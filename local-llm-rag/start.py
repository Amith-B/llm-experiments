
import gradio as gr
import subprocess
import os
import glob
import hashlib
import time

import pymupdf
import chromadb
from openai import OpenAI


# ── Configuration ──────────────────────────────────────────────────

DEFAULT_CHAT_MODEL = "llama3.2:latest"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"

PDFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
TOP_K = 5                 # context chunks to retrieve per query
BATCH_SIZE = 50           # ChromaDB upsert batch size

COLLECTION_NAME = "pdf_documents"


# ── Clients ────────────────────────────────────────────────────────

openai_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

chroma_client = chromadb.Client()  # in-memory, ephemeral


# ── Embedding Helper ──────────────────────────────────────────────

def get_embeddings(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Get embeddings for a batch of texts from Ollama."""
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            model=model,
            input=text,
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


# ── PDF Processing ─────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, returning a list of {page, text} dicts."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunk_id = hashlib.sha256(
                f"{source}:p{page}:c{index}:{chunk[:64]}".encode()
            ).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": index,
                },
            })
            index += 1

        next_start = start + chunk_size - chunk_overlap
        if next_start <= start:
            break
        start = next_start

    return chunks


# ── Indexing ───────────────────────────────────────────────────────

def index_pdfs(progress=gr.Progress()):
    """Read all PDFs in pdfs/, chunk them, embed, and store in ChromaDB."""

    # Recreate collection for a clean re-index.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    pdf_files = sorted(glob.glob(os.path.join(PDFS_DIR, "*.pdf")))

    if not pdf_files:
        return (
            "⚠️ No PDF files found in the `pdfs/` directory.\n\n"
            "Add some PDFs and click **Index PDFs** again."
        )

    total_chunks = 0
    file_summaries = []

    for file_idx, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        progress(file_idx / len(pdf_files), desc=f"Processing {filename}…")

        # Extract text.
        try:
            pages = extract_text_from_pdf(pdf_path)
        except Exception as e:
            file_summaries.append(f"❌ **{filename}**: Error reading — {e}")
            continue

        # Chunk each page.
        all_chunks = []
        for page_info in pages:
            all_chunks.extend(
                chunk_text(
                    text=page_info["text"],
                    source=filename,
                    page=page_info["page"],
                )
            )

        if not all_chunks:
            file_summaries.append(f"⚠️ **{filename}**: No extractable text.")
            continue

        # Embed and upsert in batches.
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = get_embeddings(texts)

            collection.upsert(
                ids=[c["id"] for c in batch],
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
                embeddings=embeddings,
            )

        total_chunks += len(all_chunks)
        file_summaries.append(
            f"✅ **{filename}**: {len(pages)} page(s), {len(all_chunks)} chunk(s)"
        )

    progress(1.0, desc="Done!")

    summary = (
        f"### Indexing Complete\n\n"
        f"**{len(pdf_files)} PDF(s)** processed → **{total_chunks} chunks** indexed.\n\n"
        + "\n".join(f"- {s}" for s in file_summaries)
    )
    return summary


# ── Retrieval ──────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Retrieve the most relevant chunks for a user query."""
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return ""

    if collection.count() == 0:
        return ""

    query_embedding = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_parts.append(f"[Source: {source}, Page {page}]\n{doc}")

    return "\n\n---\n\n".join(context_parts)


# ── Ollama Helpers ─────────────────────────────────────────────────

def get_ollama_models():
    """Get installed Ollama model names using `ollama list`."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:
            return []
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Could not retrieve Ollama models: {e}")
        return []


def refresh_models():
    """Refresh the model dropdown."""
    models = get_ollama_models()
    default = (
        DEFAULT_CHAT_MODEL
        if DEFAULT_CHAT_MODEL in models
        else (models[0] if models else None)
    )
    return gr.update(choices=models, value=default)


# ── Chat ───────────────────────────────────────────────────────────

RAG_SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided context \
extracted from PDF documents.

INSTRUCTIONS:
- Use the context below to answer the user's question accurately.
- If the context contains the answer, provide it clearly and cite the source \
file name and page number.
- If the context does not contain enough information, say so honestly and \
answer to the best of your general knowledge, clearly distinguishing between \
sourced and unsourced information.
- Be concise but thorough.

CONTEXT FROM DOCUMENTS:
{context}
"""

NO_DOCS_SYSTEM = (
    "You are a helpful assistant. No PDF documents have been indexed yet. "
    "Answer the user's question to the best of your ability and suggest "
    "they add PDFs to the pdfs/ folder and click 'Index PDFs' for "
    "document-grounded answers."
)


def chat(message, history, selected_model):
    """RAG-augmented streaming chat."""

    if not selected_model:
        yield "No Ollama models found. Pull a model and refresh the list."
        return

    if not message.strip():
        yield ""
        return

    if message.strip().lower() == "stop":
        yield "Stopping the chat..."
        time.sleep(1)  # Simulate delay
        return gr.close_all()


    # Retrieve relevant context from indexed PDFs.
    context = retrieve_context(message)

    system_content = (
        RAG_SYSTEM_TEMPLATE.format(context=context) if context else NO_DOCS_SYSTEM
    )

    messages = [{"role": "system", "content": system_content}]

    # Include conversation history.
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in ("user", "assistant")
    )

    messages.append({"role": "user", "content": message})

    try:
        stream = openai_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            stream=True,
        )

        response = ""
        for event in stream:
            content = event.choices[0].delta.content
            if content:
                response += content
                yield response

    except Exception as error:
        yield f"Error: {error}"


# ── Launch ─────────────────────────────────────────────────────────

models = get_ollama_models()
initial_model = (
    DEFAULT_CHAT_MODEL
    if DEFAULT_CHAT_MODEL in models
    else (models[0] if models else None)
)

with gr.Blocks(title="Local RAG Chatbot") as demo:

    gr.Markdown(
        "# 📄 Local RAG Chatbot\n"
        "Chat with your PDF documents using locally running Ollama models."
    )

    with gr.Row():

        # ── Sidebar ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")

            model_dropdown = gr.Dropdown(
                label="Chat Model",
                choices=models,
                value=initial_model,
                interactive=True,
                info="Select an installed Ollama model for chat.",
            )
            refresh_btn = gr.Button("🔄 Refresh Models")

            gr.Markdown("---")
            gr.Markdown("### 📚 PDF Index")
            gr.Markdown(
                f"Place your PDFs in:\n\n`{PDFS_DIR}`"
            )
            index_btn = gr.Button("📥 Index PDFs", variant="primary")
            index_status = gr.Markdown(
                value="PDFs have not been indexed yet. Click **Index PDFs** to start."
            )

        # ── Chat area ──────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[model_dropdown],
                additional_inputs_accordion=None,
                title="Chat",
                description="Ask questions about your indexed PDF documents.",
            )

    refresh_btn.click(fn=refresh_models, inputs=[], outputs=[model_dropdown])
    index_btn.click(fn=index_pdfs, inputs=[], outputs=[index_status])


if __name__ == "__main__":
    os.makedirs(PDFS_DIR, exist_ok=True)
    demo.launch(inbrowser=True)
