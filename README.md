# LLM Experiments

A collection of experiments and projects exploring Large Language Models (LLMs).

## Projects

### [Local LLM Chatbot](./local-llm-chatbot)

A Gradio-based chatbot interface that connects to locally running Ollama models via the OpenAI-compatible API. Features include:

- Streaming chat responses
- Model selection from installed Ollama models
- Customizable system prompt
- Conversation history support

### [Local LLM RAG](./local-llm-rag)

A RAG (Retrieval-Augmented Generation) chatbot that reads PDF documents, chunks and embeds them into a vector store, and uses the retrieved context to answer questions. Features include:

- PDF text extraction with PyMuPDF
- Configurable chunking with overlap for better context coverage
- Vector storage and semantic search via ChromaDB
- Ollama-powered embeddings (`nomic-embed-text`) and chat
- Source attribution with file name and page number
- One-click PDF re-indexing

## Getting Started

Each project folder contains its own `requirements.txt`. Navigate into a project directory and install dependencies:

```bash
cd <project-folder>
pip install -r requirements.txt
```

## License

MIT
