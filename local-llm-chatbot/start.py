
import gradio as gr
import subprocess
from openai import OpenAI
import time

DEFAULT_MODEL = "llama3.2:latest"

openai = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

system_message = """
Your helpful assistant is a local chatbot that uses Ollama models. It can answer questions, provide information, and engage in conversation.
"""


def get_ollama_models():
    """Get installed Ollama model names using `ollama list`."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        lines = result.stdout.strip().splitlines()

        if len(lines) <= 1:
            return []

        # First column contains the model name.
        models = [
            line.split()[0]
            for line in lines[1:]
            if line.strip()
        ]

        return models

    except (subprocess.SubprocessError, FileNotFoundError) as error:
        print(f"Could not retrieve Ollama models: {error}")
        return []


def refresh_models():
    """Refresh the model dropdown without restarting the app."""
    models = get_ollama_models()

    default = (
        DEFAULT_MODEL
        if DEFAULT_MODEL in models
        else (models[0] if models else None)
    )

    return gr.update(
        choices=models,
        value=default
    )


def chat(message, history, system_prompt, selected_model):
    """Generate a streaming response using the selected settings."""

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

    # Include previous conversation messages.
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(
        {
            "role": item["role"],
            "content": item["content"]
        }
        for item in history
        if item.get("role") in ("user", "assistant")
    )

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        stream = openai.chat.completions.create(
            model=selected_model,
            messages=messages,
            stream=True
        )

        response = ""

        for event in stream:
            content = event.choices[0].delta.content

            if content:
                response += content
                yield response

    except Exception as error:
        yield f"Error: {error}"


# Load installed models when the application starts.
models = get_ollama_models()

initial_model = (
    DEFAULT_MODEL
    if DEFAULT_MODEL in models
    else (models[0] if models else None)
)


with gr.Blocks(title="Local Ollama Chatbot") as demo:

    gr.Markdown("# Local Ollama Chatbot")

    # System prompt configuration.
    system_prompt = gr.Textbox(
        label="System Prompt",
        value=system_message,
        lines=8,
        max_lines=20,
        placeholder="Enter your system prompt...",
        info="Customize the instructions given to the model."
    )

    # Model selection and refresh button.
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Ollama Model",
            choices=models,
            value=initial_model,
            interactive=True,
            scale=4,
            info="Select an installed local model."
        )

        refresh_button = gr.Button(
            "Refresh Models",
            scale=1
        )

    refresh_button.click(
        fn=refresh_models,
        inputs=[],
        outputs=[model_dropdown]
    )

    # Chat interface with configurable system prompt and model.
    gr.ChatInterface(
        fn=chat,
        type="messages",
        additional_inputs=[
            system_prompt,
            model_dropdown
        ],
        additional_inputs_accordion=None,
        title="Chat",
        description="Chat with your locally running Ollama models."
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True
    )
