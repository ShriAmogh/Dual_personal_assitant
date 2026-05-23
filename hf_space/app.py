"""
app.py — HuggingFace Spaces Deployment
---------------------------------------
OSS Personal Assistant using microsoft/Phi-3-mini-4k-instruct.

Compatible with Gradio 5.x and Python 3.13.

Features:
  - Multi-turn conversation with 5-message sliding window memory
  - Professional system prompt
  - Runs on CPU (HF free tier compatible)
"""

import time
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SLIDING_WINDOW = 5
MAX_NEW_TOKENS = 512
SYSTEM_PROMPT = (
    "You are a helpful, professional personal assistant. "
    "Be concise, accurate, and friendly. "
    "If you don't know something, say so clearly rather than guessing."
)

# ── Load Model (runs once at Space startup) ───────────────────────────────────
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=False,
)
model.eval()
print("Model loaded successfully.")


# ── Inference Function ────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """
    Gradio 5 passes history as a list of dicts:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Apply sliding window on history (each entry is a dict in Gradio 5)
    recent = history[-(SLIDING_WINDOW * 2):] if len(history) > SLIDING_WINDOW * 2 else history
    for entry in recent:
        if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
            messages.append({"role": entry["role"], "content": entry["content"]})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            # Fallback for Gradio 4 tuple format
            if entry[0]:
                messages.append({"role": "user", "content": entry[0]})
            if entry[1]:
                messages.append({"role": "assistant", "content": entry[1]})

    # Add current user message
    messages.append({"role": "user", "content": message})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize and generate
    model_inputs = tokenizer([text], return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start

    new_tokens = generated_ids[0][model_inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"[Inference] {latency:.2f}s | {len(new_tokens)} tokens generated")
    return response


# ── Gradio 5 Interface ────────────────────────────────────────────────────────

with gr.Blocks(
    title="OSS Personal Assistant - Phi-3-mini",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"),
    css="""
    .gradio-container { max-width: 860px; margin: auto; }
    footer { display: none !important; }
    .badge {
        background: linear-gradient(135deg, #6B46C1, #4338CA);
        color: white; padding: 4px 14px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; display: inline-block;
        margin-bottom: 10px;
    }
    """,
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <div class="badge">Phi-3-mini-4k-instruct &middot; CPU &middot; HF Free Tier</div>
        <h1 style="margin:8px 0 4px 0; font-size:1.8rem;">OSS Personal Assistant</h1>
        <p style="color:#888; margin:0; font-size:0.93rem;">
            Open-source AI with 5-message sliding window memory &mdash;
            part of the <strong>Dual AI Assistants</strong> project.
        </p>
    </div>
    """)

    gr.ChatInterface(
        fn=chat,
        type="messages",           # Gradio 5: use dict-based message format
        chatbot=gr.Chatbot(
            height=460,
            type="messages",
            placeholder="<strong>Hi! I'm your OSS personal assistant.</strong><br>Ask me anything.",
            bubble_full_width=False,
        ),
        textbox=gr.Textbox(
            placeholder="Type a message...",
            container=False,
            scale=7,
        ),
        examples=[
            "What is the difference between machine learning and deep learning?",
            "Help me write a professional email declining a meeting.",
            "Explain quantum computing in simple terms.",
            "What are 3 tips for better productivity?",
        ],
        cache_examples=False,
    )

    gr.HTML("""
    <div style="text-align:center; padding:10px 0 4px 0; color:#999; font-size:0.82rem;">
        Model: <code>microsoft/Phi-3-mini-4k-instruct</code> &middot;
        Memory: 5-message sliding window &middot;
        Deployed on HuggingFace Spaces
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
