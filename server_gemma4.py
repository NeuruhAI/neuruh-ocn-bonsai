#!/usr/bin/env python3
"""
Gemma 4 E4B MLX FastAPI Server - Neuruh Sovereign Multimodal Inference
Asset ID: AI-G4-001-API
Port: 8200 (Bonsai stays on 8100)
Deploy: cd ~/neuruh-ocn-bonsai && source ~/.venv-gemma4/bin/activate && uvicorn server_gemma4:app --host 0.0.0.0 --port 8200

IMPORTANT: Uses mlx_vlm (NOT mlx_lm) - mlx_lm does not support gemma4 model type yet.
mlx_vlm 0.4.3+ handles both text and vision for Gemma 4.
Confirmed working: 9.48 tok/sec on M1 Mac Mini, peak memory 5.24 GB.
"""

import os
import sys
import time
import json
import base64
import hashlib
import logging
from io import BytesIO
from typing import Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gemma4")

# Config
MODEL_ID = os.environ.get("GEMMA4_MODEL_ID", "mlx-community/gemma-4-e4b-it-4bit")
MODEL_LOCAL = os.environ.get(
    "GEMMA4_MODEL_PATH",
    os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit")
)
PORT = int(os.environ.get("GEMMA4_PORT", "8200"))
MAX_CONTEXT = 128_000
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP = 0.7

# Single model instance for text + vision via mlx_vlm
_model = None
_processor = None


def get_model():
    """Lazy-load Gemma 4 via mlx_vlm (handles both text and vision)."""
    global _model, _processor
    if _model is None:
        log.info(f"Loading Gemma 4 model: {MODEL_ID}")
        from mlx_vlm import load
        _model, _processor = load(MODEL_ID)
        log.info("Gemma 4 model loaded (text + vision ready)")
    return _model, _processor


# App
app = FastAPI(
    title="Gemma 4 E4B MLX Server",
    description="Neuruh sovereign multimodal inference - Gemma 4 E4B 4-bit on Apple Silicon",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""

class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-e4b"
    messages: list[ChatMessage]
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=8192)
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, le=2.0)
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str] = None
    stream: bool = False

class VisionRequest(BaseModel):
    prompt: str
    image_base64: str
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, le=2.0)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=8192)
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, le=2.0)
    system: Optional[str] = None


# Helpers
def format_chat_prompt(messages, tools=None):
    """Format messages into Gemma 4 chat template."""
    parts = []
    if tools:
        tool_defs = [{"name": t.function.name, "description": t.function.description, "parameters": t.function.parameters} for t in tools]
        parts.append(f"<start_of_turn>system\nYou have access to these tools:\n{json.dumps(tool_defs, indent=2)}\nTo call a tool, respond with JSON: {{\"tool_call\": {{\"name\": \"fn\", \"arguments\": {{}}}}}}\n<end_of_turn>")
    for msg in messages:
        role_tag = "model" if msg.role == "assistant" else msg.role
        parts.append(f"<start_of_turn>{role_tag}\n{msg.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def generate_text(prompt, max_tokens, temperature):
    """Generate text using mlx_vlm. Returns (text, token_count, latency_ms)."""
    model, processor = get_model()
    from mlx_vlm import generate
    start = time.time()
    result = generate(model, processor, prompt=prompt, max_tokens=max_tokens, temp=temperature, verbose=False)
    latency = (time.time() - start) * 1000
    text = result.text if hasattr(result, 'text') else str(result)
    token_count = result.generation_tokens if hasattr(result, 'generation_tokens') else len(text.split())
    return text, token_count, latency


# Routes
@app.get("/health")
async def health():
    return {
        "status": "ok" if Path(MODEL_LOCAL).exists() else "model_missing",
        "model_id": MODEL_ID,
        "model_loaded": _model is not None,
        "port": PORT,
        "capabilities": ["text", "vision", "function_calling", "multilingual"],
        "context_window": MAX_CONTEXT,
        "quantization": "4-bit",
        "library": "mlx_vlm",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions."""
    prompt = format_chat_prompt(req.messages, req.tools)
    output, token_count, latency = generate_text(prompt, req.max_tokens, req.temperature)

    # Clean trailing turn markers
    output = output.split("<end_of_turn>")[0].strip()
    output = output.split("<start_of_turn>")[0].strip()

    # Check for tool calls
    tool_calls = None
    finish_reason = "stop"
    if req.tools and '"tool_call"' in output:
        try:
            parsed = json.loads(output)
            if "tool_call" in parsed:
                tc = parsed["tool_call"]
                tool_calls = [{
                    "id": f"call_{hashlib.md5(json.dumps(tc).encode()).hexdigest()[:8]}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))},
                }]
                finish_reason = "tool_calls"
                output = ""
        except json.JSONDecodeError:
            pass

    input_tokens = len(prompt.split()) * 2
    return {
        "id": f"gemma4-{hashlib.md5(prompt.encode()).hexdigest()[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output if not tool_calls else None, "tool_calls": tool_calls}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": token_count, "total_tokens": input_tokens + token_count},
        "latency_ms": round(latency, 2),
        "provider": "neuruh-gemma4-local",
    }


@app.post("/v1/vision")
async def vision(req: VisionRequest):
    """Image + text via Gemma 4 vision."""
    from mlx_vlm import generate as vlm_generate
    from PIL import Image
    model, processor = get_model()
    try:
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")

    vision_prompt = f"<start_of_turn>user\n{req.prompt}<end_of_turn>\n<start_of_turn>model\n"
    start = time.time()
    result = vlm_generate(model, processor, image=image, prompt=vision_prompt, max_tokens=req.max_tokens, temp=req.temperature, verbose=False)
    latency = (time.time() - start) * 1000
    text = result.text if hasattr(result, 'text') else str(result)
    return {"text": text, "latency_ms": round(latency, 2), "model": MODEL_ID, "mode": "vision", "success": True}


@app.post("/generate")
async def generate_simple(req: GenerateRequest):
    """Bonsai-compatible /generate endpoint."""
    full_prompt = f"<start_of_turn>user\n{req.system}\n\n{req.prompt}<end_of_turn>\n<start_of_turn>model\n" if req.system else f"<start_of_turn>user\n{req.prompt}<end_of_turn>\n<start_of_turn>model\n"
    output, token_count, latency = generate_text(full_prompt, req.max_tokens, req.temperature)
    return {"text": output, "tokens_generated": token_count, "latency_ms": round(latency, 2), "model": MODEL_ID, "success": True}


@app.post("/classify")
async def classify(text: str = "", labels: list[str] = []):
    """Classification endpoint."""
    if not text or len(labels) < 2:
        raise HTTPException(400, "Provide text and at least 2 labels")
    labels_str = ", ".join(labels)
    prompt = f"<start_of_turn>user\nClassify this text into one category.\nCategories: {labels_str}\n\nText: {text}\n\nReply with ONLY the category name.\n<end_of_turn>\n<start_of_turn>model\n"
    output, _, latency = generate_text(prompt, max_tokens=20, temperature=0.1)
    matched = labels[0]
    for label in labels:
        if label.lower() in output.lower().strip():
            matched = label
            break
    return {"label": matched, "raw_output": output, "latency_ms": round(latency, 2), "success": True}


if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Gemma 4 server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
