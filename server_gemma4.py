#!/usr/bin/env python3
"""
Gemma 4 E4B MLX FastAPI Server - Neuruh Sovereign Multimodal Inference
Asset ID: AI-G4-001-API
Port: 8200 (Bonsai stays on 8100)
v1.1.2 — Uses apply_chat_template (mlx_vlm 0.4.3 required format).
Raw prompt format returns empty string. apply_chat_template returns real output.
"""

import os, re, time, json, base64, hashlib, logging
from io import BytesIO
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gemma4")

MODEL_ID    = os.environ.get("GEMMA4_MODEL_ID", "mlx-community/gemma-4-e4b-it-4bit")
MODEL_LOCAL = os.environ.get("GEMMA4_MODEL_PATH", os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit"))
PORT               = int(os.environ.get("GEMMA4_PORT", "8200"))
MAX_CONTEXT        = 128_000
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP       = 0.7

_model = _processor = _config = None

_REASONING_PATTERNS = [
    r'<\|channel>thought[\s\S]*?<channel\|>',
    r'<\|channel>thought[\s\S]*$',
    r'<channel\|>',
    r'<think>[\s\S]*?</think>',
]

def strip_reasoning(text: str) -> str:
    for p in _REASONING_PATTERNS:
        text = re.sub(p, '', text, flags=re.DOTALL)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def get_model():
    global _model, _processor, _config
    if _model is None:
        log.info(f"Loading Gemma 4: {MODEL_ID}")
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        _model, _processor = load(MODEL_ID)
        _config = load_config(MODEL_ID)
        log.info("Gemma 4 loaded (text + vision ready)")
    return _model, _processor, _config

app = FastAPI(title="Gemma 4 E4B MLX Server", version="1.1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

def build_prompt(messages: list[ChatMessage], tools=None) -> str:
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = get_model()
    system_parts = []
    conversation = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            conversation.append(msg)
    if tools:
        tool_defs = [{"name": t.function.name, "description": t.function.description, "parameters": t.function.parameters} for t in tools]
        system_parts.append(f"You have access to these tools:\n{json.dumps(tool_defs, indent=2)}\nTo call a tool respond with JSON: {{\"tool_call\": {{\"name\": \"fn\", \"arguments\": {{}}}}}}")
    # Build text from messages
    if len(conversation) == 1 and not system_parts:
        text = conversation[0].content
    else:
        parts = []
        if system_parts:
            parts.append("[SYSTEM]\n" + "\n\n".join(system_parts) + "\n[/SYSTEM]")
        for msg in conversation:
            parts.append(f"[{msg.role.upper()}]\n{msg.content}")
        text = "\n\n".join(parts)
    return apply_chat_template(processor, config, text, num_images=0)

def generate_text(prompt: str, max_tokens: int, temperature: float):
    model, processor, config = get_model()
    from mlx_vlm import generate
    start = time.time()
    result = generate(model, processor, prompt=prompt, max_tokens=max_tokens, temp=temperature, verbose=False)
    latency = (time.time() - start) * 1000
    raw = result.text if hasattr(result, 'text') else str(result)
    tokens = result.generation_tokens if hasattr(result, 'generation_tokens') else len(raw.split())
    tps = result.generation_tps if hasattr(result, 'generation_tps') else 0
    mem = result.peak_memory if hasattr(result, 'peak_memory') else 0
    clean = strip_reasoning(raw)
    log.info(f"Generated {tokens} tokens | {tps:.1f} tok/sec | peak {mem:.2f}GB | {latency:.0f}ms")
    return clean, tokens, latency

@app.get("/health")
async def health():
    return {"status": "ok" if Path(MODEL_LOCAL).exists() else "model_missing", "version": "1.1.2", "model_id": MODEL_ID, "model_loaded": _model is not None, "port": PORT, "capabilities": ["text", "vision", "function_calling", "multilingual"], "context_window": MAX_CONTEXT, "quantization": "4-bit", "library": "mlx_vlm"}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = build_prompt(req.messages, req.tools)
    output, token_count, latency = generate_text(prompt, req.max_tokens, req.temperature)
    tool_calls = None
    finish_reason = "stop"
    if req.tools and '"tool_call"' in output:
        try:
            parsed = json.loads(output)
            if "tool_call" in parsed:
                tc = parsed["tool_call"]
                tool_calls = [{"id": f"call_{hashlib.md5(json.dumps(tc).encode()).hexdigest()[:8]}", "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))}}]
                finish_reason = "tool_calls"
                output = ""
        except json.JSONDecodeError:
            pass
    input_tokens = len(prompt.split()) * 2
    return {"id": f"gemma4-{hashlib.md5(prompt.encode()).hexdigest()[:12]}", "object": "chat.completion", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": output if not tool_calls else None, "tool_calls": tool_calls}, "finish_reason": finish_reason}], "usage": {"prompt_tokens": input_tokens, "completion_tokens": token_count, "total_tokens": input_tokens + token_count}, "latency_ms": round(latency, 2), "provider": "neuruh-gemma4-local"}

@app.post("/v1/vision")
async def vision(req: VisionRequest):
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image
    model, processor, config = get_model()
    try:
        image = Image.open(BytesIO(base64.b64decode(req.image_base64)))
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")
    prompt = apply_chat_template(processor, config, req.prompt, num_images=1)
    start = time.time()
    result = vlm_generate(model, processor, prompt=prompt, image=image, max_tokens=req.max_tokens, temp=req.temperature, verbose=False)
    latency = (time.time() - start) * 1000
    clean = strip_reasoning(result.text if hasattr(result, 'text') else str(result))
    return {"text": clean, "latency_ms": round(latency, 2), "model": MODEL_ID, "mode": "vision", "success": True}

@app.post("/generate")
async def generate_simple(req: GenerateRequest):
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = get_model()
    text = f"[SYSTEM]\n{req.system}\n[/SYSTEM]\n\n{req.prompt}" if req.system else req.prompt
    prompt = apply_chat_template(processor, config, text, num_images=0)
    output, token_count, latency = generate_text(prompt, req.max_tokens, req.temperature)
    return {"text": output, "tokens_generated": token_count, "latency_ms": round(latency, 2), "model": MODEL_ID, "success": True}

@app.post("/classify")
async def classify(text: str = "", labels: list[str] = []):
    if not text or len(labels) < 2:
        raise HTTPException(400, "Provide text and at least 2 labels")
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = get_model()
    content = f"Classify this text into exactly one of these categories: {', '.join(labels)}\n\nText: {text}\n\nReply with ONLY the category name."
    prompt = apply_chat_template(processor, config, content, num_images=0)
    output, _, latency = generate_text(prompt, max_tokens=20, temperature=0.1)
    matched = next((l for l in labels if l.lower() in output.lower().strip()), labels[0])
    return {"label": matched, "raw_output": output, "latency_ms": round(latency, 2), "success": True}

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Gemma 4 server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
