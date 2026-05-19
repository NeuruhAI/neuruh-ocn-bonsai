#!/usr/bin/env python3
"""
OCN Bonsai FastAPI Server — Neuruh Local Inference Endpoint
Asset ID: AI-CB-003-API
Deploy: cd ~/neuruh-ocn-bonsai && source .venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8100
Health: curl http://localhost:8100/health
Generate: curl -X POST http://localhost:8100/generate -H 'Content-Type: application/json' -d '{"prompt": "hello", "max_tokens": 100}'
"""

import os
import time
import subprocess
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "BONSAI_MODEL_PATH",
    os.path.expanduser("~/neuruh-ocn-bonsai/models/Bonsai-8B-mlx")
)
VENV_PYTHON = os.environ.get(
    "BONSAI_PYTHON",
    os.path.expanduser("~/neuruh-ocn-bonsai/.venv/bin/python")
)
PORT = int(os.environ.get("BONSAI_PORT", "8100"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OCN Bonsai Inference Server",
    description="Neuruh sovereign local inference — Bonsai 8B MLX on Apple Silicon",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    system: Optional[str] = Field(None, description="Optional system prompt prefix")

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float
    model: str
    success: bool

class ClassifyRequest(BaseModel):
    text: str
    labels: list[str] = Field(..., min_length=2)

class ClassifyResponse(BaseModel):
    label: str
    raw_output: str
    latency_ms: float
    success: bool

class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_exists: bool
    python_path: str
    python_exists: bool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_mlx(prompt: str, max_tokens: int = 200, temp: float = 0.7) -> str:
    """Run mlx_lm.generate via subprocess and return raw stdout."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(503, f"Model not found: {MODEL_PATH}")
    if not os.path.exists(VENV_PYTHON):
        raise HTTPException(503, f"Python not found: {VENV_PYTHON}")

    cmd = [
        VENV_PYTHON, "mlx_lm", "generate",
        "--model", MODEL_PATH,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Generation timed out (120s limit)")

    if result.returncode != 0:
        raise HTTPException(500, f"MLX error: {result.stderr.strip()}")

    return result.stdout.strip()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if os.path.exists(MODEL_PATH) else "model_missing",
        model_path=MODEL_PATH,
        model_exists=os.path.exists(MODEL_PATH),
        python_path=VENV_PYTHON,
        python_exists=os.path.exists(VENV_PYTHON),
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    full_prompt = f"{req.system}\n\n{req.prompt}" if req.system else req.prompt
    start = time.time()
    output = run_mlx(full_prompt, req.max_tokens, req.temperature)
    latency = (time.time() - start) * 1000
    tokens_est = max(1, len(output.split()))
    return GenerateResponse(
        text=output,
        tokens_generated=tokens_est,
        latency_ms=round(latency, 2),
        model=MODEL_PATH,
        success=True,
    )

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    labels_str = ", ".join(req.labels)
    prompt = (
        f"Classify the following text into exactly one category.\n"
        f"Categories: {labels_str}\n\n"
        f"Text: {req.text}\n\n"
        f"Reply with ONLY the category name, nothing else."
    )
    start = time.time()
    output = run_mlx(prompt, max_tokens=20, temp=0.1)
    latency = (time.time() - start) * 1000

    # Match output to closest label
    matched = req.labels[0]
    output_lower = output.lower().strip()
    for label in req.labels:
        if label.lower() in output_lower:
            matched = label
            break

    return ClassifyResponse(
        label=matched,
        raw_output=output,
        latency_ms=round(latency, 2),
        success=True,
    )

@app.post("/embed")
async def embed():
    """Placeholder — wire when MLX adds embedding support or swap to sentence-transformers."""
    raise HTTPException(501, "Embedding endpoint not yet implemented. Use /classify for now.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
