"""HTTP API server for FlashInfer text generation."""

from typing import List, Optional

from flashinfer.engine.factory import create_engine
from flashinfer.engine.generation_config import GenerationConfig

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Run: pip install flashinfer[server]"
    ) from exc

app = FastAPI(title="FlashInfer", version="0.3.0")
_engine = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@app.on_event("startup")
def load_model():
    global _engine
    import os

    model_name = os.environ.get("FLASHINFER_MODEL", "gpt2")
    device = os.environ.get("FLASHINFER_DEVICE", "cpu")
    dtype = os.environ.get("FLASHINFER_DTYPE", "float32")
    _engine = create_engine(model_name, device=device, dtype=dtype)


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        stop_sequences=request.stop,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
    )

    if request.stream:
        def event_stream():
            for piece in _engine.generate_stream(request.prompt, config=config):
                yield f"data: {piece}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = _engine.generate(request.prompt, config=config)
    return {"text": text}


def main():
    import uvicorn

    uvicorn.run("flashinfer.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
