# FlashInfer

A minimal LLM inference engine built from scratch in PyTorch. Supports GPT-2 and Llama-style models with KV-cache incremental decode.

## Install

```bash
pip install -e ".[dev]"
```

Optional HTTP server:

```bash
pip install -e ".[server]"
```

## Quickstart

```python
from flashinfer.engine.factory import create_engine

engine = create_engine("gpt2", device="cpu")
text = engine.generate(
    "The capital of France is",
    max_new_tokens=20,
    temperature=0.0,
    top_k=50,
    top_p=0.9,
    seed=42,
)
print(text)
```

Streaming:

```python
for piece in engine.generate_stream("Hello, world", max_new_tokens=20, temperature=0.8):
    print(piece, end="", flush=True)
```

Batch generation:

```python
results = engine.generate_batch(["Hello", "The sky is"], max_new_tokens=10, temperature=0.0)
```

## CLI

```bash
python examples/generate.py --prompt "Hello, world" --max-new-tokens 50 --stream
python examples/generate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype float16 --device cuda
```

## HTTP Server

```bash
FLASHINFER_MODEL=gpt2 flashinfer-serve
# POST http://localhost:8000/v1/completions
# {"prompt": "Hello", "max_tokens": 50, "stream": true}
```

## Benchmarks

```bash
python benchmarks/bench_generate.py --model gpt2 --prompt-tokens 32 --decode-tokens 50
```

## Features

| Version | Features |
|---------|----------|
| v1 | Custom GPT-2, HF weights, basic generation |
| v2 | KV cache, top-k / top-p sampling |
| v3 | Static batching, streaming, stop sequences, repetition penalty, seed |
| v4 | FP16/BF16 dtype, benchmarks |
| v5 | Llama (RMSNorm, RoPE, SwiGLU, GQA) via `create_engine()` |
| v6 | FastAPI server with SSE streaming |

## Project Structure

See [docs/architecture.md](docs/architecture.md) for dataflow diagrams and a full system overview.

```
flashinfer/
├── cache/        # KV cache
├── engine/       # InferenceEngine, LlamaInferenceEngine, factory
├── layers/       # GPT-2 + Llama layer primitives
├── model/        # GPT2Model, LlamaModel
├── weights/      # HF weight loaders
├── sampling/     # Sampling utilities
└── server.py     # HTTP API
```

## Tests

```bash
pytest tests/
pytest tests/ -m "not slow"   # skip TinyLlama download
```
