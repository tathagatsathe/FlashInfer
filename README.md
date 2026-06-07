# FlashInfer

A minimal GPT-2 inference engine built from scratch in PyTorch. Loads HuggingFace GPT-2 weights and runs text generation through a custom model implementation.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from flashinfer.engine.generator import InferenceEngine

engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
text = engine.generate(
    "The capital of France is",
    max_new_tokens=20,
    temperature=0.0,
    top_k=50,
    top_p=0.9,
)
print(text)
```

Or use the CLI:

```bash
python examples/generate.py --prompt "Hello, world" --max-new-tokens 50 --temperature 0.8 --top-k 50 --top-p 0.9
```

## v2 Features

- **KV cache**: Prefill runs once over the prompt; decode appends one token at a time using cached K/V tensors (O(n) per step vs O(n²) re-forward).
- **Advanced sampling**: `top_k` and `top_p` (nucleus) in addition to temperature and greedy decoding.
- **Parity preserved**: The full-sequence `GPT2Model.forward()` path is unchanged for v1 parity tests.

## Project Structure

```
flashinfer/
├── cache/        # KV cache for incremental decode
├── layers/       # LayerNorm, embeddings, attention, MLP
├── model/        # GPT2Model (forward, forward_prefill, forward_decode)
├── weights/      # HuggingFace weight loader
├── sampling/     # Greedy, temperature, top-k, top-p
└── engine/       # Inference loop with KV cache
```

## Tests

```bash
pytest tests/
```

Parity tests compare logits against the HuggingFace `transformers` reference implementation. KV-cache tests verify cached prefill/decode matches the full forward path.
