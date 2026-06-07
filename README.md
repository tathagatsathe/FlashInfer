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
text = engine.generate("The capital of France is", max_new_tokens=20, temperature=0.0)
print(text)
```

Or use the CLI:

```bash
python examples/generate.py --prompt "Hello, world" --max-new-tokens 50 --temperature 0.8
```

## Project Structure

```
flashinfer/
├── layers/       # LayerNorm, embeddings, attention, MLP
├── model/        # GPT2Model
├── weights/      # HuggingFace weight loader
├── sampling/     # Greedy and temperature sampling
└── engine/       # Inference loop
```

## Tests

```bash
pytest tests/
```

Parity tests compare logits against the HuggingFace `transformers` reference implementation.
