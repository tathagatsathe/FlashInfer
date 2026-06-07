import pytest

from flashinfer.engine.generator import InferenceEngine
from flashinfer.sampling.sampler import apply_repetition_penalty
import torch


def test_seed_reproducibility():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    prompt = "Hello"
    a = engine.generate(prompt, max_new_tokens=5, temperature=1.0, seed=123)
    b = engine.generate(prompt, max_new_tokens=5, temperature=1.0, seed=123)
    assert a == b


def test_stop_sequence():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    text = engine.generate(
        "One two three",
        max_new_tokens=20,
        temperature=0.0,
        stop_sequences=[" four"],
    )
    assert " four" not in text[len("One two three") :]


def test_repetition_penalty():
    logits = torch.tensor([1.0, 2.0, 3.0])
    penalized = apply_repetition_penalty(logits, [1, 2], penalty=2.0)
    assert penalized[1].item() == 1.0
    assert penalized[2].item() == 1.5


def test_max_context_len_raises():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    prompt = "Hello world this is a test"
    with pytest.raises(ValueError, match="max_context_len"):
        engine.generate(prompt, max_new_tokens=50, max_context_len=5)
