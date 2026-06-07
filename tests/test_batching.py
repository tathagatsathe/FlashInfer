import torch

from flashinfer.engine.generator import InferenceEngine


def test_generate_batch_same_length_greedy():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    prompts = ["Hello world", "The sky is"]
    batch_out = engine.generate_batch(
        prompts, max_new_tokens=3, temperature=0.0
    )
    assert len(batch_out) == 2
    for prompt, batch_text in zip(prompts, batch_out):
        single_text = engine.generate(
            prompt, max_new_tokens=3, temperature=0.0
        )
        assert batch_text == single_text


def test_generate_batch_count():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    prompts = ["Hello", "The capital of France is"]
    batch_out = engine.generate_batch(prompts, max_new_tokens=2, temperature=0.0)
    assert len(batch_out) == 2
    for text in batch_out:
        assert isinstance(text, str)
        assert len(text) > 0


def test_left_pad_batch_input():
    from flashinfer.engine.batching import left_pad_sequences

    batch = left_pad_sequences(
        [[1, 2, 3], [4, 5]],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert batch.input_ids.shape == (2, 3)
    assert batch.attention_mask[0].tolist() == [1, 1, 1]
    assert batch.attention_mask[1].tolist() == [0, 1, 1]
