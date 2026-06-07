import torch

from flashinfer.cache.kv_cache import KVCache
from flashinfer.model.gpt2 import GPT2Model
from flashinfer.weights.loader import config_from_hf, load_hf_weights


MODEL_NAME = "gpt2"


def _load_model():
    config = config_from_hf(MODEL_NAME)
    model = GPT2Model.from_config(config)
    load_hf_weights(model, MODEL_NAME)
    model.eval()
    return model


def _make_cache(model, batch=1):
    config = model.config
    return KVCache.empty(
        num_layers=config.n_layer,
        batch=batch,
        n_head=config.n_head,
        head_dim=config.n_embd // config.n_head,
        max_len=config.n_positions,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_prefill_logits_parity():
    model = _load_model()
    input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])

    cache = _make_cache(model)
    with torch.no_grad():
        full_logits = model(input_ids)
        cached_logits = model.forward_prefill(input_ids, cache)

    assert torch.allclose(
        full_logits[0, -1, :],
        cached_logits[0, -1, :],
        atol=1e-4,
        rtol=1e-3,
    )
    assert cache.seq_len == input_ids.size(1)


def test_decode_logits_parity():
    model = _load_model()
    input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    cache = _make_cache(model)
    generated = input_ids.tolist()[0]

    with torch.no_grad():
        model.forward_prefill(input_ids, cache)

        for _ in range(5):
            full_logits = model(torch.tensor([generated]))
            next_token = int(torch.argmax(full_logits[0, -1, :]).item())

            decode_logits = model.forward_decode(torch.tensor([[next_token]]), cache)

            generated.append(next_token)
            full_after = model(torch.tensor([generated]))

            assert torch.allclose(
                full_after[0, -1, :],
                decode_logits[0, -1, :],
                atol=1e-4,
                rtol=1e-3,
            )


def test_cached_generation_parity():
    model = _load_model()
    input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    generated_full = input_ids.tolist()[0]
    generated_cached = input_ids.tolist()[0]

    with torch.no_grad():
        for _ in range(10):
            logits = model(torch.tensor([generated_full]))
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            generated_full.append(next_token)

    cache = _make_cache(model)
    with torch.no_grad():
        logits = model.forward_prefill(input_ids, cache)
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        generated_cached.append(next_token)

        for _ in range(9):
            decode_logits = model.forward_decode(torch.tensor([[next_token]]), cache)
            next_token = int(torch.argmax(decode_logits[0, -1, :]).item())
            generated_cached.append(next_token)

    assert generated_full == generated_cached
