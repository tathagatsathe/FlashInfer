import torch

from flashinfer.sampling.sampler import sample_next_token


def test_top_k_masks():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    torch.manual_seed(0)
    for _ in range(10):
        token = sample_next_token(logits, temperature=1.0, top_k=1)
        assert token == 1


def test_top_p_nucleus():
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])

    torch.manual_seed(0)
    token_full = sample_next_token(logits, temperature=1.0, top_p=1.0)

    torch.manual_seed(0)
    token_none = sample_next_token(logits, temperature=1.0, top_p=None)
    assert token_full == token_none

    token_greedy = sample_next_token(logits, temperature=0.0, top_p=0.0)
    assert token_greedy == 0


def test_greedy_unchanged():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    token = sample_next_token(logits, temperature=0.0, top_k=2, top_p=0.5)
    assert token == 1
