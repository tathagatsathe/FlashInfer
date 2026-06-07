import torch

from flashinfer.layers.attention import CausalSelfAttention
from flashinfer.layers.embeddings import GPT2Embeddings
from flashinfer.layers.mlp import GPT2MLP
from flashinfer.layers.norm import LayerNorm


def test_layer_norm_shape():
    ln = LayerNorm(768)
    x = torch.randn(2, 10, 768)
    out = ln(x)
    assert out.shape == (2, 10, 768)


def test_embeddings_shape():
    emb = GPT2Embeddings(vocab_size=100, hidden_size=64, max_position=128)
    ids = torch.randint(0, 100, (2, 16))
    out = emb(ids)
    assert out.shape == (2, 16, 64)


def test_attention_shape_and_causal():
    attn = CausalSelfAttention(hidden_size=64, num_heads=4, max_position=32)
    x = torch.randn(1, 8, 64)
    out = attn(x)
    assert out.shape == (1, 8, 64)

    attn.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 4, 64)
        full = attn(x1)
        padded = torch.cat([x1, torch.randn(1, 4, 64)], dim=1)
        padded_out = attn(padded)
        assert torch.allclose(full, padded_out[:, :4, :], atol=1e-5)


def test_mlp_shape():
    mlp = GPT2MLP(hidden_size=64, intermediate_size=256)
    x = torch.randn(2, 10, 64)
    out = mlp(x)
    assert out.shape == (2, 10, 64)
