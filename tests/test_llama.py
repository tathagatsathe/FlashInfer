import pytest
import torch

from flashinfer.layers.llama_attention import LlamaAttention
from flashinfer.layers.rms_norm import RMSNorm
from flashinfer.layers.swiglu_mlp import SwiGLUMLP
from flashinfer.model.llama import LlamaConfig, LlamaModel


def test_rms_norm_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    assert norm(x).shape == x.shape


def test_swiglu_shape():
    mlp = SwiGLUMLP(64, 128)
    x = torch.randn(2, 8, 64)
    assert mlp(x).shape == x.shape


def test_llama_attention_shape():
    attn = LlamaAttention(64, num_heads=4, num_kv_heads=2, max_position=128)
    x = torch.randn(1, 8, 64)
    assert attn(x).shape == x.shape


def test_llama_model_forward():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaModel.from_config(config)
    ids = torch.randint(0, 100, (1, 10))
    logits = model(ids)
    assert logits.shape == (1, 10, 100)


@pytest.mark.slow
def test_tinyllama_logits_parity():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashinfer.model.llama import LlamaModel
    from flashinfer.weights.llama_loader import config_from_hf_llama, load_hf_llama_weights

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config = config_from_hf_llama(model_name)
    model = LlamaModel.from_config(config)
    load_hf_llama_weights(model, model_name)
    model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        our_logits = model(input_ids)
        hf_logits = hf_model(input_ids).logits

    assert torch.allclose(
        our_logits[0, -1, :].float(),
        hf_logits[0, -1, :].float(),
        atol=1e-3,
        rtol=1e-2,
    )
