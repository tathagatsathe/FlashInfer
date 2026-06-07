import torch
from transformers import AutoModelForCausalLM, GPT2Config as HFGPT2Config

from flashinfer.model.gpt2 import GPT2Config, GPT2Model


def _copy_conv1d_to_linear(
    hf_weight: torch.Tensor,
    hf_bias: torch.Tensor,
    linear: torch.nn.Linear,
    dtype: torch.dtype,
):
    linear.weight.data.copy_(hf_weight.t().to(dtype))
    if hf_bias is not None:
        linear.bias.data.copy_(hf_bias.to(dtype))


def load_hf_weights(
    model: GPT2Model,
    model_name: str,
    dtype: torch.dtype = torch.float32,
) -> GPT2Model:
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hf_state = hf_model.state_dict()
    hf_config: HFGPT2Config = hf_model.config

    config = GPT2Config(
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.n_positions,
        n_embd=hf_config.n_embd,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_inner=hf_config.n_inner,
    )

    expected = (
        config.vocab_size,
        config.n_positions,
        config.n_embd,
        config.n_layer,
        config.n_head,
    )
    actual = (
        model.config.vocab_size,
        model.config.n_positions,
        model.config.n_embd,
        model.config.n_layer,
        model.config.n_head,
    )
    if expected != actual:
        raise ValueError(f"Model config mismatch: expected {expected}, got {actual}")

    model.embeddings.token_emb.weight.data.copy_(
        hf_state["transformer.wte.weight"].to(dtype)
    )
    model.embeddings.pos_emb.weight.data.copy_(
        hf_state["transformer.wpe.weight"].to(dtype)
    )

    for i, block in enumerate(model.blocks):
        prefix = f"transformer.h.{i}"

        block.ln_1.weight.data.copy_(hf_state[f"{prefix}.ln_1.weight"].to(dtype))
        block.ln_1.bias.data.copy_(hf_state[f"{prefix}.ln_1.bias"].to(dtype))
        block.ln_2.weight.data.copy_(hf_state[f"{prefix}.ln_2.weight"].to(dtype))
        block.ln_2.bias.data.copy_(hf_state[f"{prefix}.ln_2.bias"].to(dtype))

        _copy_conv1d_to_linear(
            hf_state[f"{prefix}.attn.c_attn.weight"],
            hf_state[f"{prefix}.attn.c_attn.bias"],
            block.attn.qkv,
            dtype,
        )
        _copy_conv1d_to_linear(
            hf_state[f"{prefix}.attn.c_proj.weight"],
            hf_state[f"{prefix}.attn.c_proj.bias"],
            block.attn.out_proj,
            dtype,
        )
        _copy_conv1d_to_linear(
            hf_state[f"{prefix}.mlp.c_fc.weight"],
            hf_state[f"{prefix}.mlp.c_fc.bias"],
            block.mlp.fc,
            dtype,
        )
        _copy_conv1d_to_linear(
            hf_state[f"{prefix}.mlp.c_proj.weight"],
            hf_state[f"{prefix}.mlp.c_proj.bias"],
            block.mlp.proj,
            dtype,
        )

    model.ln_f.weight.data.copy_(hf_state["transformer.ln_f.weight"].to(dtype))
    model.ln_f.bias.data.copy_(hf_state["transformer.ln_f.bias"].to(dtype))

    if "lm_head.weight" in hf_state:
        model.lm_head.weight.data.copy_(hf_state["lm_head.weight"].to(dtype))
    else:
        model.lm_head.weight.data.copy_(hf_state["transformer.wte.weight"].to(dtype))

    del hf_model
    return model


def config_from_hf(model_name: str) -> GPT2Config:
    hf_config: HFGPT2Config = HFGPT2Config.from_pretrained(model_name)
    return GPT2Config(
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.n_positions,
        n_embd=hf_config.n_embd,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_inner=hf_config.n_inner,
    )
