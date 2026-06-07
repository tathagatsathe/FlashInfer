import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig as HFLlamaConfig

from flashinfer.model.llama import LlamaConfig, LlamaModel


def config_from_hf_llama(model_name: str) -> LlamaConfig:
    hf_config: HFLlamaConfig = AutoConfig.from_pretrained(model_name)
    return LlamaConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
    )


def load_hf_llama_weights(
    model: LlamaModel,
    model_name: str,
    dtype: torch.dtype = torch.float32,
) -> LlamaModel:
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hf_state = hf_model.state_dict()

    model.embed_tokens.weight.data.copy_(hf_state["model.embed_tokens.weight"].to(dtype))
    model.norm.weight.data.copy_(hf_state["model.norm.weight"].to(dtype))
    model.lm_head.weight.data.copy_(hf_state["lm_head.weight"].to(dtype))

    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        layer.input_layernorm.weight.data.copy_(
            hf_state[f"{prefix}.input_layernorm.weight"].to(dtype)
        )
        layer.post_attention_layernorm.weight.data.copy_(
            hf_state[f"{prefix}.post_attention_layernorm.weight"].to(dtype)
        )
        layer.self_attn.q_proj.weight.data.copy_(
            hf_state[f"{prefix}.self_attn.q_proj.weight"].to(dtype)
        )
        layer.self_attn.k_proj.weight.data.copy_(
            hf_state[f"{prefix}.self_attn.k_proj.weight"].to(dtype)
        )
        layer.self_attn.v_proj.weight.data.copy_(
            hf_state[f"{prefix}.self_attn.v_proj.weight"].to(dtype)
        )
        layer.self_attn.o_proj.weight.data.copy_(
            hf_state[f"{prefix}.self_attn.o_proj.weight"].to(dtype)
        )
        layer.mlp.gate_proj.weight.data.copy_(
            hf_state[f"{prefix}.mlp.gate_proj.weight"].to(dtype)
        )
        layer.mlp.up_proj.weight.data.copy_(
            hf_state[f"{prefix}.mlp.up_proj.weight"].to(dtype)
        )
        layer.mlp.down_proj.weight.data.copy_(
            hf_state[f"{prefix}.mlp.down_proj.weight"].to(dtype)
        )

    del hf_model
    return model
