from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from flashinfer.cache.kv_cache import KVCache
from flashinfer.layers.llama_attention import LlamaAttention
from flashinfer.layers.rms_norm import RMSNorm
from flashinfer.layers.swiglu_mlp import SwiGLUMLP


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = residual + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(LlamaBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        seq_len = input_ids.size(1)
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                kv_cache=kv_cache,
                layer_idx=i,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        kv_cache.advance(seq_len)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                kv_cache=kv_cache,
                layer_idx=i,
                position_ids=position_ids,
            )
        kv_cache.advance(input_ids.size(1))
        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_config(cls, config: LlamaConfig) -> "LlamaModel":
        return cls(config)
