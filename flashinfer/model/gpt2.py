from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from flashinfer.cache.kv_cache import KVCache
from flashinfer.layers.attention import CausalSelfAttention
from flashinfer.layers.embeddings import GPT2Embeddings
from flashinfer.layers.mlp import GPT2MLP
from flashinfer.layers.norm import LayerNorm


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None

    @property
    def intermediate_size(self) -> int:
        return self.n_inner if self.n_inner is not None else 4 * self.n_embd


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden = config.n_embd
        self.ln_1 = LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, config.n_head, config.n_positions)
        self.ln_2 = LayerNorm(hidden)
        self.mlp = GPT2MLP(hidden, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if kv_cache is None:
            x = x + self.attn(self.ln_1(x))
        else:
            x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embeddings = GPT2Embeddings(
            config.vocab_size, config.n_embd, config.n_positions
        )
        self.blocks = nn.ModuleList(GPT2Block(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_prefill(self, input_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        x = self.embeddings(input_ids, position_offset=0)
        seq_len = input_ids.size(1)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i)
        kv_cache.advance(seq_len)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_decode(self, input_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        position_offset = kv_cache.seq_len
        x = self.embeddings(input_ids, position_offset=position_offset)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i)
        kv_cache.advance(input_ids.size(1))
        x = self.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_config(cls, config: GPT2Config) -> "GPT2Model":
        return cls(config)
