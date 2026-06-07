import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashinfer.layers.rope import RotaryEmbedding, apply_rotary_pos_emb

if TYPE_CHECKING:
    from flashinfer.cache.kv_cache import KVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_position, rope_theta)

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        q, k, v = self._project(x)

        if position_ids is None:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
            if kv_cache is not None and seq_len == 1:
                pos = torch.full((batch, 1), kv_cache.seq_len, device=x.device, dtype=torch.long)
        else:
            pos = position_ids

        max_pos = int(pos.max().item()) + 1
        cos, sin = self.rotary(max_pos, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, pos)

        scale = 1.0 / math.sqrt(self.head_dim)

        if kv_cache is None or seq_len > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
            causal = causal.view(1, 1, seq_len, seq_len)
            if attention_mask is not None:
                pad = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
                causal = causal * pad
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + torch.where(
                causal == 0, min_value, torch.zeros_like(causal)
            )
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            if kv_cache is not None:
                kv_cache.append(layer_idx, k, v)
        else:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            cache_end = kv_cache.append(layer_idx, k, v)
            cached_k, cached_v = kv_cache.get(layer_idx, end=cache_end)
            attn_weights = torch.matmul(q, cached_k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, cached_v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.o_proj(attn_output)
