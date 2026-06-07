import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from flashinfer.cache.kv_cache import KVCache


def _combine_masks(
    causal_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = causal_mask[:, :, :seq_len, :seq_len]
    if attention_mask is not None:
        pad = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
        mask = mask * pad
    min_value = torch.finfo(dtype).min
    return torch.where(mask == 0, min_value, torch.zeros_like(mask))


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_position: int = 1024):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_position, max_position)).view(
                1, 1, max_position, max_position
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(hidden, dim=2)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)

        if kv_cache is None:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            combined = _combine_masks(
                self.causal_mask, attention_mask, seq_len, attn_weights.dtype
            )
            attn_weights = attn_weights + combined
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        elif seq_len > 1:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            combined = _combine_masks(
                self.causal_mask, attention_mask, seq_len, attn_weights.dtype
            )
            attn_weights = attn_weights + combined
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            kv_cache.append(layer_idx, k, v)
        else:
            cache_end = kv_cache.append(layer_idx, k, v)
            cached_k, cached_v = kv_cache.get(layer_idx, end=cache_end)
            attn_weights = torch.matmul(q, cached_k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, cached_v)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        )
        return self.out_proj(attn_output)
