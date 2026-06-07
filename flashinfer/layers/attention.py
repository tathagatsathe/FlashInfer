import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(hidden, dim=2)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(mask == 0, min_value)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        )
        return self.out_proj(attn_output)
