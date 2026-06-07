from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class LayerKVCache:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class KVCache:
    layers: List[LayerKVCache] = field(default_factory=list)
    seq_len: int = 0

    @classmethod
    def empty(
        cls,
        num_layers: int,
        batch: int,
        n_head: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        layers = []
        for _ in range(num_layers):
            keys = torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype)
            values = torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype)
            layers.append(LayerKVCache(keys=keys, values=values))
        return cls(layers=layers, seq_len=0)

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> int:
        start = self.seq_len
        end = start + key.size(2)
        self.layers[layer_idx].keys[:, :, start:end, :] = key
        self.layers[layer_idx].values[:, :, start:end, :] = value
        return end

    def get(self, layer_idx: int, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self.seq_len if end is None else end
        layer = self.layers[layer_idx]
        return layer.keys[:, :, :end, :], layer.values[:, :, :end, :]

    def advance(self, num_tokens: int) -> None:
        self.seq_len += num_tokens
