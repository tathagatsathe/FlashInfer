import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    )


class GPT2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, intermediate_size)
        self.proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = gelu_new(x)
        return self.proj(x)
