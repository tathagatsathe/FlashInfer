from typing import Optional

import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_position, hidden_size)
        self.max_position = max_position

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if position_ids is None:
            end_position = position_offset + seq_len
            if end_position > self.max_position:
                raise ValueError(
                    f"Sequence length {end_position} exceeds max position {self.max_position}"
                )
            position_ids = torch.arange(
                position_offset, end_position, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
        else:
            if position_ids.max().item() >= self.max_position:
                raise ValueError(
                    f"Position id exceeds max position {self.max_position}"
                )

        return self.token_emb(input_ids) + self.pos_emb(position_ids)
