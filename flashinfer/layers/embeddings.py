import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_position, hidden_size)
        self.max_position = max_position

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_position:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max position {self.max_position}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_emb(input_ids) + self.pos_emb(positions)
