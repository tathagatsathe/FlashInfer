from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class BatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_lengths: List[int]


def left_pad_sequences(
    sequences: List[List[int]],
    pad_token_id: int,
    device: torch.device,
) -> BatchInput:
    prompt_lengths = [len(seq) for seq in sequences]
    max_len = max(prompt_lengths)
    batch_size = len(sequences)

    input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        pad_len = max_len - seq_len
        input_ids[i, pad_len:] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, pad_len:] = 1
        position_ids[i, pad_len:] = torch.arange(seq_len, device=device)

    return BatchInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prompt_lengths=prompt_lengths,
    )


def last_token_indices(prompt_lengths: List[int], padded_len: int) -> torch.Tensor:
    return torch.tensor(
        [padded_len - 1 for _ in prompt_lengths],
        dtype=torch.long,
    )
