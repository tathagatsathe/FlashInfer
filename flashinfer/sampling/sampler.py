from typing import Iterable, List, Optional, Set, Union

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Iterable[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    unique_ids: Set[int] = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] = logits[token_id] / penalty
        else:
            logits[token_id] = logits[token_id] * penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample the next token from last-position logits."""
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < top_values[-1], float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_next_token_batch(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    finished: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample next token for each row in a batch of logits (batch, vocab)."""
    batch_size = logits.size(0)
    tokens = []
    for i in range(batch_size):
        if finished is not None and finished[i]:
            tokens.append(0)
            continue
        tokens.append(
            sample_next_token(
                logits[i],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
        )
    return torch.tensor(tokens, dtype=torch.long, device=logits.device)
