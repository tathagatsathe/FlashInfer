import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Sample the next token from last-position logits."""
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
