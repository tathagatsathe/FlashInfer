from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    max_context_len: Optional[int] = None
