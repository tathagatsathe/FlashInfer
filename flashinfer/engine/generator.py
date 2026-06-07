from typing import Optional

import torch
from transformers import AutoTokenizer

from flashinfer.cache.kv_cache import KVCache
from flashinfer.model.gpt2 import GPT2Model
from flashinfer.sampling.sampler import sample_next_token
from flashinfer.weights.loader import config_from_hf, load_hf_weights


class InferenceEngine:
    def __init__(
        self,
        model: GPT2Model,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cpu") -> "InferenceEngine":
        config = config_from_hf(model_name)
        model = GPT2Model.from_config(config)
        load_hf_weights(model, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer, device)

    def _make_kv_cache(self, batch: int = 1) -> KVCache:
        config = self.model.config
        return KVCache.empty(
            num_layers=config.n_layer,
            batch=batch,
            n_head=config.n_head,
            head_dim=config.n_embd // config.n_head,
            max_len=config.n_positions,
            device=torch.device(self.device),
            dtype=next(self.model.parameters()).dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.tolist()[0]
        eos_token_id = self.tokenizer.eos_token_id

        kv_cache = self._make_kv_cache(batch=input_ids.size(0))
        logits = self.model.forward_prefill(input_ids, kv_cache)
        next_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

            token = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
            logits = self.model.forward_decode(token, kv_cache)
            next_logits = logits[0, -1, :]

        return self.tokenizer.decode(generated, skip_special_tokens=True)
