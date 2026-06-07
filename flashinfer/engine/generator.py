from typing import List, Optional

import torch
from transformers import AutoTokenizer

from flashinfer.cache.kv_cache import KVCache
from flashinfer.engine.batching import left_pad_sequences
from flashinfer.model.gpt2 import GPT2Model
from flashinfer.sampling.sampler import sample_next_token, sample_next_token_batch
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
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        if not prompts:
            return []

        sequences = [self.tokenizer.encode(p) for p in prompts]
        batch = left_pad_sequences(
            sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            device=torch.device(self.device),
        )
        batch_size = len(prompts)
        generated: List[List[int]] = [seq.copy() for seq in sequences]
        eos_token_id = self.tokenizer.eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        kv_cache = self._make_kv_cache(batch=batch_size)
        logits = self.model.forward_prefill(
            batch.input_ids,
            kv_cache,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
        )

        padded_len = batch.input_ids.size(1)
        next_logits = logits[torch.arange(batch_size), padded_len - 1, :]

        for _ in range(max_new_tokens):
            if finished.all():
                break

            next_tokens = sample_next_token_batch(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                finished=finished,
            )

            for i in range(batch_size):
                if finished[i]:
                    continue
                token = int(next_tokens[i].item())
                generated[i].append(token)
                if eos_token_id is not None and token == eos_token_id:
                    finished[i] = True

            if finished.all():
                break

            position_ids = torch.full(
                (batch_size, 1),
                kv_cache.seq_len,
                dtype=torch.long,
                device=self.device,
            )
            decode_logits = self.model.forward_decode(
                next_tokens.unsqueeze(1),
                kv_cache,
                position_ids=position_ids,
            )
            next_logits = decode_logits[:, -1, :]

        return [
            self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated
        ]
