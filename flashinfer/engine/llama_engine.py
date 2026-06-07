from typing import Iterator, List, Optional, Union

import torch
from transformers import AutoTokenizer

from flashinfer.cache.kv_cache import KVCache
from flashinfer.engine.batching import left_pad_sequences
from flashinfer.engine.generation_config import GenerationConfig
from flashinfer.model.llama import LlamaModel
from flashinfer.sampling.sampler import (
    apply_repetition_penalty,
    sample_next_token,
    sample_next_token_batch,
)
from flashinfer.weights.llama_loader import config_from_hf_llama, load_hf_llama_weights


class LlamaInferenceEngine:
    def __init__(
        self,
        model: LlamaModel,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.to(device=device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cpu",
        dtype: Union[str, torch.dtype] = "float32",
    ) -> "LlamaInferenceEngine":
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        config = config_from_hf_llama(model_name)
        model = LlamaModel.from_config(config)
        load_hf_llama_weights(model, model_name, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model, tokenizer, device, dtype)

    def _make_kv_cache(self, batch: int = 1) -> KVCache:
        config = self.model.config
        return KVCache.empty(
            num_layers=config.num_hidden_layers,
            batch=batch,
            n_head=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_len=config.max_position_embeddings,
            device=torch.device(self.device),
            dtype=self.dtype,
        )

    def _make_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        return gen

    def _check_context_len(self, seq_len: int, max_context_len: Optional[int]) -> None:
        if max_context_len is not None and seq_len >= max_context_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_context_len {max_context_len}"
            )
        if seq_len >= self.model.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max positions "
                f"{self.model.config.max_position_embeddings}"
            )

    def _should_stop(
        self,
        text: str,
        stop_sequences: Optional[List[str]],
    ) -> bool:
        if not stop_sequences:
            return False
        return any(text.endswith(stop) for stop in stop_sequences)

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig(**kwargs)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.tolist()[0]
        self._check_context_len(len(generated), cfg.max_context_len)
        eos_token_id = self.tokenizer.eos_token_id
        rng = self._make_generator(cfg.seed)

        kv_cache = self._make_kv_cache(batch=1)
        logits = self.model.forward_prefill(input_ids, kv_cache)
        next_logits = logits[0, -1, :]

        for _ in range(cfg.max_new_tokens):
            self._check_context_len(len(generated), cfg.max_context_len)
            next_logits = apply_repetition_penalty(
                next_logits, generated, cfg.repetition_penalty
            )
            next_token = sample_next_token(
                next_logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                generator=rng,
            )
            generated.append(next_token)
            yield self.tokenizer.decode([next_token], skip_special_tokens=True)

            full_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if self._should_stop(full_text, cfg.stop_sequences):
                break
            if eos_token_id is not None and next_token == eos_token_id:
                break

            token = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
            pos = torch.tensor([[kv_cache.seq_len]], device=self.device, dtype=torch.long)
            logits = self.model.forward_decode(token, kv_cache, position_ids=pos)
            next_logits = logits[0, -1, :]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        cfg = config or GenerationConfig(**kwargs)
        chunks = list(self.generate_stream(prompt, config=cfg))
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prefix = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return prefix + "".join(chunks)

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> List[str]:
        cfg = config or GenerationConfig(**kwargs)
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
        rng = self._make_generator(cfg.seed)
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

        for _ in range(cfg.max_new_tokens):
            if finished.all():
                break
            for i in range(batch_size):
                if not finished[i]:
                    self._check_context_len(len(generated[i]), cfg.max_context_len)
                    next_logits[i] = apply_repetition_penalty(
                        next_logits[i], generated[i], cfg.repetition_penalty
                    )

            next_tokens = sample_next_token_batch(
                next_logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                generator=rng,
                finished=finished,
            )

            for i in range(batch_size):
                if finished[i]:
                    continue
                token = int(next_tokens[i].item())
                generated[i].append(token)
                text = self.tokenizer.decode(generated[i], skip_special_tokens=True)
                if self._should_stop(text, cfg.stop_sequences):
                    finished[i] = True
                elif eos_token_id is not None and token == eos_token_id:
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
