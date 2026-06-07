import torch
from transformers import AutoTokenizer

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

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.tolist()[0]
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            tokens = torch.tensor([generated], device=self.device, dtype=torch.long)
            logits = self.model(tokens)
            next_logits = logits[0, -1, :]
            next_token = sample_next_token(next_logits, temperature=temperature)
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)
