from typing import Union

import torch
from transformers import AutoConfig

from flashinfer.engine.generator import InferenceEngine
from flashinfer.engine.llama_engine import LlamaInferenceEngine


def create_engine(
    model_name: str,
    device: str = "cpu",
    dtype: Union[str, torch.dtype] = "float32",
) -> Union[InferenceEngine, LlamaInferenceEngine]:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "llama":
        return LlamaInferenceEngine.from_pretrained(model_name, device=device, dtype=dtype)
    if config.model_type == "gpt2":
        return InferenceEngine.from_pretrained(model_name, device=device, dtype=dtype)
    raise ValueError(f"Unsupported model type: {config.model_type}")
