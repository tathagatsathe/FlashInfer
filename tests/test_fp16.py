import torch

from flashinfer.model.gpt2 import GPT2Model
from flashinfer.weights.loader import config_from_hf, load_hf_weights


def test_fp16_logits_parity():
    config = config_from_hf("gpt2")
    model_fp32 = GPT2Model.from_config(config)
    load_hf_weights(model_fp32, "gpt2", dtype=torch.float32)
    model_fp32.eval()

    model_fp16 = GPT2Model.from_config(config)
    load_hf_weights(model_fp16, "gpt2", dtype=torch.float16)
    model_fp16.eval()

    input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])

    with torch.no_grad():
        logits_fp32 = model_fp32(input_ids)
        logits_fp16 = model_fp16(input_ids).float()

    assert torch.allclose(
        logits_fp32[0, -1, :],
        logits_fp16[0, -1, :],
        atol=1e-2,
        rtol=1e-2,
    )
