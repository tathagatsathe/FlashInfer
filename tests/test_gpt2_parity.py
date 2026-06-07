import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashinfer.model.gpt2 import GPT2Model
from flashinfer.weights.loader import config_from_hf, load_hf_weights


MODEL_NAME = "gpt2"
PROMPT = "The capital of France is"


def test_logits_parity():
    config = config_from_hf(MODEL_NAME)
    model = GPT2Model.from_config(config)
    load_hf_weights(model, MODEL_NAME)
    model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")

    with torch.no_grad():
        our_logits = model(input_ids)
        hf_logits = hf_model(input_ids).logits

    our_last = our_logits[0, -1, :]
    hf_last = hf_logits[0, -1, :]

    assert torch.allclose(our_last, hf_last, atol=1e-4, rtol=1e-3)


def test_greedy_generation_parity():
    config = config_from_hf(MODEL_NAME)
    model = GPT2Model.from_config(config)
    load_hf_weights(model, MODEL_NAME)
    model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    generated = input_ids.tolist()[0]

    with torch.no_grad():
        for _ in range(5):
            tokens = torch.tensor([generated])
            our_logits = model(tokens)
            hf_logits = hf_model(tokens).logits

            our_next = int(torch.argmax(our_logits[0, -1, :]).item())
            hf_next = int(torch.argmax(hf_logits[0, -1, :]).item())
            assert our_next == hf_next
            generated.append(our_next)
