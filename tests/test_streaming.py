from flashinfer.engine.generator import InferenceEngine


def test_generate_stream_matches_generate():
    engine = InferenceEngine.from_pretrained("gpt2", device="cpu")
    prompt = "Hello"
    streamed = "".join(
        engine.generate_stream(prompt, max_new_tokens=5, temperature=0.0)
    )
    full = engine.generate(prompt, max_new_tokens=5, temperature=0.0)
    input_prefix = engine.tokenizer.decode(
        engine.tokenizer.encode(prompt), skip_special_tokens=True
    )
    assert full == input_prefix + streamed
