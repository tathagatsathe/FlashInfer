from flashinfer.engine.factory import create_engine


def test_create_engine_gpt2():
    engine = create_engine("gpt2", device="cpu")
    text = engine.generate("Hi", max_new_tokens=2, temperature=0.0)
    assert isinstance(text, str)
    assert len(text) >= 2
