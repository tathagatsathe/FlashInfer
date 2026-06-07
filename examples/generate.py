#!/usr/bin/env python3
import argparse
import sys

from flashinfer.engine.factory import create_engine
from flashinfer.engine.generation_config import GenerationConfig


def main():
    parser = argparse.ArgumentParser(description="Generate text with FlashInfer")
    parser.add_argument("--prompt", type=str, default="Hello, world", help="Input prompt")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--stop", type=str, nargs="*", default=None, help="Stop sequences")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-context-len", type=int, default=None, help="Max context length")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    args = parser.parse_args()

    engine = create_engine(args.model, device=args.device, dtype=args.dtype)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_sequences=args.stop,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_context_len=args.max_context_len,
    )

    if args.stream:
        for piece in engine.generate_stream(args.prompt, config=config):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
    else:
        print(engine.generate(args.prompt, config=config))


if __name__ == "__main__":
    main()
