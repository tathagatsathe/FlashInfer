#!/usr/bin/env python3
import argparse
import sys

from flashinfer.engine.generator import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--prompt", type=str, default="Hello, world", help="Input prompt")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    args = parser.parse_args()

    engine = InferenceEngine.from_pretrained(args.model, device=args.device)
    if args.stream:
        for piece in engine.generate_stream(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
    else:
        print(engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ))


if __name__ == "__main__":
    main()
