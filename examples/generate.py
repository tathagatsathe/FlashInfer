#!/usr/bin/env python3
import argparse

from flashinfer.engine.generator import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--prompt", type=str, default="Hello, world", help="Input prompt")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    engine = InferenceEngine.from_pretrained(args.model, device=args.device)
    text = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
