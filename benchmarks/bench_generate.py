#!/usr/bin/env python3
"""Benchmark prefill latency, decode throughput, and memory usage."""

import argparse
import time

import torch

from flashinfer.engine.generator import InferenceEngine


def benchmark(args):
    engine = InferenceEngine.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )
    prompt = args.prompt * max(1, args.prompt_tokens // 5)
    input_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(args.device)

    if args.prompt_tokens > input_ids.size(1):
        repeat = (args.prompt_tokens // input_ids.size(1)) + 1
        ids = input_ids[0].tolist() * repeat
        input_ids = torch.tensor([ids[: args.prompt_tokens]], device=args.device)

    kv_cache = engine._make_kv_cache(batch=1)

    torch.cuda.synchronize() if args.device == "cuda" else None
    start = time.perf_counter()
    engine.model.forward_prefill(input_ids, kv_cache)
    torch.cuda.synchronize() if args.device == "cuda" else None
    prefill_ms = (time.perf_counter() - start) * 1000

    decode_tokens = args.decode_tokens
    token = torch.tensor([[0]], device=args.device, dtype=torch.long)
    pos = torch.tensor([[kv_cache.seq_len]], device=args.device, dtype=torch.long)

    torch.cuda.synchronize() if args.device == "cuda" else None
    start = time.perf_counter()
    for _ in range(decode_tokens):
        pos[0, 0] = kv_cache.seq_len
        logits = engine.model.forward_decode(token, kv_cache, position_ids=pos)
        token = torch.argmax(logits[0, -1, :]).view(1, 1)
    torch.cuda.synchronize() if args.device == "cuda" else None
    decode_s = time.perf_counter() - start

    param_bytes = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    cache_bytes = sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in kv_cache.layers
    )

    print(f"Model: {args.model}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    print(f"Prefill: {input_ids.size(1)} tokens in {prefill_ms:.2f} ms")
    print(f"Decode: {decode_tokens / decode_s:.2f} tokens/sec ({decode_tokens} tokens)")
    print(f"Model memory: {param_bytes / 1e6:.1f} MB")
    print(f"KV cache memory: {cache_bytes / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer generation")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The quick brown fox ")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
