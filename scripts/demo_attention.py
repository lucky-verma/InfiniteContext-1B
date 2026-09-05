"""A bounded synthetic attention/cache check. This does not run a language model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from training.src.attention_basics import CachedMultiHeadAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--prefill', type=int, default=32)
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--memory-limit-mib', type=int, default=128)
    args = parser.parse_args()
    if args.prefill < 1 or args.tokens < 0 or args.d_model < 1 or args.heads < 1 or args.memory_limit_mib < 1:
        parser.error('prefill, dimensions and memory limit must be positive; tokens must be nonnegative')
    if args.d_model % args.heads:
        parser.error('d-model must be divisible by heads')
    length = args.prefill + args.tokens
    estimate = 16 * args.d_model ** 2 + 8 * args.heads * length ** 2 + 32 * length * args.d_model
    if estimate > args.memory_limit_mib * 1024 ** 2:
        parser.error('synthetic workload exceeds the configured memory estimate; reduce its dimensions')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = CachedMultiHeadAttention(args.d_model, args.heads, max_seq_len=length).to(args.device).eval()
    with torch.no_grad():
        model(torch.randn(1, args.prefill, args.d_model, device=args.device))
        prefill_bytes = model.get_cache_memory_bytes()
        for _ in range(args.tokens):
            model(torch.randn(1, 1, args.d_model, device=args.device))
    actual = model.get_cache_memory_bytes()
    expected = model.count_kv_cache_bytes(1, length, dtype=torch.float32)
    assert actual == expected
    print(json.dumps({'kind': 'synthetic_attention_check', 'device': args.device,
                      'prefill_tokens': args.prefill, 'decode_steps': args.tokens,
                      'cache_tokens': model.cache_len, 'prefill_cache_bytes': prefill_bytes,
                      'cache_bytes': actual, 'expected_cache_bytes': expected,
                      'scope': 'one untrained attention layer; cache bytes exclude weights and temporary allocations'}, indent=2))


if __name__ == '__main__':
    main()
