#!/usr/bin/env python3
"""
InfiniteContext-1B — Attention Memory Scaling Demo
===================================================

This script demonstrates the KV cache memory problem that MLA solves.
It runs Multi-Head Attention at increasing sequence lengths on your GPU
and shows you exactly where standard attention runs out of memory.

Usage:
    python scripts/demo_attention.py

What you'll see:
    1. A table showing KV cache size at various context lengths
    2. A live GPU memory measurement as we run attention
    3. The point where your GPU runs out of memory (the "wall")

After running this, you'll understand viscerally why MLA's 93% cache
compression is necessary for long-context inference.
"""

import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from training.src.attention_basics import (
    MultiHeadAttention,
    CachedMultiHeadAttention,
    analyze_kv_cache_scaling,
    create_causal_mask,
)


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def demo_1_theoretical_scaling():
    """
    Part 1: Show the theoretical KV cache scaling WITHOUT running anything.
    This helps you understand the math before seeing it on real hardware.
    """
    print_header("PART 1: Theoretical KV Cache Scaling (Standard MHA)")

    # These match a typical 1B-parameter model
    config = {
        "d_model": 2048,
        "n_heads": 16,
        "n_layers": 24,
        "dtype": torch.float16,
    }

    print(f"  Model config: d_model={config['d_model']}, "
          f"n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    print(f"  d_head = d_model / n_heads = {config['d_model'] // config['n_heads']}")
    print(f"  Dtype: FP16 (2 bytes per element)")
    print()

    results = analyze_kv_cache_scaling(**config)

    # Print table
    print(f"  {'Context Length':>15}  {'KV Cache (MB)':>15}  {'KV Cache (GB)':>15}  {'Fits 8GB GPU?':>15}")
    print(f"  {'-' * 15}  {'-' * 15}  {'-' * 15}  {'-' * 15}")

    for r in results:
        ctx = f"{r['seq_len'] // 1024}k tokens"
        fits = "✅ Yes" if r['cache_gb'] < 5.5 else "❌ No (OOM)"  # ~5.5GB available after model weights
        print(f"  {ctx:>15}  {r['cache_mb']:>12.1f} MB  {r['cache_gb']:>12.2f} GB  {fits:>15}")

    print()
    print("  📊 Key Insight:")
    print("     Standard MHA KV cache grows LINEARLY with context length.")
    print("     At 128k tokens, the cache alone is ~25 GB — 3x your total VRAM!")
    print()
    print("  💡 MLA Solution:")
    print("     Compress K and V into a shared latent vector (~93% smaller).")
    print("     128k tokens → ~1.7 GB instead of ~25 GB. Fits on your 2070 Super!")


def demo_2_live_gpu_measurement():
    """
    Part 2: Actually run MHA on your GPU and measure real memory usage.
    We'll increase sequence length until we either hit a limit or OOM.
    """
    print_header("PART 2: Live GPU Memory Measurement")

    if not torch.cuda.is_available():
        print("  ⏭️  No CUDA GPU available — showing CPU-only mode")
        print("     (Memory measurements won't reflect GPU constraints)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_mem / (1024**3)
        print(f"  GPU: {props.name} ({total_gb:.1f} GB VRAM)")
        torch.cuda.reset_peak_memory_stats()

    # Use a smaller config for live testing (so we don't OOM immediately)
    d_model = 512
    n_heads = 8
    n_layers = 1  # Single layer for demo (multiply results by 24 for full model)

    print(f"  Demo config: d_model={d_model}, n_heads={n_heads}, layers={n_layers}")
    print(f"  (Multiply memory by 24 to estimate a full 1B model)")
    print()

    model = MultiHeadAttention(d_model=d_model, n_heads=n_heads).to(device).half()
    model.eval()

    # Sequence lengths to test — we'll go until OOM or our max
    test_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    print(f"  {'Seq Length':>12}  {'KV Cache':>12}  {'GPU Alloc':>12}  {'GPU Peak':>12}  {'Status':>10}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    for seq_len in test_lengths:
        try:
            # Clear previous tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Create input
            x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.float16)
            mask = create_causal_mask(seq_len, device=device)

            # Forward pass
            with torch.no_grad():
                _ = model(x, mask=mask)

            # Measure
            kv_cache_bytes = model.count_kv_cache_bytes(1, seq_len)
            kv_mb = kv_cache_bytes / (1024**2)

            if torch.cuda.is_available():
                alloc_mb = torch.cuda.memory_allocated() / (1024**2)
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                alloc_mb = 0
                peak_mb = 0

            ctx_str = f"{seq_len:,}"
            print(f"  {ctx_str:>12}  {kv_mb:>9.1f} MB  {alloc_mb:>9.1f} MB  {peak_mb:>9.1f} MB  {'✅ OK':>10}")

            # Clean up
            del x, mask

        except torch.cuda.OutOfMemoryError:
            print(f"  {seq_len:>12}  {'—':>12}  {'—':>12}  {'—':>12}  {'❌ OOM':>10}")
            torch.cuda.empty_cache()
            break

        except Exception as e:
            print(f"  {seq_len:>12}  {'—':>12}  {'—':>12}  {'—':>12}  {'⚠️ ' + str(e)[:20]:>10}")
            break

    print()
    print("  📊 Note: These numbers are for 1 layer with d_model=512.")
    print("     A full 1B model (24 layers, d_model=2048) uses ~64x more cache.")


def demo_3_cached_generation():
    """
    Part 3: Simulate autoregressive generation to show how KV cache grows
    token by token during text generation.
    """
    print_header("PART 3: Autoregressive Generation (KV Cache Growth)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    d_model = 512
    n_heads = 8
    n_tokens_to_generate = 64

    print(f"  Simulating generation of {n_tokens_to_generate} tokens...")
    print(f"  Config: d_model={d_model}, n_heads={n_heads}")
    print()

    model = CachedMultiHeadAttention(d_model=d_model, n_heads=n_heads).to(device).to(dtype)
    model.eval()
    model.reset_cache()

    # Prefill with some context (like a prompt)
    prefill_len = 128
    print(f"  Step 0: Prefill {prefill_len} tokens (the 'prompt')...")
    prompt = torch.randn(1, prefill_len, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        _ = model(prompt)
    cache_kb = model.get_cache_memory_bytes() / 1024
    print(f"           Cache size: {cache_kb:.1f} KB ({model.cache_len} tokens cached)")
    print()

    # Generate tokens one at a time
    print(f"  {'Token #':>10}  {'Cache Tokens':>15}  {'Cache Size':>12}")
    print(f"  {'-' * 10}  {'-' * 15}  {'-' * 12}")

    for i in range(n_tokens_to_generate):
        new_token = torch.randn(1, 1, d_model, device=device, dtype=dtype)
        with torch.no_grad():
            _ = model(new_token)

        if (i + 1) % 8 == 0 or i == 0:  # Print every 8 tokens
            cache_kb = model.get_cache_memory_bytes() / 1024
            print(f"  {i + 1:>10}  {model.cache_len:>15}  {cache_kb:>9.1f} KB")

    print()
    final_cache_kb = model.get_cache_memory_bytes() / 1024
    print(f"  Final cache: {final_cache_kb:.1f} KB for {model.cache_len} tokens")
    print()
    print("  📊 Key Insight:")
    print("     The cache grows with EVERY token generated.")
    print("     In a real model with 24 layers, multiply this by 24.")
    print(f"     Full model estimate: {final_cache_kb * 24 / 1024:.2f} MB for {model.cache_len} tokens")
    print()

    # Clean up
    model.reset_cache()


def print_next_steps():
    """Print what to do after running this demo."""
    print_header("NEXT STEPS")
    print("  You've now seen the KV cache memory problem firsthand.")
    print()
    print("  What you learned:")
    print("    1. KV cache grows linearly with context length")
    print("    2. Standard MHA at 128k tokens needs ~25 GB (for a 1B model)")
    print("    3. Your RTX 2070 Super can handle ~8-32k with standard MHA")
    print("    4. Each generated token adds to the cache")
    print()
    print("  What's next (Phase 3 of the roadmap):")
    print("    → Implement Multi-Head Latent Attention (MLA)")
    print("    → Compress KV cache by 93% (128 MB/1k → 8 MB/1k)")
    print("    → That's what enables 128k+ context on your 8GB GPU!")
    print()
    print("  File to create: training/src/modeling_mla.py")
    print()


def main():
    print()
    print("🧠 InfiniteContext-1B — Attention Memory Scaling Demo")
    print("=" * 70)

    demo_1_theoretical_scaling()
    demo_2_live_gpu_measurement()
    demo_3_cached_generation()
    print_next_steps()


if __name__ == "__main__":
    main()
