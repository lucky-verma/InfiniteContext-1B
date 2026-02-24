#!/usr/bin/env python3
"""
InfiniteContext-1B — GPU & Environment Verification
====================================================

Run this first to confirm your hardware and software stack is ready.

What this script checks:
  1. Python version (need 3.10+)
  2. PyTorch version & CUDA availability
  3. GPU details (name, VRAM, compute capability)
  4. Simple GPU computation (matrix multiply) to verify CUDA works
  5. Triton availability for custom kernel development

Usage:
    python scripts/verify_gpu.py
"""

import sys
import time


def check_python():
    """Verify Python version >= 3.10."""
    print("=" * 60)
    print("1. PYTHON ENVIRONMENT")
    print("=" * 60)
    v = sys.version_info
    print(f"   Python version: {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) >= (3, 10):
        print("   ✅ Python 3.10+ requirement met")
    else:
        print("   ❌ Need Python 3.10+, please upgrade")
    print()


def check_pytorch():
    """Verify PyTorch installation and CUDA support."""
    print("=" * 60)
    print("2. PYTORCH & CUDA")
    print("=" * 60)
    try:
        import torch

        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available:  {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   CUDA version:    {torch.version.cuda}")
            print(f"   cuDNN version:   {torch.backends.cudnn.version()}")
            print(f"   cuDNN enabled:   {torch.backends.cudnn.enabled}")
            print("   ✅ CUDA is ready")
        else:
            print("   ❌ CUDA not available — check your NVIDIA drivers")
            print("      Run: nvidia-smi  (should show your GPU)")
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("      Run: pip install torch>=2.4.0")
    print()


def check_gpu():
    """Print detailed GPU information."""
    print("=" * 60)
    print("3. GPU HARDWARE")
    print("=" * 60)
    try:
        import torch

        if not torch.cuda.is_available():
            print("   ⏭️  Skipped (no CUDA)")
            print()
            return

        n_gpus = torch.cuda.device_count()
        print(f"   GPU count: {n_gpus}")
        print()

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_mem / (1024**3)
            print(f"   --- GPU {i} ---")
            print(f"   Name:               {props.name}")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Total VRAM:         {total_mem_gb:.1f} GB")
            print(f"   SM Count:           {props.multi_processor_count}")
            print()

            # Check compute capability requirement (need 7.0+)
            cc = props.major + props.minor / 10
            if cc >= 7.0:
                print(f"   ✅ Compute Capability {cc:.1f} meets 7.0+ requirement")
            else:
                print(f"   ⚠️  Compute Capability {cc:.1f} — Triton needs 7.0+")

            # Memory context for the project
            print()
            print("   📊 Memory Budget for InfiniteContext-1B:")
            print(f"      Model weights (1B, FP16):  ~2.0 GB")
            print(f"      Available for KV cache:    ~{total_mem_gb - 2.5:.1f} GB")
            print(f"      MLA KV cache per 1k tokens: ~8 MB")
            est_max_ctx = int((total_mem_gb - 2.5) * 1000 / 8)
            print(f"      Estimated max context:     ~{est_max_ctx}k tokens (with MLA)")
    except ImportError:
        print("   ⏭️  Skipped (no PyTorch)")
    print()


def check_gpu_compute():
    """Run a simple matrix multiply on GPU to verify CUDA computation works."""
    print("=" * 60)
    print("4. GPU COMPUTATION TEST")
    print("=" * 60)
    try:
        import torch

        if not torch.cuda.is_available():
            print("   ⏭️  Skipped (no CUDA)")
            print()
            return

        device = torch.device("cuda:0")

        # Matrix multiply: (4096 x 4096) @ (4096 x 4096)
        # This is roughly the size of a single linear layer in a 1B model
        size = 4096
        print(f"   Running: ({size}x{size}) @ ({size}x{size}) matmul on GPU...")

        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Timed run
        start = time.perf_counter()
        n_runs = 10
        for _ in range(n_runs):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs

        # Calculate TFLOPS (2 * N^3 FLOPs for matmul)
        flops = 2 * size**3
        tflops = flops / elapsed / 1e12
        print(f"   Time per matmul:  {elapsed * 1000:.2f} ms")
        print(f"   Throughput:       {tflops:.1f} TFLOPS (FP16)")
        print(f"   Output shape:     {c.shape}")
        print(f"   Output dtype:     {c.dtype}")
        print("   ✅ GPU computation works!")

        # Show memory usage after the test
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print()
        print(f"   Memory allocated: {alloc:.0f} MB")
        print(f"   Memory reserved:  {reserved:.0f} MB")

    except Exception as e:
        print(f"   ❌ GPU compute failed: {e}")
    print()


def check_triton():
    """Check if Triton is installed and can compile a simple kernel."""
    print("=" * 60)
    print("5. TRITON (Custom Kernel Development)")
    print("=" * 60)
    try:
        import triton
        import triton.language as tl

        print(f"   Triton version: {triton.__version__}")

        # Try compiling a trivial kernel to verify the compiler toolchain works
        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
            idx = tl.arange(0, n)
            x = tl.load(x_ptr + idx)
            y = tl.load(y_ptr + idx)
            tl.store(out_ptr + idx, x + y)

        import torch

        if torch.cuda.is_available():
            x = torch.ones(128, device="cuda")
            y = torch.ones(128, device="cuda")
            out = torch.empty(128, device="cuda")
            _add_kernel[(1,)](x, y, out, n=128)
            assert torch.allclose(out, torch.full((128,), 2.0, device="cuda"))
            print("   ✅ Triton kernel compilation & execution works!")
        else:
            print("   ⚠️  Triton installed but can't test without CUDA")

    except ImportError:
        print("   ❌ Triton not installed")
        print("      Run: pip install triton>=3.0.0")
    except Exception as e:
        print(f"   ⚠️  Triton installed but kernel test failed: {e}")
        print("      This might be a CUDA toolkit version issue")
    print()


def check_einops():
    """Quick check for einops (used for readable tensor operations)."""
    print("=" * 60)
    print("6. EINOPS")
    print("=" * 60)
    try:
        import einops

        print(f"   einops version: {einops.__version__}")
        print("   ✅ Ready")
    except ImportError:
        print("   ❌ Not installed — run: pip install einops")
    print()


def main():
    print()
    print("🔍 InfiniteContext-1B — Environment Verification")
    print("=" * 60)
    print()

    check_python()
    check_pytorch()
    check_gpu()
    check_gpu_compute()
    check_triton()
    check_einops()

    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print()
    print("If all checks passed (✅), you're ready to start!")
    print("Next step: python scripts/demo_attention.py")
    print()


if __name__ == "__main__":
    main()
