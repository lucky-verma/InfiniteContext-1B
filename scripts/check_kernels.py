"""Check Triton kernels against PyTorch; interpreter results are not GPU results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interpreter', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.interpreter:
        os.environ['TRITON_INTERPRET'] = '1'
    import torch
    import triton
    import triton.testing
    from kernels.ops import mla_decode, rms_norm, rope
    from training.src.modeling_mla import rotate
    torch.set_num_threads(1)
    torch.manual_seed(51)
    device = 'cpu' if args.interpreter else 'cuda'
    rows = []
    for dtype in (torch.float32, torch.float16):
        for width in (12, 64, 256):
            x = torch.randn(7, width, device=device, dtype=dtype)
            w = torch.randn(width, device=device, dtype=dtype)
            result = rms_norm(x, w)
            expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
            torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)
            positions = torch.tensor([0, 1, 63, 511, 2048, 8192, 32768], device=device)
            actual_rope = rope(x, positions)
            expected_rope = rotate(x.float(), positions, 10000.0).to(dtype)
            torch.testing.assert_close(actual_rope, expected_rope, rtol=8e-3, atol=8e-3)
            rows.append({'op': 'rmsnorm+rope', 'dtype': str(dtype), 'width': width,
                         'rope_max_abs_error': (actual_rope.float()-expected_rope.float()).abs().max().item()})
        for length in (1, 33, 257):
            q = torch.randn(2, 3, 16, device=device, dtype=dtype)
            qr = torch.randn(2, 3, 8, device=device, dtype=dtype)
            kv = torch.randn(2, length, 16, device=device, dtype=dtype)
            kr = torch.randn(2, length, 8, device=device, dtype=dtype)
            scores = (torch.einsum('bhr,bnr->bhn', q.float(), kv.float()) + torch.einsum('bhd,bnd->bhn', qr.float(), kr.float())) / 4
            expected = torch.einsum('bhn,bnr->bhr', scores.softmax(-1), kv.float()).to(dtype)
            actual = mla_decode(q, qr, kv, kr, scale=0.25)
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
            row = {'op': 'mla_decode', 'dtype': str(dtype), 'tokens': length,
                   'max_abs_error': (actual.float()-expected.float()).abs().max().item()}
            if not args.interpreter:
                row['median_ms'] = triton.testing.do_bench(lambda: mla_decode(q, qr, kv, kr, scale=0.25))
            rows.append(row)
    result = {'status': 'passed', 'execution': 'CPU interpreter' if args.interpreter else torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'triton': triton.__version__, 'cases': rows,
              'kernel_sha256': hashlib.sha256((ROOT/'kernels/ops.py').read_bytes()).hexdigest(),
              'scope': 'numerical reference checks; no end-to-end speedup claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(f"PASS: {len(rows)} numerical cases on {result['execution']}")


if __name__ == '__main__':
    main()
