"""Run with: python -m unittest discover -s tests -v"""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch.nn import functional as F

from training.src.attention_basics import (
    CachedMultiHeadAttention,
    MultiHeadAttention,
    analyze_kv_cache_scaling,
    scaled_dot_product_attention,
)


class AttentionContract(unittest.TestCase):
    def test_development_commands_report_validation_and_failures(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, 'scripts/demo_attention.py', '--prefill', '8', '--tokens', '4'],
                                cwd=root, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['cache_bytes'], 6144)
        self.assertEqual(report['cache_bytes'], report['expected_cache_bytes'])
        result = subprocess.run([sys.executable, 'scripts/demo_attention.py', '--prefill', '0'],
                                cwd=root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, 'scripts/verify_gpu.py', '--device', 'cpu'],
                                cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'passed')
        if not torch.cuda.is_available():
            result = subprocess.run([sys.executable, 'scripts/verify_gpu.py', '--device', 'cuda'],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)['status'], 'failed')

    def test_attention_matches_torch_with_masked_rows(self):
        torch.manual_seed(17)
        q, k, v = [torch.randn(2, 3, 5, 4, dtype=torch.float64) for _ in range(3)]
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        mask[2] = False
        for candidate in (None, mask, mask.float()):
            reference_mask = None if candidate is None else candidate.bool()
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=reference_mask)
            actual = scaled_dot_product_attention(q, k, v, candidate)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_prefill_and_decoding_match_pytorch_multihead_attention(self):
        torch.manual_seed(23)
        model = CachedMultiHeadAttention(16, 4, max_seq_len=7).double().eval()
        reference = torch.nn.MultiheadAttention(16, 4, bias=False, batch_first=True).double().eval()
        x = torch.randn(2, 7, 16, dtype=torch.float64)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([model.W_Q.weight, model.W_K.weight, model.W_V.weight]))
            reference.out_proj.weight.copy_(model.W_O.weight)
            future = torch.ones(7, 7, dtype=torch.bool).triu(1)
            expected, _ = reference(x, x, x, attn_mask=future, need_weights=False)
            for chunks in ((7,), (3, 2, 2), (1,) * 7):
                model.reset_cache()
                outputs, offset = [], 0
                for width in chunks:
                    outputs.append(model(x[:, offset:offset + width]))
                    offset += width
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-10, atol=1e-10)
                self.assertEqual(model.cache_len, 7)
                self.assertEqual(model.get_cache_memory_bytes(), 2 * x.numel() * x.element_size())
            old_cache = model.cache_k
            with self.assertRaises(ValueError):
                model(x[:, :1])
            self.assertIs(model.cache_k, old_cache)
            self.assertEqual(model.cache_len, 7)
            model.reset_cache()
            self.assertEqual(model.get_cache_memory_bytes(), 0)
            model(x[:, :1])
            with self.assertRaises(ValueError):
                model(x[:1, :1])
            self.assertEqual(model.cache_len, 1)

    def test_invalid_dimensions_and_inputs_fail_before_cache_changes(self):
        for args in ((0, 1), (8, 0), (-8, 2), (9, 2)):
            with self.assertRaises(ValueError):
                MultiHeadAttention(*args)
        with self.assertRaises(ValueError):
            CachedMultiHeadAttention(8, 2, max_seq_len=0)
        model = CachedMultiHeadAttention(8, 2)
        for shape in ((1, 8), (0, 1, 8), (1, 0, 8), (1, 1, 9)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))
            self.assertEqual(model.cache_len, 0)

    def test_cache_estimate_tracks_tensor_storage(self):
        model = CachedMultiHeadAttention(16, 4).double().eval()
        with torch.no_grad():
            model(torch.zeros(1, 5, 16, dtype=torch.float64))
        estimate = analyze_kv_cache_scaling(16, 4, 1, torch.float64, [5])[0]
        self.assertEqual(estimate['cache_bytes'], model.get_cache_memory_bytes())


if __name__ == '__main__':
    unittest.main()
