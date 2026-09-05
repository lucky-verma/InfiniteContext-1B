import copy
import unittest

import torch

from training.src.modeling_mla import MLAConfig, MLAAttention, MLALanguageModel


def small_config():
    return MLAConfig(vocab_size=31, hidden_size=16, num_layers=2, intermediate_size=32,
                     num_heads=2, kv_rank=8, q_rank=8, nope_dim=4, rope_dim=4, value_dim=8)


class LatentAttentionContract(unittest.TestCase):
    def test_absorption_matches_materialized_reference_and_gradients(self):
        torch.manual_seed(11)
        model = MLAAttention(small_config()).double()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        actual, _ = model(x)
        expected, _ = reference(y, implementation='reference')
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-10)
        for left, right in zip(model.parameters(), reference.parameters()):
            self.assertIsNotNone(left.grad)
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-8, atol=1e-10)

    def test_cached_decode_and_bounded_rebasing(self):
        torch.manual_seed(19)
        config = small_config()
        model = MLALanguageModel(config).double().eval()
        tokens = torch.randint(config.vocab_size, (2, 9))
        with torch.no_grad():
            full = model(tokens)
            cache, outputs = None, []
            for chunk in (tokens[:, :4], tokens[:, 4:7], tokens[:, 7:]):
                logits, cache = model(chunk, cache, use_cache=True)
                outputs.append(logits)
            torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=1e-8, atol=1e-9)
            expected_bytes = 2 * 9 * (config.kv_rank + config.rope_dim) * 8
            self.assertTrue(all(item.nbytes == expected_bytes for item in cache))
            attention = model.layers[0].attention
            absorbed_cache, reference_cache = None, None
            for index in range(24):
                x = torch.randn(2, 1, config.hidden_size, dtype=torch.float64)
                actual, absorbed_cache = attention(x, absorbed_cache, use_cache=True, window=8, anchors=2)
                expected, reference_cache = attention(x, reference_cache, use_cache=True, implementation='reference', window=8, anchors=2)
                torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
                self.assertLessEqual(absorbed_cache.length, 8)
                self.assertEqual(absorbed_cache.total_tokens, index + 1)

    def test_reference_configuration_is_one_billion_class(self):
        with torch.device('meta'):
            model = MLALanguageModel(MLAConfig())
        count = sum(p.numel() for p in model.parameters())
        self.assertGreater(count, 950_000_000)
        self.assertLess(count, 1_050_000_000)


if __name__ == '__main__':
    unittest.main()
