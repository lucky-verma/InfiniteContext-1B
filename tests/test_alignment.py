import math
from pathlib import Path
import tempfile
import unittest

import torch

from training.alignment import PreferencePairs, completion_logps, dpo_loss


class AlignmentContract(unittest.TestCase):
    def test_completion_mask_and_dpo_update_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pairs.jsonl'
            path.write_text('{"prompt":[1,2],"chosen":[3,4],"rejected":[5]}\n')
            pairs = PreferencePairs(path, 8, 8, 'dpo')
            inputs, labels = pairs.collate([pairs[0]])
            self.assertEqual(inputs.tolist(), [[1, 2, 3], [1, 2, 0]])
            self.assertEqual(labels.tolist(), [[-100, 3, 4], [-100, 5, -100]])
            logits = torch.zeros(2, 3, 8, requires_grad=True)
            scores = completion_logps(logits, labels)
            torch.testing.assert_close(scores, torch.tensor([-2 * math.log(8), -math.log(8)]))
            (-scores.sum()).backward()
            self.assertTrue(torch.all(logits.grad[labels == -100] == 0))
            policy = scores.detach().clone().requires_grad_()
            reference = scores.detach().clone().requires_grad_()
            loss = dpo_loss(policy, reference, beta=0.1)
            self.assertAlmostEqual(loss.item(), math.log(2), places=6)
            loss.backward()
            self.assertLess(policy.grad[0].item(), 0)
            self.assertGreater(policy.grad[1].item(), 0)
            self.assertIsNone(reference.grad)
            self.assertLess(dpo_loss(policy - policy.grad, reference, 0.1).item(), loss.item())


if __name__ == '__main__':
    unittest.main()
