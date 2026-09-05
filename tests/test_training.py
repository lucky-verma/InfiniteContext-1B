"""Exercise the trainer across an epoch boundary, process restart and DDP."""

import array
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch

from training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class TrainingRecovery(unittest.TestCase):
    def test_resume_and_global_batch_ddp_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = random.Random(41)
            tokens = array.array('I', (rng.randrange(32) for _ in range(97)))
            if sys.byteorder != 'little':
                tokens.byteswap()
            data = root / 'tokens.u32'
            data.write_bytes(tokens.tobytes())

            def run(name, *, steps, resume=None, ddp=False):
                command = [sys.executable, '-m']
                if ddp:
                    command += ['torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', '-m']
                command += ['training.run', '--config', str(ROOT / 'training/recipes/smoke.json'),
                            '--data', str(data), '--output', str(root / name), '--steps', str(steps),
                            '--sequence-length', '8', '--batch-size', '1' if ddp else '2',
                            '--checkpoint-every', '2', '--threads', '1',
                            '--strategy', 'ddp' if ddp else 'single']
                if resume:
                    command += ['--resume', str(resume)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        timeout=120, env={**os.environ, 'OMP_NUM_THREADS': '1'})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return load_checkpoint(root / name / f'step-{steps:08d}.pt')

            # Six global batches per epoch: step eight exercises the saved cursor
            # after the sampler changes epochs, using nonperiodic input tokens.
            full = run('full', steps=8)
            resumed = run('resumed', steps=8, resume=root / 'full/step-00000004.pt')
            distributed = run('ddp', steps=8, ddp=True)
            for name, value in full['model'].items():
                torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(value, distributed['model'][name], rtol=1e-5, atol=1e-6)
            self.assertEqual(full['cursor'], resumed['cursor'])
            self.assertGreater(full['cursor']['epoch'], 0)
            manifest = root / 'full/step-00000008.pt.json'
            metadata = json.loads(manifest.read_text())
            metadata['sha256'] = '0' * 64
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_checkpoint(root / 'full/step-00000008.pt')


if __name__ == '__main__':
    unittest.main()
