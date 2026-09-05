"""Compare small-model CUDA training and resume with the same CPU global batch."""

import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    processes = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip()
    hardware = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,driver_version,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    free, usage = map(int, hardware.split(',')[-2:])
    if processes or free < 3072 or usage > 30:
        parser.error('GPU has active compute, less than 3 GiB free or high background utilization')
    import torch
    from training.checkpoint import load_checkpoint
    if not torch.cuda.is_available():
        parser.error('install the matching CUDA PyTorch build in a separate environment')
    args.output.mkdir(parents=True, exist_ok=False)
    (ROOT/'.runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.runs') as temporary:
        root = Path(temporary)
        rng = random.Random(417)
        tokens = array.array('I', (rng.randrange(32) for _ in range(97)))
        if sys.byteorder != 'little':
            tokens.byteswap()
        data = root/'data.u32'
        data.write_bytes(tokens.tobytes())

        def train(name, device, resume=None):
            command = [sys.executable, '-m', 'training.run', '--config', 'training/recipes/smoke.json',
                       '--data', str(data), '--output', str(root/name), '--steps', '8',
                       '--sequence-length', '8', '--batch-size', '2', '--checkpoint-every', '4',
                       '--threads', '1', '--device', device]
            if resume:
                command += ['--resume', str(resume)]
            with (args.output/(name+'.txt')).open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120,
                               env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
            return load_checkpoint(root/name/'step-00000008.pt')

        cpu = train('cpu', 'cpu')
        gpu = train('cuda', 'cuda')
        resumed = train('resumed', 'cuda', root/'cuda/step-00000004.pt')
        maximum, resumed_maximum = 0.0, 0.0
        for name, value in cpu['model'].items():
            actual, recovered = gpu['model'][name], resumed['model'][name]
            maximum = max(maximum, (value-actual).abs().max().item())
            resumed_maximum = max(resumed_maximum, (actual-recovered).abs().max().item())
            torch.testing.assert_close(actual, value, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(recovered, actual, rtol=1e-5, atol=1e-7)
        result = {'status': 'passed', 'hardware': hardware, 'torch': str(torch.__version__),
                  'cuda': torch.version.cuda, 'optimizer_steps': 8,
                  'cpu_cuda_max_abs_weight_difference': maximum,
                  'cuda_resume_max_abs_weight_difference': resumed_maximum,
                  'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                  'source_sha256': {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                    ('training/run.py', 'training/src/modeling_mla.py', 'training/checkpoint.py', 'scripts/check_training_cuda.py')},
                  'scope': 'small single-GPU FP32 training/resume equivalence; no 1B or distributed GPU scaling claim'}
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: small CUDA training, CPU comparison and checkpoint resume')


if __name__ == '__main__':
    main()
