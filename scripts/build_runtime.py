"""Build the pinned runtime and local context-shift patch in an isolated cache."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--portable', action='store_true', help='disable host-specific CPU instruction selection')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error('jobs must be between 1 and 16')
    spec = json.loads((ROOT / 'serving/runtime.json').read_text())
    patch = (ROOT / spec['patch']).resolve()
    if not patch.is_relative_to(ROOT):
        parser.error('runtime patch must be inside the repository')
    patch_bytes = patch.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != spec['patch_sha256']:
        parser.error('runtime patch does not match its pinned SHA-256')
    cache = ROOT / '.runtime'
    cache.mkdir(exist_ok=True)
    source = cache / f"llama-{spec['revision'][:12]}-{spec['patch_sha256'][:12]}-{args.backend}"
    if args.portable:
        source = source.with_name(source.name + '-portable')
    log_path = cache / f'build-{args.backend}.log'
    cmake = shutil.which('cmake') or str(Path(sys.executable).parent / 'cmake')
    flags = list(spec['build_flags']) + [f"-DGGML_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}"]
    if args.portable:
        flags.append('-DGGML_NATIVE=OFF')
    if args.backend == 'cuda':
        nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        if not Path(nvcc).is_file():
            parser.error('CUDA build requires nvcc; no system packages will be changed')
        flags += [f'-DCMAKE_CUDA_COMPILER={nvcc}', '-DCMAKE_CUDA_ARCHITECTURES=native']
    with log_path.open('w') as log:
        def run(command, cwd=None):
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)

        if not source.exists():
            print('Fetching the pinned runtime source...', flush=True)
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / 'source'
                run(['git', 'init', str(staged)])
                run(['git', 'fetch', '--depth', '1', spec['repository'], spec['revision']], staged)
                run(['git', 'checkout', '--detach', 'FETCH_HEAD'], staged)
                run(['git', 'apply', '--check', str(patch)], staged)
                run(['git', 'apply', str(patch)], staged)
                staged.rename(source)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source).decode().strip()
        difference = subprocess.check_output(['git', 'diff', '--binary', '--full-index'], cwd=source)
        if head != spec['revision'] or difference != patch_bytes:
            raise RuntimeError('runtime checkout differs from the pinned source and patch; leaving it untouched')
        print(f'Building {args.backend} runtime; details: {log_path}', flush=True)
        run([cmake, '-S', str(source), '-B', str(source / 'build'), *flags])
        targets = ['llama-server', 'test-rope']
        if args.backend == 'cuda':
            targets.append('test-backend-ops')
        run([cmake, '--build', str(source / 'build'), '--target', *targets, '--parallel', str(args.jobs)])
        print('Checking rotary translation and cache metadata...', flush=True)
        run([str(source / 'build/bin/test-rope')])
        binary = source / 'build/bin/llama-server'
        with binary.open('rb') as f:
            binary_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        version = subprocess.check_output([str(binary), '--version'], stderr=subprocess.STDOUT).decode().strip()
        artifact_hashes = {}
        for artifact in sorted(binary.parent.iterdir()):
            if artifact.is_file() and not artifact.is_symlink() and (artifact == binary or '.so' in artifact.name):
                with artifact.open('rb') as f:
                    artifact_hashes[artifact.name] = hashlib.file_digest(f, 'sha256').hexdigest()
        receipt = {'backend': args.backend, 'source_revision': head,
                   'patch_sha256': spec['patch_sha256'], 'build_flags': flags,
                   'binary': str(binary), 'binary_sha256': binary_hash,
                   'runtime_version': version, 'artifact_sha256': artifact_hashes,
                   'rotary_and_metadata_check': 'passed'}
        with tempfile.NamedTemporaryFile(mode='w', dir=cache, delete=False) as f:
            json.dump(receipt, f, indent=2)
            f.write('\n')
            temporary_receipt = Path(f.name)
        temporary_receipt.replace(cache / f'{args.backend}.json')
    print(f'PASS: runtime built and numerical checks passed. Binary: {binary}')


if __name__ == '__main__':
    main()
