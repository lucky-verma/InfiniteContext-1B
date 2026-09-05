"""Run the verified native backend with private state storage and authentication."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

from scripts.fetch_model import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--window', type=int, default=4096)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--host', choices=('127.0.0.1', '0.0.0.0'), default='127.0.0.1')
    args = parser.parse_args()
    if args.window < 256 or args.window % 256 or not 1024 <= args.port <= 65535 or args.threads < 1:
        parser.error('window must be a positive multiple of 256; port 1024-65535; threads positive')
    spec = json.loads((ROOT / 'serving/runtime.json').read_text())
    receipt = json.loads((ROOT / f'.runtime/{args.backend}.json').read_text())
    if receipt['source_revision'] != spec['revision'] or receipt['patch_sha256'] != spec['patch_sha256']:
        parser.error('runtime receipt is stale; rebuild the pinned runtime')
    binary = Path(receipt['binary'])
    with binary.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != receipt['binary_sha256']:
            parser.error('runtime binary differs from the build receipt')
    for name, expected_hash in receipt['artifact_sha256'].items():
        with (binary.parent / name).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != expected_hash:
                parser.error(f'runtime artifact differs from its build receipt: {name}')
    model = json.loads((ROOT / 'serving/model.json').read_text())
    model_path = ROOT / '.models' / model['filename']
    verify(model_path, model)
    state = ROOT / '.sessions'
    state.mkdir(mode=0o700, exist_ok=True)
    (state / 'kv').mkdir(mode=0o700, exist_ok=True)
    key = state / 'backend.key'
    try:
        descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as output:
            output.write(secrets.token_urlsafe(32) + '\n')
    except FileExistsError:
        if not key.read_text().strip():
            parser.error('backend key file is empty')
    command = [str(binary), '--model', str(model_path.relative_to(ROOT)), '--alias', model['filename'],
               '--ctx-size', str(args.window), '--parallel', '1', '--threads', str(args.threads),
               '--threads-batch', str(args.threads), '--poll', '0', '--host', args.host,
               '--port', str(args.port), '--n-gpu-layers', '99' if args.backend == 'cuda' else '0',
               '--split-mode', 'none', '--main-gpu', '0',
               '--context-shift', '--keep', '4', '--cache-reuse', '0', '--cache-ram', '0', '--metrics', '--slots',
               '--n-predict', str(args.window),
               '--slot-save-path', '.sessions/kv', '--api-key-file', '.sessions/backend.key', '--no-ui']
    environment = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent))
    print(f'Starting {args.backend} backend on {args.host}:{args.port}; window={args.window}', flush=True)
    os.chdir(ROOT)
    os.execve(binary, command, environment)


if __name__ == '__main__':
    main()
