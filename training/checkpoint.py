"""Atomic, checksummed training checkpoints loaded through weights-only deserialization."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def load_checkpoint(path, manifest_path=None):
    path = Path(path)
    metadata = json.loads(Path(manifest_path or path.with_suffix(path.suffix + '.json')).read_text())
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != metadata['sha256']:
            raise ValueError('checkpoint SHA-256 does not match its manifest')
    return torch.load(path, map_location='cpu', weights_only=True)


def save_checkpoint(directory, step, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'step-{step:08d}.pt'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        with temporary.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        os.link(temporary, target)
        manifest = {'sha256': digest, 'step': step, 'config': payload['config'],
                    'data_sha256': payload['data_sha256'], 'torch': str(torch.__version__)}
        with target.with_suffix(target.suffix + '.json').open('x') as file:
            json.dump(manifest, file, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
