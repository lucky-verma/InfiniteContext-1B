"""Download and verify the pinned public reference model using the stdlib."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def verify(path, manifest):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if path.stat().st_size != manifest['size_bytes'] or digest != manifest['sha256']:
        raise ValueError(f'Model verification failed: {path}')


def main():
    manifest = json.loads((ROOT / 'serving/model.json').read_text())
    filename = manifest['filename']
    if Path(filename).name != filename or not filename.endswith('.gguf'):
        raise ValueError('Manifest filename must be a plain GGUF filename')
    destination = ROOT / '.models' / filename
    if destination.exists():
        verify(destination, manifest)
        print('PASS: existing model matches the pinned size and SHA-256')
        return
    destination.parent.mkdir(exist_ok=True)
    url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{manifest['filename']}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        verify(temporary, manifest)
        temporary.chmod(0o644)
        # Publish complete bytes without overwriting a file created by another session.
        os.link(temporary, destination)
        print('PASS: downloaded model matches the pinned size and SHA-256')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
