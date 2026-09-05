"""Probe native generation across cache rollovers; this is not a recall benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.server_binary.resolve()
    manifest = json.loads((ROOT / 'serving/model.json').read_text())
    model = ROOT / '.models' / manifest['filename']
    with model.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == manifest['sha256'], 'model hash mismatch'
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    command = [str(binary), '--model', str(model.relative_to(ROOT)),
               '--ctx-size', '512', '--parallel', '1', '--threads', '2',
               '--threads-batch', '2', '--n-gpu-layers', '0', '--poll', '0',
               '--host', '127.0.0.1', '--port', str(port), '--no-webui',
               '--context-shift', '--keep', '4', '--no-warmup', '--metrics']
    environment = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent))
    version = subprocess.check_output([str(binary), '--version'], env=environment, stderr=subprocess.STDOUT).decode().strip()
    metadata = {'model': manifest, 'runtime': version, 'command': command,
                'window_tokens': 512, 'requested_generation_tokens': 1536,
                'device': 'cpu', 'scope': 'native generation rollover compatibility only'}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary = {'status': 'failed'}
    started = time.monotonic()
    with (args.output / 'server.log').open('x') as server_log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            base = f'http://127.0.0.1:{port}'
            while True:
                if process.poll() is not None:
                    raise RuntimeError('runtime exited; inspect server.log')
                try:
                    with urlopen(base + '/health', timeout=1) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() - started > 30:
                    raise TimeoutError('runtime did not become healthy within 30 seconds')
                time.sleep(0.25)
            request = {'prompt': 'Continue a numbered list of simple observations about the weather:\n1.',
                       'n_predict': 1536, 'temperature': 0, 'seed': 42,
                       'ignore_eos': True, 'stream': True, 'n_keep': 4}
            (args.output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
            req = Request(base + '/completion', data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
            final = None
            generation_started = time.monotonic()
            with urlopen(req, timeout=120) as response, (args.output / 'events.jsonl').open('x') as events:
                for line in response:
                    if time.monotonic() - generation_started > 120:
                        raise TimeoutError('probe exceeded its 120-second generation budget')
                    if not line.startswith(b'data: '):
                        continue
                    event = json.loads(line[6:])
                    events.write(json.dumps({'elapsed_s': time.monotonic() - generation_started, 'event': event}) + '\n')
                    events.flush()
                    if event.get('stop'):
                        final = event
            if final is None:
                raise RuntimeError('stream ended without a final completion event')
            with urlopen(base + '/metrics', timeout=5) as response:
                (args.output / 'metrics.txt').write_bytes(response.read())
            server_log.flush()
            shifts = (args.output / 'server.log').read_text().count('slot context shift,')
            predicted = final.get('tokens_predicted') or final.get('timings', {}).get('predicted_n', 0)
            assert predicted == 1536, f'incomplete generation: {predicted}'
            assert shifts >= 2, f'expected repeated context shifts, observed {shifts}'
            summary = {'status': 'passed', 'generated_tokens': predicted,
                       'context_shift_events': shifts, 'elapsed_s': time.monotonic() - generation_started,
                       'scope': 'native CPU generation rollover; input append, recall, GPU memory and recovery remain unvalidated'}
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
