"""Exercise an isolated Compose deployment, authenticated access and recovery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]


def available_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='infinitecontext:local')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    port, metrics_port = available_port(), available_port()
    project = 'ic-check-' + uuid.uuid4().hex[:12]
    environment = {**os.environ, 'IC_IMAGE': args.image, 'IC_PORT': str(port), 'IC_PROM_PORT': str(metrics_port)}
    compose = ['docker', 'compose', '--project-name', project, '-f', 'serving/compose.yaml', '-f', 'infra/monitoring/compose.yaml']
    summary = {'status': 'failed'}
    with (args.output/'commands.txt').open('x') as log:
        def run(arguments, input=None):
            result = subprocess.run(compose + arguments, cwd=ROOT, env=environment,
                                    input=input, capture_output=True, text=True, timeout=180)
            log.write(result.stdout + result.stderr)
            log.flush()
            if result.returncode:
                raise RuntimeError('Compose command failed; inspect commands.txt')
            return result.stdout

        try:
            run(['up', '--no-build', '--detach', '--wait', '--wait-timeout', '120'])
            try:
                urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3)
            except HTTPError as error:
                assert error.code in (401, 403)
            else:
                raise AssertionError('model API accepted an unauthenticated request')
            message = json.dumps({'id': 'record-1', 'text': 'The station is ready.\n', 'generate': 0})+'\n'
            cli = ['exec', '-T', 'model', 'python3', '-m', 'streaming.cli', '--window', '512']
            first = [json.loads(line) for line in run(cli, message).splitlines()][-1]
            assert first['type'] == 'complete' and not first['replayed']
            began = time.monotonic()
            run(['restart', 'model'])
            run(['up', '--no-build', '--detach', '--wait', '--wait-timeout', '120'])
            recovered = [json.loads(line) for line in run(cli, message).splitlines()][-1]
            assert recovered['replayed'] and recovered['total_input_tokens'] == first['total_input_tokens']
            recovery_seconds = time.monotonic() - began
            deadline = time.monotonic() + 30
            while True:
                with urlopen(f'http://127.0.0.1:{metrics_port}/api/v1/query?query=up', timeout=3) as response:
                    samples = json.load(response)['data']['result']
                if any(item['metric'].get('job') == 'infinitecontext' and item['value'][1] == '1' for item in samples):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('authenticated Prometheus scrape did not become healthy')
                time.sleep(1)
            identity = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
            summary = {'status': 'passed', 'image_id': identity, 'unauthenticated_request': 'rejected',
                       'session_commit': 'passed', 'container_restart_replay': 'passed',
                       'authenticated_prometheus_scrape': 'passed', 'recovery_seconds': recovery_seconds,
                       'files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                 ('serving/Dockerfile', 'serving/compose.yaml', 'infra/monitoring/compose.yaml', 'infra/monitoring/prometheus.yaml')},
                       'scope': 'single-session CPU container deployment; multi-session admission, GPU containers and autoscaling remain separate checks'}
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            run(['down', '--volumes', '--remove-orphans'])
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('PASS: container restart/replay, authentication and Prometheus scrape')


if __name__ == '__main__':
    main()
