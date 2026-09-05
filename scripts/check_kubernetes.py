"""Validate rollout and durable replay in an explicitly isolated kind cluster."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubectl', default='kubectl')
    parser.add_argument('--kubeconfig', type=Path, required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--image', default='infinitecontext:local')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.context.startswith('kind-infinitecontext-'):
        parser.error('use a disposable kind-infinitecontext-* cluster, never a shared deployment')
    command = [args.kubectl, '--kubeconfig', str(args.kubeconfig), '--context', args.context]
    existing = subprocess.run(command+['get', 'namespace', 'infinitecontext', '--ignore-not-found', '-o', 'name'],
                              capture_output=True, text=True, check=True, timeout=10)
    if existing.stdout.strip():
        parser.error('the test namespace already exists; leaving it untouched')
    args.output.mkdir(parents=True, exist_ok=False)
    summary, created = {'status': 'failed'}, False
    with (args.output/'commands.txt').open('x') as log:
        def run(arguments, input=None):
            result = subprocess.run(command+arguments, input=input, capture_output=True, text=True, timeout=240)
            log.write(result.stdout+result.stderr)
            log.flush()
            if result.returncode:
                raise RuntimeError('Kubernetes check failed; inspect commands.txt')
            return result.stdout

        def ready():
            run(['-n', 'infinitecontext', 'rollout', 'status', 'statefulset/model', '--timeout=180s'])

        cli = ['-n', 'infinitecontext', 'exec', '-i', 'model-0', '-c', 'model', '--',
               'python3', '-m', 'streaming.cli', '--window', '512']
        message = json.dumps({'id': 'station-record', 'text': 'The station reports stable pressure.\n'})+'\n'
        try:
            manifest = (ROOT/'serving/kubernetes.yaml').read_text().replace('infinitecontext:local', args.image)
            created = True
            run(['apply', '-f', '-'], manifest)
            ready()
            first = json.loads(run(cli, message).splitlines()[-1])
            assert first['type'] == 'complete' and not first['replayed']
            began = time.monotonic()
            run(['-n', 'infinitecontext', 'delete', 'pod', 'model-0', '--wait=true'])
            ready()
            recovered = json.loads(run(cli, message).splitlines()[-1])
            assert recovered['replayed'] and recovered['total_input_tokens'] == first['total_input_tokens']
            restart_seconds = time.monotonic()-began
            run(['-n', 'infinitecontext', 'set', 'image', 'statefulset/model', 'model=invalid.local/infinitecontext:unavailable'])
            deadline = time.monotonic()+60
            while True:
                pod = json.loads(run(['-n', 'infinitecontext', 'get', 'pod', 'model-0', '-o', 'json']))
                statuses = pod.get('status', {}).get('containerStatuses', [])
                if any(item.get('state', {}).get('waiting', {}).get('reason') == 'ErrImageNeverPull' for item in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('the deliberately invalid rollout did not reach its expected failure state')
                time.sleep(1)
            began = time.monotonic()
            run(['-n', 'infinitecontext', 'rollout', 'undo', 'statefulset/model'])
            # StatefulSet rollback can require deleting the failed replacement
            # pod before its controller recreates the last healthy revision.
            run(['-n', 'infinitecontext', 'delete', 'pod', 'model-0', '--wait=true'])
            ready()
            restored = json.loads(run(cli, message).splitlines()[-1])
            assert restored['replayed']
            version = json.loads(run(['version', '-o', 'json']))['serverVersion']['gitVersion']
            summary = {'status': 'passed', 'kubernetes': version, 'image': args.image,
                       'pod_replacement_replay': 'passed', 'failed_rollout_observed': 'passed',
                       'rollback_replay': 'passed', 'restart_seconds': restart_seconds,
                       'rollback_seconds': time.monotonic()-began,
                       'manifest_sha256': hashlib.sha256((ROOT/'serving/kubernetes.yaml').read_bytes()).hexdigest(),
                       'scope': 'single-replica CPU StatefulSet on local kind; GPU scheduling, autoscaling and multi-node storage are not validated'}
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            if created:
                run(['delete', 'namespace', 'infinitecontext', '--wait=true', '--timeout=120s'])
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('PASS: Kubernetes pod replacement, failed rollout and rollback preserve the session')


if __name__ == '__main__':
    main()
