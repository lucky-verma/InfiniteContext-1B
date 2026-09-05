"""Exercise a real streaming session, bounded windows, replay and cancellation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streaming.backend import json_request
from streaming.session import Session


def wait_ready(endpoint, key_file, process):
    until = time.monotonic() + 60
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError('backend exited; inspect server.txt')
        try:
            key = key_file.read_text().strip()
            if json_request(endpoint, '/health', key=key)['status'] == 'ok':
                return
        except (OSError, ValueError, RuntimeError):
            pass
        time.sleep(0.1)
    raise TimeoutError('backend readiness timed out')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--min-input-tokens', type=int, default=4096)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--max-gpu-utilization', type=int, default=10,
                        help='preflight ceiling for background graphics activity; active compute processes always block')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.window < 512 or args.window % 256 or args.min_input_tokens < 2 * args.window or args.max_seconds < 1:
        parser.error('use a window multiple of 256 >=512, at least two windows of input, and a positive budget')
    if args.backend == 'cuda':
        if not 0 <= args.max_gpu_utilization <= 30:
            parser.error('max-gpu-utilization must be between 0 and 30')
        info = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,driver_version,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip()
        processes = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).decode().strip()
        free, utilization = [int(x.strip()) for x in info.splitlines()[0].split(',')[-2:]]
        if processes or free < 3072 or utilization > args.max_gpu_utilization:
            parser.error('GPU preflight failed: require no compute processes, at least 3 GiB free, and utilization below the ceiling')
    else:
        info = 'CPU execution'
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as port_reservation:
        port_reservation.bind(('127.0.0.1', 0))
        port = port_reservation.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}'
    model = json.loads((ROOT / 'serving/model.json').read_text())
    runtime = json.loads((ROOT / 'serving/runtime.json').read_text())
    receipt = json.loads((ROOT / f'.runtime/{args.backend}.json').read_text())
    sources = ['streaming/session.py', 'streaming/backend.py', 'serving/run.py', 'scripts/check_streaming.py']
    metadata = {'backend': args.backend, 'hardware': info, 'window': args.window,
                'requested_input_tokens': args.min_input_tokens, 'model': model, 'runtime': runtime,
                'build_receipt': {k:v for k,v in receipt.items() if k != 'binary'},
                'source_sha256': {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'ignore_eos': True,
                'workload': 'repeated telemetry prose plus a station-code update; conformance, not general language quality'}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    summary = {'status': 'failed'}
    stop = threading.Event()
    samples = []
    session_id = None
    (ROOT/'.sessions').mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.sessions') as temporary, (args.output/'server.txt').open('x') as log:
        database = Path(temporary)/'session.sqlite'
        key_file = ROOT/'.sessions/backend.key'
        command = [sys.executable, '-m', 'serving.run', '--backend', args.backend,
                   '--window', str(args.window), '--port', str(port)]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        def monitor():
            with (args.output/'resources.jsonl').open('x') as output:
                while not stop.is_set():
                    sample = {'elapsed_s': time.monotonic() - began}
                    for label, pid in (('server', process.pid), ('client', os.getpid())):
                        try:
                            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                                if line.startswith('VmRSS:'):
                                    sample[label+'_rss_kib'] = int(line.split()[1])
                        except OSError:
                            pass
                    if args.backend == 'cuda':
                        try:
                            text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], timeout=3).decode()
                            for line in text.splitlines():
                                fields = [x.strip() for x in line.split(',')]
                                if len(fields) == 2 and fields[0] == str(process.pid) and fields[1].isdigit():
                                    sample['server_gpu_mib'] = int(fields[1])
                        except (OSError, subprocess.SubprocessError):
                            sample['gpu_sample_unavailable'] = True
                    samples.append(sample)
                    output.write(json.dumps(sample)+'\n')
                    output.flush()
                    stop.wait(1)

        began = time.monotonic()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            wait_ready(endpoint, key_file, process)
            parameters = {'endpoint': endpoint, 'window': args.window, 'checkpoint_interval': 16, 'ignore_eos': True}
            count, maximum, evictions = 0, 0, 0
            started = time.monotonic()
            workload = 'The sensor reports a stable reading at the monitored station. ' * 16
            with (args.output/'events.jsonl').open('x') as output, Session(database, **parameters) as session:
                session_id = session.state['session_id']
                while session.state['input_tokens'] < args.min_input_tokens:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('stream exceeded the declared execution budget')
                    result = list(session.append(f'chunk-{count}', workload))[-1]
                    assert result['runtime_generated_steps'] == 0
                    assert result['runtime_timings']['prompt_n'] >= result['input_tokens'], 'new repeated input was not processed'
                    assert result['active_tokens'] < args.window
                    maximum = max(maximum, result['active_tokens'])
                    evictions += result['evicted_tokens'] > 0
                    result['elapsed_s'] = time.monotonic() - began
                    output.write(json.dumps(result)+'\n')
                    output.flush()
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {session.state['input_tokens']} input tokens; active={result['active_tokens']}", flush=True)
                before = session.state['last_seq']
                stream_elapsed = time.monotonic() - started
                replay = list(session.append(f'chunk-{count-1}', workload))[-1]
                assert replay['replayed'] and session.state['last_seq'] == before
                invalid = list(session.state['active'])
                invalid[0] += 1
                try:
                    session.rpc('/completion', {**session.payload(invalid, 0, 1), 'stream': False})
                except RuntimeError as error:
                    assert 'n_cache_shift does not match' in str(error)
                else:
                    raise AssertionError('an inconsistent cache shift was accepted')
                update = list(session.append('station-update', '\nThe current station code is VIOLET-913. Continue the log.\n', generate=16))[-1]
                output.write(json.dumps(update)+'\n')
                input_tokens = session.state['input_tokens']
                # Record an uncommitted continuation of the live state. After a
                # cold restart, snapshot+log replay must reproduce that output.
                probe_text = '\nContinue the station report.\n'
                probe_ids = session.rpc('/tokenize', {'content': probe_text, 'add_special': False, 'parse_special': True})['tokens']
                combined = session.state['active'] + probe_ids
                dropped = max(0, len(combined) - (args.window - 16 - 1))
                retained = combined[:session.anchors] + combined[session.anchors+dropped:] if dropped else combined
                expected = session.rpc('/completion', {**session.payload(retained, 16, dropped), 'stream': False})['content']
            # Abrupt backend loss must recover from disk without resubmitting accepted input.
            process.kill()
            process.wait(timeout=10)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(endpoint, key_file, process)
            with Session(database, **parameters) as session:
                assert session.state['input_tokens'] == input_tokens
                assert list(session.append('station-update', '\nThe current station code is VIOLET-913. Continue the log.\n', generate=16))[-1]['replayed']
                continuation = list(session.append('recovery-probe', probe_text, generate=16))[-1]
                assert continuation['text'] == expected, 'restored model state changed the deterministic continuation'
                stream = session.append('cancel-retry', '\nContinue the station log:\n', generate=128)
                assert next(stream)['type'] == 'accepted'
                assert next(stream)['type'] == 'delta'
                stream.close()
                recovered = list(session.append('cancel-retry', '\nContinue the station log:\n', generate=128))[-1]
                assert recovered['type'] == 'complete'
                snapshots = list((ROOT/'.sessions/kv').glob(session_id+'-*.bin'))
                assert len(snapshots) <= 2
                assert session.state['active'] and len(session.state['active']) < args.window
                completed = session.state['input_tokens']
            assert evictions > 0
            if args.backend == 'cuda':
                assert any(sample.get('server_gpu_mib', 0) > 0 for sample in samples), 'GPU process residency was not observed'
            summary = {'status': 'passed', 'input_tokens': completed, 'stream_chunks': count,
                'window_tokens': args.window, 'max_active_tokens': maximum,
                'input_evictions': evictions, 'idempotency': 'passed', 'checkpoint_replay': 'passed',
                'invalid_shift_rejection': 'passed',
                'backend_restart': 'passed', 'stream_elapsed_s': stream_elapsed,
                'restart_continuation_equivalence': 'passed',
                'cancellation_retry': 'passed', 'retained_snapshots': len(snapshots),
                'history_database_bytes': sum(p.stat().st_size for p in Path(temporary).glob('session.sqlite*') if p.is_file()),
                'elapsed_s': time.monotonic() - started,
                'scope': 'streaming/state conformance; general quality, perfect historical recall and full-system readiness are not established'}
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            stop.set()
            watcher.join(timeout=5)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if session_id:
                for path in (ROOT/'.sessions/kv').glob(session_id+'-*.bin'):
                    path.unlink()
            summary['sample_count'] = len(samples)
            for key in ('server_rss_kib', 'client_rss_kib', 'server_gpu_mib'):
                values = [sample[key] for sample in samples if key in sample]
                if values:
                    summary['max_sampled_'+key] = max(values)
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
