"""Run the small synthetic training/alignment/evaluation/registry/serving path.

This is a systems integration check. Its artificial token data and small model
do not establish natural-language capability or target-scale readiness.
"""

import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='fresh private run directory')
    parser.add_argument('--output', type=Path, required=True, help='sanitized evidence JSON')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')

    def run(module, arguments):
        log = args.work / (module.replace('.', '-') + '-' + str(time.time_ns()) + '.txt')
        with log.open('x') as file:
            subprocess.run([sys.executable, '-m', module, *map(str, arguments)], cwd=ROOT,
                           stdout=file, stderr=subprocess.STDOUT, check=True, timeout=180)

    for name, seed in (('train', 71), ('eval', 103)):
        rng = random.Random(seed)
        # Two disjoint deterministic RNG streams; a test fixture, not a corpus.
        tokens = array.array('I', (rng.randrange(32) for _ in range(1025)))
        if sys.byteorder != 'little':
            tokens.byteswap()
        (args.work/f'{name}.u32').write_bytes(tokens.tobytes())
    pairs = [{'prompt': [i, 2], 'chosen': [3, (i+1)%32], 'rejected': [4, (i+2)%32]} for i in range(16)]
    (args.work/'preferences.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in pairs))
    common = ['--config', ROOT/'training/recipes/smoke.json', '--steps', 4, '--sequence-length', 8,
              '--batch-size', 2, '--checkpoint-every', 2, '--threads', 1]
    run('training.run', [*common, '--data', args.work/'train.u32', '--output', args.work/'pretrain', '--wandb', '--mlflow'])
    base = args.work/'pretrain/step-00000004.pt'
    run('training.run', [*common, '--data', args.work/'preferences.jsonl', '--output', args.work/'sft',
                         '--objective', 'sft', '--initialize', base])
    sft = args.work/'sft/step-00000004.pt'
    run('training.run', [*common, '--data', args.work/'preferences.jsonl', '--output', args.work/'dpo',
                         '--objective', 'dpo', '--initialize', sft, '--reference', sft])
    dpo = args.work/'dpo/step-00000004.pt'
    evaluations = {}
    for stage, checkpoint in (('pretrain', base), ('sft', sft), ('dpo', dpo)):
        report = args.work/f'{stage}-evaluation.json'
        run('training.evaluation.evaluate', ['--checkpoint', checkpoint, '--data', args.work/'eval.u32',
                                            '--sequence-length', 8, '--window', 16, '--output', report])
        evaluations[stage] = json.loads(report.read_text())
    for stage, checkpoint in (('pretrain', base), ('dpo', dpo)):
        run('training.registry', ['--checkpoint', checkpoint, '--store', args.work/'registry',
                                 '--evaluation', args.work/f'{stage}-evaluation.json'])
    import mlflow
    import pandas as pd
    mlflow.set_tracking_uri('sqlite:///' + str((args.work/'registry/registry.db').resolve()))
    client = mlflow.MlflowClient()
    name = 'InfiniteContextMLA'
    versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda x: int(x.version))
    assert len(versions) == 2
    old, candidate = [v.version for v in versions]
    assert client.get_model_version_by_alias(name, 'candidate').version == candidate
    client.set_registered_model_alias(name, 'champion', old)
    client.set_registered_model_alias(name, 'previous', old)
    client.set_registered_model_alias(name, 'champion', candidate)
    assert client.get_model_version_by_alias(name, 'champion').version == candidate
    # Pin the resolved version at process startup; moving an alias alone does
    # not reload the weights in an already-running model server.
    uri = f'models:/{name}/{candidate}'
    expected = mlflow.pyfunc.load_model(uri).predict(pd.DataFrame({'input_ids': [[0, 1, 2]]})).to_dict(orient='records')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    with (args.work/'model-server.txt').open('x') as log:
        process = subprocess.Popen([sys.executable, '-m', 'mlflow', 'models', 'serve', '-m', uri,
                                    '--env-manager', 'local', '--host', '127.0.0.1', '--port', str(port)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'MLFLOW_TRACKING_URI': mlflow.get_tracking_uri()})
        try:
            until = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError('registered-model server exited; inspect private run log')
                try:
                    with urlopen(f'http://127.0.0.1:{port}/ping', timeout=2) as response:
                        assert response.status == 200
                    break
                except OSError:
                    if time.monotonic() >= until:
                        raise TimeoutError('registered-model readiness timed out')
                    time.sleep(0.2)
            body = json.dumps({'dataframe_records': [{'input_ids': [0, 1, 2]}]}).encode()
            with urlopen(Request(f'http://127.0.0.1:{port}/invocations', body,
                                 {'Content-Type': 'application/json'}), timeout=10) as response:
                actual = json.load(response)['predictions']
            assert actual == expected
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    client.set_registered_model_alias(name, 'champion', old)
    assert client.get_model_version_by_alias(name, 'champion').version == old
    rollback = mlflow.pyfunc.load_model(f'models:/{name}@champion')
    assert len(rollback.predict(pd.DataFrame({'input_ids': [[0, 1, 2]]}))) == 1
    result = {'status': 'passed', 'stages': ['pretrain', 'sft', 'dpo', 'held-out evaluation',
              'offline W&B', 'local MLflow', 'registry promotion', 'HTTP prediction', 'registry rollback/reload'],
              'evaluations': evaluations, 'served_prediction': actual,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'training').rglob('*.py'))},
              'scope': 'small synthetic CPU integration; production quality, 1B training, distributed GPU scale and vLLM are not established'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print('PASS: small-model training-to-registry-to-HTTP path and rollback')


if __name__ == '__main__':
    main()
