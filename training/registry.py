"""Register a verified MLA checkpoint in a local MLflow model registry."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, Array, DataType

from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--name', default='InfiniteContextMLA')
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--evaluation', type=Path, required=True, help='matching held-out evaluation report')
    args = parser.parse_args()
    if args.max_sequence_length < 1:
        parser.error('max-sequence-length must be positive')
    payload = load_checkpoint(args.checkpoint)
    evaluation = json.loads(args.evaluation.read_text())
    with args.checkpoint.open('rb') as file:
        checkpoint_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    if evaluation['status'] != 'passed' or evaluation['checkpoint_sha256'] != checkpoint_hash:
        parser.error('evaluation did not pass for this exact checkpoint')
    args.store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///' + str((args.store / 'registry.db').resolve()))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name('InfiniteContext-Registry')
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        'InfiniteContext-Registry', artifact_location=(args.store / 'artifacts').resolve().as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)
    root = Path(__file__).resolve().parents[1]
    signature = ModelSignature(
        inputs=Schema([ColSpec(Array(DataType.long), 'input_ids')]),
        outputs=Schema([ColSpec(DataType.long, 'next_token_id'), ColSpec(DataType.long, 'input_tokens')]))
    with mlflow.start_run():
        mlflow.log_params({'training_step': payload['step'], 'data_sha256': payload['data_sha256']})
        mlflow.log_artifact(str(args.evaluation), artifact_path='evaluation')
        info = mlflow.pyfunc.log_model(
            name='model', python_model=str(root / 'training/mlflow_model.py'),
            artifacts={'checkpoint': str(args.checkpoint.resolve()),
                       'manifest': str(args.checkpoint.with_suffix(args.checkpoint.suffix + '.json').resolve())},
            code_paths=[str(root / 'training')], registered_model_name=args.name,
            model_config={'max_sequence_length': args.max_sequence_length}, signature=signature,
            input_example=pd.DataFrame({'input_ids': [[0, 1, 2]]}))
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        result = loaded.predict(pd.DataFrame({'input_ids': [[0, 1, 2]]}))
        assert len(result) == 1 and 0 <= result.iloc[0]['next_token_id'] < payload['config']['vocab_size']
        version = str(info.registered_model_version)
        client.set_model_version_tag(args.name, version, 'checkpoint_sha256', checkpoint_hash)
        client.set_model_version_tag(args.name, version, 'evaluation_sha256', hashlib.sha256(args.evaluation.read_bytes()).hexdigest())
        client.set_model_version_tag(args.name, version, 'validation_scope', evaluation['scope'])
        client.set_registered_model_alias(args.name, 'candidate', version)
        print('PASS: registered and reloaded model:', info.model_uri)
        print(result.to_json(orient='records'))


if __name__ == '__main__':
    main()
