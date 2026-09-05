"""MLflow model-from-code adapter using verified, weights-only checkpoints."""

import mlflow
import pandas as pd
import torch

from training.checkpoint import load_checkpoint
from training.src.modeling_mla import MLAConfig, MLALanguageModel


class MLAPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        payload = load_checkpoint(context.artifacts['checkpoint'], context.artifacts['manifest'])
        payload.pop('optimizer', None)
        self.model = MLALanguageModel(MLAConfig(**payload['config'])).eval()
        self.model.load_state_dict(payload['model'])
        self.limit = context.model_config['max_sequence_length']
        torch.set_num_threads(2)

    def predict(self, context, model_input, params=None):
        rows = []
        with torch.inference_mode():
            for values in model_input['input_ids']:
                if not 1 <= len(values) <= self.limit:
                    raise ValueError('input length exceeds the registered model signature budget')
                inputs = torch.tensor([list(values)], dtype=torch.long)
                logits = self.model(inputs)
                rows.append({'next_token_id': int(logits[0, -1].argmax()), 'input_tokens': len(values)})
        return pd.DataFrame(rows)


mlflow.models.set_model(MLAPredictor())
