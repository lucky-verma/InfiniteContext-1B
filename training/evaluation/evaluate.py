"""Score a held-out packed corpus and exercise bounded decoding of a checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from training.checkpoint import load_checkpoint
from training.data import TokenBlocks
from training.src.modeling_mla import MLAConfig, MLALanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--max-batches', type=int, default=32)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sequence_length, args.max_batches) < 1 or args.window < 8:
        parser.error('positive counts and a window of at least eight tokens are required')
    torch.set_num_threads(2)
    payload = load_checkpoint(args.checkpoint)
    with args.data.open('rb') as file:
        data_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    if data_hash == payload['data_sha256']:
        parser.error('evaluation corpus is byte-identical to the training corpus')
    # Different hashes are necessary, not proof of semantic deduplication.
    # Dataset preparation owns document-level split and contamination checks.
    model = MLALanguageModel(MLAConfig(**payload['config'])).eval()
    model.load_state_dict(payload['model'])
    dataset = TokenBlocks(args.data, args.sequence_length)
    total, tokens, caches = 0.0, 0, None
    try:
        with torch.inference_mode():
            for index, (inputs, labels) in enumerate(DataLoader(dataset, batch_size=1)):
                loss = F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten(), reduction='sum')
                total += loss.item()
                tokens += labels.numel()
                if index + 1 == args.max_batches:
                    break
            source = dataset[0][0]
            for step in range(args.window * 3):
                token = source[step % len(source)].reshape(1, 1)
                logits, caches = model(token, caches, use_cache=True, window=args.window)
                assert torch.isfinite(logits).all()
                assert all(cache.length <= args.window for cache in caches)
        nll = total / tokens
        if not math.isfinite(nll):
            raise FloatingPointError('held-out loss is not finite')
        with args.checkpoint.open('rb') as file:
            checkpoint_hash = hashlib.file_digest(file, 'sha256').hexdigest()
        report = {'status': 'passed', 'checkpoint_sha256': checkpoint_hash,
                  'evaluation_data_sha256': data_hash, 'training_data_sha256': payload['data_sha256'],
                  'tokens_scored': tokens, 'mean_nll': nll, 'perplexity': math.exp(nll),
                  'streamed_tokens': args.window * 3, 'active_window': args.window,
                  'cache_bytes': sum(cache.nbytes for cache in caches),
                  'scope': 'checkpoint conformance and held-out corpus loss; no general quality or scale claim'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
        print(json.dumps(report))
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
