"""Reproducible causal-LM training with DDP/FSDP and exact-data checkpoint resume."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from training.checkpoint import load_checkpoint, save_checkpoint
from training.data import TokenBlocks
from training.alignment import PreferencePairs, completion_logps, dpo_loss
from training.src.modeling_mla import MLAConfig, MLADecoderLayer, MLALanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True, help='absolute optimizer step to reach, including resumed steps')
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--strategy', choices=('single', 'ddp', 'fsdp'), default='single')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize', type=Path, help='initialize weights for a new training stage; not optimizer resume')
    parser.add_argument('--objective', choices=('pretrain', 'sft', 'dpo'), default='pretrain')
    parser.add_argument('--reference', type=Path, help='frozen reference checkpoint required for DPO')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--wandb', action='store_true', help='record a local/offline W&B run')
    parser.add_argument('--mlflow', action='store_true', help='record a local MLflow tracking run')
    args = parser.parse_args()
    if min(args.steps, args.sequence_length, args.batch_size, args.checkpoint_every, args.threads) < 1 or not 0 < args.learning_rate < 1:
        parser.error('counts must be positive and learning rate in (0,1)')
    if args.resume and args.initialize or (args.objective == 'dpo') != bool(args.reference) or not 0 < args.beta < float('inf'):
        parser.error('resume and initialize are exclusive; DPO requires reference; beta must be finite and positive')
    if args.objective != 'pretrain' and args.strategy != 'single':
        parser.error('alignment currently supports the single-device reference path; distributed alignment is not validated')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.strategy == 'single' and world != 1 or args.strategy != 'single' and world < 2:
        parser.error('single requires one process; ddp/fsdp require torchrun with at least two processes')
    if args.strategy == 'fsdp' and args.device != 'cuda':
        parser.error('FSDP validation requires CUDA devices')
    if args.device == 'cuda' and (not torch.cuda.is_available() or local_rank >= torch.cuda.device_count()):
        parser.error('the requested CUDA device is unavailable')
    config = MLAConfig(**json.loads(args.config.read_text()))
    with args.data.open('rb') as file:
        data_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    training = {'sequence_length': args.sequence_length, 'batch_size_per_rank': args.batch_size,
                'world_size': world, 'strategy': args.strategy, 'learning_rate': args.learning_rate,
                'seed': args.seed, 'precision': 'float32'}
    if args.objective != 'pretrain':
        training.update(objective=args.objective, beta=args.beta)
    if args.reference:
        with args.reference.open('rb') as file:
            training['reference_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
    saved = load_checkpoint(args.resume) if args.resume else None
    lineage = dict(saved.get('lineage', {})) if saved else {}
    if args.initialize:
        with args.initialize.open('rb') as file:
            lineage['initialized_from_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
    if saved and (saved['config'] != asdict(config) or saved['data_sha256'] != data_hash or saved['training'] != training):
        raise ValueError('resume configuration, data, or distributed topology differs from the checkpoint')
    start_step = saved['step'] if saved else 0
    if args.steps <= start_step:
        parser.error('steps must exceed the checkpoint step')
    with torch.device('meta'):
        probe = MLALanguageModel(config)
    parameters = sum(p.numel() for p in probe.parameters())
    if args.device == 'cuda':
        torch.cuda.set_device(local_rank)
        free, _ = torch.cuda.mem_get_info(local_rank)
        floor = parameters * (20 if args.reference else 16) // (world if args.strategy == 'fsdp' else 1)
        if floor > free * 0.8:
            parser.error('the parameter/gradient/Adam memory floor exceeds available CUDA memory; choose suitable hardware/topology')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    dataset, model, wandb_run, metrics = None, None, None, None
    mlflow = None
    try:
        if world > 1:
            dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
        dataset = TokenBlocks(args.data, args.sequence_length) if args.objective == 'pretrain' else PreferencePairs(
            args.data, args.sequence_length, config.vocab_size, args.objective)
        if len(dataset) < world * args.batch_size:
            raise ValueError('dataset does not contain a complete global batch')
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, drop_last=True)
        loader_rng = torch.Generator()
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=0,
                            generator=loader_rng, collate_fn=dataset.collate if isinstance(dataset, PreferencePairs) else None)
        base = MLALanguageModel(config)
        if saved:
            base.load_state_dict(saved['model'])
        if args.initialize:
            initial = load_checkpoint(args.initialize)
            if initial['config'] != asdict(config):
                raise ValueError('initialization checkpoint architecture differs from the requested config')
            base.load_state_dict(initial['model'])
            del initial
        base.to(device)
        reference = None
        if args.reference:
            reference_payload = load_checkpoint(args.reference)
            if reference_payload['config'] != asdict(config):
                raise ValueError('DPO reference architecture differs from the policy')
            reference = MLALanguageModel(config).to(device).eval().requires_grad_(False)
            reference.load_state_dict(reference_payload['model'])
            del reference_payload
        if args.strategy == 'ddp':
            model = DistributedDataParallel(base, device_ids=[local_rank] if args.device == 'cuda' else None)
        elif args.strategy == 'fsdp':
            model = FSDP(base, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={MLADecoderLayer}), device_id=device)
        else:
            model = base
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if saved:
            optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, saved['optimizer']) if args.strategy == 'fsdp' else saved['optimizer']
            optimizer.load_state_dict(optimizer_state)
            torch.set_rng_state(saved['rng'][rank]['cpu'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state(saved['rng'][rank]['cuda'], device)
        args.output.mkdir(parents=True, exist_ok=True)
        segment = args.output / f'metrics-from-{start_step:08d}.jsonl'
        metrics = segment.open('x') if rank == 0 else None
        if rank == 0:
            metadata = {'config': asdict(config), 'training': training, 'parameters': parameters,
                        'data_sha256': data_hash, 'torch': str(torch.__version__), 'start_step': start_step,
                        'lineage': lineage}
            (args.output / f'run-from-{start_step:08d}.json').write_text(json.dumps(metadata, indent=2)+'\n')
            if args.wandb:
                os.environ['WANDB_DISABLE_GIT'] = 'true'
                os.environ['WANDB_DISABLE_CODE'] = 'true'
                import wandb
                wandb_run = wandb.init(project='InfiniteContext', mode='offline', dir=str(args.output), config=metadata)
            if args.mlflow:
                os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
                import mlflow as tracking
                mlflow = tracking
                mlflow.set_tracking_uri('sqlite:///' + str((args.output / 'mlflow.db').resolve()))
                client = mlflow.MlflowClient()
                experiment = client.get_experiment_by_name('InfiniteContext')
                experiment_id = experiment.experiment_id if experiment else client.create_experiment(
                    'InfiniteContext', artifact_location=(args.output / 'mlartifacts').resolve().as_uri())
                mlflow.set_experiment(experiment_id=experiment_id)
                mlflow.start_run()
                mlflow.log_params({**training, 'parameters': parameters, 'data_sha256': data_hash})
        step = start_step
        epoch, skip = (saved['cursor']['epoch'], saved['cursor']['batch']) if saved else (0, 0)
        model.train()
        while step < args.steps:
            sampler.set_epoch(epoch)
            loader_rng.manual_seed(args.seed + epoch)
            for batch_index, (inputs, labels) in enumerate(loader):
                if batch_index < skip:
                    continue
                started = time.perf_counter()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                if reference is None:
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)
                else:
                    with torch.no_grad():
                        reference_logps = completion_logps(reference(inputs), labels)
                    loss = dpo_loss(completion_logps(logits, labels), reference_logps, args.beta)
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('non-finite loss; the last complete checkpoint is preserved')
                loss.backward()
                norm = model.clip_grad_norm_(1.0) if args.strategy == 'fsdp' else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError('non-finite gradient norm')
                optimizer.step()
                value = loss.detach().clone()
                if world > 1:
                    dist.all_reduce(value)
                    value /= world
                if args.device == 'cuda':
                    torch.cuda.synchronize(device)
                step += 1
                next_cursor = {'epoch': epoch + (batch_index + 1 == len(loader)), 'batch': (batch_index + 1) % len(loader)}
                record = {'step': step, 'loss': value.item(), 'gradient_norm': norm.item(),
                          'tokens': inputs.numel() * world, 'scored_tokens': int((labels != -100).sum()) * world,
                          'seconds': time.perf_counter() - started}
                if rank == 0:
                    metrics.write(json.dumps(record)+'\n')
                    metrics.flush()
                    print(json.dumps(record), flush=True)
                    if wandb_run:
                        wandb_run.log(record, step=step)
                    if mlflow:
                        mlflow.log_metrics({k:v for k,v in record.items() if k != 'step'}, step=step)
                if step % args.checkpoint_every == 0 or step == args.steps:
                    rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if args.device == 'cuda' else None}
                    states = [None] * world if rank == 0 else None
                    if world > 1:
                        dist.gather_object(rng, states, dst=0)
                    else:
                        states = [rng]
                    context = FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)) if args.strategy == 'fsdp' else nullcontext()
                    with context:
                        model_state = model.state_dict() if args.strategy == 'fsdp' else base.state_dict()
                        optimizer_state = FSDP.optim_state_dict(model, optimizer) if args.strategy == 'fsdp' else optimizer.state_dict()
                    if rank == 0:
                        path = save_checkpoint(args.output, step, {'config': asdict(config), 'training': training,
                            'model': model_state, 'optimizer': optimizer_state, 'step': step,
                            'cursor': next_cursor, 'rng': states, 'data_sha256': data_hash, 'lineage': lineage})
                        if mlflow:
                            mlflow.log_artifact(str(path), artifact_path='checkpoints')
                            mlflow.log_artifact(str(path.with_suffix(path.suffix+'.json')), artifact_path='checkpoints')
                    if world > 1:
                        dist.barrier()
                if step == args.steps:
                    break
            epoch += 1
            skip = 0
    finally:
        if metrics is not None:
            metrics.close()
        if wandb_run:
            wandb_run.finish()
        if mlflow:
            mlflow.end_run()
        if dataset:
            dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
