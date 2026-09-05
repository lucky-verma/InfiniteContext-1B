# Model, training, alignment, and lifecycle

`training/src/modeling_mla.py` implements a dense decoder with latent KV
attention, decoupled rotary keys, RMSNorm, and SwiGLU. The absorbed and
materialized paths share weights and are compared for outputs and gradients.
The cache stores latent vectors and unrotated rotary keys, allowing surviving
keys to be rebased without repeatedly rotating their stored representation.

`training/recipes/mla-1b.json` specifies **996,520,448 parameters**. It is an
architecture configuration, not a trained checkpoint. `smoke.json` is a small
synthetic test fixture. The Qwen runtime reference uses a different architecture;
its pretrained tensors cannot initialize this MLA model.

## Data and training

Pretraining consumes packed little-endian uint32 token IDs. Each sequence has
one additional target token. The loader memory-maps the file; each sample copies
only its requested token block. The trainer records the exact corpus hash.

A real corpus must supply source URLs/revisions, redistribution/license status,
tokenizer identity and special tokens, document-level split/deduplication, and
packing policy. No licensed natural-language training corpus or trained 1B
checkpoint is published here. The integration harness creates explicitly
synthetic token fixtures with separate deterministic train/evaluation streams.
Different hashes alone do not prove semantic separation.

```bash
python -m training.run --config training/recipes/smoke.json \
  --data .data/train.u32 --output .runs/train --steps 100 \
  --sequence-length 128 --batch-size 2 --checkpoint-every 25
python -m training.run --config training/recipes/smoke.json \
  --data .data/train.u32 --output .runs/resume --steps 200 \
  --sequence-length 128 --batch-size 2 --checkpoint-every 25 \
  --resume .runs/train/step-00000100.pt
```

These commands require a prepared corpus with IDs inside the configured
vocabulary. `scripts/check_pipeline.py` creates its own runnable fixtures.
Counts are absolute optimizer steps, including resumed steps. A resume must
match architecture, data hash, optimizer settings, per-rank batch size, seed,
precision, strategy, and world size. Checkpoints preserve model, Adam state,
random state, and the next epoch/batch cursor. Checksums are verified before
weights-only deserialization; incomplete or corrupt artifacts are rejected.
Existing checkpoint paths and metric segments are never overwritten.

The current trainer uses FP32 and gradient clipping. The GPU preflight uses a
parameter/gradient/Adam memory floor, not an activation-memory guarantee. The
1B configuration needs roughly 16 GB for those training states alone before
activations and runtime overhead in an unsharded run. Mixed precision,
activation checkpointing, and target-scale throughput are unqualified work.

For GPU training, install the matching CUDA wheel rather than the quick start's
CPU wheel. The checked environment used PyTorch 2.8.0+cu128. Run
`python scripts/check_training_cuda.py --output .runs/training-cuda` to compare
small-model CPU/CUDA weights and CUDA checkpoint resume. This does not qualify
the 1B configuration's hardware fit.

## Distributed execution

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 -m training.run \
  --config training/recipes/smoke.json --data .data/train.u32 \
  --output .runs/ddp --steps 100 --strategy ddp --device cuda
```

DDP uses NCCL on CUDA and Gloo on CPU. Two-process CPU DDP is compared with the
equivalent single-process global batch in `tests/test_training.py`, including
an epoch boundary. That test also requires bit-exact checkpoint resume.
FSDP uses per-decoder-layer wrapping and full model/optimizer checkpoint APIs,
but its multi-GPU execution and scaling have not been validated on the available
single-GPU hardware. Distributed alignment currently fails explicitly instead
of implying that its reference-model sharding has been qualified.

`training/slurm/train.sbatch` launches one torchrun supervisor per allocated
node. Set `IC_REPO`, `IC_DATA`, `IC_OUTPUT`, `IC_CONFIG`, `IC_STEPS`,
`IC_GPUS_PER_NODE`, and `IC_RDZV_PORT`; choose `IC_STRATEGY=ddp` or `fsdp`.
Review the cluster's partition, allocation, data paths, communication, and
compute budget before submission. Shell syntax is checked; no SLURM job or
multi-node scaling result is claimed.

## SFT and DPO

Alignment uses tokenized JSONL:

```json
{"prompt":[1,2],"chosen":[3,4],"rejected":[5,6]}
```

These IDs illustrate the schema. Supply a versioned tokenizer and real
preference provenance for meaningful training. SFT requires `prompt` and
`chosen`; DPO additionally requires a different `rejected` completion. Sequences
over budget are rejected, and right-padding/prompt positions are excluded from
the objective. DPO uses summed completion log-probabilities and a frozen
reference checkpoint, following [the DPO paper](https://arxiv.org/abs/2305.18290).

```bash
python -m training.run --config training/recipes/smoke.json \
  --data .data/preferences.jsonl --output .runs/sft --steps 100 \
  --objective sft --initialize .runs/train/step-00000100.pt
python -m training.run --config training/recipes/smoke.json \
  --data .data/preferences.jsonl --output .runs/dpo --steps 100 \
  --objective dpo --initialize .runs/sft/step-00000100.pt \
  --reference .runs/sft/step-00000100.pt --beta 0.1
```

`--initialize` starts a new optimizer stage and records checkpoint lineage;
`--resume` continues an existing stage. The native objective supports this custom
model without a Transformers adapter. TRL/torchtune interoperability and
held-out preference-quality studies remain required integration work.

## Tracking, evaluation, and registry

Install `requirements-tracking.txt` for `--wandb` and `--mlflow`. W&B runs
locally/offline with Git/code capture disabled. MLflow tracking and artifacts
stay in the selected run directory. No run uploads occur automatically.

`training.evaluation.evaluate` scores a separate packed corpus and checks bounded
cached decoding. `training.registry` requires its matching successful report,
registers a model-from-code adapter, records checkpoint/evaluation hashes, and
reloads the candidate for prediction. This verifies artifact consistency; a
finite loss does not constitute a production quality threshold.

```bash
python scripts/check_pipeline.py --work .runs/pipeline \
  --output .runs/pipeline-evidence.json
```

The harness trains and aligns the fixture, evaluates all three stages, registers
two versions, moves aliases, serves the resolved version on loopback, compares
HTTP predictions with direct loading, and rolls back/reloads. Moving a registry
alias does not reload weights in an existing server process. The adapter is a
bounded next-token reference endpoint, not a production batching engine.

## Kernel qualification

```bash
python -m pip install -r requirements-kernels.txt
python scripts/check_kernels.py --interpreter --output .runs/kernels.json
```

RMSNorm, RoPE, and fused latent decoding have PyTorch numerical comparisons.
The decoder combines scoring, online softmax, and latent-value accumulation;
projection layers remain outside it. Interpreter results do not establish
CUDA compilation, race freedom, or speed. Current Triton lists
[SM80+ NVIDIA hardware](https://github.com/triton-lang/triton#compatibility);
the reference streaming GPU is SM75. Supported-device numerical tests, profiler
traces, and matched end-to-end comparisons are needed before these kernels can
replace the model's PyTorch path.
