# Results and evidence boundaries

These results establish specific implementation contracts. The pretrained
streaming reference, synthetic custom-model pipeline, kernel checks, and
operational tests use different workloads and support different claims.

## Sustained streaming

The pinned Qwen3.5-0.8B Q8_0 reference ran through a patched llama.cpp backend
on an RTX 2070 SUPER with 8 GiB VRAM. The active window was 512 tokens; the
workload repeatedly appended a 177-token telemetry chunk with zero requested
output, then exercised generation and recovery.

| Measurement | GPU run | CPU conformance run |
|---|---:|---:|
| Total committed input tokens | 1,000,083 | 4,281 |
| Bulk ingestion chunks | 5,650 | 24 |
| Active-token maximum | 511 | 511 |
| Input eviction events | 5,648 | 22 |
| Bulk ingestion wall time | 300.541 s | 35.782 s |
| Sampled GPU process maximum | 954 MiB | Not applicable |
| Sampled server RSS maximum | 814,028 KiB | 984,772 KiB |
| Sampled client RSS maximum | 28,324 KiB | 26,364 KiB |
| History database after closure | 35,987,456 bytes | 200,704 bytes |
| Retained native snapshots | 2 | 2 |

Both runs passed request-ID idempotency, invalid-shift rejection, abrupt backend
restart, checkpoint/log replay, deterministic continuation equivalence after
restart, and cancellation/retry. The bulk GPU stream contains 1,000,050 input
tokens; the final recovery probes account for the remaining committed inputs.
The live-state continuation used as a recovery reference is deliberately not
committed to the session log.

[GPU summary](../results/streaming-million-cuda/summary.json),
[events](../results/streaming-million-cuda/events.jsonl),
[resource samples](../results/streaming-million-cuda/resources.jsonl),
[identity/build metadata](../results/streaming-million-cuda/metadata.json),
[server output](../results/streaming-million-cuda/server.txt), and
[CPU summary](../results/streaming-cpu/summary.json) preserve the evidence.

Memory is sampled approximately once per second; maxima are sampled process
values, not isolated KV tensor allocations or guaranteed hardware peaks. Model
weights, recurrent state, runtime allocations, and cache are included in the GPU
process measurement. Native snapshots ranged from 22,382,044 to 26,495,588 bytes
in the GPU run and are separate from the history database. The figure excludes
startup and post-stream recovery samples from its stream-length axis.

This is one repeated synthetic ingestion workload, not a mixed conversational
load or a speedup comparison with the shorter CPU run. Active model state stays
bounded in the observed run while history/index storage grows. A finite test
does not establish literally unlimited full-attention context.

## Retention and explicit retrieval

The CPU diagnostic uses three deterministic six-digit codes and four fact
conditions, each evaluated under four policies: **48 generated answers**. The
question uses the pinned text-only chat template with thinking disabled and
requests at most 64 output tokens. Each log crosses the active window repeatedly.

| Policy | Recent code | Evicted code | Conflicting update | Absent code |
|---|---:|---:|---:|---:|
| Rolling window | 3/3 | 0/3 | 3/3 | 3/3 |
| Anchors + recent window | 3/3 | 0/3 | 3/3 | 3/3 |
| Reset and prefill retained window | 3/3 | 0/3 | 3/3 | 3/3 |
| Anchors + lexical history retrieval | 3/3 | 3/3 | 3/3 | 3/3 |

Without retrieval, all three evicted-code answers were `UNKNOWN`. The retrieval
policy queried committed history for the exact phrase `access code`, selected
at most two matching source records, and reinjected them within the same token
budget. Its success is evidence for these lexical probes, not arbitrary
semantic recall. The small sample, regular wording, and known query phrase
limit generalization. No benefit of anchors over the plain rolling baseline is
established by this workload.

[Predictions, inputs, and retrieved records](../results/retention-cpu/predictions.jsonl),
[score counts](../results/retention-cpu/summary.json), and
[metadata](../results/retention-cpu/metadata.json) are preserved. The archived
harness matches its execution hash; the current script only removes an unused
import and redundant key-file read.

## Custom-model pipeline

The [small CPU integration result](../results/pipeline-cpu/summary.json) covers
pretraining, completion-only SFT, DPO with a frozen reference, held-out corpus
loss, bounded decoding, offline W&B, local MLflow, version registration,
promotion, HTTP prediction, and rollback/reload.

The fixture uses random token sequences and 16 artificial preference pairs.
Each training stage runs four optimizer steps; evaluation scores 256 tokens.
These are integration checks, not evidence of language or alignment quality.
The 996,520,448-parameter configuration has been counted on a meta device and
has no trained weights. CPU unit tests separately require bit-exact resumed
training and agreement between two-process DDP and its single-process global
batch across an epoch boundary.

The [single-GPU training check](../results/training-cuda/summary.json) ran eight
FP32 optimizer steps on the RTX 2070 SUPER. Its largest CPU/CUDA weight
difference was 1.53e-7; resumed CUDA weights exactly matched the uninterrupted
CUDA run in this fixture. This checks device execution and resume, without
establishing target-scale memory, throughput, or distributed GPU behavior.

## Kernels and operations

[Triton interpreter results](../results/kernels-interpreter/checks.json) compare
RMSNorm, RoPE, and fused latent decoding with PyTorch in FP32/FP16, including
non-power-of-two dimensions and partial tiles. They establish interpreter
numerical behavior only. The native runtime's CUDA operator checks passed
**138 NEOX and 68 IMROPE supported cases**; unsupported layouts remain listed
in the [NEOX](../results/cuda-operators/rope-2.txt) and
[IMROPE](../results/cuda-operators/rope-40.txt) output. Those native operators
are separate from the unqualified Triton kernels.

[Compose evidence](../results/deployment-compose/summary.json) establishes
unauthenticated-request denial, session commit, restart replay, and a successful
authenticated Prometheus scrape. [Kubernetes evidence](../results/deployment-kubernetes/summary.json)
establishes PVC-backed replay after pod replacement, an observed failed image
rollout, and rollback/replay on local kind/Kubernetes v1.36.4. Recovery timings
are single observations, not service-level objectives. GPU deployment,
autoscaling, distributed storage, and production concurrency remain open.

## Failed compatibility attempts

The stock pinned runtime generated only **498 of 1,536 requested tokens** in the
512-token generation probe and did not cross the context boundary. Its
[summary](../results/compatibility-stock/summary.json),
[server output](../results/compatibility-stock/server.txt), and
[provenance](../results/compatibility-stock/provenance.json) are retained; the
local binary path is redacted. This is a compatibility failure, not a matched
throughput baseline.

An initial guard-only patch then hit a second position assertion. Later session
attempts exposed automatic prompt-cache restoration overwriting stream state.
The [failure ledger](../results/compatibility-failures.json) records these
outcomes and original artifact hashes. The final patch handles positional
metadata, validates explicit eviction, and requires shared prompt RAM caching
to be disabled. No upstream acceptance or algorithmic novelty is claimed.

The first clean-environment CI run also exposed an undeclared NumPy dependency
inside PyTorch's distributed object gathering. NumPy is now pinned in the base
requirements; the checkpoint/resume/DDP regression passes with those installed
requirements on a separate Python 3.12 environment.
