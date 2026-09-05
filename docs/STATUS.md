# Implementation status

## Published baseline

The repository contains educational attention/KV-cache source and synthetic
development scripts. It does not yet contain a published, validated
end-to-end training-to-streaming result.

| Area | Current state |
|---|---|
| Attention foundation | Educational implementation present; correctness hardening is in progress |
| MLA model and decoupled RoPE | Planned implementation and initialization strategy |
| Data, distributed training, checkpoint/resume | Planned |
| SFT/DPO alignment | Planned |
| Streaming context manager | In development; real rollover and retention remain unvalidated |
| Triton kernels | Planned |
| Evaluation | Cache-size/demo utilities present; full protocol pending |
| W&B/MLflow lifecycle | Planned |
| vLLM serving | Planned integration |
| Ansible, SLURM, Kubernetes/K3s, monitoring/HPA | Planned implementation and operational validation |

Local development can be ahead of the published baseline. Results become
public claims only with their corresponding source, configuration, and evidence.

## Current technical problem

The streaming path needs model-specific state/position compatibility. A local
contemporary-model probe exposed a runtime guard that disables KV shifting for
multiple position components. A server flag alone is insufficient. Resolve the
actual state transformation with equivalence checks, or validate another
compatible runtime/model path.

This problem does not remove the model, distributed-training, kernel,
alignment, serving, or operations requirements. The complete scope remains in
[ROADMAP.md](ROADMAP.md), with checks in [VALIDATION.md](VALIDATION.md).

## Next execution

Finish the attention/setup checks and the first validated streaming rollover.
Preserve the failed compatibility probe, establish the positional contract,
and keep independent training/infrastructure prerequisites moving where their
inputs are available. Update this file as implementations and evidence land.
