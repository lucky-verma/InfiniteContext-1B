# Full-system roadmap and scope coverage

This roadmap preserves the complete end-to-end project. Work is sequenced by
dependencies and validated artifacts; [STATUS.md](STATUS.md) identifies the
implemented components and their remaining qualification work.

## Scope inventory

| Area | Required capability | Completion artifact |
|---|---|---|
| GPU infrastructure | Ansible provisioning, NVIDIA Container Toolkit, reproducible runtime setup | Idempotent setup and hardware/runtime verification |
| Training orchestration | SLURM launchers, PyTorch DDP/FSDP, NCCL, checkpoint/resume | Successful baseline/distributed runs and recovery evidence |
| Model architecture | 1B-class MLA, decoupled RoPE, MHA/GQA comparators | Tested implementation, model configuration, and trained-checkpoint provenance |
| Streaming context | Rolling state, position handling, continuity, retention, and recovery | Sustained real-model stream with memory and recall/failure traces |
| GPU optimization | Triton RoPE/RMSNorm and fused MLA decoding | Numerical checks, profiler evidence, and matched benchmarks |
| Alignment | SFT, DPO, versioned preference data, TRL/torchtune integration | Reproducible training and held-out comparisons |
| Quality evaluation | Retrieval/passkey, long-context/stream quality, historical recall, judge checks where used | Versioned workload, outputs, scoring, and limitations |
| Experiment/model lifecycle | W&B tracking, MLflow registry, model promotion/rollback | Run-to-checkpoint-to-evaluation-to-service trace |
| Inference | vLLM integration, batching, bounded requests, streaming responses | Reproducible real-model service and load results |
| Deployment/operations | Kubernetes/K3s, health/readiness, HPA, Prometheus/Grafana | Deployment, monitoring, rollout/rollback, and failure-test evidence |
| Release | Setup, model/data provenance, tests, benchmark commands, documentation | A clean reproduction of the integrated system |

## Dependency order

1. **Foundation and infrastructure:** correct attention/cache behavior, tested
   development setup, provisioning and launcher foundations, and an explicit
   model/data strategy.
2. **Training pipeline:** single-device training, experiment records,
   checkpoints, resume, and distributed equivalence/scaling checks.
3. **Model architecture:** MLA and decoupled RoPE with reference comparisons;
   validate small configurations before spending on the target scale.
4. **Streaming integration:** establish model/runtime state compatibility,
   implement rollover, and validate continued input/output and retention.
5. **Kernel optimization:** implement and check the planned kernels, then
   measure their effect on the actual training or serving path.
6. **Alignment and evaluation:** SFT/DPO comparisons, long-context and
   long-stream evaluation, quality/performance tradeoffs, and provenance.
7. **Serving and operations:** vLLM, registry promotion, Kubernetes deployment,
   monitoring, autoscaling, failure recovery, and complete-system reproduction.

Work on an independent prerequisite can proceed alongside the active technical
problem. A resource blocker leaves a capability pending; it does not remove
that capability from the scope.

## Release boundary

Use [VALIDATION.md](VALIDATION.md) to assess each component and the integrated
path. Record incomplete or hardware-blocked work in [STATUS.md](STATUS.md).
A verified attention module, model endpoint, or documentation reorganization
is useful progress but does not complete the end-to-end system.
