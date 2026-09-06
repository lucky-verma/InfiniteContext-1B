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
| Service networking | CoreDNS, one CNI (Cilium or Calico), NetworkPolicy, Gateway API/TLS | Service discovery, allowed/denied traffic, authenticated routing, and fault-recovery evidence |
| GitOps delivery | Argo CD, versioned manifests, reconciliation, drift detection | Deployment from Git, drift repair, failed-release detection, and recovery to a known version |
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
   DNS/network isolation, GitOps, monitoring, session routing, autoscaling,
   failure recovery, and complete-system reproduction.

Work on an independent prerequisite can proceed alongside the active technical
problem. A resource blocker leaves a capability pending; it does not remove
that capability from the scope.

## Infrastructure and operations milestones

Extend the [existing deployment checks](OPERATIONS.md#kubernetes) with the
following qualification work. These are planned requirements; implementation
and measured results remain in [STATUS.md](STATUS.md). DNS, networking, and
GitOps checks can use the existing single-session service while model work
continues. Qualify one CNI in an isolated test cluster and document the choice
between Cilium and Calico.

| Area | Capability to exercise | Completion evidence |
|---|---|---|
| Kubernetes/K3s | StatefulSets/Jobs, probes, resource requests/limits, RBAC, PVC/CSI storage, and GPU scheduling | Reproduce deployment on K3s; diagnose unavailable GPU capacity, failed probes, and resource exhaustion; verify restricted permissions and persistent-session recovery |
| [CoreDNS](https://kubernetes.io/docs/tasks/administer-cluster/dns-debugging-resolution/) | Service discovery, DNS search paths, and upstream resolution | Reproduce and diagnose a failed lookup and upstream timeout; restore resolution and verify the service workload |
| [Cilium](https://docs.cilium.io/en/stable/overview/intro/) or [Calico](https://docs.tigera.io/calico/latest/about/) | Pod/service routing and enforced [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/) | Demonstrate default-deny isolation, required DNS/artifact/metrics traffic, and denied unauthorized traffic; trace and repair an intentionally blocked connection |
| [Gateway API](https://gateway-api.sigs.k8s.io/docs/) and TLS | Authenticated application routing, certificate validation, streaming timeouts, and cancellation | Exercise a stream through one maintained gateway implementation; verify authentication denial, cancellation, and isolation of state-management/admin endpoints |
| [Argo CD](https://argo-cd.readthedocs.io/en/stable/) | Git as desired state, image/configuration identity, sync health, and drift repair | Reconcile versioned manifests with credentials supplied outside Git; detect a bad image release, restore the known version through Git, and verify committed-session replay |
| Session routing and HPA | Isolated session state, one writer per session, bounded admission, graceful draining, and a declared scaling signal | Verify concurrent-session isolation and safe retries; exercise scale-out, scale-in, and node loss with no duplicate committed records or concurrent state writers |
| Prometheus/Grafana | Durable metrics, rendered dashboards, latency, errors, queue pressure, storage growth, and GPU use | Run a declared load, trigger an alert, and record recovery against predeclared latency/error/recovery limits and available capacity |

Keep the reference StatefulSet at one replica until session ownership, routing,
storage access, and drain/recovery checks pass. A replica increase cannot supply
those semantics. A Git revert restores deployment configuration; persistent
state compatibility and backup/restore require their own checks.

The integrated exercise deploys the service through Git, injects DNS and policy
faults, recovers a failed release, and replaces a worker. Preserve commands,
cluster/CNI/controller versions, manifest and image identities, workload,
events, metrics, and committed-record replay results. Record CPU-only and GPU
evidence separately under the [validation contract](VALIDATION.md).

### Conditional infrastructure

- **[Rook/Ceph](https://rook.io/docs/rook/latest/Getting-Started/intro/):** adopt
  when the target deployment requires operating shared/distributed storage.
  First qualify PVC/CSI behavior with the selected storage provider. A Ceph
  deployment additionally needs node/disk-failure, attachment, and backup/restore
  evidence; replication alone does not establish recoverability.
- **[Twingate](https://www.twingate.com/docs/how-twingate-works):** consider when
  private operator access is required and the existing access route does not
  satisfy it. Verify authorized/denied access, revocation, and connector-failure
  recovery. Keep application authentication and Kubernetes RBAC in place.

These product-specific additions become release requirements only when adopted
for the declared deployment. Core networking, storage recovery, and access
control remain required regardless of the selected products.

## Release boundary

Use [VALIDATION.md](VALIDATION.md) to assess each component and the integrated
path. Record incomplete or hardware-blocked work in [STATUS.md](STATUS.md).
A verified attention module, model endpoint, or documentation reorganization
is useful progress but does not complete the end-to-end system.
