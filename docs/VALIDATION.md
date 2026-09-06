# Validation and release evidence

## Component checks

| Component | Required checks |
|---|---|
| Attention/model | Reference numerical behavior, causal masking, prefill/decode equivalence, cache boundaries, gradients, serialization, and declared architecture/configuration |
| Data/training | Data identity and split separation, checkpoint integrity, resume semantics, repeatability, and baseline/distributed comparison |
| Alignment | Preference-data provenance, held-out evaluation, SFT/DPO comparison, and quality regressions |
| Streaming | Continued input/output, repeated rollover, valid position/state transitions, bounded active memory, historical retention/failures, and recovery |
| Kernels | Reference agreement, supported shapes/dtypes/layouts, boundary handling, and end-to-end performance under matched conditions |
| Serving | Model identity, request/token limits, concurrency, cancellation, errors, overload, health/readiness, and streaming completion semantics |
| Operations | Provisioning, deployment, restart, model promotion/rollback, monitoring, and autoscaling behavior on actual infrastructure |

Operations also requires the [infrastructure milestones](ROADMAP.md#infrastructure-and-operations-milestones):
DNS fault recovery, enforced network isolation, authenticated gateway routing,
GitOps drift/release recovery, and session-safe scaling. Qualify the chosen CNI;
Rook/Ceph and Twingate checks apply only when those products are adopted.

## Benchmark contract

Record code, runtime, model/revision, quantization, data/workload, hardware,
configuration, and commands. Preserve raw outputs and failed attempts. Declare
warm-up, repetition, sampling/decoding, input/output lengths, concurrency, and
the meaning of every timing and memory metric.

For streaming, compare ordinary bounded context, a rolling-window baseline,
and the chosen policy. Include a feasible full-context reference when useful,
with its different resource cost. Cross the active window repeatedly and
record token offsets, eviction events, latency, throughput, cache/state bytes,
actual device memory, and failures.

Probe recent and distant facts, conflicting updates, and facts no longer
available to the model. Report denominators, missing answers, and retrieval
failures. Preserve quality checks when changing cache policy, precision,
batching, or kernels.

Distinguish CPU checks, synthetic workloads, real-model GPU runs, distributed
runs, and operational evidence. A small configuration validates only its tested
behavior; it does not establish larger-model quality, scale, or hardware fit.

## End-to-end release check

A clean environment must be able to reproduce the declared path:

1. Prepare the versioned data and configuration.
2. Train or adapt the declared model and verify checkpoint/resume.
3. Run quality, alignment, and streaming-context evaluation.
4. Register the evaluated model and deploy the corresponding version.
5. Exercise the real inference and continuous-session workload under load.
6. Observe resource use and failures, interrupt/recover the service, and roll
   back to a known model version.
7. Reproduce the reported kernel/system comparisons and their limitations.

The release package includes usable setup and execution instructions, source
and model/data license information, tests, raw results, and an independent
reproduction record when available. A missing load-bearing implementation or
validation remains an explicit blocker to calling the full system complete.
