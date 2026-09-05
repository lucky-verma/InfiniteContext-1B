# Implementation and qualification status

The repository now contains executable model, training/alignment, streaming,
evaluation, registry, kernel-reference, and deployment paths. The strongest
real-model result is a million-token bounded-window stream with deterministic
continuation after recovery. The full target system is **not release-complete**:
the 1B model is untrained, and several scale/integration gates remain open.

## Current evidence

| Area | Verified result | Remaining qualification |
|---|---|---|
| Attention and MLA | CPU reference outputs/gradients, causal decoding, bounded cache, exact parameter count | GQA comparator and matched whole-model comparisons |
| Training | Small-model CPU/CUDA training and resume, bit-exact CPU resume, two-process CPU DDP/global-batch equivalence | Multi-GPU FSDP/NCCL, target-scale memory/throughput, precision/activation optimization |
| Data | Mapped uint32 loader, tokenized preferences, data/checkpoint hashes, distinct synthetic evaluation fixture | Licensed real corpus, tokenizer/source manifests, semantic deduplication and contamination audit |
| Alignment | Native completion-only SFT and reference-relative DPO; small pipeline and objective checks | TRL/torchtune adapters and meaningful held-out preference-quality comparisons |
| Streaming | 1,000,083 input tokens; 512-token window; at most 511 active; repeated eviction; disk recovery and identical deterministic continuation | Diverse long streams, concurrent sessions, model/version upgrades, stronger quality and failure envelopes |
| Historical retention | 48 synthetic probes across four policies; explicit lexical retrieval recovers the three evicted codes missed by the other policies | Larger independently held-out tasks, semantic retrieval, adversarial/conflicting sources, retrieval latency and storage growth studies |
| Kernels | CPU interpreter comparisons for Triton RMSNorm/RoPE/latent decode; native CUDA rotary operator checks | SM80+ Triton execution, profiler traces, matched performance, and integration into the trained model |
| Model lifecycle | Offline W&B, local MLflow, checkpoint/evaluation identity, registration, alias promotion, HTTP prediction, rollback/reload | Real-model acceptance thresholds, artifact storage/security operations, deployed model reload orchestration |
| Native serving | Pinned artifact verification, authentication, single-slot ownership, cancellation/retry, bounded requests | Multi-session admission, fairness, load/overload and batching studies |
| Containers and monitoring | Non-root/read-only Compose deployment, authentication denial, restart replay, authenticated Prometheus scrape | GPU image qualification, durable metrics, Grafana rendering and alert/load tests |
| Kubernetes | Local kind StatefulSet, PVC-backed pod-replacement replay, observed bad rollout and successful rollback | K3s, GPU scheduling, multi-node storage/failure, network-policy enforcement, and HPA with session routing |
| Host/cluster provisioning | Ansible and SLURM launch recipes; syntax checks | Actual host provisioning/idempotence, NVIDIA Container Toolkit/driver lifecycle, and SLURM allocation |
| vLLM | Integration requirements and environment boundary documented | Custom MLA model/weight adapter and verified batching/session integration |

[RESULTS.md](RESULTS.md) links the raw evidence. [VALIDATION.md](VALIDATION.md)
remains the release contract; a passing synthetic pipeline does not close its
quality, scale, or production-operation gates.

## Next decisive execution

1. Choose and version a licensed corpus/tokenizer and held-out workloads. Train
   a useful smaller MLA checkpoint before allocating target-scale compute.
2. Qualify the same checkpoint and training path on the intended GPU topology;
   measure memory, resume, FSDP/NCCL scaling, and communication cost.
3. Run broader streaming and retrieval-quality comparisons with contamination
   checks and meaningful absent/conflicting-fact cases.
4. Qualify Triton kernels on supported hardware and integrate only measured wins.
5. Implement the vLLM model/weight path and session ownership under batching,
   then qualify multi-session service operation and autoscaling.

An upstream patch has not been submitted or accepted. The root repository does
not yet declare a project-wide source license; the carried llama.cpp patch
retains the upstream MIT notice and the downloaded model has its own license.
These decisions remain explicit before a broader distributable release.
