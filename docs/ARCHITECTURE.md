# End-to-end ML systems architecture

## Model and attention

The model target is a configurable 1B-class language model using Multi-Head
Latent Attention (MLA) and decoupled RoPE. Keep conventional MHA and GQA
comparators available for correctness and matched memory/performance studies.
Small configurations support fast checks of the same implementation; they do
not establish the quality or performance of a trained 1B model.

MLA compresses retained KV state. Validate projection shapes, positional
handling, prefill/decoding equivalence, cache storage, and gradients before
training. A claimed memory reduction must specify the comparator, layer/head
configuration, dtype, and which allocations it counts.

The initialization strategy must be explicit: compatible released weights,
distillation, or pretraining. Standard-model weights cannot be assumed to fit
an MLA architecture. Contemporary pretrained models also serve as runtime and
quality comparators; they do not replace the target model work.

## Data, distributed training, and alignment

Version public data sources, licenses, preprocessing, tokenization, train/eval
splits, and sequence packing. Keep evaluation examples outside training and
preference-pair construction where they are intended as held-out measurements.

Training includes a single-device baseline, PyTorch DDP/FSDP, NCCL
communication, SLURM launchers, checkpointing, and resume. Preserve model,
optimizer, scheduler, random-state, data-position, and configuration identity
as needed for a valid continuation. Measure scaling on actual devices and
report communication and data-loading costs.

SFT and DPO remain part of the alignment workflow. Use maintained TRL or
torchtune components where compatible. Preference construction, train/eval
separation, and comparison with the unaligned/SFT model must be inspectable.
Improved retrieval or reduced hallucination is an empirical question.

W&B records experiment traces; MLflow supplies model/artifact registry and
lifecycle metadata. Their responsibilities should be distinct. A promoted
model must link to its checkpoint, configuration, evaluation, and serving
version, with a repeatable rollback path.

## Streaming context

A session should continue accepting input and producing output as total history
passes its active window. The context manager owns cache retention/eviction,
position handling, request ordering, state identity, and recovery.

| Resource | Contract to establish | Evidence |
|---|---|---|
| Active token window | Fixed configured capacity, including retained anchors, recent tokens, and any retrieved material | Accounting at append, prefill, decode, and rollover |
| KV/recurrent state | Model-specific bounded allocation and valid state transformation | Cache/state bytes and actual device memory across repeated rollovers |
| Persistent history | Durable accepted input when historical retention is enabled; storage grows with history | Ordered records, storage growth, replay, and interruption tests |
| Historical recall | Explicitly limited by retention and retrieval policy | Distance-based probes, conflicts, absent facts, and failures |
| Recovery | Resume from acknowledged input with visible reconstruction cost | Checkpoint/restart behavior, offsets, and timing |

Continued streaming, a large native attention window, and exact recall of all
history are different capabilities. Report the achieved stream length, active
window, storage costs, and observed retention separately. Finite experiments
do not establish literally unlimited full-attention context.

The implementation must establish which state survives eviction, how rotary
positions change, and whether a request reuses cache or re-prefills a
reconstructed prompt. Hybrid recurrent models need their own validated state
contract. Streaming HTTP output alone is not streaming-context management.

Compare a plain rolling window with an anchor/sink policy on the same stream.
If persistent retrieval is included, report its added storage, latency, and
recall contribution separately. The intended retention of evicted history
remains an explicit design decision.

The established [StreamingLLM baseline](https://github.com/mit-han-lab/streaming-llm)
retains recent tokens and attention sinks while discarding middle tokens. Its
FAQ distinguishes this behavior from long-term recall. Attribute any reused
method and validate its transfer to the selected model/runtime.

## GPU kernels

Triton RoPE, RMSNorm, and fused MLA decoding remain planned components. Keep a
PyTorch reference for numerical and gradient checks where relevant. Test
supported shapes, lengths, dtypes, layouts, masking, and boundary conditions.

Use profiler traces to identify the bottleneck, then measure the kernel and
its end-to-end effect. Include launch, memory movement, and fallback costs.
Avoid a speedup claim based only on a favorable shape or incompatible baseline.

## Evaluation

Evaluate model quality, alignment effects, retrieval/passkey accuracy,
long-stream behavior, historical recall, cache memory, training efficiency,
and serving performance. Separate theoretical byte counts from allocator
measurements and complete-model memory.

Long-context experiments may include 4K, 32K, 128K, and 1M token regimes when
supported by the model and hardware. Distinguish total streamed tokens from
tokens simultaneously available to attention. An out-of-memory result or
quality failure is part of the result, not a missing table cell.

An automated judge such as Prometheus-Eval can supplement task-specific
metrics with an explicit rubric and reliability checks. It is distinct from
Prometheus service monitoring. The full protocol is in
[VALIDATION.md](VALIDATION.md).

## Serving and operations

vLLM integration is the serving target, including model identity, bounded
requests/concurrency, streaming output, batching, KV behavior, and error paths.
Other maintained runtimes can provide comparisons and compatibility probes.
A successful alternate-runtime probe does not establish vLLM integration.

Provision GPU hosts using Ansible and validate the driver, CUDA/runtime,
NVIDIA Container Toolkit, storage, and communication configuration. SLURM
supports training execution; Kubernetes/K3s supports service deployment.

Operational work includes container builds, health/readiness, resource limits,
request admission, cancellation, overload handling, restart recovery, rollout,
rollback, and autoscaling. Prometheus/Grafana expose latency, errors, queue
pressure, model/cache resources, and device utilization. HPA behavior must be
validated against the selected scaling signal and available capacity.

Keep the serving surface authenticated and appropriately isolated when
exposed beyond a local development machine. Protect model/artifact integrity,
secrets, and private inputs. Record the deployment, operational failure tests,
and model version together.
