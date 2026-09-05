# InfiniteContext contributor and agent contract

## Full project objective

Build the complete ML systems stack described in README.md: a 1B-class MLA
model, distributed training, GPU kernels, SFT/DPO alignment, long-context and
streaming evaluation, model lifecycle, vLLM serving, and infrastructure/operations.
Streaming-context management is a central capability within this end-to-end
system. Supporting components remain in scope.

Preserve the agreed capabilities. A deadline, resource limitation, or missing
implementation changes execution order or validation status; it does not
silently remove requirements. Do not replace the system with a generic serving
demo or mark an aspirational architecture as implemented.

## Read first

1. `README.md` — public scope and implementation boundary.
2. `docs/STATUS.md` — current work and next execution.
3. `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, and `docs/VALIDATION.md` — system
   contracts, complete scope, and completion criteria.
4. Applicable source, callers of changed functions, and relevant checks.

Keep durable technical context in `docs/`. Editor configuration, tool caches,
session memory, credentials, and downloaded weights are not tracked context.

## Public and private information

Public files, commit messages, issues, and repository metadata contain technical
project information only. Keep personal deadlines, availability, career plans,
job targets, private budgets, employer details, and personal conversations in
the user's private workspace. Permission to reorganize or publish technical
files is not permission to publish that private context.

Before any remote update, review the exact changed lines for private context,
unsupported claims, and unrelated work. Public roadmaps describe capabilities
and dependencies, not personal scheduling. Preserve source and license
attribution without adding AI co-author or attribution lines.

## Model and streaming decisions

- Select models using current primary sources, architecture, runtime support,
  workload, and actual hardware. Record revision and quantization provenance.
  Label smoke-test fixtures explicitly; they are not benchmark defaults.
- Keep the target MLA model and contemporary pretrained comparators distinct.
  An incompatible pretrained checkpoint is not an MLA initialization strategy.
- Separate total stream length, active window, cache/recurrent state, persistent
  storage, and historical recall. State which resources grow with history.
- Verify eviction, positional encoding, and recurrent-state behavior on the
  actual model/runtime. A transformer-only method may not transfer unchanged
  to a hybrid architecture.
- Bounded cache does not establish perfect access to every past token. Measure
  information loss, retrieval failures, and reconstruction costs explicitly.
- Reuse maintained methods where they satisfy the contract. Attribute them and
  compare with the simplest relevant baseline. Do not claim algorithmic novelty
  from integration or rename an existing technique as a new one.

## Implementation and checks

The owning implementations are `training/src/modeling_mla.py` (custom MLA),
`training/run.py` (pretraining/alignment), `streaming/session.py` (durable
session), and the pinned patch under `patches/llama.cpp/`. The pretrained native
reference and custom MLA checkpoint/MLflow service are distinct execution paths.

Run `python -m unittest discover -s tests -v` for CPU contracts. Native patch or
session changes also require `scripts/check_streaming.py`; alignment/lifecycle
changes require `scripts/check_pipeline.py` with the tracking dependencies.
Triton interpreter checks, real CUDA checks, Compose checks, and Kubernetes
checks have separate scripts and evidence scopes. Preserve exact source hashes
with benchmark results; never relabel an earlier measurement as a new run.

- Trace the complete affected flow and every caller before a shared-function
  fix. Prefer existing code, stdlib, native features, and installed dependencies.
- Keep implementations small while preserving the requested capabilities,
  input validation, causal semantics, resource limits, and data recovery.
- Give nontrivial behavior a runnable regression check. Test boundary and
  failure behavior as well as successful execution.
- Check live occupancy before GPU work. Do not interrupt another workload or
  infer spending authority from urgency.
- Pin code, model, runtime, data, and workload identity. Preserve raw output;
  compare equivalent workloads and report quality and failures alongside speed.
- Separate CPU checks, synthetic tests, real-model GPU runs, distributed runs,
  and operational evidence. None substitutes silently for another.
- Expect a dirty tree. Preserve unrelated changes. Stage explicit paths only
  and inspect the staged diff and commit path list before publishing.
- Commit, push, publish, spend, or contact maintainers only when authorized for
  that action. Use agents only when the current request permits them.

## Completion and continuity

Update `docs/STATUS.md` after material results, including evidence, limitations,
and the next decisive step. Keep architecture and roadmap aligned with explicit
user changes. A component remains pending or blocked until its checks pass.

Before calling the full system complete, execute the end-to-end path and the
relevant component checks in `docs/VALIDATION.md`. Report missing evidence
plainly; documentation cleanup or a model HTTP endpoint does not establish
system readiness.
