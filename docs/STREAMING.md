# Streaming context and recovery

The runtime processes a growing text stream while retaining a bounded active
window. A SQLite log stores accepted input and completed output on disk. An
FTS5 index can retrieve committed input records for explicit reinjection. Neither
an attention sink nor a recurrent state guarantees exact recall of evicted text.

## Model and runtime identity

`serving/model.json` pins Qwen3.5-0.8B, its GGUF conversion revision, Q8_0 file,
size, SHA-256, and Apache-2.0 source license. It is a contemporary small hybrid
reference that fits the validation hardware. It is separate from the custom
MLA architecture and is not selected as the newest or highest-quality model in
its size class. A model update must repeat compatibility, quality, and resource
checks before replacing the reference.

`serving/runtime.json` pins llama.cpp source and the carried patch. The builder
checks the patch hash, refuses unexpected source changes, runs rotary/metadata
checks, and records the binary and shared-library hashes. The launcher verifies
those receipts and the model bytes before starting. CUDA builds select the
visible GPU architecture; portable CPU container builds disable host-native
instruction selection. Build receipts identify those different binaries.

## Why the patch exists

The stock pinned runtime rejected shifting for the reference model's multiple
position components. Enabling one guard exposed a second assertion. The carried
patch permits the supported MROPE/IMROPE translation, updates spatial cache
metadata, and checks the translation against the existing rotary reference.
The scalar NEOX translation builds on upstream
[issue 13865](https://github.com/ggml-org/llama.cpp/issues/13865) and
[PR 13870](https://github.com/ggml-org/llama.cpp/pull/13870).

A separate server change adds `n_cache_shift`: the exact count to remove after
`n_keep`. Before an explicit shift, the backend checks that the requested window
matches the retained token prefix and tail. This makes new repeated input
unambiguous even when its text equals previously cached text. Shared prompt RAM
caching is disabled (`--cache-ram 0`), because automatic restoration of a
content-matched prompt can replace the session's state. A zero-output request
also evaluates its input without sampling an unwanted token.

The tested path is text-only and uses one leased slot. Multimodal shifting,
arbitrary rotary modes, parallel slots, and vLLM behavior are not established by
this patch. The hybrid recurrent state is retained as history advances; its
information content differs from a fresh prefill of the same surviving tokens.
The retention evaluation therefore includes a reset-and-prefill comparator.

## Session contract

`python -m streaming.cli` accepts JSONL objects with `id`, `text`, and optional
`generate`. It produces JSONL events:

| Event | Meaning |
|---|---|
| `accepted` | Input and its ID have been committed to the local request log |
| `delta` | Provisional generated text; cancellation/restart can replace it |
| `complete` | Output and active-token accounting have been committed |

A duplicate completed ID returns the stored result. Changing its input or
sampling budget is an error. An unfinished ID must be retried before a later
request, and its next `accepted` event has `restart: true`. Consumers should
replace any earlier provisional text for that ID.

The session reserves generation capacity before eviction, retains configured
anchors plus recent tokens, and leaves room for recurrent-state continuity.
Inputs over the byte/token budget fail before acceptance. A process lease
protects both the database and backend slot. This intentionally serializes
sessions on one backend; it is not a multi-tenant scheduling layer.

Snapshots contain native model state and checksums. Recovery restores the last
snapshot and replays committed consumed-token prefixes after it. The last
sampled token may still be pending in the model, so the log records the exact
cached-token count. Cancellation closes the backend response before recovery
waits for the slot to become idle. Two snapshots are retained. SQLite history,
token records, and its lexical index grow with the stream; disk capacity remains
a real limit. Preserve the database, WAL, snapshots, model, and runtime identity
when moving a session.

## Historical retrieval

```bash
python -m streaming.cli --window 512 --search '"station code"'
python -m streaming.cli --window 512 --history-after 0
```

Search returns matching committed source records, newest first, with their
request IDs and sequence numbers. It does not silently modify the prompt. A
caller can cite selected records and submit their text within the same active
token budget. Search is lexical FTS5 matching, not semantic retrieval or perfect
historical recall. Current CLI history/search opens a session and requires its
backend; direct SQLite access remains available for offline inspection.

## Chat formatting

The session accepts raw text/tokens; it does not silently wrap each chunk as an
independent chat turn. For a multi-chunk Qwen user message, start with
`<|im_start|>user\n`, append body chunks, then close with
`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
and set `generate` on that final record. This follows the pinned tokenizer's
text-only, thinking-disabled template. Ordinary input containing these special
tokens can affect prompt structure; treat it as model input, not executable code.

## Reproduction

```bash
python scripts/check_streaming.py --backend cuda \
  --min-input-tokens 1000000 --max-seconds 1800 --output .runs/million
python -m serving.run --window 512 --port 18082
# In another terminal:
python scripts/evaluate_retention.py --endpoint http://127.0.0.1:18082 \
  --seeds 3 --output .runs/retention
```

The stream harness requires no active GPU compute process and sufficient free
memory. `--max-gpu-utilization` permits a declared ceiling of at most 30% for
background graphics activity; it never bypasses the active-compute check. The
million-token workload is repeated telemetry prose with zero requested output
per ingestion chunk, followed by generation/recovery checks. Its throughput is
not a mixed interactive chat benchmark.

The retention harness compares rolling context, anchors plus recent context,
reset-and-prefill, and explicit lexical retrieval. It logs full synthetic inputs,
expected answers, generated answers, retrieved records, and exact-match scores.
The source workload is small and diagnostic; see [RESULTS.md](RESULTS.md).
