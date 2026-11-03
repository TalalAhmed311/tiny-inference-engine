# Stage 5 — Scheduler and continuous batching

**Flags:** `-f static` · `-f batching` · `-f chunked` (pulls batching → paged)  
**Presets:** `--preset batching`  
**Benchmark:** `python benchmarks/stage5_scheduling.py`

## Problem

KV makes one sequence cheap to extend. FIFO with one seat still serializes clients — the GPU reads weights to produce one token while other requests wait.

## What happens

The scheduler does **not** run the model. Each `step` it returns:

```text
[(request, num_new_tokens), ...]
```

- `num_new_tokens == remaining` → sample this step  
- `num_new_tokens < remaining` → prefill chunk, no sample (`num_logits=0`)

### Policies

| Policy | Behavior |
|--------|----------|
| **FIFO** | `max_num_seqs=1` — one request until done |
| **Static** | Admit a batch; no new joins until the whole batch finishes |
| **Continuous** (`-f batching`) | Every step: run current seqs, then admit waiters while seats / KV / token budget allow |
| **Chunked** (`-f chunked`) | Continuous + hard token budget; long prompts split so they share steps with decode |

Shared helpers in `scheduler/base.py`: `_allocate`, `_try_admit`, `_schedule_running`, `_preempt` (drop KV, recompute later).

### Packed forward (batching execution)

```text
BatchItem per scheduled seq (new tokens only)
CachedModelRunner packs them into one flat batch
attention: short queries (decode) batched; long prefills often alone
sample only rows with num_logits > 0
```

That is why concurrency raises tok/s: many sequences share one weight read.

## Code

| File | Role |
|------|------|
| `scheduler/base.py` | Queues, admit, preempt |
| `scheduler/fifo.py` / `static.py` / `continuous.py` | Policies |
| `engine._step_cached` | Multi `BatchItem` |
| `model/cached_runner.py` | Pack + execute |

## Run

```bash
tiny-engine serve -f paged,batching --port 8001
tiny-engine serve -f paged,batching,chunked --port 8001
python benchmarks/stage5_scheduling.py
```

## Expect

Largest multi-request QPS jump in this engine (often ~10×+ vs FIFO). Cap still exists (`max_num_seqs`, Python/gather overhead vs vLLM).
