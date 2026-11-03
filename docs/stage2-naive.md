# Stage 2 — Naive engine (no KV)

**Flag:** defaults / `--preset v0`  
**Benchmark:** `tiny-engine bench 2` or `python benchmarks/bench_offline.py --kv-caches none`

## Problem

Need a working autoregressive loop and an OpenAI-compatible server before adding caches or batching.

## What happens

```text
generate / HTTP
  → encode prompt
  → add_request → scheduler.waiting
  → while unfinished:
       step():
         schedule()                 # FIFO, max 1 request
         ModelRunner.forward(ALL token_ids)   # use_cache=False
         sample one token
         append / maybe finish
```

Because `kv_cache="none"`, `LLMEngine.step` takes `_step_no_cache`. Every step re-runs the **full** sequence through Hugging Face. Cost grows with prompt + outputs so far.

Serving: FastAPI → `AsyncEngine` (background thread) → same `step()` loop. Port **8001**.

## Code

| File | Role |
|------|------|
| `engine.py` | `add_request`, `step`, `_step_no_cache`, `generate` |
| `model/loader.py` | Load Qwen2 weights |
| `model/runner.py` | Full-sequence forward, last-position logits |
| `scheduler/fifo.py` | One request at a time |
| `request.py` / `sampling.py` | State, stops, Sampler |
| `async_engine.py` / `serving/` | HTTP ↔ engine thread |

## Run

```bash
tiny-engine generate --show-steps --temperature 0 "Hi"
tiny-engine serve --port 8001
```

## Expect

Step times climb as the sequence grows. Under concurrency, second clients wait (FIFO). That baseline is what KV and batching improve.
