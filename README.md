# Tiny Inference Engine

<p align="center">
  <img src="assets/inference-engine-man-emoji.jpg" alt="Inference Engine Man" width="160" />
</p>

A small **LLM inference / serving engine** in Python + PyTorch, built around Hugging Face **Qwen2** weights (default: `Qwen/Qwen2.5-1.5B-Instruct`).

The transformer is **not** reimplemented. Embeddings, norms, Q/K/V/O, MLP, RoPE, and the LM head stay Hugging Face modules. This project owns what a real inference engine owns:

- request lifecycle and the `add_request` / `step` / `generate` loop  
- KV cache (contiguous + paged) and prefix reuse  
- scheduling (FIFO, static, continuous batching, chunked prefill)  
- sampling, streaming, weight-only quantization  
- an **OpenAI-compatible HTTP server** (chat + completions + metrics)

Same API shape as vLLM, so the same smoke tests and load clients can hit this server on port **8001**.

---

## Why this exists

Modern serving stacks (vLLM, SGLang) are hard to learn from the outside. This repo is a **teaching engine**: every major idea is a feature switch you can turn on one at a time, measure, and read in a few hundred lines instead of a full production tree.

Typical loop:

```text
naive autoregressive serve
  → contiguous KV
  → paged KV
  → continuous batching (+ chunked prefill)
  → prefix caching
  → optional int8/int4/fp8
  → optional CUDA graphs (decode)
```

---

## Quick start

```bash
cd ~/tiny-inference-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# optional: cache Hugging Face downloads
export HF_HOME=~/hf-cache

# interactive menu
tiny-engine

# or one-shot
python -m tiny_engine generate --temperature 0 "Explain the KV cache"
python -m tiny_engine serve --port 8001
```

GPU recommended for real benches. CPU/MPS can still run small models for learning the control path.

### OpenAI-compatible server

```bash
# production-like feature set for serving
tiny-engine serve --features paged,batching,chunked,prefix --port 8001

# naive baseline (no KV, one request at a time)
tiny-engine serve --port 8001
```

Then:

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Endpoints: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/metrics`.

---

## Features (stage map)

| Feature / flag | Stage idea | What turns on |
|----------------|------------|----------------|
| *(default)* | Naive loop | Full re-forward every step, FIFO, no KV |
| `-f kv` | Contiguous KV | One reserved slot range per request |
| `-f paged` | Paged KV | 16-token blocks + block tables |
| `-f static` | Static batching | Fixed batches drain before new admits |
| `-f batching` | Continuous batching | Join/leave every engine step |
| `-f chunked` | Chunked prefill | Long prompts split under a token budget |
| `-f prefix` | Prefix caching | Reuse KV blocks for shared prompt prefixes |
| `-f int8` / `int4` / `fp8` | Weight-only quant | Swap attn/MLP Linears for QuantLinear |
| `--enable-cuda-graph` | Decode graphs | Capture/replay pure-decode buckets |

Rules of thumb:

- `prefix` requires `paged`  
- `chunked` pulls in `batching` → paged  
- Alternatives in one group: pick one KV layout, one scheduler, one quant  

```bash
tiny-engine features
tiny-engine config --features paged,batching,prefix
tiny-engine chat --features paged,batching,prefix
```

---

## How a step works

```text
waiting → Scheduler (who runs, how many new tokens)
            ↓
         KV manager (allocate / prefix hit / block tables)
            ↓
         CachedModelRunner packs new tokens into one batch
            ↓
         per layer: HF modules + our attention over the KV pool
            ↓
         Sampler → append tokens → stream / free finished KV
```

Naive mode (`kv_cache=none`) skips the pool and re-runs the whole sequence through HF every step (`ModelRunner`).

Serving wraps the same engine in `AsyncEngine`: HTTP stays on asyncio; the blocking forward runs on a background thread.

---

## Repository layout

```text
tiny-inference-engine/
├── tiny_engine/           # the package
│   ├── engine.py          # LLMEngine: add_request / step / generate
│   ├── async_engine.py    # thread ↔ asyncio for the server
│   ├── request.py         # per-request state, stops, TTFT
│   ├── sampling.py        # SamplingParams + Sampler
│   ├── config.py / cli.py / features.py / main.py
│   ├── cache/             # pool, contiguous, paged, prefix
│   ├── scheduler/         # fifo, static, continuous
│   ├── model/             # loader, runner, cached_runner, forward, attention, cuda_graph
│   ├── quantization/      # QuantLinear + quantize_model
│   └── serving/           # FastAPI OpenAI server
├── benchmarks/            # offline + stage benches + online wrapper
├── scripts/               # generate / serve helpers
├── tests/                 # unit + greedy HF parity + API tests
├── docs/                  # one doc per stage (see docs/README.md)
├── pyproject.toml
└── requirements.txt
```

### Docs (one per stage)

Start at [`docs/README.md`](docs/README.md). Then:

| Doc | Stage |
|-----|--------|
| `docs/stage2-naive.md` | No KV, FIFO |
| `docs/stage3-kv-cache.md` | Contiguous KV |
| `docs/stage4-paged-kv.md` | Paged blocks |
| `docs/stage5-scheduling.md` | Continuous / static / chunked |
| `docs/stage6-prefix-caching.md` | Prefix block reuse |
| `docs/stage8-quantization.md` | int8 / int4 / fp8 |
| `docs/cuda-graphs.md` | Decode CUDA graphs |

---

## CLI reference

```text
tiny-engine                         interactive menu
tiny-engine features                list features
tiny-engine config [options]        resolve config (no model load)
tiny-engine chat [options]          terminal chat + metrics
tiny-engine generate [options] "…"  one-shot (+ --show-steps)
tiny-engine serve [options]         OpenAI server (default :8001)
tiny-engine bench STAGE             2 | 3 | 4 | 5 | 6 | 8 | online
```

Also: `python -m tiny_engine …` if the console script is not on `PATH`.

---

## Tests

```bash
TINY_SKIP_MODEL_TESTS=1 pytest -q    # caches, scheduler, attention, quant (no big download)
pytest -q                            # + greedy match vs transformers (small Qwen)
```

Key correctness checks: engine modes (contiguous / paged / static / continuous / chunked / prefix / preemption) must still match HF greedy tokens when those tests run.

---

## Results (what actually moved)

Measured on **NVIDIA A10G**, model **`Qwen/Qwen2.5-1.5B-Instruct`**, short chat prompts, greedy / fixed output lengths unless noted. This engine is for teaching and ablation — not for beating vLLM on a leaderboard.

### 1) Single request — no concurrency win from paging

One completion, 41 input tokens → 64 output tokens:

| Backend | TTFT | TPOT | tok/s |
|---------|-----:|-----:|------:|
| HuggingFace `generate` | 26 ms | 25 ms | 40 |
| tiny Stage 2 (no KV) | 22 ms | 22 ms | 46 |
| tiny paged + continuous | 24 ms | 27 ms | 37 |

**Takeaway:** at batch size 1, everyone is in the same ballpark. Paged attention here **gathers** K/V then runs SDPA — a memory layout, not a fused CUDA kernel — so it does not buy single-request latency.

### 2) Multi-request — continuous batching is the leap

Same host, concurrency ramp to **16**, `max_tokens=64`. Plain HF FIFO server vs tiny feature sets:

| Config | QPS @16 | out tok/s | TTFT p50 |
|--------|--------:|----------:|---------:|
| Plain HF (FIFO `generate`) | 0.58 | 38 | 26 s |
| tiny v0 / kv / paged (no batching) | ~0.55–0.68 | ~35–43 | 22–27 s |
| **tiny paged + continuous** | **6.81** | **436** | **136 ms** |
| **+ chunked prefill + prefix** | **7.16** | **458** | **79 ms** |
| all + int8 (no speculative) | 5.03 | 322 | 109 ms |

**Takeaways:**

- KV alone does **not** raise multi-request QPS — still one-at-a-time without batching; TTFT just queues.  
- **Continuous batching ≈ 12× QPS / tok/s** vs plain HF or tiny without batching.  
- Chunk + prefix help TTFT further on shared-prefix-ish load.  
- Our **int8** path works but **slows** TPOT (~49 ms vs ~34 ms bf16) because QuantLinear dequantizes every matmul — memory win, not a fused-kernel win.

Naive concurrent HF (`max_parallel=4`) **hurt** throughput (TPOT ~26 → ~163 ms) — contention, not capacity.

### 3) Offline KV — recompute vs cache

`bench_offline` (e.g. prompt 2048 → 128 out):

| Mode | TPOT | Recompute ratio (model tokens / outputs) |
|------|-----:|-----------------------------------------:|
| no KV | ~142 ms | ~2000× |
| contiguous / paged | ~31–32 ms | ~17× |

Without a cache, step cost grows with context. With KV, decode TPOT stays roughly flat; remaining gap to vLLM (~8 ms TPOT on the same model) is Python + gather/SDPA, not “missing the idea of a cache.”

### 4) Paged capacity under a tight KV budget

Forced ~**1.0 GiB** KV pool, **256** requests (Stage 4 exhaustion):

| Allocator | req/s | out tok/s | running peak | KV waste |
|-----------|------:|----------:|-------------:|---------:|
| contiguous / max_model_len | 0.32 | 105 | 4 | ~94% |
| contiguous / max_tokens | 0.80 | 253 | 28 | ~59% |
| **paged** | **1.81** | **569** | **122** | **~1.4%** |

Paged packs many more variable-length sequences; cost when full is **preemption** (160 in that run), not silent OOM.

### Reproduce

```bash
python benchmarks/bench_offline.py --kv-caches none contiguous paged
python benchmarks/stage4_paged_capacity.py
python benchmarks/stage5_scheduling.py
python benchmarks/stage6_prefix_caching.py
python benchmarks/stage8_quantization.py

tiny-engine serve -f paged,batching,chunked,prefix --port 8001
bash benchmarks/run_online.sh --concurrency 1 4 16 64
```

---

## Design choices (short)

| Choice | Why |
|--------|-----|
| Keep HF modules | Fair weight/tokenizer parity; focus on the engine |
| Explicit `step()` | Same mental model as vLLM EngineCore |
| Feature flags | Ablate one idea at a time |
| OpenAI API | Reuse external smoke/bench clients |
| Gather + SDPA (not fused paged attn) | Correct + readable; speed gap is intentional teaching fuel |
| Weight-only quant via dequant+`F.linear` | Memory win; shows why fused int4 kernels matter |

---

## Requirements

- Python ≥ 3.10  
- `torch`, `transformers`, `fastapi`, `uvicorn`, `pydantic` (see `pyproject.toml`)  
- Dev extras: `pytest`, `httpx`, `openai`  

Optional GPU + CUDA for graphs and serious throughput tests.

---
