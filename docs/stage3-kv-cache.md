# Stage 3 — Contiguous KV cache

**Flag:** `-f kv` / `--preset kv` (`kv_cache="contiguous"`)  
**Benchmark:** `python benchmarks/bench_offline.py --kv-caches none contiguous`

## Problem

Without a cache, step *i* re-runs prompt + *i* tokens through the model just to keep the last logits. Recomputing K/V that never change is pure waste.

## What happens

### Init (`engine._init_kv_cache`)

```text
KVCacheSpec from HF config
  bytes_per_token = 2 × layers × kv_heads × head_dim × dtype
  (~28 KiB for Qwen2.5-1.5B bf16 — GQA uses KV heads, not Q heads)
budget = leftover VRAM after a profile forward (or kv_cache_memory_gib)
num_slots = budget / bytes_per_token

kv   = ContiguousKVManager(num_slots)   # free_ranges = [(0, slots)]
pool = KVPool(k,v shape [layers, slots, kv_heads, head_dim])
runner = CachedModelRunner(model, kv, pool)
```

**Pool** = physical K/V tensors. **Manager** = which slot each `(seq, position)` uses.  
`slot(position) = start + position` for that request’s reserved range.

### Generate / step

```text
add_request → waiting (allocate happens on admit)
step → _step_cached:
  schedule → allocate contiguous range (reserve ≈ prompt + max_tokens)
  BatchItem = only NEW tokens (start_pos = num_computed_tokens)
  CachedModelRunner:
    slot_table → build_metadata
    Qwen2Forward → attention WRITES new K/V into pool, READS context
  sample → num_computed_tokens += n
```

- **Prefill:** run all prompt tokens once; write their K/V.  
- **Decode:** run **one** new token; read the rest from the pool.

## Code

| File | Role |
|------|------|
| `cache/pool.py` | `KVPool`, `KVCacheSpec`, `kv_budget_bytes` |
| `cache/contiguous.py` | Reserve range, `slot_table`, free/merge |
| `model/cached_runner.py` | Pack new tokens only |
| `model/forward.py` / `attention.py` | Layer loop; write/read pool |
| `engine.py` | `_init_kv_cache`, `_step_cached` |

## Run

```bash
tiny-engine generate -f kv --show-steps --temperature 0 "Explain KV"
```

## Expect

After prefill, step time stays roughly flat (vs climbing in Stage 2).  
Cost: reservation waste — you reserve `max_tokens` you may never use. That is Stage 4.
