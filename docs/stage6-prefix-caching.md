# Stage 6 — Prefix caching

**Flag:** `-f prefix` (requires `paged`)  
**Preset:** `--preset prefix`  
**Benchmark:** `python benchmarks/stage6_prefix_caching.py`

## Problem

Many requests share the same system prompt. Without reuse, every request prefills those tokens again — identical K/V recomputed, TTFT wasted.

## What happens

Only works with **paged** blocks.

```text
prompt → split into blocks of 16
       → chained hashes:
            h0 = hash(tokens[0:16])
            h1 = hash(h0, tokens[16:32])
            …
on admit:
  lookup_prefix → list of physical block ids that hit
  num_computed_tokens = hits × 16
  allocate remaining blocks; skip prefill for hits
after steps:
  commit full blocks into PrefixCacheIndex (hash → block id)
```

Refcounts: shared blocks stay alive while any sequence uses them. Unused hashed blocks can sit in an LRU (`evictable`) until space is needed.

**Footgun guard:** if a lookup would cover the *entire* prompt, the last block is dropped so at least one token still runs (need logits to sample).

## Code

| File | Role |
|------|------|
| `cache/prefix.py` | `compute_block_hashes`, `PrefixCacheIndex` |
| `cache/paged.py` | `lookup_prefix`, `commit`, refcount / LRU |
| `scheduler/base.py` | `_try_admit` calls lookup |

## Run

```bash
tiny-engine chat -f prefix
# set /system with a long shared prompt; watch /stats for hits
python benchmarks/stage6_prefix_caching.py
```

## Expect

Lower TTFT when many clients share a system prompt. Random unique prompts defeat the cache (by design in fair benches).
