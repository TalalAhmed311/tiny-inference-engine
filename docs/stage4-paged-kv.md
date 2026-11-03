# Stage 4 — Paged KV

**Flag:** `-f paged` / `--preset paged`  
**Benchmark:** `python benchmarks/stage4_paged_capacity.py`

## Problem

Contiguous KV reserves a long range per request. Variable lengths waste memory and fragment the free list — total free space can be enough while the largest hole is too small.

## What happens

Same pool and same `_step_cached` path. Only the **manager** changes.

```text
Pool: still k/v[layer, slot, ...]
Manager: PagedKVManager
  pool cut into blocks (default block_size = 16 tokens)
  each sequence has a block table: list of physical block ids

logical position p
  → block_table[p // 16], offset p % 16
  → physical slot
```

On admit/allocate: take free blocks as needed (grow during decode). No huge upfront reservation.  
`slot_table` still feeds attention; `slots_are_contiguous=False` so reads **gather** by index (correct, slower than fused paged kernels).

When the pool is full, the scheduler can **preempt**: free a request’s blocks, reset `num_computed_tokens`, requeue (recompute later).

## Code

| File | Role |
|------|------|
| `cache/paged.py` | Blocks, free list, allocate/free, `slot_table` |
| `cache/__init__.py` | `make_kv_manager("paged", …)` |
| `model/attention.py` | Same write/read; gather path |

## Run

```bash
tiny-engine generate -f paged --temperature 0 "Hi"
python benchmarks/stage4_paged_capacity.py
```

## Expect

Under a fixed small KV budget, many more concurrent sequences than contiguous, with low waste (~1% vs tens of %). Next problem: one request still hogging the GPU — Stage 5.
