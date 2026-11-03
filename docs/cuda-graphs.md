# CUDA graphs (decode)

**Flag:** `--enable-cuda-graph`  
**Also:** `--cuda-graph-buckets`, `--cuda-graph-context-buckets`  
**Needs:** CUDA + KV (`CachedModelRunner`)

## Problem

After continuous batching, decode TPOT is still high vs vLLM partly because every step pays Python/CUDA **launch** overhead for many small kernels.

## What happens

### Init

```text
reserve pad slots at the tail of the KV pool (for fake batch rows)
capture one CUDA graph per (batch_bucket × context_bucket)
  defaults: batch 1,2,4,…,64 × context 512,2048,8192
```

### Each step

```text
CachedModelRunner.execute(items):
  if every item is decode (1 new token) AND batch fits a bucket:
      pad to bucket → CUDAGraph.replay()    # hit
  else:
      eager path                            # miss (prefill, mixed, oversized)
```

Pad rows write into scratch KV slots and are masked out. Metrics: `cuda_graph_hits_total` / `cuda_graph_misses_total` on `/metrics`.

Chunked prefill lowers hit rate until the batch is decode-only.

## Code

| File | Role |
|------|------|
| `model/cuda_graph.py` | `CudaGraphDecoder` capture / replay |
| `engine._init_cuda_graphs` | Pad slots + enable on runner |
| `model/cached_runner.py` | Hit/miss branch in `execute` |

## Run

```bash
tiny-engine serve -f paged,batching --enable-cuda-graph --port 8001
```

## Expect

Often helps **short** decode (lower launch tax). Can **hurt** if context padding is too fat. Does not close the gap to fused paged-attention kernels.
