# Docs — one page per stage

Each file matches what the engine does when that feature is on. Read in order.

| Stage | Doc | Flag | Code |
|------:|-----|------|------|
| 2 | [stage2-naive.md](stage2-naive.md) | defaults / `--preset v0` | `engine.py`, `model/runner.py`, `scheduler/fifo.py` |
| 3 | [stage3-kv-cache.md](stage3-kv-cache.md) | `-f kv` | `cache/pool.py`, `contiguous.py`, `cached_runner.py` |
| 4 | [stage4-paged-kv.md](stage4-paged-kv.md) | `-f paged` | `cache/paged.py` |
| 5 | [stage5-scheduling.md](stage5-scheduling.md) | `-f batching` / `static` / `chunked` | `scheduler/` |
| 6 | [stage6-prefix-caching.md](stage6-prefix-caching.md) | `-f prefix` | `cache/prefix.py` + paged |
| 8 | [stage8-quantization.md](stage8-quantization.md) | `-f int8` / `int4` / `fp8` | `quantization/` |
| — | [cuda-graphs.md](cuda-graphs.md) | `--enable-cuda-graph` | `model/cuda_graph.py` |

Serving (all stages): `serving/api_server.py` + `async_engine.py` on port **8001**.
