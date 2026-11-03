# Stage 8 — Weight-only quantization

**Flags:** `-f int8` · `-f int4` · `-f fp8`  
**Benchmark:** `python benchmarks/stage8_quantization.py`

## Problem

Weights dominate memory at small batches. Narrower weights free VRAM for KV. With fused kernels, they can also cut HBM traffic and TPOT — this teaching path shows the *memory* half and why naive dequant is not free speed.

## What happens

### Init only (before KV)

```text
load HF model (bf16 Linears)
quantize_model(model, method):
  for each decoder layer:
    replace q/k/v/o and gate/up/down with QuantLinear
  # embed, norms, lm_head stay bf16
then _init_kv_cache / scheduler as usual
```

~7 projections × N layers. No separate generate path: `attn.q_proj(h)` still runs; the module is now `QuantLinear`.

### Every matmul

```text
QuantLinear.forward(x):
  W_bf16 = dequant(qweight, scales)
  return F.linear(x, W_bf16)
```

| Method | Store |
|--------|--------|
| int8 | per-row scale, int8 weights |
| int4 | group scales, packed nibbles |
| fp8 | per-row scale, float8 |

Activations and KV stay bf16.

## Code

| File | Role |
|------|------|
| `quantization/quantize.py` | Walk layers, swap modules |
| `quantization/linear.py` | `from_linear`, `dequantize`, `forward` |
| `engine.__init__` | Calls `quantize_model` when configured |

## Run

```bash
tiny-engine generate -f int8 --temperature 0 "Hi"
tiny-engine generate -f int4 --temperature 0 "Hi"
python benchmarks/stage8_quantization.py
```

## Expect

Smaller weight footprint (more room for KV). TPOT here often **worse** than bf16 — full bf16 weight is rebuilt every call. Production AWQ/GPTQ fuse dequant into the GEMM; that gap is intentional.
