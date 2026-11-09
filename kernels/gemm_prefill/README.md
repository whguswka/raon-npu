# GEMM prefill stage 2a

This package contains the exact-BF16, 9-bit-weight prefill projection body used by
`NpuBackend.linear_batch`. It is an `s2a` artifact family and does not import
the retired `gemv_batch` experiment.

## Device contract

- One xclbin uses columns 0..6 for `rda_in`, `f_b`, `o_proj`, `mla_in`, `q_b`,
  `k_b_v_b`, `g`, and `dense_down`, and column 7 for RDA. Every GEMM column owns
  four compute rows, each producing 32 outputs.
- The core program is shape-independent. Each shape has one command stream that
  writes `(K/256, rounds_for_column, diagnostic_mode)` to every core RTP.
- Physical T is always 24. A final logical chunk of size `t` is zero-padded and
  computes `24-t` unused rows; no T-specific command stream or xclbin is created.
- X DMA order is `[K/256, 32 K8 groups, 24 rows, 8 bf16]`. W is streamed in
  128-output chunks and split to four 32-output cores. Y is scattered to `[24,M]`.
- The exact kernel reconstructs packed 9-bit weights as BF16 and uses AIE2P
  4x8x8 BF16 mmul with FP32 accumulation. Only the epilogue narrows to BF16 RNE.
- The 9-bit unpack converts two K values at once in one 32-lane conversion and multiply.

`PrefillGEMM` defaults to the final layout: columns 0..6 use GEMM and column 7
uses the existing RDA channel builder. GEMM and RDA command streams share the
same xclbin and the backend shares RDA state/input/output BOs with decode.

## Static budget

T=24 core use is 62,472 bytes including two W buffers, two X buffers, FP32 C,
two Y buffers, 12,288-byte stack reservation, RTP and a 1,024-byte margin. The
memory tile holds only the W split and the Y join (both depth 2): 86,016 bytes
for every shape at T=24; X is not staged there (broadcast FIFO from the shim to
the column's cores, DESIGN-prefill.md section 45). Modeled maxima are core/memtile/shim BD
6/24/9 and core/memtile locks 7/24. Actual ELF text, stack metadata, MLIR buffer
placement, BD and lock counts remain notebook link checks.
The AIE2P core program-memory limit is 16,384 bytes (0x4000), distinct from
the 65,536-byte core data-memory limit. The prototype gates each linked core's
`.text` against 16 KiB and derives L1 usage from addressed buffer end values.

## Notebook commands

From `C:\dev\raon\stage15` in the existing AIE environment:

```powershell
python -m bench.gemm_prefill.benchmark C:\dev\raon\weights-d --iterations 10 --switch-iterations 100 --output prefill-results\s2a-engine.json
python -m bench.gemm_prefill.rda_context C:\dev\raon\weights-d --output prefill-results\s2a-rda-context.json
python -m bench.gemm_prefill.prototype C:\dev\raon\weights-d --iterations 10000 --output prefill-results\s2a-prototype.json
python -m kernels.gemm_prefill.audit --elf C:\absolute\build\core_0_2.elf --elf C:\absolute\build\core_0_3.elf
```

The engine benchmark checks all nine shapes at every T=1..24 against row-wise
`linear`, alternates timing windows, and records context switches in one JSON.
The prototype builds both plates, includes decode ELF data, audits only each
xclbin's own `<name>.mlir.d` tree, and alternates one GEMM and one RDA call.
GEMM parity uses row-wise decode GEMV; RDA state/output parity uses the same RDA
kernel through the decode-unified backend. NumPy supplies max-absolute-error
and NRMSE metrics only.
