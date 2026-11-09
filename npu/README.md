<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon NPU Execution Backend

Provides manifest-driven validation, pre-allocation of NPU weight tensors, bundled GEMV execution, recurrent RDA updates, MoE dispatch, and native IRON/AIE runtime orchestration for AMD XDNA 2.

## GEMV Engine Selection (`RAON_NPU_GEMV` & `run_golden.py --gemv`)

Raon NPU supports two GEMV execution engines: `unified` (default) and `legacy`.
The engine can be configured via the environment variable `RAON_NPU_GEMV` (`unified` or `legacy`) or via the CLI flag `run_golden.py --gemv {unified,legacy}`.

- **`unified` (default)**: Maps all layer bundles to unified shapes (`ALL_SHAPES`) using `pad_x` and shares a single merged configuration xclbin (`cols=7` GEMV + `rda_col=7` RDA) across GEMV, MoE, and RDA recurrent state update operators. This eliminates all hardware xclbin context switches during decode (0 switches per token).
- **`legacy`**: Retains per-shape individual GEMV operators, separate MoE templates (`gemv_{fmt}_k{k}_m{m}`, `moe_gate_up`, `moe_down`), and the standalone 4-column `rda_step` xclbin path (`_prepare_rda_kernel`), tracking context switches as before.

The unified backend also selects a runtime-only DMA sequence plan per shape.
`UNIFIED_SEQ_PLAN="per_call"` enables the allowlist in `UNIFIED_PER_CALL_SHAPES`:
- **per_call shapes (`UNIFIED_PER_CALL_SHAPES`)**: `f_b`, `q_b`, `k_b_v_b`, `dense_gu`, `moe_b1_6` (B1), and `moe_b2_9` (B2).
- **per_round retained shapes & rationale**:
  - `dense_down`: `K_blocks=47` (odd). The 47-BD core chain exceeds the core tile hardware BD pool limit (16 BDs) and cannot form a legal per_call core chain (`test_dense_down_is_not_a_legal_per_call_core_chain`).
  - `lm_head`: Large output dimension (`M=260,096`, `chunks=2,032`, 290-291 rounds/col on the 7-column engine; 254 is the 8-column value). Retained as per_round due to DDR bandwidth saturation (DDR-bound) and shim DMA iteration hardware limits (≤ 64) splitting overhead and core BD bounds.
  - `rda_in`, `mla_in`, `o_proj`, `g`: DDR-bound or small fixed-round operators retained as byte-compatible baseline per_round.
- **Rollback switch**: Setting `UNIFIED_SEQ_PLAN="per_round"` is the single engine-wide rollback switch for every unified GEMV, MoE, and merged-RDA operator, reverting all command streams to the byte-compatible baseline.
The per-call instruction binaries use an `_pc` suffix while both plans reuse the same configuration xclbin.

### Merged Unified Configuration (7-Column GEMV + Column 7 RDA)

Unified engine executes entirely within a single merged xclbin configuration without switching between GEMV and RDA:
- **Zero xclbin switches**: Average xclbin context switches per token is 0 (previously 36 switches per token due to separate `rda_step` xclbin).
- **Laptop gate verification results**:
  - Exact bit-for-bit identity across device bodies between configuration, RDA, and shape instruction streams.
  - Bit-exact output match across all dense GEMV shapes.
  - RDA state and output match standalone `rda_step` (4-column separate xclbin) with 100% uint32 bit-exact parity across 3 seeds x 200 tokens.
  - GEMV <-> RDA alternation additional latency p50 ≈ 0 ms (previous separate xclbins incurred approximately 2.3 ms per pair).
  - Legacy engine path and behavior remain unchanged.

### 2026-11-01 Verification Benchmark (per_call vs per_round A/B & Golden)

Official golden verification on laptop (steps 3-16 decode: baseline af44167 vs this tree with per_call, alternating A/B):
- **Overall Decode Latency**: 207.9 / 205.8 -> 192.3 / 191.0 ms/token
- **Pure NPU Compute Time**: 156 -> 140 ms
- **Golden Parity**: 100% identical logits and intermediate activation hashes across all 16 decode steps.
- **Artifact Invariance (`plan_artifact_check --compile-insts`)**: Compares all 13 per_round GEMV shapes + RDA using normalized MLIR, public `_make_extern` input fingerprints, and per-round `insts.bin` SHA-256. Each of the six allowlisted current-tree shapes must also have an error-free per_call record whose name differs from its per_round name and ends in `_pc`; `call_device_equal` compares their normalized device bodies and `call_name_ok` reports that naming contract. Per-round-only shapes print `-` in both per_call columns. `round_equal` combines normalized-MLIR and per-round-name equality, while `round_name_equal`, `raw_sha_diff`, `norm_sha_equal`, and `insts_sha_equal` expose the individual checks. Repository include-directory fingerprints cover only C/C++ source/header suffixes (`.h`, `.hh`, `.hpp`, `.hxx`, `.inc`, `.ipp`, `.c`, `.cc`, `.cpp`, `.cxx`, case-insensitive) and exclude hidden directories and `__pycache__`. Device/architecture MLIR lines are retained. The check does not compare compiled `.o` or xclbin bytes.

The A/B command requires a weight bundle file or a directory containing `<shape>.bin`/`<shape>.npy`:

```powershell
python -m bench.gemv_unified.plan_ab_bench --shape f_b --weights C:\dev\raon-weights --iterations 400 --timeout 300
```

#### Measured Shape A/B Benchmark (`lap/plan_ab.py`, 200 iterations, bit-exact, canary valid)

| Shape | K / M | p50 Latency (per_round -> per_call) | TXN Ops (per_round -> per_call) | DDR_PATCH (per_round -> per_call) | Notes |
|---|---|---:|---:|---:|---|
| `f_b` | 256 / 5,120 | 320 -> 231 µs (-27.8%) | 1,198 -> 478 (-60.1%) | - | K_blocks=1 (odd) |
| `q_b` | 768 / 7,680 | 541 -> 344 µs (-36.4%) | 3,168 -> 620 (-80.4%) | - | K_blocks=3 (odd) |
| `k_b_v_b` | 512 / 5,120 | 346 -> 241 µs (-30.3%) | 1,488 -> 504 (-66.1%) | - | K_blocks=2 (even) |
| `moe_b2_9` (B2) | 1,280 / 17,920 | 1,133 -> 739 µs (-34.8%) | 7,028 -> 672 (-90.4%) | 287 -> 21 (-92.7%) | 7-slot down |
| `moe_b1_6` (B1) | 2,560 / 13,056 | 782 -> 681 µs (-12.9%) | 3,534 -> 453 (-87.2%) | 211 -> 19 (-91.0%) | 6-expert gate/up |
| `dense_gu` | 2,560 / 24,064 | 1,051 -> 937 µs (-10.8%) | 6,372 -> 504 (-92.1%) | - | K_blocks=10 (even), 188 chunks |

### 2026-10-31 Verification Benchmark (25 layers, 15 steps)

Official golden verification on laptop (steps 3-16 decode: 245.7 -> 198.9 ms/token versus separate-RDA engine):

| Engine | Latency | Xclbin Switches / Token | Golden Parity |
|---|---|---|---|
| `unified` (merged c7+rda7) | 200.7 / 198.7 ms/token | 0 | Identical |
| `legacy` | 583.1 ms/token | 182 | Identical |

#### Previous Benchmark (2026-10-30, Separate RDA Engine)

Prior to merging RDA into column 7 (separate 4-column RDA engine with xclbin switches):

| Engine | Latency | Xclbin Switches / Token | Golden Parity |
|---|---|---|---|
| `unified` (separate RDA) | 406.5 ms/token | 36 | Identical |
| `legacy` | 874.1 ms/token | 182 | Identical |
