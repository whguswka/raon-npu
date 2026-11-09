# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
#
"""RDA (Raon Delta Attention) fp32 state update microbenchmark.

Implements one-layer token-serial RDA state update microkernel on AMD XDNA 2 (AIE2P).
Evaluates worker scaling (N in {1, 4, 8}), state ObjectFifo streaming (depth 2),
accuracy against numpy reference (NRMSE <= 1e-4), and NPU execution time via run_iters.
"""

import argparse
from pathlib import Path
import sys
import time

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.config import cxx_header_path
import numpy as np

# Geometric constants for Raon-1-mini RDA
V_DIM = 128
K_DIM = 128
TILE_V = 32
TILES_PER_HEAD = V_DIM // TILE_V  # 4 tiles per head (4 * 32 = 128 rows)

# Head bundle geometry:
# Packed contiguous layout: q[128] + k[128] + a[128] + v[128] + beta[1] + pad[127] = 640 floats (2,560 B)
# Conforms to AIE2P Core Tile DMA channel limit (maximum 2 S2MM input channels: 1 state tile + 1 head bundle).
# Offsets:
#   q:    0..127   (128 floats = 512 B, 64-byte aligned)
#   k:    128..255 (128 floats = 512 B, 64-byte aligned)
#   a:    256..383 (128 floats = 512 B, 64-byte aligned)
#   v:    384..511 (128 floats = 512 B, 64-byte aligned; per-tile v32 at 384 + tile_idx * 32)
#   beta: 512      (1 float = 4 B, 64-byte aligned)
#   pad:  513..639 (127 floats = 508 B, 0-filled for 64-byte vector alignment)
HEAD_BUNDLE_SIZE = 640

_KERNEL_PATH = Path(__file__).resolve().parent.parent.parent / "kernels" / "rda_state.cc"


def pack_head_bundle(q, k, a, v, beta):
    """Pack per-head vectors into contiguous head bundle [H, 640] fp32.

    Offsets within each 640-float head slice:
      0..127   : q [128]
      128..255 : k [128]
      256..383 : a [128] = exp(log_alpha)
      384..511 : v [128] (slice for tile t in 0..3 is at 384 + t * 32)
      512      : beta [1]
      513..639 : zero padding [127] (for 64-byte vector alignment)
    """
    H = q.shape[0]
    bundle = np.zeros((H, HEAD_BUNDLE_SIZE), dtype=np.float32)
    bundle[:, 0:128] = q
    bundle[:, 128:256] = k
    bundle[:, 256:384] = a
    bundle[:, 384:512] = v
    bundle[:, 512] = beta
    return bundle


@iron.jit
def rda_step(
    s_in: In,
    s_out: Out,
    head_in: In,
    out: Out,
    *,
    num_heads: CompileTime[int] = 40,
    num_workers: CompileTime[int] = 4,
    use_memtile: CompileTime[bool] = True,
    trace_size: CompileTime[int] = 0,
):
    """Run one decode token step of recurrent RDA for all heads across N workers."""
    H = num_heads
    N = num_workers
    heads_per_worker = H // N

    # --------------------------------------------------------------------------
    # Type descriptors
    # --------------------------------------------------------------------------
    # L3 external memory types
    s_l3_ty = np.ndarray[(H, V_DIM, K_DIM), np.dtype[np.float32]]
    head_in_l3_ty = np.ndarray[(H, HEAD_BUNDLE_SIZE), np.dtype[np.float32]]
    out_l3_ty = np.ndarray[(H, V_DIM), np.dtype[np.float32]]

    # L1 core tile types
    # State tile: 32 rows x 128 keys fp32 = 4096 floats = 16 KiB
    tile_s_ty = np.ndarray[(TILE_V, K_DIM), np.dtype[np.float32]]
    # Packed head bundle: 640 floats = 2,560 B
    tile_head_bundle_ty = np.ndarray[(HEAD_BUNDLE_SIZE,), np.dtype[np.float32]]
    # Output slice: 32 floats = 128 B
    tile_out_ty = np.ndarray[(TILE_V,), np.dtype[np.float32]]

    # --------------------------------------------------------------------------
    # External C++ kernel
    # --------------------------------------------------------------------------
    rda_kernel = ExternalFunction(
        "rda_update_tile",
        source_file=str(_KERNEL_PATH),
        arg_types=[
            tile_s_ty,
            tile_s_ty,
            tile_head_bundle_ty,
            np.int32,
            tile_out_ty,
        ],
        include_dirs=[cxx_header_path()],
    )

    # --------------------------------------------------------------------------
    # Core worker task
    # --------------------------------------------------------------------------
    def core_fn(of_s_in, of_s_out, of_head_in, of_out, kernel):
        for _ in range_(heads_per_worker):
            # Acquire per-head parameters once across the 4 V-tiles
            elem_head_in = of_head_in.acquire(1)

            for tile_idx in range_(TILES_PER_HEAD):
                elem_s_in = of_s_in.acquire(1)
                elem_s_out = of_s_out.acquire(1)
                elem_out = of_out.acquire(1)

                kernel(
                    elem_s_in,
                    elem_s_out,
                    elem_head_in,
                    tile_idx,
                    elem_out,
                )

                of_s_in.release(1)
                of_s_out.release(1)
                of_out.release(1)

            of_head_in.release(1)

    # --------------------------------------------------------------------------
    # ObjectFifos and Workers creation
    # --------------------------------------------------------------------------
    workers = []
    s_in_prods = []
    s_out_conses = []
    head_in_prods = []
    out_conses = []

    for w in range(N):
        col = w
        core_tile = Tile(col, 2)
        shim_tile = Tile(col, 0)

        if use_memtile:
            # Memory tile intermediate staging:
            # Shim -> MemTile (depth 2, 32 KiB) -> Core L1 (depth 1, 16 KiB)
            of_s_in_l3l2 = ObjectFifo(tile_s_ty, depth=2, name=f"s_in_l3l2_{w}")
            s_in_prods.append(of_s_in_l3l2.prod(tile=shim_tile))
            of_s_in_core = of_s_in_l3l2.cons().forward(depth=1, name=f"s_in_l2l1_{w}")

            # Core L1 (depth 1, 16 KiB) -> MemTile -> Shim (depth 2, 32 KiB)
            of_s_out_core = ObjectFifo(tile_s_ty, depth=1, name=f"s_out_l1l2_{w}")
            of_s_out_l2l3 = of_s_out_core.cons().forward(depth=2, name=f"s_out_l2l3_{w}")
            s_out_conses.append(of_s_out_l2l3.cons(tile=shim_tile))

            core_s_in_handle = of_s_in_core.cons()
            core_s_out_handle = of_s_out_core.prod()
        else:
            raise ValueError(
                "Direct Shim-to-Core routing (--use-memtile 0) is disabled because it exceeds "
                "the 64 KiB L1 budget (81.25 KiB needed with depth 2 state fifos and 12 KiB stack). "
                "Always route state via Memory Tile (use_memtile=True)."
            )

        # Head bundle ObjectFifo (Shim -> Core, depth 2, 5 KiB)
        # Consumes exactly 1 input DMA channel on Core tile (leaving 1 for s_in)
        of_head_in = ObjectFifo(tile_head_bundle_ty, depth=2, name=f"head_in_{w}")
        head_in_prods.append(of_head_in.prod(tile=shim_tile))

        # Output slice ObjectFifo (Core -> Shim, depth 2, 256 B)
        # Consumes exactly 1 output DMA channel on Core tile (leaving 1 for s_out)
        of_out = ObjectFifo(tile_out_ty, depth=2, name=f"out_{w}")
        out_conses.append(of_out.cons(tile=shim_tile))

        worker = Worker(
            core_fn,
            [
                core_s_in_handle,
                core_s_out_handle,
                of_head_in.cons(),
                of_out.prod(),
                rda_kernel,
            ],
            tile=core_tile,
            stack_size=12288,
        )
        workers.append(worker)

    # --------------------------------------------------------------------------
    # Tensor Access Patterns: partition across N workers along heads axis
    # --------------------------------------------------------------------------
    s_taps = TensorAccessPattern.full((H, V_DIM, K_DIM)).partition(N)
    head_in_taps = TensorAccessPattern.full((H, HEAD_BUNDLE_SIZE)).partition(N)
    out_taps = TensorAccessPattern.full((H, V_DIM)).partition(N)

    # --------------------------------------------------------------------------
    # Runtime sequence
    # --------------------------------------------------------------------------
    def sequence(
        s_in_arg,
        s_out_arg,
        head_in_arg,
        out_arg,
        s_in_prods_seq,
        s_out_conses_seq,
        head_in_prods_seq,
        out_conses_seq,
    ):
        tg = TaskGroup()
        for w in range(N):
            s_in_prods_seq[w].fill(s_in_arg, s_taps[w], group=tg)
            head_in_prods_seq[w].fill(head_in_arg, head_in_taps[w], group=tg)

            s_out_conses_seq[w].drain(s_out_arg, s_taps[w], wait=True, group=tg)
            out_conses_seq[w].drain(out_arg, out_taps[w], wait=True, group=tg)

        tg.finish()

    rt = Runtime(
        sequence,
        [
            s_l3_ty,
            s_l3_ty,
            head_in_l3_ty,
            out_l3_ty,
            s_in_prods,
            s_out_conses,
            head_in_prods,
            out_conses,
        ],
    )

    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()


def numpy_recurrent_rda_step(S, q, k, a, v, beta):
    """Numpy reference calculation for one RDA decode step.

    Matches torch_recurrent_rda (modeling_raon1_mini.py:241-274).
    S: [H, V, K], q/k/a: [H, K], v: [H, V], beta: [H]
    """
    # 1. Decay
    S_decay = S * a[:, None, :]  # [H, V, K]
    # 2. Predicted
    pred = np.einsum("hvk,hk->hv", S_decay, k)  # [H, V]
    # 3. Error
    err = v - pred  # [H, V]
    # 4. Rank-1 state update
    scale = (beta[:, None] * err)[:, :, None]  # [H, V, 1]
    S_new = S_decay + scale * k[:, None, :]  # [H, V, K]
    # 5. Output projection
    out = np.einsum("hvk,hk->hv", S_new, q)  # [H, V]
    return S_new, out


def main():
    parser = argparse.ArgumentParser(description="Raon-1-mini RDA fp32 State Update Microbenchmark")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        choices=[1, 4, 8],
        help="Number of AIE worker cores (1, 4, 8; default: 4)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="Number of sequential decode tokens (default: 64)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=40,
        help="Total number of RDA heads (default: 40)",
    )
    parser.add_argument(
        "--use-memtile",
        type=int,
        default=1,
        choices=[0, 1],
        help="Route state ObjectFifo via Memory Tile (1: yes (default); 0: direct route disabled due to L1 overflow)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for JIT and timing (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.use_memtile == 0:
        sys.exit(
            "ERROR: Direct Shim-to-Core routing (--use-memtile 0) is disabled.\n"
            "Reason: Direct routing requires depth 2 state FIFOs, taking 81.25 KiB L1 (> 64 KiB)\n"
            "with stack_size=12 KiB. Even with output depth 1, it requires 65.25 KiB (> 64 KiB).\n"
            "Always use MemTile staging (--use-memtile 1)."
        )

    H = args.heads
    T = args.tokens
    N = args.workers
    V = V_DIM
    K = K_DIM

    if H % N != 0:
        sys.exit(f"Number of heads ({H}) must be divisible by workers ({N})")

    print("=" * 70)
    print("Raon-1-mini RDA fp32 State Update Microkernel Benchmark")
    print(f"Configuration: Workers={N}, Heads={H}, Tokens={T}, MemTile={bool(args.use_memtile)}")
    print(f"State Size: {H}x{V}x{K} fp32 ({H * V * K * 4 / (1024 * 1024):.2f} MiB per state buffer)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Synthetic realistic test data generation
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    # Unit-length query and key vectors
    q_all = rng.standard_normal((T, H, K)).astype(np.float32)
    q_all /= np.linalg.norm(q_all, axis=-1, keepdims=True) + 1e-12

    k_all = rng.standard_normal((T, H, K)).astype(np.float32)
    k_all /= np.linalg.norm(k_all, axis=-1, keepdims=True) + 1e-12

    # log_alpha in [-5, 0] -> a = exp(log_alpha) in [e^-5, 1]
    log_alpha_all = rng.uniform(-5.0, 0.0, size=(T, H, K)).astype(np.float32)
    a_all = np.exp(log_alpha_all)

    # beta in (0, 1)
    beta_all = rng.uniform(0.01, 0.99, size=(T, H)).astype(np.float32)

    # v ~ N(0, 1)
    v_all = rng.standard_normal((T, H, V)).astype(np.float32)

    # Initial state S: small random values
    S_init = rng.standard_normal((H, V, K)).astype(np.float32) * 0.01

    # --------------------------------------------------------------------------
    # Golden reference computation (Numpy fp32)
    # --------------------------------------------------------------------------
    print("\n[1/3] Computing CPU Golden Reference (numpy fp32)...")
    ref_outs = np.zeros((T, H, V), dtype=np.float32)
    ref_states_first3 = []
    S_ref = S_init.copy()
    for t in range(T):
        S_ref, ref_outs[t] = numpy_recurrent_rda_step(
            S_ref, q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t]
        )
        if t < 3:
            ref_states_first3.append(S_ref.copy())
    print("Golden reference computed successfully.")

    # --------------------------------------------------------------------------
    # NPU buffer allocation and sequential execution
    # --------------------------------------------------------------------------
    print(f"\n[2/3] Allocating NPU device memory and warming up JIT ({args.warmup} runs)...")
    s_in_npu = iron.tensor(S_init, device="npu")
    s_out_npu = iron.zeros_like(s_in_npu)

    # Conforms to AIE2P Core Tile DMA channel budget:
    # 2 S2MM input channels: s_in (16 KiB state tile) + head_in (2.5 KiB packed bundle)
    # 2 MM2S output channels: s_out (16 KiB updated state) + out (128 B output slice)
    head_in_npu = iron.zeros((H, HEAD_BUNDLE_SIZE), dtype=np.float32, device="npu")
    out_npu = iron.zeros((H, V), dtype=np.float32, device="npu")

    # Warmup invocations to compile and prime JIT cache
    head_in_npu[:] = pack_head_bundle(q_all[0], k_all[0], a_all[0], v_all[0], beta_all[0])

    for _ in range(args.warmup):
        rda_step(
            s_in_npu,
            s_out_npu,
            head_in_npu,
            out_npu,
            num_heads=H,
            num_workers=N,
            use_memtile=bool(args.use_memtile),
        )

    print(f"\n[3/3] Executing {T} sequential tokens on NPU (state chained in DDR)...")
    s_in_npu[:] = S_init
    npu_outs = np.zeros((T, H, V), dtype=np.float32)
    token_latencies_us = []
    early_step_results = []

    for t in range(T):
        head_in_npu[:] = pack_head_bundle(
            q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t]
        )

        t_start = time.perf_counter()
        rda_step(
            s_in_npu,
            s_out_npu,
            head_in_npu,
            out_npu,
            num_heads=H,
            num_workers=N,
            use_memtile=bool(args.use_memtile),
        )
        t_end = time.perf_counter()
        token_latencies_us.append((t_end - t_start) * 1e6)

        npu_outs[t] = out_npu.numpy()

        # Step-check: compare state after each of the first 3 tokens against golden reference
        if t < min(T, 3):
            s_step_npu = s_out_npu.numpy()
            diff_step = s_step_npu - ref_states_first3[t]
            rmse_step = float(np.sqrt(np.mean(diff_step**2)))
            range_step = float(np.max(ref_states_first3[t]) - np.min(ref_states_first3[t]) + 1e-12)
            nrmse_step = rmse_step / range_step
            rel_l2_step = float(np.linalg.norm(diff_step) / (np.linalg.norm(ref_states_first3[t]) + 1e-12))
            status_str = "PASS" if nrmse_step <= 1e-4 else "FAIL"
            print(f"  [Step-Check Token {t+1}/{min(T, 3)}] State NRMSE: {nrmse_step:.6e}, Rel L2: {rel_l2_step:.6e} -> {status_str}")
            early_step_results.append((t + 1, nrmse_step, rel_l2_step))
            if nrmse_step > 1e-4:
                sys.exit(f"VERDICT: FAIL (Early state verification failed at token {t+1}: NRMSE {nrmse_step:.6e} > 1e-4)")

        # Chain state forward in DDR: output state becomes next input state
        s_in_npu[:] = s_out_npu

    # Capture final state immediately after sequential loop before any further benchmark calls
    s_final_npu = s_out_npu.numpy().copy()

    # --------------------------------------------------------------------------
    # Accuracy evaluation (vs FP32 Golden Reference)
    # --------------------------------------------------------------------------
    out_diff = npu_outs - ref_outs
    out_max_err = float(np.max(np.abs(out_diff)))
    out_rel_l2 = float(np.linalg.norm(out_diff) / (np.linalg.norm(ref_outs) + 1e-12))

    s_diff = s_final_npu - S_ref
    s_max_err = float(np.max(np.abs(s_diff)))
    s_rel_l2 = float(np.linalg.norm(s_diff) / (np.linalg.norm(S_ref) + 1e-12))
    s_rmse = float(np.sqrt(np.mean(s_diff**2)))
    s_range = float(np.max(S_ref) - np.min(S_ref) + 1e-12)
    s_nrmse = s_rmse / s_range

    # --------------------------------------------------------------------------
    # Benchmark via run_iters (using dedicated scratch buffers to avoid state contamination)
    # --------------------------------------------------------------------------
    print("\nRunning benchmark timing via aie.utils.benchmark.run_iters...")
    s_in_bench = iron.tensor(S_init, device="npu")
    s_out_bench = iron.zeros_like(s_in_bench)
    bench = run_iters(
        rda_step,
        s_in_bench,
        s_out_bench,
        head_in_npu,
        out_npu,
        num_heads=H,
        num_workers=N,
        use_memtile=bool(args.use_memtile),
        warmup=2,
        iters=min(T, 20),
    )
    print_benchmark(bench)

    p50_us = float(np.percentile(token_latencies_us, 50))
    p99_us = float(np.percentile(token_latencies_us, 99))
    avg_us = float(np.mean(token_latencies_us))

    print("\n" + "=" * 70)
    print("RDA State Update Microbenchmark Evaluation Report")
    print("=" * 70)
    print(f"Tokens Processed                     : {T}")
    print(f"Heads / Workers                      : {H} / {N} ({H // N} heads/worker)")
    print(f"Memory Tile Staging                  : {bool(args.use_memtile)}")
    print("-" * 70)
    print("EARLY STEP VERIFICATION (Tokens 1..3 State Check):")
    for step_num, step_nrmse, step_rel_l2 in early_step_results:
        verdict = "PASS" if step_nrmse <= 1e-4 else "FAIL"
        print(f"  Token {step_num} State NRMSE : {step_nrmse:.6e}, Relative L2 : {step_rel_l2:.6e} ({verdict} <= 1e-4)")
    print("-" * 70)
    print("ACCURACY COMPARISON (vs FP32 Golden Reference):")
    print(f"  Per-token Output Max Absolute Error: {out_max_err:.6e}")
    print(f"  Per-token Output Relative L2 Error : {out_rel_l2:.6e}")
    print(f"  Final State Max Absolute Error     : {s_max_err:.6e}")
    print(f"  Final State Relative L2 Error      : {s_rel_l2:.6e}")
    print(f"  Final State RMSE                   : {s_rmse:.6e}")
    print(f"  Final State NRMSE                  : {s_nrmse:.6e}  (Threshold <= 1.0e-4)")
    print("-" * 70)
    print("TIMING & THROUGHPUT:")
    print(f"  Host Python Call Latency p50       : {p50_us:.1f} us ({p50_us / 1000.0:.3f} ms)")
    print(f"  Host Python Call Latency p99       : {p99_us:.1f} us ({p99_us / 1000.0:.3f} ms)")
    print(f"  Host Python Call Latency mean      : {avg_us:.1f} us ({avg_us / 1000.0:.3f} ms)")

    # Data movement volume per token:
    # State Read: H * V * K * 4 = 2,621,440 B (2.5 MiB)
    # State Write: H * V * K * 4 = 2,621,440 B (2.5 MiB)
    # Head bundle Read: H * HEAD_BUNDLE_SIZE * 4 = 40 * 640 * 4 = 102,400 B (100 KiB)
    # Output Write: H * V * 4 = 40 * 128 * 4 = 20,480 B (20 KiB)
    # Total traffic per token: 5,365,760 bytes (~5.12 MiB)
    total_bytes_per_token = 2 * (H * V * K * 4) + (H * HEAD_BUNDLE_SIZE * 4) + (H * V * 4)

    if bench.npu is not None:
        npu_avg_us = bench.npu.avg_us
        bw_gbps = (total_bytes_per_token / (npu_avg_us * 1e-6)) / 1e9
        print(f"  NPU Kernel Time (avg/min/max)      : {bench.npu.avg_us:.1f} / {bench.npu.min_us:.1f} / {bench.npu.max_us:.1f} us")
        print(f"  NPU Effective DDR Bandwidth        : {bw_gbps:.2f} GB/s")
    print("=" * 70)

    # Threshold verdict
    state_pass = s_nrmse <= 1e-4
    out_pass = out_rel_l2 <= 1e-5
    if state_pass and out_pass:
        print(
            f"VERDICT: PASS (State NRMSE {s_nrmse:.6e} <= 1e-4 and "
            f"Output Relative L2 {out_rel_l2:.6e} <= 1e-5 satisfied)"
        )
    else:
        failures = []
        if not state_pass:
            failures.append(f"State NRMSE {s_nrmse:.6e} exceeded 1e-4 threshold")
        if not out_pass:
            failures.append(f"Output Relative L2 {out_rel_l2:.6e} exceeded 1e-5 threshold")
        print(f"VERDICT: FAIL ({', '.join(failures)})")
        sys.exit(1)


if __name__ == "__main__":
    main()
