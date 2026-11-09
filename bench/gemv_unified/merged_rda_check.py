# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Laptop gate check for merged GEMV (cols=7) + RDA (col 7) in a single PDI.

Stages executed in the same process:
  (a) Bit-exact verification of registered GEMV shapes on the shared xclbin.
  (b) Bit-exact verification of UnifiedRDA (shared xclbin) vs standalone rda_step (separate xclbin)
      across 3 seeds x 200 tokens.
  (c) Alternation latency: back-to-back GEMV + RDA dispatches comparing shared PDI
      against separate xclbins (overhead p50 / p95).

Outputs 'RESULT: PASS' or 'RESULT: FAIL <reason>' on the final stdout line.
Exceptions and mismatches in required checks (all registered dense shapes,
dense_gu, MoE B1, and separate xclbin comparison) are always recorded as FAIL;
MoE B2 has no functional check here and stays COMPILE_ONLY (its functional
verdict belongs to the engine golden), never counted as a PASS.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IRON_DIR = os.environ.get("RAON_IRON_DIR")
if IRON_DIR:
    iron_path = Path(IRON_DIR).expanduser()
    if str(iron_path) not in sys.path:
        sys.path.insert(0, str(iron_path))

from kernels.gemv_unified import (
    ALL_SHAPES,
    FORMAT_BFP16,
    M_PER_CHUNK,
    MAX_K,
    UnifiedGEMV,
    UnifiedMoEDown9,
    UnifiedMoEGateUp,
    UnifiedRDA,
    get_shape,
    pad_x,
    reset_xclbin_signature_cache,
)
from bench.gemv_unified.unified_bench import (
    SENTINEL,
    compare_bf16_bits,
    make_shape_test_case,
)
try:
    from bench.rda_state.rda_state import (
        K_DIM,
        V_DIM,
        pack_head_bundle,
        rda_step,
    )
except (ImportError, ModuleNotFoundError):
    from bench.rda_state.rda_state_1col import (
        K_DIM,
        V_DIM,
        pack_head_bundle,
    )
    rda_step = None  # type: ignore[assignment]


def configure_streams() -> None:
    """Ensure stdout and stderr do not raise UnicodeEncodeError on cp949/Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except Exception:
                pass


configure_streams()


def log(msg: str) -> None:
    print(msg, flush=True)


# Status recorded for MoE B2: only its compilation on the shared xclbin is
# verified here; its functional verdict comes from the engine golden.
COMPILE_ONLY_STATUS = "COMPILE_ONLY (engine golden required)"


def judge_results(results: dict[str, str]) -> dict[str, Any]:
    """Judge the stage-(a) per-key result table into counts, failures, verdict.

    Buckets every entry into exactly one class:
      * ``PASS`` -- a required check (all registered dense shapes, dense_gu,
        MoE B1) passed functionally (or compiled, for B1's template entry).
      * ``FAIL`` -- a required check errored or mismatched; any single FAIL
        makes the verdict (and therefore the final ``RESULT`` line) FAIL.
      * ``COMPILE_ONLY`` -- B2 compiled on the shared xclbin but its functional
        verdict lives in the engine golden; never counted as a functional PASS.
      * ``skipped`` -- remaining non-required skips.
    """
    counts = {"pass": 0, "compile_only": 0, "skipped": 0}
    failures: list[str] = []
    for key, status in results.items():
        text = str(status)
        if text.startswith("FAIL"):
            failures.append(f"{key}={text}")
        elif text.startswith("PASS"):
            counts["pass"] += 1
        elif text.startswith("COMPILE_ONLY"):
            counts["compile_only"] += 1
        else:
            counts["skipped"] += 1
    return {
        "counts": counts,
        "failures": failures,
        "verdict": "FAIL" if failures else "PASS",
    }


def log_gemv_summary(verdict: dict[str, Any]) -> str:
    """Format and log the GEMV checks summary line."""
    counts = verdict["counts"]
    line = (
        f"GEMV checks summary: {counts['pass']} passed, "
        f"{counts['compile_only']} compile-only (engine golden required), "
        f"{counts['skipped']} skipped, {len(verdict['failures'])} failed"
    )
    log(line)
    return line


def judge_and_log_summary(results: dict[str, str]) -> dict[str, Any]:
    """Judge results and print the summary line to stdout."""
    verdict = judge_results(results)
    log_gemv_summary(verdict)
    return verdict


def _find_first_mismatch(a: np.ndarray, b: np.ndarray) -> tuple[int, ...] | None:
    """Return index of first mismatch viewing arrays as uint32."""
    a_u32 = np.asarray(a).view(np.uint32)
    b_u32 = np.asarray(b).view(np.uint32)
    diff = a_u32 != b_u32
    if not np.any(diff):
        return None
    idx = np.unravel_index(np.argmax(diff), a_u32.shape)
    return tuple(int(i) for i in idx)


class MergedShapeRunner:
    """Runs one GEMV shape on the merged (cols=7, rda_col=7) configuration."""

    def __init__(
        self,
        shape_spec: Any,
        *,
        cols: int,
        rda_col: int,
        continuous_mem_w: bool,
        context: Any,
        aie: Any,
        bfloat16_dtype: Any,
        seed: int,
    ) -> None:
        self.shape_spec = shape_spec
        self.shape = shape_spec if hasattr(shape_spec, "K") else get_shape(str(shape_spec))
        self.cols = cols
        self.rda_col = rda_col
        self.aie = aie
        self.bfloat16 = bfloat16_dtype

        self.op = UnifiedGEMV.for_shape(
            self.shape,
            cols=cols,
            rda_col=rda_col,
            continuous_mem_w=continuous_mem_w,
            context=context,
        )
        self.op.set_up_artifacts()
        self.op.compile()
        self.fn = self.op.get_callable()

        w_words, x_u16, y_ref_u16 = make_shape_test_case(self.shape.K, self.shape.M, seed=seed)
        self.y_ref_u16 = y_ref_u16

        x_padded = np.zeros(MAX_K, dtype=np.uint16)
        x_padded[: self.shape.K] = x_u16
        self.X_bo = self.aie.tensor(x_padded.view(bfloat16_dtype))
        self.W_bo = self.aie.tensor(w_words)
        self.Y_bo = self.aie.full((self.shape.M,), SENTINEL, dtype=bfloat16_dtype)

    def invoke(self) -> Any:
        return self.fn(self.Y_bo, self.W_bo, self.X_bo)

    def read_output_u16(self) -> np.ndarray:
        return np.asarray(self.Y_bo.numpy()).view(np.uint16).copy()

    def verify(self) -> dict[str, Any]:
        got_u16 = self.read_output_u16()
        return compare_bf16_bits(got_u16, self.y_ref_u16)


def run_gemv_checks(
    *,
    cols: int,
    rda_col: int,
    continuous_mem_w: bool,
    context: Any,
    aie: Any,
    bfloat16_dtype: Any,
    seed: int,
) -> dict[str, Any]:
    log("=" * 70)
    log(f"(가) Merged GEMV Shape Verification (cols={cols}, rda_col={rda_col})")
    log("=" * 70)

    results: dict[str, str] = {}

    # 1. BFP16 shapes
    bfp16_shapes = [s for s in ALL_SHAPES if s.format == FORMAT_BFP16]
    for idx, shape in enumerate(bfp16_shapes):
        if shape.name == "moe_b2_9":
            log(f"  [COMPILE_ONLY -- 기능 판정은 엔진 골든에서] {shape.name:12s} (K={shape.K:5d}, M={shape.M:6d}, x_slots=7)")
            results[shape.name] = COMPILE_ONLY_STATUS
            continue

        t0 = time.perf_counter()
        try:
            runner = MergedShapeRunner(
                shape,
                cols=cols,
                rda_col=rda_col,
                continuous_mem_w=continuous_mem_w,
                context=context,
                aie=aie,
                bfloat16_dtype=bfloat16_dtype,
                seed=seed + idx * 31,
            )
            runner.invoke()
            diff = runner.verify()
            dt = time.perf_counter() - t0
            if diff["mismatches"] == 0:
                log(f"  [PASS] {shape.name:12s} (K={shape.K:5d}, M={shape.M:6d}) in {dt:.2f}s")
                results[shape.name] = "PASS"
            else:
                log(f"  [FAIL] {shape.name:12s} mismatches={diff['mismatches']}")
                results[shape.name] = f"FAIL: mismatches={diff['mismatches']}"
        except Exception as exc:
            log(f"  [FAIL] {shape.name:12s} error: {exc}")
            results[shape.name] = f"FAIL: {exc}"

    # 2. Q4NX shapes: dense_gu
    try:
        from bench.gemv_unified.stage2_bench import _dense_case
        s, w, x, ref = _dense_case(seed)
        t0 = time.perf_counter()
        dense_op = UnifiedGEMV.for_shape(
            "dense_gu",
            cols=cols,
            rda_col=rda_col,
            continuous_mem_w=continuous_mem_w,
            context=context,
        )
        dense_op.compile()
        dense_fn = dense_op.get_callable()
        y_bo = aie.full((s.M,), SENTINEL, dtype=bfloat16_dtype)
        dense_fn(y_bo, aie.tensor(w), aie.tensor(pad_x(x, s)))
        dt = time.perf_counter() - t0

        from bench.moe_d.moe_d_bench import _check
        _check("merged dense_gu", y_bo.numpy(), ref)
        log(f"  [PASS] dense_gu     (K={s.K:5d}, M={s.M:6d}) in {dt:.2f}s")
        results["dense_gu"] = "PASS"
    except Exception as exc:
        # dense_gu is a required check: an error here must fail the gate.
        log(f"  [FAIL] dense_gu: {exc}")
        results["dense_gu"] = f"FAIL: {exc}"

    # 3. MoE B1 / B2 templates must share the same merged xclbin.  For B1,
    #    the result goes under a separate '#template' key so compiling never
    #    overwrites the shape's functional result from step 1.  For B2, which has
    #    no functional check in step 1, the single shape key is used so B2 is
    #    counted exactly once as compile-only.
    for name, TemplateCls in [("moe_b1_6", UnifiedMoEGateUp), ("moe_b2_9", UnifiedMoEDown9)]:
        key = name if name == "moe_b2_9" else f"{name}#template"
        try:
            tmpl = TemplateCls(
                cols=cols,
                rda_col=rda_col,
                continuous_mem_w=continuous_mem_w,
                context=context,
            )
            tmpl.compile()
            if name == "moe_b2_9":
                results[key] = COMPILE_ONLY_STATUS
                log(f"  [COMPILE_ONLY] {name:12s} compiled on shared xclbin (engine golden required)")
            else:
                results[key] = "PASS"
                log(f"  [PASS] {name:12s} template compiled on shared xclbin")
        except Exception as exc:
            log(f"  [FAIL] {name:12s}: template compile error: {exc}")
            results[key] = f"FAIL: {exc}"

    judge_and_log_summary(results)
    return results


def run_rda_checks(
    *,
    cols: int,
    rda_col: int,
    heads: int,
    tokens: int,
    seeds: list[int],
    continuous_mem_w: bool,
    context: Any,
    aie: Any,
) -> None:
    log("=" * 70)
    log(f"(나) Merged RDA Verification vs Original 4-Col (heads={heads}, tokens={tokens})")
    log("=" * 70)

    # 1. Merged UnifiedRDA
    merged_rda_op = UnifiedRDA(
        cols=cols,
        rda_col=rda_col,
        rda_heads=heads,
        continuous_mem_w=continuous_mem_w,
        context=context,
    )
    merged_rda_op.compile()
    merged_rda_fn = merged_rda_op.get_callable()
    log(f"  UnifiedRDA ready: xclbin={merged_rda_fn.xclbin_path.name} "
        f"(mtime={merged_rda_fn.xclbin_mtime_ns})")

    # Allocate NPU ping-pong buffers
    s_in_m = aie.zeros((heads, V_DIM, K_DIM), dtype=np.float32)
    s_out_m = aie.zeros_like(s_in_m)
    head_in_m = aie.zeros((heads, 640), dtype=np.float32)
    out_m = aie.zeros((heads, V_DIM), dtype=np.float32)

    s_in_orig = aie.zeros((heads, V_DIM, K_DIM), dtype=np.float32)
    s_out_orig = aie.zeros_like(s_in_orig)
    head_in_orig = aie.zeros((heads, 640), dtype=np.float32)
    out_orig = aie.zeros((heads, V_DIM), dtype=np.float32)

    # Warmup
    for _ in range(3):
        merged_rda_fn(s_in_m, s_out_m, head_in_m, out_m)
        rda_step(
            s_in_orig,
            s_out_orig,
            head_in_orig,
            out_orig,
            num_heads=heads,
            num_workers=4,
            use_memtile=True,
        )

    for seed in seeds:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Initialize both with identical random state
        S_init = rng.standard_normal((heads, V_DIM, K_DIM)).astype(np.float32) * 0.01
        s_in_m[:] = S_init
        s_in_m.to("npu")
        s_in_orig[:] = S_init
        s_in_orig.to("npu")

        cur_s_in_m, cur_s_out_m = s_in_m, s_out_m
        cur_s_in_o, cur_s_out_o = s_in_orig, s_out_orig

        for t in range(tokens):
            q = rng.standard_normal((heads, K_DIM)).astype(np.float32)
            k = rng.standard_normal((heads, K_DIM)).astype(np.float32)
            a = np.exp(rng.uniform(-5.0, 0.0, size=(heads, K_DIM)).astype(np.float32))
            v = rng.standard_normal((heads, V_DIM)).astype(np.float32)
            beta = rng.uniform(0.01, 0.99, size=heads).astype(np.float32)

            bundle = pack_head_bundle(q, k, a, v, beta)
            head_in_m[:] = bundle
            head_in_m.to("npu")
            head_in_orig[:] = bundle
            head_in_orig.to("npu")

            merged_rda_fn(cur_s_in_m, cur_s_out_m, head_in_m, out_m)
            rda_step(
                cur_s_in_o,
                cur_s_out_o,
                head_in_orig,
                out_orig,
                num_heads=heads,
                num_workers=4,
                use_memtile=True,
            )

            # Check uint32 exact equality
            s_m_arr = cur_s_out_m.numpy()
            s_o_arr = cur_s_out_o.numpy()
            s_mismatch = _find_first_mismatch(s_m_arr, s_o_arr)
            if s_mismatch is not None:
                raise AssertionError(
                    f"Seed {seed} token {t}: s_out mismatch at {s_mismatch} "
                    f"merged={s_m_arr[s_mismatch]} vs orig={s_o_arr[s_mismatch]}"
                )

            out_m_arr = out_m.numpy()
            out_o_arr = out_orig.numpy()
            out_mismatch = _find_first_mismatch(out_m_arr, out_o_arr)
            if out_mismatch is not None:
                raise AssertionError(
                    f"Seed {seed} token {t}: out mismatch at {out_mismatch} "
                    f"merged={out_m_arr[out_mismatch]} vs orig={out_o_arr[out_mismatch]}"
                )

            # Ping-pong swap
            cur_s_in_m, cur_s_out_m = cur_s_out_m, cur_s_in_m
            cur_s_in_o, cur_s_out_o = cur_s_out_o, cur_s_in_o

        dt = time.perf_counter() - t0
        log(f"  [PASS] seed {seed}: {tokens} tokens uint32 bit-exact match in {dt:.2f}s "
            f"({dt / tokens * 1e3:.2f} ms/token)")


def run_alternation_bench(
    *,
    cols: int,
    rda_col: int,
    heads: int,
    iterations: int,
    continuous_mem_w: bool,
    context: Any,
    aie: Any,
    bfloat16_dtype: Any,
) -> None:
    log("=" * 70)
    log(f"(다) Alternation Latency Benchmark (iters={iterations})")
    log("=" * 70)

    # 1. Setup shared PDI operators (GEMV 'o_proj' + RDA)
    gemv_runner = MergedShapeRunner(
        "o_proj",
        cols=cols,
        rda_col=rda_col,
        continuous_mem_w=continuous_mem_w,
        context=context,
        aie=aie,
        bfloat16_dtype=bfloat16_dtype,
        seed=42,
    )
    rda_op = UnifiedRDA(
        cols=cols,
        rda_col=rda_col,
        rda_heads=heads,
        continuous_mem_w=continuous_mem_w,
        context=context,
    )
    rda_op.compile()
    rda_fn = rda_op.get_callable()

    s_in = aie.zeros((heads, V_DIM, K_DIM), dtype=np.float32)
    s_out = aie.zeros_like(s_in)
    head_in = aie.zeros((heads, 640), dtype=np.float32)
    out = aie.zeros((heads, V_DIM), dtype=np.float32)

    # Warmup
    for _ in range(10):
        gemv_runner.invoke()
        rda_fn(s_in, s_out, head_in, out)

    # Measure shared PDI consecutive vs alternating
    t_gemv_only = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        gemv_runner.invoke()
        t1 = time.perf_counter_ns()
        t_gemv_only.append((t1 - t0) / 1e3)

    t_rda_only = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        rda_fn(s_in, s_out, head_in, out)
        t1 = time.perf_counter_ns()
        t_rda_only.append((t1 - t0) / 1e3)

    t_shared_alt = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        gemv_runner.invoke()
        rda_fn(s_in, s_out, head_in, out)
        t1 = time.perf_counter_ns()
        t_shared_alt.append((t1 - t0) / 1e3)

    base_pair_us = np.median(t_gemv_only) + np.median(t_rda_only)
    shared_overhead_us = np.array(t_shared_alt) - base_pair_us
    p50_shared = float(np.percentile(shared_overhead_us, 50))
    p95_shared = float(np.percentile(shared_overhead_us, 95))

    log(f"  [Shared PDI]  GEMV solo: p50={np.percentile(t_gemv_only, 50):.1f}us | "
        f"RDA solo: p50={np.percentile(t_rda_only, 50):.1f}us")
    log(f"  [Shared PDI]  Alternating extra overhead: p50={p50_shared:+.1f}us, p95={p95_shared:+.1f}us")

    # 2. Compare against separate xclbin alternation (8-col GEMV + standalone 4-col RDA)
    compare_separate_xclbins(
        gemv_runner=gemv_runner,
        s_in=s_in,
        s_out=s_out,
        head_in=head_in,
        out=out,
        heads=heads,
        iterations=iterations,
        context=context,
    )


def compare_separate_xclbins(
    *,
    gemv_runner: Any,
    s_in: Any,
    s_out: Any,
    head_in: Any,
    out: Any,
    heads: int,
    iterations: int,
    context: Any,
    sep_gemv_fn: Any = None,
    rda_step_fn: Any = None,
) -> dict[str, float]:
    """Benchmark separate 8-col GEMV + 4-col RDA alternation as a required check.

    In a single process, the merged xclbin (cols=7, rda_col=7), 8-col GEMV
    xclbin (cols=8, rda_col=None), and separate RDA xclbin are used together.
    Before creating the 8-col callable, we must call reset_xclbin_signature_cache()
    so the 8-col unified xclbin signature is cleanly established.
    """
    try:
        # 한 프로세스에서 합친 xclbin (cols=7, rda_col=7), 8열 GEMV xclbin (cols=8, rda_col=None),
        # 분리 RDA xclbin이 함께 쓰이므로, 8열 UnifiedGEMV callable 생성 직전에
        # reset_xclbin_signature_cache()를 호출하여 8열 통합 xclbin 서명을 새로 잡게 한다.
        reset_xclbin_signature_cache()

        if sep_gemv_fn is None:
            from kernels.gemv_unified.op import UnifiedGEMV as SepGEMV

            sep_gemv_op = SepGEMV.for_shape(
                "o_proj",
                cols=8,
                rda_col=None,
                continuous_mem_w=True,
                context=context,
            )
            sep_gemv_op.compile()
            sep_gemv_fn = sep_gemv_op.get_callable()

        if rda_step_fn is None:
            if rda_step is None:
                raise RuntimeError(
                    "standalone rda_step is unavailable (failed to import from bench.rda_state)"
                )
            rda_step_fn = rda_step

        warmup_iters = min(10, iterations)
        for _ in range(warmup_iters):
            sep_gemv_fn(gemv_runner.Y_bo, gemv_runner.W_bo, gemv_runner.X_bo)
            rda_step_fn(
                s_in,
                s_out,
                head_in,
                out,
                num_heads=heads,
                num_workers=4,
                use_memtile=True,
            )

        t_sep_gemv_only: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            sep_gemv_fn(gemv_runner.Y_bo, gemv_runner.W_bo, gemv_runner.X_bo)
            t1 = time.perf_counter_ns()
            t_sep_gemv_only.append((t1 - t0) / 1e3)

        t_sep_rda_only: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            rda_step_fn(
                s_in,
                s_out,
                head_in,
                out,
                num_heads=heads,
                num_workers=4,
                use_memtile=True,
            )
            t1 = time.perf_counter_ns()
            t_sep_rda_only.append((t1 - t0) / 1e3)

        t_sep_alt: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            sep_gemv_fn(gemv_runner.Y_bo, gemv_runner.W_bo, gemv_runner.X_bo)
            rda_step_fn(
                s_in,
                s_out,
                head_in,
                out,
                num_heads=heads,
                num_workers=4,
                use_memtile=True,
            )
            t1 = time.perf_counter_ns()
            t_sep_alt.append((t1 - t0) / 1e3)

        base_sep_pair_us = np.median(t_sep_gemv_only) + np.median(t_sep_rda_only)
        sep_overhead_us = np.array(t_sep_alt) - base_sep_pair_us
        p50_sep = float(np.percentile(sep_overhead_us, 50))
        p95_sep = float(np.percentile(sep_overhead_us, 95))
        p50_gemv = float(np.percentile(t_sep_gemv_only, 50))
        p50_rda = float(np.percentile(t_sep_rda_only, 50))

        log(
            f"  [Separate XCLBINs] GEMV solo: p50={p50_gemv:.1f}us | "
            f"RDA solo: p50={p50_rda:.1f}us"
        )
        log(
            f"  [Separate XCLBINs] Alternating extra overhead: "
            f"p50={p50_sep:+.1f}us, p95={p95_sep:+.1f}us"
        )
        return {
            "gemv_solo_p50": p50_gemv,
            "rda_solo_p50": p50_rda,
            "overhead_p50": p50_sep,
            "overhead_p95": p95_sep,
        }
    except Exception as exc:
        log(f"  [Separate XCLBINs] Comparison failed: {exc}")
        raise RuntimeError(f"Separate XCLBINs comparison failed: {exc}") from exc


def _brace_delta(line: str) -> int:
    """Count braces outside MLIR string literals."""
    delta = 0
    quoted = False
    escaped = False
    for char in line:
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            delta += 1
        elif not quoted and char == "}":
            delta -= 1
    return delta


def strip_runtime_sequence(mlir: str) -> str:
    """Remove the complete runtime sequence operation from textual MLIR.

    생성된 MLIR을 한 번 출력해 보고 runtime sequence 블록을 잘라내는 방식을 결정함:
    IRON/mlir-aie에서는 런타임 시퀀스가 `aie.runtime_sequence` 또는 `aiex.runtime_sequence`
    블록으로 생성된다. 따라서 MLIR 텍스트를 줄 단위로 순회하며 해당 op의 시작부터
    문자열 리터럴을 고려한 중괄호 깊이(_brace_delta)를 추적하여 닫는 중괄호까지 블록 전체를
    제거하고, Tiles, Buffers, Locks, ObjectFifos, Workers 등 장치 본체 선언부만 남겨 비교한다.
    """
    kept: list[str] = []
    removing = False
    depth = 0
    found = False
    for line in mlir.splitlines():
        if not removing and ("runtime_sequence" in line):
            removing = True
            found = True
            depth = _brace_delta(line)
            if depth <= 0:
                removing = False
            continue
        if removing:
            depth += _brace_delta(line)
            if depth <= 0:
                removing = False
            continue
        kept.append(line.rstrip())
    if not found or removing:
        raise ValueError("could not isolate a complete runtime_sequence operation from MLIR")
    return "\n".join(kept) + "\n"


def check_device_body_identity(
    *,
    cols: int,
    rda_col: int,
    heads: int,
    continuous_mem_w: bool,
    context: Any,
) -> bool:
    """Verify that Config (GEMV), RDA, and GEMV shape MLIRs produce identical device bodies."""
    log("=== Stage (0): Device Body Identity Gate ===")
    log("  Generating 3 MLIR modules via IRON DesignGenerator:")
    log(f"    1. Config MLIR (GEMV reference shape, cols={cols}, rda_col={rda_col})")
    log(f"    2. RDA instruction stream MLIR (mode='rda', heads={heads})")
    log("    3. Sample shape instruction stream MLIR (o_proj)")

    # 1. Config MLIR (gemv mode, reference shape K=256, M=cols * M_PER_CHUNK)
    ref_k = 256
    ref_m = cols * M_PER_CHUNK
    config_op = UnifiedGEMV(
        K=ref_k,
        M=ref_m,
        cols=cols,
        rda_col=rda_col,
        rda_heads=heads,
        experimental_shape=True,
        continuous_mem_w=continuous_mem_w,
        context=context,
    )
    config_art = config_op._mlir_artifact(f"{config_op.config_name}.mlir", ref_k, ref_m)
    config_mlir = config_art.generator()

    # 2. RDA command stream MLIR (rda mode)
    rda_op = UnifiedRDA(
        cols=cols,
        rda_col=rda_col,
        rda_heads=heads,
        continuous_mem_w=continuous_mem_w,
        context=context,
    )
    rda_art = rda_op._mlir_artifact(f"{rda_op.name}.mlir")
    rda_mlir = rda_art.generator()

    # 3. Sample shape command stream MLIR (o_proj)
    sample_op = UnifiedGEMV.for_shape(
        "o_proj",
        cols=cols,
        rda_col=rda_col,
        rda_heads=heads,
        continuous_mem_w=continuous_mem_w,
        context=context,
    )
    sample_art = sample_op.get_mlir_artifact()
    sample_mlir = sample_art.generator()

    # Strip runtime sequence block to isolate device body
    body_config = strip_runtime_sequence(config_mlir)
    body_rda = strip_runtime_sequence(rda_mlir)
    body_sample = strip_runtime_sequence(sample_mlir)

    def print_differences(label1: str, text1: str, label2: str, text2: str, limit: int = 15) -> None:
        shown = 0
        from itertools import zip_longest
        for line_no, (left, right) in enumerate(
            zip_longest(text1.splitlines(), text2.splitlines(), fillvalue="<EOF>"),
            start=1,
        ):
            if left == right:
                continue
            log(f"    line {line_no}:")
            log(f"      [{label1}]: {left}")
            log(f"      [{label2}]: {right}")
            shown += 1
            if shown >= limit:
                break

    if body_config != body_rda:
        log("ERROR: Device body differs between GEMV config and RDA command stream!")
        print_differences("Config", body_config, "RDA", body_rda)
        print("RESULT: FAIL device body differs")
        return False

    if body_config != body_sample:
        log("ERROR: Device body differs between GEMV config and Sample GEMV shape command stream!")
        print_differences("Config", body_config, "SampleShape", body_sample)
        print("RESULT: FAIL device body differs")
        return False

    log("  [PASS] Device body is bit-for-bit identical across Config, RDA, and GEMV shape streams.")
    return True


def main() -> int:
    configure_streams()
    parser = argparse.ArgumentParser(
        description="Verify merged 7-col GEMV + col 7 RDA in single PDI"
    )
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--rda-col", type=int, default=7)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    try:
        from ml_dtypes import bfloat16
        import aie.utils as aie
        from iron.common import AIEContext
    except (ImportError, ModuleNotFoundError) as exc:
        log(f"IRON/NPU toolchain not available on this host: {exc}")
        print("RESULT: FAIL IRON/NPU environment unavailable on CPU-only host")
        return 1

    ctx = AIEContext()

    try:
        # Step (0): Verify device body identity across Config, RDA, and GEMV command streams
        if not check_device_body_identity(
            cols=args.cols,
            rda_col=args.rda_col,
            heads=args.heads,
            continuous_mem_w=True,
            context=ctx,
        ):
            return 1

        # Step (a): Verify all GEMV shapes on shared xclbin
        gemv_res = run_gemv_checks(
            cols=args.cols,
            rda_col=args.rda_col,
            continuous_mem_w=True,
            context=ctx,
            aie=aie,
            bfloat16_dtype=bfloat16,
            seed=123,
        )
        failures = [k for k, v in gemv_res.items() if v.startswith("FAIL")]
        if failures:
            print(f"RESULT: FAIL GEMV shape mismatches on {failures}")
            return 1

        # Step (b): Verify RDA bit-exactness across seeds
        run_rda_checks(
            cols=args.cols,
            rda_col=args.rda_col,
            heads=args.heads,
            tokens=args.tokens,
            seeds=args.seeds,
            continuous_mem_w=True,
            context=ctx,
            aie=aie,
        )

        # Step (c): Alternation benchmark
        run_alternation_bench(
            cols=args.cols,
            rda_col=args.rda_col,
            heads=args.heads,
            iterations=args.iters,
            continuous_mem_w=True,
            context=ctx,
            aie=aie,
            bfloat16_dtype=bfloat16,
        )

        print("RESULT: PASS")
        return 0
    except Exception as exc:
        log(f"Execution failed with exception: {exc}")
        print(f"RESULT: FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
