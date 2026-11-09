# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Build/audit the stage-2a 7-column+RDA prefill plate on the notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from bench.rda_state.rda_state_1col import (
    MODEL_RDA_HEADS,
    pack_head_bundle,
    numpy_recurrent_rda_step,
)
from host.weights import f32_to_bf16_rtne
from kernels.gemm_prefill.audit import section_sizes, text_size
from kernels.gemm_prefill.budget import (
    CORE_BD_LIMIT,
    CORE_DATA_LIMIT_BYTES,
    CORE_LOCK_LIMIT,
    CORE_PROGRAM_LIMIT_BYTES,
    MEMTILE_BD_LIMIT,
    MEMTILE_LIMIT_BYTES,
    MEMTILE_LOCK_LIMIT,
    SHIM_BD_LIMIT,
    resource_budget,
)
from kernels.gemm_prefill.op import PrefillGEMM
from kernels.gemm_prefill.shapes import MAX_BATCH_T, PREFILL_SHAPE_NAMES
from kernels.gemv_unified.op import UnifiedGEMV
from npu.backend import NpuBackend, bfloat16

try:
    import aie.utils as aie_utils
except (ImportError, ModuleNotFoundError):
    aie_utils = None


RDA_STATE_NRMSE_LIMIT = 1e-4
RDA_OUTPUT_NRMSE_LIMIT = 1e-5


def _build_root(path):
    """Return only this xclbin's compiler directory, never a shared parent."""
    return path.parent / f"{path.stem}.mlir.d"


def _tile_kind(row):
    if row == 0:
        return "shim"
    if row == 1:
        return "memtile"
    return "core"


def _core_role(label, col):
    if label == "prefill-7col-rda":
        return "gemm" if col < 7 else "rda"
    if label == "decode-unified":
        return "decode-gemv-moe" if col < 7 else "rda"
    return "gemm"


def _braced_region(source, start):
    opening = source.find("{", start)
    if opening < 0:
        return ""
    depth = 0
    for pos in range(opening, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:pos + 1]
    return source[opening:]


def _memref_bytes(type_text):
    match = re.fullmatch(r"((?:\d+x)+)(bf16|f16|f32|f64|i\d+|ui\d+)", type_text)
    if match is None:
        return None
    count = 1
    for dim in match.group(1).rstrip("x").split("x"):
        count *= int(dim)
    dtype = match.group(2)
    bits = 16 if dtype == "bf16" else int(re.search(r"\d+", dtype).group())
    return (count * bits + 7) // 8


def _address_resources(path):
    """Read allocated addresses/IDs from a post-address-assignment MLIR."""
    source = path.read_text(encoding="utf-8", errors="replace")
    tiles = {
        name: (int(col), int(row))
        for name, col, row in re.findall(
            r"%(\w+)\s*=\s*aie\.tile\(\s*(\d+)\s*,\s*(\d+)\s*\)", source
        )
    }
    resources = {}

    def entry(coord):
        col, row = coord
        return resources.setdefault(
            coord,
            {"col": col, "row": row, "kind": _tile_kind(row),
             "buffer_end_bytes": 0, "bd_ids": set(), "lock_ids": set(),
             "runtime_bd_peak": 0},
        )

    buffer_pattern = re.compile(
        r"%\w+\s*=\s*aie\.buffer\(%(\w+)\).*?"
        r"address\s*=\s*(0x[0-9a-fA-F]+|\d+).*?"
        r":\s*memref<([^>]+)>",
    )
    for tile_name, address_text, memref_type in buffer_pattern.findall(source):
        if tile_name not in tiles:
            continue
        size = _memref_bytes(memref_type)
        if size is None:
            continue
        item = entry(tiles[tile_name])
        item["buffer_end_bytes"] = max(
            item["buffer_end_bytes"], int(address_text, 0) + size
        )

    for tile_name, lock_id in re.findall(
        r"aie\.lock\(%(\w+)\s*,\s*(\d+)\)", source
    ):
        if tile_name in tiles:
            entry(tiles[tile_name])["lock_ids"].add(int(lock_id))

    region_pattern = re.compile(
        r"(?:%\w+\s*=\s*)?aie\.(mem|memtile_dma|shim_dma)\(%(\w+)\)\s*\{"
    )
    for match in region_pattern.finditer(source):
        tile_name = match.group(2)
        if tile_name not in tiles:
            continue
        region = _braced_region(source, match.start())
        ids = {int(value) for value in re.findall(r"bd_id\s*=\s*(\d+)", region)}
        entry(tiles[tile_name])["bd_ids"].update(ids)

    shim_allocs = {
        symbol: tiles[tile_name]
        for symbol, tile_name in re.findall(
            r"aie\.shim_dma_allocation\s+@(\w+)\s*\(%(\w+)", source
        )
        if tile_name in tiles
    }
    # Shim BDs are allocated in the runtime sequence, not necessarily in an
    # aie.shim_dma device region.  input_with_addresses.mlir can still omit
    # bd_id attributes, so track configured tasks until dma_free_task and use
    # the peak number of live aie.dma_bd operations for that shim tile.
    runtime_tasks = {}
    live_runtime_bds = {}
    runtime_pattern = re.compile(
        r"(?P<task>%[\w.]+)\s*=\s*"
        r"aiex\.dma_configure_task_for\s+@(?P<symbol>\w+)\s*\{"
        r"|aiex\.dma_free_task\(\s*(?P<free>%[\w.]+)\s*\)"
    )
    for match in runtime_pattern.finditer(source):
        task = match.group("task")
        if task is not None:
            coord = shim_allocs.get(match.group("symbol"))
            if coord is None:
                continue
            region = _braced_region(source, match.start())
            ids = {int(value) for value in re.findall(r"bd_id\s*=\s*(\d+)", region)}
            bd_count = len(re.findall(r"\baie\.dma_bd\s*\(", region))
            if not bd_count:
                continue
            item = entry(coord)
            item["bd_ids"].update(ids)
            runtime_tasks[task] = (coord, bd_count)
            live_runtime_bds[coord] = live_runtime_bds.get(coord, 0) + bd_count
            item["runtime_bd_peak"] = max(
                item["runtime_bd_peak"], live_runtime_bds[coord]
            )
            continue
        freed = runtime_tasks.pop(match.group("free"), None)
        if freed is not None:
            coord, bd_count = freed
            live_runtime_bds[coord] -= bd_count

    rows = []
    for coord in sorted(resources):
        item = resources[coord]
        rows.append({
            "col": item["col"],
            "row": item["row"],
            "kind": item["kind"],
            "buffer_end_bytes": item["buffer_end_bytes"],
            "bd_count": max(len(item["bd_ids"]), item["runtime_bd_peak"]),
            "bd_ids": sorted(item["bd_ids"]),
            "lock_count": len(item["lock_ids"]),
            "lock_ids": sorted(item["lock_ids"]),
        })
    return rows


def _link_audit_passed(reports):
    """Apply measured program, memory, BD, and lock limits to every plate."""
    limits = {
        "core": (CORE_DATA_LIMIT_BYTES, CORE_BD_LIMIT, CORE_LOCK_LIMIT),
        "memtile": (MEMTILE_LIMIT_BYTES, MEMTILE_BD_LIMIT, MEMTILE_LOCK_LIMIT),
        "shim": (None, SHIM_BD_LIMIT, None),
    }
    for report in reports:
        if not report["elf"] or not report["addressed_mlir"]:
            return False
        if any(row["text_bytes"] > CORE_PROGRAM_LIMIT_BYTES for row in report["elf"]):
            return False
        for mlir in report["addressed_mlir"]:
            allocations = mlir["allocations"]
            if not allocations or not any(row["kind"] == "shim" for row in allocations):
                return False
            for row in allocations:
                memory_limit, bd_limit, lock_limit = limits[row["kind"]]
                if memory_limit is not None and row["buffer_end_bytes"] > memory_limit:
                    return False
                if row["bd_count"] > bd_limit:
                    return False
                if lock_limit is not None and row["lock_count"] > lock_limit:
                    return False
                if row["kind"] == "shim" and row["bd_count"] == 0:
                    return False
    return True


def _artifact_rows(label, artifact):
    path = Path(artifact.filename).resolve()
    build_root = _build_root(path)
    elf_rows = []
    if build_root.is_dir():
        for core_dir in sorted(build_root.glob("elfs_main_core_*_*")):
            match = re.fullmatch(r"elfs_main_core_(\d+)_(\d+)", core_dir.name)
            if match is None or not core_dir.is_dir():
                continue
            col, row = map(int, match.groups())
            paths = []
            text_bytes = stack_bytes = data_bytes = 0
            for elf in sorted(core_dir.rglob("*.elf")):
                try:
                    sections = section_sizes(elf)
                    text_bytes += text_size(elf)
                    stack_bytes += sections.get(".stack_sizes", 0)
                    data_bytes += sum(
                        size for name, size in sections.items()
                        if name in (".data", ".bss", ".rodata")
                        or name.startswith(".data.")
                    )
                    paths.append(str(elf))
                except ValueError:
                    continue
            if paths:
                elf_rows.append({
                    "col": col,
                    "row": row,
                    "role": _core_role(label, col),
                    "paths": paths,
                    "text_bytes": text_bytes,
                    "program_limit_bytes": CORE_PROGRAM_LIMIT_BYTES,
                    "program_margin_bytes": CORE_PROGRAM_LIMIT_BYTES - text_bytes,
                    "stack_sizes_section_bytes": stack_bytes,
                    "data_section_bytes": data_bytes,
                })
    mlir_rows = []
    if build_root.is_dir():
        for mlir in sorted(build_root.rglob("input_with_addresses.mlir")):
            mlir_rows.append({
                "path": str(mlir),
                "bytes": mlir.stat().st_size,
                "allocations": _address_resources(mlir),
            })
    return {
        "label": label,
        "artifact": str(path),
        "build_root": str(build_root),
        "elf": elf_rows,
        "addressed_mlir": mlir_rows,
    }


def _compile_plate(cols, rda_col):
    ops = []
    for shape_name in PREFILL_SHAPE_NAMES:
        op = PrefillGEMM(shape_name, MAX_BATCH_T, cols=cols, rda_col=rda_col)
        op.compile()
        ops.append(op)
    rda = None
    if rda_col is not None:
        rda = PrefillGEMM(
            "rda_in", MAX_BATCH_T, cols=cols, rda_col=rda_col, execution="rda"
        )
        rda.compile()
    return ops, rda


def _overwrite(tensor, value):
    overwrite = getattr(tensor, "overwrite", None)
    if overwrite is None:
        tensor[:] = value
    else:
        with overwrite() as view:
            view[...] = value


def _record_first_failure(first_failure, iteration, item):
    if first_failure is None:
        return {"iteration": iteration, "item": item}
    return first_failure


def _error_metrics(actual, reference, nrmse_limit):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    diff = actual - reference
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))
    value_range = float(np.max(reference) - np.min(reference))
    nrmse = rmse / (value_range + 1e-12)
    return {
        "max_abs": max_abs,
        "nrmse": nrmse,
        "nrmse_limit": nrmse_limit,
        "passed": nrmse <= nrmse_limit,
    }


def _alternate(args, gemm_op, rda_op):
    if aie_utils is None:
        raise RuntimeError("AIE runtime is unavailable")
    backend = NpuBackend(weights_dir=args.weights_dir, mock=False)
    name = f"model.layers.{args.rda_layer}.self_attn.q_proj.weight"
    comp = backend.comp_map[name]
    bmeta = backend.bundle_map[comp.bundle_name]
    rng = np.random.default_rng(args.seed)
    x = f32_to_bf16_rtne(rng.standard_normal((MAX_BATCH_T, bmeta.k), dtype=np.float32))
    backend.set_step(1)
    gemm_ref = np.stack([backend.linear(name, row) for row in x])
    x_tensor = aie_utils.tensor(x.reshape(-1).view(bfloat16))
    y_tensor = aie_utils.full((MAX_BATCH_T * bmeta.m,), 0.0, dtype=bfloat16)
    w_tensor = backend._get_weight_tensor(comp.bundle_name)
    gemm_fn = gemm_op.get_callable()

    h = MODEL_RDA_HEADS
    state = rng.standard_normal((h, 128, 128), dtype=np.float32) * np.float32(0.01)
    q = rng.standard_normal((h, 128), dtype=np.float32) * np.float32(0.1)
    k = rng.standard_normal((h, 128), dtype=np.float32) * np.float32(0.1)
    log_alpha = np.full((h, 128), np.log(np.float32(0.99)), dtype=np.float32)
    a = np.exp(log_alpha)
    v = rng.standard_normal((h, 128), dtype=np.float32) * np.float32(0.1)
    beta = np.full(h, np.float32(0.2))
    state_ref, out_ref = numpy_recurrent_rda_step(state, q, k, a, v, beta)
    head = pack_head_bundle(q, k, a, v, beta)

    # Byte reference: run the identical RDA kernel through the decode-unified
    # backend once. NumPy remains an accuracy metric, not the parity oracle.
    backend.rda_states[args.rda_layer] = [
        aie_utils.tensor(state),
        aie_utils.tensor(np.zeros_like(state)),
    ]
    backend.rda_active_slot[args.rda_layer] = 0
    backend.rda_update(args.rda_layer, q, k, v, log_alpha, beta)
    decode_state = np.array(
        backend.rda_states[args.rda_layer][1].numpy(), copy=True
    )
    decode_out = np.array(
        backend._prepared_buffer("rda_out").numpy(), copy=True
    )
    s_in = aie_utils.tensor(state)
    s_out = aie_utils.tensor(np.zeros_like(state))
    head_in = aie_utils.tensor(head)
    out = aie_utils.tensor(np.zeros((h, 128), np.float32))
    rda_fn = rda_op.get_callable()

    mismatches = {"gemm": 0, "rda_state": 0, "rda_output": 0}
    first_failure = None
    timeouts = 0
    gemm_seconds = 0.0
    rda_seconds = 0.0
    for iteration in range(1, args.iterations + 1):
        try:
            started = time.perf_counter()
            gemm_fn(y_tensor, w_tensor, x_tensor)
            gemm_seconds += time.perf_counter() - started
            got = np.asarray(y_tensor.numpy()).view(np.uint16).reshape(MAX_BATCH_T, bmeta.m)
            if not np.array_equal(got[:, comp.row_start:comp.row_end], gemm_ref):
                mismatches["gemm"] += 1
                first_failure = _record_first_failure(
                    first_failure, iteration, "gemm"
                )
        except TimeoutError:
            timeouts += 1
            first_failure = _record_first_failure(
                first_failure, iteration, "gemm_timeout"
            )

        _overwrite(s_out, np.zeros_like(state))
        _overwrite(out, np.zeros((h, 128), np.float32))
        try:
            started = time.perf_counter()
            rda_fn(s_in, s_out, head_in, out)
            rda_seconds += time.perf_counter() - started
            got_state = np.asarray(s_out.numpy())
            got_out = np.asarray(out.numpy())
            if not np.array_equal(got_state, decode_state):
                mismatches["rda_state"] += 1
                first_failure = _record_first_failure(
                    first_failure, iteration, "rda_state"
                )
            if not np.array_equal(got_out, decode_out):
                mismatches["rda_output"] += 1
                first_failure = _record_first_failure(
                    first_failure, iteration, "rda_output"
                )
        except TimeoutError:
            timeouts += 1
            first_failure = _record_first_failure(
                first_failure, iteration, "rda_timeout"
            )
    accuracy = {
        "reference": "numpy_recurrent_rda_step",
        "state": _error_metrics(decode_state, state_ref, RDA_STATE_NRMSE_LIMIT),
        "output": _error_metrics(decode_out, out_ref, RDA_OUTPUT_NRMSE_LIMIT),
    }
    mismatch_total = sum(mismatches.values())
    return {
        "iterations": args.iterations,
        "byte_reference": "decode-unified-rda",
        "mismatches": mismatches,
        "mismatch_total": mismatch_total,
        "first_failure": first_failure,
        "timeouts": timeouts,
        "numpy_accuracy": accuracy,
        "gemm_ms_per_call": gemm_seconds * 1e3 / args.iterations,
        "rda_ms_per_call": rda_seconds * 1e3 / args.iterations,
        "passed": (
            mismatch_total == 0
            and timeouts == 0
            and accuracy["state"]["passed"]
            and accuracy["output"]["passed"]
        ),
    }


def _stage2_estimates(reports):
    by_label = {report["label"]: report for report in reports}
    gemm = by_label["prefill-7col-rda"]
    decode = by_label["decode-unified"]

    def text_map(report):
        return {
            (row["col"], row["row"]): row["text_bytes"]
            for row in report["elf"]
        }

    def l1_map(report):
        result = {}
        for mlir in report["addressed_mlir"]:
            for row in mlir["allocations"]:
                if row["kind"] == "core":
                    coord = (row["col"], row["row"])
                    result[coord] = max(
                        result.get(coord, 0), row["buffer_end_bytes"]
                    )
        return result

    gemm_text = text_map(gemm)
    decode_text = text_map(decode)
    program = []
    for coord in sorted(set(gemm_text) & set(decode_text)):
        col, row = coord
        if col > 6:
            continue
        combined = gemm_text[coord] + decode_text[coord]
        program.append({
            "col": col,
            "row": row,
            "gemm_text_bytes": gemm_text[coord],
            "decode_gemv_moe_text_bytes": decode_text[coord],
            "estimated_combined_text_bytes": combined,
            "program_limit_bytes": CORE_PROGRAM_LIMIT_BYTES,
            "estimated_margin_bytes": CORE_PROGRAM_LIMIT_BYTES - combined,
            "estimated": True,
            "link_measured": False,
        })

    l1 = []
    for source, usage in (("gemm", l1_map(gemm)), ("decode", l1_map(decode))):
        for (col, row), used in sorted(usage.items()):
            if col > 6:
                continue
            l1.append({
                "source": source,
                "col": col,
                "row": row,
                "buffer_address_end_bytes": used,
                "data_memory_limit_bytes": CORE_DATA_LIMIT_BYTES,
                "estimated_margin_bytes": CORE_DATA_LIMIT_BYTES - used,
                "estimated": True,
                "link_measured": False,
            })
    return {
        "description": (
            "Estimated sums/margins for a future single linked program; "
            "these are not a combined-link measurement."
        ),
        "program_memory": program,
        "l1_data_memory": l1,
    }


def _budgets():
    return {
        f"{shape}-c{cols}": resource_budget(shape, MAX_BATCH_T, cols=cols).as_dict()
        for cols in (7, 8)
        for shape in PREFILL_SHAPE_NAMES
    }


def run(args):
    plate8, _ = _compile_plate(8, None)
    plate7, rda = _compile_plate(7, 7)
    decode = UnifiedGEMV.for_shape("rda_in", cols=7, rda_col=7)
    decode.compile()
    reports = []
    reports.append(_artifact_rows("prefill-8col", plate8[0].xclbin_artifact))
    reports.append(_artifact_rows("prefill-7col-rda", plate7[0].xclbin_artifact))
    reports.append(_artifact_rows("decode-unified", decode.xclbin_artifact))
    budgets = _budgets()
    alternate = _alternate(args, plate7[0], rda)
    link_audit_passed = _link_audit_passed(reports)
    return {
        "schema": "raon-prefill-prototype-s2a-v1",
        "artifacts": reports,
        "budgets": budgets,
        "stage2_estimates": _stage2_estimates(reports),
        "alternating": alternate,
        "link_audit_passed": link_audit_passed,
        "passed": alternate["passed"] and link_audit_passed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir")
    parser.add_argument("--rda-layer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
