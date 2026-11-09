# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Prefill replay comparison harness for byte-level verification."""

from __future__ import annotations

from collections.abc import Collection, Sequence
import time
from typing import Any

import numpy as np

from host.raon import RaonModel
from host.weights import bf16_to_f32
from serve.prefill_plan import prefill_segments


def _copy_conv_states(conv_states: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
    copied: dict[int, dict[str, np.ndarray]] = {}
    for layer, d in conv_states.items():
        copied[layer] = {k: np.array(v, copy=True) for k, v in d.items()}
    return copied


def _compare_arrays(
    name: str,
    arr_a: np.ndarray | None,
    arr_b: np.ndarray | None,
    *,
    layer: int | None = None,
    token: int | None = None,
) -> dict[str, Any] | None:
    if arr_a is None and arr_b is None:
        return None
    if arr_a is None or arr_b is None:
        return {
            "item": name,
            "layer": layer,
            "token": token,
            "reason": "one array is None",
            "max_abs_diff": float("inf"),
            "mismatched_elements": -1,
        }
    if arr_a.shape != arr_b.shape:
        return {
            "item": name,
            "layer": layer,
            "token": token,
            "reason": f"shape mismatch ({arr_a.shape} vs {arr_b.shape})",
            "max_abs_diff": float("inf"),
            "mismatched_elements": -1,
        }
    if arr_a.dtype != arr_b.dtype:
        return {
            "item": name,
            "layer": layer,
            "token": token,
            "reason": f"dtype mismatch ({arr_a.dtype} vs {arr_b.dtype})",
            "max_abs_diff": float("inf"),
            "mismatched_elements": -1,
        }
    if arr_a.tobytes() == arr_b.tobytes():
        return None

    # Calculate numeric difference for diagnostics
    if arr_a.dtype == np.uint16:
        a_f32 = bf16_to_f32(arr_a)
        b_f32 = bf16_to_f32(arr_b)
    else:
        a_f32 = arr_a.astype(np.float32)
        b_f32 = arr_b.astype(np.float32)

    diff = np.abs(a_f32 - b_f32)
    max_abs_diff = float(np.max(diff))
    mismatched = int(np.count_nonzero(arr_a != arr_b))

    # Find first token index with mismatch if leading dimension is token
    first_tok = token
    if first_tok is None and arr_a.ndim >= 1 and arr_a.shape[0] > 0:
        for t in range(arr_a.shape[0]):
            if arr_a[t].tobytes() != arr_b[t].tobytes():
                first_tok = t
                break

    return {
        "item": name,
        "layer": layer,
        "token": first_tok,
        "max_abs_diff": max_abs_diff,
        "mismatched_elements": mismatched,
    }


def run_prefill_replay(
    model: RaonModel,
    token_ids: Sequence[int],
    *,
    chunk: int = 16,
    checkpoint_len: int | None = None,
    logits_positions: Collection[int] | None = None,
) -> dict[str, Any]:
    """Execute token sequence through Path A (advance) and Path B (prefill), comparing bit-for-bit.

    Returns a JSON-serializable dictionary with match boolean, timings, and first mismatch details.
    """
    total_tokens = len(token_ids)
    if total_tokens == 0:
        return {
            "match": True,
            "tokens": 0,
            "chunk": chunk,
            "checkpoint_len": checkpoint_len,
            "time_a_ms": 0.0,
            "time_b_ms": 0.0,
            "ms_per_token_a": 0.0,
            "ms_per_token_b": 0.0,
            "speedup": 1.0,
            "first_mismatch": None,
        }

    cp_len = checkpoint_len if checkpoint_len is not None else total_tokens
    if logits_positions is None:
        target_positions = set()
        if 0 < cp_len <= total_tokens:
            target_positions.add(cp_len - 1)
        if total_tokens > 0:
            target_positions.add(total_tokens - 1)
    else:
        target_positions = set(logits_positions)

    num_layers = model.config.num_hidden_layers

    # =========================================================================
    # Path A: Token-serial advance()
    # =========================================================================
    model.reset_state()
    model.backend.reset()

    a_streams_in: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    a_streams_out: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    a_router_top: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    a_router_weights: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    a_logits: dict[int, np.ndarray] = {}

    t_a0 = time.perf_counter()
    for i, tid in enumerate(token_ids):
        if hasattr(model.backend, "set_step"):
            model.backend.set_step(i + 1)
        need_logits = (i in target_positions)
        lg = model.advance(int(tid), need_logits=need_logits)
        if need_logits and lg is not None:
            a_logits[i] = np.array(lg, copy=True)

        for idx in range(num_layers):
            pfx = f"layer_{idx:03d}"
            sin = model.last_intermediates.get(f"{pfx}.streams_in")
            if sin is not None:
                a_streams_in[idx].append(sin[0, 0].copy())
            sout = model.last_intermediates.get(f"{pfx}.streams_out")
            if sout is not None:
                a_streams_out[idx].append(sout[0, 0].copy())
            if model.config.is_moe_layer(idx):
                rtop = model.last_intermediates.get(f"{pfx}.router_top")
                if rtop is not None:
                    a_router_top[idx].append(rtop[0].copy())
                rw = model.last_intermediates.get(f"{pfx}.router_weights")
                if rw is not None:
                    a_router_weights[idx].append(rw[0].copy())

    t_a = time.perf_counter() - t_a0

    a_conv_states = _copy_conv_states(model.conv_states)
    a_rda_states = {idx: model.backend.get_rda_state(idx) for idx in range(num_layers)}
    a_mla_caches = {idx: model.backend.get_mla_cache(idx) for idx in range(num_layers)}

    # =========================================================================
    # Path B: Batched prefill() with server chunk boundary rules
    # =========================================================================
    model.reset_state()
    model.backend.reset()

    b_streams_in: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    b_streams_out: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    b_router_top: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    b_router_weights: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_layers)}
    b_logits: dict[int, np.ndarray] = {}
    p_idx = 0
    t_b0 = time.perf_counter()

    # Pre-determine chunks according to server rules
    segments = prefill_segments(
        cached=0,
        n=total_tokens,
        chunk=chunk,
        checkpoint_len=cp_len,
        prompt_tokens=total_tokens,
        target_logits_positions=target_positions,
    )
    p_idx = 0
    t_b0 = time.perf_counter()

    for step_size, chunk_logits_positions in segments:
        chunk_tokens = [int(x) for x in token_ids[p_idx : p_idx + step_size]]

        if hasattr(model.backend, "set_step"):
            model.backend.set_step(p_idx + 1)

        logits_map = model.prefill(
            chunk_tokens,
            logits_positions=chunk_logits_positions,
            chunk=step_size,
            record_intermediates=True,
        )

        for t_local, lg in logits_map.items():
            b_logits[p_idx + t_local] = np.array(lg, copy=True)

        for idx in range(num_layers):
            pfx = f"layer_{idx:03d}"
            sin = model.last_intermediates.get(f"{pfx}.streams_in")
            if sin is not None:
                b_streams_in[idx].append(sin.copy())
            sout = model.last_intermediates.get(f"{pfx}.streams_out")
            if sout is not None:
                b_streams_out[idx].append(sout.copy())
            if model.config.is_moe_layer(idx):
                rtop = model.last_intermediates.get(f"{pfx}.router_top")
                if rtop is not None:
                    b_router_top[idx].append(rtop.copy())
                rw = model.last_intermediates.get(f"{pfx}.router_weights")
                if rw is not None:
                    b_router_weights[idx].append(rw.copy())

        p_idx += step_size

    t_b = time.perf_counter() - t_b0

    b_conv_states = _copy_conv_states(model.conv_states)
    b_rda_states = {idx: model.backend.get_rda_state(idx) for idx in range(num_layers)}
    b_mla_caches = {idx: model.backend.get_mla_cache(idx) for idx in range(num_layers)}

    # =========================================================================
    # Byte-by-byte Comparison
    # =========================================================================
    first_mismatch: dict[str, Any] | None = None

    # 1. streams_in & streams_out
    for idx in range(num_layers):
        arr_a_in = np.stack(a_streams_in[idx], axis=0) if a_streams_in[idx] else None
        arr_b_in = np.concatenate(b_streams_in[idx], axis=0) if b_streams_in[idx] else None
        mismatch = _compare_arrays(f"layer_{idx}.streams_in", arr_a_in, arr_b_in, layer=idx)
        if mismatch and first_mismatch is None:
            first_mismatch = mismatch

        arr_a_out = np.stack(a_streams_out[idx], axis=0) if a_streams_out[idx] else None
        arr_b_out = np.concatenate(b_streams_out[idx], axis=0) if b_streams_out[idx] else None
        mismatch = _compare_arrays(f"layer_{idx}.streams_out", arr_a_out, arr_b_out, layer=idx)
        if mismatch and first_mismatch is None:
            first_mismatch = mismatch

        # 2. Router top and weights
        if model.config.is_moe_layer(idx):
            arr_a_top = np.stack(a_router_top[idx], axis=0) if a_router_top[idx] else None
            arr_b_top = np.concatenate(b_router_top[idx], axis=0) if b_router_top[idx] else None
            mismatch = _compare_arrays(f"layer_{idx}.router_top", arr_a_top, arr_b_top, layer=idx)
            if mismatch and first_mismatch is None:
                first_mismatch = mismatch

            arr_a_rw = np.stack(a_router_weights[idx], axis=0) if a_router_weights[idx] else None
            arr_b_rw = np.concatenate(b_router_weights[idx], axis=0) if b_router_weights[idx] else None
            mismatch = _compare_arrays(f"layer_{idx}.router_weights", arr_a_rw, arr_b_rw, layer=idx)
            if mismatch and first_mismatch is None:
                first_mismatch = mismatch

    # 3. Logits at selected positions
    for pos in sorted(target_positions):
        lg_a = a_logits.get(pos)
        lg_b = b_logits.get(pos)
        mismatch = _compare_arrays(f"logits_pos_{pos}", lg_a, lg_b, token=pos)
        if mismatch and first_mismatch is None:
            first_mismatch = mismatch

    # 4. Final Conv States
    for idx in range(num_layers):
        ca = a_conv_states.get(idx, {})
        cb = b_conv_states.get(idx, {})
        for kind in ("q", "k", "v"):
            mismatch = _compare_arrays(
                f"layer_{idx}.conv_{kind}", ca.get(kind), cb.get(kind), layer=idx
            )
            if mismatch and first_mismatch is None:
                first_mismatch = mismatch

    # 5. Final RDA Recurrent States
    for idx in range(num_layers):
        if model.config.is_rda_layer(idx):
            mismatch = _compare_arrays(
                f"layer_{idx}.rda_state", a_rda_states.get(idx), b_rda_states.get(idx), layer=idx
            )
            if mismatch and first_mismatch is None:
                first_mismatch = mismatch

    # 6. Final MLA Caches
    for idx in range(num_layers):
        if not model.config.is_rda_layer(idx):
            ka, va = a_mla_caches.get(idx, (None, None))
            kb, vb = b_mla_caches.get(idx, (None, None))
            mismatch_k = _compare_arrays(f"layer_{idx}.mla_key_cache", ka, kb, layer=idx)
            if mismatch_k and first_mismatch is None:
                first_mismatch = mismatch_k
            mismatch_v = _compare_arrays(f"layer_{idx}.mla_val_cache", va, vb, layer=idx)
            if mismatch_v and first_mismatch is None:
                first_mismatch = mismatch_v

    ms_a = t_a * 1000.0
    ms_b = t_b * 1000.0
    ms_per_tok_a = ms_a / total_tokens if total_tokens > 0 else 0.0
    ms_per_tok_b = ms_b / total_tokens if total_tokens > 0 else 0.0
    speedup = (ms_a / ms_b) if ms_b > 0 else 1.0

    return {
        "match": first_mismatch is None,
        "tokens": total_tokens,
        "chunk": chunk,
        "checkpoint_len": checkpoint_len,
        "time_a_ms": ms_a,
        "time_b_ms": ms_b,
        "ms_per_token_a": ms_per_tok_a,
        "ms_per_token_b": ms_per_tok_b,
        "speedup": speedup,
        "first_mismatch": first_mismatch,
    }


def measure_timing_repeats(
    model: RaonModel,
    token_ids: Sequence[int],
    repeats: int,
    *,
    chunk: int = 16,
    checkpoint_len: int | None = None,
    logits_positions: Collection[int] | None = None,
) -> dict[str, Any]:
    """Execute Path A (advance) and Path B (prefill) alternately for N pairs without recording intermediates.

    Each run resets model and backend state beforehand.
    Pair 1 is marked as warmup (준비 실행) and excluded from the summary (median).
    Returns list of pair timings and summary statistics.
    """
    total_tokens = len(token_ids)
    if total_tokens == 0 or repeats <= 0:
        return {"pairs": [], "summary": None}

    cp_len = checkpoint_len if checkpoint_len is not None else total_tokens
    if logits_positions is None:
        target_positions = set()
        if 0 < cp_len <= total_tokens:
            target_positions.add(cp_len - 1)
        if total_tokens > 0:
            target_positions.add(total_tokens - 1)
    else:
        target_positions = set(logits_positions)

    pairs: list[dict[str, Any]] = []

    segments = prefill_segments(
        cached=0,
        n=total_tokens,
        chunk=chunk,
        checkpoint_len=cp_len,
        prompt_tokens=total_tokens,
        target_logits_positions=target_positions,
    )

    for pair_idx in range(1, repeats + 1):
        is_warmup = (pair_idx == 1)

        # Path A: Token-serial advance without recording intermediates
        model.reset_state()
        model.backend.reset()
        t_a0 = time.perf_counter()
        for i, tid in enumerate(token_ids):
            if hasattr(model.backend, "set_step"):
                model.backend.set_step(i + 1)
            need_logits = (i in target_positions)
            model.advance(int(tid), need_logits=need_logits)
        t_a = time.perf_counter() - t_a0
        ms_per_tok_a = (t_a * 1000.0) / total_tokens

        # Path B: Batched prefill without recording intermediates
        model.reset_state()
        model.backend.reset()
        p_idx = 0
        t_b0 = time.perf_counter()
        for step_size, chunk_logits_positions in segments:
            chunk_tokens = [int(x) for x in token_ids[p_idx : p_idx + step_size]]
            if hasattr(model.backend, "set_step"):
                model.backend.set_step(p_idx + 1)
            model.prefill(
                chunk_tokens,
                logits_positions=chunk_logits_positions,
                chunk=step_size,
                record_intermediates=False,
            )
            p_idx += step_size
        t_b = time.perf_counter() - t_b0
        ms_per_tok_b = (t_b * 1000.0) / total_tokens

        speedup = (ms_per_tok_a / ms_per_tok_b) if ms_per_tok_b > 0 else 1.0

        pair_info = {
            "pair": pair_idx,
            "warmup": is_warmup,
            "time_a_ms": t_a * 1000.0,
            "time_b_ms": t_b * 1000.0,
            "ms_per_token_a": ms_per_tok_a,
            "ms_per_token_b": ms_per_tok_b,
            "speedup": speedup,
        }
        pairs.append(pair_info)

    # Summary: exclude warmup (pair 1)
    non_warmup_pairs = [p for p in pairs if not p["warmup"]]
    if non_warmup_pairs:
        med_a = float(np.median([p["ms_per_token_a"] for p in non_warmup_pairs]))
        med_b = float(np.median([p["ms_per_token_b"] for p in non_warmup_pairs]))
        med_speedup = float(np.median([p["speedup"] for p in non_warmup_pairs]))
        summary = {
            "count": len(non_warmup_pairs),
            "median_ms_per_token_a": med_a,
            "median_ms_per_token_b": med_b,
            "median_speedup": med_speedup,
        }
    else:
        summary = {
            "count": 0,
            "note": "Only warmup pair executed; no non-warmup iterations for median summary",
            "median_ms_per_token_a": pairs[0]["ms_per_token_a"],
            "median_ms_per_token_b": pairs[0]["ms_per_token_b"],
            "median_speedup": pairs[0]["speedup"],
        }

    return {
        "pairs": pairs,
        "summary": summary,
    }
