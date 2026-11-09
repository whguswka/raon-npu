# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon NPU computation backend implementing the host Backend interface.

Provides:
- Manifest-driven validation and pre-allocation of NPU weight tensors and operator reuse.
- DWeights adapter decoding exported D-mode bundles (q4nx / bfp16) for host NumPy parity.
- Bundled GEMV execution with input-identity and step-aware result slicing (1D and 2D inputs).
- Double-buffered (ping-pong) NPU recurrent state for token-serial RDA update.
- MoE execution: B1-6 -> shared gate+up -> unweighted RTNE(SiTU-GLU) -> B2-9 -> weighted FP32 sum in ascending expert ID order.
- Native NPU acceleration when IRON / AIE is available, with pure Python/NumPy fallback for testing.

GEMV Engine Selection:
- Configured via environment variable `RAON_NPU_GEMV` ('unified' [default] or 'legacy') or `run_golden.py --gemv {unified,legacy}`.
- 2026-10-31 laptop official golden (25 layers, 15 steps): unified (merged 7-column GEMV + column-7 RDA) 200.7 / 198.7 ms/token with 0 xclbin switches per token (steps 3-16: 245.7 -> 198.9 versus the separate-RDA engine); legacy 583.1 ms/token with 182 switches; identical golden metrics.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import gc
import hashlib
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Sequence
import weakref

import numpy as np

# Import host backend contract and utilities
from host.backend import Backend, situ_glu, _MlaLayerCache, mla_context
from host.strict_json import loads_strict
from host.weights import (
    F32_CACHE_MAX_TENSOR_BYTES,
    SafeTensorsReader,
    Weights,
    bf16_to_f32,
    f32_to_bf16_rtne,
)
from quant.bfp16 import parse_bfp16_gemv
from quant.q4nx import dequantize_simple, parse_gemv
from kernels.gemv_unified.shapes import MAX_K, get_shape
from kernels.gemv_unified.stage2 import pad_x
from kernels.moe_grouped.layout import B1 as GROUPED_B1, B2 as GROUPED_B2, K_TILE as GROUPED_K_TILE, SLOTS, TMAX
from bench.rda_state.rda_state_multi import RDA_TOKEN_CHOICES, check_rda_tokens
from kernels.moe_grouped.planner import (
    accumulate_call,
    build_plan,
    gather_call,
    gather_inputs,
    make_b2_inputs,
    scatter_b1,
    scatter_b2,
    shared_situ,
    situ_call,
)


DWEIGHTS_F32_CACHE_MAX_TENSOR_BYTES = F32_CACHE_MAX_TENSOR_BYTES

# Try importing AIE / IRON runtime
try:
    import aie.utils as aie_utils
    from iron.common import AIEContext
    import aie.iron as iron
    HAVE_NPU = True
except (ImportError, ModuleNotFoundError):
    aie_utils = None
    AIEContext = None
    iron = None
    HAVE_NPU = False

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    # Fallback 2-byte float dtype so uint16 assignment without .view(bfloat16)
    # performs numeric conversion in tests just like ml_dtypes.bfloat16.
    bfloat16 = np.float16


# -----------------------------------------------------------------------------
# Component and Bundle Metadata
# -----------------------------------------------------------------------------
@dataclass
class ComponentMeta:
    hf_name: str
    bundle_name: str
    row_start: int
    row_end: int
    orig_k: int
    padded_k: int
    orig_m: int
    padded_m: int
    fmt: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BundleMeta:
    name: str
    rel_file: str
    fmt: str  # 'bfp16_gemv', 'q4nx_gemv', 'q4nx_gemv_bundle', 'bfp16_gemv_bundle'
    orig_shape: list[int]
    padded_shape: list[int]
    k: int
    m: int
    components: list[dict[str, Any]]
    expert_bytes: int = 0
    expert_count: int = 0
    size_bytes: int = 0
    sha256: str = ""


# -----------------------------------------------------------------------------
# Unified GEMV Shape Resolution Mappings (survey section 2)
# -----------------------------------------------------------------------------
UNIFIED_SHAPE_BY_K_M_FMT: dict[tuple[int, int, str], str] = {
    (2560, 20736, "bfp16_gemv"): "rda_in",
    (256, 5120, "bfp16_gemv"): "f_b",
    (5120, 2560, "bfp16_gemv"): "o_proj",
    (2560, 1152, "bfp16_gemv"): "mla_in",
    (768, 7680, "bfp16_gemv"): "q_b",
    (512, 5120, "bfp16_gemv"): "k_b_v_b",
    (2560, 5120, "bfp16_gemv"): "g",
    (2560, 2176, "bfp16_gemv"): "shared_gu",
    (12032, 2560, "bfp16_gemv"): "dense_down",
    (2560, 260096, "bfp16_gemv"): "lm_head",
    (2560, 24064, "q4nx_gemv"): "dense_gu",
    (2560, 2176, "q4nx_gemv_bundle"): "moe_b1_6",
    (1280, 2560, "bfp16_gemv_bundle"): "moe_b2_9",
}

UNIFIED_SHAPE_BY_KIND: dict[str, str] = {
    "rda_in": "rda_in",
    "f_b": "f_b",
    "o_proj": "o_proj",
    "mla_in": "mla_in",
    "q_b": "q_b",
    "k_b": "k_b_v_b",
    "v_b": "k_b_v_b",
    "g": "g",
    "shared_gu": "shared_gu",
    "dense_down": "dense_down",
    "lm_head": "lm_head",
    "dense_gu": "dense_gu",
    "moe_b1": "moe_b1_6",
    "moe_b2": "moe_b2_9",
}

# Engine-wide switch: all unified GEMV and MoE operators use the same continuous W
# xclbin so decode does not pay extra xclbin context switches between shapes.
UNIFIED_CONTINUOUS_MEM_W = True
UNIFIED_SEQ_PLAN = "per_call"
UNIFIED_PER_CALL_SHAPES = frozenset(
    {
        "f_b",
        "q_b",
        "k_b_v_b",
        "dense_gu",
        "moe_b1_6",
        "moe_b2_9",
    }
)
UNIFIED_COLS = 7
UNIFIED_RDA_COL = 7

PREFILL_GEMM_MAX_T = 24  # rows per plate GEMM call = kernels.gemm_prefill.shapes.MAX_BATCH_T (DESIGN-prefill.md section 45)
PREFILL_GEMM_XCLBIN_ID = "gemm_prefill_s2a"
PREFILL_GEMM_SHAPES = frozenset(
    {"rda_in", "f_b", "o_proj", "mla_in", "q_b", "k_b_v_b", "g", "shared_gu", "dense_down"}
)


def unified_sequence_plan(shape_name: str) -> str:
    """Resolve the engine-wide rollback switch and per-shape opt-in set."""
    if UNIFIED_SEQ_PLAN not in ("per_round", "per_call"):
        raise ValueError(f"invalid UNIFIED_SEQ_PLAN {UNIFIED_SEQ_PLAN!r}")
    if UNIFIED_SEQ_PLAN == "per_round":
        return "per_round"
    return "per_call" if shape_name in UNIFIED_PER_CALL_SHAPES else "per_round"


def resolve_unified_shape_name(k: int, m: int, fmt: str, bundle_name: str | None = None) -> str:
    """Resolve bundle dimensions and format to an ALL_SHAPES shape name.

    Shape resolution uses only (K, M, fmt).  If a bundle_name is supplied, its
    kind (the leaf after the last '/') must map to the same shape; otherwise a
    ValueError is raised so that mismatched dimensions cannot slip through via
    the bundle name.

    Raises ValueError if the bundle cannot be mapped to a unified GEMV shape.
    """
    key = (k, m, fmt)
    shape_name = UNIFIED_SHAPE_BY_K_M_FMT.get(key)
    if shape_name is None:
        raise ValueError(
            f"No unified GEMV shape matching bundle (K={k}, M={m}, format={fmt!r}, bundle={bundle_name!r})"
        )
    if bundle_name is not None:
        kind = bundle_name.split("/")[-1]
        expected_by_kind = UNIFIED_SHAPE_BY_KIND.get(kind)
        if expected_by_kind is None:
            raise ValueError(
                f"Unknown bundle kind '{kind}' for {bundle_name!r} "
                f"(K={k}, M={m}, format={fmt!r}); resolved shape {shape_name!r} from (K,M,fmt)"
            )
        if expected_by_kind != shape_name:
            raise ValueError(
                f"Bundle kind '{kind}' ({bundle_name!r}) maps to shape {expected_by_kind!r}, "
                f"but (K={k}, M={m}, format={fmt!r}) resolves to {shape_name!r}"
            )
    # Validate against ALL_SHAPES registry
    get_shape(shape_name)
    return shape_name


def _xclbin_path_from_op(op: Any) -> Path:
    """Return the resolved xclbin path for a compiled UnifiedGEMV/MoE operator."""
    art = getattr(op, "xclbin_artifact", None)
    if art is None:
        raise RuntimeError(f"operator {op!r} has no xclbin_artifact")
    filename = getattr(art, "filename", None)
    if not filename:
        raise RuntimeError(f"operator {op!r} has no xclbin filename")
    return Path(filename).resolve()


def _require_single_unified_xclbin(
    ops: Iterable[Any],
    expected: Path | None = None,
) -> Path:
    """Verify every compiled unified operator resolves to the same xclbin path.

    Raises RuntimeError with per-operator paths if any operator disagrees.
    """
    labels: list[str] = []
    paths: list[Path] = []
    for op in ops:
        try:
            path = _xclbin_path_from_op(op)
        except RuntimeError as exc:
            raise RuntimeError(f"unified xclbin check failed: {exc}") from exc
        paths.append(path)
        name = getattr(op, "name", repr(op))
        labels.append(f"  {name}: {path}")
    if not paths:
        raise RuntimeError("unified xclbin check received no operators")
    first = paths[0]
    if expected is not None and first != expected:
        raise RuntimeError(
            "unified xclbin path mismatch:\n" + "\n".join(labels) +
            f"\n  expected: {expected}"
        )
    for idx, p in enumerate(paths[1:], start=1):
        if p != first:
            raise RuntimeError(
                "unified xclbin path mismatch:\n" + "\n".join(labels) +
                f"\n  expected (from item 0): {first}"
            )
    return first


def legacy_gemv_xclbin_name(fmt: str, k: int, m: int) -> str:
    """Format canonical xclbin identifier for legacy per-shape GEMV operators."""
    return f"gemv_{fmt}_k{k}_m{m}"


# A successful digest is reusable across backend recreation as long as the
# resolved path and the file identity relevant to integrity have not changed.
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _bundle_path(weights_dir: Path, rel_file: str, bundle_name: str) -> Path:
    """Resolve a manifest file without allowing it to escape weights_dir."""
    rel = Path(rel_file)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe file path for {bundle_name}: {rel_file!r}")
    root = weights_dir.resolve()
    resolved = (root / rel).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest file escapes weights directory for {bundle_name}: {rel_file!r}"
        ) from exc
    return resolved


def _read_verified_bundle(weights_dir: Path, meta: BundleMeta) -> bytes:
    """Read a bundle once, checking its size and cached SHA-256 while reading."""
    path = _bundle_path(weights_dir, meta.rel_file, meta.name)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing D bundle binary: {path}") from exc
    if stat.st_size != meta.size_bytes:
        raise ValueError(
            f"Manifest size_bytes mismatch for {meta.name}: "
            f"file has {stat.st_size}, manifest has {meta.size_bytes}"
        )
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_SHA256_CACHE.get(key)
    data = path.read_bytes()
    digest = cached if cached is not None else hashlib.sha256(data).hexdigest()
    if digest != meta.sha256:
        raise ValueError(
            f"Manifest sha256 mismatch for {meta.name}: {digest} != {meta.sha256}"
        )
    if cached is None:
        _FILE_SHA256_CACHE[key] = digest
    return data


def _verify_bundle_file(weights_dir: Path, meta: BundleMeta) -> Path:
    """Verify a bundle without retaining it, streaming only on a cache miss."""
    path = _bundle_path(weights_dir, meta.rel_file, meta.name)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing D bundle binary: {path}") from exc
    if stat.st_size != meta.size_bytes:
        raise ValueError(
            f"Manifest size_bytes mismatch for {meta.name}: "
            f"file has {stat.st_size}, manifest has {meta.size_bytes}"
        )
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    digest = _FILE_SHA256_CACHE.get(key)
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    if digest != meta.sha256:
        raise ValueError(
            f"Manifest sha256 mismatch for {meta.name}: {digest} != {meta.sha256}"
        )
    _FILE_SHA256_CACHE[key] = digest
    return path


# Default RDA / MLA layer definitions matching Raon-1-mini preview
RDA_LAYERS = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22}
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 24}
HIDDEN_SIZE = 2560
NUM_ROUTED_EXPERTS = 128
NUM_TOTAL_EXPERTS = 129


def build_default_manifest_specs() -> tuple[dict[str, ComponentMeta], dict[str, BundleMeta]]:
    """Build canonical bundle and component specifications as defined by export_d.py."""
    comp_map: dict[str, ComponentMeta] = {}
    bundle_map: dict[str, BundleMeta] = {}

    # Layer 0 Dense MLP
    comp_map["model.layers.0.mlp.gate_proj.weight"] = ComponentMeta(
        hf_name="model.layers.0.mlp.gate_proj.weight",
        bundle_name="L00/dense_gu",
        row_start=0,
        row_end=12032,
        orig_k=2560,
        padded_k=2560,
        orig_m=12032,
        padded_m=24064,
        fmt="q4nx_gemv",
    )
    comp_map["model.layers.0.mlp.up_proj.weight"] = ComponentMeta(
        hf_name="model.layers.0.mlp.up_proj.weight",
        bundle_name="L00/dense_gu",
        row_start=12032,
        row_end=24064,
        orig_k=2560,
        padded_k=2560,
        orig_m=12032,
        padded_m=24064,
        fmt="q4nx_gemv",
    )
    bundle_map["L00/dense_gu"] = BundleMeta(
        name="L00/dense_gu",
        rel_file="L00/dense_gu.bin",
        fmt="q4nx_gemv",
        orig_shape=[24064, 2560],
        padded_shape=[24064, 2560],
        k=2560,
        m=24064,
        components=[],
    )

    comp_map["model.layers.0.mlp.down_proj.weight"] = ComponentMeta(
        hf_name="model.layers.0.mlp.down_proj.weight",
        bundle_name="L00/dense_down",
        row_start=0,
        row_end=2560,
        orig_k=12032,
        padded_k=12032,
        orig_m=2560,
        padded_m=2560,
        fmt="bfp16_gemv",
    )
    bundle_map["L00/dense_down"] = BundleMeta(
        name="L00/dense_down",
        rel_file="L00/dense_down.bin",
        fmt="bfp16_gemv",
        orig_shape=[2560, 12032],
        padded_shape=[2560, 12032],
        k=12032,
        m=2560,
        components=[],
    )

    for layer in range(25):
        lpfx = f"model.layers.{layer}"

        # 1. Attention site
        if layer in RDA_LAYERS:
            bname = f"L{layer:02d}/rda_in"
            bundle_map[bname] = BundleMeta(
                name=bname,
                rel_file=f"L{layer:02d}/rda_in.bin",
                fmt="bfp16_gemv",
                orig_shape=[20688, 2560],
                padded_shape=[20736, 2560],
                k=2560,
                m=20736,
                components=[],
            )
            t_specs = [
                ("q_proj.weight", 0, 5120),
                ("k_proj.weight", 5120, 10240),
                ("v_proj.weight", 10240, 15360),
                ("g_proj.weight", 15360, 20480),
                ("f_a_proj.weight", 20480, 20608),
                ("b_proj.weight", 20608, 20648),
                ("rda_w_proj.weight", 20648, 20688),
            ]
            for sname, r0, r1 in t_specs:
                hfname = f"{lpfx}.self_attn.{sname}"
                comp_map[hfname] = ComponentMeta(
                    hf_name=hfname,
                    bundle_name=bname,
                    row_start=r0,
                    row_end=r1,
                    orig_k=2560,
                    padded_k=2560,
                    orig_m=r1 - r0,
                    padded_m=20736,
                    fmt="bfp16_gemv",
                )

            # f_b projection (K padded 128 -> 256)
            fb_name = f"L{layer:02d}/f_b"
            bundle_map[fb_name] = BundleMeta(
                name=fb_name,
                rel_file=f"L{layer:02d}/f_b.bin",
                fmt="bfp16_gemv",
                orig_shape=[5120, 128],
                padded_shape=[5120, 256],
                k=256,
                m=5120,
                components=[],
            )
            comp_map[f"{lpfx}.self_attn.f_b_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.f_b_proj.weight",
                bundle_name=fb_name,
                row_start=0,
                row_end=5120,
                orig_k=128,
                padded_k=256,
                orig_m=5120,
                padded_m=5120,
                fmt="bfp16_gemv",
            )
        else:
            # MLA layer
            min_name = f"L{layer:02d}/mla_in"
            bundle_map[min_name] = BundleMeta(
                name=min_name,
                rel_file=f"L{layer:02d}/mla_in.bin",
                fmt="bfp16_gemv",
                orig_shape=[1152, 2560],
                padded_shape=[1152, 2560],
                k=2560,
                m=1152,
                components=[],
            )
            comp_map[f"{lpfx}.self_attn.q_a_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.q_a_proj.weight",
                bundle_name=min_name,
                row_start=0,
                row_end=576,
                orig_k=2560,
                padded_k=2560,
                orig_m=576,
                padded_m=1152,
                fmt="bfp16_gemv",
            )
            comp_map[f"{lpfx}.self_attn.kv_a_proj_with_mqa.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.kv_a_proj_with_mqa.weight",
                bundle_name=min_name,
                row_start=576,
                row_end=1152,
                orig_k=2560,
                padded_k=2560,
                orig_m=576,
                padded_m=1152,
                fmt="bfp16_gemv",
            )

            # q_b projection (K padded 576 -> 768)
            qb_name = f"L{layer:02d}/q_b"
            bundle_map[qb_name] = BundleMeta(
                name=qb_name,
                rel_file=f"L{layer:02d}/q_b.bin",
                fmt="bfp16_gemv",
                orig_shape=[7680, 576],
                padded_shape=[7680, 768],
                k=768,
                m=7680,
                components=[],
            )
            comp_map[f"{lpfx}.self_attn.q_b_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.q_b_proj.weight",
                bundle_name=qb_name,
                row_start=0,
                row_end=7680,
                orig_k=576,
                padded_k=768,
                orig_m=7680,
                padded_m=7680,
                fmt="bfp16_gemv",
            )

            # k_b and v_b (both K=512, M=5120)
            kb_name = f"L{layer:02d}/k_b"
            vb_name = f"L{layer:02d}/v_b"
            bundle_map[kb_name] = BundleMeta(
                name=kb_name,
                rel_file=f"L{layer:02d}/k_b.bin",
                fmt="bfp16_gemv",
                orig_shape=[5120, 512],
                padded_shape=[5120, 512],
                k=512,
                m=5120,
                components=[],
            )
            bundle_map[vb_name] = BundleMeta(
                name=vb_name,
                rel_file=f"L{layer:02d}/v_b.bin",
                fmt="bfp16_gemv",
                orig_shape=[5120, 512],
                padded_shape=[5120, 512],
                k=512,
                m=5120,
                components=[],
            )
            # HF kv_b_proj.weight maps to combined k_b + v_b
            comp_map[f"{lpfx}.self_attn.kv_b_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.kv_b_proj.weight",
                bundle_name=f"L{layer:02d}/kv_b_combined",
                row_start=0,
                row_end=10240,
                orig_k=512,
                padded_k=512,
                orig_m=10240,
                padded_m=10240,
                fmt="mla_kv_b",
                extra={"kb_bundle": kb_name, "vb_bundle": vb_name},
            )

            # g_proj
            g_name = f"L{layer:02d}/g"
            bundle_map[g_name] = BundleMeta(
                name=g_name,
                rel_file=f"L{layer:02d}/g.bin",
                fmt="bfp16_gemv",
                orig_shape=[5120, 2560],
                padded_shape=[5120, 2560],
                k=2560,
                m=5120,
                components=[],
            )
            comp_map[f"{lpfx}.self_attn.g_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.self_attn.g_proj.weight",
                bundle_name=g_name,
                row_start=0,
                row_end=5120,
                orig_k=2560,
                padded_k=2560,
                orig_m=5120,
                padded_m=5120,
                fmt="bfp16_gemv",
            )

        # Output projection for both RDA and MLA
        o_name = f"L{layer:02d}/o_proj"
        bundle_map[o_name] = BundleMeta(
            name=o_name,
            rel_file=f"L{layer:02d}/o_proj.bin",
            fmt="bfp16_gemv",
            orig_shape=[2560, 5120],
            padded_shape=[2560, 5120],
            k=5120,
            m=2560,
            components=[],
        )
        comp_map[f"{lpfx}.self_attn.o_proj.weight"] = ComponentMeta(
            hf_name=f"{lpfx}.self_attn.o_proj.weight",
            bundle_name=o_name,
            row_start=0,
            row_end=2560,
            orig_k=5120,
            padded_k=5120,
            orig_m=2560,
            padded_m=2560,
            fmt="bfp16_gemv",
        )

        # 2. MLP site (Layers 1..24 MoE)
        if layer >= 1:
            # MoE B1
            b1_name = f"L{layer:02d}/moe_b1"
            bundle_map[b1_name] = BundleMeta(
                name=b1_name,
                rel_file=f"L{layer:02d}/moe_b1.bin",
                fmt="q4nx_gemv_bundle",
                orig_shape=[NUM_ROUTED_EXPERTS, 2176, 2560],
                padded_shape=[NUM_ROUTED_EXPERTS, 2176, 2560],
                k=2560,
                m=2176,
                expert_bytes=3481600,
                expert_count=NUM_ROUTED_EXPERTS,
                components=[],
            )

            # MoE B2
            b2_name = f"L{layer:02d}/moe_b2"
            bundle_map[b2_name] = BundleMeta(
                name=b2_name,
                rel_file=f"L{layer:02d}/moe_b2.bin",
                fmt="bfp16_gemv_bundle",
                orig_shape=[NUM_TOTAL_EXPERTS, 2560, 1088],
                padded_shape=[NUM_TOTAL_EXPERTS, 2560, 1280],
                k=1280,
                m=2560,
                expert_bytes=3686400,
                expert_count=NUM_TOTAL_EXPERTS,
                components=[],
            )

            # MoE shared gate+up
            sh_gu_name = f"L{layer:02d}/shared_gu"
            bundle_map[sh_gu_name] = BundleMeta(
                name=sh_gu_name,
                rel_file=f"L{layer:02d}/shared_gu.bin",
                fmt="bfp16_gemv",
                orig_shape=[2176, 2560],
                padded_shape=[2176, 2560],
                k=2560,
                m=2176,
                components=[],
            )
            comp_map[f"{lpfx}.moe.shared_experts.gate_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.moe.shared_experts.gate_proj.weight",
                bundle_name=sh_gu_name,
                row_start=0,
                row_end=1088,
                orig_k=2560,
                padded_k=2560,
                orig_m=1088,
                padded_m=2176,
                fmt="bfp16_gemv",
            )
            comp_map[f"{lpfx}.moe.shared_experts.up_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.moe.shared_experts.up_proj.weight",
                bundle_name=sh_gu_name,
                row_start=1088,
                row_end=2176,
                orig_k=2560,
                padded_k=2560,
                orig_m=1088,
                padded_m=2176,
                fmt="bfp16_gemv",
            )
            # Shared down projection maps to expert index 128 of moe_b2
            comp_map[f"{lpfx}.moe.shared_experts.down_proj.weight"] = ComponentMeta(
                hf_name=f"{lpfx}.moe.shared_experts.down_proj.weight",
                bundle_name=b2_name,
                row_start=0,
                row_end=2560,
                orig_k=1088,
                padded_k=1280,
                orig_m=2560,
                padded_m=2560,
                fmt="shared_down",
                extra={"expert_id": 128},
            )

    # 3. LM Head
    comp_map["lm_head.weight"] = ComponentMeta(
        hf_name="lm_head.weight",
        bundle_name="lm_head",
        row_start=0,
        row_end=259840,
        orig_k=2560,
        padded_k=2560,
        orig_m=259840,
        padded_m=260096,
        fmt="bfp16_gemv",
    )
    bundle_map["lm_head"] = BundleMeta(
        name="lm_head",
        rel_file="lm_head.bin",
        fmt="bfp16_gemv",
        orig_shape=[259840, 2560],
        padded_shape=[260096, 2560],
        k=2560,
        m=260096,
        components=[],
    )

    return comp_map, bundle_map


def _canonical_bundle_size_bytes(meta: BundleMeta) -> tuple[int, int]:
    """Return (whole bundle bytes, bytes per matrix) from canonical geometry."""
    from quant import bfp16, q4nx

    if len(meta.padded_shape) < 2:
        raise ValueError(f"Canonical padded_shape is not a matrix for {meta.name}")
    rows, cols = meta.padded_shape[-2:]
    if meta.fmt in {"q4nx_gemv", "q4nx_gemv_bundle"}:
        if rows % q4nx.M_TILE or cols % q4nx.K_TILE:
            raise ValueError(f"Canonical q4nx shape is not tile-aligned for {meta.name}")
        matrix_bytes = (
            (rows // q4nx.M_TILE)
            * (cols // q4nx.K_TILE)
            * q4nx.BLOCK_BYTES
        )
    elif meta.fmt in {"bfp16_gemv", "bfp16_gemv_bundle"}:
        if cols % bfp16.BLOCK:
            raise ValueError(f"Canonical bfp16 shape is not block-aligned for {meta.name}")
        matrix_bytes = rows * (cols // bfp16.BLOCK) * bfp16.BLOCK_BYTES
    else:
        raise ValueError(f"Unknown canonical bundle format for {meta.name}: {meta.fmt!r}")

    bundle_bytes = matrix_bytes * meta.expert_count if meta.expert_count else matrix_bytes
    return bundle_bytes, matrix_bytes


def _canonical_component_names(
    comp_map: dict[str, ComponentMeta],
    meta: BundleMeta,
) -> tuple[set[str], set[str]]:
    """Return required and allowed manifest component names for a canonical bundle."""
    required = {
        component.hf_name
        for component in comp_map.values()
        if component.bundle_name == meta.name
    }
    allowed = set(required)
    allowed.add("padding")

    layer_dir, _, leaf = meta.name.partition("/")
    if layer_dir.startswith("L") and layer_dir[1:].isdigit():
        layer = int(layer_dir[1:])
        prefix = f"model.layers.{layer}"
        if leaf == "moe_b1" and meta.expert_count:
            allowed.add(
                f"{prefix}.moe.experts.0..{meta.expert_count - 1}.(gate|up)_proj.weight"
            )
        elif leaf == "moe_b2" and meta.expert_count:
            allowed.add(
                f"{prefix}.moe.experts.0..{meta.expert_count - 2}.down_proj.weight"
            )
        elif leaf == "k_b":
            allowed.add(f"{prefix}.self_attn.kv_b_proj.weight (key_part)")
        elif leaf == "v_b":
            allowed.add(f"{prefix}.self_attn.kv_b_proj.weight (val_part)")
    return required, allowed


def _load_manifest_specs(
    path: Path,
    comp_map: dict[str, ComponentMeta],
    bundle_map: dict[str, BundleMeta],
    *,
    weights_dir: Path | None = None,
    require_complete: bool = True,
) -> None:
    """Validate manifest.json against canonical specs and update bundle file paths."""
    payload = loads_strict(path.read_bytes())

    if isinstance(payload, dict):
        if "bundles" not in payload or not isinstance(payload["bundles"], list):
            raise ValueError(f"Manifest dict missing 'bundles' list: {path}")
        entries = payload["bundles"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Invalid manifest top-level structure in {path}")

    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid manifest bundle entry: {item!r}")
        fmt = item.get("format")
        if fmt == "host_safetensors":
            continue
        bname = item.get("bundle")
        meta = bundle_map.get(bname) if bname else None
        if meta is None:
            raise ValueError(f"Unknown NPU bundle in manifest: {bname!r}")
        if bname in seen:
            raise ValueError(f"Duplicate NPU bundle in manifest: {bname}")
        seen.add(bname)
        if fmt != meta.fmt:
            raise ValueError(
                f"Manifest format mismatch for {meta.name}: {fmt!r} != {meta.fmt!r}"
            )
        padded_shape = item.get("padded_shape")
        if padded_shape is None or list(padded_shape) != list(meta.padded_shape):
            raise ValueError(
                f"Manifest padded_shape mismatch for {meta.name}: "
                f"{padded_shape!r} != {meta.padded_shape!r}"
            )
        original_shape = item.get("original_shape")
        if original_shape is None or list(original_shape) != list(meta.orig_shape):
            raise ValueError(
                f"Manifest original_shape mismatch for {meta.name}: "
                f"{original_shape!r} != {meta.orig_shape!r}"
            )
        if meta.expert_count:
            canonical_size, canonical_expert_bytes = _canonical_bundle_size_bytes(meta)
            if meta.expert_bytes != canonical_expert_bytes:
                raise ValueError(
                    f"Canonical expert_bytes mismatch for {meta.name}: "
                    f"{meta.expert_bytes} != {canonical_expert_bytes}"
                )
            if item.get("expert_count") != meta.expert_count:
                raise ValueError(
                    f"Manifest expert_count mismatch for {meta.name}: "
                    f"{item.get('expert_count')!r} != {meta.expert_count}"
                )
            if item.get("expert_bytes") != meta.expert_bytes:
                raise ValueError(
                    f"Manifest expert_bytes mismatch for {meta.name}: "
                    f"{item.get('expert_bytes')!r} != {meta.expert_bytes}"
                )
        rel_file = item.get("file")
        if not isinstance(rel_file, str) or not rel_file:
            raise ValueError(f"Manifest file missing for {meta.name}")
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Invalid size_bytes for {meta.name}: {size_bytes!r}")
        if not meta.expert_count:
            canonical_size, _ = _canonical_bundle_size_bytes(meta)
        if size_bytes != canonical_size:
            raise ValueError(
                f"Manifest canonical size_bytes mismatch for {meta.name}: "
                f"{size_bytes} != {canonical_size}"
            )
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            raise ValueError(f"Invalid sha256 for {meta.name}: {sha256!r}")
        if weights_dir is not None:
            resolved = _bundle_path(weights_dir, rel_file, meta.name)
            try:
                actual_size = resolved.stat().st_size
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Missing D bundle binary: {resolved}") from exc
            if actual_size != size_bytes:
                raise ValueError(
                    f"Manifest size_bytes mismatch for {meta.name}: "
                    f"file has {actual_size}, manifest has {size_bytes}"
                )
        meta.rel_file = rel_file
        meta.size_bytes = size_bytes
        meta.sha256 = sha256.lower()

        components = item.get("components")
        if not isinstance(components, list):
            raise ValueError(f"Invalid components for {meta.name}: {components!r}")
        required_components, allowed_components = _canonical_component_names(comp_map, meta)
        seen_components: set[str] = set()
        for c in components:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                raise ValueError(f"Invalid component entry for {meta.name}: {c!r}")
            cname = c.get("name")
            if cname not in allowed_components:
                raise ValueError(f"Unknown component for {meta.name}: {cname!r}")
            seen_components.add(cname)
            cm = comp_map.get(cname)
            if cm is None:
                continue
            if cm.bundle_name != bname:
                raise ValueError(
                    f"Component {cm.hf_name} bundle mismatch: {bname} != {cm.bundle_name}"
                )
            if "row_start" in c or "row_end" in c:
                if (c.get("row_start"), c.get("row_end")) != (cm.row_start, cm.row_end):
                    raise ValueError(
                        f"Row range mismatch for {cm.hf_name}: "
                        f"({c.get('row_start')}, {c.get('row_end')}) != ({cm.row_start}, {cm.row_end})"
                    )
            if "expert_id" in c and "expert_id" in cm.extra:
                if c["expert_id"] != cm.extra["expert_id"]:
                    raise ValueError(
                        f"Expert ID mismatch for {cm.hf_name}: "
                        f"{c['expert_id']} != {cm.extra['expert_id']}"
                    )
        if require_complete:
            missing_components = required_components - seen_components
            if missing_components:
                raise ValueError(
                    f"Manifest missing canonical components for {meta.name}: "
                    + ", ".join(sorted(missing_components))
                )

    if require_complete:
        missing = set(bundle_map) - seen
        if missing:
            raise ValueError(
                "Manifest missing NPU bundles: " + ", ".join(sorted(missing))
            )


# -----------------------------------------------------------------------------
# DWeights Adapter for Host NumPy Parity
# -----------------------------------------------------------------------------
class DWeights(Weights):
    """Weights adapter serving host tensors from host Weights and decoding NPU D-mode
    bundles (q4nx / bfp16) on demand with bounded LRU caching.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        host_weights: Weights | None = None,
        manifest_path: str | Path | None = None,
        selected_layers: Sequence[int] | None = None,
        max_bundle_cache_entries: int = 4,
        max_expert_cache_entries: int = 32,
    ):
        self.weights_dir = Path(weights_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else (self.weights_dir / "manifest.json")
        )
        self.selected_layers = (
            set(selected_layers) if selected_layers is not None else None
        )
        self.load_global = True
        self.skip_moe_experts = True

        self.comp_map, self.bundle_map = build_default_manifest_specs()
        if self.manifest_path.is_file():
            _load_manifest_specs(
                self.manifest_path,
                self.comp_map,
                self.bundle_map,
                weights_dir=self.weights_dir,
            )

        if host_weights is not None:
            self.host_weights = host_weights
            self.tensors: dict[str, np.ndarray] = dict(host_weights.tensors)
        else:
            self.host_weights = None
            self.tensors = {}
            self._load_host_safetensors(selected_layers)

        self.max_bundle_cache_entries = max(1, int(max_bundle_cache_entries))
        self.max_expert_cache_entries = max(1, int(max_expert_cache_entries))
        self._bundle_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._expert_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._f32_cache: dict[str, np.ndarray] = {}

    def _load_host_safetensors(self, selected_layers: Sequence[int] | None) -> None:
        """Load host safetensors from weights_dir if host_weights was not supplied."""
        layers_list = list(selected_layers) if selected_layers is not None else None
        for candidate in (self.weights_dir, self.weights_dir / "host"):
            if candidate.is_dir() and (
                (candidate / "model.safetensors.index.json").is_file()
                or (candidate / "model.safetensors").is_file()
            ):
                hw = Weights(
                    candidate,
                    selected_layers=layers_list,
                    skip_moe_experts=True,
                )
                self.host_weights = hw
                self.tensors = dict(hw.tensors)
                return

        # Fallback: scan individual host_global.safetensors and L*/host.safetensors
        st_files: list[Path] = []
        global_st = self.weights_dir / "host_global.safetensors"
        if global_st.is_file():
            st_files.append(global_st)
        for ly in range(25):
            if self.selected_layers is not None and ly not in self.selected_layers:
                continue
            ly_st = self.weights_dir / f"L{ly:02d}" / "host.safetensors"
            if ly_st.is_file():
                st_files.append(ly_st)

        for st_path in st_files:
            with SafeTensorsReader(st_path) as reader:
                for name in reader.keys():
                    raw, dtype_str, _ = reader.get_tensor_raw(name)
                    if dtype_str == "BF16":
                        self.tensors[name] = np.array(raw, copy=True, dtype=np.uint16)
                    elif dtype_str == "F32":
                        self.tensors[name] = np.array(raw, copy=True, dtype=np.float32)
                    elif dtype_str == "F16":
                        self.tensors[name] = f32_to_bf16_rtne(raw.astype(np.float32))
                    else:
                        self.tensors[name] = np.array(raw, copy=True)

    @staticmethod
    def _parse_routed_expert_name(name: str) -> tuple[int, int, str] | None:
        """Parse 'model.layers.{L}.moe.experts.{E}.{gate_proj|up_proj|down_proj}.weight'."""
        parts = name.split(".")
        if (
            len(parts) == 8
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "moe"
            and parts[4] == "experts"
            and parts[7] == "weight"
            and parts[2].isdigit()
            and parts[5].isdigit()
            and parts[6] in ("gate_proj", "up_proj", "down_proj")
        ):
            layer = int(parts[2])
            expert_id = int(parts[5])
            proj = parts[6]
            if 1 <= layer < 25 and 0 <= expert_id < NUM_ROUTED_EXPERTS:
                return layer, expert_id, proj
        return None

    def _put_bundle_cache(self, bundle_name: str, mat_f32: np.ndarray) -> None:
        self._bundle_cache[bundle_name] = mat_f32
        self._bundle_cache.move_to_end(bundle_name)
        while len(self._bundle_cache) > self.max_bundle_cache_entries:
            self._bundle_cache.popitem(last=False)

    def _put_expert_cache(self, key: str, mat_f32: np.ndarray) -> None:
        self._expert_cache[key] = mat_f32
        self._expert_cache.move_to_end(key)
        while len(self._expert_cache) > self.max_expert_cache_entries:
            self._expert_cache.popitem(last=False)

    def _decode_2d_bundle(self, bundle_name: str) -> np.ndarray:
        """Decode a 2D NPU bundle (q4nx_gemv or bfp16_gemv) to float32 [M_pad, K_pad]."""
        if bundle_name in self._bundle_cache:
            self._bundle_cache.move_to_end(bundle_name)
            return self._bundle_cache[bundle_name]

        bmeta = self.bundle_map.get(bundle_name)
        if bmeta is None:
            raise KeyError(f"Unknown bundle '{bundle_name}'")
        bin_path = self.weights_dir / bmeta.rel_file
        if not bin_path.is_file():
            raise FileNotFoundError(f"Missing D bundle binary: {bin_path}")

        raw_bytes = (
            _read_verified_bundle(self.weights_dir, bmeta)
            if bmeta.sha256
            else bin_path.read_bytes()
        )
        if bmeta.fmt == "q4nx_gemv":
            codes, scales, mins = parse_gemv(raw_bytes, K=bmeta.k, M=bmeta.m, rows=4)
            mat_f32 = dequantize_simple(codes, scales, mins)
        elif bmeta.fmt == "bfp16_gemv":
            mat_f32 = parse_bfp16_gemv(raw_bytes, K=bmeta.k, M=bmeta.m, rows=4)
        else:
            raise ValueError(f"Unsupported 2D bundle format '{bmeta.fmt}' for {bundle_name}")

        self._put_bundle_cache(bundle_name, mat_f32)
        return mat_f32

    def _read_expert_slice(self, bundle_name: str, expert_id: int) -> bytes:
        """Read a single expert's packed byte slice from a 3D MoE bundle file."""
        bmeta = self.bundle_map.get(bundle_name)
        if bmeta is None:
            raise KeyError(f"Unknown MoE bundle '{bundle_name}'")
        bin_path = (
            _verify_bundle_file(self.weights_dir, bmeta)
            if bmeta.sha256
            else self.weights_dir / bmeta.rel_file
        )
        if not bin_path.is_file():
            raise FileNotFoundError(f"Missing D MoE bundle binary: {bin_path}")

        ebytes = bmeta.expert_bytes
        if ebytes <= 0 or not (0 <= expert_id < bmeta.expert_count):
            raise ValueError(
                f"Invalid expert slice request: {bundle_name} expert={expert_id}"
            )
        offset = expert_id * ebytes
        with open(bin_path, "rb") as f:
            f.seek(offset)
            chunk = f.read(ebytes)
        if len(chunk) != ebytes:
            raise ValueError(
                f"Short read in {bin_path} for expert {expert_id}: "
                f"got {len(chunk)} bytes, expected {ebytes}"
            )
        return chunk

    def __contains__(self, name: str) -> bool:
        if self.host_weights is not None and name in self.host_weights:
            return True
        if name in self.tensors:
            return True
        if name in self.comp_map:
            comp = self.comp_map[name]
            if comp.fmt == "mla_kv_b":
                kb = self.bundle_map[comp.extra["kb_bundle"]]
                vb = self.bundle_map[comp.extra["vb_bundle"]]
                return (self.weights_dir / kb.rel_file).is_file() and (
                    self.weights_dir / vb.rel_file
                ).is_file()
            bmeta = self.bundle_map.get(comp.bundle_name)
            return bmeta is not None and (self.weights_dir / bmeta.rel_file).is_file()
        routed = self._parse_routed_expert_name(name)
        if routed is not None:
            layer, _, proj = routed
            bname = (
                f"L{layer:02d}/moe_b1"
                if proj in ("gate_proj", "up_proj")
                else f"L{layer:02d}/moe_b2"
            )
            bmeta = self.bundle_map.get(bname)
            return bmeta is not None and (self.weights_dir / bmeta.rel_file).is_file()
        return False

    def get(self, name: str, to_float: bool = True) -> np.ndarray:
        if not to_float:
            return self._get_uncached(name, to_float=False)

        # Branch-1 host tensors are cached permanently; check the cache first so
        # repeated to_float=True requests avoid redundant host lookup/conversion.
        cached = self._f32_cache.get(name)
        if cached is not None:
            return cached

        host = self._get_host_tensor(name, to_float=True)
        if host is not None:
            return self._cache_f32_host(name, host)

        return self._get_uncached(name, to_float=True)

    def _cache_f32_host(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Cache branch-1 host tensor f32 results with size bound and read-only copy.

        If the incoming array is already read-only float32 in C order, reuse it
        directly so DWeights does not keep a second copy behind Weights.get's
        own f32 cache.
        """
        cached = self._f32_cache.get(name)
        if cached is not None:
            return cached

        is_cached_host = (
            arr.dtype == np.float32
            and (arr.ndim == 0 or arr.flags.c_contiguous)
            and not arr.flags.writeable
        )
        if is_cached_host and arr.nbytes <= DWEIGHTS_F32_CACHE_MAX_TENSOR_BYTES:
            self._f32_cache[name] = arr
            return arr

        if arr.nbytes <= DWEIGHTS_F32_CACHE_MAX_TENSOR_BYTES:
            cached = np.array(arr, dtype=np.float32, copy=True, order="C")
            cached.flags.writeable = False
            self._f32_cache[name] = cached
            return cached
        return arr

    def _get_host_tensor(self, name: str, to_float: bool) -> np.ndarray | None:
        """Branch 1: direct host weights / self.tensors lookup."""
        if self.host_weights is not None and name in self.host_weights:
            return self.host_weights.get(name, to_float=to_float)
        if name in self.tensors:
            arr = self.tensors[name]
            if arr.dtype == np.uint16:
                return bf16_to_f32(arr) if to_float else arr
            if to_float and arr.dtype != np.float32:
                return arr.astype(np.float32)
            return arr
        return None

    def _get_uncached(self, name: str, to_float: bool = True) -> np.ndarray:
        # 1. Host tensors
        host = self._get_host_tensor(name, to_float)
        if host is not None:
            return host

        # 2. Cached expert / special component tensor
        if name in self._expert_cache:
            self._expert_cache.move_to_end(name)
            arr_f32 = self._expert_cache[name]
            return arr_f32 if to_float else f32_to_bf16_rtne(arr_f32)

        # 3. Registered component in comp_map
        if name in self.comp_map:
            comp = self.comp_map[name]
            if comp.fmt == "mla_kv_b":
                kb_mat = self._decode_2d_bundle(comp.extra["kb_bundle"])  # [5120, 512]
                vb_mat = self._decode_2d_bundle(comp.extra["vb_bundle"])  # [5120, 512]
                k_h = kb_mat.reshape(40, 128, 512)
                v_h = vb_mat.reshape(40, 128, 512)
                kv_f32 = np.ascontiguousarray(
                    np.concatenate([k_h, v_h], axis=1).reshape(10240, 512),
                    dtype=np.float32,
                )
                self._put_expert_cache(name, kv_f32)
                return kv_f32 if to_float else f32_to_bf16_rtne(kv_f32)

            if comp.fmt == "shared_down":
                expert_id = int(comp.extra.get("expert_id", 128))
                bmeta = self.bundle_map[comp.bundle_name]
                chunk = self._read_expert_slice(comp.bundle_name, expert_id)
                dn_pad = parse_bfp16_gemv(chunk, K=bmeta.k, M=bmeta.m, rows=4)
                dn_f32 = np.ascontiguousarray(
                    dn_pad[: comp.orig_m, : comp.orig_k], dtype=np.float32
                )
                self._put_expert_cache(name, dn_f32)
                return dn_f32 if to_float else f32_to_bf16_rtne(dn_f32)

            full_f32 = self._decode_2d_bundle(comp.bundle_name)
            sliced_f32 = np.ascontiguousarray(
                full_f32[comp.row_start : comp.row_end, : comp.orig_k],
                dtype=np.float32,
            )
            return sliced_f32 if to_float else f32_to_bf16_rtne(sliced_f32)

        # 4. Routed MoE expert (gate_proj, up_proj, down_proj)
        routed = self._parse_routed_expert_name(name)
        if routed is not None:
            layer, expert_id, proj = routed
            if proj in ("gate_proj", "up_proj"):
                bname = f"L{layer:02d}/moe_b1"
                bmeta = self.bundle_map[bname]
                chunk = self._read_expert_slice(bname, expert_id)
                codes, scales, mins = parse_gemv(chunk, K=bmeta.k, M=bmeta.m, rows=4)
                gu_f32 = dequantize_simple(codes, scales, mins)  # [2176, 2560]
                half = bmeta.m // 2
                gate_f32 = np.ascontiguousarray(gu_f32[:half, :], dtype=np.float32)
                up_f32 = np.ascontiguousarray(gu_f32[half:, :], dtype=np.float32)
                g_key = f"model.layers.{layer}.moe.experts.{expert_id}.gate_proj.weight"
                u_key = f"model.layers.{layer}.moe.experts.{expert_id}.up_proj.weight"
                self._put_expert_cache(g_key, gate_f32)
                self._put_expert_cache(u_key, up_f32)
                out_f32 = gate_f32 if proj == "gate_proj" else up_f32
                return out_f32 if to_float else f32_to_bf16_rtne(out_f32)

            bname = f"L{layer:02d}/moe_b2"
            bmeta = self.bundle_map[bname]
            chunk = self._read_expert_slice(bname, expert_id)
            dn_pad = parse_bfp16_gemv(chunk, K=bmeta.k, M=bmeta.m, rows=4)
            orig_k = bmeta.orig_shape[-1] if len(bmeta.orig_shape) >= 3 else 1088
            dn_f32 = np.ascontiguousarray(dn_pad[:, :orig_k], dtype=np.float32)
            self._put_expert_cache(name, dn_f32)
            return dn_f32 if to_float else f32_to_bf16_rtne(dn_f32)

        raise KeyError(f"Tensor '{name}' not found in DWeights")


def ensure_mla_cache_expanded(
    model: Any,
    snapshot_file_or_dict: str | Path | dict[str, np.ndarray],
) -> int:
    """Expand MLA latent caches via backend.linear(kv_b, lat) for any MLA layer
    whose expanded KV cache was not populated by model.load_snapshot().
    Returns the number of layers expanded by this fallback.
    """
    if isinstance(snapshot_file_or_dict, (str, Path)):
        npz = np.load(str(snapshot_file_or_dict))
        snap = {k: npz[k] for k in npz.files}
    else:
        snap = snapshot_file_or_dict

    cfg = model.config
    be = model.backend
    expanded_count = 0

    for i in range(cfg.num_hidden_layers):
        if cfg.is_rda_layer(i):
            continue
        k_existing, v_existing = be.get_mla_cache(i)
        if k_existing is not None and v_existing is not None:
            continue

        lk = f"layer_{i:03d}.mla_kv_lat_cache"
        pk = f"layer_{i:03d}.mla_k_pe_cache"
        if lk not in snap or pk not in snap:
            continue

        raw_lat = snap[lk].squeeze(0) if snap[lk].ndim == 3 else snap[lk]
        raw_pe = snap[pk].squeeze(0) if snap[pk].ndim == 3 else snap[pk]
        lat_u16 = (
            np.ascontiguousarray(raw_lat, dtype=np.uint16)
            if raw_lat.dtype == np.uint16
            else f32_to_bf16_rtne(raw_lat)
        ).reshape(-1, cfg.kv_lora_rank)
        pe_u16 = (
            np.ascontiguousarray(raw_pe, dtype=np.uint16)
            if raw_pe.dtype == np.uint16
            else f32_to_bf16_rtne(raw_pe)
        ).reshape(-1, cfg.qk_rope_head_dim)

        H = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        v_dim = cfg.v_head_dim
        t_len = lat_u16.shape[0]

        kv_u16 = be.linear(
            f"model.layers.{i}.self_attn.kv_b_proj.weight", lat_u16
        ).reshape(t_len, H, nope + v_dim)
        k_nope = kv_u16[:, :, :nope]
        vals = kv_u16[:, :, nope:]
        pe_exp = np.tile(pe_u16[:, np.newaxis, :], (1, H, 1))
        keys = np.concatenate([k_nope, pe_exp], axis=-1)

        be.set_mla_cache(
            i,
            np.ascontiguousarray(keys.transpose(1, 0, 2)),
            np.ascontiguousarray(vals.transpose(1, 0, 2)),
        )
        expanded_count += 1

    return expanded_count


@dataclass(eq=False)
class _RdaTicket:
    """One overlapped prefill RDA call of ``tokens`` tokens (DESIGN-prefill.md sections 18 and 36).

    ``runner`` is the direct runner the call was started on (wait and abort use it); the mock path keeps the outputs
    and the state after the call in ``mock_out`` / ``mock_state`` and commits the state only when the wait succeeds.
    """

    layer: int
    io: int
    phase: str  # "staged", "launched", "waited", "done"
    xclbin_id: str
    tokens: int = 1
    slot: int | None = None
    run: Any = None
    keep: Any = None
    runner: Any = None
    mock_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
    mock_out: np.ndarray | None = None
    mock_state: np.ndarray | None = None
    t_launch: float = 0.0


def _prepare_xrt_handle(runtime: Any, npu_kernel: Any) -> Any:
    """Load and cache an NPU kernel handle through the runtime loader."""
    return runtime.load(npu_kernel)


def _poll_wait_states(pyxrt: Any) -> tuple[Any, ...]:
    """The pending command states a direct runner polls, or () when RAON_NPU_POLL_WAIT is off (DESIGN-prefill.md section 38)."""
    if os.environ.get("RAON_NPU_POLL_WAIT", "1") in ("0", "false", "False", "no"):
        return ()
    states = pyxrt.ert_cmd_state
    return tuple(
        getattr(states, f"ERT_CMD_STATE_{name}")
        for name in ("NEW", "QUEUED", "RUNNING", "SUBMITTED")
        if hasattr(states, f"ERT_CMD_STATE_{name}")
    )


def _poll_run(run: Any, pending: tuple[Any, ...]) -> None:
    """Poll a started run until it leaves ``pending``, for at most 1 s, giving up the time slice between polls.

    The waiting thread stays busy instead of sleeping in run.wait(), so its core keeps its clock and the host work
    between NPU calls runs faster: prefill -0.2 to -2.0 ms/token in same-engine A/B runs, bytes unchanged
    (DESIGN-prefill.md section 38, R3-3). time.sleep(0) is Windows Sleep(0): it hands the rest of the time slice to a
    ready thread and releases the GIL. Polling only changes timing: a run still pending after 1 s, or a run.state()
    that raises, is left to the caller's blocking run.wait(), which waits and reports the outcome as before (so an
    abort still drains a started run).
    """
    deadline = time.perf_counter() + 1.0
    try:
        while run.state() in pending and time.perf_counter() < deadline:
            time.sleep(0)
    except Exception:
        pass


class _XrtDirectRunner:
    """Start and wait a loaded XRT kernel the way aie.utils.hostruntime.xrtruntime.XRTHostRuntime.run does, with the start and the wait separated so host work can run in between (DESIGN-prefill.md section 18, R2)."""

    def __init__(
        self, runtime: Any, npu_kernel: Any, completed_state: Any, pending_states: tuple[Any, ...] = ()
    ) -> None:
        self.runtime = runtime
        self.npu_kernel = npu_kernel
        self.completed_state = completed_state
        self.pending_states = pending_states
        self.handle = _prepare_xrt_handle(self.runtime, self.npu_kernel)

    def start(self, args: list[Any]) -> tuple[Any, Any]:
        """Validate handle, migrate buffers, and launch kernel asynchronously."""
        if not getattr(self.handle, "_is_valid", True) or getattr(self.handle, "insts_bo", None) is None:
            self.handle = _prepare_xrt_handle(self.runtime, self.npu_kernel)
        insts_bo = getattr(self.handle, "insts_bo", None)
        if insts_bo is None:
            raise RuntimeError("NPU kernel handle does not have insts_bo")
        insts_bytes = self.handle.insts.nbytes
        for a in args:
            a.to("npu")
        bos = [a.buffer_object() for a in args]
        run = self.handle.kernel(3, insts_bo, insts_bytes, *bos)
        keep = (self.handle, args, bos)
        return run, keep

    def wait(self, run: Any) -> None:
        """Wait for kernel completion and verify termination state."""
        if self.pending_states:
            _poll_run(run, self.pending_states)
        state = run.wait()
        if state != self.completed_state:
            raise RuntimeError(
                f"Kernel execution failed with state {state} (expected {self.completed_state})"
            )


def _make_xrt_direct_runner(op: Any) -> _XrtDirectRunner:
    import pyxrt
    from aie.utils.npukernel import NPUKernel

    npu_kernel = NPUKernel(
        xclbin_path=op.xclbin_artifact.filename,
        kernel_name=op.xclbin_artifact.kernel_name,
        insts_path=op.insts_artifact.filename,
    )
    runtime = aie_utils.DefaultNPURuntime
    completed_state = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    return _XrtDirectRunner(runtime, npu_kernel, completed_state, _poll_wait_states(pyxrt))


@dataclass(frozen=True)
class _RdaPrefillStream:
    """One prefill RDA instruction stream of the plate (DESIGN-prefill.md section 36, R3-2): T tokens per call.

    ``heads[io]`` is the [40, T, 640] head BO and ``outs[io]`` the [40, 4, T, 32] output BO of overlap slot ``io``.
    """

    tokens: int
    runner: _XrtDirectRunner
    heads: tuple[Any, Any]
    outs: tuple[Any, Any]


_LIVE_GROUPED_RUNNERS: weakref.WeakSet[Any] = weakref.WeakSet()
_GROUPED_RUNNER_ATEXIT_REGISTERED = False


def _close_live_grouped_runners() -> None:
    for runner in list(_LIVE_GROUPED_RUNNERS):
        runner.close()
    gc.collect()


def _grouped_dispatch_scalars(shape: Any, call: Any) -> dict[str, int]:
    scalars = {
        f"e{slot}": int(expert)
        for slot, expert in enumerate(call.expert_slots)
    }
    if shape is GROUPED_B1:
        for slot, count in enumerate(call.rows):
            scalars[f"r{slot}"] = int(count)
    return scalars


def _grouped_x_tiled(values: np.ndarray, shape: Any) -> np.ndarray:
    """Grouped X as the device reads it: per slot [k_block][32][TMAX][8], contiguous, from [SLOTS, TMAX, K].

    The grouped design reads each slot's X as one contiguous run (DESIGN-prefill.md section 49); the cores receive
    the same bytes in the same order as the former 16-byte walk over [SLOTS, TMAX, K].
    """
    x = np.asarray(values, dtype=np.uint16).reshape(
        SLOTS, TMAX, shape.k // GROUPED_K_TILE, GROUPED_K_TILE // 8, 8
    )
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1, 4))


class _GroupedDirectRunner:
    """Double-buffered direct XRT runner for one grouped MoE shape."""

    def __init__(
        self,
        backend: "NpuBackend",
        shape: Any,
        kernel: Any,
        runtime: Any,
        completed_state: Any,
        flags: Any,
        x_buffers: tuple[Any, Any],
        y_buffers: tuple[Any, Any],
        pending_states: tuple[Any, ...] = (),
    ) -> None:
        self.backend = backend
        self.shape = shape
        self.kernel = kernel
        self.runtime = runtime
        self.completed_state = completed_state
        self.pending_states = pending_states
        self.flags = flags
        self.x = x_buffers
        self.y = y_buffers
        self.handle = _prepare_xrt_handle(runtime, kernel)
        zero_scalars = {name: 0 for name in kernel.dispatch_params}
        self.n_words = len(kernel.instructions(**zero_scalars))
        self._validate_abi()
        self.insts = self._new_instruction_buffers()
        self._kept: list[Any | None] = [None, None]
        self._closed = False

    def _validate_abi(self) -> None:
        kernels = self.handle.xclbin.get_kernels()
        selected = None
        if self.handle.name is not None:
            selected = next(
                (item for item in kernels if item.get_name() == self.handle.name), None
            )
        if selected is None and kernels:
            selected = kernels[0]
        declared = 0 if selected is None else selected.get_num_args() - 3
        if declared < 3:
            raise RuntimeError(
                f"grouped {self.shape.name} xclbin declares {declared} host buffer "
                "argument(s); at least 3 are required"
            )

    def _new_instruction_buffers(self) -> tuple[Any, Any]:
        group_id = self.handle.kernel.group_id(1)
        make = self.runtime._tensor_class
        return tuple(
            make(
                np.zeros(self.n_words, dtype=np.uint32),
                flags=self.flags,
                group_id=group_id,
                xrt_device=self.runtime._device,
            )
            for _ in range(2)
        )

    def prepare(self, io: int, call: Any, values: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("grouped MoE direct runner is closed")
        if not getattr(self.handle, "_is_valid", True):
            self.handle = _prepare_xrt_handle(self.runtime, self.kernel)
            self.insts = self._new_instruction_buffers()
        self.backend._overwrite_buffer(
            self.x[io],
            _grouped_x_tiled(values, self.shape).reshape(-1).view(bfloat16),
        )
        self.x[io].to("npu")
        scalars = _grouped_dispatch_scalars(self.shape, call)
        words = np.asarray(self.kernel.instructions(**scalars), dtype=np.uint32)
        if len(words) != self.n_words:
            raise RuntimeError(
                f"grouped {self.shape.name} instruction length changed from "
                f"{self.n_words} to {len(words)}"
            )
        self.backend._overwrite_buffer(self.insts[io], words)
        self.insts[io].to("npu")

    def start(self, io: int, layer: int) -> Any:
        if self._closed:
            raise RuntimeError("grouped MoE direct runner is closed")
        if not getattr(self.handle, "_is_valid", True):
            raise RuntimeError("grouped MoE XRT handle became invalid after prepare")
        self.runtime.check_device_consistency()
        weight = self.backend._get_prefill_weight_tensor(
            f"L{layer:02d}/moe_{self.shape.name}"
        )
        args = [self.y[io], weight, self.x[io]]
        if not all(isinstance(arg, self.runtime._tensor_class) for arg in args):
            raise RuntimeError(
                f"grouped {self.shape.name} arguments must use "
                f"{self.runtime._tensor_class.__name__}"
            )
        for arg in args:
            arg.to("npu")
        bos = [arg.buffer_object() for arg in args]
        run = self.handle.kernel(
            3, self.insts[io].buffer_object(), self.n_words * 4, *bos
        )
        self._kept[io] = (self.handle, self.insts[io], args, bos, run)
        return run

    def wait(self, run: Any) -> None:
        if self.pending_states:
            _poll_run(run, self.pending_states)
        state = run.wait()
        for io, kept in enumerate(self._kept):
            if kept is not None and kept[-1] is run:
                self._kept[io] = None
                break
        if state != self.completed_state:
            raise RuntimeError(
                f"Kernel execution failed with state {state} "
                f"(expected {self.completed_state})"
            )

    def close(self) -> None:
        """Release this runner's XRT objects while the IRON context is alive."""
        if self._closed:
            return
        self._closed = True
        self._kept[:] = [None, None]
        self.insts = (None, None)
        self.x = (None, None)
        self.y = (None, None)
        self.handle = None
        self.kernel = None

    def read(self, io: int) -> np.ndarray:
        return np.array(self.y[io].numpy().view(np.uint16), copy=True).reshape(
            SLOTS, TMAX, self.shape.m
        )


def _make_grouped_direct_runner(
    backend: "NpuBackend", shape: Any, kernel: Any, x0: Any, y0: Any
) -> _GroupedDirectRunner:
    """Create a grouped direct runner; importing pyxrt stays hardware-only."""
    global _GROUPED_RUNNER_ATEXIT_REGISTERED

    import pyxrt

    runtime = aie_utils.DefaultNPURuntime
    x1 = aie_utils.tensor(np.zeros(SLOTS * TMAX * shape.k, dtype=bfloat16))
    y1 = aie_utils.full((SLOTS * TMAX * shape.m,), 0.0, dtype=bfloat16)
    runner = _GroupedDirectRunner(
        backend,
        shape,
        kernel,
        runtime,
        pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED,
        pyxrt.bo.cacheable,
        (x0, x1),
        (y0, y1),
        _poll_wait_states(pyxrt),
    )
    _LIVE_GROUPED_RUNNERS.add(runner)
    if not _GROUPED_RUNNER_ATEXIT_REGISTERED:
        # IRON registers its cleanup when its runtime is created.  Runners can
        # only be made after that, so atexit's last-registered-first-run order
        # releases our XRT objects while the IRON context is still alive.
        atexit.register(_close_live_grouped_runners)
        _GROUPED_RUNNER_ATEXIT_REGISTERED = True
    return runner


_LIVE_DECODE_RUNNERS: weakref.WeakSet[Any] = weakref.WeakSet()
_DECODE_RUNNER_ATEXIT_REGISTERED = False


def _close_live_decode_runners() -> None:
    for runner in list(_LIVE_DECODE_RUNNERS):
        runner.close()
    gc.collect()


@dataclass
class _DecodeGemvEntry:
    bundle: str
    handle: Any
    x: Any
    y: Any
    x_host: np.ndarray
    y_host: np.ndarray
    run: Any
    k: int
    m: int
    shape_name: str | None
    timing_key: str


@dataclass
class _DecodeMoeShape:
    handle: Any
    insts: Any
    insts_host: np.ndarray
    n_words: int
    x: Any
    y: Any
    x_host: np.ndarray
    y_host: np.ndarray
    runs: dict[int, Any]
    staged: Any = None


@dataclass
class _DecodeRdaEntry:
    handle: Any
    head: Any
    out: Any
    head_host: np.ndarray
    out_host: np.ndarray
    run: Any


@dataclass
class _DecodeGemvBuffers:
    x: Any
    y: Any
    x_host: np.ndarray
    y_host: np.ndarray


@dataclass
class _DecodeMoeBuffers:
    handle: Any
    insts: Any
    insts_host: np.ndarray
    n_words: int
    x: Any
    y: Any
    x_host: np.ndarray
    y_host: np.ndarray


@dataclass
class _DecodeRdaBuffers:
    handle: Any
    head: Any
    out: Any
    head_host: np.ndarray
    out_host: np.ndarray


class _DecodeDirectRunner:
    """Reusable direct XRT runs for unified decode kernels (DESIGN-prefill 41)."""

    def __init__(self, runtime: Any, pyxrt: Any, xclbin_path: str | Path) -> None:
        self.runtime = runtime
        self.pyxrt = pyxrt
        self.xclbin_path = Path(xclbin_path)
        self.completed_state = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        self.pending_states = _poll_wait_states(pyxrt)
        self.to_device = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.from_device = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self._gemv: dict[str, _DecodeGemvEntry] = {}
        self._moe: dict[str, _DecodeMoeShape] = {}
        self._rda: _DecodeRdaEntry | None = None
        self._gemv_bufs: dict[str, _DecodeGemvBuffers] = {}
        self._moe_bufs: dict[str, _DecodeMoeBuffers] = {}
        self._rda_buf: _DecodeRdaBuffers | None = None
        self._handles: dict[tuple[str, str], Any] = {}
        self._settle_s: float = 0.0
        self._pending_settle: bool = False
        self._prepared_gemv: set[str] = set()
        self._prepared_moe: set[str] = set()
        self._prepared_rda: bool = False
        self._settle_warned: set[str] = set()
        self._gemv_specs: dict[str, tuple[Any, int, int]] = {}
        self._moe_specs: dict[str, tuple[Any, int, int]] = {}
        self._rda_op: Any | None = None
        self._closed = False
        runtime.check_device_consistency()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("decode direct runner is closed")

    @staticmethod
    def _tensor_bo(tensor: Any) -> Any:
        return tensor.buffer_object()

    @staticmethod
    def _handle_valid(handle: Any, *, needs_insts: bool) -> bool:
        return bool(
            getattr(handle, "_is_valid", True)
            and getattr(handle, "kernel", None) is not None
            and (
                not needs_insts
                or (
                    getattr(handle, "insts_bo", None) is not None
                    and getattr(handle, "insts", None) is not None
                )
            )
        )

    def _check_handle(
        self, handle: Any, *, needs_insts: bool, family: str | None = None
    ) -> bool:
        if not self._handle_valid(handle, needs_insts=needs_insts):
            self._discard_entries()
            self._reload_handles_and_settle(family)
            return False
        return True

    def _discard_entries(self) -> None:
        self._gemv.clear()
        self._moe.clear()
        self._rda = None
        self._handles.clear()

    def _prepare_moe_handle(self, key: str, kernel: Any) -> Any:
        handle_key = ("moe", key)
        handle = self.runtime.load(kernel)
        self._validate_abi(handle, self._kernel_name(kernel), 3)
        self._handles[handle_key] = handle
        return handle

    def _reload_handles_and_settle(self, family: str | None = None) -> None:
        """After invalidation, reload every remembered handle and settle once.

        If ``prepare`` was called we have a complete spec: reload all GEMV, MoE,
        and RDA handles, reconnect them to the existing BO pools, mark every
        prepared family, and sleep exactly once.  If no spec was recorded, mark
        pending settle so the next start call settles after handles and BOs are
        recreated.
        """
        if self._gemv_specs or self._moe_specs or self._rda_op:
            seen_insts: set[str] = set()
            for op, _k, _m in self._gemv_specs.values():
                insts_file = str(op.insts_artifact.filename)
                if insts_file not in seen_insts:
                    self._prepare_static_handle(op, 3)
                    seen_insts.add(insts_file)
            for key, (kernel, _x, _y) in self._moe_specs.items():
                handle = self._prepare_moe_handle(key, kernel)
                buf = self._moe_bufs.get(key)
                if buf is not None:
                    buf.handle = handle
                    buf.insts = self.runtime._tensor_class(
                        np.zeros(buf.n_words, dtype=np.uint32),
                        flags=self.pyxrt.bo.cacheable,
                        group_id=handle.kernel.group_id(1),
                        xrt_device=self.runtime._device,
                    )
                    buf.insts_host = buf.insts.data.view(np.uint32)
            if self._rda_op is not None:
                handle = self._prepare_static_handle(self._rda_op, 4)
                if self._rda_buf is not None:
                    self._rda_buf.handle = handle
            self._prepared_gemv.update(self._gemv_specs)
            self._prepared_moe.update(self._moe_specs)
            self._prepared_rda = self._rda_op is not None
            if self._settle_s > 0.0:
                time.sleep(self._settle_s)
        else:
            self._pending_settle = True

    def _settle(self, family: str) -> None:
        if self._settle_s <= 0.0:
            return
        if family not in self._settle_warned:
            print(
                f"[decode-direct] {family}: BO not prepared; settling "
                f"{self._settle_s:.3f}s",
                file=sys.stderr,
                flush=True,
            )
            self._settle_warned.add(family)
        time.sleep(self._settle_s)

    def _make_gemv_buffers(
        self, bundle: str, k: int, m: int
    ) -> _DecodeGemvBuffers:
        del bundle, k
        x = aie_utils.tensor(np.zeros(MAX_K, dtype=bfloat16))
        y = aie_utils.full((m,), 0.0, dtype=bfloat16)
        return _DecodeGemvBuffers(
            x, y, x.data.view(np.uint16), y.data.view(np.uint16)
        )

    def _ensure_gemv_buffers(
        self, bundle: str, k: int, m: int
    ) -> _DecodeGemvBuffers:
        buf = self._gemv_bufs.get(bundle)
        if buf is None:
            buf = self._make_gemv_buffers(bundle, k, m)
            self._gemv_bufs[bundle] = buf
            if bundle not in self._prepared_gemv:
                self._settle("gemv")
        return buf

    def _prepare_moe_buffers_new(
        self, key: str, kernel: Any, x_elems: int, y_elems: int
    ) -> _DecodeMoeBuffers:
        handle_key = ("moe", key)
        handle = self._handles.get(handle_key)
        if handle is None:
            handle = self._prepare_moe_handle(key, kernel)
        zero_scalars = {name: 0 for name in kernel.dispatch_params}
        n_words = len(kernel.instructions(**zero_scalars))
        insts = self.runtime._tensor_class(
            np.zeros(n_words, dtype=np.uint32),
            flags=self.pyxrt.bo.cacheable,
            group_id=handle.kernel.group_id(1),
            xrt_device=self.runtime._device,
        )
        x = aie_utils.tensor(np.zeros(x_elems, dtype=bfloat16))
        y = aie_utils.full((y_elems,), 0.0, dtype=bfloat16)
        return _DecodeMoeBuffers(
            handle,
            insts,
            insts.data.view(np.uint32),
            n_words,
            x,
            y,
            x.data.view(np.uint16),
            y.data.view(np.uint16),
        )

    def _prepare_moe_shape_buffers(
        self, key: str, kernel: Any, x_elems: int, y_elems: int
    ) -> _DecodeMoeBuffers:
        buf = self._moe_bufs.get(key)
        if buf is None:
            buf = self._prepare_moe_buffers_new(key, kernel, x_elems, y_elems)
            self._moe_bufs[key] = buf
            if key not in self._prepared_moe:
                self._settle("moe")
        elif not self._handle_valid(buf.handle, needs_insts=False):
            handle = self._prepare_moe_handle(key, kernel)
            buf.handle = handle
            buf.insts = self.runtime._tensor_class(
                np.zeros(buf.n_words, dtype=np.uint32),
                flags=self.pyxrt.bo.cacheable,
                group_id=handle.kernel.group_id(1),
                xrt_device=self.runtime._device,
            )
            buf.insts_host = buf.insts.data.view(np.uint32)
        return buf

    def _make_rda_buffers(self, op: Any) -> _DecodeRdaBuffers:
        handle = self._prepare_static_handle(op, 4)
        head = iron.zeros((40, 640), dtype=np.float32, device="npu")
        out = iron.zeros((40, 128), dtype=np.float32, device="npu")
        return _DecodeRdaBuffers(handle, head, out, head.data, out.data)

    def _ensure_rda_buffers(self, op: Any) -> _DecodeRdaBuffers:
        if self._rda_buf is None:
            self._rda_buf = self._make_rda_buffers(op)
            if not self._prepared_rda:
                self._settle("rda")
        elif not self._handle_valid(self._rda_buf.handle, needs_insts=True):
            self._rda_buf.handle = self._prepare_static_handle(op, 4)
        return self._rda_buf

    def prepare(
        self,
        gemv_specs: dict[str, tuple[Any, int, int]],
        moe_specs: dict[str, tuple[Any, int, int]],
        rda_op: Any | None,
        settle_s: float,
    ) -> None:
        """Pre-allocate all decode BOs and settle once before first use."""
        self._check_open()
        self._settle_s = max(0.0, float(settle_s))
        self._gemv_specs = dict(gemv_specs)
        self._moe_specs = dict(moe_specs)
        self._rda_op = rda_op
        for bundle in gemv_specs:
            self._prepared_gemv.add(bundle)
        for key in moe_specs:
            self._prepared_moe.add(key)
        if rda_op is not None:
            self._prepared_rda = True
        # Load every handle before allocating any data BO so that every
        # command-stream BO is created before the settle sleep.
        seen_insts: set[str] = set()
        for op, _k, _m in gemv_specs.values():
            insts_file = str(op.insts_artifact.filename)
            if insts_file not in seen_insts:
                self._prepare_static_handle(op, 3)
                seen_insts.add(insts_file)
        for key, (kernel, x_elems, y_elems) in moe_specs.items():
            self._prepare_moe_shape_buffers(key, kernel, x_elems, y_elems)
        if rda_op is not None:
            self._ensure_rda_buffers(rda_op)
        # Allocate data BOs after all handles are loaded; with prepared flags
        # already set these calls will not settle.
        for bundle, (_op, k, m) in gemv_specs.items():
            self._ensure_gemv_buffers(bundle, k, m)
        if self._settle_s > 0.0:
            time.sleep(self._settle_s)

    @staticmethod
    def _kernel_name(kernel: Any) -> str | None:
        return getattr(kernel, "kernel_name", None)

    def _validate_abi(self, handle: Any, kernel_name: str | None, required: int) -> None:
        kernels = handle.xclbin.get_kernels()
        selected = next(
            (item for item in kernels if item.get_name() == kernel_name), None
        )
        if selected is None:
            raise RuntimeError(f"kernel {kernel_name!r} is not present in decode xclbin")
        declared = selected.get_num_args() - 3
        if declared < required:
            raise RuntimeError(
                f"decode kernel {kernel_name} declares {declared} host buffer "
                f"argument(s); at least {required} are required"
            )

    def _prepare_static_handle(self, op: Any, required: int) -> Any:
        from aie.utils.npukernel import NPUKernel

        key = ("static", str(op.insts_artifact.filename))
        cached = self._handles.get(key)
        if cached is not None and self._handle_valid(cached, needs_insts=True):
            return cached
        kernel = NPUKernel(
            xclbin_path=op.xclbin_artifact.filename,
            kernel_name=op.xclbin_artifact.kernel_name,
            insts_path=op.insts_artifact.filename,
        )
        handle = self.runtime.load(kernel)
        self._validate_abi(handle, kernel.kernel_name, required)
        self._handles[key] = handle
        return handle

    def gemv_cached_entry(self, bundle: str) -> _DecodeGemvEntry | None:
        self._check_open()
        entry = self._gemv.get(bundle)
        if entry is not None and not self._check_handle(
            entry.handle, needs_insts=True, family="gemv"
        ):
            return None
        return entry

    def gemv_entry(
        self, bundle: str, op: Any, w_tensor: Any, k: int, m: int,
        shape_name: str | None = None, timing_key: str | None = None,
        *, already_checked: bool = False,
    ) -> _DecodeGemvEntry:
        cached = self._gemv.get(bundle) if already_checked else self.gemv_cached_entry(bundle)
        if cached is not None:
            return cached
        handle = self._prepare_static_handle(op, 3)
        buf = self._ensure_gemv_buffers(bundle, k, m)
        run = self.pyxrt.run(handle.kernel)
        args = (
            3,
            handle.insts_bo,
            handle.insts.nbytes,
            self._tensor_bo(buf.y),
            self._tensor_bo(w_tensor),
            self._tensor_bo(buf.x),
        )
        for index, arg in enumerate(args):
            run.set_arg(index, arg)
        entry = _DecodeGemvEntry(
            bundle, handle, buf.x, buf.y, buf.x_host, buf.y_host,
            run, k, m, shape_name, timing_key or f"gemv:{bundle.split('/')[-1]}"
        )
        self._gemv[bundle] = entry
        return entry

    def gemv_start(self, entry: _DecodeGemvEntry, x_u16: np.ndarray) -> Any:
        entry.x_host[: entry.k] = np.asarray(x_u16, dtype=np.uint16).reshape(-1)[: entry.k]
        self._tensor_bo(entry.x).sync(self.to_device)
        if self._pending_settle:
            self._pending_settle = False
            if self._settle_s > 0.0:
                time.sleep(self._settle_s)
        entry.run.start()
        return entry.run

    def gemv_finish(self, entry: _DecodeGemvEntry, run: Any) -> np.ndarray:
        self.wait(run)
        self._tensor_bo(entry.y).sync(self.from_device)
        return np.array(entry.y_host, copy=True)

    def gemv(self, entry: _DecodeGemvEntry, x_u16: np.ndarray) -> np.ndarray:
        run = self.gemv_start(entry, x_u16)
        return self.gemv_finish(entry, run)

    def _prepare_moe_shape(
        self, key: str, kernel: Any, w_tensor: Any, words: np.ndarray,
        x_elems: int, y_elems: int, layer: int,
    ) -> _DecodeMoeShape:
        created = False
        shape = self._moe.get(key)
        if shape is None:
            buf = self._prepare_moe_shape_buffers(key, kernel, x_elems, y_elems)
            shape = _DecodeMoeShape(
                buf.handle, buf.insts, buf.insts_host, buf.n_words,
                buf.x, buf.y, buf.x_host, buf.y_host, {},
            )
            created = True
        if len(words) != shape.n_words:
            raise RuntimeError(
                f"decode MoE {key} instruction length changed from "
                f"{shape.n_words} to {len(words)}"
            )
        if layer not in shape.runs:
            run = self.pyxrt.run(shape.handle.kernel)
            args = (
                3, self._tensor_bo(shape.insts), shape.n_words * 4,
                self._tensor_bo(shape.y), self._tensor_bo(w_tensor),
                self._tensor_bo(shape.x),
            )
            for index, arg in enumerate(args):
                run.set_arg(index, arg)
            shape.runs[layer] = run
        if created:
            self._moe[key] = shape
        return shape

    def moe_start(
        self, key: str, layer: int, kernel: Any, w_tensor: Any,
        words: np.ndarray, x_rows: np.ndarray, x_elems: int, y_elems: int,
    ) -> Any:
        self._check_open()
        shape = self._moe.get(key)
        if shape is not None:
            self._check_handle(shape.handle, needs_insts=False, family="moe")
        words_u32 = np.asarray(words, dtype=np.uint32).reshape(-1)
        shape = self._prepare_moe_shape(
            key, kernel, w_tensor, words_u32, x_elems, y_elems, layer
        )
        staged, shape.staged = shape.staged, None
        if staged is not words:  # moe_stage already wrote and synced these exact words
            shape.insts_host[:] = words_u32
            self._tensor_bo(shape.insts).sync(self.to_device)
        rows = np.asarray(x_rows, dtype=np.uint16)
        if key == "b1":
            shape.x_host[: rows.size] = rows.reshape(-1)
        elif key == "b2":
            shape.x_host.reshape(7, MAX_K)[:, : rows.shape[1]] = rows
        else:
            raise ValueError(f"unknown decode MoE shape {key!r}")
        self._tensor_bo(shape.x).sync(self.to_device)
        run = shape.runs[layer]
        if self._pending_settle:
            self._pending_settle = False
            if self._settle_s > 0.0:
                time.sleep(self._settle_s)
                self._tensor_bo(shape.insts).sync(self.to_device)
        run.start()
        return run

    def moe_stage(
        self, key: str, layer: int, kernel: Any, w_tensor: Any,
        words: np.ndarray, x_elems: int, y_elems: int,
    ) -> None:
        """Write and sync a MoE call's instruction words ahead of its start (DESIGN-prefill.md section 46).

        The next moe_start of ``key`` with this same ``words`` object skips its own copy and sync; other words, or a
        shape rebuilt in between (handle reload), are written as before. Other runs may be in flight here, so this
        never discards entries, reloads handles or builds a shape: without an existing shape whose handle is valid
        and whose word count matches, nothing is staged and moe_start does all of it after those runs finish.
        """
        self._check_open()
        shape = self._moe.get(key)
        words_u32 = np.asarray(words, dtype=np.uint32).reshape(-1)
        if (
            shape is None
            or len(words_u32) != shape.n_words
            or not self._handle_valid(shape.handle, needs_insts=False)
        ):
            return
        shape.insts_host[:] = words_u32
        self._tensor_bo(shape.insts).sync(self.to_device)
        shape.staged = words

    def overlap_ready(self, entry: _DecodeGemvEntry) -> bool:
        """True when the handle checks made while ``entry``'s run is in flight cannot discard or reload (section 46).

        moe_experts looks ``entry`` up again and starts B1 while a prestarted shared run is in flight. Those checks
        change nothing only if ``entry`` is still the cached entry with a valid handle and the B1 shape exists with a
        valid handle. This method itself changes nothing.
        """
        shape = self._moe.get("b1")
        return (
            self._gemv.get(entry.bundle) is entry
            and self._handle_valid(entry.handle, needs_insts=True)
            and shape is not None
            and self._handle_valid(shape.handle, needs_insts=False)
        )

    def moe_clear_staged(self) -> None:
        """Forget staged instruction words (a staged MoE call that will not start)."""
        for shape in self._moe.values():
            shape.staged = None

    def moe_finish(self, key: str, run: Any) -> np.ndarray:
        self.wait(run)
        shape = self._moe[key]
        self._tensor_bo(shape.y).sync(self.from_device)
        return np.array(shape.y_host, copy=True)

    def rda(self, op: Any, s_in: Any, s_out: Any, bundle_host: np.ndarray) -> np.ndarray:
        self._check_open()
        if self._rda is not None:
            self._check_handle(self._rda.handle, needs_insts=True, family="rda")
        if self._rda is None:
            buf = self._ensure_rda_buffers(op)
            run = self.pyxrt.run(buf.handle.kernel)
            fixed = (
                (0, 3), (1, buf.handle.insts_bo), (2, buf.handle.insts.nbytes),
                (5, self._tensor_bo(buf.head)), (6, self._tensor_bo(buf.out)),
            )
            for index, arg in fixed:
                run.set_arg(index, arg)
            self._rda = _DecodeRdaEntry(
                buf.handle, buf.head, buf.out, buf.head_host, buf.out_host, run
            )
        entry = self._rda
        s_in.to("npu")
        s_out.to("npu")
        entry.run.set_arg(3, self._tensor_bo(s_in))
        entry.run.set_arg(4, self._tensor_bo(s_out))
        entry.head_host[...] = np.asarray(bundle_host, dtype=np.float32)
        self._tensor_bo(entry.head).sync(self.to_device)
        if self._pending_settle:
            self._pending_settle = False
            if self._settle_s > 0.0:
                time.sleep(self._settle_s)
        entry.run.start()
        self.wait(entry.run)
        self._tensor_bo(entry.out).sync(self.from_device)
        return np.array(entry.out_host, dtype=np.float32, copy=True)

    def wait(self, run: Any) -> None:
        if self.pending_states:
            _poll_run(run, self.pending_states)
        state = run.wait()
        if state != self.completed_state:
            raise RuntimeError(
                f"Kernel execution failed with state {state} "
                f"(expected {self.completed_state})"
            )

    def drain(self, runs: Iterable[Any]) -> None:
        for run in runs:
            try:
                run.wait()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_settle = False
        self._discard_entries()
        self._gemv_bufs.clear()
        self._moe_bufs.clear()
        self._rda_buf = None
        self._handles.clear()
        self._gemv_specs.clear()
        self._moe_specs.clear()
        self._rda_op = None
        self.runtime = None
        self.pyxrt = None


def _make_decode_direct_runner(xclbin_path: str | Path) -> _DecodeDirectRunner:
    """Create the decode runner while keeping the pyxrt import hardware-only."""
    global _DECODE_RUNNER_ATEXIT_REGISTERED

    import pyxrt

    runner = _DecodeDirectRunner(aie_utils.DefaultNPURuntime, pyxrt, xclbin_path)
    _LIVE_DECODE_RUNNERS.add(runner)
    if not _DECODE_RUNNER_ATEXIT_REGISTERED:
        # IRON registers first; LIFO atexit therefore releases these XRT
        # objects before the IRON runtime tears its context down.
        atexit.register(_close_live_decode_runners)
        _DECODE_RUNNER_ATEXIT_REGISTERED = True
    return runner


class _GroupedNumpyRunner:
    """Synchronous NumPy runner with the same pipeline interface."""

    def __init__(self, backend: "NpuBackend", shape: Any, dw: DWeights | None) -> None:
        self.backend = backend
        self.shape = shape
        self.dw = dw
        self._prepared: list[Any | None] = [None, None]
        self._outputs: list[np.ndarray | None] = [None, None]

    def prepare(self, io: int, call: Any, values: np.ndarray) -> None:
        self._prepared[io] = (call, values)

    def start(self, io: int, layer: int) -> Any:
        prepared = self._prepared[io]
        if prepared is None:
            raise RuntimeError("grouped NumPy runner was started before prepare")
        call, values = prepared
        output, _ = self.backend._run_grouped_moe_numpy_call(
            self.shape, layer, call, values, self.dw, record_xclbin=False
        )
        self._outputs[io] = output
        return object()

    def wait(self, run: Any) -> None:
        del run

    def read(self, io: int) -> np.ndarray:
        output = self._outputs[io]
        if output is None:
            raise RuntimeError("grouped NumPy output is not ready")
        return np.array(output, copy=True)


class _GroupedCallableRunner:
    """Compatibility adapter for injected synchronous grouped callables."""

    def __init__(
        self, backend: "NpuBackend", shape: Any, grouped_entry: tuple[Any, Any, Any]
    ) -> None:
        self.backend = backend
        self.shape = shape
        self.run, self.x, self.y = grouped_entry
        self._calls: list[Any | None] = [None, None]
        self._outputs: list[np.ndarray | None] = [None, None]

    def prepare(self, io: int, call: Any, values: np.ndarray) -> None:
        self._calls[io] = call
        self.backend._overwrite_buffer(
            self.x,
            _grouped_x_tiled(values, self.shape).reshape(-1).view(bfloat16),
        )

    def start(self, io: int, layer: int) -> Any:
        call = self._calls[io]
        if call is None:
            raise RuntimeError("grouped callable runner was started before prepare")
        weight = self.backend._get_prefill_weight_tensor(
            f"L{layer:02d}/moe_{self.shape.name}"
        )
        scalars = _grouped_dispatch_scalars(self.shape, call)
        self.run(self.y, weight, self.x, **scalars)
        self._outputs[io] = np.array(
            self.y.numpy().view(np.uint16), copy=True
        ).reshape(SLOTS, TMAX, self.shape.m)
        return object()

    def wait(self, run: Any) -> None:
        del run

    def read(self, io: int) -> np.ndarray:
        output = self._outputs[io]
        if output is None:
            raise RuntimeError("grouped callable output is not ready")
        return np.array(output, copy=True)


@dataclass(eq=False)
class _PendingLinear:
    """A ``linear_start`` token (DESIGN-prefill.md section 48); ``run`` is None when ``linear_finish`` does the GEMV."""

    name: str
    x: np.ndarray
    bundle: str | None = None
    entry: Any = None
    run: Any = None
    dt_start: float = 0.0


# -----------------------------------------------------------------------------
# NpuBackend Class
# -----------------------------------------------------------------------------
class NpuBackend(Backend):
    """Production NPU computation backend for Raon-1-mini with hardware execution and mock fallbacks."""

    _BASE_TIMING_KEYS = (
        "linear_npu",
        "linear_host",
        "rda_npu",
        "rda_npu_prefill",
        "rda_host",
        "moe_b1_npu",
        "moe_sh_npu",
        "moe_dn_npu",
        "moe_grouped_b1_npu",
        "moe_grouped_b2_npu",
        "moe_grouped_host",
        "moe_pipe_wait",
        "moe_host",
        "mla_attend",
        "mla_host",
        "host_prep",
        "rda_pipe_stage",
        "rda_pipe_launch",
        "rda_pipe_wait",
        "rda_pipe_read",
    )

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        host_weights: Weights | None = None,
        mock: bool = False,
        device: str = "npu",
        gemv: str | None = None,
    ):
        # GEMV engine selection: constructor arg takes precedence over RAON_NPU_GEMV
        if gemv is None:
            gemv = os.environ.get("RAON_NPU_GEMV", "unified")
        if gemv not in ("unified", "legacy"):
            raise ValueError(f"Invalid gemv engine {gemv!r}; choose 'unified' or 'legacy'")
        self.gemv: str = gemv

        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else ((self.weights_dir / "manifest.json") if self.weights_dir else None)
        )
        self.host_weights = host_weights
        self.mock = mock or (not HAVE_NPU) or (device == "cpu")
        self.device = device

        # Xclbin tracking metrics
        self.xclbin_calls: dict[str, int] = defaultdict(int)
        self.xclbin_switches: int = 0
        self._last_active_xclbin: str | None = None
        self._unified_xclbin_path: Path | None = None
        self._prefill_xclbin_path: Path | None = None
        self.prefill_gemm_enabled = os.environ.get("RAON_PREFILL_GEMM", "1") != "0"
        self.prefill_rda_enabled = os.environ.get("RAON_PREFILL_RDA", "1") != "0"
        self.prefill_share_weights = os.environ.get("RAON_PREFILL_SHARE_WEIGHTS", "1") != "0"
        self._prefill_shared_precompute: bool | None = None
        self._prefill_grouped_moe: bool | None = None
        self._prefill_rda_overlap: bool | None = None
        self._prefill_moe_overlap: bool | None = None
        self._decode_direct: bool | None = None
        self._decode_direct_env = os.environ.get("RAON_DECODE_DIRECT", "1") not in (
            "0", "false", "False", "no"
        )
        self._decode_direct_warned: set[str] = set()
        self._decode_unified_xclbin_resolved: Path | None = None
        self._decode_moe_eligibility: tuple[
            tuple[Any, Any], bool, Any, tuple[bool, str]
        ] | None = None
        self._decode_moe_overlap: bool | None = None
        self._decode_moe_overlap_env = os.environ.get("RAON_DECODE_MOE_OVERLAP", "1") not in (
            "0", "false", "False", "no"
        )
        self._moe_shared_pre: tuple[int, Any, Any, np.ndarray] | None = None
        self._decode_mla_group: bool | None = None
        self._decode_mla_group_env = os.environ.get("RAON_DECODE_MLA_GROUP", "1") not in (
            "0", "false", "False", "no"
        )
        self._decode_linear_start: bool | None = None
        self._decode_linear_start_env = os.environ.get("RAON_DECODE_LINEAR_START", "1") not in (
            "0", "false", "False", "no"
        )
        self._linear_pending: list[_PendingLinear] = []
        self._decode_rda_eligibility: tuple[Any, tuple[bool, str]] | None = None

        # RDA overlapped pipelining state
        self._rda_live: list[_RdaTicket] = []
        self._rda_head_busy: list[bool] = [False, False]
        self._rda_out_pending: list[bool] = [False, False]
        self._rda_io_next: int = 0
        self._rda_prefill: dict[int, _RdaPrefillStream] | None = None

        # Load specs
        self.comp_map, self.bundle_map = build_default_manifest_specs()
        if self.manifest_path and self.manifest_path.is_file():
            self._load_manifest(self.manifest_path)
        elif not self.mock:
            raise FileNotFoundError(
                f"manifest.json is required for NPU execution: {self.manifest_path}"
            )
        elif self.gemv == "unified":
            self.validate_manifest_shapes()

        # Execution tracking and caches
        self._step_id: int = 0
        self._bundle_cache: dict[str, tuple[int, int, np.ndarray]] = {}  # bundle -> (step, id(x), y_out)
        self._cached_inputs: dict[str, np.ndarray] = {}  # hold reference to avoid id reuse
        self._batch_bundle_cache: dict[str, tuple[int, int, np.ndarray]] = {}
        self._batch_cached_inputs: dict[str, np.ndarray] = {}
        self._cached_shared: dict[int, dict[str, Any]] = {}  # layer -> {step, sh_gate, sh_up, sh_act_u16, sh_down}

        # Model states
        self.rda_states: dict[int, list[Any]] = {}  # layer -> [s0, s1] (ping-pong)
        self.rda_active_slot: dict[int, int] = {}   # layer -> active index (0 or 1)
        self.mla_key_cache: dict[int, np.ndarray] = {}
        self.mla_value_cache: dict[int, np.ndarray] = {}
        self._mla_caches: dict[int, _MlaLayerCache] = {}

        # Operator callables and hardware buffers
        self._compiled_ops: dict[str, Any] = {}
        self._npu_weights: dict[str, Any] = {}
        self._prefill_weights: dict[str, Any] = {}
        self._work_buffers: dict[str, Any] = {}

        # Mock reference storage for testing without NPU
        self.mock_weight_arrays: dict[str, np.ndarray] = {}
        self._mock_d_weights: DWeights | None = None

        # Timing and profiling metrics
        self.timings: dict[str, float] = {}
        self.call_counts: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = self.call_counts
        self.reset_timings()
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Lazy init on first use or explicit load
        if self.weights_dir and not self.mock:
            self._init_hardware_operators()

    @property
    def prefill_shared_precompute(self) -> bool:
        if not self.prefill_gemm_enabled:
            return False
        if self._prefill_shared_precompute is not None:
            return bool(self._prefill_shared_precompute)
        return os.environ.get("RAON_PREFILL_SHARED_GEMM", "1") not in ("0", "false", "False", "no")

    @prefill_shared_precompute.setter
    def prefill_shared_precompute(self, value: bool | None) -> None:
        self._prefill_shared_precompute = value

    @property
    def prefill_grouped_moe(self) -> bool:
        if not self.prefill_shared_precompute or self.gemv != "unified":
            return False
        if self._prefill_grouped_moe is not None:
            return bool(self._prefill_grouped_moe)
        return os.environ.get("RAON_PREFILL_GROUPED_MOE", "1") not in (
            "0", "false", "False", "no"
        )

    @prefill_grouped_moe.setter
    def prefill_grouped_moe(self, value: bool | None) -> None:
        self._prefill_grouped_moe = value

    @property
    def prefill_rda_overlap(self) -> bool:
        if self._prefill_rda_overlap is not None:
            return bool(self._prefill_rda_overlap)
        return os.environ.get("RAON_PREFILL_RDA_OVERLAP", "1") not in (
            "0", "false", "False", "no"
        )

    @prefill_rda_overlap.setter
    def prefill_rda_overlap(self, value: bool | None) -> None:
        self._prefill_rda_overlap = value

    @property
    def prefill_moe_overlap(self) -> bool:
        if self._prefill_moe_overlap is not None:
            return bool(self._prefill_moe_overlap)
        return os.environ.get("RAON_PREFILL_MOE_OVERLAP", "1") not in (
            "0", "false", "False", "no"
        )

    @prefill_moe_overlap.setter
    def prefill_moe_overlap(self, value: bool | None) -> None:
        self._prefill_moe_overlap = value

    @property
    def decode_direct(self) -> bool:
        if self._decode_direct is not None:
            return bool(self._decode_direct)
        return self._decode_direct_env

    @decode_direct.setter
    def decode_direct(self, value: bool | None) -> None:
        self._decode_direct = value

    @property
    def decode_moe_overlap(self) -> bool:
        if self._decode_moe_overlap is not None:
            return bool(self._decode_moe_overlap)
        return self._decode_moe_overlap_env

    @decode_moe_overlap.setter
    def decode_moe_overlap(self, value: bool | None) -> None:
        self._decode_moe_overlap = value

    @property
    def decode_mla_group(self) -> bool:
        if self._decode_mla_group is not None:
            return bool(self._decode_mla_group)
        return self._decode_mla_group_env

    @decode_mla_group.setter
    def decode_mla_group(self, value: bool | None) -> None:
        self._decode_mla_group = value

    @property
    def decode_linear_start(self) -> bool:
        if self._decode_linear_start is not None:
            return bool(self._decode_linear_start)
        return self._decode_linear_start_env

    @decode_linear_start.setter
    def decode_linear_start(self, value: bool | None) -> None:
        self._decode_linear_start = value

    def _decode_runner(self) -> _DecodeDirectRunner | None:
        if not (
            self.decode_direct
            and self.gemv == "unified"
            and not self.mock
            and HAVE_NPU
            and self._unified_xclbin_path is not None
        ):
            return None
        runner = self._compiled_ops.get("decode_direct")
        if runner is None:
            runner = _make_decode_direct_runner(self._unified_xclbin_path)
            self._compiled_ops["decode_direct"] = runner
            gemv_specs = self._collect_decode_gemv_specs()
            moe_specs = self._collect_decode_moe_specs()
            rda_op = self._compiled_ops.get("rda_step_op")
            if not self._decode_rda_eligible(rda_op)[0]:
                rda_op = None
            runner.prepare(gemv_specs, moe_specs, rda_op, self._decode_settle_s())
        return runner

    def _decode_settle_s(self) -> float:
        raw = os.environ.get("RAON_DECODE_DIRECT_SETTLE_MS", "500")
        try:
            ms = int(raw)
            if ms < 0:
                raise ValueError
        except ValueError:
            ms = 500
            key = "RAON_DECODE_DIRECT_SETTLE_MS"
            if key not in self._decode_direct_warned:
                print(
                    f"[decode-direct] invalid {key}; using default 500ms",
                    file=sys.stderr,
                    flush=True,
                )
                self._decode_direct_warned.add(key)
        return ms / 1000.0

    def _collect_decode_gemv_specs(self) -> dict[str, tuple[Any, int, int]]:
        specs: dict[str, tuple[Any, int, int]] = {}
        for bname, bmeta in self.bundle_map.items():
            if bmeta.fmt not in ("q4nx_gemv", "bfp16_gemv"):
                continue
            fn = self._get_or_compile_gemv(bmeta.k, bmeta.m, bmeta.fmt)
            op = getattr(fn, "op", None)
            if self._decode_op_eligible(op)[0]:
                specs[bname] = (op, bmeta.k, bmeta.m)
        return specs

    def _collect_decode_moe_specs(self) -> dict[str, tuple[Any, int, int]]:
        specs: dict[str, tuple[Any, int, int]] = {}
        kernels = self._compiled_ops.get("unified_moe_kernels")
        if not self._decode_moe_eligible(False)[0]:
            return specs
        if not isinstance(kernels, tuple) or len(kernels) != 2:
            return specs
        gu_kernel, dn_kernel = kernels
        from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT, NUM_ACTIVE_TOTAL

        specs["b1"] = (gu_kernel, MAX_K, 6 * B1_M_PER_EXPERT)
        specs["b2"] = (dn_kernel, 7 * MAX_K, NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT)
        return specs

    def _decode_fallback(self, family: str, reason: str) -> None:
        self._add_timing(f"decode_direct_fallback_{family}", 0.0, count=1)
        if family not in self._decode_direct_warned:
            print(
                f"[decode-direct] {family}: IRON path ({reason})",
                file=sys.stderr,
                flush=True,
            )
            self._decode_direct_warned.add(family)

    def _decode_unified_path(self) -> Path | None:
        if self._unified_xclbin_path is None:
            return None
        if self._decode_unified_xclbin_resolved is None:
            self._decode_unified_xclbin_resolved = Path(
                self._unified_xclbin_path
            ).resolve()
        return self._decode_unified_xclbin_resolved

    def _decode_op_eligible(self, op: Any) -> tuple[bool, str]:
        if op is None:
            return False, "operator metadata is missing"
        xclbin = getattr(op, "xclbin_artifact", None)
        insts = getattr(op, "insts_artifact", None)
        filename = getattr(xclbin, "filename", None)
        kernel_name = getattr(xclbin, "kernel_name", None)
        insts_name = getattr(insts, "filename", None)
        if not filename or not kernel_name or not insts_name:
            return False, "operator artifacts are incomplete"
        unified = self._decode_unified_path()
        if unified is None:
            return False, "unified xclbin is not prepared"
        if Path(filename).resolve() != unified:
            return False, "operator xclbin does not match the unified xclbin"
        return True, ""

    def _decode_moe_eligible(
        self, need_shared: bool, shared_fn: Any = None
    ) -> tuple[bool, str]:
        kernels = self._compiled_ops.get("unified_moe_kernels")
        shared_op = getattr(shared_fn, "op", None) if need_shared else None
        cached = self._decode_moe_eligibility
        if (
            cached is not None
            and cached[0] is kernels
            and cached[1] == need_shared
            and cached[2] is shared_op
        ):
            return cached[3]
        if not isinstance(kernels, tuple) or len(kernels) != 2:
            result = (False, "MoE kernel templates are missing")
            self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
            return result
        unified = self._decode_unified_path()
        if unified is None:
            result = (False, "unified xclbin is not prepared")
            self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
            return result
        for kernel in kernels:
            if not hasattr(kernel, "instructions") or not hasattr(kernel, "dispatch_params"):
                result = (False, "MoE kernel template metadata is incomplete")
                self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
                return result
            path = getattr(kernel, "xclbin_path", None)
            if not path or Path(path).resolve() != unified:
                result = (False, "MoE kernel xclbin does not match the unified xclbin")
                self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
                return result
        if need_shared:
            eligible, reason = self._decode_op_eligible(shared_op)
            if not eligible:
                result = (False, f"shared GEMV: {reason}")
                self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
                return result
        result = (True, "")
        self._decode_moe_eligibility = (kernels, need_shared, shared_op, result)
        return result

    def _decode_rda_eligible(self, op: Any) -> tuple[bool, str]:
        cached = self._decode_rda_eligibility
        if cached is not None and cached[0] is op:
            return cached[1]
        result = self._decode_op_eligible(op)
        self._decode_rda_eligibility = (op, result)
        return result

    def rda_overlap_ready(self) -> bool:
        return bool(
            self.prefill_rda_overlap
            and (self.mock or not HAVE_NPU or self._use_prefill_rda())
        )

    def reset_timings(self) -> None:
        """Reset per-category timing accumulators, call counts, and xclbin counters."""
        self.timings = {k: 0.0 for k in self._BASE_TIMING_KEYS}
        self.call_counts.clear()
        self.xclbin_calls.clear()
        self.xclbin_switches = 0

    def _record_xclbin_use(self, xclbin_id: str) -> None:
        """Track active xclbin and count context switches."""
        if self._last_active_xclbin is not None and self._last_active_xclbin != xclbin_id:
            self.xclbin_switches += 1
        self._last_active_xclbin = xclbin_id
        self.xclbin_calls[xclbin_id] = self.xclbin_calls.get(xclbin_id, 0) + 1

    def validate_manifest_shapes(self) -> None:
        """Ensure all NPU bundles in bundle_map map to known unified GEMV shapes."""
        if self.gemv != "unified":
            return
        unmapped = []
        for bname, bmeta in self.bundle_map.items():
            try:
                resolve_unified_shape_name(bmeta.k, bmeta.m, bmeta.fmt, bundle_name=bname)
            except (ValueError, KeyError) as err:
                unmapped.append((bname, bmeta.k, bmeta.m, bmeta.fmt, str(err)))
        if unmapped:
            details = "\n".join(f"  - {u[0]}: k={u[1]}, m={u[2]}, fmt={u[3]} ({u[4]})" for u in unmapped)
            raise ValueError(
                f"NpuBackend(gemv='unified') found {len(unmapped)} bundle(s) without valid unified GEMV shape mapping:\n"
                f"{details}"
            )

    def _add_timing(self, key: str, dt: float, count: int = 1) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + dt
        if count:
            self.call_counts[key] = self.call_counts.get(key, 0) + count

    def set_step(self, step: int) -> None:
        """Advance token decode step ID for cache invalidation."""
        self._drain_linear_pending()
        self._drain_moe_shared_pre()
        self._step_id = step
        # Clear per-step caches while preserving recurrent/KV state
        self._bundle_cache.clear()
        self._cached_inputs.clear()
        self._batch_bundle_cache.clear()
        self._batch_cached_inputs.clear()
        self._cached_shared.clear()

    def reset(self) -> None:
        """Reset internal recurrent and cache states."""
        self._drain_linear_pending()
        self.rda_abort_all()
        self.moe_abort_prestart_shared()
        self._step_id = 0
        self._bundle_cache.clear()
        self._cached_inputs.clear()
        self._batch_bundle_cache.clear()
        self._batch_cached_inputs.clear()
        self._cached_shared.clear()
        self.rda_states.clear()
        self.rda_active_slot.clear()
        self.mla_key_cache.clear()
        self.mla_value_cache.clear()
        self._mla_caches.clear()
        self.reset_timings()

    def _load_manifest(self, path: Path) -> None:
        """Validate manifest.json against default specs and take bundle file paths."""
        _load_manifest_specs(
            path,
            self.comp_map,
            self.bundle_map,
            weights_dir=self.weights_dir,
        )
        if self.gemv == "unified":
            self.validate_manifest_shapes()

    def _get_mock_d_weights(self) -> DWeights | None:
        """Lazily create a DWeights decoder when running mock mode against a real D weights dir."""
        if self._mock_d_weights is not None:
            return self._mock_d_weights
        if self.weights_dir is not None and self.weights_dir.is_dir():
            self._mock_d_weights = DWeights(
                self.weights_dir,
                host_weights=self.host_weights,
                manifest_path=self.manifest_path if (self.manifest_path and self.manifest_path.is_file()) else None,
            )
            return self._mock_d_weights
        return None

    # -------------------------------------------------------------------------
    # Hardware Operator Initialization and Preloading
    # -------------------------------------------------------------------------
    def _get_or_compile_gemv(self, k: int, m: int, fmt: str) -> Callable[..., None]:
        """Compile or retrieve cached static GEMV operator callable."""
        if self.gemv == "unified":
            shape_name = resolve_unified_shape_name(k, m, fmt)
            op_key = f"unified_{shape_name}"
            if op_key in self._compiled_ops:
                return self._compiled_ops[op_key]

            if self.mock:
                raise RuntimeError(f"Mock mode does not compile hardware operators ({op_key})")

            ctx = AIEContext() if callable(AIEContext) else None
            from kernels.gemv_unified.op import UnifiedGEMV
            op = UnifiedGEMV.for_shape(
                shape_name,
                cols=UNIFIED_COLS,
                rda_col=UNIFIED_RDA_COL,
                context=ctx,
                continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W,
                sequence_plan=unified_sequence_plan(shape_name),
            )
            op.compile()
            self._unified_xclbin_path = _require_single_unified_xclbin(
                [op], expected=self._unified_xclbin_path
            )
            fn = op.get_callable()
            self._compiled_ops[op_key] = fn
            return fn
        else:
            op_key = f"legacy_{fmt}_k{k}_m{m}"
            if op_key in self._compiled_ops:
                return self._compiled_ops[op_key]

            if self.mock:
                raise RuntimeError(f"Mock mode does not compile hardware operators ({op_key})")

            ctx = AIEContext() if callable(AIEContext) else None
            if fmt == "q4nx_gemv":
                from kernels.gemv.op import GEMV
                op = GEMV(K=k, M=m, cols=8, context=ctx)
            elif fmt == "bfp16_gemv":
                from kernels.gemv_bfp16.op import GEMV_BFP16
                op = GEMV_BFP16(K=k, M=m, cols=8, context=ctx)
            else:
                raise ValueError(f"Unknown GEMV format: {fmt}")

            op.compile()
            fn = op.get_callable()
            self._compiled_ops[op_key] = fn
            return fn

    def _get_or_compile_moe_ops(self) -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
        """Compile or retrieve B1-6, Shared GU, and B2-9 MoE operators."""
        op_key = f"{self.gemv}_moe_ops"
        if op_key in self._compiled_ops:
            return self._compiled_ops[op_key]

        if self.mock:
            raise RuntimeError(f"Mock mode does not compile hardware operators ({op_key})")

        ctx = AIEContext() if callable(AIEContext) else None
        if self.gemv == "unified":
            from kernels.gemv_unified.op import UnifiedGEMV
            from kernels.gemv_unified.stage2 import UnifiedMoEDown9, UnifiedMoEGateUp
            gu = UnifiedMoEGateUp(
                context=ctx,
                continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W,
                sequence_plan=unified_sequence_plan("moe_b1_6"),
                cols=UNIFIED_COLS,
                rda_col=UNIFIED_RDA_COL,
            )
            sh = UnifiedGEMV.for_shape(
                "shared_gu",
                cols=UNIFIED_COLS,
                rda_col=UNIFIED_RDA_COL,
                context=ctx,
                continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W,
                sequence_plan=unified_sequence_plan("shared_gu"),
            )
            dn = UnifiedMoEDown9(
                context=ctx,
                continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W,
                sequence_plan=unified_sequence_plan("moe_b2_9"),
                cols=UNIFIED_COLS,
                rda_col=UNIFIED_RDA_COL,
            )
            for op in (gu, sh, dn):
                op.compile()
            self._unified_xclbin_path = _require_single_unified_xclbin(
                (gu, sh, dn), expected=self._unified_xclbin_path
            )
            gu_kernel = gu.get_callable()
            dn_kernel = dn.get_callable()
            self._compiled_ops["unified_moe_kernels"] = (gu_kernel, dn_kernel)
            fns = (
                self._prepare_dynamic_kernel(gu_kernel),
                sh.get_callable(),
                self._prepare_dynamic_kernel(dn_kernel),
            )
            self._compiled_ops[op_key] = fns
            return fns
        else:
            from kernels.gemv_bfp16.op import GEMV_BFP16
            from kernels.moe.op import MoEDownBfp16, MoEGateUp
            from kernels.moe.pack import B1_K, B1_M_PER_EXPERT

            gu = MoEGateUp(slots=6, context=ctx)
            sh = GEMV_BFP16(K=B1_K, M=B1_M_PER_EXPERT, cols=8, context=ctx)
            dn = MoEDownBfp16(context=ctx)

            for op in (gu, sh, dn):
                op.compile()

            gu_kernel = gu.get_callable()
            dn_kernel = dn.get_callable()
            fns = (
                self._prepare_dynamic_kernel(gu_kernel),
                sh.get_callable(),
                self._prepare_dynamic_kernel(dn_kernel),
            )
            self._compiled_ops[op_key] = fns
            return fns

    def _get_or_compile_grouped_moe(self) -> dict[str, tuple[Callable[..., None], Any, Any]]:
        """Compile and cache the two mode-specific grouped MoE callables and BOs."""
        op_key = "grouped_moe"
        cached = self._compiled_ops.get(op_key)
        if cached is not None:
            return cached
        if self.mock:
            raise RuntimeError("Mock mode does not compile grouped MoE operators")

        from kernels.moe_grouped.op import GroupedMoE, validate_design_pair

        ctx = AIEContext() if callable(AIEContext) else None
        grouped: dict[str, tuple[Callable[..., None], Any, Any]] = {}
        kernels: dict[str, Any] = {}
        xclbins: dict[str, str] = {}
        for shape in (GROUPED_B1, GROUPED_B2):
            op = GroupedMoE(shape.name, context=ctx)
            config_mode, stream_mode = op.artifact_designs
            validate_design_pair(config_mode, stream_mode)
            op.compile()
            fn = op.get_callable()
            kernels[shape.name] = fn
            validate_design_pair(shape, fn.shape)
            path = str(getattr(fn, "xclbin_path", getattr(fn, "_xclbin_path", "")))
            if not path:
                raise RuntimeError(f"grouped {shape.name} callable has no xclbin path")
            if f"_{shape.name}_" not in Path(path).name:
                raise RuntimeError(
                    f"grouped {shape.name} xclbin does not identify its mode: {path}"
                )
            xclbins[shape.name] = path
            run = self._prepare_dynamic_kernel(fn)
            x_tensor = aie_utils.tensor(
                np.zeros(SLOTS * TMAX * shape.k, dtype=bfloat16)
            )
            y_tensor = aie_utils.full(
                (SLOTS * TMAX * shape.m,), 0.0, dtype=bfloat16
            )
            grouped[shape.name] = (run, x_tensor, y_tensor)
        if xclbins[GROUPED_B1.name] == xclbins[GROUPED_B2.name]:
            raise RuntimeError(
                "grouped B1/B2 must use separate mode-specific xclbins: "
                f"{xclbins[GROUPED_B1.name]}"
            )
        self._compiled_ops[op_key] = grouped
        self._compiled_ops["grouped_moe_kernels"] = kernels
        return grouped

    @staticmethod
    def _is_xrt_einval(exc: RuntimeError) -> bool:
        """Match XRT IOCTL EINVAL messages like the original CallableDesign path."""
        msg = str(exc)
        return "err=-22" in msg or (
            "Invalid argument" in msg and ("XRT" in msg or "xrt" in msg)
        )

    @staticmethod
    def _prepare_dynamic_kernel(kernel: Any) -> Callable[..., None]:
        """Load a patched-template kernel once and reuse its runtime handle."""
        runtime = aie_utils.DefaultNPURuntime
        handle = runtime.load(kernel)
        dispatch_names = set(kernel.dispatch_params)

        def run(*args: Any, **kwargs: Any) -> None:
            nonlocal handle
            if not getattr(handle, "_is_valid", True):
                handle = runtime.load(kernel)
            unknown = set(kwargs) - dispatch_names
            if unknown:
                raise TypeError(f"Unexpected keyword argument(s): {sorted(unknown)}")
            dispatch_scalars = {
                name: value for name, value in kwargs.items() if name in dispatch_names
            }
            instructions = kernel.instructions(**dispatch_scalars)
            runtime.run(handle, list(args), dispatch_insts=instructions)

        return run

    @staticmethod
    def _prepare_rda_kernel() -> Callable[..., None]:
        """Specialize and load the fixed-shape RDA design once at startup."""
        from bench.rda_state.rda_state import rda_step

        ensure_device = getattr(aie_utils, "ensure_current_device", None)
        if ensure_device is not None:
            ensure_device()
        compile_kwargs = {
            "num_heads": 40,
            "num_workers": 4,
            "use_memtile": True,
        }
        sample_args = (
            np.empty((40, 128, 128), dtype=np.float32),
            np.empty((40, 128, 128), dtype=np.float32),
            np.empty((40, 640), dtype=np.float32),
            np.empty((40, 128), dtype=np.float32),
        )
        compilable = rda_step._build_compilable(compile_kwargs, sample_args)
        kernel = rda_step._compile_and_build_kernel(compilable, None, None)
        runtime = aie_utils.DefaultNPURuntime
        handle = runtime.load(kernel)

        def run(*args: Any) -> None:
            nonlocal handle, kernel
            try:
                if not getattr(handle, "_is_valid", True):
                    handle = runtime.load(kernel)
                return runtime.run(handle, list(args))
            except RuntimeError as exc:
                if not NpuBackend._is_xrt_einval(exc):
                    raise
                # Evict the stale context and rebuild exactly once, matching the
                # original CallableDesign retry behavior. This also covers the
                # case where the fresh handle loaded above fails on its first run.
                if getattr(kernel, "xclbin_path", None) is not None:
                    runtime.evict_context(kernel.xclbin_path)
                compilable = rda_step._build_compilable(compile_kwargs, sample_args)
                kernel = rda_step._compile_and_build_kernel(compilable, None, None)
                handle = runtime.load(kernel)
                return runtime.run(handle, list(args))

        return run

    def _prepare_unified_rda_kernel(self) -> Callable[..., None]:
        """Compile and initialize the merged UnifiedRDA operator in the unified xclbin."""
        from kernels.gemv_unified.rda import UnifiedRDA

        ctx = AIEContext() if callable(AIEContext) else None
        rda_op = UnifiedRDA(
            cols=UNIFIED_COLS,
            rda_col=UNIFIED_RDA_COL,
            continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W,
            sequence_plan=unified_sequence_plan("rda_in"),
            context=ctx,
        )
        rda_op.compile()
        self._unified_xclbin_path = _require_single_unified_xclbin(
            [rda_op], expected=self._unified_xclbin_path
        )
        self._compiled_ops["rda_step_op"] = rda_op
        return rda_op.get_callable()

    def _get_weight_tensor(self, bundle_name: str) -> Any:
        """Load and cache weight buffer on NPU as uint32 words."""
        if bundle_name in self._npu_weights:
            return self._npu_weights[bundle_name]

        if self.mock:
            return self.mock_weight_arrays.get(bundle_name)

        if not self.weights_dir:
            raise FileNotFoundError(f"weights_dir not provided; cannot load {bundle_name}")

        bmeta = self.bundle_map.get(bundle_name)
        if not bmeta:
            raise KeyError(f"Bundle {bundle_name} not found in metadata")

        raw_bytes = _read_verified_bundle(self.weights_dir, bmeta)
        if len(raw_bytes) % np.dtype(np.uint32).itemsize:
            raise ValueError(f"Weight binary is not uint32-aligned: {bmeta.rel_file}")
        words = np.frombuffer(raw_bytes, dtype=np.uint32)
        tensor = aie_utils.tensor(words)
        self._npu_weights[bundle_name] = tensor
        return tensor

    def _get_prefill_weight_tensor(self, bundle_name: str) -> Any:
        """Return the decode tensor, or an explicitly selected second-context copy."""
        if self.prefill_share_weights:
            return self._get_weight_tensor(bundle_name)
        if bundle_name in self._prefill_weights:
            return self._prefill_weights[bundle_name]
        if self.mock:
            return self.mock_weight_arrays.get(bundle_name)
        if not self.weights_dir:
            raise FileNotFoundError(f"weights_dir not provided; cannot load {bundle_name}")
        bmeta = self.bundle_map[bundle_name]
        raw_bytes = _read_verified_bundle(self.weights_dir, bmeta)
        if len(raw_bytes) % np.dtype(np.uint32).itemsize:
            raise ValueError(f"Weight binary is not uint32-aligned: {bmeta.rel_file}")
        tensor = aie_utils.tensor(np.frombuffer(raw_bytes, dtype=np.uint32))
        self._prefill_weights[bundle_name] = tensor
        return tensor

    def _get_or_compile_prefill_gemm(self, shape_name: str) -> Callable[..., None]:
        """Lazily compile one shape command stream against the shared prefill body."""
        op_key = f"prefill_gemm:{shape_name}"
        if op_key in self._compiled_ops:
            return self._compiled_ops[op_key]
        if self.mock:
            raise RuntimeError(f"Mock mode does not compile hardware operators ({op_key})")
        from kernels.gemm_prefill.op import PrefillGEMM

        ctx = AIEContext() if callable(AIEContext) else None
        op = PrefillGEMM.for_shape(
            shape_name,
            batch_t=PREFILL_GEMM_MAX_T,
            max_batch_t=PREFILL_GEMM_MAX_T,
            cols=7,
            rda_col=7,
            context=ctx,
        )
        op.compile()
        path = _xclbin_path_from_op(op)
        if self._prefill_xclbin_path is not None and path != self._prefill_xclbin_path:
            raise RuntimeError(
                f"prefill GEMM xclbin path mismatch: {path} != {self._prefill_xclbin_path}"
            )
        self._prefill_xclbin_path = path
        fn = op.get_callable()
        self._compiled_ops[op_key] = fn
        return fn

    def _get_or_compile_prefill_rda(self) -> Callable[..., None]:
        """Prepare the RDA streams (T = 1, 2, 4, 8) from the already-compiled prefill plate.

        Returns the T = 1 callable for the synchronous path, whose [40, 640] / [40, 128] BOs have the bytes of the
        T = 1 stream's [40, 1, 640] / [40, 4, 1, 32]; the overlapped calls use ``_rda_prefill``.
        """
        op_key = "prefill_rda"
        if op_key in self._compiled_ops:
            return self._compiled_ops[op_key]
        if self.mock:
            raise RuntimeError(f"Mock mode does not compile hardware operators ({op_key})")
        if self._prefill_xclbin_path is None:
            raise RuntimeError("prefill RDA requires an already-compiled prefill plate")

        from kernels.gemm_prefill.op import PrefillGEMM

        ctx = AIEContext() if callable(AIEContext) else None
        streams: dict[int, _RdaPrefillStream] = {}
        fn = None
        for tokens in RDA_TOKEN_CHOICES:
            op = PrefillGEMM.for_shape(
                "rda_in",
                batch_t=PREFILL_GEMM_MAX_T,
                max_batch_t=PREFILL_GEMM_MAX_T,
                cols=7,
                rda_col=7,
                execution="rda",
                rda_tokens=tokens,
                context=ctx,
            )
            op.compile()
            path = _xclbin_path_from_op(op)
            if path != self._prefill_xclbin_path:
                raise RuntimeError(
                    f"prefill RDA xclbin path mismatch: {path} != {self._prefill_xclbin_path}"
                )
            streams[tokens] = _RdaPrefillStream(
                tokens=tokens,
                runner=_make_xrt_direct_runner(op),
                heads=tuple(
                    iron.zeros((40, tokens, 640), dtype=np.float32, device="npu") for _ in range(2)
                ),
                outs=tuple(
                    iron.zeros((40, 4, tokens, 32), dtype=np.float32, device="npu") for _ in range(2)
                ),
            )
            if tokens == 1:
                fn = op.get_callable()
        self._rda_prefill = streams
        self._compiled_ops[op_key] = fn
        return fn

    def _prefill_rda_stream(self, tokens: int) -> _RdaPrefillStream:
        """The prepared prefill RDA stream of ``tokens`` tokens per call (compiled on first use)."""
        if self._rda_prefill is None:
            self._get_or_compile_prefill_rda()
        if self._rda_prefill is None:
            raise RuntimeError("prefill RDA streams are not initialized")
        return self._rda_prefill[tokens]

    def _use_prefill_rda(self) -> bool:
        """Select prefill RDA only while its compiled plate is the active context."""
        return (
            self.prefill_rda_enabled
            and self._prefill_xclbin_path is not None
            and self._last_active_xclbin == PREFILL_GEMM_XCLBIN_ID
        )

    def preload(self, num_layers: int | None = None) -> dict[str, Any]:
        """Preload all (or up to num_layers) weight bundles into NPU tensors before decode."""
        t0 = time.perf_counter()
        target_bundles: list[BundleMeta] = []
        for bname, bmeta in self.bundle_map.items():
            if num_layers is not None and bname.startswith("L") and "/" in bname:
                ly_str = bname[1 : bname.index("/")]
                if ly_str.isdigit() and int(ly_str) >= num_layers:
                    continue
            target_bundles.append(bmeta)

        total_bytes = 0
        loaded_count = 0
        total_bundles = len(target_bundles)
        print(
            f"[preload] Preloading {total_bundles} NPU weight bundles "
            f"(mock={self.mock})...",
            flush=True,
        )

        for idx, bmeta in enumerate(target_bundles, 1):
            bname = bmeta.name
            if self.mock:
                arr = self.mock_weight_arrays.get(bname)
                if arr is not None:
                    nbytes = int(arr.nbytes)
                    self._npu_weights[bname] = arr
                    total_bytes += nbytes
                    loaded_count += 1
                elif self.weights_dir is not None:
                    bin_path = self.weights_dir / bmeta.rel_file
                    if bin_path.is_file():
                        nbytes = int(bin_path.stat().st_size)
                        total_bytes += nbytes
                        loaded_count += 1
            else:
                tensor = self._get_weight_tensor(bname)
                nbytes = int(getattr(tensor, "nbytes", 0))
                if nbytes == 0 and self.weights_dir is not None:
                    bin_path = self.weights_dir / bmeta.rel_file
                    if bin_path.is_file():
                        nbytes = int(bin_path.stat().st_size)
                total_bytes += nbytes
                loaded_count += 1

            if idx == 1 or idx == total_bundles or idx % 20 == 0:
                print(
                    f"[preload] ({idx}/{total_bundles}) {bname}: "
                    f"cumulative {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB)",
                    flush=True,
                )

        dt = time.perf_counter() - t0
        print(
            f"[preload] Done: {loaded_count}/{total_bundles} bundles, "
            f"{total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB) in {dt:.2f}s",
            flush=True,
        )
        self._decode_runner()
        return {
            "bundles_loaded": loaded_count,
            "bundles_total": total_bundles,
            "total_bytes": total_bytes,
            "seconds": dt,
        }

    def _prepared_buffer(self, name: str) -> Any:
        """Return a work buffer created during hardware operator preparation.

        Raises RuntimeError if the buffer is missing, which means
        `_init_hardware_operators` was not called (or did not complete) before
        attempting NPU execution.
        """
        buf = self._work_buffers.get(name)
        if buf is None:
            raise RuntimeError(
                f"{name} work buffer missing — hardware operators were not prepared"
            )
        return buf

    @staticmethod
    def _overwrite_buffer(tensor: Any, value: np.ndarray) -> None:
        """Replace a prepared tensor without first pulling stale device bytes."""
        overwrite = getattr(tensor, "overwrite", None)
        if overwrite is None:
            # Lightweight fake tensors used by the CPU-only test suite do not
            # implement IRON's coherence API.
            tensor[:] = value
            return
        with overwrite() as host_view:
            host_view[...] = value

    def _init_hardware_operators(self) -> None:
        """Pre-allocate static work buffers for zero-allocation decode loop."""
        if self.mock or not HAVE_NPU:
            return
        from kernels.moe.pack import B1_K, B1_M_PER_EXPERT, B2_K_PAD, B2_M_PER_EXPERT
        self._work_buffers["moe_y_gu"] = aie_utils.full((6 * B1_M_PER_EXPERT,), 0.0, dtype=bfloat16)
        self._work_buffers["moe_y_sh"] = aie_utils.full((B1_M_PER_EXPERT,), 0.0, dtype=bfloat16)
        self._work_buffers["moe_y_dn"] = aie_utils.full((7 * B2_M_PER_EXPERT,), 0.0, dtype=bfloat16)
        if self.gemv == "unified":
            self._work_buffers["moe_x_gu"] = aie_utils.tensor(np.zeros(MAX_K, dtype=bfloat16))
            self._work_buffers["moe_x_dn"] = aie_utils.tensor(np.zeros(7 * MAX_K, dtype=bfloat16))
            self._work_buffers["gemv_x_unified"] = aie_utils.tensor(np.zeros(MAX_K, dtype=bfloat16))
        else:
            self._work_buffers["moe_x_gu"] = aie_utils.tensor(np.zeros(B1_K, dtype=bfloat16))
            self._work_buffers["moe_x_dn"] = aie_utils.tensor(np.zeros(7 * B2_K_PAD, dtype=bfloat16))
            for k in {meta.k for meta in self.bundle_map.values()}:
                self._work_buffers[f"gemv_x_{k}"] = aie_utils.tensor(
                    np.zeros(k, dtype=bfloat16)
                )
        for m in {meta.m for meta in self.bundle_map.values()}:
            self._work_buffers[f"gemv_y_{m}"] = aie_utils.full(
                (m,), 0.0, dtype=bfloat16
            )
        self._work_buffers["rda_head_in"] = iron.zeros(
            (40, 640), dtype=np.float32, device="npu"
        )
        self._work_buffers["rda_out"] = iron.zeros(
            (40, 128), dtype=np.float32, device="npu"
        )
        if self.gemv == "unified":
            self._get_or_compile_moe_ops()
            self._compiled_ops["rda_step"] = self._prepare_unified_rda_kernel()
        else:
            self._compiled_ops["rda_step"] = self._prepare_rda_kernel()
            self._get_or_compile_moe_ops()

    # -------------------------------------------------------------------------
    # Linear Projection with Bundled GEMV and Reuse (1D and 2D / ND inputs)
    # -------------------------------------------------------------------------
    def linear_batch(self, name: str, x: np.ndarray) -> np.ndarray:
        """Run an exact bundled prefill GEMM for ``x[T,K]`` when supported."""
        self._drain_linear_pending()
        if not isinstance(x, np.ndarray) or x.dtype != np.uint16 or x.ndim != 2:
            raise ValueError("linear_batch x must be a [T,K] uint16 BF16-bit ndarray")
        if x.shape[0] < 1:
            raise ValueError("linear_batch requires T >= 1")

        comp = self.comp_map.get(name)
        if comp is not None and x.shape[1] != comp.orig_k:
            raise ValueError(
                f"linear_batch input K={x.shape[1]} does not match {name} K={comp.orig_k}"
            )
        if comp is None or not self.prefill_gemm_enabled:
            return self.linear(name, x)
        if comp.fmt == "mla_kv_b":
            return self._linear_batch_mla_kv_b(comp, x)
        if comp.fmt == "shared_down":
            return self.linear(name, x)

        bmeta = self.bundle_map.get(comp.bundle_name)
        if bmeta is None:
            return self.linear(name, x)
        try:
            shape_name = resolve_unified_shape_name(
                bmeta.k, bmeta.m, bmeta.fmt, comp.bundle_name
            )
        except (KeyError, ValueError):
            return self.linear(name, x)
        if shape_name not in PREFILL_GEMM_SHAPES:
            return self.linear(name, x)

        cached = self._batch_bundle_cache.get(comp.bundle_name)
        cached_input = self._batch_cached_inputs.get(comp.bundle_name)
        if cached is not None:
            c_step, c_id, y_full = cached
            if c_step == self._step_id and c_id == id(x) and cached_input is x:
                self.cache_hits += 1
                return y_full[:, comp.row_start : comp.row_end].copy()

        self.cache_misses += 1
        if comp.padded_k > comp.orig_k:
            x_padded = np.zeros((x.shape[0], comp.padded_k), dtype=np.uint16)
            x_padded[:, : comp.orig_k] = x
        else:
            x_padded = x

        chunks = []
        for start in range(0, x.shape[0], PREFILL_GEMM_MAX_T):
            chunks.append(
                self._execute_gemm_batch(
                    comp.bundle_name,
                    bmeta,
                    shape_name,
                    x_padded[start : start + PREFILL_GEMM_MAX_T],
                )
            )
        y_full = np.concatenate(chunks, axis=0)
        self._batch_bundle_cache[comp.bundle_name] = (self._step_id, id(x), y_full)
        self._batch_cached_inputs[comp.bundle_name] = x
        return y_full[:, comp.row_start : comp.row_end].copy()

    def _linear_batch_mla_kv_b(
        self, comp: ComponentMeta, x: np.ndarray
    ) -> np.ndarray:
        """Run k_b/v_b GEMMs and interleave each head as [k_h, v_h]."""
        cache_key = comp.bundle_name
        cached = self._batch_bundle_cache.get(cache_key)
        cached_input = self._batch_cached_inputs.get(cache_key)
        if cached is not None:
            c_step, c_id, y_full = cached
            if c_step == self._step_id and c_id == id(x) and cached_input is x:
                self.cache_hits += 1
                return y_full.copy()

        self.cache_misses += 1
        outputs = []
        for bundle_name in (comp.extra["kb_bundle"], comp.extra["vb_bundle"]):
            bmeta = self.bundle_map[bundle_name]
            chunks = [
                self._execute_gemm_batch(
                    bundle_name,
                    bmeta,
                    "k_b_v_b",
                    x[start : start + PREFILL_GEMM_MAX_T],
                )
                for start in range(0, x.shape[0], PREFILL_GEMM_MAX_T)
            ]
            outputs.append(np.concatenate(chunks, axis=0))

        t_h0 = time.perf_counter()
        k_heads = outputs[0].reshape(x.shape[0], 40, 128)
        v_heads = outputs[1].reshape(x.shape[0], 40, 128)
        y_full = np.concatenate([k_heads, v_heads], axis=-1).reshape(
            x.shape[0], comp.orig_m
        )
        self._batch_bundle_cache[cache_key] = (self._step_id, id(x), y_full)
        self._batch_cached_inputs[cache_key] = x
        dt_h = time.perf_counter() - t_h0
        self._add_timing("linear_host", dt_h, count=1)
        self._add_timing("host_prep", dt_h, count=1)
        return y_full.copy()

    def _execute_gemm_batch(
        self,
        bundle_name: str,
        bmeta: BundleMeta,
        shape_name: str,
        x_u16: np.ndarray,
    ) -> np.ndarray:
        """Execute one 1..PREFILL_GEMM_MAX_T-row exact GEMM, zero-padding the physical BO to PREFILL_GEMM_MAX_T rows."""
        actual_t = x_u16.shape[0]
        if not 1 <= actual_t <= PREFILL_GEMM_MAX_T:
            raise ValueError(f"prefill chunk T must be in 1..{PREFILL_GEMM_MAX_T}")
        timing_key = f"gemm:{shape_name}"

        if self.mock:
            t0 = time.perf_counter()
            if bundle_name in self.mock_weight_arrays:
                w_mat = self.mock_weight_arrays[bundle_name]
                w_f32 = (
                    bf16_to_f32(w_mat)
                    if w_mat.dtype == np.uint16
                    else w_mat.astype(np.float32)
                )
            else:
                dw = self._get_mock_d_weights()
                if dw is None or self.weights_dir is None or not (
                    self.weights_dir / bmeta.rel_file
                ).is_file():
                    raise FileNotFoundError(
                        f"No mock array or D bundle file available for {bundle_name}"
                    )
                w_f32 = dw._decode_2d_bundle(bundle_name)
            rows = [
                f32_to_bf16_rtne(
                    np.dot(bf16_to_f32(row[: bmeta.k]), w_f32.T)
                )
                for row in x_u16
            ]
            result = np.stack(rows, axis=0)
            if self._prefill_xclbin_path is None:
                self._prefill_xclbin_path = Path(f"{PREFILL_GEMM_XCLBIN_ID}.xclbin")
            self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
        else:
            fn = self._get_or_compile_prefill_gemm(shape_name)
            w_tensor = self._get_prefill_weight_tensor(bundle_name)
            x_key = f"prefill_x:{shape_name}"
            y_key = f"prefill_y:{shape_name}"
            if x_key not in self._work_buffers:
                self._work_buffers[x_key] = aie_utils.tensor(
                    np.zeros(PREFILL_GEMM_MAX_T * bmeta.k, dtype=bfloat16)
                )
                self._work_buffers[y_key] = aie_utils.full(
                    (PREFILL_GEMM_MAX_T * bmeta.m,), 0.0, dtype=bfloat16
                )
            x_pad = np.zeros((PREFILL_GEMM_MAX_T, bmeta.k), dtype=np.uint16)
            x_pad[:actual_t] = x_u16[:, : bmeta.k]
            x_tensor = self._prepared_buffer(x_key)
            y_tensor = self._prepared_buffer(y_key)
            self._overwrite_buffer(x_tensor, x_pad.reshape(-1).view(bfloat16))
            t0 = time.perf_counter()
            fn(y_tensor, w_tensor, x_tensor)
            self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
            result = np.array(y_tensor.numpy().view(np.uint16), copy=True).reshape(
                PREFILL_GEMM_MAX_T, bmeta.m
            )[:actual_t]

        dt = time.perf_counter() - t0
        self._add_timing(timing_key, dt, count=1)
        self._add_timing("linear_npu", dt, count=1)
        return result

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        """Matrix multiplication with bundled GEMV reuse and slicing."""
        t_h0 = time.perf_counter()
        is_f32 = (x.dtype == np.float32)
        x_u16 = f32_to_bf16_rtne(x) if is_f32 else np.asarray(x, dtype=np.uint16)

        comp = self.comp_map.get(name)

        # Fallback to host_weights directly when running mock mode without mock_weight_arrays
        # or when testing custom-dimension tiny models
        if comp is None or (
            self.mock
            and comp.bundle_name not in self.mock_weight_arrays
            and self.host_weights is not None
            and name in self.host_weights
        ):
            if self.host_weights is not None and name in self.host_weights:
                w_f32 = self.host_weights.get(name, to_float=True)
                x_f32 = x if is_f32 else bf16_to_f32(x)
                y_f32 = np.dot(x_f32, w_f32.T)
                bias_name = name[:-6] + "bias" if name.endswith("weight") else name + ".bias"
                if bias_name in self.host_weights:
                    y_f32 += self.host_weights.get(bias_name, to_float=True)
                out = y_f32.astype(np.float32) if is_f32 else f32_to_bf16_rtne(y_f32)
                dt_h = time.perf_counter() - t_h0
                self._add_timing("linear_host", dt_h, count=1)
                self._add_timing("host_prep", dt_h, count=1)
                return out
            # In mock mode with a D weights directory, allow routed experts via _mock_d_weights
            if self.mock:
                dw = self._get_mock_d_weights()
                if dw is not None and name in dw:
                    w_f32 = dw.get(name, to_float=True)
                    x_f32 = x if is_f32 else bf16_to_f32(x)
                    y_f32 = np.dot(x_f32, w_f32.T)
                    out = y_f32.astype(np.float32) if is_f32 else f32_to_bf16_rtne(y_f32)
                    dt_h = time.perf_counter() - t_h0
                    self._add_timing("linear_host", dt_h, count=1)
                    self._add_timing("host_prep", dt_h, count=1)
                    return out
            raise KeyError(f"Weight parameter '{name}' not found in NPU bundle map or host weights")

        # Special Case 1: MLA kv_b_proj combined projection
        if comp.fmt == "mla_kv_b":
            dt_h = time.perf_counter() - t_h0
            self._add_timing("linear_host", dt_h, count=0)
            self._add_timing("host_prep", dt_h, count=0)
            if x_u16.ndim == 1:
                return self._linear_mla_kv_b(comp, x_u16, is_f32, key=x)
            lead_shape = x_u16.shape[:-1]
            rows = x_u16.reshape(-1, x_u16.shape[-1])
            out_rows = [self._linear_mla_kv_b(comp, r, is_f32=False) for r in rows]
            stacked = (
                np.stack(out_rows, axis=0)
                if out_rows
                else np.empty((0, comp.orig_m), dtype=np.uint16)
            ).reshape(*lead_shape, comp.orig_m)
            return bf16_to_f32(stacked) if is_f32 else stacked

        # Special Case 2: Shared expert down projection via MoE B2
        if comp.fmt == "shared_down":
            dt_h = time.perf_counter() - t_h0
            self._add_timing("linear_host", dt_h, count=0)
            self._add_timing("host_prep", dt_h, count=0)
            if x_u16.ndim == 1:
                return self._linear_shared_down(comp, x_u16, is_f32)
            lead_shape = x_u16.shape[:-1]
            rows = x_u16.reshape(-1, x_u16.shape[-1])
            out_rows = [self._linear_shared_down(comp, r, is_f32=False) for r in rows]
            stacked = (
                np.stack(out_rows, axis=0)
                if out_rows
                else np.empty((0, comp.orig_m), dtype=np.uint16)
            ).reshape(*lead_shape, comp.orig_m)
            return bf16_to_f32(stacked) if is_f32 else stacked

        bundle_name = comp.bundle_name
        bmeta = self.bundle_map[bundle_name]

        # ---------------------------------------------------------------------
        # Check Bundled GEMV Cache
        # Reuse rule: same bundle, same token step, and same input object identity
        # ---------------------------------------------------------------------
        cached_entry = self._bundle_cache.get(bundle_name)
        if cached_entry is not None:
            c_step, c_id, cached_out = cached_entry
            cached_input = self._cached_inputs.get(bundle_name)
            if c_step == self._step_id and c_id == id(x) and (x is cached_input):
                self.cache_hits += 1
                sliced = cached_out[..., comp.row_start : comp.row_end].copy()
                out = bf16_to_f32(sliced) if is_f32 else sliced
                dt_h = time.perf_counter() - t_h0
                self._add_timing("linear_host", dt_h, count=1)
                self._add_timing("host_prep", dt_h, count=1)
                return out

        self.cache_misses += 1

        # ---------------------------------------------------------------------
        # Execute Bundled GEMV on NPU (or Mock reference)
        # Supports 1D [K] and multi-dimensional [..., K] inputs row by row.
        # ---------------------------------------------------------------------
        if x_u16.ndim == 1:
            if comp.padded_k > comp.orig_k:
                x_in = np.zeros(comp.padded_k, dtype=np.uint16)
                x_in[: comp.orig_k] = x_u16
            else:
                x_in = x_u16
            dt_h1 = time.perf_counter() - t_h0
            self._add_timing("linear_host", dt_h1, count=0)
            self._add_timing("host_prep", dt_h1, count=0)

            y_full = self._execute_gemv(bundle_name, bmeta, x_in)
        else:
            lead_shape = x_u16.shape[:-1]
            rows = x_u16.reshape(-1, x_u16.shape[-1])
            dt_h1 = time.perf_counter() - t_h0
            self._add_timing("linear_host", dt_h1, count=0)
            self._add_timing("host_prep", dt_h1, count=0)

            out_rows = []
            for r in rows:
                t_pad = time.perf_counter()
                if comp.padded_k > comp.orig_k:
                    x_in = np.zeros(comp.padded_k, dtype=np.uint16)
                    x_in[: comp.orig_k] = r
                else:
                    x_in = r
                dt_pad = time.perf_counter() - t_pad
                self._add_timing("linear_host", dt_pad, count=0)
                self._add_timing("host_prep", dt_pad, count=0)
                out_rows.append(self._execute_gemv(bundle_name, bmeta, x_in))

            t_stack = time.perf_counter()
            y_full = (
                np.stack(out_rows, axis=0)
                if out_rows
                else np.empty((0, bmeta.m), dtype=np.uint16)
            ).reshape(*lead_shape, bmeta.m)
            dt_stack = time.perf_counter() - t_stack
            self._add_timing("linear_host", dt_stack, count=0)
            self._add_timing("host_prep", dt_stack, count=0)

        t_h2 = time.perf_counter()
        self._bundle_cache[bundle_name] = (self._step_id, id(x), y_full)
        self._cached_inputs[bundle_name] = x

        sliced = y_full[..., comp.row_start : comp.row_end].copy()
        out = bf16_to_f32(sliced) if is_f32 else sliced
        dt_h2 = time.perf_counter() - t_h2
        self._add_timing("linear_host", dt_h2, count=1)
        self._add_timing("host_prep", dt_h2, count=1)
        return out

    def prefetch_linears(self, items: tuple[tuple[str, np.ndarray], ...]) -> None:
        """Run the decode GEMVs behind several ``linear(name, x)`` calls back to back (DESIGN-prefill.md section 47).

        Only with ``decode_mla_group`` on the decode direct path, for 1-D inputs of bundled GEMV or MLA kv_b
        components whose direct entries already exist; otherwise nothing happens and ``linear`` computes as before.
        Every entry is fetched (the only handle checks) before any run starts; then every GEMV starts, then each one
        finishes in order. Outputs go into the bundle cache under linear's own reuse rule (same step, same input
        object), so the following ``linear`` calls return the bytes they would have computed.
        """
        self._drain_linear_pending()
        if not self.decode_mla_group:
            return
        decode_runner = self._decode_runner()
        if decode_runner is None:
            return
        t_h0 = time.perf_counter()
        work: list[tuple[str, np.ndarray, np.ndarray]] = []  # (bundle, linear's input object, GEMV input)
        for name, x in items:
            comp = self.comp_map.get(name)
            if comp is None or np.ndim(x) != 1:
                return
            x_u16 = f32_to_bf16_rtne(x) if x.dtype == np.float32 else np.asarray(x, dtype=np.uint16)
            if comp.fmt == "mla_kv_b":
                gemvs = ((comp.extra["kb_bundle"], x_u16), (comp.extra["vb_bundle"], x_u16))
            elif comp.fmt in ("q4nx_gemv", "bfp16_gemv"):
                x_in = x_u16
                if comp.padded_k > comp.orig_k:
                    x_in = np.zeros(comp.padded_k, dtype=np.uint16)
                    x_in[: comp.orig_k] = x_u16
                gemvs = ((comp.bundle_name, x_in),)
            else:
                return
            for bundle, x_in in gemvs:
                # One start per bundle and call (a direct entry has a single run): a second input for the same bundle,
                # or a bundle already cached for this step and this input, is left to its own linear call.
                cached = self._bundle_cache.get(bundle)
                if any(b == bundle for b, _, _ in work) or (
                    cached is not None
                    and cached[0] == self._step_id
                    and cached[1] == id(x)
                    and self._cached_inputs.get(bundle) is x
                ):
                    continue
                work.append((bundle, x, x_in))
        if not work:
            return
        entries = [decode_runner.gemv_cached_entry(bundle) for bundle, _, _ in work]
        if any(entry is None for entry in entries):
            return
        dt_h = time.perf_counter() - t_h0
        self._add_timing("linear_host", dt_h, count=0)
        self._add_timing("host_prep", dt_h, count=0)

        runs: list[Any] = []
        dt_start: list[float] = []
        try:
            for entry, (_, _, x_in) in zip(entries, work):
                t0 = time.perf_counter()
                runs.append(decode_runner.gemv_start(entry, x_in))
                dt_start.append(time.perf_counter() - t0)
        except BaseException:
            # A run that never started returns from drain at once (its wait raises, which drain swallows).
            decode_runner.drain([entry.run for entry in entries[: len(runs) + 1]])
            raise
        t_npu = time.perf_counter()
        for i, (entry, run, (bundle, x, _)) in enumerate(zip(entries, runs, work)):
            try:
                y = decode_runner.gemv_finish(entry, run)
            except BaseException:
                decode_runner.drain([later.run for later in entries[i:]])
                raise
            t_done = time.perf_counter()
            # Its own start plus the wait since the previous finish: the recorded times add up to the group's time.
            dt = dt_start[i] + (t_done - t_npu)
            self._bundle_cache[bundle] = (self._step_id, id(x), y)
            self._cached_inputs[bundle] = x
            self.cache_misses += 1
            self._record_xclbin_use("gemv_unified")
            self._add_timing(entry.timing_key, dt, count=1)
            self._add_timing("linear_npu", dt, count=1)
            self._add_timing("decode_direct_gemv", 0.0, count=1)
            t_npu = t_done
        self._add_timing("decode_mla_group", 0.0, count=1)

    def _drain_linear_pending(self) -> None:
        """Wait out GEMVs started by linear_start and not yet finished (DESIGN-prefill.md section 48)."""
        if not self._linear_pending:
            return
        pending, self._linear_pending = self._linear_pending, []
        runner = self._compiled_ops.get("decode_direct")
        if runner is not None:
            runner.drain([p.entry.run for p in pending])

    def linear_start(self, name: str, x: np.ndarray) -> _PendingLinear:
        """Start ``linear(name, x)``'s decode GEMV now; ``linear_finish`` waits for it (DESIGN-prefill.md section 48).

        Only with ``decode_linear_start`` on the decode direct path, for a 1-D input of a bundled GEMV component whose
        bundle is not cached for this step and input and whose direct entry already exists; otherwise the GEMV is left
        to ``linear_finish``, which then calls ``linear``. The entry is fetched (the only handle check) before the run
        starts, and every other entry point of this backend that starts NPU runs or checks handles waits out a started
        run first, so no handle check or reload happens while it is in flight. ``x`` must not change before the finish.
        """
        self._drain_linear_pending()
        t_h0 = time.perf_counter()
        pending = _PendingLinear(name, x)
        comp = self.comp_map.get(name)
        if (
            not self.decode_linear_start
            or comp is None
            or np.ndim(x) != 1
            or comp.fmt not in ("q4nx_gemv", "bfp16_gemv")
        ):
            return pending
        decode_runner = self._decode_runner()
        if decode_runner is None:
            return pending
        cached = self._bundle_cache.get(comp.bundle_name)
        if (
            cached is not None
            and cached[0] == self._step_id
            and cached[1] == id(x)
            and self._cached_inputs.get(comp.bundle_name) is x
        ):
            return pending
        entry = decode_runner.gemv_cached_entry(comp.bundle_name)
        if entry is None:
            return pending
        x_u16 = f32_to_bf16_rtne(x) if x.dtype == np.float32 else np.asarray(x, dtype=np.uint16)
        x_in = x_u16
        if comp.padded_k > comp.orig_k:
            x_in = np.zeros(comp.padded_k, dtype=np.uint16)
            x_in[: comp.orig_k] = x_u16
        dt_prep = time.perf_counter() - t_h0
        self._add_timing("linear_host", dt_prep, count=0)
        self._add_timing("host_prep", dt_prep, count=0)
        t0 = time.perf_counter()
        try:
            pending.run = decode_runner.gemv_start(entry, x_in)
            pending.bundle, pending.entry = comp.bundle_name, entry
            self._linear_pending.append(pending)
        except BaseException:
            decode_runner.drain([entry.run])
            if pending in self._linear_pending:
                self._linear_pending.remove(pending)
            pending.run = None
            raise
        pending.dt_start = time.perf_counter() - t0
        return pending

    def linear_finish(self, pending: _PendingLinear) -> np.ndarray:
        """Wait for a ``linear_start`` GEMV and return exactly ``linear(name, x)`` (DESIGN-prefill.md section 48).

        The output goes into the bundle cache under linear's reuse rule, so ``linear`` returns it with its own
        slicing. A token without a run, or whose run was already waited out elsewhere, is computed by ``linear``.
        The GEMV's kind key and ``linear_npu`` get the host time blocked on it (its start call and the wait here), as
        prefetch_linears does; the part hidden behind the caller's own work is not counted.
        """
        if pending.run is None or pending not in self._linear_pending:
            return self.linear(pending.name, pending.x)
        self._linear_pending.remove(pending)
        runner = self._compiled_ops["decode_direct"]
        t0 = time.perf_counter()
        try:
            y = runner.gemv_finish(pending.entry, pending.run)
        except BaseException:
            runner.drain([pending.entry.run])
            raise
        dt = pending.dt_start + (time.perf_counter() - t0)
        self._bundle_cache[pending.bundle] = (self._step_id, id(pending.x), y)
        self._cached_inputs[pending.bundle] = pending.x
        self.cache_misses += 1
        self._record_xclbin_use("gemv_unified")
        self._add_timing(pending.entry.timing_key, dt, count=1)
        self._add_timing("linear_npu", dt, count=1)
        self._add_timing("decode_direct_gemv", 0.0, count=1)
        return self.linear(pending.name, pending.x)

    def linear_abort(self, pending: _PendingLinear) -> None:
        """Drop a ``linear_start`` whose result is not needed: wait out its run (DESIGN-prefill.md section 48)."""
        if pending.run is not None and pending in self._linear_pending:
            self._linear_pending.remove(pending)
            runner = self._compiled_ops.get("decode_direct")
            if runner is not None:
                runner.drain([pending.entry.run])

    def _execute_gemv(self, bundle_name: str, bmeta: BundleMeta, x_u16: np.ndarray) -> np.ndarray:
        """Run single 1D GEMV (q4nx or bfp16) on NPU or mock reference."""
        self._drain_linear_pending()
        decode_runner = self._decode_runner()
        if decode_runner is not None:
            t_prep0 = time.perf_counter()
            entry = decode_runner.gemv_cached_entry(bundle_name)
            if entry is not None:
                dt_prep = time.perf_counter() - t_prep0
                self._add_timing("linear_host", dt_prep, count=0)
                self._add_timing("host_prep", dt_prep, count=0)

                t_npu0 = time.perf_counter()
                out_copy = decode_runner.gemv(entry, x_u16)
                self._record_xclbin_use("gemv_unified")
                dt_npu = time.perf_counter() - t_npu0
                self._add_timing(entry.timing_key, dt_npu, count=1)
                self._add_timing("linear_npu", dt_npu, count=1)
                self._add_timing("decode_direct_gemv", 0.0, count=1)
                return out_copy

        k = bmeta.k
        m = bmeta.m
        fmt = bmeta.fmt
        kind = bundle_name.split("/")[-1]
        gemv_key = f"gemv:{kind}"

        if self.gemv == "unified":
            shape_name = resolve_unified_shape_name(k, m, fmt, bundle_name)
            xclbin_id = "gemv_unified"
        else:
            shape_name = None
            xclbin_id = legacy_gemv_xclbin_name(fmt, k, m)

        if self.mock:
            t0 = time.perf_counter()
            if bundle_name in self.mock_weight_arrays:
                w_mat = self.mock_weight_arrays[bundle_name]
                w_f32 = bf16_to_f32(w_mat) if w_mat.dtype == np.uint16 else w_mat.astype(np.float32)
            else:
                dw = self._get_mock_d_weights()
                if dw is not None and self.weights_dir is not None and (self.weights_dir / bmeta.rel_file).is_file():
                    w_f32 = dw._decode_2d_bundle(bundle_name)
                else:
                    raise FileNotFoundError(
                        f"No mock array or D bundle file available for {bundle_name}"
                    )
            y_f32 = np.dot(bf16_to_f32(x_u16[:k]), w_f32.T)
            out = f32_to_bf16_rtne(y_f32)
            self._record_xclbin_use(xclbin_id)
            dt = time.perf_counter() - t0
            self._add_timing(gemv_key, dt, count=1)
            self._add_timing("linear_npu", dt, count=1)
            return out

        # Host preparation: retrieve compiled op, weight tensor, work buffers, and write input bits
        t_prep0 = time.perf_counter()
        fn = self._get_or_compile_gemv(k, m, fmt)
        w_tensor = self._get_weight_tensor(bundle_name)

        if decode_runner is not None:
            eligible, reason = self._decode_op_eligible(getattr(fn, "op", None))
            if eligible:
                entry = decode_runner.gemv_entry(
                    bundle_name, fn.op, w_tensor, k, m, shape_name, gemv_key,
                    already_checked=True,
                )
                dt_prep = time.perf_counter() - t_prep0
                self._add_timing("linear_host", dt_prep, count=0)
                self._add_timing("host_prep", dt_prep, count=0)

                t_npu0 = time.perf_counter()
                out_copy = decode_runner.gemv(entry, x_u16)
                self._record_xclbin_use(xclbin_id)
                dt_npu = time.perf_counter() - t_npu0
                self._add_timing(gemv_key, dt_npu, count=1)
                self._add_timing("linear_npu", dt_npu, count=1)
                self._add_timing("decode_direct_gemv", 0.0, count=1)
                return out_copy
            self._decode_fallback("gemv", reason)

        y_buf_key = f"gemv_y_{m}"

        if self.gemv == "unified":
            x_tensor = self._prepared_buffer("gemv_x_unified")
            x_pad = pad_x(x_u16[:k], shape_name)
            self._overwrite_buffer(
                x_tensor, np.ascontiguousarray(x_pad, dtype=np.uint16).view(bfloat16)
            )
        else:
            x_buf_key = f"gemv_x_{k}"
            x_tensor = self._prepared_buffer(x_buf_key)
            # Reinterpret uint16 BF16 bits as bfloat16 (do NOT numerically cast uint16 -> bfloat16)
            self._overwrite_buffer(
                x_tensor,
                np.ascontiguousarray(x_u16[:k], dtype=np.uint16).view(bfloat16),
            )

        y_tensor = self._prepared_buffer(y_buf_key)
        dt_prep = time.perf_counter() - t_prep0
        self._add_timing("linear_host", dt_prep, count=0)
        self._add_timing("host_prep", dt_prep, count=0)

        # Pure NPU execution + output read (copy out of reusable work buffer)
        t_npu0 = time.perf_counter()
        fn(y_tensor, w_tensor, x_tensor)
        self._record_xclbin_use(xclbin_id)
        out_arr = y_tensor.numpy()
        out_copy = np.array(out_arr.view(np.uint16), copy=True)
        dt_npu = time.perf_counter() - t_npu0

        self._add_timing(gemv_key, dt_npu, count=1)
        self._add_timing("linear_npu", dt_npu, count=1)
        return out_copy

    def _cached_gemv_output(self, bundle_name: str, key: Any) -> np.ndarray | None:
        """The bundle cache's GEMV output for this step and the input object ``key`` (linear's reuse rule), or None."""
        if key is None:
            return None
        cached = self._bundle_cache.get(bundle_name)
        if (
            cached is None
            or cached[0] != self._step_id
            or cached[1] != id(key)
            or self._cached_inputs.get(bundle_name) is not key
        ):
            return None
        self.cache_hits += 1
        return cached[2]

    def _linear_mla_kv_b(
        self, comp: ComponentMeta, x_u16: np.ndarray, is_f32: bool, key: Any = None
    ) -> np.ndarray:
        """MLA kv_b_proj executing k_b and v_b GEMVs and interleaving 40 heads [k_h, v_h].

        With ``key`` (linear's input object), k_b / v_b outputs that prefetch_linears put in the bundle cache for this
        step and this object are used instead of running the GEMVs (DESIGN-prefill.md section 47).
        """
        kb_bundle = comp.extra["kb_bundle"]
        vb_bundle = comp.extra["vb_bundle"]
        kb_meta = self.bundle_map[kb_bundle]
        vb_meta = self.bundle_map[vb_bundle]

        k_out = self._cached_gemv_output(kb_bundle, key)
        if k_out is None:
            k_out = self._execute_gemv(kb_bundle, kb_meta, x_u16)  # [5120] (40 heads x 128)
        v_out = self._cached_gemv_output(vb_bundle, key)
        if v_out is None:
            v_out = self._execute_gemv(vb_bundle, vb_meta, x_u16)  # [5120] (40 heads x 128)

        t_h0 = time.perf_counter()
        k_heads = k_out.reshape(40, 128)
        v_heads = v_out.reshape(40, 128)
        kv_heads = np.concatenate([k_heads, v_heads], axis=-1).reshape(10240)
        out = bf16_to_f32(kv_heads) if is_f32 else kv_heads
        dt_h = time.perf_counter() - t_h0
        self._add_timing("linear_host", dt_h, count=1)
        self._add_timing("host_prep", dt_h, count=1)
        return out

    def _linear_shared_down(self, comp: ComponentMeta, x_u16: np.ndarray, is_f32: bool) -> np.ndarray:
        """Shared expert down projection (expert 128 of moe_b2).

        Uses cached B2 slot-6 output ONLY when step matches and input BF16 bits
        equal the activation placed into B2 slot 6 during moe_experts; raises otherwise.
        """
        t_h0 = time.perf_counter()
        layer_parts = comp.hf_name.split(".")
        layer_idx = int(layer_parts[2]) if len(layer_parts) > 2 and layer_parts[2].isdigit() else 0

        x_flat = np.asarray(x_u16, dtype=np.uint16).reshape(-1)
        cached = self._cached_shared.get(layer_idx)
        if cached is None:
            raise ValueError(
                f"Shared down projection '{comp.hf_name}' called without a valid "
                f"B2 slot-6 cache entry for layer {layer_idx} at step {self._step_id}"
            )

        c_step = cached.get("step")
        sh_act_u16 = cached.get("sh_act_u16")
        sh_down = cached.get("sh_down")
        if (
            c_step != self._step_id
            or sh_act_u16 is None
            or sh_down is None
            or x_flat.shape != sh_act_u16.shape
            or not np.array_equal(x_flat, sh_act_u16)
        ):
            raise ValueError(
                f"Shared down cache mismatch at layer {layer_idx}: "
                f"step ({c_step} vs {self._step_id}) or input activation bits "
                "do not match B2 slot-6 input"
            )

        self.cache_hits += 1
        out = bf16_to_f32(sh_down) if is_f32 else sh_down.copy()
        dt_h = time.perf_counter() - t_h0
        self._add_timing("linear_host", dt_h, count=1)
        self._add_timing("host_prep", dt_h, count=1)
        return out

    # -------------------------------------------------------------------------
    # RDA Recurrent State Update (Double-Buffered Ping-Pong)
    # -------------------------------------------------------------------------
    def rda_update(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        log_alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """V-first delta rule recurrence on NPU with double-buffered recurrent state."""
        self._drain_linear_pending()
        if self._rda_live:
            raise RuntimeError(
                "Cannot execute synchronous rda_update while overlapped RDA tickets are active"
            )
        t_h0 = time.perf_counter()
        # pack_head_bundle/exp only read these arrays; same-dtype inputs need no copy.
        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32, copy=False)
        k_f32 = bf16_to_f32(k) if k.dtype == np.uint16 else k.astype(np.float32, copy=False)
        v_f32 = bf16_to_f32(v) if v.dtype == np.uint16 else v.astype(np.float32, copy=False)
        log_alpha_f32 = log_alpha.astype(np.float32, copy=False)
        beta_f32 = bf16_to_f32(beta) if beta.dtype == np.uint16 else beta.astype(np.float32, copy=False)

        H, V, K = 40, 128, 128

        decode_xclbin_id = "gemv_unified" if self.gemv == "unified" else "rda_step"
        use_prefill_rda = self._use_prefill_rda()
        xclbin_id = PREFILL_GEMM_XCLBIN_ID if use_prefill_rda else decode_xclbin_id

        if self.mock or not HAVE_NPU:
            dt_h = time.perf_counter() - t_h0
            self._add_timing("rda_host", dt_h, count=1)
            self._add_timing("host_prep", dt_h, count=1)
            t_npu0 = time.perf_counter()
            out = self._rda_update_numpy(layer, q_f32, k_f32, v_f32, log_alpha_f32, beta_f32)
            self._record_xclbin_use(xclbin_id)
            dt_npu = time.perf_counter() - t_npu0
            self._add_timing("rda_npu", dt_npu, count=1)
            if use_prefill_rda:
                self._add_timing("rda_npu_prefill", dt_npu, count=1)
            self._add_timing("rda", dt_npu, count=1)
            return out

        # ---------------------------------------------------------------------
        # NPU RDA Step with Ping-Pong Double Buffering
        # ---------------------------------------------------------------------
        from bench.rda_state.rda_state import pack_head_bundle

        if layer not in self.rda_states:
            s0 = iron.zeros((H, V, K), dtype=np.float32, device="npu")
            s1 = iron.zeros((H, V, K), dtype=np.float32, device="npu")
            self.rda_states[layer] = [s0, s1]
            self.rda_active_slot[layer] = 0

        slot = self.rda_active_slot[layer]
        s_in = self.rda_states[layer][slot]
        s_out = self.rda_states[layer][1 - slot]

        run_rda = (
            self._get_or_compile_prefill_rda()
            if use_prefill_rda
            else self._compiled_ops.get("rda_step")
        )
        if run_rda is None:
            raise RuntimeError("rda_step kernel was not prepared")

        a_f32 = np.exp(log_alpha_f32)
        bundle_host = pack_head_bundle(q_f32, k_f32, a_f32, v_f32, beta_f32)
        decode_runner = (
            self._decode_runner()
            if self.gemv == "unified" and not use_prefill_rda
            else None
        )
        direct_op = self._compiled_ops.get("rda_step_op")
        direct = False
        if decode_runner is not None:
            eligible, reason = self._decode_rda_eligible(direct_op)
            if eligible:
                direct = True
            else:
                self._decode_fallback("rda", reason)

        if direct:
            dt_prep = time.perf_counter() - t_h0
            t_npu0 = time.perf_counter()
            out_f32 = decode_runner.rda(direct_op, s_in, s_out, bundle_host)
            self._record_xclbin_use(xclbin_id)
            self.rda_active_slot[layer] = 1 - slot
            dt_npu = time.perf_counter() - t_npu0
            self._add_timing("decode_direct_rda", 0.0, count=1)
        else:
            head_in_npu = self._prepared_buffer("rda_head_in")
            out_npu = self._prepared_buffer("rda_out")
            self._overwrite_buffer(head_in_npu, bundle_host)
            dt_prep = time.perf_counter() - t_h0

            t_npu0 = time.perf_counter()
            run_rda(
                s_in,
                s_out,
                head_in_npu,
                out_npu,
            )
            self._record_xclbin_use(xclbin_id)
            self.rda_active_slot[layer] = 1 - slot
            out_f32 = np.array(out_npu.numpy(), copy=True)
            dt_npu = time.perf_counter() - t_npu0

        t_post0 = time.perf_counter()
        res = f32_to_bf16_rtne(out_f32)
        dt_post = time.perf_counter() - t_post0

        self._add_timing("rda_npu", dt_npu, count=1)
        if use_prefill_rda:
            self._add_timing("rda_npu_prefill", dt_npu, count=1)
        self._add_timing("rda", dt_npu, count=1)
        self._add_timing("rda_host", dt_prep + dt_post, count=1)
        self._add_timing("host_prep", dt_prep + dt_post, count=1)
        return res

    def _rda_update_numpy(
        self,
        layer: int,
        q_f32: np.ndarray,
        k_f32: np.ndarray,
        v_f32: np.ndarray,
        log_alpha_f32: np.ndarray,
        beta_f32: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference RDA update."""
        H, K = k_f32.shape[-2], k_f32.shape[-1]
        V = v_f32.shape[-1]
        if layer not in self.rda_states:
            self.rda_states[layer] = [np.zeros((H, V, K), dtype=np.float32), None]
            self.rda_active_slot[layer] = 0

        slot = self.rda_active_slot[layer]
        state, out = self._rda_numpy_step(
            self.rda_states[layer][slot], q_f32, k_f32, v_f32, log_alpha_f32, beta_f32
        )
        self.rda_states[layer][slot] = state
        return out

    @staticmethod
    def _rda_numpy_step(
        state: np.ndarray,
        q_f32: np.ndarray,
        k_f32: np.ndarray,
        v_f32: np.ndarray,
        log_alpha_f32: np.ndarray,
        beta_f32: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One NumPy reference RDA token on ``state`` [H, V, K], which it does not modify: (new state, BF16 output)."""
        # 1. Decay
        decay = np.exp(log_alpha_f32)[:, np.newaxis, :]
        state = state * decay

        # 2. Predicted
        predicted = np.einsum("hvk,hk->hv", state, k_f32)

        # 3. Error
        error = v_f32 - predicted

        # 4. State update
        update = np.einsum("hv,hk->hvk", error, k_f32)
        state = state + beta_f32[:, np.newaxis, np.newaxis] * update

        # 5. Output
        out_f32 = np.einsum("hvk,hk->hv", state, q_f32)
        return state, f32_to_bf16_rtne(out_f32)

    def rda_stage(
        self,
        layer: int,
        inputs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> _RdaTicket:
        """Stage one prefill RDA call of len(inputs) tokens (1, 2, 4 or 8), each (q, k, v, log_alpha, beta).

        DESIGN-prefill.md section 36 (R3-2): the head bundle of token t goes to [:, t, :] of the call's [40, T, 640]
        head BO.
        """
        self._drain_linear_pending()
        t_stage0 = time.perf_counter()
        tokens = check_rda_tokens(len(inputs))
        io = self._rda_io_next
        if self._rda_head_busy[io]:
            raise RuntimeError(f"RDA head buffer io={io} is currently busy")

        converted = [
            (
                bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32, copy=False),
                bf16_to_f32(k) if k.dtype == np.uint16 else k.astype(np.float32, copy=False),
                bf16_to_f32(v) if v.dtype == np.uint16 else v.astype(np.float32, copy=False),
                log_alpha.astype(np.float32, copy=False),
                bf16_to_f32(beta) if beta.dtype == np.uint16 else beta.astype(np.float32, copy=False),
            )
            for q, k, v, log_alpha, beta in inputs
        ]

        decode_xclbin_id = "gemv_unified" if self.gemv == "unified" else "rda_step"
        use_prefill_rda = self._use_prefill_rda()
        xclbin_id = PREFILL_GEMM_XCLBIN_ID if use_prefill_rda else decode_xclbin_id

        if self.mock or not HAVE_NPU:
            ticket = _RdaTicket(
                layer=layer,
                io=io,
                phase="staged",
                xclbin_id=xclbin_id,
                tokens=tokens,
                mock_inputs=converted,
            )
        else:
            try:
                from bench.rda_state.rda_state import pack_head_bundle
            except (ImportError, ModuleNotFoundError):
                from bench.rda_state.rda_state_1col import pack_head_bundle

            stream = self._prefill_rda_stream(tokens)
            bundles = np.stack(
                [pack_head_bundle(q, k, np.exp(la), v, b) for q, k, v, la, b in converted], axis=1
            )
            self._overwrite_buffer(stream.heads[io], bundles)
            ticket = _RdaTicket(
                layer=layer,
                io=io,
                phase="staged",
                xclbin_id=xclbin_id,
                tokens=tokens,
            )

        dt_prep = time.perf_counter() - t_stage0
        self._add_timing("rda_host", dt_prep, count=1)
        self._add_timing("host_prep", dt_prep, count=1)
        self._add_timing("rda_pipe_stage", dt_prep, count=1)

        self._rda_io_next = 1 - io
        self._rda_live.append(ticket)
        return ticket

    def rda_launch(self, ticket: _RdaTicket) -> None:
        self._drain_linear_pending()
        if ticket.phase != "staged":
            raise RuntimeError(f"Ticket phase must be 'staged', got '{ticket.phase}'")
        if self._rda_out_pending[ticket.io]:
            raise RuntimeError(f"RDA out buffer io={ticket.io} is pending read")
        if any(t.phase == "launched" for t in self._rda_live if t is not ticket):
            raise RuntimeError("Another ticket is already launched and not waited")

        io = ticket.io
        t_launch0 = time.perf_counter()

        if self.mock or not HAVE_NPU:
            # The state after the call is a candidate until rda_wait succeeds.
            states = self.rda_states.get(ticket.layer)
            if states is None:
                q0, k0, v0 = ticket.mock_inputs[0][:3]
                state = np.zeros((k0.shape[-2], v0.shape[-1], k0.shape[-1]), dtype=np.float32)
            else:
                state = states[self.rda_active_slot[ticket.layer]]
            outs = []
            for token_inputs in ticket.mock_inputs:
                state, out = self._rda_numpy_step(state, *token_inputs)
                outs.append(out)
            ticket.mock_state = state
            ticket.mock_out = np.stack(outs)
            ticket.t_launch = t_launch0
            self._rda_head_busy[io] = True
            self._rda_out_pending[io] = True
            self._add_timing("rda_pipe_launch", time.perf_counter() - t_launch0, count=1)
            ticket.phase = "launched"
            return

        if not self._use_prefill_rda():
            raise RuntimeError("Overlapped RDA launch requested outside prefill RDA context")

        H, V, K = 40, 128, 128
        if ticket.layer not in self.rda_states:
            s0 = iron.zeros((H, V, K), dtype=np.float32, device="npu")
            s1 = iron.zeros((H, V, K), dtype=np.float32, device="npu")
            self.rda_states[ticket.layer] = [s0, s1]
            self.rda_active_slot[ticket.layer] = 0

        slot = self.rda_active_slot[ticket.layer]
        s_in = self.rda_states[ticket.layer][slot]
        s_out = self.rda_states[ticket.layer][1 - slot]
        stream = self._prefill_rda_stream(ticket.tokens)
        args = [s_in, s_out, stream.heads[io], stream.outs[io]]

        ticket.run, ticket.keep = stream.runner.start(args)
        ticket.runner = stream.runner
        ticket.slot = slot
        self._rda_head_busy[io] = True
        self._rda_out_pending[io] = True
        ticket.t_launch = time.perf_counter()
        self._add_timing("rda_pipe_launch", time.perf_counter() - t_launch0, count=1)
        ticket.phase = "launched"

    def rda_wait(self, ticket: _RdaTicket) -> None:
        if ticket.phase != "launched":
            raise RuntimeError(f"Ticket phase must be 'launched', got '{ticket.phase}'")

        io = ticket.io
        t0 = time.perf_counter()

        if self.mock or not HAVE_NPU:
            if ticket.layer not in self.rda_states:
                self.rda_states[ticket.layer] = [ticket.mock_state, None]
                self.rda_active_slot[ticket.layer] = 0
            else:
                self.rda_states[ticket.layer][self.rda_active_slot[ticket.layer]] = ticket.mock_state
            ticket.mock_state = None
            now = time.perf_counter()
            dt_npu = now - ticket.t_launch
            self._record_xclbin_use(ticket.xclbin_id)
            self._rda_head_busy[io] = False
            self._add_timing("rda_npu", dt_npu, count=1)
            if ticket.xclbin_id == PREFILL_GEMM_XCLBIN_ID:
                self._add_timing("rda_npu_prefill", dt_npu, count=1)
            self._add_timing("rda", dt_npu, count=1)
            self._add_timing("rda_pipe_wait", now - t0, count=1)
            ticket.phase = "waited"
            return

        # Hardware execution:
        # If wait fails, do NOT switch slot and do NOT record xclbin use.
        ticket.runner.wait(ticket.run)

        # Success:
        now = time.perf_counter()
        self.rda_active_slot[ticket.layer] = 1 - ticket.slot
        self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
        self._rda_head_busy[io] = False
        dt_npu = now - ticket.t_launch
        self._add_timing("rda_npu", dt_npu, count=1)
        self._add_timing("rda_npu_prefill", dt_npu, count=1)
        self._add_timing("rda", dt_npu, count=1)
        self._add_timing("rda_pipe_wait", now - t0, count=1)
        ticket.run = None
        ticket.keep = None
        ticket.runner = None
        ticket.phase = "waited"

    def rda_read(self, ticket: _RdaTicket) -> np.ndarray:
        """BF16 outputs of a waited call, [T, 40, 128]: token t from [:, :, t, :] of the [40, 4, T, 32] output BO."""
        if ticket.phase != "waited":
            raise RuntimeError(f"Ticket phase must be 'waited', got '{ticket.phase}'")

        io = ticket.io
        if self.mock or not HAVE_NPU:
            self._rda_out_pending[io] = False
            ticket.phase = "done"
            if ticket in self._rda_live:
                self._rda_live.remove(ticket)
            return ticket.mock_out

        t_read0 = time.perf_counter()
        tokens = ticket.tokens
        out_buf = self._prefill_rda_stream(tokens).outs[io]
        out_f32 = np.asarray(out_buf.numpy()).reshape(40, 4, tokens, 32)
        res = f32_to_bf16_rtne(out_f32.transpose(2, 0, 1, 3).reshape(tokens, 40, 128))
        self._rda_out_pending[io] = False
        dt_post = time.perf_counter() - t_read0
        self._add_timing("rda_host", dt_post, count=0)
        self._add_timing("host_prep", dt_post, count=0)
        self._add_timing("rda_pipe_read", dt_post, count=1)
        ticket.phase = "done"
        if ticket in self._rda_live:
            self._rda_live.remove(ticket)
        return res

    def rda_abort_all(self) -> None:
        """Abort all live RDA overlapped tickets.

        Waits for any launched-but-unwaited hardware ticket to finish (on the runner it was started on) so no
        in-flight NPU execution is left running. Exceptions from this wait are swallowed. Active state slots are NOT
        switched, mock candidate states are dropped, and no usage counters are incremented. After this call, internal
        RDA state is undefined and callers must reset() model state before continuing.
        """
        for ticket in list(self._rda_live):
            if ticket.phase == "launched":
                if ticket.runner is not None and ticket.run is not None:
                    try:
                        ticket.runner.wait(ticket.run)
                    except Exception:
                        pass
                ticket.run = None
                ticket.keep = None
                ticket.runner = None
                ticket.phase = "waited"
            ticket.mock_state = None
        self._rda_live.clear()
        self._rda_head_busy = [False, False]
        self._rda_out_pending = [False, False]
        self._rda_io_next = 0

    def set_rda_state(self, layer: int, state: np.ndarray) -> None:
        """Set RDA recurrent state [H, V, K] (FP32)."""
        if self._rda_live:
            raise RuntimeError("Cannot set RDA state while overlapped RDA tickets are active")
        st = state.squeeze(0) if state.ndim == 4 else state
        st_f32 = np.ascontiguousarray(st, dtype=np.float32)
        if self.mock or not HAVE_NPU:
            self.rda_states[layer] = [st_f32.copy(), None]
            self.rda_active_slot[layer] = 0
            return

        if layer not in self.rda_states:
            s0 = iron.tensor(st_f32, device="npu")
            s1 = iron.zeros_like(s0)
            self.rda_states[layer] = [s0, s1]
            self.rda_active_slot[layer] = 0
        else:
            slot = self.rda_active_slot[layer]
            self.rda_states[layer][slot][:] = st_f32

    def get_rda_state(self, layer: int) -> np.ndarray | None:
        """Get RDA recurrent state [1, H, V, K] (FP32)."""
        if self._rda_live:
            raise RuntimeError("Cannot read RDA state while overlapped RDA tickets are active")
        states = self.rda_states.get(layer)
        if states is None:
            return None
        slot = self.rda_active_slot.get(layer, 0)
        curr = states[slot]
        if self.mock or not HAVE_NPU:
            return curr.copy()[np.newaxis, ...]
        return np.array(curr.numpy(), copy=True)[np.newaxis, ...]

    # -------------------------------------------------------------------------
    # MoE Execution: B1-6 -> Shared GU -> Unweighted RTNE(SiTU-GLU) -> B2-9 -> Weighted Ascending FP32 Sum
    # -------------------------------------------------------------------------
    @staticmethod
    def _fit_u16_input_width(inp_u16: np.ndarray, target_width: int) -> np.ndarray:
        """Trim or zero-pad an input vector to match the weight projection width."""
        inp = np.asarray(inp_u16, dtype=np.uint16).reshape(-1)
        if inp.size > target_width:
            return inp[:target_width]
        if inp.size < target_width:
            padded = np.zeros(target_width, dtype=np.uint16)
            padded[:inp.size] = inp
            return padded
        return inp

    def _moe_named_projection_numpy(
        self, name: str, inp_u16: np.ndarray, dw: DWeights | None
    ) -> np.ndarray:
        """Project one BF16 row through host weights or decoded DWeights."""
        if self.host_weights is not None and name in self.host_weights:
            w_f32 = self.host_weights.get(name, to_float=True)
        elif dw is not None and name in dw:
            w_f32 = dw.get(name, to_float=True)
        else:
            raise KeyError(f"Weight parameter '{name}' not found in host weights or DWeights")
        inp = self._fit_u16_input_width(inp_u16, w_f32.shape[1])
        return f32_to_bf16_rtne(np.dot(bf16_to_f32(inp), w_f32.T))

    def _moe_b1_row_numpy(
        self, layer: int, expert_id: int, x_u16: np.ndarray, dw: DWeights | None
    ) -> np.ndarray:
        """Compute one routed expert gate+up row using the decode mock expression."""
        bundle = f"L{layer:02d}/moe_b1"
        gate_name = f"model.layers.{layer}.moe.experts.{expert_id}.gate_proj.weight"
        up_name = f"model.layers.{layer}.moe.experts.{expert_id}.up_proj.weight"
        if bundle in self.mock_weight_arrays and (
            self.host_weights is None or gate_name not in self.host_weights
        ):
            w_b1 = self.mock_weight_arrays[bundle][expert_id]
            w_f32 = bf16_to_f32(w_b1) if w_b1.dtype == np.uint16 else w_b1.astype(np.float32)
            return f32_to_bf16_rtne(np.dot(bf16_to_f32(x_u16), w_f32.T))
        gate = self._moe_named_projection_numpy(gate_name, x_u16, dw)
        up = self._moe_named_projection_numpy(up_name, x_u16, dw)
        return np.concatenate([gate, up])

    def _moe_b2_row_numpy(
        self, layer: int, expert_id: int, x_u16: np.ndarray, dw: DWeights | None
    ) -> np.ndarray:
        """Compute one routed/shared down row from a consistently padded input."""
        bundle = f"L{layer:02d}/moe_b2"
        if expert_id == 128:
            if bundle in self.mock_weight_arrays:
                w_b2 = self.mock_weight_arrays[bundle][expert_id]
                w_f32 = bf16_to_f32(w_b2) if w_b2.dtype == np.uint16 else w_b2.astype(np.float32)
            else:
                if dw is None:
                    raise ValueError(
                        f"Cannot compute shared down at layer {layer}: missing DWeights"
                    )
                name = f"model.layers.{layer}.moe.shared_experts.down_proj.weight"
                if name not in dw:
                    raise KeyError(f"Shared down weight '{name}' not found in DWeights")
                w_f32 = dw.get(name, to_float=True)
            inp = self._fit_u16_input_width(x_u16, w_f32.shape[1])
            return f32_to_bf16_rtne(np.dot(bf16_to_f32(inp), w_f32.T))

        name = f"model.layers.{layer}.moe.experts.{expert_id}.down_proj.weight"
        use_bundle = bundle in self.mock_weight_arrays and (
            self.host_weights is None or name not in self.host_weights
        )
        if use_bundle:
            w_b2 = self.mock_weight_arrays[bundle][expert_id]
            w_f32 = bf16_to_f32(w_b2) if w_b2.dtype == np.uint16 else w_b2.astype(np.float32)
            inp = self._fit_u16_input_width(x_u16, w_f32.shape[1])
            return f32_to_bf16_rtne(np.dot(bf16_to_f32(inp), w_f32.T))
        return self._moe_named_projection_numpy(name, x_u16, dw)

    def _moe_shared_gu_reference_numpy(
        self, layer: int, x_u16: np.ndarray, dw: DWeights | None
    ) -> np.ndarray:
        """Compute the shared gate+up reference row used to validate precompute input."""
        bundle = f"L{layer:02d}/shared_gu"
        if bundle in self.mock_weight_arrays:
            w_sh = self.mock_weight_arrays[bundle]
            w_f32 = bf16_to_f32(w_sh) if w_sh.dtype == np.uint16 else w_sh.astype(np.float32)
            return f32_to_bf16_rtne(np.dot(bf16_to_f32(x_u16), w_f32.T))
        if dw is not None:
            w_f32 = dw._decode_2d_bundle(bundle)
            return f32_to_bf16_rtne(np.dot(bf16_to_f32(x_u16), w_f32.T))
        raise ValueError(f"Cannot verify shared_gu at layer {layer}: missing shared weights")

    def _run_grouped_moe_hardware_call(
        self,
        shape: Any,
        layer: int,
        expert_slots: tuple[int, ...],
        values: np.ndarray,
        grouped: dict[str, tuple[Callable[..., None], Any, Any]],
        rows: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run one grouped hardware call and return its output and run-only time."""
        if shape is GROUPED_B1 and rows is None:
            raise ValueError("B1 grouped hardware call requires rows")
        run, x_tensor, y_tensor = grouped[shape.name]
        w_tensor = self._get_prefill_weight_tensor(f"L{layer:02d}/moe_{shape.name}")
        self._overwrite_buffer(
            x_tensor,
            _grouped_x_tiled(values, shape).reshape(-1).view(bfloat16),
        )
        scalars = {f"e{i}": int(expert) for i, expert in enumerate(expert_slots)}
        if rows is not None:
            for i, count in enumerate(rows):
                scalars[f"r{i}"] = int(count)
        started = time.perf_counter()
        run(y_tensor, w_tensor, x_tensor, **scalars)
        run_s = time.perf_counter() - started
        self._record_xclbin_use(f"moe_grouped_{shape.name}")
        output = np.array(y_tensor.numpy().view(np.uint16), copy=True).reshape(
            SLOTS, TMAX, shape.m
        )
        return output, run_s

    def _run_grouped_moe_numpy_call(
        self,
        shape: Any,
        layer: int,
        call: Any,
        values: np.ndarray,
        dw: DWeights | None,
        *,
        record_xclbin: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Emulate one grouped call one assigned row at a time."""
        output = np.zeros((SLOTS, TMAX, shape.m), dtype=np.uint16)
        started = time.perf_counter()
        for slot, job in enumerate(call.jobs):
            if job is None:
                continue
            for row, _assignment in enumerate(job.assignments):
                if shape is GROUPED_B1:
                    output[slot, row] = self._moe_b1_row_numpy(
                        layer, job.expert, values[slot, row], dw
                    )
                else:
                    output[slot, row] = self._moe_b2_row_numpy(
                        layer, job.expert, values[slot, row], dw
                    )
        run_s = time.perf_counter() - started
        if record_xclbin:
            self._record_xclbin_use(f"moe_grouped_{shape.name}")
        return output, run_s

    def _get_grouped_overlap_runner(
        self, shape: Any, dw: DWeights | None
    ) -> _GroupedDirectRunner | _GroupedNumpyRunner | _GroupedCallableRunner:
        """Lazily create the per-shape runner on the first overlap call."""
        cache = self._compiled_ops.setdefault("grouped_moe_overlap", {})
        runner = cache.get(shape.name)
        if runner is not None:
            return runner
        if self.mock or not HAVE_NPU:
            runner = _GroupedNumpyRunner(self, shape, dw)
        else:
            grouped = self._get_or_compile_grouped_moe()
            kernels = self._compiled_ops.get("grouped_moe_kernels")
            if not isinstance(kernels, dict) or shape.name not in kernels:
                runner = _GroupedCallableRunner(self, shape, grouped[shape.name])
            else:
                _run, x0, y0 = grouped[shape.name]
                runner = _make_grouped_direct_runner(
                    self, shape, kernels[shape.name], x0, y0
                )
        cache[shape.name] = runner
        return runner

    def _moe_grouped_overlapped(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        shared_gu: np.ndarray,
        gate_beta: float,
        value_beta: float,
        dw: DWeights | None,
        wall_started: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Overlap host work with grouped calls.

        The B1/B2 envelope timings run from just after launch through successful
        wait completion.  They include any host-work tail and are not pure
        device execution times; ``moe_pipe_wait`` records only time blocked in
        ``wait``.
        """
        from kernels.moe.pack import (
            B2_K_PAD,
            B2_M_PER_EXPERT,
            NUM_ACTIVE_ROUTED,
        )

        b1_plan = build_plan(top_ids)
        b2_plan = build_plan(top_ids, include_shared=True)
        token_count = x.shape[0]
        routed_in = np.zeros(
            (token_count, NUM_ACTIVE_ROUTED, B2_K_PAD), dtype=np.uint16
        )
        routed_acc = np.zeros((token_count, B2_M_PER_EXPERT), dtype=np.float32)
        shared_down = np.zeros((token_count, B2_M_PER_EXPERT), dtype=np.uint16)
        holder: dict[str, np.ndarray] = {}
        envelope_by_shape = {GROUPED_B1.name: 0.0, GROUPED_B2.name: 0.0}
        wait_s = 0.0

        def b1_first() -> None:
            holder["shared_in"] = shared_situ(
                shared_gu, gate_beta=gate_beta, value_beta=value_beta
            )

        def run_phase(
            shape: Any,
            plan: Any,
            gather: Callable[[int], np.ndarray],
            process: Callable[[Any, np.ndarray], None],
            first: Callable[[], None] | None = None,
        ) -> None:
            nonlocal wait_s
            runner = self._get_grouped_overlap_runner(shape, dw)
            count = len(plan.calls)
            pending = None
            try:
                runner.prepare(0, plan.calls[0], gather(0))
                pending = runner.start(0, layer)
                started = time.perf_counter()
                if first is not None:
                    first()
                for index in range(count):
                    if index > 0:
                        process(
                            plan.calls[index - 1], runner.read((index - 1) % 2)
                        )
                    if index + 1 < count:
                        runner.prepare(
                            (index + 1) % 2,
                            plan.calls[index + 1],
                            gather(index + 1),
                        )
                    wait_started = time.perf_counter()
                    current = pending
                    pending = None
                    runner.wait(current)
                    wait_s += time.perf_counter() - wait_started
                    envelope_by_shape[shape.name] += time.perf_counter() - started
                    self._record_xclbin_use(f"moe_grouped_{shape.name}")
                    if index + 1 < count:
                        pending = runner.start((index + 1) % 2, layer)
                        started = time.perf_counter()
                process(plan.calls[-1], runner.read((count - 1) % 2))
            finally:
                if pending is not None:
                    try:
                        runner.wait(pending)
                    except Exception:
                        pass
                    pending = None

        run_phase(
            GROUPED_B1,
            b1_plan,
            lambda index: gather_call(b1_plan, index, x),
            lambda call, out: situ_call(
                call,
                out,
                routed_in,
                gate_beta=gate_beta,
                value_beta=value_beta,
            ),
            first=b1_first,
        )
        run_phase(
            GROUPED_B2,
            b2_plan,
            lambda index: gather_call(
                b2_plan, index, routed_in, holder["shared_in"]
            ),
            lambda call, out: accumulate_call(
                call, out, routed_acc, shared_down, top_weights
            ),
        )
        routed = f32_to_bf16_rtne(routed_acc)
        b1_s = envelope_by_shape[GROUPED_B1.name]
        b2_s = envelope_by_shape[GROUPED_B2.name]
        host_s = max(0.0, time.perf_counter() - wall_started - b1_s - b2_s)
        self._add_timing("moe_grouped_b1_npu", b1_s, count=len(b1_plan.calls))
        self._add_timing("moe_grouped_b2_npu", b2_s, count=len(b2_plan.calls))
        self._add_timing(
            "moe_pipe_wait", wait_s, count=len(b1_plan.calls) + len(b2_plan.calls)
        )
        self._add_timing("moe_grouped_host", host_s, count=1)
        self._add_timing("moe", b1_s + b2_s, count=token_count)
        self._add_timing("moe_host", host_s, count=1)
        self._add_timing("host_prep", host_s, count=1)
        if routed.shape != (token_count, B2_M_PER_EXPERT):
            raise RuntimeError("grouped MoE produced an invalid routed output shape")
        return routed, shared_down

    def moe_experts_grouped(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        shared_gu: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return grouped routed sums and shared down rows for ``T`` tokens."""
        self._drain_linear_pending()
        from kernels.moe.pack import (
            B1_K,
            B1_M_PER_EXPERT,
            B2_M_PER_EXPERT,
            NUM_ACTIVE_ROUTED,
        )

        if not isinstance(layer, (int, np.integer)):
            raise ValueError("layer must be an integer")
        layer = int(layer)
        if not isinstance(x, np.ndarray) or x.dtype != np.uint16 or x.ndim != 2:
            raise ValueError("x must be a [T, 2560] uint16 ndarray")
        if x.shape[0] < 1 or x.shape[1] != B1_K:
            raise ValueError("x must be a [T, 2560] uint16 ndarray with T >= 1")
        token_count = x.shape[0]
        if (
            not isinstance(top_ids, np.ndarray)
            or not np.issubdtype(top_ids.dtype, np.integer)
            or top_ids.shape != (token_count, NUM_ACTIVE_ROUTED)
        ):
            raise ValueError("top_ids must be an integer ndarray with shape [T, 6]")
        if np.any(top_ids < 0) or np.any(top_ids >= 128):
            raise ValueError("top_ids values must be in 0..127")
        sorted_ids = np.sort(top_ids, axis=1)
        if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("each token must route to 6 distinct experts")
        if (
            not isinstance(top_weights, np.ndarray)
            or top_weights.dtype != np.float32
            or top_weights.shape != (token_count, NUM_ACTIVE_ROUTED)
        ):
            raise ValueError("top_weights must be a float32 ndarray with shape [T, 6]")
        if (
            not isinstance(shared_gu, np.ndarray)
            or shared_gu.dtype != np.uint16
            or shared_gu.shape != (token_count, B1_M_PER_EXPERT)
        ):
            raise ValueError("shared_gu must be a [T, 2176] uint16 ndarray")
        for suffix in ("moe_b1", "moe_b2"):
            if f"L{layer:02d}/{suffix}" not in self.bundle_map:
                raise ValueError(f"layer {layer} is not an exported MoE layer")

        self._cached_shared.pop(layer, None)
        wall_started = time.perf_counter()
        dw = self._get_mock_d_weights() if (self.mock or not HAVE_NPU) else None

        if dw is not None or self.mock or not HAVE_NPU:
            for token in range(token_count):
                reference = self._moe_shared_gu_reference_numpy(layer, x[token], dw)
                if not np.array_equal(shared_gu[token], reference):
                    raise ValueError(
                        f"shared_gu mismatch in grouped moe_experts at layer {layer}, "
                        f"token {token}"
                    )

        if self.prefill_moe_overlap:
            return self._moe_grouped_overlapped(
                layer,
                x,
                top_ids,
                top_weights,
                shared_gu,
                gate_beta,
                value_beta,
                dw,
                wall_started,
            )

        b1_plan = build_plan(top_ids)
        b1_inputs = gather_inputs(b1_plan, x)
        grouped = None if (self.mock or not HAVE_NPU) else self._get_or_compile_grouped_moe()

        b1_outputs: list[np.ndarray] = []
        b1_run_s = 0.0
        for call, values in zip(b1_plan.calls, b1_inputs):
            if grouped is None:
                output, run_s = self._run_grouped_moe_numpy_call(
                    GROUPED_B1, layer, call, values, dw
                )
            else:
                output, run_s = self._run_grouped_moe_hardware_call(
                    GROUPED_B1,
                    layer,
                    call.expert_slots,
                    values,
                    grouped,
                    rows=call.rows,
                )
            b1_outputs.append(output)
            b1_run_s += run_s
        b1 = scatter_b1(b1_plan, b1_outputs)
        routed_in, shared_in = make_b2_inputs(
            b1, shared_gu, gate_beta=gate_beta, value_beta=value_beta
        )
        b2_plan = build_plan(top_ids, include_shared=True)
        b2_inputs = gather_inputs(b2_plan, routed_in, shared_in)
        b2_outputs: list[np.ndarray] = []
        b2_run_s = 0.0
        for call, values in zip(b2_plan.calls, b2_inputs):
            if grouped is None:
                output, run_s = self._run_grouped_moe_numpy_call(
                    GROUPED_B2, layer, call, values, dw
                )
            else:
                output, run_s = self._run_grouped_moe_hardware_call(
                    GROUPED_B2, layer, call.expert_slots, values, grouped
                )
            b2_outputs.append(output)
            b2_run_s += run_s
        routed, shared_down = scatter_b2(
            b2_plan, b2_outputs, top_ids, top_weights
        )

        host_s = max(0.0, time.perf_counter() - wall_started - b1_run_s - b2_run_s)
        self._add_timing("moe_grouped_b1_npu", b1_run_s, count=len(b1_plan.calls))
        self._add_timing("moe_grouped_b2_npu", b2_run_s, count=len(b2_plan.calls))
        self._add_timing("moe_grouped_host", host_s, count=1)
        self._add_timing("moe", b1_run_s + b2_run_s, count=token_count)
        self._add_timing("moe_host", host_s, count=1)
        self._add_timing("host_prep", host_s, count=1)
        if routed.shape != (token_count, B2_M_PER_EXPERT):
            raise RuntimeError("grouped MoE produced an invalid routed output shape")
        return routed, shared_down

    def _drain_moe_shared_pre(self) -> None:
        """Wait out a shared gate/up run started by moe_prestart_shared that moe_experts did not take."""
        pre, self._moe_shared_pre = self._moe_shared_pre, None
        if pre is not None:
            runner = self._compiled_ops.get("decode_direct")
            if runner is not None:
                runner.drain([pre[2]])

    def moe_prestart_shared(self, layer: int, x: np.ndarray) -> None:
        """Start one decode token's shared-expert gate/up GEMV before the host router (DESIGN-prefill.md section 46).

        moe_experts takes the started run when the layer, the GEMV entry and the input bytes match, and otherwise
        waits it out and starts its own, so the result never depends on this call. Only on the decode direct path
        with ``decode_moe_overlap`` on; elsewhere nothing happens.
        """
        self._drain_linear_pending()
        self._drain_moe_shared_pre()
        if not self.decode_moe_overlap or self.mock or not HAVE_NPU or self.gemv != "unified":
            return
        decode_runner = self._decode_runner()
        if decode_runner is None:
            return
        from kernels.moe.pack import B1_K, B1_M_PER_EXPERT

        _, run_sh, _ = self._get_or_compile_moe_ops()
        if not self._decode_moe_eligible(True, run_sh)[0]:
            return
        x_u16 = f32_to_bf16_rtne(x) if x.dtype == np.float32 else np.asarray(x, dtype=np.uint16)
        x_sh = np.array(x_u16.reshape(-1)[:B1_K], dtype=np.uint16, copy=True)
        w_sh = self._get_weight_tensor(f"L{layer:02d}/shared_gu")
        entry = decode_runner.gemv_entry(
            f"L{layer:02d}/shared_gu", run_sh.op, w_sh, B1_K, B1_M_PER_EXPERT,
        )
        try:
            run = decode_runner.gemv_start(entry, x_sh)
            self._moe_shared_pre = (layer, entry, run, x_sh)
        except BaseException:
            decode_runner.drain([entry.run])
            self._moe_shared_pre = None
            raise

    def moe_abort_prestart_shared(self) -> None:
        """Wait out a prestarted shared gate/up and forget staged B2 words when a token does not finish its MoE step."""
        self._drain_moe_shared_pre()
        runner = self._compiled_ops.get("decode_direct")
        if runner is not None:
            runner.moe_clear_staged()

    def moe_experts(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
        shared_gu: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the weighted sum of 6 routed experts via patched NPU pipeline.

        The SiTU betas come from the caller (the model passes the config's
        ``activation_situ_beta``/``activation_situ_linear_beta``); the defaults
        are the released-checkpoint 4.0/25.0 pair. The SiTU-GLU activation is
        computed on the host here, so non-default betas are supported as-is.
        """
        self._drain_linear_pending()
        if shared_gu is not None:
            if (
                not isinstance(shared_gu, np.ndarray)
                or shared_gu.dtype != np.uint16
                or shared_gu.shape != (2176,)
            ):
                raise ValueError(
                    f"shared_gu must be a [2176] uint16 ndarray, got {type(shared_gu)} with "
                    f"dtype={getattr(shared_gu, 'dtype', None)}, shape={getattr(shared_gu, 'shape', None)}"
                )

        t_h0 = time.perf_counter()
        x_u16 = f32_to_bf16_rtne(x) if x.dtype == np.float32 else np.asarray(x, dtype=np.uint16)

        if self.mock or not HAVE_NPU:
            dt_h = time.perf_counter() - t_h0
            self._add_timing("moe_host", dt_h, count=0)
            self._add_timing("host_prep", dt_h, count=0)
            return self._moe_experts_numpy(
                layer,
                x,
                x_u16,
                top_ids,
                top_weights,
                gate_beta=gate_beta,
                value_beta=value_beta,
                shared_gu=shared_gu,
            )

        # ---------------------------------------------------------------------
        # Full NPU MoE Pipeline
        # ---------------------------------------------------------------------
        from kernels.moe.pack import (
            B1_K,
            B1_M_PER_EXPERT,
            B2_K_PAD,
            B2_M_PER_EXPERT,
            MOE_INTERMEDIATE_SIZE,
            NUM_ACTIVE_TOTAL,
        )

        run_gu, run_sh, run_dn = self._get_or_compile_moe_ops()
        w_gu = self._get_weight_tensor(f"L{layer:02d}/moe_b1")
        w_sh = self._get_weight_tensor(f"L{layer:02d}/shared_gu")
        w_dn = self._get_weight_tensor(f"L{layer:02d}/moe_b2")

        decode_runner = self._decode_runner() if self.gemv == "unified" else None
        direct = False
        direct_kernels: tuple[Any, Any] | None = None
        if decode_runner is not None:
            eligible, reason = self._decode_moe_eligible(shared_gu is None, run_sh)
            if eligible:
                direct = True
                direct_kernels = self._compiled_ops["unified_moe_kernels"]
            else:
                self._decode_fallback("moe", reason)

        if not direct:
            self._drain_moe_shared_pre()
            y_gu = self._prepared_buffer("moe_y_gu")
            y_sh = self._prepared_buffer("moe_y_sh")
            y_dn = self._prepared_buffer("moe_y_dn")
            x_gu = self._prepared_buffer("moe_x_gu")
            x_dn = self._prepared_buffer("moe_x_dn")

        top_ids_int = [int(i) for i in top_ids]
        scalars = {f"e{j}": top_ids_int[j] for j in range(6)}

        if self.gemv == "unified":
            if not direct:
                x_pad_gu = pad_x(x_u16.reshape(-1)[:B1_K], "moe_b1_6")
                self._overwrite_buffer(
                    x_gu, np.ascontiguousarray(x_pad_gu, dtype=np.uint16).view(bfloat16)
                )
            b1_xclbin = "gemv_unified"
            sh_xclbin = "gemv_unified"
            dn_xclbin = "gemv_unified"
        else:
            self._overwrite_buffer(
                x_gu,
                np.ascontiguousarray(x_u16.reshape(-1)[:B1_K], dtype=np.uint16).view(bfloat16),
            )
            b1_xclbin = "moe_gate_up"
            sh_xclbin = legacy_gemv_xclbin_name("bfp16_gemv", B1_K, B1_M_PER_EXPERT)
            dn_xclbin = "moe_down"

        dt_prep1 = time.perf_counter() - t_h0

        # 1. B1-6 (Patched Template for routed experts 0..5)
        t_b1 = time.perf_counter()
        if direct:
            gu_kernel, dn_kernel = direct_kernels
            sh_entry = None
            sh_run = None
            dn_words = None
            # A shared gate/up run started before the router (moe_prestart_shared, DESIGN-prefill.md section 46) stays
            # in self._moe_shared_pre until it is taken or waited out, so an abort can still drain it. While it is in
            # flight no handle check below may discard or reload: when one could, it is waited out first.
            pre = self._moe_shared_pre
            if pre is not None and not decode_runner.overlap_ready(pre[1]):
                decode_runner.drain([pre[2]])
                self._moe_shared_pre = pre = None
            # 3-round order: obtain the shared GEMV entry before B1 moe_start
            # so B1 never waits on handle checks / runtime.load / entry warmup.
            if shared_gu is None:
                sh_entry = decode_runner.gemv_entry(
                    f"L{layer:02d}/shared_gu", run_sh.op, w_sh,
                    B1_K, B1_M_PER_EXPERT,
                )
            # The prestarted run is taken only for this layer, this entry and these input bytes; anything else is
            # waited out first.
            if pre is not None and not (
                shared_gu is None
                and pre[0] == layer
                and pre[1] is sh_entry
                and np.array_equal(pre[3], x_u16.reshape(-1)[:B1_K])
            ):
                decode_runner.drain([pre[2]])
                self._moe_shared_pre = pre = None
            try:
                b1_run = decode_runner.moe_start(
                    "b1", layer, gu_kernel, w_gu,
                    gu_kernel.instructions(**scalars),
                    x_u16.reshape(-1)[:B1_K], MAX_K, 6 * B1_M_PER_EXPERT,
                )
            except BaseException:
                if pre is not None:
                    decode_runner.drain([pre[2]])
                    self._moe_shared_pre = None
                raise
            if shared_gu is None:
                if pre is not None:
                    sh_run = pre[2]
                    self._moe_shared_pre = None
                else:
                    try:
                        sh_run = decode_runner.gemv_start(
                            sh_entry, x_u16.reshape(-1)[:B1_K]
                        )
                    except BaseException:
                        decode_runner.drain([b1_run])
                        raise
            if self.decode_moe_overlap:
                # The B2 words depend only on the routing: make, write and sync them while B1 runs.
                try:
                    dn_words = dn_kernel.instructions(**scalars)
                    decode_runner.moe_stage(
                        "b2", layer, dn_kernel, w_dn, dn_words,
                        7 * MAX_K, NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,
                    )
                except BaseException:
                    decode_runner.drain([r for r in (b1_run, sh_run) if r is not None])
                    raise
            try:
                gu_arr = decode_runner.moe_finish("b1", b1_run).reshape(
                    6, B1_M_PER_EXPERT
                )
            except BaseException:
                if sh_run is not None:
                    decode_runner.drain([sh_run])
                raise
            self._record_xclbin_use(b1_xclbin)
            self._add_timing("decode_direct_moe", 0.0, count=1)
            dt_b1 = time.perf_counter() - t_b1

            if shared_gu is not None:
                sh_flat_u16 = np.asarray(shared_gu, dtype=np.uint16).reshape(B1_M_PER_EXPERT)
                dt_sh = 0.0
            else:
                t_sh = time.perf_counter()
                sh_flat_u16 = decode_runner.gemv_finish(
                    sh_entry, sh_run
                ).reshape(B1_M_PER_EXPERT)
                self._record_xclbin_use(sh_xclbin)
                self._add_timing("decode_direct_moe", 0.0, count=1)
                dt_sh = time.perf_counter() - t_sh
        else:
            run_gu(y_gu, w_gu, x_gu, **scalars)
            self._record_xclbin_use(b1_xclbin)
            gu_arr = np.array(y_gu.numpy().view(np.uint16), copy=True).reshape(6, B1_M_PER_EXPERT)
            dt_b1 = time.perf_counter() - t_b1

            # 2. Shared Expert gate+up (9-bit bfp16 GEMV)
            if shared_gu is not None:
                sh_flat_u16 = np.asarray(shared_gu, dtype=np.uint16).reshape(B1_M_PER_EXPERT)
                dt_sh = 0.0
            else:
                t_sh = time.perf_counter()
                run_sh(y_sh, w_sh, x_gu)
                self._record_xclbin_use(sh_xclbin)
                sh_flat_u16 = np.array(y_sh.numpy().view(np.uint16), copy=True).reshape(B1_M_PER_EXPERT)
                dt_sh = time.perf_counter() - t_sh

        # 3. Cache shared_gu in _bundle_cache so subsequent linear(shared_experts.gate/up_proj, x) reuses it
        t_h1 = time.perf_counter()
        sh_bundle = f"L{layer:02d}/shared_gu"
        self._bundle_cache[sh_bundle] = (self._step_id, id(x), sh_flat_u16)
        self._cached_inputs[sh_bundle] = x

        sh_arr = sh_flat_u16.reshape(1, B1_M_PER_EXPERT)
        gate_up_all = np.concatenate([gu_arr, sh_arr], axis=0)  # [7, 2176]

        # Host SiTU-GLU -> unweighted RTNE(act) for all 7 slots
        x_down_u16 = np.zeros((NUM_ACTIVE_TOTAL, B2_K_PAD), dtype=np.uint16)
        sh_gate_u16 = sh_flat_u16[:MOE_INTERMEDIATE_SIZE].copy()
        sh_up_u16 = sh_flat_u16[MOE_INTERMEDIATE_SIZE:].copy()
        act_f32 = situ_glu(
            gate_up_all[:, :MOE_INTERMEDIATE_SIZE],
            gate_up_all[:, MOE_INTERMEDIATE_SIZE:],
            gate_beta=gate_beta,
            value_beta=value_beta,
        )
        act_u16 = f32_to_bf16_rtne(act_f32)
        x_down_u16[:, :MOE_INTERMEDIATE_SIZE] = act_u16
        sh_act_u16 = act_u16[6].copy()

        if self.gemv == "unified":
            if not direct:
                x_pad_dn = pad_x(x_down_u16, "moe_b2_9")
                self._overwrite_buffer(
                    x_dn, np.ascontiguousarray(x_pad_dn, dtype=np.uint16).view(bfloat16)
                )
        else:
            self._overwrite_buffer(x_dn, x_down_u16.reshape(-1).view(bfloat16))
        dt_prep2 = time.perf_counter() - t_h1

        # 4. B2-9 (7 slots: 0..5 routed, slot 6 shared)
        t_dn = time.perf_counter()
        if direct:
            _, dn_kernel = direct_kernels
            dn_run = decode_runner.moe_start(
                "b2", layer, dn_kernel, w_dn,
                dn_words if dn_words is not None else dn_kernel.instructions(**scalars), x_down_u16,
                7 * MAX_K, NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,
            )
            dn_arr = decode_runner.moe_finish("b2", dn_run).reshape(
                NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT
            )
            self._record_xclbin_use(dn_xclbin)
            self._add_timing("decode_direct_moe", 0.0, count=1)
        else:
            run_dn(y_dn, w_dn, x_dn, **scalars)
            self._record_xclbin_use(dn_xclbin)
            dn_arr = np.array(y_dn.numpy().view(np.uint16), copy=True).reshape(NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT)
        dt_dn = time.perf_counter() - t_dn

        # 5. Host weighted sum in ascending expert ID order
        t_h2 = time.perf_counter()
        routed_order = sorted(range(6), key=lambda j: top_ids_int[j])
        routed_sum_f32 = np.zeros(B2_M_PER_EXPERT, dtype=np.float32)
        dn_f32 = bf16_to_f32(dn_arr)
        for j in routed_order:
            routed_sum_f32 += dn_f32[j] * float(top_weights[j])

        # Cache shared expert outputs with step and slot-6 input bits for validation in _linear_shared_down
        sh_down_u16 = dn_arr[6].copy()
        self._cached_shared[layer] = {
            "step": self._step_id,
            "sh_gate": sh_gate_u16,
            "sh_up": sh_up_u16,
            "sh_act_u16": sh_act_u16,
            "sh_down": sh_down_u16,
        }

        out_u16 = f32_to_bf16_rtne(routed_sum_f32)
        dt_prep3 = time.perf_counter() - t_h2

        dt_host_total = dt_prep1 + dt_prep2 + dt_prep3
        self._add_timing("moe_b1_npu", dt_b1, count=1)
        if shared_gu is None:
            self._add_timing("moe_sh_npu", dt_sh, count=1)
        self._add_timing("moe_dn_npu", dt_dn, count=1)
        self._add_timing("moe", dt_b1 + dt_sh + dt_dn, count=1)
        self._add_timing("moe_host", dt_host_total, count=1)
        self._add_timing("host_prep", dt_host_total, count=1)
        return out_u16

    def _moe_experts_numpy(
        self,
        layer: int,
        x_orig: np.ndarray,
        x_u16: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
        shared_gu: np.ndarray | None = None,
    ) -> np.ndarray:
        """NumPy reference MoE execution adhering strictly to ascending expert order accumulation.

        ``gate_beta``/``value_beta`` are the SiTU activation betas forwarded
        from ``moe_experts``; the defaults are the released-checkpoint pair.
        """
        if shared_gu is not None:
            if (
                not isinstance(shared_gu, np.ndarray)
                or shared_gu.dtype != np.uint16
                or shared_gu.shape != (2176,)
            ):
                raise ValueError(
                    f"shared_gu must be a [2176] uint16 ndarray, got {type(shared_gu)} with "
                    f"dtype={getattr(shared_gu, 'dtype', None)}, shape={getattr(shared_gu, 'shape', None)}"
                )

        t_h0 = time.perf_counter()
        dim = x_u16.shape[-1]
        weighted_out = np.zeros(dim, dtype=np.float32)

        selected = sorted(
            ((int(expert_id), float(weight)) for expert_id, weight in zip(top_ids, top_weights)),
            key=lambda item: item[0],
        )

        b2_name = f"L{layer:02d}/moe_b2"
        sh_gu_name = f"L{layer:02d}/shared_gu"
        dw = self._get_mock_d_weights()

        dt_host = time.perf_counter() - t_h0
        dt_b1 = 0.0
        dt_sh = 0.0
        dt_dn = 0.0

        if self.gemv == "unified":
            b1_xclbin = "gemv_unified"
            sh_xclbin = "gemv_unified"
            dn_xclbin = "gemv_unified"
        else:
            b1_xclbin = "moe_gate_up"
            sh_xclbin = legacy_gemv_xclbin_name("bfp16_gemv", 2560, 2176)
            dn_xclbin = "moe_down"

        # 1. B1-6 (gate + up for routed experts)
        routed_acts: list[tuple[int, float, np.ndarray]] = []

        for expert_id, weight in selected:
            t_b1_0 = time.perf_counter()
            gu_u16 = self._moe_b1_row_numpy(layer, expert_id, x_u16, dw)
            half = gu_u16.shape[-1] // 2
            gate_out = gu_u16[:half]
            up_out = gu_u16[half:]
            dt_b1 += time.perf_counter() - t_b1_0

            t_act0 = time.perf_counter()
            act_f32 = situ_glu(gate_out, up_out, gate_beta=gate_beta, value_beta=value_beta)
            act_u16 = f32_to_bf16_rtne(act_f32)
            routed_acts.append((expert_id, weight, act_u16))
            dt_host += time.perf_counter() - t_act0

        self._record_xclbin_use(b1_xclbin)

        # 2. Shared expert gate+up
        has_sh_gu = (sh_gu_name in self.mock_weight_arrays) or (
            dw is not None
            and self.weights_dir is not None
            and (self.weights_dir / self.bundle_map[sh_gu_name].rel_file).is_file()
        )
        has_b2 = (b2_name in self.mock_weight_arrays) or (
            dw is not None
            and self.weights_dir is not None
            and (self.weights_dir / self.bundle_map[b2_name].rel_file).is_file()
        )
        sh_act_u16: np.ndarray | None = None
        sh_g: np.ndarray | None = None
        sh_u: np.ndarray | None = None
        if has_sh_gu and x_u16.shape[-1] == HIDDEN_SIZE:
            t_sh0 = time.perf_counter()
            sh_flat_u16_ref = self._moe_shared_gu_reference_numpy(layer, x_u16, dw)
            if shared_gu is not None:
                if not np.array_equal(shared_gu, sh_flat_u16_ref):
                    raise ValueError(
                        f"shared_gu mismatch in mock moe_experts at layer {layer}: "
                        "supplied shared_gu bits do not match reference calculation"
                    )
                sh_flat_u16 = shared_gu.copy()
            else:
                sh_flat_u16 = sh_flat_u16_ref
                self._record_xclbin_use(sh_xclbin)
                dt_sh += time.perf_counter() - t_sh0

            t_sh_h0 = time.perf_counter()
            self._bundle_cache[sh_gu_name] = (self._step_id, id(x_orig), sh_flat_u16)
            self._cached_inputs[sh_gu_name] = x_orig
            dt_host += time.perf_counter() - t_sh_h0

            if has_b2:
                t_sh_act0 = time.perf_counter()
                sh_g = sh_flat_u16[:1088]
                sh_u = sh_flat_u16[1088:2176]
                sh_act_u16 = f32_to_bf16_rtne(situ_glu(sh_g, sh_u, gate_beta=gate_beta, value_beta=value_beta))
                dt_host += time.perf_counter() - t_sh_act0
        elif shared_gu is not None:
            raise ValueError(
                f"Cannot verify shared_gu at layer {layer}: "
                f"missing shared weights (has_sh_gu={has_sh_gu}) or hidden dim mismatch ({x_u16.shape[-1]} != {HIDDEN_SIZE})"
            )

        # 3. B2-9 (down proj for routed and shared experts)
        for expert_id, weight, act_u16 in routed_acts:
            t_dn_0 = time.perf_counter()
            expert_out = self._moe_b2_row_numpy(layer, expert_id, act_u16, dw)
            dt_dn += time.perf_counter() - t_dn_0

            t_sum0 = time.perf_counter()
            weighted_out += bf16_to_f32(expert_out) * float(weight)
            dt_host += time.perf_counter() - t_sum0

        if has_sh_gu and x_u16.shape[-1] == HIDDEN_SIZE and has_b2 and sh_act_u16 is not None:
            t_sh_dn0 = time.perf_counter()
            sh_down_u16 = self._moe_b2_row_numpy(layer, 128, sh_act_u16, dw)
            dt_dn += time.perf_counter() - t_sh_dn0

            t_sh_c0 = time.perf_counter()
            self._cached_shared[layer] = {
                "step": self._step_id,
                "sh_gate": sh_g.copy(),
                "sh_up": sh_u.copy(),
                "sh_act_u16": sh_act_u16.copy(),
                "sh_down": sh_down_u16.copy(),
            }
            dt_host += time.perf_counter() - t_sh_c0

        self._record_xclbin_use(dn_xclbin)

        t_end0 = time.perf_counter()
        res = f32_to_bf16_rtne(weighted_out)
        dt_host += time.perf_counter() - t_end0

        self._add_timing("moe_b1_npu", dt_b1, count=1)
        if has_sh_gu and x_u16.shape[-1] == HIDDEN_SIZE:
            self._add_timing("moe_sh_npu", dt_sh, count=1)
        self._add_timing("moe_dn_npu", dt_dn, count=1)
        self._add_timing("moe", dt_b1 + dt_sh + dt_dn, count=1)
        self._add_timing("moe_host", dt_host, count=1)
        self._add_timing("host_prep", dt_host, count=1)
        return res

    # -------------------------------------------------------------------------
    # MLA Attention (Host NumPy Reference)
    # -------------------------------------------------------------------------
    def mla_attend(
        self,
        layer: int,
        q: np.ndarray,
        k_cur: np.ndarray,
        v_cur: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """MLA attention computation and cache update for 1 token."""
        t0 = time.perf_counter()
        k_cur_u16 = f32_to_bf16_rtne(k_cur) if k_cur.dtype == np.float32 else k_cur
        v_cur_u16 = f32_to_bf16_rtne(v_cur) if v_cur.dtype == np.float32 else v_cur
        if k_cur_u16.ndim == 3:
            k_cur_u16 = k_cur_u16.squeeze(1)
        if v_cur_u16.ndim == 3:
            v_cur_u16 = v_cur_u16.squeeze(1)

        cache = self._mla_caches.get(layer)
        if cache is None:
            h, dk = k_cur_u16.shape[0], k_cur_u16.shape[-1]
            _, dv = v_cur_u16.shape[0], v_cur_u16.shape[-1]
            margin = max(256, 1 // 4)
            cache = _MlaLayerCache(h, dk, dv, 1 + margin)
            self._mla_caches[layer] = cache

        cache.append(k_cur_u16, v_cur_u16)
        t = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :t, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :t, :]

        # Attend in FP32
        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        k_f32 = cache.k_f32[:, :t, :]
        v_f32 = cache.v_f32[:, :t, :]

        context_f32 = mla_context(q_f32[np.newaxis], k_f32, v_f32, scale)[0]
        flat_context = context_f32.reshape(-1)
        out = f32_to_bf16_rtne(flat_context)
        dt = time.perf_counter() - t0
        self._add_timing("mla_attend", dt, count=1)
        self._add_timing("mla_host", dt, count=1)
        return out

    def mla_attend_chunk(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """mla_attend for T consecutive tokens at once (q/k [T, H, QK], v [T, H, V]) -> [T, H*V] BF16 uint16.

        The cache gets the same appends token by token; mla_context gives the bytes of T mla_attend calls. The
        timings count one call per chunk."""
        t0 = time.perf_counter()
        k_u16 = f32_to_bf16_rtne(k) if k.dtype == np.float32 else k
        v_u16 = f32_to_bf16_rtne(v) if v.dtype == np.float32 else v
        cache = self._mla_caches.get(layer)
        if cache is None:  # the capacity mla_attend starts with
            cache = _MlaLayerCache(k_u16.shape[1], k_u16.shape[2], v_u16.shape[2], 1 + max(256, 1 // 4))
            self._mla_caches[layer] = cache
        for i in range(k_u16.shape[0]):
            cache.append(k_u16[i], v_u16[i])
        n = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :n, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :n, :]

        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        context_f32 = mla_context(q_f32, cache.k_f32[:, :n, :], cache.v_f32[:, :n, :], scale)
        out = f32_to_bf16_rtne(context_f32.reshape(k_u16.shape[0], -1))
        dt = time.perf_counter() - t0
        self._add_timing("mla_attend", dt, count=1)
        self._add_timing("mla_host", dt, count=1)
        return out

    def set_mla_cache(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        k = key.squeeze(0) if key.ndim == 4 else key
        v = value.squeeze(0) if value.ndim == 4 else value
        k_u16 = np.ascontiguousarray(k, dtype=np.uint16)
        v_u16 = np.ascontiguousarray(v, dtype=np.uint16)
        cache = _MlaLayerCache.from_existing(k_u16, v_u16)
        self._mla_caches[layer] = cache
        t = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :t, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :t, :]

    def get_mla_cache(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        cache = self._mla_caches.get(layer)
        if cache is None or cache.length == 0:
            return None, None
        t = cache.length
        return cache.k_u16[:, :t, :][np.newaxis, ...], cache.v_u16[:, :t, :][np.newaxis, ...]
