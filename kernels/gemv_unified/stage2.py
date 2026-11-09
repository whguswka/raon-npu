# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 q4nx and MoE command streams for the shared GEMV xclbin.

The MoE callables keep the static instruction templates emitted by
``UnifiedGEMV`` and only patch weight ``DDR_PATCH.arg_plus`` fields.  They do
not use the DispatchTime lowering path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B2_BFP16_W_PER_CHUNK_BYTES,
    B2_CHUNKS_PER_EXPERT,
    DN9_BYTES,
    GU_BYTES,
    SHARED_EXPERT_ID,
)
from kernels.moe.patch import PatchedTemplateKernel

from .op import UnifiedGEMV
from .shapes import COLS, MAX_K, UnifiedShape, get_shape


def pad_x(x: np.ndarray, shape: str | UnifiedShape) -> np.ndarray:
    """Pad a general vector or B2's per-column vectors to fixed receiver BDs."""
    s = get_shape(shape) if isinstance(shape, str) else shape
    values = np.asarray(x)
    if s.x_slots == 1:
        flat = values.reshape(-1)
        if flat.size != s.K:
            raise ValueError(f"{s.name}: expected {s.K} x values, got {flat.size}")
        out = np.zeros(MAX_K, dtype=values.dtype)
        out[: s.K] = flat
        return out
    if values.size != s.x_slots * s.K:
        raise ValueError(
            f"{s.name}: expected {s.x_slots * s.K} x values, got {values.size}"
        )
    slots = values.reshape(s.x_slots, s.K)
    if slots.shape[1] != s.K:
        raise ValueError(
            f"{s.name}: expected ({s.x_slots}, {s.K}) x values, got {slots.shape}"
        )
    out = np.zeros((s.x_slots, MAX_K), dtype=values.dtype)
    out[:, : s.K] = slots
    return out.reshape(-1)


@dataclass
class UnifiedMoETemplate:
    """Compile a B1-6 or B2-9 stream against the common unified xclbin."""

    shape_name: str
    context: object = field(default=None, repr=False)
    continuous_mem_w: bool = False
    sequence_plan: str = "per_round"
    cols: int = COLS
    rda_col: int | None = None
    rda_heads: int = 40

    def __post_init__(self) -> None:
        if self.shape_name not in ("moe_b1_6", "moe_b2_9"):
            raise ValueError("shape_name must be 'moe_b1_6' or 'moe_b2_9'")
        self.shape = get_shape(self.shape_name)
        expected_chunk_bytes = (
            B1_W_PER_CHUNK_BYTES
            if self.shape_name == "moe_b1_6"
            else B2_BFP16_W_PER_CHUNK_BYTES
        )
        if self.shape.w_chunk_bytes != expected_chunk_bytes:
            raise AssertionError(
                f"{self.shape_name}: runtime W DMA chunk is "
                f"{self.shape.w_chunk_bytes} B, patch template expects "
                f"{expected_chunk_bytes} B"
            )
        self._op = UnifiedGEMV.for_shape(
            self.shape,
            cols=self.cols,
            context=self.context,
            continuous_mem_w=self.continuous_mem_w,
            sequence_plan=self.sequence_plan,
            rda_col=self.rda_col,
            rda_heads=self.rda_heads,
        )

    @property
    def name(self) -> str:
        return self._op.name

    @property
    def xclbin_artifact(self):
        return self._op.xclbin_artifact

    @property
    def insts_artifact(self):
        return self._op.insts_artifact

    def set_up_artifacts(self) -> None:
        self._op.set_up_artifacts()

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        return self._op.compile(*args, **kwargs)

    def get_arg_spec(self):
        return self._op.get_arg_spec()

    def get_callable(self) -> PatchedTemplateKernel:
        if self.xclbin_artifact is None or self.insts_artifact is None:
            raise RuntimeError("compile/set_up_artifacts must run before get_callable")
        b1 = self.shape_name == "moe_b1_6"
        return PatchedTemplateKernel(
            xclbin_path=Path(self.xclbin_artifact.filename).resolve(),
            insts_path=Path(self.insts_artifact.filename).resolve(),
            w_arg_idx=1,
            chunks_per_expert=(B1_CHUNKS_PER_EXPERT if b1 else B2_CHUNKS_PER_EXPERT),
            chunk_bytes=(B1_W_PER_CHUNK_BYTES if b1 else B2_BFP16_W_PER_CHUNK_BYTES),
            expert_bytes=(GU_BYTES if b1 else DN9_BYTES),
            expected_w_patches=(6 if b1 else 7)
            if self.sequence_plan == "per_call"
            else (6 * B1_CHUNKS_PER_EXPERT if b1 else 7 * B2_CHUNKS_PER_EXPERT),
            slots=(6 if b1 else 7),
            shared_expert_id=SHARED_EXPERT_ID,
            num_host_bos=3,
        )


def assert_one_xclbin(callables: Iterable[Any]) -> tuple[Path, int]:
    """Assert resolved path and mtime identity across normal and patched callables."""
    identity: tuple[Path, int] | None = None
    for fn in callables:
        raw_path = getattr(fn, "xclbin_path", None)
        if raw_path is None:
            raw_path = getattr(fn, "_xclbin_path", None)
        if raw_path is None:
            raise AssertionError(f"callable {fn!r} exposes no xclbin path")
        path = Path(raw_path).resolve()
        current = (path, path.stat().st_mtime_ns)
        if identity is None:
            identity = current
        elif current != identity:
            raise AssertionError(f"xclbin identity mismatch: {current} != {identity}")
    if identity is None:
        raise ValueError("assert_one_xclbin requires at least one callable")
    return identity


class UnifiedMoEGateUp(UnifiedMoETemplate):
    def __init__(
        self,
        *,
        context: object = None,
        continuous_mem_w: bool = False,
        sequence_plan: str = "per_round",
        cols: int = COLS,
        rda_col: int | None = None,
        rda_heads: int = 40,
    ) -> None:
        super().__init__(
            "moe_b1_6",
            context=context,
            continuous_mem_w=continuous_mem_w,
            sequence_plan=sequence_plan,
            cols=cols,
            rda_col=rda_col,
            rda_heads=rda_heads,
        )


class UnifiedMoEDown9(UnifiedMoETemplate):
    def __init__(
        self,
        *,
        context: object = None,
        continuous_mem_w: bool = False,
        sequence_plan: str = "per_round",
        cols: int = COLS,
        rda_col: int | None = None,
        rda_heads: int = 40,
    ) -> None:
        super().__init__(
            "moe_b2_9",
            context=context,
            continuous_mem_w=continuous_mem_w,
            sequence_plan=sequence_plan,
            cols=cols,
            rda_col=rda_col,
            rda_heads=rda_heads,
        )
