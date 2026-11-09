# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""MoE MLIROperators for Raon-1-mini static template and patched-dispatch execution.

Provides:
1. `MoEGateUpTemplate`:
   - Statically compiled GEMV operator template for B1 (gate+up) across 8 columns x 4 rows.
   - Sized for 7 contiguous experts (0..6) with full 129-expert W buffer [TOTAL_GU_WORDS].
   - Compiles static `insts.bin` and `xclbin` with robust BD0 lifetime and token synchronization.

2. `MoEDownTemplate`:
   - Statically compiled GEMV operator template for B2 (down) across 7 columns x 4 rows.
   - Sized for 7 contiguous experts (0..6) with full 129-expert W buffer [TOTAL_DN_WORDS].
   - Dedicated per-column ObjectFifos for each column's slice x_j [1280].

3. `MoEGateUp`:
   - High-level operator wrapper returning a `PatchedTemplateKernel` for B1.
   - Declares dispatch parameters e0..e5.

4. `MoEDown`:
   - High-level operator wrapper returning a `PatchedTemplateKernel` for B2.
   - Declares dispatch parameters e0..e5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.uint16

try:
    import aie.utils as aie_utils
    from iron.common import (
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )

    HAVE_IRON = True
except (ImportError, ModuleNotFoundError):
    HAVE_IRON = False

    class MLIROperator:  # type: ignore[no-redef]
        def __init__(self, context=None):
            self.context = context

        def compile(self, *args, **kwargs):
            pass

        def get_callable(self):
            return None

    class AIERuntimeArgSpec:  # type: ignore[no-redef]
        def __init__(self, direction, shape, dtype=None):
            self.direction = direction
            self.shape = shape
            self.dtype = dtype

    class DesignGenerator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class KernelObjectArtifact:  # type: ignore[no-redef]
        @classmethod
        def from_extern(cls, *args, **kwargs):
            return cls()

    class PythonGeneratedMLIRArtifact:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    aie_utils = None

from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_K,
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B2_CHUNKS_PER_EXPERT,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    B2_W_PER_CHUNK_BYTES,
    B2_BFP16_W_PER_CHUNK_BYTES,
    DN_BYTES,
    DN9_BYTES,
    DN_WORDS,
    GU_BYTES,
    GU_WORDS,
    NUM_ACTIVE_TOTAL,
    NUM_ACTIVE_ROUTED,
    TOTAL_DN_WORDS,
    TOTAL_DN9_WORDS,
    TOTAL_GU_WORDS,
    ROUTED_GU_WORDS,
)
from kernels.moe.patch import PatchedTemplateKernel


# -----------------------------------------------------------------------------
# 1. MoEGateUpTemplate (B1 Static Template MLIROperator)
# -----------------------------------------------------------------------------
@dataclass
class MoEGateUpTemplate(MLIROperator):
    """Statically compiled B1 (gate+up) template operator.

    K in-features: 2560 (B1_K).
    M out-features: 15232 (7 * 2176).
    cols: 8 active columns (32 cores).
    Y receives output vector [15232] (bf16).
    W holds full layer buffer [TOTAL_GU_WORDS] (uint32).
    X holds input vector [2560] (bf16).
    """

    K: int = B1_K
    M: int = NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT  # 15232
    cols: int = 8
    context: object = field(default=None, repr=False)
    slots: int = field(default=NUM_ACTIVE_TOTAL, kw_only=True)

    def __post_init__(self):
        if self.slots not in (6, 7):
            raise ValueError(f"slots must be 6 or 7, got {self.slots}")
        default_m = NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT
        if self.slots != NUM_ACTIVE_TOTAL and self.M == default_m:
            self.M = self.slots * B1_M_PER_EXPERT
        if self.M != self.slots * B1_M_PER_EXPERT:
            raise ValueError("M must equal slots * M_per_expert")
        if HAVE_IRON and aie_utils is not None:
            dev = aie_utils.get_current_device()
            from kernels.moe.design import check_moe_device

            check_moe_device(dev)
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev_name = "aie2p"
        if HAVE_IRON and aie_utils is not None:
            try:
                dev_name = aie_utils.get_current_device().resolve().name
            except Exception:
                pass
        slot_tag = "" if self.slots == NUM_ACTIVE_TOTAL else f"_s{self.slots}"
        return f"Raon_MoEGateUpTemplate_k{self.K}_m{self.M}{slot_tag}_c{self.cols}_{dev_name}"

    def _kernel(self):
        """The custom raon_q4nx_gemv kernel for K=2560."""
        from kernels.gemv.kernel import raon_q4nx_gemv
        from quant.q4nx import GROUP, K_TILE, M_TILE

        return raon_q4nx_gemv(K=self.K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP)

    def get_mlir_artifact(self):
        op_dir = Path(__file__).resolve().parent
        dev = aie_utils.get_current_device() if (aie_utils is not None) else None
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                op_dir / "design.py",
                "moe_gate_up_template",
                (
                    dev,
                    self.K,
                    self.M,
                    self.cols,
                ),
                {"gemv_kernel": self._kernel(), "slots": self.slots},
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(self._kernel())]

    def get_arg_spec(self):
        # The runtime sequence's argument order: Y, W, X.
        return [
            AIERuntimeArgSpec("out", (self.M,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in",
                (
                    (
                        ROUTED_GU_WORDS
                        if self.slots == NUM_ACTIVE_ROUTED
                        else TOTAL_GU_WORDS
                    ),
                ),
                dtype=np.uint32,
            ),
            AIERuntimeArgSpec("in", (self.K,), dtype=bfloat16),
        ]


# -----------------------------------------------------------------------------
# 2. MoEDownTemplate (B2 Static Template MLIROperator)
# -----------------------------------------------------------------------------
@dataclass
class MoEDownTemplate(MLIROperator):
    """Statically compiled B2 (down) template operator.

    K in-features: 1280 (B2_K_PAD).
    M out-features: 17920 (7 * 2560).
    cols: 7 active columns (col 7 unused).
    Y receives output vector [17920] (bf16 partial sums).
    W holds full layer buffer [TOTAL_DN_WORDS] (uint32).
    X holds concatenated input vector [7 x 1280] = [8960] (bf16).
    """

    K: int = B2_K_PAD
    M: int = NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT  # 17920
    cols: int = NUM_ACTIVE_TOTAL  # 7
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if HAVE_IRON and aie_utils is not None:
            dev = aie_utils.get_current_device()
            from kernels.moe.design import check_moe_device

            check_moe_device(dev)
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev_name = "aie2p"
        if HAVE_IRON and aie_utils is not None:
            try:
                dev_name = aie_utils.get_current_device().resolve().name
            except Exception:
                pass
        return f"Raon_MoEDownTemplate_k{self.K}_m{self.M}_c{self.cols}_{dev_name}"

    def _kernel(self):
        """The custom raon_q4nx_gemv kernel for K=1280."""
        from kernels.gemv.kernel import raon_q4nx_gemv
        from quant.q4nx import GROUP, K_TILE, M_TILE

        return raon_q4nx_gemv(K=self.K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP)

    def get_mlir_artifact(self):
        op_dir = Path(__file__).resolve().parent
        dev = aie_utils.get_current_device() if (aie_utils is not None) else None
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                op_dir / "design.py",
                "moe_down_template",
                (dev,),
                {"gemv_kernel": self._kernel()},
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(self._kernel())]

    def get_arg_spec(self):
        # The runtime sequence's argument order: Y, W, X.
        return [
            AIERuntimeArgSpec("out", (self.M,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (TOTAL_DN_WORDS,), dtype=np.uint32),
            AIERuntimeArgSpec("in", (NUM_ACTIVE_TOTAL * self.K,), dtype=bfloat16),
        ]


# -----------------------------------------------------------------------------
# 3. MoEGateUp (Patched Template Operator for B1)
# -----------------------------------------------------------------------------
@dataclass
class MoEGateUp:
    """B1 (gate+up) operator combining static compilation and runtime address patching.

    Compiles `MoEGateUpTemplate` to obtain static xclbin and insts.bin,
    and returns a `PatchedTemplateKernel` configured with ``slots * 17`` patches.
    """

    K: int = B1_K
    M_per_expert: int = B1_M_PER_EXPERT
    active_experts: int = NUM_ACTIVE_TOTAL
    cols: int = 8
    context: object = field(default=None, repr=False)
    slots: int = field(default=NUM_ACTIVE_TOTAL, kw_only=True)

    def __post_init__(self):
        if self.active_experts != NUM_ACTIVE_TOTAL:
            if self.slots != NUM_ACTIVE_TOTAL and self.slots != self.active_experts:
                raise ValueError("slots and active_experts disagree")
            self.slots = self.active_experts
        self._template_op = MoEGateUpTemplate(
            K=self.K,
            M=self.slots * self.M_per_expert,
            cols=self.cols,
            slots=self.slots,
            context=self.context,
        )

    @property
    def name(self) -> str:
        return self._template_op.name

    def get_dispatch_params(self) -> dict[str, type]:
        """Dispatch parameters representing the 6 routed expert IDs."""
        return {
            "e0": np.int32,
            "e1": np.int32,
            "e2": np.int32,
            "e3": np.int32,
            "e4": np.int32,
            "e5": np.int32,
        }

    def get_arg_spec(self):
        return self._template_op.get_arg_spec()

    def compile(self, *args, **kwargs):
        """Compile the underlying static template."""
        return self._template_op.compile(*args, **kwargs)

    def get_callable(self) -> PatchedTemplateKernel:
        """Create and return PatchedTemplateKernel wrapping the compiled template."""
        xclbin_path = self._template_op.xclbin_artifact.filename
        insts_path = self._template_op.insts_artifact.filename
        kernel_name = self._template_op.xclbin_artifact.kernel_name

        return PatchedTemplateKernel(
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            kernel_name=kernel_name,
            w_arg_idx=1,
            chunks_per_expert=B1_CHUNKS_PER_EXPERT,
            chunk_bytes=B1_W_PER_CHUNK_BYTES,
            expert_bytes=GU_BYTES,
            expected_w_patches=self.slots * B1_CHUNKS_PER_EXPERT,
            slots=self.slots,
        )


# -----------------------------------------------------------------------------
# 4. MoEDown (Patched Template Operator for B2)
# -----------------------------------------------------------------------------
@dataclass
class MoEDown:
    """B2 (down) operator combining static compilation and runtime address patching.

    Compiles `MoEDownTemplate` to obtain static xclbin and insts.bin,
    and returns a `PatchedTemplateKernel` configured with 140 weight patches.
    """

    K: int = B2_K_PAD
    M_per_expert: int = B2_M_PER_EXPERT
    active_experts: int = NUM_ACTIVE_TOTAL
    cols: int = NUM_ACTIVE_TOTAL
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        self._template_op = MoEDownTemplate(
            K=self.K,
            M=self.active_experts * self.M_per_expert,
            cols=self.cols,
            context=self.context,
        )

    @property
    def name(self) -> str:
        return self._template_op.name

    def get_dispatch_params(self) -> dict[str, type]:
        """Dispatch parameters representing the 6 routed expert IDs."""
        return {
            "e0": np.int32,
            "e1": np.int32,
            "e2": np.int32,
            "e3": np.int32,
            "e4": np.int32,
            "e5": np.int32,
        }

    def get_arg_spec(self):
        return self._template_op.get_arg_spec()

    def compile(self, *args, **kwargs):
        """Compile the underlying static template."""
        return self._template_op.compile(*args, **kwargs)

    def get_callable(self) -> PatchedTemplateKernel:
        """Create and return PatchedTemplateKernel wrapping the compiled template."""
        xclbin_path = self._template_op.xclbin_artifact.filename
        insts_path = self._template_op.insts_artifact.filename
        kernel_name = self._template_op.xclbin_artifact.kernel_name

        return PatchedTemplateKernel(
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            kernel_name=kernel_name,
            w_arg_idx=1,
            chunks_per_expert=B2_CHUNKS_PER_EXPERT,
            chunk_bytes=B2_W_PER_CHUNK_BYTES,
            expert_bytes=DN_BYTES,
            expected_w_patches=140,
        )


# -----------------------------------------------------------------------------
# 5. MoEDownBfp16Template / MoEDownBfp16 (D mixed_full B2)
# -----------------------------------------------------------------------------
@dataclass
class MoEDownBfp16Template(MLIROperator):
    """Seven-column B2 template backed by the 9-bit bfp16ebs8 GEMV kernel."""

    K: int = B2_K_PAD
    M: int = NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT
    cols: int = NUM_ACTIVE_TOTAL
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.K != B2_K_PAD or self.M != NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT:
            raise ValueError("MoEDownBfp16Template requires K=1280 and M=7*2560")
        if self.cols != NUM_ACTIVE_TOTAL:
            raise ValueError("MoEDownBfp16Template requires exactly 7 columns")
        if HAVE_IRON and aie_utils is not None:
            from kernels.moe.design import check_moe_device

            check_moe_device(aie_utils.get_current_device())
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev_name = "aie2p"
        if HAVE_IRON and aie_utils is not None:
            try:
                dev_name = aie_utils.get_current_device().resolve().name
            except Exception:
                pass
        return f"Raon_MoEDownBfp16Template_k{self.K}_m{self.M}_c{self.cols}_{dev_name}"

    def _kernel(self):
        from kernels.gemv_bfp16.kernel import raon_bfp16_gemv
        from quant.bfp16 import GROUP, K_TILE, M_TILE

        return raon_bfp16_gemv(K=self.K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP)

    def get_mlir_artifact(self):
        op_dir = Path(__file__).resolve().parent
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                op_dir / "design.py",
                "moe_down_bfp16_template",
                (dev,),
                {"gemv_kernel": self._kernel()},
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(self._kernel())]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("out", (self.M,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (TOTAL_DN9_WORDS,), dtype=np.uint32),
            AIERuntimeArgSpec("in", (NUM_ACTIVE_TOTAL * self.K,), dtype=bfloat16),
        ]


@dataclass
class MoEDownBfp16:
    """Compile and patch the seven-slot 9-bit B2 static template."""

    K: int = B2_K_PAD
    M_per_expert: int = B2_M_PER_EXPERT
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        self._template_op = MoEDownBfp16Template(
            K=self.K, M=NUM_ACTIVE_TOTAL * self.M_per_expert, context=self.context
        )

    @property
    def name(self) -> str:
        return self._template_op.name

    def get_dispatch_params(self) -> dict[str, type]:
        return {f"e{i}": np.int32 for i in range(NUM_ACTIVE_ROUTED)}

    def get_arg_spec(self):
        return self._template_op.get_arg_spec()

    def compile(self, *args, **kwargs):
        return self._template_op.compile(*args, **kwargs)

    def get_callable(self) -> PatchedTemplateKernel:
        return PatchedTemplateKernel(
            xclbin_path=self._template_op.xclbin_artifact.filename,
            insts_path=self._template_op.insts_artifact.filename,
            kernel_name=self._template_op.xclbin_artifact.kernel_name,
            w_arg_idx=1,
            chunks_per_expert=B2_CHUNKS_PER_EXPERT,
            chunk_bytes=B2_BFP16_W_PER_CHUNK_BYTES,
            expert_bytes=DN9_BYTES,
            expected_w_patches=NUM_ACTIVE_TOTAL * B2_CHUNKS_PER_EXPERT,
            slots=NUM_ACTIVE_TOTAL,
        )
