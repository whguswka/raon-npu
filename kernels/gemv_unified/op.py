# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unified stage-1 9-bit bfp16ebs8 GEMV operator.

Uses the configuration-only xclbin + shape-specific instruction stream pattern:
  - One `XclbinArtifact` (`config_name.xclbin`) is compiled once from a reference
    shape and shared across all 10 stage-1 shapes.
  - Each `(format, K, M)` shape compiles only its own `InstsBinArtifact`
    (`name.bin`), containing RTP writes, worker barrier releases, and the
    format-length runtime mem/core W BD sequences.
  - All shapes share the same resolved xclbin path, mtime, kernel name (`MLIR_AIE`),
    and 3-BO ABI `(Y, W, X)`, verified via runtime assertions before execution.
  - Dispatch-time (`DispatchTime`) compilation paths are strictly prohibited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kernels.gemv_unified.design import check_shape
from kernels.gemv_unified.kernel import unified_bfp16_kernel
from kernels.gemv_unified.shapes import (
    ALL_SHAPES,
    COLS,
    MAX_K,
    M_PER_CHUNK,
    UnifiedShape,
    get_shape,
    selected_shapes,
)
from bench.rda_state.rda_state_1col import MODEL_RDA_HEADS, make_rda_kernel_object

_IMPORT_ERROR: Exception | None = None
try:
    import aie.utils as aie_utils
    from ml_dtypes import bfloat16

    from iron.common import (
        AIEContext,
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact, XclbinArtifact
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:  # pragma: no cover - CPU-only host lacks IRON runtime
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    aie_utils = None
    bfloat16 = None  # type: ignore[assignment]
    AIEContext = object  # type: ignore[assignment]
    AIERuntimeArgSpec = object  # type: ignore[assignment]
    DesignGenerator = object  # type: ignore[assignment]
    KernelObjectArtifact = object  # type: ignore[assignment]
    MLIROperator = object  # type: ignore[assignment,misc]
    PythonGeneratedMLIRArtifact = object  # type: ignore[assignment]
    InstsBinArtifact = object  # type: ignore[assignment]
    XclbinArtifact = object  # type: ignore[assignment]

KERNEL_NAME = "MLIR_AIE"

# Tracks the active resolved (xclbin_path, mtime_ns, kernel_name, abi_signature)
# so every shape callable asserts compatibility at initialization time.
_ACTIVE_XCLBIN_SIGNATURE: dict[str, Any] = {}


def _shape_tag(shape: UnifiedShape) -> str:
    """Format the unique six-field signature tag for a UnifiedShape."""
    c_tag = "auto" if shape.chunks_per_col is None else str(shape.chunks_per_col)
    return (
        f"f{shape.format}_k{shape.K}_m{shape.M}_a{shape.active_cols}_x"
        f"{shape.x_slots}_c{c_tag}"
    )


def reset_xclbin_signature_cache() -> None:
    """Clear the cached xclbin signature (used by unit tests)."""
    _ACTIVE_XCLBIN_SIGNATURE.clear()


def _abi_signature_from_specs(specs: Iterable[Any]) -> tuple[tuple[str, str, int], ...]:
    """Extract direction, dtype, and rank for each BO in the 3-BO ABI (Y, W, X)."""
    sig = []
    for s in specs:
        direction = getattr(s, "direction", None)
        dtype = str(np.dtype(getattr(s, "dtype", None)))
        shape = getattr(s, "shape", ())
        sig.append((str(direction), dtype, len(shape)))
    return tuple(sig)


class SharedXclbinCallable:
    """Thin callable wrapper that preserves metadata for shared-xclbin assertions."""

    def __init__(
        self,
        inner: Any,
        *,
        op: "UnifiedGEMV",
        xclbin_path: Path,
        xclbin_mtime_ns: int,
        insts_path: Path,
        kernel_name: str,
        arg_spec: list[Any],
    ) -> None:
        self._inner = inner
        self.op = op
        self.K = op.K
        self.M = op.M
        self.cols = op.cols
        self.xclbin_path = xclbin_path
        self.xclbin_mtime_ns = xclbin_mtime_ns
        self.insts_path = insts_path
        self.kernel_name = kernel_name
        self.arg_spec = list(arg_spec)
        self.abi_signature = _abi_signature_from_specs(self.arg_spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) != 3 or kwargs:
            raise ValueError(
                f"UnifiedGEMV callable requires positional 3-BO ABI (Y, W, X), "
                f"got {len(args)} args and kwargs={list(kwargs)}"
            )
        return self._inner(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def assert_shared_xclbin(items: Iterable[Any]) -> dict[str, Any]:
    """Assert that all provided UnifiedGEMV ops or callables share one xclbin and 3-BO ABI.

    Checks:
      1. Resolved `xclbin_path` is identical across all items.
      2. `xclbin` file exists and its `st_mtime_ns` matches across all items.
      3. `kernel_name == "MLIR_AIE"` for all items.
      4. 3-BO ABI `(Y: out bf16 1D, W: in uint32 1D, X: in bf16 1D)` matches across all items.
    """
    items_list = list(items)
    if not items_list:
        raise ValueError("assert_shared_xclbin requires at least one operator or callable")

    expected: dict[str, Any] | None = None
    for idx, item in enumerate(items_list):
        if isinstance(item, SharedXclbinCallable):
            xclbin_path = item.xclbin_path.resolve()
            mtime_ns = item.xclbin_mtime_ns
            kernel_name = item.kernel_name
            arg_spec = item.arg_spec
        else:
            xclbin_art = getattr(item, "xclbin_artifact", None)
            if xclbin_art is None:
                raise AssertionError(f"Item {idx} ({item!r}) has no xclbin_artifact")
            xclbin_path = Path(xclbin_art.filename).resolve()
            mtime_ns = xclbin_path.stat().st_mtime_ns if xclbin_path.exists() else -1
            kernel_name = getattr(xclbin_art, "kernel_name", KERNEL_NAME)
            arg_spec = list(item.get_arg_spec())

        assert len(arg_spec) == 3, (
            f"Item {idx} must use 3-BO ABI (Y, W, X), got {len(arg_spec)} args"
        )
        abi_sig = _abi_signature_from_specs(arg_spec)
        expected_abi = (
            ("out", str(np.dtype(bfloat16)) if bfloat16 is not None else "bfloat16", 1),
            ("in", str(np.dtype(np.uint32)), 1),
            ("in", str(np.dtype(bfloat16)) if bfloat16 is not None else "bfloat16", 1),
        )
        assert abi_sig == expected_abi, (
            f"Item {idx} ABI {abi_sig} does not match expected 3-BO ABI (Y, W, X) {expected_abi}"
        )
        assert kernel_name == KERNEL_NAME, (
            f"Item {idx} kernel_name={kernel_name!r}, expected {KERNEL_NAME!r}"
        )

        cur = {
            "xclbin_path": xclbin_path,
            "xclbin_mtime_ns": mtime_ns,
            "kernel_name": kernel_name,
            "abi_signature": abi_sig,
        }
        if expected is None:
            expected = cur
        else:
            assert cur["xclbin_path"] == expected["xclbin_path"], (
                f"xclbin path mismatch at item {idx}: "
                f"{cur['xclbin_path']} != {expected['xclbin_path']}"
            )
            assert cur["xclbin_mtime_ns"] == expected["xclbin_mtime_ns"], (
                f"xclbin mtime mismatch at item {idx}: "
                f"{cur['xclbin_mtime_ns']} != {expected['xclbin_mtime_ns']}"
            )
            assert cur["kernel_name"] == expected["kernel_name"], (
                f"kernel_name mismatch at item {idx}: "
                f"{cur['kernel_name']} != {expected['kernel_name']}"
            )
            assert cur["abi_signature"] == expected["abi_signature"], (
                f"ABI signature mismatch at item {idx}: "
                f"{cur['abi_signature']} != {expected['abi_signature']}"
            )

    assert expected is not None
    return expected


@dataclass
class UnifiedGEMV(MLIROperator):
    """Unified 9-bit bfp16ebs8 GEMV operator sharing one configuration-only xclbin."""

    K: int
    M: int
    cols: int = COLS
    shape_name: str | None = None
    format: int = 0
    active_cols: int | None = None
    x_slots: int = 1
    chunks_per_col: int | None = None
    continuous_mem_w: bool = False
    sequence_plan: str = "per_round"
    diagnostic_receive_only: bool = False
    diagnostic_static_core: bool = False
    diagnostic_small_x: bool = False
    rda_col: int | None = None
    rda_heads: int = MODEL_RDA_HEADS
    experimental_shape: bool = False
    context: object = field(default=None, repr=False)

    _reference_shape: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.sequence_plan not in ("per_round", "per_call"):
            raise ValueError(
                "sequence_plan must be 'per_round' or 'per_call', "
                f"got {self.sequence_plan!r}"
            )
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        raw_shape: UnifiedShape = check_shape(
            dev, self.K, self.M, self.cols, format=self.format,
            active_cols=self.active_cols, x_slots=self.x_slots,
            chunks_per_col=self.chunks_per_col,
        )
        effective_active_cols = raw_shape.resolve_active_cols(self.cols)
        if effective_active_cols != raw_shape.active_cols:
            self.shape = UnifiedShape(
                name=raw_shape.name,
                K=raw_shape.K,
                M=raw_shape.M,
                format=raw_shape.format,
                active_cols=effective_active_cols,
                x_slots=raw_shape.x_slots,
                chunks_per_col=raw_shape.chunks_per_col,
            )
        else:
            self.shape = raw_shape

        if self.rda_col is not None:
            if not (0 <= self.rda_col < COLS):
                raise ValueError(
                    f"rda_col must be within device columns 0..{COLS - 1}, got {self.rda_col}"
                )
            if self.rda_col < self.cols:
                raise ValueError(
                    f"rda_col {self.rda_col} overlaps GEMV active columns 0..{self.cols - 1}"
                )
            if self.rda_heads != MODEL_RDA_HEADS:
                raise ValueError(
                    f"the merged config path supports only rda_heads={MODEL_RDA_HEADS} "
                    f"(the model contract head count): the shared xclbin/insts artifact "
                    f"names carry no head count, and BD chains with runs=H//2 could "
                    f"leave heads unconsumed at any other count, got {self.rda_heads}"
                )

        # Minimal reference where every device column executes exactly one round.
        self._reference_shape = (256, self.cols * M_PER_CHUNK)

        supported = next(
            (
                shape
                for shape in ALL_SHAPES
                if self._shape_fields(shape, self.cols) == self._shape_fields(self.shape, self.cols)
            ),
            None,
        )
        diagnostic = (
            self.diagnostic_receive_only
            or self.diagnostic_static_core
            or self.diagnostic_small_x
        )
        if supported is None and not (self.experimental_shape or diagnostic):
            raise ValueError(self._unsupported_shape_message())
        if self.shape_name is None:
            self.shape_name = supported.name if supported is not None else self.shape.name
        if MLIROperator is not object:
            MLIROperator.__init__(self, context=self.context)

    @staticmethod
    def _shape_fields(
        shape: UnifiedShape,
        cols: int | None = None,
    ) -> tuple[int, int, int, int, int, int | None]:
        active = shape.resolve_active_cols(cols)
        return (
            shape.K,
            shape.M,
            shape.format,
            active,
            shape.x_slots,
            shape.chunks_per_col,
        )

    def _unsupported_shape_message(self) -> str:
        requested = self._shape_fields(self.shape, self.cols)
        field_names = (
            "K",
            "M",
            "format",
            "active_cols",
            "x_slots",
            "chunks_per_col",
        )
        closest = min(
            ALL_SHAPES,
            key=lambda shape: sum(
                left != right
                for left, right in zip(requested, self._shape_fields(shape, self.cols))
            ),
        )
        differences = ", ".join(
            f"{name}={actual!r} (closest {closest.name}={expected!r})"
            for name, actual, expected in zip(
                field_names, requested, self._shape_fields(closest, self.cols)
            )
            if actual != expected
        )
        names = ", ".join(shape.name for shape in ALL_SHAPES)
        return (
            "unsupported unified GEMV shape: "
            f"{differences}; supported names: {names}. "
            "Use experimental_shape=True only for explicit diagnostic/experimental runs."
        )

    @classmethod
    def for_shape(
        cls,
        shape: str | UnifiedShape,
        *,
        cols: int = COLS,
        context: object = None,
        experimental_shape: bool = False,
        continuous_mem_w: bool = False,
        sequence_plan: str = "per_round",
        rda_col: int | None = None,
        rda_heads: int = MODEL_RDA_HEADS,
    ) -> "UnifiedGEMV":
        """Construct an operator from a registered name or explicit shape object."""
        resolved = get_shape(shape) if isinstance(shape, str) else shape
        # B1's baseline seven-column round-robin layout is part of the
        # byte-for-byte per_round contract.  Only the per_call stream derives
        # a six-column contiguous layout, reducing patchable W BDs from 102 to
        # six while leaving K/M and the device body unchanged.
        if sequence_plan == "per_call" and resolved.name == "moe_b1_6":
            resolved = UnifiedShape(
                resolved.name,
                resolved.K,
                resolved.M,
                format=resolved.format,
                active_cols=6,
                x_slots=resolved.x_slots,
                chunks_per_col=17,
            )
            experimental_shape = True
        return cls(
            K=resolved.K,
            M=resolved.M,
            cols=cols,
            shape_name=resolved.name,
            format=resolved.format,
            active_cols=resolved.active_cols,
            x_slots=resolved.x_slots,
            chunks_per_col=resolved.chunks_per_col,
            continuous_mem_w=continuous_mem_w,
            sequence_plan=sequence_plan,
            rda_col=rda_col,
            rda_heads=rda_heads,
            experimental_shape=experimental_shape,
            context=context,
        )

    @property
    def config_name(self) -> str:
        """Artifact prefix for the xclbin.

        Non-diagnostic operators keep the shape-independent name so every
        stage-1 shape shares one configuration-only xclbin.  Diagnostic
        operators (receive-only / static-core / small-x) build a fresh xclbin
        for each exact shape and configuration, so all six shape fields and
        active diagnostic flags are baked into the name.
        """
        dev = (
            aie_utils.get_current_device().resolve().name
            if aie_utils is not None
            else "npu2"
        )
        options = []
        if self.continuous_mem_w:
            options.append("contw")
        if self.diagnostic_receive_only:
            options.append("recv")
        if self.diagnostic_static_core:
            options.append("static")
        if self.diagnostic_small_x:
            options.append("smallx")
        suffix = "_" + "_".join(options) if options else ""
        rda_part = f"_rda{self.rda_col}" if self.rda_col is not None else ""
        diagnostic = (
            self.diagnostic_receive_only
            or self.diagnostic_static_core
            or self.diagnostic_small_x
        )
        if diagnostic:
            return (
                f"Raon_GEMV_Unified_c{self.cols}{rda_part}_{dev}_{_shape_tag(self.shape)}{suffix}"
            )
        return f"Raon_GEMV_Unified_c{self.cols}{rda_part}_{dev}{suffix}"

    @property
    def name(self) -> str:
        """Format/shape-dependent prefix for the per-stream instruction binary."""
        plan_suffix = "_pc" if self.sequence_plan == "per_call" else ""
        return f"{self.config_name}_{_shape_tag(self.shape)}{plan_suffix}"

    def _kernel(self):
        """The unified bfp16 kernel bound to the current device.

        ``diagnostic_small_x`` uses the actual K for the X buffer signature,
        matching ``design.py`` and ``get_kernel_artifacts``; the shared
        configuration path keeps the MAX_K signature.
        """
        if aie_utils is not None:
            try:
                import aie.utils.hostruntime as hostruntime
                hostruntime.set_current_device(aie_utils.get_current_device())
            except Exception:
                pass
        if self.diagnostic_small_x:
            return unified_bfp16_kernel(K=self.K)
        return unified_bfp16_kernel()

    def get_kernel_artifacts(self):
        artifacts = [KernelObjectArtifact.from_extern(self._kernel())]
        if self.rda_col is not None:
            rda_kernel = make_rda_kernel_object()
            if rda_kernel is not None:
                artifacts.append(KernelObjectArtifact.from_extern(rda_kernel))
        return artifacts

    def _mlir_artifact(
        self,
        filename: str,
        K: int,
        M: int,
        *,
        shape: UnifiedShape | None = None,
        mode: str = "gemv",
    ) -> PythonGeneratedMLIRArtifact:
        design_path = Path(__file__).parent / "design.py"
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        s = shape or self.shape
        return PythonGeneratedMLIRArtifact(
            filename,
            DesignGenerator(
                design_path,
                "gemv_unified",
                (
                    dev,
                    K,
                    M,
                    self.cols,
                    0,
                ),
                {
                    "gemv_kernel": self._kernel(),
                    "format": s.format,
                    "active_cols": s.active_cols,
                    "x_slots": s.x_slots,
                    "chunks_per_col": s.chunks_per_col,
                    "continuous_mem_w": self.continuous_mem_w,
                    "sequence_plan": self.sequence_plan,
                    "diagnostic_receive_only": self.diagnostic_receive_only,
                    "diagnostic_static_core": self.diagnostic_static_core,
                    "diagnostic_small_x": self.diagnostic_small_x,
                    "rda_col": self.rda_col,
                    "rda_heads": self.rda_heads,
                    "mode": mode,
                },
            ),
        )

    def get_mlir_artifact(self):
        return self._mlir_artifact(f"{self.name}.mlir", self.K, self.M)

    def get_xclbin_artifact(self, config_mlir, kernels):
        return XclbinArtifact(
            f"{self.config_name}.xclbin",
            mlir_input=config_mlir,
            dependencies=[config_mlir] + kernels,
            kernel_name=KERNEL_NAME,
            extra_flags=["--dynamic-objFifos"],
        )

    def get_insts_artifact(self, shape_mlir, kernels):
        return InstsBinArtifact(
            f"{self.name}.bin",
            mlir_input=shape_mlir,
            dependencies=[shape_mlir] + kernels,
            extra_flags=["--dynamic-objFifos"],
        )

    def set_up_artifacts(self) -> None:
        """Register one device xclbin and a format-length runtime DMA stream."""
        kernels = self.get_kernel_artifacts()

        diagnostic_device = (
            self.diagnostic_receive_only
            or self.diagnostic_static_core
            or self.diagnostic_small_x
        )
        ref_K, ref_M = (
            (self.K, self.M) if diagnostic_device else self._reference_shape
        )
        config_shape = self.shape if diagnostic_device else UnifiedShape(
            "configuration", ref_K, ref_M
        )
        config_mlir = self._mlir_artifact(
            f"{self.config_name}.mlir", ref_K, ref_M,
            shape=config_shape,
        )
        if diagnostic_device or (
            self._shape_fields(self.shape) == self._shape_fields(config_shape)
        ):
            shape_mlir = config_mlir
        else:
            shape_mlir = self.get_mlir_artifact()

        self.mlir_artifact = shape_mlir
        self.xclbin_artifact = self.get_xclbin_artifact(config_mlir, kernels)
        self.insts_artifact = self.get_insts_artifact(shape_mlir, kernels)

        if hasattr(self, "add_artifacts") and callable(getattr(self, "add_artifacts")):
            self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def register_artifacts(self, artifacts: list[Any]) -> None:
        """Compatibility alias for add_artifacts."""
        if hasattr(self, "add_artifacts") and callable(getattr(self, "add_artifacts")):
            self.add_artifacts(artifacts)

    def get_arg_spec(self):
        """Fixed 3-BO ABI; X is MAX_K-padded for the fixed receiver BD."""
        bf16_dtype = bfloat16 if bfloat16 is not None else np.uint16
        x_elems = self.K if self.diagnostic_small_x else MAX_K
        return [
            AIERuntimeArgSpec("out", (self.M,), dtype=bf16_dtype),
            AIERuntimeArgSpec("in", (self.shape.weight_words,), dtype=np.uint32),
            AIERuntimeArgSpec(
                "in", (self.shape.x_slots * x_elems,), dtype=bf16_dtype
            ),
        ]

    def get_callable(self) -> SharedXclbinCallable:
        """Return an XRT callable and assert shared-xclbin invariants at startup."""
        assert self.xclbin_artifact is not None, "xclbin_artifact not initialized"
        assert self.insts_artifact is not None, "insts_artifact not initialized"

        xclbin_path = Path(self.xclbin_artifact.filename).resolve()
        insts_path = Path(self.insts_artifact.filename).resolve()
        assert xclbin_path.is_file(), f"Shared xclbin file does not exist: {xclbin_path}"
        assert insts_path.is_file(), f"Shape insts.bin file does not exist: {insts_path}"

        mtime_ns = xclbin_path.stat().st_mtime_ns
        kernel_name = getattr(self.xclbin_artifact, "kernel_name", KERNEL_NAME)
        arg_spec = self.get_arg_spec()

        inner = super().get_callable()
        wrapped = SharedXclbinCallable(
            inner,
            op=self,
            xclbin_path=xclbin_path,
            xclbin_mtime_ns=mtime_ns,
            insts_path=insts_path,
            kernel_name=kernel_name,
            arg_spec=arg_spec,
        )

        # Verify against our own operator spec and any previously created shape callable
        sig = assert_shared_xclbin([wrapped])
        if _ACTIVE_XCLBIN_SIGNATURE:
            assert sig == _ACTIVE_XCLBIN_SIGNATURE, (
                f"UnifiedGEMV callable for {self.name} violated shared-xclbin invariant: "
                f"{sig} != {_ACTIVE_XCLBIN_SIGNATURE}"
            )
        else:
            _ACTIVE_XCLBIN_SIGNATURE.update(sig)

        return wrapped


def build_shape_ops(
    shape_names: Iterable[str] | None = None,
    *,
    cols: int = COLS,
    context: object = None,
) -> dict[str, UnifiedGEMV]:
    """Instantiate UnifiedGEMV operators for the requested stage-1 shapes."""
    shapes = selected_shapes(shape_names)
    if context is None and AIEContext is not None:
        context = AIEContext()
    ops: dict[str, UnifiedGEMV] = {}
    for s in shapes:
        op = UnifiedGEMV.for_shape(s, cols=cols, context=context)
        if hasattr(op, "set_up_artifacts") and op.context is not None:
            op.set_up_artifacts()
        ops[s.name] = op
    if ops:
        assert_shared_xclbin(ops.values())
    return ops


GEMV_Unified = UnifiedGEMV
