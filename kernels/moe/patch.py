# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""TXN decoding and runtime address patching for Raon MoE static templates.

Instead of synthesizing new instruction streams via dynamic dispatch lowering
(which triggers driver TDR hangs due to premature BD0 deallocation on column 0),
this module parses statically-compiled instruction templates (whose BD allocation
and token synchronization are verified and robust) and patches the DDR_PATCH
`arg_plus` offsets for weight DMA transfers per execution call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np

try:
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.hostruntime import HostRuntimeError
except (ImportError, ModuleNotFoundError):

    class NPUKernel:  # type: ignore[no-redef]
        def __init__(
            self,
            xclbin_path=None,
            insts_path=None,
            device_index=0,
            kernel_name="MLIR_AIE",
            trace_config=None,
            num_host_bos=None,
            elf_path=None,
            dispatch_params=None,
            dispatch_lib_path=None,
        ):
            self._xclbin_path = xclbin_path
            self._insts_path = insts_path
            self._elf_path = elf_path
            self._kernel_name = kernel_name
            self._trace_config = trace_config
            self._device_index = device_index
            self._num_host_bos = num_host_bos
            self._dispatch_params = list(dispatch_params) if dispatch_params else []
            self._dispatch_lib_path = dispatch_lib_path
            self._dispatch_bridge = None

        @property
        def dispatch_params(self) -> list[str]:
            return list(self._dispatch_params)

        @property
        def xclbin_path(self):
            return self._xclbin_path

        @property
        def insts_path(self):
            return self._insts_path

        @property
        def kernel_name(self):
            return self._kernel_name

        @property
        def trace_config(self):
            return self._trace_config

        @property
        def num_host_bos(self):
            return self._num_host_bos

        def _generate_dispatch_insts(self, dispatch_scalars: dict | None):
            raise NotImplementedError

        def instructions(self, **dispatch_scalars) -> np.ndarray:
            words = self._generate_dispatch_insts(dispatch_scalars)
            if words is not None:
                return words
            if self._insts_path is None:
                raise ValueError("this kernel has no separate instruction stream")
            return np.fromfile(self._insts_path, dtype=np.uint32)

        def __call__(self, *args, **kwargs):
            dispatch_names = set(self._dispatch_params)
            unknown = set(kwargs) - dispatch_names - {"retry"}
            if unknown:
                raise TypeError(f"Unexpected keyword argument(s): {sorted(unknown)}")
            try:
                from aie.utils import DefaultNPURuntime
            except Exception:
                DefaultNPURuntime = None
            if DefaultNPURuntime is None:
                raise Exception("Cannot run kernel; DefaultNPURuntime not set.")
            dispatch_scalars = {k: v for k, v in kwargs.items() if k in dispatch_names}
            other_kwargs = {k: v for k, v in kwargs.items() if k not in dispatch_names}
            return DefaultNPURuntime.load_and_run(
                self,
                list(args),
                dispatch_scalars=dispatch_scalars or None,
                **other_kwargs,
            )

    class HostRuntimeError(Exception):  # type: ignore[no-redef]
        pass


from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    GU_BYTES,
    SHARED_EXPERT_ID,
)

# -----------------------------------------------------------------------------
# TXN Firmware Opcodes (mlir-aie/include/aie/Runtime/TxnEncoding.h)
# -----------------------------------------------------------------------------
TXN_OPC_WRITE: int = 0
TXN_OPC_BLOCKWRITE: int = 1
TXN_OPC_BLOCKSET: int = 2
TXN_OPC_MASKWRITE: int = 3
TXN_OPC_MASKPOLL: int = 4
TXN_OPC_NOOP: int = 5
TXN_OPC_PREEMPT: int = 6
TXN_OPC_MASKPOLL_BUSY: int = 7
TXN_OPC_LOADPDI: int = 8
TXN_OPC_LOAD_PM_START: int = 9
TXN_OPC_CREATE_SCRATCHPAD: int = 10
TXN_OPC_UPDATE_STATE_TABLE: int = 11
TXN_OPC_UPDATE_REG: int = 12
TXN_OPC_UPDATE_SCRATCH: int = 13
TXN_OPC_CONFIG_SHIMDMA_BD: int = 14
TXN_OPC_CONFIG_SHIMDMA_DMABUF_BD: int = 15

TXN_OPC_CUSTOM_OP_BEGIN: int = 1 << 7  # 128
TXN_OPC_TCT: int = TXN_OPC_CUSTOM_OP_BEGIN  # 128
TXN_OPC_DDR_PATCH: int = TXN_OPC_CUSTOM_OP_BEGIN + 1  # 129
TXN_OPC_READ_REGS: int = TXN_OPC_CUSTOM_OP_BEGIN + 2  # 130
TXN_OPC_RECORD_TIMER: int = TXN_OPC_CUSTOM_OP_BEGIN + 3  # 131
TXN_OPC_MERGE_SYNC: int = TXN_OPC_CUSTOM_OP_BEGIN + 4  # 132


def parse_txn_ops(words: np.ndarray) -> list[tuple[int, int, int]]:
    """Parse a TXN uint32 word stream into a sequence of operations.

    Args:
        words: 1D uint32 numpy array containing a TXN binary with a 4-word header.

    Returns:
        List of (opcode, word_offset, word_size) tuples.
    """
    if len(words) < 4:
        raise ValueError(
            f"TXN stream is too short: {len(words)} words (minimum header is 4 words)"
        )

    # Validate header total size
    total_bytes = int(words[3])
    if total_bytes != len(words) * 4:
        raise ValueError(
            f"TXN header size mismatch: header declares {total_bytes} bytes, array has {len(words) * 4} bytes"
        )

    ops: list[tuple[int, int, int]] = []
    pos = 4  # skip 4-word TXN header
    n_words = len(words)

    while pos < n_words:
        opc = int(words[pos] & 0xFF)
        if opc == TXN_OPC_WRITE:
            sz = int(words[pos + 5]) // 4
        elif opc == TXN_OPC_BLOCKWRITE:
            sz = int(words[pos + 3]) // 4
        elif opc in (TXN_OPC_MASKWRITE, TXN_OPC_MASKPOLL):
            sz = int(words[pos + 6]) // 4
        elif opc == TXN_OPC_PREEMPT:
            sz = 1
        elif opc == TXN_OPC_LOADPDI:
            sz = 4
        elif opc == TXN_OPC_TCT:
            sz = int(words[pos + 1]) // 4
        elif opc == TXN_OPC_DDR_PATCH:
            sz = int(words[pos + 1]) // 4
        elif opc in (TXN_OPC_READ_REGS, TXN_OPC_RECORD_TIMER, TXN_OPC_MERGE_SYNC):
            sz = (
                int(words[pos + 1]) // 4
                if (pos + 1 < n_words and words[pos + 1] > 0)
                else 4
            )
        else:
            raise ValueError(f"Unknown TXN opcode {opc} at word {pos}")

        if sz <= 0:
            raise ValueError(
                f"Invalid operation size {sz} words for opcode {opc} at word {pos}"
            )
        if pos + sz > n_words:
            raise ValueError(
                f"Operation {opc} at word {pos} with size {sz} overflows TXN buffer ({n_words} words)"
            )

        ops.append((opc, pos, sz))
        pos += sz

    return ops


def find_ddr_patches(
    words: np.ndarray, target_arg_idx: int | None = None
) -> list[tuple[int, int, int]]:
    """Scan TXN word stream and extract DDR_PATCH operations.

    Args:
        words: 1D uint32 numpy array.
        target_arg_idx: Optional buffer argument index filter.

    Returns:
        List of (word_offset, arg_idx, arg_plus) tuples.
    """
    ops = parse_txn_ops(words)
    patches = []
    for opc, pos, _ in ops:
        if opc == TXN_OPC_DDR_PATCH:
            arg_idx = int(words[pos + 8])
            if target_arg_idx is None or arg_idx == target_arg_idx:
                arg_plus = int(words[pos + 10]) | (int(words[pos + 11]) << 32)
                patches.append((pos, arg_idx, arg_plus))
    return patches


class PatchedTemplateKernel(NPUKernel):
    """NPUKernel executing a static instruction template with per-call W DDR_PATCH modifications.

    Maintains the robust BD lifetime and token synchronization of statically-compiled GEMV
    programs while allowing dynamic runtime selection of MoE experts by rewriting W DDR_PATCH
    offsets directly in the instruction stream before submitting the buffer object to XRT.
    """

    def __init__(
        self,
        xclbin_path: str | Path,
        insts_path: str | Path,
        *,
        w_arg_idx: int = 1,
        chunks_per_expert: int = B1_CHUNKS_PER_EXPERT,  # 17 for B1, 20 for B2
        chunk_bytes: int = B1_W_PER_CHUNK_BYTES,  # 204,800 (0x32000) for B1, 102,400 (0x19000) for B2
        expert_bytes: int = GU_BYTES,  # 3,481,600 (0x352000) for B1, 2,048,000 (0x1f4000) for B2
        expected_w_patches: int = 119,  # 119 for B1 (7x17), 140 for B2 (7x20)
        slots: int = 7,
        shared_expert_id: int = SHARED_EXPERT_ID,  # 128
        dispatch_params: Sequence[str] | None = None,
        device_index: int = 0,
        kernel_name: str = "MLIR_AIE",
        trace_config=None,
        num_host_bos: int | None = None,
    ):
        if dispatch_params is None:
            dispatch_params = ["e0", "e1", "e2", "e3", "e4", "e5"]

        # Note: Do not pass insts_path to super().__init__ if we want XRT to allocate
        # per-call instruction BO from _generate_dispatch_insts.
        super().__init__(
            xclbin_path=xclbin_path,
            insts_path=None,
            device_index=device_index,
            kernel_name=kernel_name,
            trace_config=trace_config,
            num_host_bos=num_host_bos,
            dispatch_params=list(dispatch_params),
        )

        self.template_insts_path = Path(insts_path)
        self.w_arg_idx = w_arg_idx
        self.chunks_per_expert = chunks_per_expert
        self.chunk_bytes = chunk_bytes
        self.expert_bytes = expert_bytes
        self.expected_w_patches = expected_w_patches
        self.slots = slots
        self.shared_expert_id = shared_expert_id

        if not self.template_insts_path.exists():
            raise FileNotFoundError(
                f"Template insts file not found: {self.template_insts_path}"
            )

        self._template_words = np.fromfile(self.template_insts_path, dtype=np.uint32)
        self._w_patch_entries = self._parse_and_validate_w_patches()

    @classmethod
    def from_words(
        cls,
        xclbin_path: str | Path | None,
        template_words: np.ndarray,
        *,
        w_arg_idx: int = 1,
        chunks_per_expert: int = B1_CHUNKS_PER_EXPERT,
        chunk_bytes: int = B1_W_PER_CHUNK_BYTES,
        expert_bytes: int = GU_BYTES,
        expected_w_patches: int = 119,
        slots: int = 7,
        shared_expert_id: int = SHARED_EXPERT_ID,
        dispatch_params: Sequence[str] | None = None,
    ) -> "PatchedTemplateKernel":
        """Instantiate kernel directly from in-memory uint32 template words."""
        obj = cls.__new__(cls)
        if dispatch_params is None:
            dispatch_params = ["e0", "e1", "e2", "e3", "e4", "e5"]
        super(PatchedTemplateKernel, obj).__init__(
            xclbin_path=xclbin_path,
            insts_path=None,
            device_index=0,
            kernel_name="MLIR_AIE",
            trace_config=None,
            num_host_bos=None,
            dispatch_params=list(dispatch_params),
        )
        obj.template_insts_path = Path("in_memory.bin")
        obj.w_arg_idx = w_arg_idx
        obj.chunks_per_expert = chunks_per_expert
        obj.chunk_bytes = chunk_bytes
        obj.expert_bytes = expert_bytes
        obj.expected_w_patches = expected_w_patches
        obj.slots = slots
        obj.shared_expert_id = shared_expert_id
        obj._template_words = np.asarray(template_words, dtype=np.uint32)
        obj._w_patch_entries = obj._parse_and_validate_w_patches()
        return obj

    def _parse_and_validate_w_patches(self) -> list[tuple[int, int]]:
        """Parse template instructions and extract (word_pos, chunk_idx) for weight patches."""
        if self.slots not in (6, 7):
            raise ValueError(f"slots must be 6 or 7, got {self.slots}")
        valid_counts = {
            self.slots,
            self.slots * self.chunks_per_expert,
        }
        if self.expected_w_patches not in valid_counts:
            raise ValueError(
                "expected_w_patches must equal slots (per-call template) or "
                "slots * chunks_per_expert (per-round template): "
                f"got {self.expected_w_patches}, expected one of {sorted(valid_counts)}"
            )
        patches = find_ddr_patches(self._template_words, target_arg_idx=self.w_arg_idx)
        if len(patches) != self.expected_w_patches:
            raise ValueError(
                f"W patch count mismatch: expected exactly {self.expected_w_patches} W patches "
                f"(arg_idx={self.w_arg_idx}), but found {len(patches)}"
            )

        # Recover chunk index c from template_arg_plus (c = template_arg_plus // chunk_bytes)
        # Note: Do NOT rely on traversal order! Determine c directly from the offset value.
        patch_entries: list[tuple[int, int]] = []
        seen_chunks: set[int] = set()

        for pos, arg_idx, template_arg_plus in patches:
            if template_arg_plus % self.chunk_bytes != 0:
                raise ValueError(
                    f"Template arg_plus {hex(template_arg_plus)} at word {pos} is not "
                    f"a multiple of chunk_bytes {hex(self.chunk_bytes)}"
                )
            c = template_arg_plus // self.chunk_bytes
            if c in seen_chunks:
                raise ValueError(
                    f"Duplicate chunk index {c} recovered from template arg_plus {hex(template_arg_plus)}"
                )
            seen_chunks.add(c)
            patch_entries.append((pos, c))

        expected_chunk_set = (
            {slot * self.chunks_per_expert for slot in range(self.slots)}
            if self.expected_w_patches == self.slots
            else set(range(self.expected_w_patches))
        )
        if seen_chunks != expected_chunk_set:
            missing = expected_chunk_set - seen_chunks
            raise ValueError(
                f"Recovered chunk indices are incomplete! Missing chunks: {sorted(missing)}"
            )

        return patch_entries

    def _generate_dispatch_insts(self, dispatch_scalars: dict | None) -> np.ndarray:
        """Validate dispatch scalar parameters and generate patched instruction stream."""
        dispatch_scalars = dispatch_scalars or {}
        missing = set(self._dispatch_params) - set(dispatch_scalars)
        extra = set(dispatch_scalars) - set(self._dispatch_params)
        if missing or extra:
            raise HostRuntimeError(
                f"dispatch scalar mismatch: missing={missing or None} "
                f"extra={extra or None}; design expects exactly {self._dispatch_params}"
            )

        for key in ["e0", "e1", "e2", "e3", "e4", "e5"]:
            val = dispatch_scalars[key]
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise ValueError(
                    f"Expert index for '{key}' must be an integer, got {type(val).__name__} ({val!r})"
                )
            if not (0 <= int(val) < 128):
                raise ValueError(
                    f"Expert index for '{key}' out of range [0, 128): got {val}"
                )

        expert_slots = [
            int(dispatch_scalars["e0"]),
            int(dispatch_scalars["e1"]),
            int(dispatch_scalars["e2"]),
            int(dispatch_scalars["e3"]),
            int(dispatch_scalars["e4"]),
            int(dispatch_scalars["e5"]),
        ]
        if self.slots == 7:
            expert_slots.append(self.shared_expert_id)

        # Create a fresh copy of template instructions
        patched = self._template_words.copy()

        # Update each weight DDR_PATCH arg_plus
        for pos, c in self._w_patch_entries:
            j = c // self.chunks_per_expert
            l = c % self.chunks_per_expert
            expert_id = expert_slots[j]
            new_arg_plus = expert_id * self.expert_bytes + l * self.chunk_bytes
            patched[pos + 10] = new_arg_plus & 0xFFFFFFFF
            patched[pos + 11] = (new_arg_plus >> 32) & 0xFFFFFFFF

        return patched

    def instructions(self, **dispatch_scalars) -> np.ndarray:
        """Return the instruction words with patched weight offsets."""
        return self._generate_dispatch_insts(dispatch_scalars)
