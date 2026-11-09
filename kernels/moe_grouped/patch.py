# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Patch all slot/chunk DDR_PATCH offsets in a grouped static template."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kernels.moe.patch import (
    TXN_OPC_DDR_PATCH,
    TXN_OPC_WRITE,
    HostRuntimeError,
    NPUKernel,
    find_ddr_patches,
    parse_txn_ops,
)
from .layout import (
    COLS,
    FORMAT_Q4NX,
    ROWS,
    ROWS_WORD_TEMPLATE,
    SLOTS,
    TMAX,
    GroupedShape,
    get_shape,
    stream_plan,
)


def find_rows_words(words: np.ndarray, shape: str | GroupedShape) -> np.ndarray:
    """Return the 32 RTP word-4 positions in a grouped TXN template."""
    resolved = get_shape(shape)
    arr = np.asarray(words, dtype=np.uint32)
    ops = parse_txn_ops(arr)
    first_ddr = next(
        (i for i, (opc, _pos, _sz) in enumerate(ops) if opc == TXN_OPC_DDR_PATCH),
        None,
    )
    if first_ddr is None:
        raise ValueError(f"{resolved.name}: no DDR_PATCH operation found in template")
    hits = [
        i
        for i, (opc, pos, _sz) in enumerate(ops[:first_ddr])
        if opc == TXN_OPC_WRITE and int(arr[pos + 4]) == ROWS_WORD_TEMPLATE
    ]
    expected_count = COLS * ROWS
    if len(hits) != expected_count:
        raise ValueError(
            f"{resolved.name}: {len(hits)} rows-word writes before first DDR_PATCH, "
            f"expected {expected_count}"
        )
    cores: set[tuple[int, int]] = set()
    positions: list[int] = []
    for i in hits:
        _opc, pos, size = ops[i]
        if (
            size != 6
            or int(arr[pos + 5]) != 24
            or int(arr[pos + 3]) != 0
            or not (6 <= i < len(ops) - 1)
        ):
            raise ValueError(
                f"rows-word write at word {pos} has an unexpected form or place"
            )
        addr = int(arr[pos + 2])
        col = (addr >> 25) & 0x7F
        row = (addr >> 20) & 0x1F
        if not (0 <= col < COLS and 2 <= row <= ROWS + 1):
            raise ValueError(
                f"rows-word write at word {pos} targets out-of-range core ({col}, {row})"
            )
        want = (
            resolved.k_blocks,
            resolved.rounds_by_col[col],
            resolved.mode,
            SLOTS,
        )
        for j in range(4):
            opc, p, s = ops[i - 4 + j]
            if (
                opc != TXN_OPC_WRITE
                or s != 6
                or int(arr[p + 5]) != 24
                or int(arr[p + 2]) != addr - 16 + 4 * j
                or int(arr[p + 3]) != 0
                or int(arr[p + 4]) != want[j]
            ):
                raise ValueError(
                    f"RTP word {j} before the rows word at {pos} (core {col},{row}) "
                    f"is invalid (expected {want[j]})"
                )
        # words 5-6 (balanced extra round count, its slot = the column) are written right before word 0
        for j, want_word in enumerate((1 if resolved.balanced else 0, col)):
            opc, p, s = ops[i - 6 + j]
            if (
                opc != TXN_OPC_WRITE
                or s != 6
                or int(arr[p + 5]) != 24
                or int(arr[p + 2]) != addr + 4 + 4 * j
                or int(arr[p + 3]) != 0
                or int(arr[p + 4]) != want_word
            ):
                raise ValueError(
                    f"RTP word {5 + j} before word 0 of the rows word at {pos} (core {col},{row}) "
                    f"is invalid (expected {want_word})"
                )
        opc, p, s = ops[i + 1]
        if (
            opc != TXN_OPC_WRITE
            or s != 6
            or int(arr[p + 5]) != 24
            or int(arr[p + 3]) != 0
            or (int(arr[p + 2]) >> 20) != (addr >> 20)
            or int(arr[p + 4]) != 1
        ):
            raise ValueError(
                f"no barrier write of core {col},{row} after the rows word at {pos}"
            )
        cores.add((col, row))
        positions.append(pos + 4)
    expected_cores = {(c, r) for c in range(COLS) for r in range(2, ROWS + 2)}
    if cores != expected_cores:
        raise ValueError(
            f"rows words reach {len(cores)} distinct cores, expected {expected_count}"
        )
    return np.array(positions, dtype=np.int64)


def pack_rows(rows: Sequence[int]) -> int:
    """Validate 8 slot row counts in 0..TMAX and pack into a 32-bit integer."""
    items = list(rows)
    if len(items) != SLOTS:
        raise ValueError(f"slot rows must have {SLOTS} values, got {len(items)}")
    packed = 0
    for slot, value in enumerate(items):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"r{slot} must be an integer row count, got {value!r}")
        count = int(value)
        if not 0 <= count <= TMAX:
            raise ValueError(f"r{slot}={count} outside 0..{TMAX}")
        packed |= count << (4 * slot)
    return packed


class GroupedPatchedTemplateKernel(NPUKernel):
    """NPU callable whose dynamic values are eight expert addresses and optional B1 rows."""

    def __init__(
        self,
        xclbin_path: str | Path,
        insts_path: str | Path,
        shape: str | GroupedShape,
        *,
        w_arg_idx: int = 1,
        dispatch_params: Sequence[str] | None = None,
        num_host_bos: int = 3,
    ) -> None:
        self.shape = get_shape(shape)
        row_names = [f"r{i}" for i in range(SLOTS)]
        if dispatch_params is None:
            expert_params = [f"e{i}" for i in range(SLOTS)]
        else:
            expert_params = [name for name in dispatch_params if name not in row_names]
        row_params = row_names if self.shape.mode == FORMAT_Q4NX else []
        params = expert_params + row_params
        super().__init__(
            xclbin_path=xclbin_path, insts_path=None, kernel_name="MLIR_AIE",
            num_host_bos=num_host_bos, dispatch_params=params,
        )
        self._expert_params = tuple(expert_params)
        self._row_params = tuple(row_params)
        self.w_arg_idx = w_arg_idx
        self.template_insts_path = Path(insts_path)
        self._template_words = np.fromfile(self.template_insts_path, dtype=np.uint32)
        self._entries = self._parse_entries()
        self._rows_pos = find_rows_words(self._template_words, self.shape)

    @classmethod
    def from_words(
        cls,
        template_words: np.ndarray,
        shape: str | GroupedShape,
        *,
        w_arg_idx: int = 1,
    ) -> "GroupedPatchedTemplateKernel":
        obj = cls.__new__(cls)
        obj.shape = get_shape(shape)
        expert_params = [f"e{i}" for i in range(SLOTS)]
        row_params = (
            [f"r{i}" for i in range(SLOTS)]
            if obj.shape.mode == FORMAT_Q4NX
            else []
        )
        params = expert_params + row_params
        super(GroupedPatchedTemplateKernel, obj).__init__(
            xclbin_path=None, insts_path=None, kernel_name="MLIR_AIE",
            num_host_bos=3, dispatch_params=params,
        )
        obj._expert_params = tuple(expert_params)
        obj._row_params = tuple(row_params)
        obj.w_arg_idx = w_arg_idx
        obj.template_insts_path = Path("in_memory.bin")
        obj._template_words = np.asarray(template_words, dtype=np.uint32)
        obj._entries = obj._parse_entries()
        obj._rows_pos = find_rows_words(obj._template_words, obj.shape)
        return obj

    def _parse_entries(self) -> tuple[tuple[int, int, int], ...]:
        patches = find_ddr_patches(self._template_words, target_arg_idx=self.w_arg_idx)
        if len(patches) != self.shape.patch_count:
            raise ValueError(
                f"{self.shape.name} W patch count mismatch: expected "
                f"{self.shape.patch_count}, found {len(patches)}"
            )
        # The W patches appear in stream_plan order (a balanced shape's extra chunks come last). Template offsets
        # repeat for every slot, so slot identity comes from that order.
        entries: list[tuple[int, int, int]] = []
        for (pos, _arg, offset), rec in zip(patches, stream_plan(self.shape)):
            slot, expected_chunk = rec["slot"], rec["chunk"]
            if offset % self.shape.chunk_bytes:
                raise ValueError(f"unaligned W patch offset {offset} at word {pos}")
            chunk = offset // self.shape.chunk_bytes
            if chunk != expected_chunk:
                raise ValueError(
                    f"non-canonical patch order at word {pos}: slot {slot} "
                    f"expected chunk {expected_chunk}, got {chunk}"
                )
            entries.append((pos, slot, chunk))
        return tuple(entries)

    def _generate_dispatch_insts(self, dispatch_scalars: dict | None) -> np.ndarray:
        values = dispatch_scalars or {}
        expert_set = set(self._expert_params)
        full_set = set(self._dispatch_params)
        given = set(values)
        if given != expert_set and given != full_set:
            raise HostRuntimeError(
                f"dispatch scalar mismatch: expected {sorted(full_set)}, got {sorted(given)}"
            )
        experts: list[int] = []
        for name in self._expert_params:
            value = values[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer expert ID")
            expert = int(value)
            if not 0 <= expert <= self.shape.max_expert:
                raise ValueError(f"{name}={expert} outside 0..{self.shape.max_expert}")
            experts.append(expert)
        rows_word: int | None = None
        if self._row_params and given == full_set:
            rows_word = pack_rows([values[name] for name in self._row_params])
        patched = self._template_words.copy()
        for pos, slot, chunk in self._entries:
            offset = experts[slot] * self.shape.expert_bytes + chunk * self.shape.chunk_bytes
            patched[pos + 10] = offset & 0xFFFFFFFF
            patched[pos + 11] = (offset >> 32) & 0xFFFFFFFF
        if rows_word is not None:
            patched[self._rows_pos] = np.uint32(rows_word)
        return patched

    def instructions(self, **dispatch_scalars: int) -> np.ndarray:
        return self._generate_dispatch_insts(dispatch_scalars)
