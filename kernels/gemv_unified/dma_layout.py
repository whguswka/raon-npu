# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python DMA access-pattern specifications for unified GEMV.

The IRON builders and the host-only BD-limit tests consume these same
``DmaTapSpec`` objects.  Keeping the arithmetic here prevents a test from
silently validating a second, independently reconstructed set of dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .shapes import (
    COLS,
    FORMAT_BFP16,
    MAX_K,
    M_PER_CHUNK,
    M_TILE,
    Q4_BLOCK_BYTES,
    ROWS,
    W_BLOCK_BYTES,
    UnifiedShape,
    x_slot_offset,
)

W_ELEMS_PER_BLOCK = W_BLOCK_BYTES // 2
DMA_SEGMENT_BYTES = 1024
SHIM_WRAP_MAX = 1023
SHIM_STEP_MAX_WORDS = 1 << 20
SHIM_ITER_MAX = 64


@dataclass(frozen=True)
class DmaTapSpec:
    """One outermost-first TensorAccessPattern plus its BD context."""

    name: str
    tensor_dims: tuple[int, ...]
    offset: int
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    tile: str
    element_bits: int
    preserve_dimensions: bool = False


def _linear(
    name: str,
    tensor_dims: tuple[int, ...],
    offset: int,
    length: int,
    *,
    tile: str,
    element_bits: int,
    shim_form: bool = False,
) -> DmaTapSpec:
    sizes = (1, 1, 1, length) if shim_form else (length,)
    strides = (1, 1, 1, 1) if shim_form else (1,)
    return DmaTapSpec(
        name, tensor_dims, offset, sizes, strides, tile, element_bits
    )


def _row_receive(
    name: str,
    *,
    rows: int,
    storage_per_row: int,
    active_per_row: int,
    element_bits: int,
) -> DmaTapSpec:
    """Return a linear full-row transfer or a hardware-safe gapped-row ND one."""
    tensor_dims = (rows, storage_per_row)
    if active_per_row == storage_per_row:
        return _linear(
            name,
            tensor_dims,
            0,
            rows * active_per_row,
            tile="mem",
            element_bits=element_bits,
        )
    segment = DMA_SEGMENT_BYTES * 8 // element_bits
    if active_per_row % segment:
        raise ValueError(
            f"{name}: active row {active_per_row} is not divisible by {segment}"
        )
    return DmaTapSpec(
        name=name,
        tensor_dims=tensor_dims,
        offset=0,
        sizes=(rows, active_per_row // segment, segment),
        strides=(storage_per_row, segment, 1),
        tile="mem",
        element_bits=element_bits,
        # iron.dataflow.tile_dma.Bd normally coalesces adjacent contiguous
        # dimensions.  That would recreate the illegal [4, 1280] q4 BD.
        preserve_dimensions=True,
    )


def gemv_dma_tap_specs(
    shape: UnifiedShape, cols: int = COLS
) -> dict[str, tuple[DmaTapSpec, ...]]:
    """Return every explicit shim/memory/core TAP emitted by ``gemv_unified``."""
    active = (
        W_ELEMS_PER_BLOCK
        if shape.format == FORMAT_BFP16
        else Q4_BLOCK_BYTES // 2
    )
    x = tuple(
        _linear(
            f"x[{col}]",
            (1, shape.x_slots * MAX_K),
            x_slot_offset(shape, col, MAX_K),
            MAX_K,
            tile="shim",
            element_bits=16,
            shim_form=True,
        )
        for col in range(cols)
    )
    w_host = tuple(
        _linear(
            f"w_host[{chunk}]",
            (1, shape.weight_words),
            chunk * shape.w_chunk_words,
            shape.w_chunk_words,
            tile="shim",
            element_bits=32,
            shim_form=True,
        )
        for chunk in range(shape.chunks)
    )
    y = tuple(
        _linear(
            f"y[{chunk}]",
            (1, shape.M),
            chunk * M_PER_CHUNK,
            M_PER_CHUNK,
            tile="shim",
            element_bits=16,
            shim_form=True,
        )
        for chunk in range(shape.chunks)
    )
    mem_in = (
        _row_receive(
            "mem_receive",
            rows=ROWS,
            storage_per_row=W_ELEMS_PER_BLOCK,
            active_per_row=active,
            element_bits=16,
        ),
    )
    mem_out = tuple(
        _linear(
            f"mem_send[{row}]",
            (ROWS * W_ELEMS_PER_BLOCK,),
            row * W_ELEMS_PER_BLOCK,
            active,
            tile="mem",
            element_bits=16,
        )
        for row in range(ROWS)
    )
    core_in = (
        _linear(
            "core_receive",
            (W_ELEMS_PER_BLOCK,),
            0,
            active,
            tile="core",
            element_bits=16,
        ),
    )
    # ObjectFifo lowering creates these linear core/shim BDs from the buffer
    # types.  design.py derives those type lengths from these same specs.
    x_core = (
        _linear(
            "x_core_receive", (MAX_K,), 0, MAX_K,
            tile="core", element_bits=16,
        ),
    )
    y_core = (
        _linear(
            "y_core_send", (M_TILE,), 0, M_TILE,
            tile="core", element_bits=16,
        ),
    )
    y_shim = (
        _linear(
            "y_shim_receive", (M_PER_CHUNK,), 0, M_PER_CHUNK,
            tile="shim", element_bits=16,
        ),
    )
    return {
        "x": x,
        "w_host": w_host,
        "y": y,
        "mem_in": mem_in,
        "mem_out": mem_out,
        "core_in": core_in,
        "x_core": x_core,
        "y_core": y_core,
        "y_shim": y_shim,
    }


def _split_contiguous_length(length: int) -> tuple[int, int]:
    """Split a contiguous shim run into two legal 10-bit wrap dimensions."""
    for inner in range(min(length, SHIM_WRAP_MAX), 0, -1):
        if length % inner == 0 and length // inner <= SHIM_WRAP_MAX:
            return length // inner, inner
    raise ValueError(f"cannot represent contiguous shim length {length} in two wrap dimensions")


def per_call_shim_tap_specs(
    shape: UnifiedShape, cols: int = COLS
) -> dict[int, dict[str, tuple[DmaTapSpec, ...]]]:
    """Return minimal per-column shim TAPs for the per-call sequence.

    A regular column walk is one rank-4 BD: rounds are the iteration dimension
    and the contiguous chunk is split across the two 10-bit wrap dimensions.
    If the outer W step exceeds the shim's 20-bit iteration-step field, the W
    walk is conservatively split into one linear BD per chunk.
    """
    active = shape.resolve_active_cols(cols)
    result: dict[int, dict[str, tuple[DmaTapSpec, ...]]] = {}
    for col, rounds in enumerate(shape.rounds_by_col(cols)):
        if col >= active or rounds == 0:
            continue
        first_chunk = col * shape.chunks_per_col if shape.chunks_per_col is not None else col
        chunk_step = 1 if shape.chunks_per_col is not None else active
        entries: dict[str, tuple[DmaTapSpec, ...]] = {}
        for kind, total, chunk_len, bits in (
            ("w", shape.weight_words, shape.w_chunk_words, 32),
            ("y", shape.M, M_PER_CHUNK, 16),
        ):
            outer_step = chunk_step * chunk_len
            step_words = outer_step * bits // 32
            if rounds > 1 and step_words > SHIM_STEP_MAX_WORDS:
                taps = tuple(
                    _linear(
                        f"{kind}_per_call[{col}:{rnd}]", (1, total),
                        (first_chunk + rnd * chunk_step) * chunk_len, chunk_len,
                        tile="shim", element_bits=bits, shim_form=True,
                    )
                    for rnd in range(rounds)
                )
            else:
                outer, inner = _split_contiguous_length(chunk_len)
                taps = tuple(
                    DmaTapSpec(
                        name=f"{kind}_per_call[{col}:{start}]",
                        tensor_dims=(1, total),
                        offset=(first_chunk + start * chunk_step) * chunk_len,
                        sizes=(min(SHIM_ITER_MAX, rounds - start), 1, outer, inner),
                        strides=(outer_step, 0, inner, 1),
                        tile="shim",
                        element_bits=bits,
                        preserve_dimensions=True,
                    )
                    for start in range(0, rounds, SHIM_ITER_MAX)
                )
            entries[kind] = taps
        result[col] = entries
    return result


def short_dma_tap_specs(transfer_words_per_row: int) -> dict[str, tuple[DmaTapSpec, ...]]:
    """Return every explicit TAP emitted by the short-DMA experiment."""
    full = W_BLOCK_BYTES // 4
    if transfer_words_per_row not in (full, Q4_BLOCK_BYTES // 4):
        raise ValueError(f"unsupported short-DMA row length {transfer_words_per_row}")
    mem_in = (
        _row_receive(
            "exp_mem_receive",
            rows=ROWS,
            storage_per_row=full,
            active_per_row=transfer_words_per_row,
            element_bits=32,
        ),
    )
    mem_out = tuple(
        _linear(
            f"exp_mem_send[{row}]",
            (ROWS * full,),
            row * full,
            transfer_words_per_row,
            tile="mem",
            element_bits=32,
        )
        for row in range(ROWS)
    )
    core = (
        _linear(
            "exp_core_receive",
            (full,),
            0,
            transfer_words_per_row,
            tile="core",
            element_bits=32,
        ),
    )
    host_w = (
        _linear(
            "exp_host_w",
            (1, ROWS * transfer_words_per_row),
            0,
            ROWS * transfer_words_per_row,
            tile="shim",
            element_bits=32,
            shim_form=True,
        ),
    )
    host_y = (
        _linear(
            "exp_host_y",
            (1, ROWS * full),
            0,
            ROWS * full,
            tile="shim",
            element_bits=32,
            shim_form=True,
        ),
    )
    y_core = (
        _linear(
            "exp_y_core_send", (full,), 0, full,
            tile="core", element_bits=32,
        ),
    )
    y_shim = (
        _linear(
            "exp_y_shim_receive", (ROWS * full,), 0, ROWS * full,
            tile="shim", element_bits=32,
        ),
    )
    return {
        "mem_in": mem_in,
        "mem_out": mem_out,
        "core_in": core,
        "host_w": host_w,
        "host_y": host_y,
        "y_core": y_core,
        "y_shim": y_shim,
    }
