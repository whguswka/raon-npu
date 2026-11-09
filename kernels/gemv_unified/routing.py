# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Static stream topology and AIE2p switchbox budget accounting.

The device builder consumes the same topology records that the host-only routing
gate checks.  This keeps the gate tied to the design instead of maintaining a
second hand-written list of W flows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from kernels.gemv_unified.shapes import COLS, ROWS

# Whether the memory-tile / core-tile W runtime tasks issue a completion token.
# The design (``design.py``) and the minimal experiment pass this to
# ``configure_ping_pong_tasks``.  ``False`` since fix3: a token from a non-shim
# tile makes ``aie-generate-column-control-overlay`` add a TileControl -> shim
# South packet flow (see ``overlay_packet_paths``).
W_TASKS_ISSUE_TOKEN = False


@dataclass(frozen=True, order=True)
class StreamTile:
    col: int
    row: int


@dataclass(frozen=True)
class StreamPath:
    name: str
    kind: str
    source: StreamTile
    destinations: tuple[StreamTile, ...]


@dataclass(frozen=True, order=True)
class PortUse:
    col: int
    row: int
    direction: str  # "source" or "dest"
    bundle: str


def unified_stream_topology(cols: int = COLS) -> tuple[StreamPath, ...]:
    """Return every data stream emitted by the design for ``cols`` columns.

    ``x`` is one multicast stream per column.  A joined ``y`` ObjectFifo is
    represented by its four core-to-memory streams and one memory-to-shim
    stream.  W contains the explicit Flow objects created in ``design.py``.
    """

    paths: list[StreamPath] = []
    for col in range(cols):
        shim = StreamTile(col, 0)
        mem = StreamTile(col, 1)
        cores = tuple(StreamTile(col, row + 2) for row in range(ROWS))
        paths.append(StreamPath(f"x{col}", "x", shim, cores))
        paths.append(StreamPath(f"w_in_{col}", "w_in", shim, (mem,)))
        paths.extend(
            StreamPath(f"w_out_{col}_{row}", "w_out", mem, (core,))
            for row, core in enumerate(cores)
        )
        paths.extend(
            StreamPath(f"y{col}_{row}", "y_join_in", core, (mem,))
            for row, core in enumerate(cores)
        )
        paths.append(StreamPath(f"y{col}", "y_join_out", mem, (shim,)))
    return tuple(paths)


def token_issuing_tiles(
    w_tasks_issue_token: bool = W_TASKS_ISSUE_TOKEN,
    cols: int = COLS,
) -> tuple[StreamTile, ...]:
    """Non-shim tiles whose runtime-sequence tasks issue a completion token.

    Shim tasks (x fill, W-in fill, Y drain) are not listed: their tokens are
    consumed by the shim itself and need no overlay path.  With
    ``w_tasks_issue_token`` (the pre-fix design) the memory tile and the four
    cores of every column issue tokens for their W tasks.
    """

    if not w_tasks_issue_token:
        return ()
    return tuple(StreamTile(col, row) for col in range(cols) for row in range(1, ROWS + 2))


def overlay_packet_paths(
    token_tiles: tuple[StreamTile, ...],
) -> tuple[StreamPath, ...]:
    """Column-control overlay packet flows (``route-shim-to-tct=shim-only``).

    ``aie-generate-column-control-overlay`` (``AIEPasses.td:653``) routes, for every
    tile a runtime sequence task issues a token from, one packet flow
    TileControl -> shim South (``keep_pkt_header``, ``priority_route``): a DOWN
    path through the column.  Names start with ``tct_``.
    """

    return tuple(
        StreamPath(f"tct_{tile.col}_{tile.row}", "tct_packet", tile, (StreamTile(tile.col, 0),))
        for tile in token_tiles
        if tile.row != 0
    )


def _packet_ports(path: StreamPath) -> set[PortUse]:
    """Switchbox ports a TileControl -> shim South packet flow occupies."""

    src = path.source
    ports = {
        PortUse(src.col, src.row, "source", "TileControl"),
        PortUse(src.col, src.row, "dest", "South"),
    }
    for row in range(src.row - 1, -1, -1):
        ports.add(PortUse(src.col, row, "source", "North"))
        ports.add(PortUse(src.col, row, "dest", "South"))
    return ports


def _add_vertical_path(
    demand: Counter[PortUse], source: StreamTile, destinations: tuple[StreamTile, ...]
) -> None:
    if not destinations:
        raise ValueError("a stream must have at least one destination")
    if any(dst.col != source.col or dst.row == source.row for dst in destinations):
        raise ValueError("unified GEMV streams must stay within one column")
    signs = {1 if dst.row > source.row else -1 for dst in destinations}
    if len(signs) != 1:
        raise ValueError("a multicast stream cannot cross both vertical directions")

    step = signs.pop()
    target_rows = {dst.row for dst in destinations}
    last_row = max(target_rows) if step > 0 else min(target_rows)
    outward = "North" if step > 0 else "South"
    inward = "South" if step > 0 else "North"

    demand[PortUse(source.col, source.row, "source", "DMA")] += 1
    demand[PortUse(source.col, source.row, "dest", outward)] += 1
    row = source.row + step
    while True:
        demand[PortUse(source.col, row, "source", inward)] += 1
        if row in target_rows:
            demand[PortUse(source.col, row, "dest", "DMA")] += 1
        if row == last_row:
            break
        demand[PortUse(source.col, row, "dest", outward)] += 1
        row += step


def switchbox_demand(
    paths: tuple[StreamPath, ...] | None = None,
    token_tiles: tuple[StreamTile, ...] | None = None,
    cols: int = COLS,
) -> dict[PortUse, int]:
    """Count occupied source/destination ports for the supplied stream tree.

    Circuit-switched data streams take one port each.  Overlay packet flows
    (``kind == "tct_packet"``, derived from ``token_tiles``, default
    ``token_issuing_tiles()``) are packet-switched toward one destination and
    share ports: each distinct port they cross is counted once for all of them
    (assumption/추정, consistent with the observed ``y join 4 + 1`` budget).
    """

    demand: Counter[PortUse] = Counter()
    paths = unified_stream_topology(cols) if paths is None else paths
    if token_tiles is None:
        token_tiles = token_issuing_tiles(cols=cols)
    paths = paths + overlay_packet_paths(token_tiles)
    packet_ports: set[PortUse] = set()
    for path in paths:
        if path.kind == "tct_packet":
            packet_ports |= _packet_ports(path)
        else:
            _add_vertical_path(demand, path.source, path.destinations)
    for use in packet_ports:
        demand[use] += 1
    return dict(demand)


def target_port_limit(use: PortUse) -> int:
    """Return the NPU2/AIE2p port limit mirrored from ``AIETargetModel.cpp``.

    NPU2 inherits AIE2 switchbox counts.  Shim DMA endpoints are in the shim
    mux, whose source and destination DMA limits are both two.
    """

    if use.bundle == "TileControl":
        return 1
    if use.row == 0:
        if use.bundle == "DMA":
            return 2
        shim = {
            "source": {"North": 4, "South": 8},
            "dest": {"North": 6, "South": 6},
        }
        return shim[use.direction].get(use.bundle, 0)
    if use.row == 1:
        mem = {
            "source": {"DMA": 6, "North": 4, "South": 6},
            "dest": {"DMA": 6, "North": 6, "South": 4},
        }
        return mem[use.direction].get(use.bundle, 0)
    core = {
        "source": {"DMA": 2, "North": 4, "South": 6},
        "dest": {"DMA": 2, "North": 6, "South": 4},
    }
    return core[use.direction].get(use.bundle, 0)


def routing_overages(
    paths: tuple[StreamPath, ...] | None = None,
    token_tiles: tuple[StreamTile, ...] | None = None,
    cols: int = COLS,
) -> tuple[tuple[PortUse, int, int], ...]:
    """Return ``(port, demand, limit)`` entries that exceed the target model."""

    return tuple(
        (use, count, target_port_limit(use))
        for use, count in sorted(switchbox_demand(paths, token_tiles, cols=cols).items())
        if count > target_port_limit(use)
    )
