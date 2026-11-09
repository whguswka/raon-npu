# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only NPU2 stream-switch routing budget gate.

Limits mirror ``AIETargetModel.cpp:1019-1165``; NPU2 inherits from the AIE2
model at ``AIETargetModel.h:1210-1256``.  Shim mux DMA source/destination limits
come from ``AIETargetModel.cpp:1169-1205``.
"""

from __future__ import annotations

from collections import Counter

from kernels.gemv_unified.routing import (
    W_TASKS_ISSUE_TOKEN,
    PortUse,
    StreamPath,
    StreamTile,
    routing_overages,
    switchbox_demand,
    overlay_packet_paths,
    target_port_limit,
    token_issuing_tiles,
    unified_stream_topology,
)
from kernels.gemv_unified.shapes import COLS, ROWS


def test_design_stream_list_is_complete_and_column_local() -> None:
    paths = unified_stream_topology()
    kinds = Counter(path.kind for path in paths)
    assert kinds == {
        "x": COLS,
        "w_in": COLS,
        "w_out": COLS * ROWS,
        "y_join_in": COLS * ROWS,
        "y_join_out": COLS,
    }
    assert len({path.name for path in paths}) == len(paths) == 88
    assert all(
        destination.col == path.source.col
        for path in paths
        for destination in path.destinations
    )


def test_design_streams_fit_every_npu2_switchbox_port_limit() -> None:
    assert routing_overages() == ()
    demand = switchbox_demand()

    # Per-column bottlenecks after y join is fixed to the W Memory Tile.
    expected = {
        PortUse(0, 0, "source", "DMA"): 2,  # x + W
        PortUse(0, 0, "dest", "DMA"): 1,  # joined y
        PortUse(0, 1, "source", "DMA"): 5,  # W rows + joined y
        PortUse(0, 1, "dest", "DMA"): 5,  # W in + four y rows
        PortUse(0, 1, "source", "North"): 4,  # four y rows, exact limit
        PortUse(0, 1, "dest", "North"): 5,  # x + four W rows
        PortUse(0, 2, "source", "South"): 5,  # x + four W rows
        PortUse(0, 2, "dest", "South"): 4,  # four y rows, exact limit
        PortUse(0, 2, "dest", "DMA"): 2,  # x + local W, exact limit
    }
    for use, count in expected.items():
        assert demand[use] == count
        assert count <= target_port_limit(use)


def test_gate_detects_one_extra_memory_to_core_stream() -> None:
    paths = unified_stream_topology() + (
        StreamPath(
            "over_budget",
            "test",
            StreamTile(0, 1),
            (StreamTile(0, 2),),
        ),
    )
    overages = routing_overages(paths)
    assert (PortUse(0, 2, "dest", "DMA"), 3, 2) in overages


def test_overlay_adds_one_downward_packet_flow_per_token_tile() -> None:
    """``route-shim-to-tct=shim-only``: TileControl -> shim South per token tile."""
    tiles = token_issuing_tiles(True)
    assert len(tiles) == COLS * (ROWS + 1)  # memory tile + 4 cores per column
    paths = overlay_packet_paths(tiles)
    assert len(paths) == len(tiles)
    assert all(p.kind == "tct_packet" and p.destinations[0].row == 0 for p in paths)
    assert all(p.source.row > 0 for p in paths)
    assert overlay_packet_paths(()) == ()


def test_pre_fix_token_issuing_w_tasks_exceed_downward_port_limit() -> None:
    """Old design (mem/core W tasks issue tokens): y join 4 + token path 1 = 5 > 4."""
    overages = routing_overages(None, token_issuing_tiles(True))
    assert overages, "pre-fix design must fail the routing budget"
    for col in range(COLS):
        # memory tile North source (cores -> mem): 4 y rows + packet path
        assert (PortUse(col, 1, "source", "North"), 5, 4) in overages
        # bottom core South dest (core -> mem): 4 y rows + packet path
        assert (PortUse(col, 2, "dest", "South"), 5, 4) in overages
    assert len(overages) == 2 * COLS


def test_fixed_design_issues_no_non_shim_tokens_and_fits() -> None:
    """Post-fix: tokens only from shim tasks, so no overlay flows and no overage."""
    assert W_TASKS_ISSUE_TOKEN is False
    assert token_issuing_tiles() == ()
    assert routing_overages() == ()
    demand_default = switchbox_demand()
    assert demand_default == switchbox_demand(None, ())
    assert demand_default[PortUse(0, 1, "source", "North")] == 4
    assert demand_default[PortUse(0, 2, "dest", "South")] == 4
