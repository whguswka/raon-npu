# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""CPU host-host lock-order unit tests for the short-DMA --variant sweep."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load exp_short_dma as a sibling module without importing the full bench tree.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gemv_unified_exp_short_dma", HERE / "exp_short_dma.py"
)
assert _spec is not None and _spec.loader is not None
_exp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp_mod)

VARIANTS = _exp_mod.VARIANTS
simulate_short_dma_variant = _exp_mod.simulate_short_dma_variant


def test_variant_list_is_complete_and_unique() -> None:
    names = [v.name for v in VARIANTS]
    assert len(names) == len(set(names))
    assert "V0" in names
    assert "V1S1" in names
    assert "V5S2" in names


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_variant_lock_order_and_data_integrity(variant: _exp_mod.VariantSpec) -> None:
    if variant.alt_sizes:
        sim = _exp_mod.simulate_short_dma_ping_pong(iters=200)
        assert sim["sentinel_preserved"] is True
        return

    sim = simulate_short_dma_variant(variant, calls=5)
    assert sim["sentinel_preserved"] is True
    assert sim["blocks"] == variant.blocks

    def expected_task_count(chain: bool, blocks: int) -> int:
        return 1 if (chain and blocks % 2 == 0) else blocks

    assert sim["mem_tasks"] == expected_task_count(variant.mem_chain, variant.blocks)
    assert sim["core_tasks_per_row"] == expected_task_count(
        variant.core_chain, variant.blocks
    )
    assert sim["shim_tasks"] == expected_task_count(variant.shim_chain, variant.blocks)
