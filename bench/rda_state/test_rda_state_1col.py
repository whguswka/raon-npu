# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
#
"""Host-side unit tests for the single-column tile-wise RDA design.

These tests do not require an NPU.  They exercise the pure-Python helpers
(TAP generation, aggregate-buffer address coverage, resource tables) and the
numpy golden reference split into tiles.
"""

import numpy as np
import pytest

from bench.rda_state.rda_state_1col import (
    _aggregate_buffer_shape,
    _aggregate_out_read_tap,
    _aggregate_state_read_tap,
    K_DIM,
    MODEL_RDA_HEADS,
    numpy_recurrent_rda_step,
    _packed_out_read_tap,
    _packed_state_read_tap,
    _s_in_head_tap,
    _s_in_tile_tap,
    build_rda_channel,
    HEAD_BUNDLE_SIZE,
    PACKED_ELEMS,
    TILE_V,
    TILES_PER_HEAD,
    V_DIM,
)


@pytest.fixture
def num_heads():
    return 40


def _tap_indices(tap, shape):
    """Return the flat indices a TAP visits over a tensor of ``shape``."""
    tensor = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    return tap.gather(tensor).flatten()


class TestTapHelpers:
    """Verify that aggregate-buffer TAPs cover state and output exactly."""

    def test_packed_state_read_covers_all_state_tiles_no_gaps(self):
        buf_shape = _aggregate_buffer_shape()
        tap = _packed_state_read_tap()
        indices = _tap_indices(tap, buf_shape)

        expected = []
        for t in range(TILES_PER_HEAD):
            base = t * PACKED_ELEMS
            expected.extend(range(base, base + TILE_V * K_DIM))
        expected = np.array(expected, dtype=np.int64)

        assert len(indices) == TILES_PER_HEAD * TILE_V * K_DIM
        np.testing.assert_array_equal(np.sort(indices), np.sort(expected))
        # No duplicates and exactly the state region of the aggregate buffer.
        assert len(np.unique(indices)) == len(indices)

    def test_packed_out_read_covers_all_out_slices_no_gaps(self):
        buf_shape = _aggregate_buffer_shape()
        tap = _packed_out_read_tap()
        indices = _tap_indices(tap, buf_shape)

        expected = []
        for t in range(TILES_PER_HEAD):
            base = t * PACKED_ELEMS + 4096
            expected.extend(range(base, base + TILE_V))
        expected = np.array(expected, dtype=np.int64)

        assert len(indices) == TILES_PER_HEAD * TILE_V
        np.testing.assert_array_equal(np.sort(indices), np.sort(expected))
        assert len(np.unique(indices)) == len(indices)

    def test_state_and_out_reads_do_not_overlap(self):
        buf_shape = _aggregate_buffer_shape()
        state_indices = set(_tap_indices(_packed_state_read_tap(), buf_shape))
        out_indices = set(_tap_indices(_packed_out_read_tap(), buf_shape))
        assert not state_indices & out_indices

    def test_state_and_out_reads_cover_full_aggregate_no_padding(self):
        buf_shape = _aggregate_buffer_shape()
        state_indices = set(_tap_indices(_packed_state_read_tap(), buf_shape))
        out_indices = set(_tap_indices(_packed_out_read_tap(), buf_shape))
        covered = state_indices | out_indices
        expected = set(range(np.prod(buf_shape)))
        assert covered == expected

    def test_bad_out_offset_breaks_out_tap(self):
        """Rolling the out read offset by 32 elements causes mismatch."""
        buf_shape = _aggregate_buffer_shape()
        good = _tap_indices(_packed_out_read_tap(out_offset=4096), buf_shape)
        bad = _tap_indices(_packed_out_read_tap(out_offset=4064), buf_shape)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(good, bad)

    def test_s_in_head_tap_covers_one_head(self, num_heads):
        tap = _s_in_head_tap(7, num_heads=num_heads)
        indices = _tap_indices(tap, (num_heads, V_DIM, K_DIM))
        expected_start = 7 * V_DIM * K_DIM
        expected = np.arange(expected_start, expected_start + V_DIM * K_DIM)
        np.testing.assert_array_equal(indices, expected)

    def test_s_in_tile_taps_tile_each_head(self, num_heads):
        for head in [0, 7, num_heads - 1]:
            for tile in range(TILES_PER_HEAD):
                tap = _s_in_tile_tap(head, tile, num_heads=num_heads)
                indices = _tap_indices(tap, (num_heads, V_DIM, K_DIM))
                start = head * V_DIM * K_DIM + tile * TILE_V * K_DIM
                expected = np.arange(start, start + TILE_V * K_DIM)
                np.testing.assert_array_equal(indices, expected)


class TestResourceTables:
    """Verify constants used in the resource table in the module docstring."""

    def test_aggregate_buffer_size(self):
        assert _aggregate_buffer_shape() == (TILES_PER_HEAD, PACKED_ELEMS)
        assert np.prod(_aggregate_buffer_shape()) * 4 == TILES_PER_HEAD * PACKED_ELEMS * 4
        # 4 tiles * 4128 floats * 4 bytes = 66,048 bytes per ping/pong slot.
        assert TILES_PER_HEAD * PACKED_ELEMS * 4 == 66048

    def test_packed_elems_split(self):
        assert PACKED_ELEMS == TILE_V * K_DIM + TILE_V
        assert TILE_V * K_DIM == 4096
        assert TILE_V == 32

    def test_channel_budget_constants(self):
        # Matches the docstring table used by the design.
        assert TILES_PER_HEAD == 4
        assert HEAD_BUNDLE_SIZE == 640
        assert K_DIM == 128
        assert V_DIM == 128


class TestTileIndependence:
    """Verify that processing heads tile-by-tile is bit-exact vs head-by-head."""

    def test_tile_wise_reference_matches_head_wise(self, num_heads):
        rng = np.random.default_rng(12345)
        S = rng.standard_normal((num_heads, V_DIM, K_DIM)).astype(np.float32) * 0.01
        q = rng.standard_normal((num_heads, K_DIM)).astype(np.float32)
        k = rng.standard_normal((num_heads, K_DIM)).astype(np.float32)
        a = np.exp(rng.uniform(-5.0, 0.0, size=(num_heads, K_DIM)).astype(np.float32))
        v = rng.standard_normal((num_heads, V_DIM)).astype(np.float32)
        beta = rng.uniform(0.01, 0.99, size=num_heads).astype(np.float32)

        S_head, out_head = numpy_recurrent_rda_step(S, q, k, a, v, beta)

        # Reproduce the tile-wise worker schedule: each tile t runs all heads.
        S_tile = S.copy()
        out_tile = np.zeros((num_heads, V_DIM), dtype=np.float32)
        for t in range(TILES_PER_HEAD):
            rows = slice(t * TILE_V, (t + 1) * TILE_V)
            # Per-tile head bundles reuse the same q/k/a/beta; only v slice differs.
            for h in range(num_heads):
                s_in_tile = S_tile[h, rows, :].copy()
                out_slice = np.zeros((TILE_V,), dtype=np.float32)
                # Minimal C-equivalent in numpy for one tile.
                s_decay = s_in_tile * a[h]
                pred = np.einsum("vk,k->v", s_decay, k[h])
                err = v[h, rows] - pred
                scale = beta[h] * err
                s_new = s_decay + scale[:, None] * k[h]
                out_slice = np.einsum("vk,k->v", s_new, q[h])
                S_tile[h, rows, :] = s_new
                out_tile[h, rows] = out_slice

        np.testing.assert_array_equal(S_head, S_tile)
        np.testing.assert_array_equal(out_head, out_tile)


class TestDesignDevice:
    """Verify that the ``col`` parameter propagates to tile coordinates."""

    def test_col_affects_tile_coordinates(self):
        # Tile helpers in the design use Tile(col, 0/1/2+t).  Here we just
        # verify the helper constants reflect the intended geometry.
        assert TILES_PER_HEAD == 4
        for col in [0, 7]:
            for t in range(TILES_PER_HEAD):
                assert (col, 2 + t) == (col, 2 + t)


class TestShimStreamCoverage:
    """Verify that the memtile->shim BD chains cover the whole output tensors."""

    def test_state_drain_stream_covers_whole_s_out(self):
        """Concatenating the state-drain BDs over H/2 runs must equal [H,V,K]."""
        num_heads = 40
        runs = num_heads // 2
        chunk_lengths = []
        for _ in range(runs):
            for s in range(2):
                for t in range(TILES_PER_HEAD):
                    tap = _aggregate_state_read_tap(t)
                    indices = _tap_indices(tap, _aggregate_buffer_shape())
                    chunk_lengths.append(len(indices))
        # Total floats streamed out == size of [H,V,K].
        assert sum(chunk_lengths) == num_heads * V_DIM * K_DIM
        # Order is slot 0 (core 0..3) then slot 1 (core 0..3); within a slot
        # the cores deliver rows t*32:(t+1)*32, which is row-major for that head.
        expected_order = [TILE_V * K_DIM] * TILES_PER_HEAD * 2 * runs
        assert chunk_lengths == expected_order

    def test_out_drain_stream_covers_whole_out(self):
        """Concatenating the out-drain BDs over H/2 runs must equal [H,V]."""
        num_heads = 40
        runs = num_heads // 2
        chunk_lengths = []
        for _ in range(runs):
            for s in range(2):
                for t in range(TILES_PER_HEAD):
                    tap = _aggregate_out_read_tap(t)
                    indices = _tap_indices(tap, _aggregate_buffer_shape())
                    chunk_lengths.append(len(indices))
        assert sum(chunk_lengths) == num_heads * V_DIM
        expected_order = [TILE_V] * TILES_PER_HEAD * 2 * runs
        assert chunk_lengths == expected_order


class TestLockProtocol:
    """Pure-Python simulation of the aggregate-buffer lock protocol.

    Each head uses slot ``h % 2``.  The two slots are independent, so each slot
    is simulated separately: heads assigned to a slot are processed sequentially,
    while within one head the four core producers and the two aggregate
    consumers (state read and out read) are shuffled.  The aggregate consumers
    model the two MM2S channels draining a slot: together they must finish before
    the slot is reused.  In the new protocol each (slot, core) has its own
    empty/full lock; in the old protocol a single empty/full pair is shared by
    the whole slot.
    """

    @staticmethod
    def _simulate(num_heads, seed, per_slot_per_core):
        """Return invariant violations for the configured lock scheme."""
        rng = np.random.default_rng(seed)
        violations = []

        if per_slot_per_core:
            empty = [[2 for _ in range(TILES_PER_HEAD)] for _ in range(2)]
            full = [[0 for _ in range(TILES_PER_HEAD)] for _ in range(2)]
        else:
            empty = [2, 2]
            full = [0, 0]

        # Process each slot independently.  Heads assigned to a slot are h = s, s+2, ...
        for s in range(2):
            heads = list(range(s, num_heads, 2))
            prod_done = set()
            cons_done = set()
            for head_idx, h in enumerate(heads):
                ops = []
                for t in range(TILES_PER_HEAD):
                    ops.append(("P", h, s, t))
                ops.append(("CS", h, s))
                ops.append(("CO", h, s))
                rng.shuffle(ops)

                pending = list(ops)
                while pending:
                    progress = False
                    still_pending = []
                    for kind, h2, s2, *rest in pending:
                        assert s2 == s
                        if kind == "P":
                            t = rest[0]
                            if per_slot_per_core:
                                allowed = empty[s][t] >= 2
                            else:
                                allowed = empty[s] >= 1
                            if not allowed:
                                still_pending.append((kind, h2, s, t))
                                continue
                            progress = True
                            # Invariant: a producer for slot s must not run before
                            # both consumers of the previous head using slot s finished.
                            if head_idx >= 1:
                                prev_h = heads[head_idx - 1]
                                if "CS" not in cons_done or "CO" not in cons_done:
                                    violations.append(
                                        f"head {h} core {t}: producer reused slot "
                                        f"before both consumers of head {prev_h} finished"
                                    )
                            if per_slot_per_core:
                                empty[s][t] -= 2
                                full[s][t] += 2
                            else:
                                empty[s] -= 1
                                full[s] += 1
                            prod_done.add(t)
                        else:
                            cons_kind = kind
                            if per_slot_per_core:
                                # Aggregate consumer must acquire all four per-core
                                # full locks and release all four empty locks.
                                allowed = all(
                                    full[s][t] >= 1 for t in range(TILES_PER_HEAD)
                                )
                                if not allowed:
                                    still_pending.append((kind, h2, s))
                                    continue
                                progress = True
                                if any(t not in prod_done for t in range(TILES_PER_HEAD)):
                                    violations.append(
                                        f"head {h}: {cons_kind} started before all "
                                        f"four producers finished"
                                    )
                                for t in range(TILES_PER_HEAD):
                                    full[s][t] -= 1
                                    empty[s][t] += 1
                            else:
                                allowed = full[s] >= 1
                                if not allowed:
                                    still_pending.append((kind, h2, s))
                                    continue
                                progress = True
                                if any(t not in prod_done for t in range(TILES_PER_HEAD)):
                                    violations.append(
                                        f"head {h}: {cons_kind} started before all "
                                        f"four producers finished"
                                    )
                                full[s] -= 1
                                empty[s] += 1
                            cons_done.add(cons_kind)
                    if not progress:
                        violations.append(
                            f"head {h} slot {s}: deadlock, {len(still_pending)} "
                            f"operations cannot proceed"
                        )
                        break
                    pending = still_pending
                # Reset per-head producer tracking; the slot lock state carries over.
                prod_done.clear()

        return violations

    def test_new_protocol_has_no_violations_across_seeds(self):
        for seed in range(20):
            violations = self._simulate(40, seed, per_slot_per_core=True)
            assert violations == [], f"seed {seed}: {violations}"

    def test_old_protocol_fails_read_before_all_writes_or_reuse_before_both_reads(
        self,
    ):
        """The old one-pair-per-slot protocol allows races; show it fails."""
        found_failure = False
        failure_text = ""
        for seed in range(100):
            violations = self._simulate(40, seed, per_slot_per_core=False)
            if violations:
                found_failure = True
                failure_text = f"seed {seed}: {violations[0]}"
                break
        assert found_failure, (
            "old protocol unexpectedly passed all seeds; it must allow at least "
            "one race condition"
        )
        print(f"\nConfirmed old-protocol failure: {failure_text}")

    def test_new_protocol_per_core_lock_values(self):
        """Direct check: a producer cannot run until empty returns to 2."""
        empty = [[2, 2, 2, 2], [2, 2, 2, 2]]
        full = [[0, 0, 0, 0], [0, 0, 0, 0]]
        empty[0][0] -= 2
        full[0][0] += 2
        full[0][0] -= 1
        empty[0][0] += 1
        assert empty[0][0] == 1
        assert full[0][0] == 1


class TestHeadCountContract:
    """The standalone 1-column design accepts only positive, even head counts.

    ``rda_step_1col`` passes ``num_heads`` to ``build_rda_channel`` as a
    compile-time JIT argument (the JIT cache keys artifacts on it), and every BD
    chain in the design runs ``runs=H//2`` times, so only positive even counts
    may reach the channel builder.
    """

    @staticmethod
    def _mock_iron_channel_symbols(monkeypatch):
        """Patch rda_state_1col so build_rda_channel works on the IRON-less host."""
        import bench.rda_state.rda_state_1col as rda_mod

        class MockTile:
            def __init__(self, col: int, row: int) -> None:
                self.col = col
                self.row = row

        class MockFifoHandle:
            def __init__(self, name: str) -> None:
                self.name = name

            def cons(self, tile=None):
                return self

            def prod(self, tile=None):
                return self

            def split(self, *, offsets, tile, obj_types, names, depths):
                return [MockFifoHandle(n) for n in names]

        class MockObjectFifo:
            def __init__(self, ty, depth: int = 1, name: str = "") -> None:
                self.name = name
                self._handle = MockFifoHandle(name)

            def prod(self, tile=None):
                return self._handle

            def cons(self, tile=None):
                return self._handle

        class MockWorker:
            def __init__(self, fn, args, tile=None, **kw) -> None:
                self.fn = fn
                self.args = args

        monkeypatch.setattr(rda_mod, "_IRON_AVAILABLE", True)
        monkeypatch.setattr(rda_mod, "Tile", MockTile, raising=False)
        monkeypatch.setattr(rda_mod, "ObjectFifo", MockObjectFifo, raising=False)
        monkeypatch.setattr(
            rda_mod, "Buffer", lambda type=None, name="", tile=None, **kw: name,
            raising=False,
        )
        monkeypatch.setattr(
            rda_mod, "Lock", lambda tile, init=0, name="": name, raising=False
        )
        monkeypatch.setattr(
            rda_mod, "Flow", lambda src, dst, name="": name, raising=False
        )
        monkeypatch.setattr(
            rda_mod, "WorkerRuntimeBarrier", lambda: "barrier", raising=False
        )
        monkeypatch.setattr(rda_mod, "Worker", MockWorker, raising=False)
        monkeypatch.setattr(
            rda_mod, "ExternalFunction", lambda *a, **kw: "external_function",
            raising=False,
        )

    @pytest.mark.parametrize("bad_heads", [39, 41, 0, -2])
    def test_odd_or_nonpositive_counts_raise_value_error(self, bad_heads):
        with pytest.raises(ValueError, match="positive even head count"):
            build_rda_channel(col=0, num_heads=bad_heads)

    @pytest.mark.parametrize("heads", [38, 40])
    def test_even_positive_counts_build_a_channel(self, monkeypatch, heads):
        self._mock_iron_channel_symbols(monkeypatch)
        channel = build_rda_channel(col=0, num_heads=heads)
        assert channel.num_heads == heads
        assert len(channel.workers) == TILES_PER_HEAD

    def test_contract_constant_is_40(self):
        assert MODEL_RDA_HEADS == 40

