# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for overlapped RDA execution on fake hardware backend."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.gemm_prefill.rda_context import _rda_inputs
from bench.rda_state.rda_state_1col import pack_head_bundle
from host.weights import bf16_to_f32, f32_to_bf16_rtne
import npu.backend as backend_mod
from npu.backend import (
    NpuBackend,
    PREFILL_GEMM_XCLBIN_ID,
    _RdaPrefillStream,
    _XrtDirectRunner,
)


DECODE_XCLBIN_ID = "gemv_unified"


def _inputs(seed: int = 20261103) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return _rda_inputs(rng, 1)[0]


class _FakeTensor:
    """Fake NPU tensor recording buffer migration, buffer object extraction, and overwrites."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = np.array(value, copy=True)
        self.to_calls: list[str] = []
        self.bo_calls = 0
        self.overwrite_count = 0

    def to(self, device: str) -> _FakeTensor:
        self.to_calls.append(device)
        return self

    def buffer_object(self) -> _FakeTensor:
        self.bo_calls += 1
        return self

    def numpy(self) -> np.ndarray:
        return self.value

    @contextmanager
    def overwrite(self):
        self.overwrite_count += 1
        yield self.value

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.value[key] = value


class _FakeDirectRunner:
    """Fake hardware direct runner implementing start() and wait()."""

    def __init__(self, tokens: int, succeed: bool = True) -> None:
        self.tokens = tokens
        self.succeed = succeed
        self.start_calls: list[list[Any]] = []
        self.wait_calls = 0

    def start(self, args: list[Any]) -> tuple[Any, Any]:
        for a in args:
            a.to("npu")
        bos = [a.buffer_object() for a in args]
        s_in, s_out, head_buf, out_buf = args
        state = np.array(s_in.value, copy=True)
        for t in range(self.tokens):
            state += head_buf.value[:, t, 0][:, None, None]
            out_buf.value[:, :, t, :] = state[:, 0, :].reshape(40, 4, 32)
        s_out.value[...] = state

        self.start_calls.append(args)
        run = _FakeRun(self)
        keep = (self, args, bos)
        return run, keep

    def wait(self, run: Any) -> None:
        self.wait_calls += 1
        if not self.succeed:
            raise RuntimeError("Fake direct runner execution failed")


class _FakeRun:
    def __init__(self, runner: _FakeDirectRunner) -> None:
        self.runner = runner


def _fake_hardware_backend(
    monkeypatch: pytest.MonkeyPatch,
    succeed: bool = True,
    succeed_by_tokens: dict[int, bool] | None = None,
) -> tuple[NpuBackend, dict[int, _FakeDirectRunner]]:
    backend = NpuBackend(mock=True)
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend.mock = False
    backend.prefill_rda_enabled = True
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")
    backend._last_active_xclbin = PREFILL_GEMM_XCLBIN_ID
    backend.rda_states[1] = [
        _FakeTensor(np.zeros((40, 128, 128), np.float32)),
        _FakeTensor(np.zeros((40, 128, 128), np.float32)),
    ]
    backend.rda_active_slot[1] = 0
    backend._work_buffers["rda_head_in"] = _FakeTensor(np.zeros((40, 640), np.float32))
    backend._work_buffers["rda_out"] = _FakeTensor(np.zeros((40, 128), np.float32))

    monkeypatch.setitem(
        sys.modules,
        "bench.rda_state.rda_state",
        types.SimpleNamespace(pack_head_bundle=pack_head_bundle),
    )

    runners = {
        t: _FakeDirectRunner(
            t,
            succeed=succeed if succeed_by_tokens is None else succeed_by_tokens.get(t, succeed),
        )
        for t in (1, 2, 4, 8)
    }
    backend._rda_prefill = {
        t: _RdaPrefillStream(
            t,
            runners[t],
            tuple(_FakeTensor(np.zeros((40, t, 640), np.float32)) for _ in range(2)),
            tuple(_FakeTensor(np.zeros((40, 4, t, 32), np.float32)) for _ in range(2)),
        )
        for t in (1, 2, 4, 8)
    }

    def run_sync(s_in: Any, s_out: Any, head_in: Any, out: Any) -> None:
        delta = head_in.value[:, 0][:, None, None]
        s_out.value[...] = s_in.value + delta
        out.value[...] = s_out.value[:, 0, :]

    backend._compiled_ops["prefill_rda"] = run_sync
    backend._compiled_ops["rda_step"] = run_sync
    return backend, runners


def _run_overlapped(
    backend: NpuBackend,
    layer: int,
    steps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sizes: tuple[int, ...],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Run blocks in the same stage/read/wait/launch order as RaonModel."""
    assert sizes and sum(sizes) == len(steps)
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size

    def stage(block: int) -> Any:
        start = offsets[block]
        return backend.rda_stage(layer, steps[start : start + sizes[block]])

    def observe(event: str, block: int, cur: Any = None, nxt: Any = None) -> None:
        observations.append(
            {
                "event": event,
                "block": block,
                "live": len(backend._rda_live),
                "live_phases": tuple(ticket.phase for ticket in backend._rda_live),
                "launched": sum(ticket.phase == "launched" for ticket in backend._rda_live),
                "cur_phase": None if cur is None else cur.phase,
                "cur_io": None if cur is None else cur.io,
                "next_phase": None if nxt is None else nxt.phase,
                "next_io": None if nxt is None else nxt.io,
                "head_busy": tuple(backend._rda_head_busy),
                "out_pending": tuple(backend._rda_out_pending),
            }
        )

    outputs: list[np.ndarray] = []
    observations: list[dict[str, Any]] = []
    cur = stage(0)
    backend.rda_launch(cur)
    observe("launch", 0, cur=cur)
    prev = None
    for block in range(len(sizes)):
        nxt = stage(block + 1) if block + 1 < len(sizes) else None
        if nxt is not None:
            observe("stage", block + 1, cur=cur, nxt=nxt)
        if prev is not None:
            observe("read", block - 1, cur=cur)
            outputs.extend(backend.rda_read(prev))
        backend.rda_wait(cur)
        prev = cur
        if nxt is not None:
            backend.rda_launch(nxt)
            observe("launch", block + 1, cur=nxt)
        cur = nxt
    outputs.extend(backend.rda_read(prev))
    observe("final", len(sizes) - 1)
    return outputs, observations


def _assert_overlap_observations(
    observations: list[dict[str, Any]],
    expected_stage_live: list[int],
    *,
    check_buffers: bool,
) -> None:
    launches = [obs for obs in observations if obs["event"] == "launch"]
    stages = [obs for obs in observations if obs["event"] == "stage"]
    reads = [obs for obs in observations if obs["event"] == "read"]
    final = observations[-1]

    assert len(launches) == len(expected_stage_live) + 1
    for obs in launches:
        assert obs["launched"] == 1
        assert obs["cur_phase"] == "launched"

    assert [obs["live"] for obs in stages] == expected_stage_live
    for obs in stages:
        assert obs["launched"] == 1
        assert obs["cur_phase"] == "launched"
        assert obs["next_phase"] == "staged"
        assert obs["next_io"] == 1 - obs["cur_io"]
        if check_buffers:
            assert obs["head_busy"][obs["cur_io"]] is True
            assert obs["head_busy"][obs["next_io"]] is False

    assert len(reads) == len(expected_stage_live)
    for obs in reads:
        assert obs["cur_phase"] == "launched"
        assert obs["launched"] == 1

    assert final["event"] == "final"
    assert final["live"] == 0
    assert final["live_phases"] == ()
    assert final["launched"] == 0
    if check_buffers:
        assert final["head_busy"] == (False, False)
        assert final["out_pending"] == (False, False)


def test_switch_and_ready(monkeypatch: pytest.MonkeyPatch):
    """Test 1: Environment variable switch and rda_overlap_ready state."""
    # Env switch: '0', 'false', 'False', 'no' disable
    for off_val in ("0", "false", "False", "no"):
        monkeypatch.setenv("RAON_PREFILL_RDA_OVERLAP", off_val)
        backend = NpuBackend(mock=True)
        assert backend.prefill_rda_overlap is False
        assert backend.rda_overlap_ready() is False

    # Setter overrides environment
    monkeypatch.setenv("RAON_PREFILL_RDA_OVERLAP", "0")
    backend = NpuBackend(mock=True)
    backend.prefill_rda_overlap = True
    assert backend.prefill_rda_overlap is True
    assert backend.rda_overlap_ready() is True

    # In mock, switch ON gives ready == True
    backend_mock = NpuBackend(mock=True)
    backend_mock.prefill_rda_overlap = True
    assert backend_mock.rda_overlap_ready() is True

    # In fake hardware, ready is True when _last_active_xclbin is PREFILL_GEMM_XCLBIN_ID, False otherwise
    hw_backend, _ = _fake_hardware_backend(monkeypatch)
    hw_backend.prefill_rda_overlap = True
    assert hw_backend._last_active_xclbin == PREFILL_GEMM_XCLBIN_ID
    assert hw_backend.rda_overlap_ready() is True

    hw_backend._last_active_xclbin = DECODE_XCLBIN_ID
    assert hw_backend.rda_overlap_ready() is False


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_rda_stage_contracts(monkeypatch: pytest.MonkeyPatch, tokens: int):
    """Test 2: rda_stage buffer contents, alternating IO, and rda_host timing accumulation."""
    backend, _ = _fake_hardware_backend(monkeypatch)
    inputs = [_inputs(20261103 + t) for t in range(tokens)]

    initial_host_time = backend.timings["rda_host"]
    initial_pipe_stage = backend.timings["rda_pipe_stage"]

    # Stage io=0
    ticket0 = backend.rda_stage(1, inputs)
    assert ticket0.io == 0
    assert ticket0.phase == "staged"
    assert backend.timings["rda_host"] > initial_host_time
    assert backend.timings["rda_pipe_stage"] > initial_pipe_stage

    # Check buffer content matches pack_head_bundle
    expected_bundles = np.stack(
        [
            pack_head_bundle(
                bf16_to_f32(q), bf16_to_f32(k), np.exp(log_alpha),
                bf16_to_f32(v), bf16_to_f32(beta),
            )
            for q, k, v, log_alpha, beta in inputs
        ],
        axis=1,
    )
    stream = backend._rda_prefill[tokens]
    np.testing.assert_array_equal(stream.heads[0].value, expected_bundles)

    # Stage io=1
    ticket1 = backend.rda_stage(1, inputs)
    assert ticket1.io == 1
    assert ticket1.phase == "staged"
    np.testing.assert_array_equal(stream.heads[1].value, expected_bundles)

    # Attempting to stage io=0 while busy raises RuntimeError
    backend._rda_head_busy[0] = True
    backend._rda_io_next = 0
    with pytest.raises(RuntimeError, match="RDA head buffer io=0 is currently busy"):
        backend.rda_stage(1, inputs)


def test_rda_launch_contracts(monkeypatch: pytest.MonkeyPatch):
    """Test 3: rda_launch argument passing, .to('npu') migration, and phase/state error checks."""
    backend, runners = _fake_hardware_backend(monkeypatch)
    q, k, v, log_alpha, beta = _inputs(20261103)

    ticket = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    backend.rda_launch(ticket)
    runner = runners[1]

    assert ticket.phase == "launched"
    assert runner.wait_calls == 0
    assert len(runner.start_calls) == 1

    # Check passed arguments: [s_in, s_out, head_buf, out_buf]
    args = runner.start_calls[0]
    assert len(args) == 4
    s_in, s_out, head_buf, out_buf = args
    assert s_in is backend.rda_states[1][0]
    assert s_out is backend.rda_states[1][1]
    stream = backend._rda_prefill[1]
    assert head_buf is stream.heads[ticket.io]
    assert out_buf is stream.outs[ticket.io]
    assert ticket.runner is runner

    # Check all four args were migrated to NPU
    for arg in args:
        assert "npu" in arg.to_calls

    # Cannot launch again when already launched
    with pytest.raises(RuntimeError, match="Ticket phase must be 'staged'"):
        backend.rda_launch(ticket)

    # Cannot launch another ticket while one is launched and not waited
    ticket2 = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    with pytest.raises(RuntimeError, match="Another ticket is already launched and not waited"):
        backend.rda_launch(ticket2)

    # Cannot launch if out[io] is pending read
    backend.rda_abort_all()
    ticket_out_pending = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    backend._rda_out_pending[ticket_out_pending.io] = True
    with pytest.raises(RuntimeError, match="is pending read"):
        backend.rda_launch(ticket_out_pending)

    # Cannot launch outside prefill RDA context
    backend.rda_abort_all()
    ticket_ctx = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    backend._last_active_xclbin = DECODE_XCLBIN_ID
    with pytest.raises(RuntimeError, match="outside prefill RDA context"):
        backend.rda_launch(ticket_ctx)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_rda_wait_success_and_failure(monkeypatch: pytest.MonkeyPatch, tokens: int):
    """Test 4: rda_wait updates slot and xclbin count on success, retains slot on failure."""
    # 4(a) Success
    backend, runners = _fake_hardware_backend(monkeypatch, succeed=True)
    ticket = backend.rda_stage(1, [_inputs(20261103 + t) for t in range(tokens)])
    backend.rda_launch(ticket)

    assert backend.rda_active_slot[1] == 0
    assert backend.xclbin_calls[PREFILL_GEMM_XCLBIN_ID] == 0

    backend.rda_wait(ticket)
    assert ticket.phase == "waited"
    assert backend.rda_active_slot[1] == 1  # Slot flipped
    assert backend.xclbin_calls[PREFILL_GEMM_XCLBIN_ID] == 1
    assert backend._rda_head_busy[ticket.io] is False
    assert {t: runner.wait_calls for t, runner in runners.items()} == {
        t: int(t == tokens) for t in (1, 2, 4, 8)
    }

    # 4(b) Failure
    backend_fail, runners_fail = _fake_hardware_backend(
        monkeypatch, succeed_by_tokens={tokens: False}
    )
    ticket_fail = backend_fail.rda_stage(
        1, [_inputs(20261203 + t) for t in range(tokens)]
    )
    backend_fail.rda_launch(ticket_fail)

    with pytest.raises(RuntimeError, match="Fake direct runner execution failed"):
        backend_fail.rda_wait(ticket_fail)

    assert backend_fail.rda_active_slot[1] == 0  # Slot NOT flipped
    assert backend_fail.xclbin_calls[PREFILL_GEMM_XCLBIN_ID] == 0
    assert backend_fail._rda_head_busy[ticket_fail.io] is True
    assert ticket_fail.phase == "launched"
    assert {t: runner.wait_calls for t, runner in runners_fail.items()} == {
        t: int(t == tokens) for t in (1, 2, 4, 8)
    }

    # 4(c) After cleaning up the failed live ticket, another token stream succeeds
    # on its own runner in the same backend.
    backend_fail.rda_abort_all()
    other_tokens = {1: 2, 2: 4, 4: 8, 8: 1}[tokens]
    before_other_waits = {
        t: runner.wait_calls for t, runner in runners_fail.items()
    }
    ticket_other = backend_fail.rda_stage(
        1, [_inputs(20261303 + t) for t in range(other_tokens)]
    )
    backend_fail.rda_launch(ticket_other)
    backend_fail.rda_wait(ticket_other)
    assert ticket_other.phase == "waited"
    assert backend_fail.rda_active_slot[1] == 1
    assert backend_fail.xclbin_calls[PREFILL_GEMM_XCLBIN_ID] == 1
    assert backend_fail._rda_head_busy[ticket_other.io] is False
    assert {
        t: runner.wait_calls - before_other_waits[t]
        for t, runner in runners_fail.items()
    } == {t: int(t == other_tokens) for t in (1, 2, 4, 8)}


def test_rda_read_contracts(monkeypatch: pytest.MonkeyPatch):
    """Test 5: rda_read returns RTNE BF16 array, clears out_pending, and transitions to done."""
    backend, runners = _fake_hardware_backend(monkeypatch, succeed=True)
    q, k, v, log_alpha, beta = _inputs(20261103)

    ticket = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    backend.rda_launch(ticket)
    backend.rda_wait(ticket)

    assert backend._rda_out_pending[ticket.io] is True
    res = backend.rda_read(ticket)

    assert ticket.phase == "done"
    assert backend._rda_out_pending[ticket.io] is False
    assert res.dtype == np.uint16
    assert res.shape == (1, 40, 128)
    raw = backend._rda_prefill[1].outs[ticket.io].numpy()
    expected_out = f32_to_bf16_rtne(raw.transpose(2, 0, 1, 3).reshape(1, 40, 128))
    np.testing.assert_array_equal(res, expected_out)
    assert ticket not in backend._rda_live


def test_overlapped_vs_sync_byte_parity_and_io_alternation(monkeypatch: pytest.MonkeyPatch):
    """Test 6: 7-step comparison between pipelined overlap and synchronous rda_update on fake hardware."""
    backend_overlap, runners_overlap = _fake_hardware_backend(monkeypatch)
    backend_sync, _ = _fake_hardware_backend(monkeypatch)

    rng = np.random.default_rng(20261103)
    steps = _rda_inputs(rng, 7)

    # Run 7 tokens as genuinely overlapped calls of 4, 2 and 1.
    outputs_overlap, observations = _run_overlapped(
        backend_overlap, 1, steps, (4, 2, 1)
    )
    _assert_overlap_observations(
        observations, [2, 3], check_buffers=True
    )

    # Run synchronous rda_update
    outputs_sync = []
    for t in range(7):
        outputs_sync.append(backend_sync.rda_update(1, *steps[t]))

    # Verify exact byte parity of outputs
    assert len(outputs_overlap) == len(outputs_sync) == 7
    for t, (out_ov, out_sy) in enumerate(zip(outputs_overlap, outputs_sync)):
        assert np.array_equal(out_ov, out_sy), f"Output mismatch at step {t}"

    # Verify final recurrent state parity
    final_slot_ov = backend_overlap.rda_active_slot[1]
    final_slot_sy = backend_sync.rda_active_slot[1]
    assert final_slot_ov == final_slot_sy
    state_ov = backend_overlap.rda_states[1][final_slot_ov].numpy()
    state_sy = backend_sync.rda_states[1][final_slot_sy].numpy()
    assert np.array_equal(state_ov, state_sy)

    # Verify each T runner received the alternating io=0,1,0 head BO.
    for size, io in zip((4, 2, 1), (0, 1, 0)):
        assert len(runners_overlap[size].start_calls) == 1
        call_args = runners_overlap[size].start_calls[0]
        assert call_args[2] is backend_overlap._rda_prefill[size].heads[io]


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_rda_abort_all_contracts(monkeypatch: pytest.MonkeyPatch, tokens: int):
    """Test 7: rda_abort_all waits for launched ticket, clears tracking, retains slot, and integrates with reset()."""
    backend, runners = _fake_hardware_backend(monkeypatch)
    runner = runners[tokens]
    inputs = [_inputs(20261103 + t) for t in range(tokens)]

    ticket = backend.rda_stage(1, inputs)
    backend.rda_launch(ticket)

    assert runner.wait_calls == 0
    backend.rda_abort_all()
    assert runner.wait_calls == 1
    assert len(backend._rda_live) == 0
    assert not any(backend._rda_head_busy)
    assert not any(backend._rda_out_pending)
    assert backend.rda_active_slot[1] == 0  # Slot retained
    assert {t: candidate.wait_calls for t, candidate in runners.items()} == {
        t: int(t == tokens) for t in (1, 2, 4, 8)
    }

    # Verify reset() calls rda_abort_all()
    ticket2 = backend.rda_stage(1, inputs)
    backend.rda_launch(ticket2)
    assert runner.wait_calls == 1  # Previous abort
    backend.reset()
    assert runner.wait_calls == 2  # Aborted during reset
    assert {t: candidate.wait_calls for t, candidate in runners.items()} == {
        t: 2 * int(t == tokens) for t in (1, 2, 4, 8)
    }


def test_sync_rda_update_fails_when_live_tickets_exist(monkeypatch: pytest.MonkeyPatch):
    """Test 8: rda_update raises RuntimeError if live overlapped tickets exist."""
    backend, _ = _fake_hardware_backend(monkeypatch)
    q, k, v, log_alpha, beta = _inputs(20261103)

    backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    with pytest.raises(RuntimeError, match="Cannot execute synchronous rda_update while overlapped RDA tickets are active"):
        backend.rda_update(1, q, k, v, log_alpha, beta)

    backend.rda_abort_all()
    # Now synchronous update should succeed
    out = backend.rda_update(1, q, k, v, log_alpha, beta)
    assert out.shape == (40, 128)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_stage_writes_each_token_to_its_own_head_axis(
    monkeypatch: pytest.MonkeyPatch, tokens: int
):
    backend, _ = _fake_hardware_backend(monkeypatch)
    inputs = []
    for t in range(tokens):
        base = np.arange(40 * 128, dtype=np.float32).reshape(40, 128) / 1024
        q = f32_to_bf16_rtne(base + t * 16)
        k = f32_to_bf16_rtne(base + 100 + t * 16)
        v = f32_to_bf16_rtne(base + 200 + t * 16)
        log_alpha = np.full((40, 128), -0.01 * (t + 1), np.float32)
        beta = f32_to_bf16_rtne(np.full(40, 0.125 * (t + 1), np.float32))
        inputs.append((q, k, v, log_alpha, beta))

    ticket = backend.rda_stage(1, inputs)
    actual = backend._rda_prefill[tokens].heads[ticket.io].value
    for t, (q, k, v, log_alpha, beta) in enumerate(inputs):
        expected = pack_head_bundle(
            bf16_to_f32(q), bf16_to_f32(k), np.exp(log_alpha),
            bf16_to_f32(v), bf16_to_f32(beta),
        )
        np.testing.assert_array_equal(actual[:, t, :], expected)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_read_gathers_head_tile_token_cell_axes(
    monkeypatch: pytest.MonkeyPatch, tokens: int
):
    backend, _ = _fake_hardware_backend(monkeypatch)
    inputs = [_inputs(700 + t) for t in range(tokens)]
    ticket = backend.rda_stage(1, inputs)
    out = backend._rda_prefill[tokens].outs[ticket.io].value
    h, tile, t, cell = np.indices(out.shape)
    marker = h * 100_000 + tile * 10_000 + t * 1_000 + cell
    out[...] = marker.astype(np.float32)
    ticket.phase = "waited"
    backend._rda_out_pending[ticket.io] = True

    actual = backend.rda_read(ticket)
    expected = f32_to_bf16_rtne(
        marker.transpose(2, 0, 1, 3).reshape(tokens, 40, 128).astype(np.float32)
    )
    np.testing.assert_array_equal(actual, expected)


def test_short_call_does_not_read_previous_long_call_output(monkeypatch: pytest.MonkeyPatch):
    backend, _ = _fake_hardware_backend(monkeypatch)
    for tokens, fill in ((8, 111.0), (1, 222.0), (2, 333.0)):
        ticket = backend.rda_stage(1, [_inputs(800 + t) for t in range(tokens)])
        backend._rda_prefill[tokens].outs[ticket.io].value.fill(fill)
        ticket.phase = "waited"
        result = backend.rda_read(ticket)
    assert result.shape == (2, 40, 128)
    np.testing.assert_array_equal(result, f32_to_bf16_rtne(np.full(result.shape, 333.0, np.float32)))


@pytest.mark.parametrize("tokens", (0, 3, 5, 16))
def test_stage_rejects_invalid_token_count_without_changing_tracking(
    monkeypatch: pytest.MonkeyPatch, tokens: int
):
    backend, _ = _fake_hardware_backend(monkeypatch)
    before = (backend._rda_io_next, list(backend._rda_live), list(backend._rda_head_busy))
    with pytest.raises(ValueError):
        backend.rda_stage(1, [_inputs(900 + t) for t in range(tokens)])
    assert (backend._rda_io_next, backend._rda_live, backend._rda_head_busy) == before


def test_mock_candidate_state_commit_abort_and_block_parity():
    candidate = NpuBackend(mock=True)
    inputs = _rda_inputs(np.random.default_rng(1001), 23)
    first = candidate.rda_stage(1, inputs[:8])
    candidate.rda_launch(first)
    assert 1 not in candidate.rda_states
    candidate.rda_abort_all()
    assert 1 not in candidate.rda_states
    assert first.mock_state is None

    committed = NpuBackend(mock=True)
    committed_ticket = committed.rda_stage(1, inputs[:8])
    committed.rda_launch(committed_ticket)
    assert 1 not in committed.rda_states
    assert committed_ticket.mock_state is not None
    committed.rda_wait(committed_ticket)
    assert 1 in committed.rda_states
    assert committed_ticket.mock_state is None
    committed.rda_read(committed_ticket)

    overlap = NpuBackend(mock=True)
    sync = NpuBackend(mock=True)
    overlap_outputs, observations = _run_overlapped(
        overlap, 1, inputs, (8, 8, 4, 2, 1)
    )
    _assert_overlap_observations(
        observations, [2, 3, 3, 3], check_buffers=False
    )
    sync_outputs = [sync.rda_update(1, *values) for values in inputs]
    np.testing.assert_array_equal(np.stack(overlap_outputs), np.stack(sync_outputs))
    np.testing.assert_array_equal(overlap.get_rda_state(1), sync.get_rda_state(1))


def test_state_access_is_rejected_while_ticket_is_live():
    backend = NpuBackend(mock=True)
    ticket = backend.rda_stage(1, [_inputs(1100)])
    with pytest.raises(RuntimeError, match="Cannot read RDA state"):
        backend.get_rda_state(1)
    with pytest.raises(RuntimeError, match="Cannot set RDA state"):
        backend.set_rda_state(1, np.zeros((40, 128, 128), np.float32))
    backend.rda_launch(ticket)
    backend.rda_wait(ticket)
    backend.rda_read(ticket)
    state = backend.get_rda_state(1)
    backend.set_rda_state(1, state)


def test_xrt_direct_runner_eviction_and_invalid_handle():
    """Test 9: _XrtDirectRunner handle validation and re-loading upon cache eviction."""
    class FakeHandle:
        def __init__(self, is_valid: bool = True, has_bo: bool = True):
            self._is_valid = is_valid
            self.insts_bo = object() if has_bo else None
            self.insts = type("Insts", (), {"nbytes": 1024})()
            self.kernel_calls = []

        def kernel(self, mode: int, insts_bo: Any, insts_bytes: int, *bos: Any) -> Any:
            self.kernel_calls.append((mode, insts_bo, insts_bytes, bos))
            return type("Run", (), {"wait": lambda: 1})()

    class FakeRuntime:
        def __init__(self, handle_factory):
            self.handle_factory = handle_factory
            self.load_calls = 0

        def load(self, kernel: Any) -> Any:
            self.load_calls += 1
            return self.handle_factory()

    class FakeTensorArg:
        def to(self, device: str) -> None:
            pass

        def buffer_object(self) -> Any:
            return object()

    # 9(a) Handle initially invalid triggers re-loading
    load_handles = [FakeHandle(is_valid=False), FakeHandle(is_valid=True)]
    runtime = FakeRuntime(lambda: load_handles.pop(0))
    runner = _XrtDirectRunner(runtime, "kernel", completed_state=1)
    assert runtime.load_calls == 1  # From __init__

    args = [FakeTensorArg()]
    run, keep = runner.start(args)
    assert runtime.load_calls == 2  # Re-loaded in start()
    assert runner.handle._is_valid is True

    # 9(b) If re-loaded handle still lacks insts_bo, raise RuntimeError
    bad_runtime = FakeRuntime(lambda: FakeHandle(is_valid=True, has_bo=False))
    bad_runner = _XrtDirectRunner(bad_runtime, "kernel", completed_state=1)
    with pytest.raises(RuntimeError, match="NPU kernel handle does not have insts_bo"):
        bad_runner.start(args)


def test_rda_launch_lazy_compilation(monkeypatch: pytest.MonkeyPatch):
    """Test 10: the first hardware stage initializes the prefill streams once."""
    backend, runners = _fake_hardware_backend(monkeypatch)
    streams = backend._rda_prefill
    backend._rda_prefill = None

    compile_calls = 0

    def fake_get_or_compile():
        nonlocal compile_calls
        compile_calls += 1
        backend._rda_prefill = streams
        return lambda *args: None

    monkeypatch.setattr(backend, "_get_or_compile_prefill_rda", fake_get_or_compile)

    q, k, v, log_alpha, beta = _inputs(20261103)

    # The BO lives in the stream, so compilation happens during the first stage.
    ticket1 = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    assert compile_calls == 1
    backend.rda_launch(ticket1)
    assert ticket1.phase == "launched"

    backend.rda_wait(ticket1)
    backend.rda_read(ticket1)

    # Second launch: _rda_direct is already initialized, should NOT call compile again
    ticket2 = backend.rda_stage(1, [(q, k, v, log_alpha, beta)])
    assert compile_calls == 1
    backend.rda_launch(ticket2)
    assert compile_calls == 1
    assert ticket2.phase == "launched"

    backend.rda_wait(ticket2)
    backend.rda_read(ticket2)


def test_get_or_compile_prefill_rda_builds_all_token_streams_once(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, Any]] = []
    callable_by_t = {t: object() for t in (1, 2, 4, 8)}

    class FakeOp:
        def __init__(self, tokens: int):
            self.tokens = tokens
            self.compiles = 0

        def compile(self):
            self.compiles += 1

        def get_callable(self):
            return callable_by_t[self.tokens]

    class FakePrefillGEMM:
        @classmethod
        def for_shape(cls, shape, **kwargs):
            calls.append({"shape": shape, **kwargs})
            return FakeOp(kwargs["rda_tokens"])

    class FakeIron:
        shapes: list[tuple[int, ...]] = []

        @classmethod
        def zeros(cls, shape, **kwargs):
            cls.shapes.append(tuple(shape))
            return _FakeTensor(np.zeros(shape, np.float32))

    fake_module = types.SimpleNamespace(PrefillGEMM=FakePrefillGEMM)
    monkeypatch.setitem(sys.modules, "kernels.gemm_prefill.op", fake_module)
    monkeypatch.setattr(backend_mod, "iron", FakeIron)
    monkeypatch.setattr(backend_mod, "_make_xrt_direct_runner", lambda op: f"runner-{op.tokens}")
    monkeypatch.setattr(backend_mod, "_xclbin_path_from_op", lambda op: Path("plate.xclbin"))

    backend = NpuBackend(mock=True)
    backend.mock = False
    backend._prefill_xclbin_path = Path("plate.xclbin")
    result = backend._get_or_compile_prefill_rda()
    assert result is callable_by_t[1]
    assert [call["rda_tokens"] for call in calls] == [1, 2, 4, 8]
    assert all(
        call["shape"] == "rda_in"
        and call["execution"] == "rda"
        and call["rda_col"] == 7
        for call in calls
    )
    assert FakeIron.shapes == [
        shape
        for t in (1, 2, 4, 8)
        for shape in ((40, t, 640), (40, t, 640), (40, 4, t, 32), (40, 4, t, 32))
    ]
    assert backend._get_or_compile_prefill_rda() is result
    assert len(calls) == 4

    mismatch = NpuBackend(mock=True)
    mismatch.mock = False
    mismatch._prefill_xclbin_path = Path("plate.xclbin")
    monkeypatch.setattr(
        backend_mod,
        "_xclbin_path_from_op",
        lambda op: Path("other.xclbin") if op.tokens == 2 else Path("plate.xclbin"),
    )
    with pytest.raises(RuntimeError, match="xclbin path mismatch"):
        mismatch._get_or_compile_prefill_rda()


def test_xrt_direct_runner_start_and_wait_contracts():
    """Test 11: _XrtDirectRunner start() and wait() contracts verifying buffer migration order and completion check."""
    events: list[str] = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.bo = f"bo_{name}"

        def to(self, device: str) -> None:
            events.append(f"{self.name}.to({device})")

        def buffer_object(self) -> str:
            events.append(f"{self.name}.buffer_object")
            return self.bo

    class FakeHandle:
        def __init__(self):
            self._is_valid = True
            self.insts_bo = "INSTS_BO"
            self.insts = type("Insts", (), {"nbytes": 4096})()
            self.kernel_calls: list[tuple[Any, ...]] = []

        def kernel(self, mode: int, insts_bo: Any, insts_bytes: int, *bos: Any) -> Any:
            events.append("handle.kernel")
            self.kernel_calls.append((mode, insts_bo, insts_bytes, *bos))
            return fake_run

    class FakeRun:
        def __init__(self, return_state: int):
            self.return_state = return_state
            self.wait_called = False

        def wait(self) -> int:
            self.wait_called = True
            events.append("run.wait")
            return self.return_state

    class FakeRuntime:
        def __init__(self, handle: FakeHandle):
            self.handle = handle

        def load(self, kernel: Any) -> FakeHandle:
            return self.handle

    handle = FakeHandle()
    runtime = FakeRuntime(handle)
    completed_state = 100
    runner = _XrtDirectRunner(runtime, "kernel_desc", completed_state=completed_state)

    fake_run = FakeRun(return_state=completed_state)

    t1 = FakeTensor("t1")
    t2 = FakeTensor("t2")
    t3 = FakeTensor("t3")
    args = [t1, t2, t3]

    # 1. start() verification
    run, keep = runner.start(args)

    # (a) Verify all args' .to("npu") are called BEFORE handle.kernel(...)
    kernel_event_idx = events.index("handle.kernel")
    for name in ("t1", "t2", "t3"):
        to_event_idx = events.index(f"{name}.to(npu)")
        assert to_event_idx < kernel_event_idx, f"{name}.to('npu') must be called before handle.kernel"

    # (b) Verify handle.kernel received (3, insts_bo, insts.nbytes, *bos) where bos are buffer_object() in order
    assert len(handle.kernel_calls) == 1
    call_args = handle.kernel_calls[0]
    expected_bos = ["bo_t1", "bo_t2", "bo_t3"]
    assert call_args == (3, "INSTS_BO", 4096, "bo_t1", "bo_t2", "bo_t3")

    # (c) Verify keep is (handle, args, bos)
    assert keep == (handle, args, expected_bos)
    assert run is fake_run

    # 2. wait() verification
    # (a) When run returns completed_state, wait succeeds without error
    runner.wait(fake_run)
    assert fake_run.wait_called is True

    # (b) When run returns a different state, wait raises RuntimeError
    bad_run = FakeRun(return_state=999)
    with pytest.raises(RuntimeError, match="Kernel execution failed with state 999"):
        runner.wait(bad_run)


def test_overlapped_timing_call_counts(monkeypatch: pytest.MonkeyPatch):
    """Test 12: each block contributes one host/stage/read timing call."""
    backend, _ = _fake_hardware_backend(monkeypatch)
    blocks = (4, 1, 8, 2)
    rng = np.random.default_rng(20261103)
    steps = _rda_inputs(rng, sum(blocks))

    # Initial call counts
    assert backend.call_counts["rda_host"] == 0
    assert backend.call_counts["host_prep"] == 0

    offset = 0
    for size in blocks:
        ticket = backend.rda_stage(1, steps[offset : offset + size])
        backend.rda_launch(ticket)
        backend.rda_wait(ticket)
        backend.rda_read(ticket)
        offset += size

    assert backend.call_counts["rda_host"] == len(blocks)
    assert backend.call_counts["host_prep"] == len(blocks)
    assert backend.call_counts["rda_pipe_stage"] == len(blocks)
    assert backend.call_counts["rda_pipe_read"] == len(blocks)
    assert backend.timings["rda_host"] > 0.0
    assert backend.timings["host_prep"] > 0.0
