# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for overlapped RDA prefill execution with byte parity and scheduling contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from bench.prefill_replay.harness import run_prefill_replay
from host.raon import RaonModel, rda_block_plan
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.test_prefill_parity import NpuCoupledMockBackend
from host.weights import Weights
from serve.prefill_plan import prefill_segments

_ = tiny_raon_model


@dataclass
class _MockTicket:
    ticket_no: int
    layer: int
    io: int
    phase: str
    inputs: tuple[tuple[Any, ...], ...] | None = None
    out: np.ndarray | None = None


class OverlapMockBackend(NpuCoupledMockBackend):
    """Mock backend tracking and verifying RDA overlapped pipeline contracts."""

    def __init__(self, weights: Weights) -> None:
        super().__init__(weights)
        self.prefill_rda_overlap = True
        self.events: list[tuple[str, int, int]] = []
        self._ticket_counter = 0
        self._head_busy = [False, False]
        self._out_pending = [False, False]
        self._io_next = 0
        self._live_tickets: list[_MockTicket] = []
        self.aborted = False
        self.launch_count = 0
        self.wait_fail_ticket: int | None = None

    def reset(self) -> None:
        self.rda_abort_all()
        super().reset()
        self.events.clear()
        self.launch_count = 0
        self.aborted = False
        self.wait_fail_ticket = None

    def rda_overlap_ready(self) -> bool:
        return bool(self.prefill_rda_overlap)

    def rda_stage(self, layer: int, inputs: list[tuple[Any, ...]]) -> _MockTicket:
        io = self._io_next
        if self._head_busy[io]:
            raise RuntimeError(f"RDA head buffer io={io} is currently busy")
        self._ticket_counter += 1
        ticket = _MockTicket(
            ticket_no=self._ticket_counter,
            layer=layer,
            io=io,
            phase="staged",
            inputs=tuple(inputs),
        )
        self.events.append(("stage", ticket.ticket_no, io))
        self._io_next = 1 - io
        self._live_tickets.append(ticket)
        return ticket

    def rda_launch(self, ticket: _MockTicket) -> None:
        if ticket.phase != "staged":
            raise RuntimeError(f"Ticket phase must be 'staged', got '{ticket.phase}'")
        if self._out_pending[ticket.io]:
            raise RuntimeError(f"RDA out buffer io={ticket.io} is pending read")
        if any(t.phase == "launched" for t in self._live_tickets if t is not ticket):
            raise RuntimeError("Another ticket is already launched and not waited")

        io = ticket.io
        self._head_busy[io] = True
        self._out_pending[io] = True
        ticket.out = np.stack(
            [self.rda_update(ticket.layer, *inputs) for inputs in ticket.inputs]
        )
        ticket.phase = "launched"
        self.launch_count += 1
        self.events.append(("launch", ticket.ticket_no, io))

    def rda_wait(self, ticket: _MockTicket) -> None:
        if ticket.phase != "launched":
            raise RuntimeError(f"Ticket phase must be 'launched', got '{ticket.phase}'")
        if self.wait_fail_ticket is not None and ticket.ticket_no == self.wait_fail_ticket:
            raise RuntimeError(f"Simulated wait failure for ticket {ticket.ticket_no}")
        io = ticket.io
        self._head_busy[io] = False
        ticket.phase = "waited"
        self.events.append(("wait", ticket.ticket_no, io))

    def rda_read(self, ticket: _MockTicket) -> np.ndarray:
        if ticket.phase != "waited":
            raise RuntimeError(f"Ticket phase must be 'waited', got '{ticket.phase}'")
        io = ticket.io
        self._out_pending[io] = False
        ticket.phase = "done"
        self.events.append(("read", ticket.ticket_no, io))
        if ticket in self._live_tickets:
            self._live_tickets.remove(ticket)
        return ticket.out

    def rda_abort_all(self) -> None:
        self.aborted = True
        for ticket in list(self._live_tickets):
            ticket.phase = "waited"
        self._live_tickets.clear()
        self._head_busy = [False, False]
        self._out_pending = [False, False]
        self._io_next = 0


@pytest.mark.parametrize(
    ("token_count", "chunk"),
    [
        (1, 16),
        (2, 16),
        (3, 2),
        (5, 1),
        (16, 16),
        (17, 16),
        (17, 64),
        (64, 64),
    ],
)
@pytest.mark.parametrize("block", (1, 8))
def test_prefill_replay_matches_overlapped(
    tiny_raon_model, token_count: int, chunk: int, block: int
):
    """Test 1: run_prefill_replay matches byte-for-byte across token and chunk configurations."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = OverlapMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    model.prefill_rda_block = block

    rng = np.random.default_rng(20261103 + token_count + chunk)
    tokens = rng.integers(0, host_cfg.vocab_size, size=token_count).tolist()

    backend.launch_count = 0
    res = run_prefill_replay(model, tokens, chunk=chunk)
    assert res["match"] is True, f"Mismatch in prefill replay: {res['first_mismatch']}"

    # Verify overlapped path was actually taken during prefill
    rda_layers = [i for i in range(host_cfg.num_hidden_layers) if host_cfg.is_rda_layer(i)]
    segments = prefill_segments(
        cached=0,
        n=token_count,
        chunk=chunk,
        checkpoint_len=token_count,
        prompt_tokens=token_count,
    )
    expected_launches = len(rda_layers) * sum(
        len(rda_block_plan(step, block)) for step, _ in segments
    )
    assert backend.launch_count == expected_launches


@pytest.mark.parametrize("block", (1, 8))
def test_overlapped_scheduling_order(tiny_raon_model, block: int):
    """Test 2: Detailed ordering constraints on events within layers and chunks."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = OverlapMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    model.prefill_rda_block = block

    tokens = [1, 2, 3, 4, 5, 6, 7]
    model.reset_state()
    backend.events.clear()
    model.prefill(tokens, chunk=4)

    events = backend.events
    assert len(events) > 0

    # Group events by ticket_no
    ticket_events: dict[int, dict[str, int]] = {}
    ticket_ios: dict[int, int] = {}
    for idx, (op, ticket_no, io) in enumerate(events):
        ticket_events.setdefault(ticket_no, {})[op] = idx
        ticket_ios[ticket_no] = io

    # Check each ticket received launch, wait, read exactly once in order
    for t_no, ops in ticket_events.items():
        assert "stage" in ops, f"Ticket {t_no} missing stage"
        assert "launch" in ops, f"Ticket {t_no} missing launch"
        assert "wait" in ops, f"Ticket {t_no} missing wait"
        assert "read" in ops, f"Ticket {t_no} missing read"
        assert ops["stage"] < ops["launch"] < ops["wait"] < ops["read"]

    # Verify single in-flight execution at all times
    in_flight = 0
    for op, _, _ in events:
        if op == "launch":
            in_flight += 1
            assert in_flight <= 1, "More than one launched execution in flight"
        elif op == "wait":
            in_flight -= 1
            assert in_flight >= 0

    # Every pipeline batch uses the same global two-slot discipline.  This also
    # covers block=8, where a chunk may contain only one RDA call.
    all_tickets = sorted(ticket_events.keys())
    for i in range(len(all_tickets) - 1):
        t1, t2 = all_tickets[i], all_tickets[i + 1]
        ops1, ops2 = ticket_events[t1], ticket_events[t2]
        assert ops2["launch"] > ops1["wait"]
        assert ticket_ios[t2] == 1 - ticket_ios[t1]
        if i >= 1:
            t0 = all_tickets[i - 1]
            ops0 = ticket_events[t0]
            assert ops0["read"] < ops2["launch"]


def test_overlapped_on_vs_off_byte_parity(tiny_raon_model):
    """Test 3: Bit-for-bit parity of logits, RDA state, and conv state between ready=True and ready=False."""
    _, host_cfg, weights, _ = tiny_raon_model
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]

    # Run with overlap ON
    backend_on = OverlapMockBackend(weights)
    backend_on.prefill_rda_overlap = True
    model_on = RaonModel(host_cfg, weights, backend=backend_on)
    model_on.reset_state()
    logits_on = model_on.prefill(tokens, chunk=4)
    conv_on = {
        i: {k: v.copy() for k, v in states.items()}
        for i, states in model_on.conv_states.items()
    }
    rda_on = {
        i: backend_on.get_rda_state(i).copy()
        for i in range(host_cfg.num_hidden_layers)
        if host_cfg.is_rda_layer(i)
    }

    # Run with overlap OFF
    backend_off = OverlapMockBackend(weights)
    backend_off.prefill_rda_overlap = False
    model_off = RaonModel(host_cfg, weights, backend=backend_off)
    model_off.reset_state()
    logits_off = model_off.prefill(tokens, chunk=4)
    conv_off = {
        i: {k: v.copy() for k, v in states.items()}
        for i, states in model_off.conv_states.items()
    }
    rda_off = {
        i: backend_off.get_rda_state(i).copy()
        for i in range(host_cfg.num_hidden_layers)
        if host_cfg.is_rda_layer(i)
    }

    assert np.array_equal(logits_on, logits_off)
    for layer in conv_on:
        for k in conv_on[layer]:
            assert np.array_equal(conv_on[layer][k], conv_off[layer][k])
    for layer in rda_on:
        assert np.array_equal(rda_on[layer], rda_off[layer])


def test_failure_in_rda_inputs_triggers_abort_all(tiny_raon_model):
    """Test 4(a): Exception in _rda_inputs bubbles up, triggers rda_abort_all, and leaves no running execution."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = OverlapMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    call_count = 0
    orig_rda_inputs = model._rda_inputs

    def failing_rda_inputs(layer_idx, q, k, v, beta_raw, rda_delta, decay_raw):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise ValueError("Simulated error in _rda_inputs")
        return orig_rda_inputs(layer_idx, q, k, v, beta_raw, rda_delta, decay_raw)

    model._rda_inputs = failing_rda_inputs

    with pytest.raises(ValueError, match="Simulated error in _rda_inputs"):
        model.prefill([1, 2, 3, 4], chunk=4)

    assert backend.aborted is True
    assert len(backend._live_tickets) == 0
    assert not any(backend._head_busy)
    assert not any(backend._out_pending)


def test_failure_in_rda_wait_does_not_commit_slot(tiny_raon_model):
    """Test 4(b): Exception in rda_wait bubbles up, aborts execution, and leaves no slot commit."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = OverlapMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    backend.wait_fail_ticket = 2

    with pytest.raises(RuntimeError, match="Simulated wait failure for ticket 2"):
        model.prefill([1, 2, 3, 4], chunk=4)

    assert backend.aborted is True
    assert len(backend._live_tickets) == 0
