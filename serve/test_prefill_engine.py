# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit test for server engine prefill batching equivalence (RAON_PREFILL_CHUNK=0 vs 16)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import numpy as np
import pytest

from host.backend import NumpyBackend
from host.raon import RaonModel
from host.test_host_small import tiny_raon_model  # noqa: F401
from serve.prefill_plan import prefill_segments
from serve.test_engine_small import Engine, FakeTokenizer, GenOptions

_ = tiny_raon_model


class TinyFakeTokenizer(FakeTokenizer):
    """FakeTokenizer that constrains token IDs to fit tiny model's small vocab_size."""

    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_ids = [2]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        return [int(x) % self.vocab_size for x in ids]

    def chat_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        raw = super().chat_ids(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        return [int(x) % self.vocab_size for x in raw]


def _run_engine_turns(tiny_raon_model, chunk_env: str) -> dict[str, Any]:
    _, host_cfg, weights, _ = tiny_raon_model

    # Set up engine with real RaonModel + NumpyBackend + TinyFakeTokenizer
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    tokenizer = TinyFakeTokenizer(vocab_size=host_cfg.vocab_size)

    engine = Engine(weights_dir=None, model_dir=tokenizer, backend=backend, model=model)

    old_env = os.environ.get("RAON_PREFILL_CHUNK")
    try:
        os.environ["RAON_PREFILL_CHUNK"] = chunk_env

        messages_turn1 = [
            {"role": "user", "content": "Explain prefill chunk scheduling in detail with examples."}
        ]
        opts = GenOptions(max_tokens=8, temperature=0.0)

        chunks_turn1 = list(engine.chat(messages_turn1, options=opts))
        text_turn1 = "".join(c.text for c in chunks_turn1)
        final_chunk_turn1 = chunks_turn1[-1]

        cp_turn1 = None
        if engine._checkpoint is not None:
            cp_turn1 = {
                "tokens": list(engine._checkpoint.tokens),
                "last_logits": np.array(engine._checkpoint.last_logits, copy=True),
                "rda_states": {k: np.array(v, copy=True) for k, v in engine._checkpoint.rda_states.items()},
                "conv_states": {
                    k: {sub_k: np.array(sub_v, copy=True) for sub_k, sub_v in v.items()}
                    for k, v in engine._checkpoint.conv_states.items()
                },
                "mla_lengths": dict(engine._checkpoint.mla_lengths),
            }

        # Turn 2: User asks follow-up, triggering prefix reuse from checkpoint
        messages_turn2 = list(messages_turn1)
        messages_turn2.append({"role": "assistant", "content": text_turn1})
        messages_turn2.append({"role": "user", "content": "Tell me more about it."})

        chunks_turn2 = list(engine.chat(messages_turn2, options=opts))
        text_turn2 = "".join(c.text for c in chunks_turn2)
        final_chunk_turn2 = chunks_turn2[-1]

        return {
            "text_turn1": text_turn1,
            "completion_tokens_turn1": final_chunk_turn1.completion_tokens,
            "checkpoint_turn1": cp_turn1,
            "text_turn2": text_turn2,
            "reused_prefix_turn2": final_chunk_turn2.reused_prefix_tokens,
            "completion_tokens_turn2": final_chunk_turn2.completion_tokens,
        }
    finally:
        if old_env is None:
            os.environ.pop("RAON_PREFILL_CHUNK", None)
        else:
            os.environ["RAON_PREFILL_CHUNK"] = old_env


def test_server_prefill_chunk_zero_vs_sixteen(tiny_raon_model):
    """(다) Verify server RAON_PREFILL_CHUNK=0 and 16 produce identical tokens, checkpoints, and prefix reuse."""
    res_chunk0 = _run_engine_turns(tiny_raon_model, "0")
    res_chunk16 = _run_engine_turns(tiny_raon_model, "16")

    # 1. Turn 1 generated text and token count
    assert res_chunk0["text_turn1"] == res_chunk16["text_turn1"]
    assert res_chunk0["completion_tokens_turn1"] == res_chunk16["completion_tokens_turn1"]

    # 2. Checkpoint parity after Turn 1
    cp0 = res_chunk0["checkpoint_turn1"]
    cp16 = res_chunk16["checkpoint_turn1"]
    assert (cp0 is None) == (cp16 is None)
    if cp0 is not None and cp16 is not None:
        assert cp0["tokens"] == cp16["tokens"]
        assert cp0["last_logits"].tobytes() == cp16["last_logits"].tobytes()
        assert cp0["mla_lengths"] == cp16["mla_lengths"]

        # RDA states byte comparison
        assert set(cp0["rda_states"].keys()) == set(cp16["rda_states"].keys())
        for k in cp0["rda_states"]:
            assert cp0["rda_states"][k].tobytes() == cp16["rda_states"][k].tobytes()

        # Conv states byte comparison
        assert set(cp0["conv_states"].keys()) == set(cp16["conv_states"].keys())
        for k in cp0["conv_states"]:
            for sub_k in cp0["conv_states"][k]:
                assert cp0["conv_states"][k][sub_k].tobytes() == cp16["conv_states"][k][sub_k].tobytes()

    # 3. Turn 2 prefix reuse and generated text
    assert res_chunk0["reused_prefix_turn2"] == res_chunk16["reused_prefix_turn2"]
    assert res_chunk0["reused_prefix_turn2"] > 0
    assert res_chunk0["text_turn2"] == res_chunk16["text_turn2"]
    assert res_chunk0["completion_tokens_turn2"] == res_chunk16["completion_tokens_turn2"]


def test_prefill_segments_checkpoint_mid_chunk():
    """체크포인트가 묶음 중간에 오는 경우: 체크포인트 위치에서 묶음을 자르고 logits 수집."""
    segs = prefill_segments(cached=0, n=20, chunk=16, checkpoint_len=5, prompt_tokens=20)
    assert segs == [(5, [4]), (15, [14])]


def test_prefill_segments_checkpoint_already_passed():
    """체크포인트가 이미 지난 경우: 체크포인트 분할 없이 chunk 단위로 나누고 끝에서 logits 수집."""
    segs = prefill_segments(cached=8, n=12, chunk=4, checkpoint_len=5, prompt_tokens=20)
    assert segs == [(4, []), (4, []), (4, [3])]


def test_prefill_segments_remaining_less_than_chunk():
    """남은 토큰이 chunk 보다 적은 경우: 남은 크기만큼의 마지막 묶음 생성."""
    segs = prefill_segments(cached=0, n=7, chunk=4, checkpoint_len=10, prompt_tokens=7)
    assert segs == [(4, []), (3, [2])]


def test_prefill_segments_chunk_one():
    """chunk=1 인 경우: 1토큰씩 개별 분할되며 체크포인트와 끝 위치에서 logits 수집."""
    segs = prefill_segments(cached=0, n=4, chunk=1, checkpoint_len=2, prompt_tokens=4)
    assert segs == [(1, []), (1, [0]), (1, []), (1, [0])]


def test_prefill_segments_explicit_target_logits_positions():
    """임의의 logits_positions가 전달되었을 때 각 묶음의 상대 위치로 정확히 변환."""
    segs = prefill_segments(
        cached=0,
        n=10,
        chunk=4,
        checkpoint_len=0,
        prompt_tokens=10,
        target_logits_positions=[2, 5, 9],
    )
    assert segs == [(4, [2]), (4, [1]), (2, [1])]


class FailingSetStepBackend(NumpyBackend):
    """NumpyBackend that provides set_step and can fail on demand."""

    def __init__(self, weights):
        super().__init__(weights)
        self.fail_on_set_step = False
        self.step_history: list[int] = []

    def set_step(self, step: int) -> None:
        self.step_history.append(step)
        if self.fail_on_set_step:
            raise RuntimeError("simulated set_step failure")


def _run_engine_turn1_then_set_step_failure(tiny_raon_model, chunk_env: str) -> dict[str, Any]:
    _, host_cfg, weights, _ = tiny_raon_model

    backend = FailingSetStepBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    tokenizer = TinyFakeTokenizer(vocab_size=host_cfg.vocab_size)
    engine = Engine(weights_dir=None, model_dir=tokenizer, backend=backend, model=model)

    old_env = os.environ.get("RAON_PREFILL_CHUNK")
    try:
        os.environ["RAON_PREFILL_CHUNK"] = chunk_env
        opts = GenOptions(max_tokens=4, temperature=0.0)

        # Turn 1: 정상 실행하여 _checkpoint, _cached_tokens, _last_logits 생성
        messages_turn1 = [{"role": "user", "content": "Explain prefill chunk scheduling."}]
        chunks1 = list(engine.chat(messages_turn1, options=opts))
        text1 = "".join(c.text for c in chunks1)

        assert engine._checkpoint is not None
        assert len(engine._cached_tokens) > 0
        assert engine._last_logits is not None
        assert len(backend.rda_states) > 0

        # Turn 2: set_step 실패 활성화
        backend.fail_on_set_step = True

        messages_turn2 = list(messages_turn1)
        messages_turn2.append({"role": "assistant", "content": text1})
        messages_turn2.append({"role": "user", "content": "Tell me more about it."})

        with pytest.raises(RuntimeError, match="simulated set_step failure"):
            list(engine.chat(messages_turn2, options=opts))

        return {
            "cached_tokens": list(engine._cached_tokens),
            "last_logits": engine._last_logits,
            "checkpoint": engine._checkpoint,
            "conv_states_len": len(model.conv_states),
            "rda_states_len": len(backend.rda_states),
            "mla_k_len": len(backend.mla_key_cache),
            "mla_v_len": len(backend.mla_value_cache),
        }
    finally:
        if old_env is None:
            os.environ.pop("RAON_PREFILL_CHUNK", None)
        else:
            os.environ["RAON_PREFILL_CHUNK"] = old_env


def test_prefill_set_step_failure_recovery_both_paths(tiny_raon_model):
    """(가) set_step이 예외를 낼 때 _cached_tokens, _last_logits, _checkpoint, 모델 상태 초기화 검증.
    chunk=16(묶음)과 chunk=0(토큰별) 두 경로에서 결과가 동일함을 확인."""
    res_chunk16 = _run_engine_turn1_then_set_step_failure(tiny_raon_model, "16")
    res_chunk0 = _run_engine_turn1_then_set_step_failure(tiny_raon_model, "0")

    for res, name in [(res_chunk16, "chunk16"), (res_chunk0, "chunk0")]:
        assert res["cached_tokens"] == [], f"{name}: cached_tokens should be empty"
        assert res["last_logits"] is None, f"{name}: last_logits should be None"
        assert res["checkpoint"] is None, f"{name}: checkpoint should be None"
        assert res["conv_states_len"] == 0, f"{name}: conv_states should be empty"
        assert res["rda_states_len"] == 0, f"{name}: rda_states should be empty"
        assert res["mla_k_len"] == 0, f"{name}: mla_k should be empty"
        assert res["mla_v_len"] == 0, f"{name}: mla_v should be empty"


def test_prefill_second_chunk_failure_recovery(tiny_raon_model):
    """(나) model.prefill이 두 번째 묶음에서 예외를 낼 때 _cached_tokens, _last_logits,
    _checkpoint, 모델 상태가 (가)와 같이 초기화되는지 검증."""
    _, host_cfg, weights, _ = tiny_raon_model

    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    tokenizer = TinyFakeTokenizer(vocab_size=host_cfg.vocab_size)
    engine = Engine(weights_dir=None, model_dir=tokenizer, backend=backend, model=model)

    old_env = os.environ.get("RAON_PREFILL_CHUNK")
    try:
        os.environ["RAON_PREFILL_CHUNK"] = "16"
        opts = GenOptions(max_tokens=4, temperature=0.0)

        # Turn 1 정상 실행하여 checkpoint와 cached_tokens 생성
        messages_turn1 = [{"role": "user", "content": "Explain prefill chunk scheduling."}]
        chunks1 = list(engine.chat(messages_turn1, options=opts))
        text1 = "".join(c.text for c in chunks1)

        assert engine._checkpoint is not None
        assert len(engine._cached_tokens) > 0
        assert engine._last_logits is not None

        # model.prefill이 2번째 호출에서 예외를 발생시키도록 설정
        orig_prefill = model.prefill
        prefill_call_count = 0

        def failing_prefill(chunk_tokens, *, logits_positions=None, chunk=None):
            nonlocal prefill_call_count
            prefill_call_count += 1
            if prefill_call_count == 2:
                raise RuntimeError("simulated model.prefill failure on chunk 2")
            return orig_prefill(chunk_tokens, logits_positions=logits_positions, chunk=chunk)

        model.prefill = failing_prefill

        messages_turn2 = list(messages_turn1)
        messages_turn2.append({"role": "assistant", "content": text1})
        messages_turn2.append({
            "role": "user",
            "content": "Explain prefill chunk scheduling in detail with examples and diagrams.",
        })

        with pytest.raises(RuntimeError, match="simulated model.prefill failure on chunk 2"):
            list(engine.chat(messages_turn2, options=opts))

        assert prefill_call_count == 2
        assert engine._cached_tokens == []
        assert engine._last_logits is None
        assert engine._checkpoint is None
        assert len(model.conv_states) == 0
        assert len(backend.rda_states) == 0
        assert len(backend.mla_key_cache) == 0
        assert len(backend.mla_value_cache) == 0
    finally:
        if old_env is None:
            os.environ.pop("RAON_PREFILL_CHUNK", None)
        else:
            os.environ["RAON_PREFILL_CHUNK"] = old_env


def test_prefill_cancellation_between_chunks(tiny_raon_model):
    """(다) 묶음 사이 취소 검증:
    첫 묶음 뒤 is_cancelled가 참이 되면 finish_reason='cancelled' 청크로 끝나고,
    _cached_tokens에는 끝난 묶음의 토큰만 들어 있다."""
    _, host_cfg, weights, _ = tiny_raon_model

    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    tokenizer = TinyFakeTokenizer(vocab_size=host_cfg.vocab_size)
    engine = Engine(weights_dir=None, model_dir=tokenizer, backend=backend, model=model)

    old_env = os.environ.get("RAON_PREFILL_CHUNK")
    try:
        os.environ["RAON_PREFILL_CHUNK"] = "16"
        opts = GenOptions(max_tokens=4, temperature=0.0)

        # 29 토큰 프롬프트 -> segments: [(16, []), (8, [7]), (5, [4])]
        messages = [{
            "role": "user",
            "content": "Explain prefill chunk scheduling in detail with examples and diagrams.",
        }]
        prompt_ids = tokenizer.chat_ids(messages, add_generation_prompt=True)
        assert len(prompt_ids) > 16

        # 첫 묶음(16개)이 완료되면 len(engine._cached_tokens) >= 16이 되므로 그때 cancel=True
        cancel_fn = lambda: len(engine._cached_tokens) >= 16

        chunks = list(engine.chat(messages, options=opts, cancel=cancel_fn))

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.done is True
        assert chunk.finish_reason == "cancelled"
        assert chunk.completion_tokens == 0
        assert chunk.prompt_tokens == len(prompt_ids)

        # _cached_tokens에는 끝난 첫 번째 묶음의 16개 토큰만 들어 있어야 함
        assert len(engine._cached_tokens) == 16
        assert engine._cached_tokens == prompt_ids[:16]
    finally:
        if old_env is None:
            os.environ.pop("RAON_PREFILL_CHUNK", None)
        else:
            os.environ["RAON_PREFILL_CHUNK"] = old_env

