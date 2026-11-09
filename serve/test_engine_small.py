# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for serve/tokenize.py and serve/engine.py (without NPU hardware)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from serve.tokenize import DEFAULT_EOS_IDS, SPECIAL_TOKENS, IncrementalDecoder, RaonTokenizer


def _load_real_engine_module():
    """Load serve/engine.py directly so test_server_small.py's sys.modules stub cannot shadow it."""
    engine_path = Path(__file__).resolve().parent / "engine.py"
    spec = importlib.util.spec_from_file_location("_real_serve_engine", engine_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_real_serve_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


_engine_mod = _load_real_engine_module()
Engine = _engine_mod.Engine
GenOptions = _engine_mod.GenOptions
Chunk = _engine_mod.Chunk
sample_token = _engine_mod.sample_token

# Real-tokenizer / golden-manifest tests read $RAON_MODEL_DIR (HF checkpoint dir with
# tokenizer.json) and $RAON_GOLDEN_DIR (golden run dir with manifest.json), the same
# variables host/check_golden.py uses, and skip with a reason when they are unset or
# the files are absent.
_MODEL_DIR_ENV = os.environ.get("RAON_MODEL_DIR")
_GOLDEN_DIR_ENV = os.environ.get("RAON_GOLDEN_DIR")
HF_MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else None
GOLDEN_MANIFEST = Path(_GOLDEN_DIR_ENV) / "manifest.json" if _GOLDEN_DIR_ENV else None


@pytest.fixture(scope="module")
def tokenizer() -> RaonTokenizer:
    if HF_MODEL_DIR is None or not (HF_MODEL_DIR / "tokenizer.json").is_file():
        pytest.skip(
            "Raon HF checkpoint with tokenizer.json not found "
            f"(RAON_MODEL_DIR={_MODEL_DIR_ENV!r}); set RAON_MODEL_DIR to the checkpoint directory"
        )
    return RaonTokenizer(HF_MODEL_DIR)


# --------------------------------------------------------------------------- #
# 가짜 모델 / 가짜 토크나이저 / 가짜 백엔드
# --------------------------------------------------------------------------- #


class FakeModel:
    """Small deterministic stand-in for RaonModel."""

    def __init__(
        self,
        vocab_size: int = 260000,
        logits_fn: Callable[[int, list[int]], np.ndarray] | None = None,
        token_sequence: list[int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.logits_fn = logits_fn
        self.token_sequence = list(token_sequence) if token_sequence is not None else []
        self.step_tokens: list[int] = []
        self.reset_count: int = 0
        self.decode_cursor: int = 0

    def reset_state(self) -> None:
        self.step_tokens.clear()
        self.reset_count += 1

    def step(self, token_id: int) -> np.ndarray:
        self.step_tokens.append(int(token_id))
        if self.logits_fn is not None:
            return self.logits_fn(int(token_id), self.step_tokens)
        logits = np.full(self.vocab_size, -10.0, dtype=np.float32)
        if self.decode_cursor < len(self.token_sequence):
            target = self.token_sequence[self.decode_cursor]
            self.decode_cursor += 1
        else:
            target = DEFAULT_EOS_IDS[0]
        logits[int(target)] = 10.0
        return logits


class StatefulFakeBackend:
    """Fake backend implementing the get/set RDA state and MLA cache contract."""

    def __init__(self, num_layers: int = 4, rda_layers: set[int] | None = None) -> None:
        self.num_layers = num_layers
        self.rda_layers = rda_layers if rda_layers is not None else {0, 1, 2}
        self.rda_states: dict[int, np.ndarray] = {}
        self.mla_k: dict[int, np.ndarray] = {}
        self.mla_v: dict[int, np.ndarray] = {}
        self.step_id: int = 0
        self.get_rda_calls: int = 0
        self.set_rda_calls: int = 0
        self.get_mla_calls: int = 0
        self.set_mla_calls: int = 0

    def set_step(self, step: int) -> None:
        self.step_id = step

    def reset(self) -> None:
        self.rda_states.clear()
        self.mla_k.clear()
        self.mla_v.clear()
        self.step_id = 0

    def get_rda_state(self, layer: int) -> np.ndarray | None:
        self.get_rda_calls += 1
        st = self.rda_states.get(layer)
        return None if st is None else st.copy()[np.newaxis, ...]

    def set_rda_state(self, layer: int, state: np.ndarray) -> None:
        self.set_rda_calls += 1
        st = state.squeeze(0) if state.ndim == 4 else state
        self.rda_states[layer] = np.array(st, dtype=np.float32, copy=True)

    def get_mla_cache(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.get_mla_calls += 1
        k = self.mla_k.get(layer)
        v = self.mla_v.get(layer)
        if k is None or v is None:
            return None, None
        return k[np.newaxis, ...], v[np.newaxis, ...]

    def set_mla_cache(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        self.set_mla_calls += 1
        k = key.squeeze(0) if key.ndim == 4 else key
        v = value.squeeze(0) if value.ndim == 4 else value
        self.mla_k[layer] = np.array(k, dtype=np.uint16, copy=True)
        self.mla_v[layer] = np.array(v, dtype=np.uint16, copy=True)


class StatefulFakeModel:
    """Deterministic model that mutates RDA states, conv_states, and MLA caches on every token step."""

    def __init__(
        self,
        backend: StatefulFakeBackend,
        vocab_size: int = 260000,
        script_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.backend = backend
        self.vocab_size = vocab_size
        self.script_fn = script_fn
        self.config = type("Cfg", (), {"num_hidden_layers": backend.num_layers})()
        self.conv_states: dict[int, dict[str, np.ndarray]] = {}
        self.reset_count: int = 0
        self.step_history: list[int] = []

    def reset_state(self) -> None:
        self.conv_states.clear()
        self.backend.reset()
        self.step_history.clear()
        self.reset_count += 1

    def step(self, token_id: int) -> np.ndarray:
        tid = int(token_id)
        self.step_history.append(tid)
        val = float((tid % 997) + 1)
        for l in range(self.backend.num_layers):
            if l in self.backend.rda_layers:
                prev_rda = self.backend.rda_states.get(l)
                if prev_rda is None:
                    prev_rda = np.zeros((2, 4, 4), dtype=np.float32)
                self.backend.rda_states[l] = prev_rda * 0.9 + val * (l + 1)

                lconv = self.conv_states.setdefault(l, {})
                for kind in ("q", "k", "v"):
                    prev_c = lconv.get(kind)
                    if prev_c is None:
                        prev_c = np.zeros((4, 3), dtype=np.float32)
                    next_col = np.full((4, 1), val + l, dtype=np.float32)
                    lconv[kind] = np.concatenate([prev_c[:, 1:], next_col], axis=-1)
            else:
                k_tok = np.full((2, 1, 4), tid & 0xFFFF, dtype=np.uint16)
                v_tok = np.full((2, 1, 4), (tid + l) & 0xFFFF, dtype=np.uint16)
                if l not in self.backend.mla_k:
                    self.backend.mla_k[l] = k_tok
                    self.backend.mla_v[l] = v_tok
                else:
                    self.backend.mla_k[l] = np.concatenate([self.backend.mla_k[l], k_tok], axis=1)
                    self.backend.mla_v[l] = np.concatenate([self.backend.mla_v[l], v_tok], axis=1)

        logits = np.full(self.vocab_size, -50.0, dtype=np.float32)
        target = self.script_fn(tid) if self.script_fn is not None else DEFAULT_EOS_IDS[0]
        logits[int(target)] = 20.0
        return logits


class FakeTokenizer:
    """Lightweight deterministic fake tokenizer for unit testing without HF files."""

    eos_ids = [259202, 259200]

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = dict(SPECIAL_TOKENS)
        self._tok2id["\n"] = 10
        self._tok2id["\n\n"] = 269
        self._id2tok: dict[int, str] = {v: k for k, v in self._tok2id.items()}
        self._next_id = 1000

    def _intern(self, word: str) -> int:
        if word not in self._tok2id:
            tid = self._next_id
            self._next_id += 1
            self._tok2id[word] = tid
            self._id2tok[tid] = word
        return self._tok2id[word]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        # Split keeping special tokens and newlines intact
        pattern = r"(<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>|<think>|</think>|\n\n|\n|[^\s<\n]+|[ \t]+)"
        import re
        tokens = re.findall(pattern, text)
        return [self._intern(t) for t in tokens]

    def decode(self, ids: Iterable[int] | int, *, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        parts: list[str] = []
        for tid in ids:
            tid_int = int(tid)
            if skip_special_tokens and tid_int in SPECIAL_TOKENS.values():
                continue
            parts.append(self._id2tok.get(tid_int, f"<{tid_int}>"))
        return "".join(parts)

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        out: list[str] = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            reasoning = m.get("reasoning_content") or ""
            if role == "assistant":
                if reasoning:
                    out.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n")
                else:
                    out.append(f"<|im_start|>assistant\n<think></think>\n\n{content}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            if enable_thinking is False:
                out.append("<|im_start|>assistant\n<think></think>\n\n")
            else:
                out.append("<|im_start|>assistant\n<think>\n")
        return "".join(out)

    def chat_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        return self.encode(
            self.render_chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=add_generation_prompt,
            )
        )


# --------------------------------------------------------------------------- #
# (가) 골든 prompt -> chat_ids 가 prompt_token_ids 와 같음
# --------------------------------------------------------------------------- #


def test_golden_prompt_chat_ids_match(tokenizer: RaonTokenizer) -> None:
    if GOLDEN_MANIFEST is None or not GOLDEN_MANIFEST.is_file():
        pytest.skip(
            f"golden manifest.json not found (RAON_GOLDEN_DIR={_GOLDEN_DIR_ENV!r}); "
            "set RAON_GOLDEN_DIR to the golden run directory"
        )
    manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    prompt = manifest["prompt"]
    expected_ids = manifest["prompt_token_ids"]

    messages = [{"role": "user", "content": prompt}]
    actual_ids = tokenizer.chat_ids(messages, add_generation_prompt=True)

    assert len(expected_ids) == 20
    assert actual_ids == expected_ids
    assert tokenizer.eos_ids == [259202, 259200]

    for tok_str, tok_id in SPECIAL_TOKENS.items():
        assert tokenizer.encode(tok_str) == [tok_id]


def test_chat_template_options_and_tools(tokenizer: RaonTokenizer) -> None:
    msgs = [{"role": "user", "content": "안녕"}]
    s_think = tokenizer.render_chat(msgs, enable_thinking=True, add_generation_prompt=True)
    assert s_think.endswith("<|im_start|>assistant\n<think>\n")

    s_no_think = tokenizer.render_chat(msgs, enable_thinking=False, add_generation_prompt=True)
    assert s_no_think.endswith("<|im_start|>assistant\n<think></think>\n\n")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    s_tools = tokenizer.render_chat(msgs, tools=tools, add_generation_prompt=True)
    assert "<tools>" in s_tools
    assert '"name": "get_weather"' in s_tools

    with pytest.raises(Exception):
        tokenizer.render_chat([])


def test_openai_tool_history_renders_with_real_template(tokenizer: RaonTokenizer) -> None:
    from serve.server import Handler

    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "checking",
         "tool_calls": [{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"city\": \"Seoul\"}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "summarize"},
    ]
    rendered = tokenizer.render_chat(
        Handler._normalize_openai_messages(messages), add_generation_prompt=True
    )

    assert "<think>\nchecking\n</think>" in rendered
    assert "<function=lookup>" in rendered
    assert "<parameter=city>\nSeoul\n</parameter>" in rendered
    assert "<tool_response>\nsunny\n</tool_response>" in rendered
    assert "summarize" in rendered


def test_ollama_tool_history_renders_with_real_template(tokenizer: RaonTokenizer) -> None:
    from serve.server import Handler

    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "thinking": "checking", "tool_calls": [{
            "function": {"name": "lookup", "arguments": {"city": "Seoul"}},
        }]},
        {"role": "tool", "tool_name": "lookup", "content": "sunny"},
        {"role": "user", "content": "summarize"},
    ]
    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object"}
    }}]
    rendered = tokenizer.render_chat(
        Handler._normalize_ollama_messages(messages), tools=tools, add_generation_prompt=True
    )

    assert "<tools>" in rendered
    assert "<think>\nchecking\n</think>" in rendered
    assert "checking" in rendered
    assert "<function=lookup>" in rendered
    assert "<parameter=city>\nSeoul\n</parameter>" in rendered
    assert "<tool_response>\nsunny\n</tool_response>" in rendered
    assert "summarize" in rendered
    assert messages[1]["thinking"] == "checking"
    assert "reasoning_content" not in messages[1]


@pytest.mark.parametrize("api_type", ["openai", "ollama"])
def test_chat_accepted_fields_rendered_or_metadata(
    api_type: str, tokenizer: RaonTokenizer
) -> None:
    """검증기가 받는 모든 필드가 템플릿에 렌더되거나 렌더되지 않는 메타데이터 목록에 포함됨을 검증."""
    from serve.server import (
        CHAT_FREE_FORM_VALUE_PATHS,
        Handler,
        OPENAI_ACCEPTED_FIELDS,
        OLLAMA_ACCEPTED_FIELDS,
    )
    from serve.chat_contract_test_utils import (
        build_all_fields_messages,
        chat_used_fields,
    )

    is_ollama = api_type == "ollama"
    accepted_fields = OLLAMA_ACCEPTED_FIELDS if is_ollama else OPENAI_ACCEPTED_FIELDS

    if is_ollama:
        unrendered_metadata = {
            "tool_name",
        }
        rendered_fields = {
            "role",
            "content",
            "content.type",
            "content.text",
            "thinking",
            "tool_calls",
            "tool_call.function.name",
            "tool_call.function.arguments",
        }
    else:
        unrendered_metadata = {
            "loss",
            "tool_call_id",
            "tool_call.index",
            "tool_call.id",
            "tool_call.type",
        }
        rendered_fields = {
            "role",
            "content",
            "content.type",
            "content.text",
            "reasoning_content",
            "tool_calls",
            "tool_call.function.name",
            "tool_call.function.arguments",
        }

    # 검증기가 받는 필드 집합 = 렌더되는 필드 ∪ 메타데이터 목록
    assert accepted_fields == rendered_fields | unrendered_metadata
    assert not (rendered_fields & unrendered_metadata), "렌더 필드와 메타데이터는 서로소여야 함"

    # 검증기가 받는 모든 필드를 사용하는 대화 이력 작성
    sys_content = "지침: 안전하고 정확하게 답하십시오."
    user1_content = "현재 서울 날씨와 시간 알려줘."
    asst_thought = "서울 날씨 확인을 위해 check_weather 도구를 먼저 호출해야 한다."
    asst_content = "서울 날씨를 조회하겠습니다."
    tool_name = "check_weather"
    tool_resp_content = '{"status": "ok", "temp": 18}'
    user2_content = "결과 고마워."

    messages = build_all_fields_messages(api_type)

    # 대화 이력이 검증기가 받는 모든 필드를 사용했는지 검증
    used_fields = chat_used_fields(messages, CHAT_FREE_FORM_VALUE_PATHS[api_type])
    assert used_fields == accepted_fields, f"대화 이력에 빠진 필드가 있음: {accepted_fields - used_fields}"

    # 서버의 실제 정규화 경로(검증 → 정규화 함수)
    handler = object.__new__(Handler)
    handler._invalid = lambda message, param: False
    assert handler._validate_messages(messages, ollama=is_ollama) is True

    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "parameters": {"type": "object"},
        },
    }]

    if is_ollama:
        normalized = Handler._normalize_ollama_messages(messages)
    else:
        normalized = Handler._normalize_openai_messages(messages)

    # 실제 템플릿으로 렌더
    rendered = tokenizer.render_chat(normalized, tools=tools, add_generation_prompt=True)

    # 내용을 담는 모든 값이 렌더 결과의 맞는 자리에 있는지 검사
    pos_sys = rendered.find(sys_content)
    assert pos_sys != -1, f"system content {sys_content!r} 누락"

    pos_user1 = rendered.find(user1_content)
    assert pos_user1 != -1, f"user 1 content {user1_content!r} 누락"
    assert pos_sys < pos_user1

    pos_asst_tag = rendered.find("<|im_start|>assistant\n", pos_user1)
    assert pos_asst_tag != -1
    pos_asst_end = rendered.find("<|im_end|>", pos_asst_tag)
    assert pos_asst_end != -1
    asst_body = rendered[pos_asst_tag:pos_asst_end]

    # 생각 글은 assistant 차례의 <think>...</think> 안
    assert f"<think>\n{asst_thought}\n</think>" in asst_body, (
        f"assistant 차례에 생각 글이 <think>...</think> 태그 안에 없음: {asst_body!r}"
    )
    assert asst_content in asst_body, f"assistant content {asst_content!r} 누락"
    assert f"<function={tool_name}>" in asst_body, f"도구 이름 {tool_name!r} 누락"
    assert "<parameter=location>\nSeoul\n</parameter>" in asst_body
    assert "<parameter=format>\ncelsius\n</parameter>" in asst_body

    think_idx = asst_body.find("<think>")
    content_idx = asst_body.find(asst_content)
    tool_idx = asst_body.find("<tool_call>")
    assert think_idx < content_idx < tool_idx, "assistant 내부 순서가 올바르지 않음"

    pos_tool_resp = rendered.find(
        f"<tool_response>\n{tool_resp_content}\n</tool_response>", pos_asst_end
    )
    assert pos_tool_resp != -1, f"도구 응답 {tool_resp_content!r} 누락"

    pos_user2 = rendered.find(f"<|im_start|>user\n{user2_content}<|im_end|>", pos_tool_resp)
    assert pos_user2 != -1, f"그다음 user content {user2_content!r} 누락"


# --------------------------------------------------------------------------- #
# (나) 한국어·이모지 IncrementalDecoder 가 전체 decode 와 같음
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sample_text",
    [
        "대한민국의 수도는 서울입니다. 반갑습니다!",
        "갹굠뷁쀍 한글 희귀 음절과 이모지 🦊🧑‍💻🇰🇷 😀🎉✨ 조합 테스트!",
        "<think>\n생각을 해봅시다: 갹굠🦊\n</think>\n\n정답은 서울입니다.",
        "Mixed English, 한국어, 日本語, and emojis 🚀🔥💡!",
    ],
)
def test_incremental_decoder_korean_and_emoji(tokenizer: RaonTokenizer, sample_text: str) -> None:
    token_ids = tokenizer.encode(sample_text)
    assert len(token_ids) > 1

    full_decoded = tokenizer.decode(token_ids)
    assert full_decoded == sample_text

    inc = tokenizer.incremental_decoder()
    pieces: list[str] = []
    for tid in token_ids:
        piece = inc.step(tid)
        # Incomplete UTF-8 bytes must never leak as replacement characters
        assert "\ufffd" not in piece
        pieces.append(piece)
    pieces.append(inc.flush())

    assert "".join(pieces) == full_decoded
    assert inc.text == full_decoded


def test_incremental_decoder_holds_split_multibyte_token(tokenizer: RaonTokenizer) -> None:
    # '갹' splits into 2 byte-level tokens [346, 185] in Raon's byte BPE
    ids = tokenizer.encode("갹")
    assert len(ids) >= 2
    inc = IncrementalDecoder(tokenizer)
    first_out = inc.step(ids[0])
    assert first_out == ""
    rest_out = "".join(inc.step(tid) for tid in ids[1:]) + inc.flush()
    assert first_out + rest_out == "갹"


# --------------------------------------------------------------------------- #
# (다) 가짜 모델로 greedy·seed 재현·top_k/top_p·presence_penalty·eos·stop·max_tokens
# --------------------------------------------------------------------------- #


def test_sampling_greedy_and_seed_reproducibility(tokenizer: RaonTokenizer) -> None:
    vocab_size = 1000
    base_logits = np.linspace(-5.0, 5.0, vocab_size, dtype=np.float32)
    base_logits[123] = 12.0
    base_logits[456] = 11.8
    base_logits[789] = 11.5

    # Greedy (temperature=0) picks argmax (123) deterministically
    rng = np.random.default_rng(0)
    picked = sample_token(base_logits, [], GenOptions(temperature=0.0, presence_penalty=0.0), rng)
    assert picked == 123

    # Seed reproducibility under stochastic sampling
    def make_model() -> FakeModel:
        def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
            arr = np.full(260000, -20.0, dtype=np.float32)
            # Give 10 candidate tokens similar logits
            candidates = [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]
            for c_idx, cid in enumerate(candidates):
                arr[cid] = 5.0 - 0.05 * c_idx
            return arr

        return FakeModel(logits_fn=logits_fn)

    eng1 = Engine(None, tokenizer, model=make_model())
    eng2 = Engine(None, tokenizer, model=make_model())
    eng3 = Engine(None, tokenizer, model=make_model())

    opts_s1 = GenOptions(max_tokens=12, temperature=1.0, top_k=10, top_p=0.95, seed=2026)
    opts_s2 = GenOptions(max_tokens=12, temperature=1.0, top_k=10, top_p=0.95, seed=2026)
    opts_diff = GenOptions(max_tokens=12, temperature=1.0, top_k=10, top_p=0.95, seed=9999)

    out1 = "".join(c.text for c in eng1.generate("테스트", raw=True, options=opts_s1))
    out2 = "".join(c.text for c in eng2.generate("테스트", raw=True, options=opts_s2))
    out3 = "".join(c.text for c in eng3.generate("테스트", raw=True, options=opts_diff))

    assert out1 == out2
    assert out1 != out3


def test_sampling_top_k_and_top_p() -> None:
    logits = np.full(100, -100.0, dtype=np.float32)
    # Token 10: logit 2.0, Token 20: logit 1.0, Token 30: logit 0.0, Token 40: logit -1.0
    logits[10] = 2.0
    logits[20] = 1.0
    logits[30] = 0.0
    logits[40] = -1.0

    # top_k=2 -> only tokens 10 and 20 can ever be sampled
    rng = np.random.default_rng(7)
    opts_topk = GenOptions(temperature=1.0, top_k=2, top_p=1.0, presence_penalty=0.0)
    sampled_topk = {sample_token(logits, [], opts_topk, rng) for _ in range(100)}
    assert sampled_topk <= {10, 20}
    assert sampled_topk == {10, 20}

    # top_p=0.6 -> softmax([2.0, 1.0, 0.0, -1.0]) is approx [0.6439, 0.2369, 0.0871, 0.0321]
    # Since token 10 alone has probability 0.6439 >= 0.6, top_p=0.6 keeps ONLY token 10!
    opts_topp = GenOptions(temperature=1.0, top_k=10, top_p=0.6, presence_penalty=0.0)
    sampled_topp = {sample_token(logits, [], opts_topp, rng) for _ in range(50)}
    assert sampled_topp == {10}


def test_sampling_presence_penalty(tokenizer: RaonTokenizer) -> None:
    tok_a = tokenizer.encode("가")[0]
    tok_b = tokenizer.encode("나")[0]

    def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
        arr = np.full(260000, -50.0, dtype=np.float32)
        arr[tok_a] = 10.0
        arr[tok_b] = 9.0
        return arr

    model = FakeModel(logits_fn=logits_fn)
    engine = Engine(None, tokenizer, model=model)
    # Step 1: tok_a (10.0 > 9.0) is chosen.
    # Step 2: tok_a gets penalized to 10.0 - 1.5 = 8.5 < 9.0, so tok_b (9.0) is chosen!
    opts = GenOptions(max_tokens=2, temperature=0.0, presence_penalty=1.5)
    chunks = list(engine.generate("시작", raw=True, options=opts))
    text = "".join(c.text for c in chunks)
    assert text == tokenizer.decode([tok_a, tok_b])
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "length"


def test_stop_on_eos_and_stop_strings_and_max_tokens(tokenizer: RaonTokenizer) -> None:
    hello_ids = tokenizer.encode("안녕하세요 반갑습니다")
    assert len(hello_ids) >= 2

    # 1. EOS stop
    seq_eos = [hello_ids[0], hello_ids[1], DEFAULT_EOS_IDS[0], hello_ids[0]]
    prompt_ids = tokenizer.encode("질문")
    n_prompt = len(prompt_ids)

    def eos_logits_fn(_tid: int, hist: list[int]) -> np.ndarray:
        arr = np.full(260000, -50.0, dtype=np.float32)
        decode_idx = len(hist) - n_prompt
        if 0 <= decode_idx < len(seq_eos):
            arr[seq_eos[decode_idx]] = 20.0
        else:
            arr[DEFAULT_EOS_IDS[0]] = 20.0
        return arr

    eng_eos = Engine(None, tokenizer, model=FakeModel(logits_fn=eos_logits_fn))
    chunks_eos = list(eng_eos.generate("질문", raw=True, options=GenOptions(max_tokens=10, temperature=0.0, presence_penalty=0.0)))
    assert "".join(c.text for c in chunks_eos) == tokenizer.decode(hello_ids[:2])
    assert chunks_eos[-1].done is True
    assert chunks_eos[-1].finish_reason == "stop"
    assert chunks_eos[-1].completion_tokens == 2

    # 2. Stop string (cut from output across multi-token boundary)
    stop_sample_ids = tokenizer.encode("서울입니다. STOP 뒤에 오는 말")

    def stop_logits_fn(_tid: int, hist: list[int]) -> np.ndarray:
        arr = np.full(260000, -50.0, dtype=np.float32)
        decode_idx = len(hist) - n_prompt
        if 0 <= decode_idx < len(stop_sample_ids):
            arr[stop_sample_ids[decode_idx]] = 20.0
        else:
            arr[DEFAULT_EOS_IDS[0]] = 20.0
        return arr

    eng_stop = Engine(None, tokenizer, model=FakeModel(logits_fn=stop_logits_fn))
    opts_stop = GenOptions(max_tokens=20, temperature=0.0, presence_penalty=0.0, stop=["STOP"])
    chunks_stop = list(eng_stop.generate("질문", raw=True, options=opts_stop))
    full_stop_text = "".join(c.text for c in chunks_stop)
    assert full_stop_text == "서울입니다. "
    assert "STOP" not in full_stop_text
    assert chunks_stop[-1].done is True
    assert chunks_stop[-1].finish_reason == "stop"

    # 3. max_tokens limit
    eng_len = Engine(None, tokenizer, model=FakeModel(logits_fn=stop_logits_fn))
    opts_len = GenOptions(max_tokens=3, temperature=0.0, presence_penalty=0.0)
    chunks_len = list(eng_len.generate("질문", raw=True, options=opts_len))
    assert chunks_len[-1].done is True
    assert chunks_len[-1].finish_reason == "length"
    assert chunks_len[-1].completion_tokens == 3
    assert "".join(c.text for c in chunks_len) == tokenizer.decode(stop_sample_ids[:3])


# --------------------------------------------------------------------------- #
# (라) 생각과 답 나누기 (토큰 259214 </think>, 생각 안 max_tokens, 생각 끔)
# --------------------------------------------------------------------------- #


def test_thinking_and_content_splitting_with_fake_tokenizer() -> None:
    fake_tok = FakeTokenizer()
    gen_body_ids = fake_tok.encode("단계별 생각\n</think>\n\n최종 답변입니다.")
    assert SPECIAL_TOKENS["</think>"] in gen_body_ids

    msgs = [{"role": "user", "content": "질문"}]
    prompt_think_ids = fake_tok.chat_ids(msgs, enable_thinking=True, add_generation_prompt=True)

    script_ids: list[int] = []
    cursor = 0

    def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
        nonlocal cursor
        arr = np.full(260000, -50.0, dtype=np.float32)
        if cursor < len(script_ids):
            arr[script_ids[cursor]] = 20.0
            cursor += 1
        else:
            arr[DEFAULT_EOS_IDS[0]] = 20.0
        return arr

    # 1. 기본(생각 켬): </think> 앞은 thinking, 뒤(앞쪽 빈 줄 제거)는 text
    engine = Engine(None, fake_tok, model=FakeModel(logits_fn=logits_fn))
    script_ids = [0] * (len(prompt_think_ids) - 1) + gen_body_ids + [DEFAULT_EOS_IDS[0]]
    cursor = 0
    chunks = list(engine.chat(msgs, options=GenOptions(max_tokens=30, temperature=0.0, presence_penalty=0.0)))
    assert "".join(c.thinking for c in chunks) == "단계별 생각\n"
    assert "".join(c.text for c in chunks) == "최종 답변입니다."
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"

    # 2. max_tokens가 생각 안에서 끝날 때: text는 비고 finish_reason == "length"
    engine_short = Engine(None, fake_tok, model=FakeModel(logits_fn=logits_fn))
    script_ids = [0] * (len(prompt_think_ids) - 1) + gen_body_ids + [DEFAULT_EOS_IDS[0]]
    cursor = 0
    chunks_short = list(engine_short.chat(msgs, options=GenOptions(max_tokens=2, temperature=0.0, presence_penalty=0.0)))
    assert "".join(c.thinking for c in chunks_short) != ""
    assert "".join(c.text for c in chunks_short) == ""
    assert chunks_short[-1].done is True
    assert chunks_short[-1].finish_reason == "length"

    # 3. 생각 끔 (enable_thinking=False): 템플릿이 <think></think>\n\n을 넣고 전부 text로 나감
    prompt_nothink_ids = fake_tok.chat_ids(msgs, enable_thinking=False, add_generation_prompt=True)
    ans_only_ids = fake_tok.encode("바로 답변합니다.")
    engine_nothink = Engine(None, fake_tok, model=FakeModel(logits_fn=logits_fn))
    script_ids = [0] * (len(prompt_nothink_ids) - 1) + ans_only_ids + [DEFAULT_EOS_IDS[0]]
    cursor = 0
    chunks_nothink = list(
        engine_nothink.chat(
            msgs,
            enable_thinking=False,
            options=GenOptions(max_tokens=20, temperature=0.0, presence_penalty=0.0),
        )
    )
    assert "".join(c.thinking for c in chunks_nothink) == ""
    assert "".join(c.text for c in chunks_nothink) == "바로 답변합니다."
    assert chunks_nothink[-1].finish_reason == "stop"


# --------------------------------------------------------------------------- #
# (마) 접두사 재사용 및 대화 이어 쓰기 지점(프롬프트 저장 지점 복원)
# --------------------------------------------------------------------------- #


def test_prefix_reuse_and_reset_on_mismatch(tokenizer: RaonTokenizer) -> None:
    # Note: chat_ids with add_generation_prompt=True already ends with "<think>\n",
    # so the model generates the tokens AFTER "<think>\n": "생각1\n</think>\n\n답변1"
    gen_body1_ids = tokenizer.encode("생각1\n</think>\n\n답변1")
    gen_body2_ids = tokenizer.encode("생각2\n</think>\n\n답변2")

    script_ids: list[int] = []
    cursor = 0

    def scripted_logits(_tid: int, _hist: list[int]) -> np.ndarray:
        nonlocal cursor
        arr = np.full(260000, -50.0, dtype=np.float32)
        if cursor < len(script_ids):
            arr[script_ids[cursor]] = 20.0
            cursor += 1
        else:
            arr[DEFAULT_EOS_IDS[0]] = 20.0
        return arr

    model = FakeModel(logits_fn=scripted_logits)
    engine = Engine(None, tokenizer, model=model)

    # Turn 1: single user message
    msgs_turn1 = [{"role": "user", "content": "첫 번째 질문입니다."}]
    prompt1_ids = tokenizer.chat_ids(msgs_turn1, add_generation_prompt=True)
    script_ids = [0] * (len(prompt1_ids) - 1) + gen_body1_ids + [DEFAULT_EOS_IDS[0]]
    cursor = 0

    chunks1 = list(
        engine.chat(
            msgs_turn1,
            options=GenOptions(max_tokens=30, temperature=0.0, presence_penalty=0.0),
        )
    )
    thinking1 = "".join(c.thinking for c in chunks1)
    text1 = "".join(c.text for c in chunks1)
    assert thinking1 == "생각1\n"
    assert text1 == "답변1"
    assert chunks1[-1].reused_prefix_tokens == 0
    assert chunks1[-1].prompt_tokens == len(prompt1_ids)
    assert chunks1[-1].completion_tokens == len(gen_body1_ids)
    total_t1_tokens = len(prompt1_ids) + len(gen_body1_ids)
    assert len(model.step_tokens) == total_t1_tokens

    # Turn 2A: exact full-token continuation (assistant includes reasoning_content)
    msgs_turn2_full = [
        {"role": "user", "content": "첫 번째 질문입니다."},
        {"role": "assistant", "content": text1, "reasoning_content": thinking1.strip()},
        {"role": "user", "content": "두 번째 질문입니다."},
    ]
    prompt2_ids = tokenizer.chat_ids(msgs_turn2_full, add_generation_prompt=True)
    assert prompt2_ids[:total_t1_tokens] == prompt1_ids + gen_body1_ids

    new_prefill_count = len(prompt2_ids) - total_t1_tokens
    assert new_prefill_count > 0

    script_ids = [0] * (new_prefill_count - 1) + gen_body2_ids + [DEFAULT_EOS_IDS[0]]
    cursor = 0
    steps_before_t2 = len(model.step_tokens)
    resets_before_t2 = model.reset_count

    chunks2 = list(
        engine.chat(
            msgs_turn2_full,
            options=GenOptions(max_tokens=30, temperature=0.0, presence_penalty=0.0),
        )
    )
    assert "".join(c.thinking for c in chunks2) == "생각2\n"
    assert "".join(c.text for c in chunks2) == "답변2"
    assert model.reset_count == resets_before_t2
    assert chunks2[-1].reused_prefix_tokens == total_t1_tokens
    steps_in_t2 = len(model.step_tokens) - steps_before_t2
    assert steps_in_t2 == new_prefill_count + len(gen_body2_ids)

    # Turn 3: completely different conversation -> must reset and prefill all tokens
    msgs_turn3 = [{"role": "user", "content": "완전히 새로운 대화입니다."}]
    prompt3_ids = tokenizer.chat_ids(msgs_turn3, add_generation_prompt=True)
    script_ids = [0] * (len(prompt3_ids) - 1) + [DEFAULT_EOS_IDS[0]]
    cursor = 0

    chunks3 = list(
        engine.chat(
            msgs_turn3,
            options=GenOptions(max_tokens=10, temperature=0.0, presence_penalty=0.0),
        )
    )
    assert model.reset_count == resets_before_t2 + 1
    assert chunks3[-1].reused_prefix_tokens == 0
    assert len(model.step_tokens) == len(prompt3_ids)


def test_checkpoint_reuse_when_assistant_thinking_stripped() -> None:
    """두 번째 턴에서 이전 assistant 메시지의 생각이 지워지고 `<think></think>\\n\\n+답`으로 그려질 때
    프롬프트 저장 지점(RDA 상태·합성곱 상태·MLA 캐시 길이)에서 정확히 이어 처리되는지 검증한다.
    """
    fake_tok = FakeTokenizer()
    gen1_ids = fake_tok.encode("긴생각1\n</think>\n\n짧은답1")
    gen2_ids = fake_tok.encode("긴생각2\n</think>\n\n짧은답2")

    msgs1 = [{"role": "user", "content": "안녕 반갑습니다 오늘 날씨는 어떤가요"}]
    prompt1_ids = fake_tok.chat_ids(msgs1, add_generation_prompt=True)
    base1_ids = fake_tok.chat_ids(msgs1, add_generation_prompt=False)
    gen_prompt_ids = fake_tok.encode("<|im_start|>assistant\n<think>\n")
    assert prompt1_ids == base1_ids + gen_prompt_ids

    # Turn 2: 클라이언트는 assistant의 답(content)만 돌려보내므로 템플릿이 <think></think>\n\n짧은답1 로 그림
    msgs2 = [
        {"role": "user", "content": "안녕 반갑습니다 오늘 날씨는 어떤가요"},
        {"role": "assistant", "content": "짧은답1"},
        {"role": "user", "content": "그럼 내일은요"},
    ]
    prompt2_ids = fake_tok.chat_ids(msgs2, add_generation_prompt=True)
    # 전체 토큰열(prompt1 + gen1)로는 시작하지 않지만, base1_ids 접두사로는 시작함!
    assert prompt2_ids[: len(prompt1_ids)] != prompt1_ids
    assert prompt2_ids[: len(base1_ids)] == base1_ids

    # 1) 이어 쓰기 엔진 실행 (Turn 1 -> Turn 2)
    be_reuse = StatefulFakeBackend()
    seq_iter1 = iter([0] * (len(prompt1_ids) - 1) + gen1_ids + [DEFAULT_EOS_IDS[0]])
    mdl_reuse = StatefulFakeModel(be_reuse, script_fn=lambda _t: next(seq_iter1, DEFAULT_EOS_IDS[0]))
    eng_reuse = Engine(None, fake_tok, model=mdl_reuse, backend=be_reuse)

    chunks1 = list(eng_reuse.chat(msgs1, options=GenOptions(max_tokens=20, temperature=0.0, presence_penalty=0.0)))
    assert "".join(c.thinking for c in chunks1) == "긴생각1\n"
    assert "".join(c.text for c in chunks1) == "짧은답1"
    assert be_reuse.get_rda_calls > 0
    assert be_reuse.get_mla_calls > 0

    # Turn 1 생성 후 MLA 캐시 길이는 len(prompt1_ids) + len(gen1_ids)까지 늘어났음
    assert be_reuse.mla_k[3].shape[1] == len(prompt1_ids) + len(gen1_ids)

    # Turn 2 실행: base1_ids 저장 지점 복원 -> prompt2_ids[len(base1_ids):] 만 step
    remaining_prefill2 = len(prompt2_ids) - len(base1_ids)
    seq_iter2 = iter([0] * (remaining_prefill2 - 1) + gen2_ids + [DEFAULT_EOS_IDS[0]])
    mdl_reuse.script_fn = lambda _t: next(seq_iter2, DEFAULT_EOS_IDS[0])
    resets_before_t2 = mdl_reuse.reset_count
    steps_before_t2 = len(mdl_reuse.step_history)

    chunks2 = list(eng_reuse.chat(msgs2, options=GenOptions(max_tokens=20, temperature=0.0, presence_penalty=0.0)))
    assert "".join(c.thinking for c in chunks2) == "긴생각2\n"
    assert "".join(c.text for c in chunks2) == "짧은답2"
    assert mdl_reuse.reset_count == resets_before_t2
    assert chunks2[-1].reused_prefix_tokens == len(base1_ids)
    assert be_reuse.set_rda_calls > 0
    assert be_reuse.set_mla_calls > 0
    assert len(mdl_reuse.step_history) - steps_before_t2 == remaining_prefill2 + len(gen2_ids)

    # 2) 처음부터 새로 돌린 기준 엔진(eng_fresh)과 RDA 상태·합성곱 상태·MLA 캐시 길이가 완전히 일치하는지 검증
    be_fresh = StatefulFakeBackend()
    seq_iter_fresh = iter([0] * (len(prompt2_ids) - 1) + gen2_ids + [DEFAULT_EOS_IDS[0]])
    mdl_fresh = StatefulFakeModel(be_fresh, script_fn=lambda _t: next(seq_iter_fresh, DEFAULT_EOS_IDS[0]))
    eng_fresh = Engine(None, fake_tok, model=mdl_fresh, backend=be_fresh)
    list(eng_fresh.chat(msgs2, options=GenOptions(max_tokens=20, temperature=0.0, presence_penalty=0.0)))

    for l in range(be_reuse.num_layers):
        if l in be_reuse.rda_layers:
            np.testing.assert_allclose(be_reuse.rda_states[l], be_fresh.rda_states[l])
            for kind in ("q", "k", "v"):
                np.testing.assert_allclose(mdl_reuse.conv_states[l][kind], mdl_fresh.conv_states[l][kind])
        else:
            np.testing.assert_array_equal(be_reuse.mla_k[l], be_fresh.mla_k[l])
            np.testing.assert_array_equal(be_reuse.mla_v[l], be_fresh.mla_v[l])

    # 3) 불일치 프롬프트(Turn 3)가 오면 reset_state() 후 처음부터 처리
    msgs3 = [{"role": "user", "content": "전혀 다른 주제의 새 대화"}]
    prompt3_ids = fake_tok.chat_ids(msgs3, add_generation_prompt=True)
    seq_iter3 = iter([0] * (len(prompt3_ids) - 1) + [DEFAULT_EOS_IDS[0]])
    mdl_reuse.script_fn = lambda _t: next(seq_iter3, DEFAULT_EOS_IDS[0])

    chunks3 = list(eng_reuse.chat(msgs3, options=GenOptions(max_tokens=5, temperature=0.0, presence_penalty=0.0)))
    assert mdl_reuse.reset_count == resets_before_t2 + 1
    assert chunks3[-1].reused_prefix_tokens == 0
    assert be_reuse.mla_k[3].shape[1] == len(prompt3_ids)


# --------------------------------------------------------------------------- #
# (바) cancel 동작, 동시성 직렬화, unload 검증
# --------------------------------------------------------------------------- #


def test_cancel_stops_before_next_token_and_keeps_state_synced(tokenizer: RaonTokenizer) -> None:
    tok_ids = tokenizer.encode("첫째 둘째 셋째 넷째 다섯째")
    assert len(tok_ids) >= 4

    def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
        arr = np.full(260000, -50.0, dtype=np.float32)
        arr[tok_ids[len(_hist) % len(tok_ids)]] = 20.0
        return arr

    model = FakeModel(logits_fn=logits_fn)
    engine = Engine(None, tokenizer, model=model)

    cancel_flag = False
    received: list[Chunk] = []

    for ch in engine.generate(
        "프롬프트",
        raw=True,
        options=GenOptions(max_tokens=10, temperature=0.0, presence_penalty=0.0),
        cancel=lambda: cancel_flag,
    ):
        received.append(ch)
        if len(received) == 2:
            cancel_flag = True

    assert len(received) == 3
    assert received[-1].done is True
    assert received[-1].finish_reason == "cancelled"
    assert received[-1].completion_tokens == 2

    # State and _cached_tokens must remain strictly in sync after cancellation
    assert engine._cached_tokens == model.step_tokens
    cached_after_cancel = list(engine._cached_tokens)

    # Continuing from the exact prefix after cancellation should reuse those tokens
    cont_text = tokenizer.decode(cached_after_cancel) + " 추가"
    cont_ids = tokenizer.encode(cont_text)
    assert cont_ids[: len(cached_after_cancel)] == cached_after_cancel
    new_count = len(cont_ids) - len(cached_after_cancel)

    steps_before = len(model.step_tokens)
    chunks_cont = list(
        engine.generate(
            cont_text,
            raw=True,
            options=GenOptions(max_tokens=1, temperature=0.0, presence_penalty=0.0),
        )
    )
    assert chunks_cont[-1].reused_prefix_tokens == len(cached_after_cancel)
    assert len(model.step_tokens) - steps_before == new_count + 1


def test_engine_lock_serializes_concurrent_calls_and_unload(tokenizer: RaonTokenizer) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        arr = np.full(260000, -50.0, dtype=np.float32)
        arr[DEFAULT_EOS_IDS[0]] = 20.0
        return arr

    engine = Engine(None, tokenizer, model=FakeModel(logits_fn=slow_logits_fn))
    errors: list[Exception] = []

    def worker(msg: str) -> None:
        try:
            list(engine.generate(msg, raw=True, options=GenOptions(max_tokens=2, temperature=0.0)))
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    assert max_active == 1
    assert engine.loaded is True
    engine.unload()
    assert engine.loaded is False
    assert engine.model is None
    assert engine.backend is None


# --------------------------------------------------------------------------- #
# (2a) 경계 의미 표: NUMERIC_OPTION_SPECS 의 모든 칸 × {아래, 위}
# --------------------------------------------------------------------------- #


def _check_temperature_lower() -> None:
    # 0 -> argmax
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float64)
    rng = np.random.default_rng(42)
    for _ in range(50):
        picked = sample_token(logits, [], GenOptions(temperature=0.0, presence_penalty=0.0), rng)
        assert picked == 1, f"temperature=0 must be greedy argmax (expected 1, got {picked})"


def _check_temperature_upper() -> None:
    # 2 -> 유한한 확률로 표본
    logits = np.array([2.0, 1.9, 1.8], dtype=np.float64)
    rng = np.random.default_rng(42)
    sampled = [
        sample_token(logits, [], GenOptions(temperature=2.0, top_k=0, top_p=1.0, presence_penalty=0.0), rng)
        for _ in range(100)
    ]
    assert len(set(sampled)) > 1, "temperature=2.0 must sample across multiple tokens with finite probability"
    assert all(0 <= t < 3 for t in sampled)


def _check_top_p_lower() -> None:
    # 0 -> argmax (keeps top-1 token)
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float64)
    rng = np.random.default_rng(42)
    # Under temperature=1.0, top_p=0.0 must filter all except top-1 (index 1)
    for _ in range(200):
        picked = sample_token(logits, [], GenOptions(temperature=1.0, top_p=0.0, presence_penalty=0.0), rng)
        assert picked == 1, f"top_p=0.0 must retain only top-1 token (expected 1, got {picked})"


def _check_top_p_upper() -> None:
    # 1 -> 거르지 않음(여러 토큰이 나올 수 있음)
    logits = np.array([2.0, 1.9, 1.8], dtype=np.float64)
    rng = np.random.default_rng(42)
    sampled = [
        sample_token(logits, [], GenOptions(temperature=1.0, top_k=0, top_p=1.0, presence_penalty=0.0), rng)
        for _ in range(100)
    ]
    assert len(set(sampled)) > 1, "top_p=1.0 must not filter logits"


def _check_top_k_lower() -> None:
    # 0 -> 거르지 않음
    logits = np.array([2.0, 1.9, 1.8], dtype=np.float64)
    rng = np.random.default_rng(42)
    sampled = [
        sample_token(logits, [], GenOptions(temperature=1.0, top_k=0, top_p=1.0, presence_penalty=0.0), rng)
        for _ in range(100)
    ]
    assert len(set(sampled)) > 1, "top_k=0 must not filter logits"


def _check_top_k_upper() -> None:
    # 어휘 크기 이상 -> 거르지 않음
    from serve.server import MAX_VOCAB_SIZE
    logits = np.array([2.0, 1.9, 1.8], dtype=np.float64)
    rng = np.random.default_rng(42)
    sampled = [
        sample_token(logits, [], GenOptions(temperature=1.0, top_k=MAX_VOCAB_SIZE, top_p=1.0, presence_penalty=0.0), rng)
        for _ in range(100)
    ]
    assert len(set(sampled)) > 1, "top_k >= vocab_size must not filter logits"


def _check_presence_penalty_lower() -> None:
    # -2 -> 이미 나온 토큰 점수에 그만큼 더함 (-(-2) = +2)
    logits = np.array([1.0, 2.0], dtype=np.float64)
    seen = [0]
    rng = np.random.default_rng(42)
    picked = sample_token(logits, seen, GenOptions(temperature=0.0, presence_penalty=-2.0), rng)
    assert picked == 0, f"presence_penalty=-2 must add 2.0 to seen token score (expected 0, got {picked})"


def _check_presence_penalty_upper() -> None:
    # 2 -> 이미 나온 토큰 점수에서 그만큼 뺌 (-2)
    logits = np.array([1.0, 2.0], dtype=np.float64)
    seen = [1]
    rng = np.random.default_rng(42)
    picked = sample_token(logits, seen, GenOptions(temperature=0.0, presence_penalty=2.0), rng)
    assert picked == 0, f"presence_penalty=2 must subtract 2.0 from seen token score (expected 0, got {picked})"


def _check_max_tokens_lower() -> None:
    # 0 -> 답 토큰 0개, finish_reason 은 지금 규칙대로
    engine = Engine(None, FakeTokenizer(), model=FakeModel())
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=0)))
    assert len(chunks) == 1
    assert chunks[0].completion_tokens == 0
    assert chunks[0].text == ""
    assert chunks[0].finish_reason == "length"


def _check_max_tokens_upper() -> None:
    # 상한값 -> 받아서 생성 시작(가짜 모델이 곧 EOS)
    from serve.server import MAX_POSITION_EMBEDDINGS
    engine = Engine(None, FakeTokenizer(), model=FakeModel(token_sequence=[10, 20]))
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=MAX_POSITION_EMBEDDINGS)))
    assert len(chunks) > 0
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"


def _check_max_completion_tokens_lower() -> None:
    # 0 -> 답 토큰 0개, finish_reason 은 지금 규칙대로
    engine = Engine(None, FakeTokenizer(), model=FakeModel())
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=0)))
    assert len(chunks) == 1
    assert chunks[0].completion_tokens == 0
    assert chunks[0].finish_reason == "length"


def _check_max_completion_tokens_upper() -> None:
    # 상한값 -> 받아서 생성 시작(가짜 모델이 곧 EOS)
    from serve.server import MAX_POSITION_EMBEDDINGS
    engine = Engine(None, FakeTokenizer(), model=FakeModel(token_sequence=[10, 20]))
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=MAX_POSITION_EMBEDDINGS)))
    assert len(chunks) > 0
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"


def _check_num_predict_lower() -> None:
    # 0 -> 답 토큰 0개, finish_reason 은 지금 규칙대로
    engine = Engine(None, FakeTokenizer(), model=FakeModel())
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=0)))
    assert len(chunks) == 1
    assert chunks[0].completion_tokens == 0
    assert chunks[0].finish_reason == "length"


def _check_num_predict_upper() -> None:
    # 상한값 -> 받아서 생성 시작(가짜 모델이 곧 EOS)
    from serve.server import MAX_POSITION_EMBEDDINGS
    engine = Engine(None, FakeTokenizer(), model=FakeModel(token_sequence=[10, 20]))
    chunks = list(engine.generate("test prompt", raw=True, options=GenOptions(max_tokens=MAX_POSITION_EMBEDDINGS)))
    assert len(chunks) > 0
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"


def _check_seed_lower() -> None:
    # 음수 하한 -> 고정 안 함
    from serve.server import MIN_SEED
    opts = GenOptions(max_tokens=10, temperature=1.0, top_k=0, top_p=1.0, seed=MIN_SEED)
    resolved_seed = opts.seed if (opts.seed is not None and opts.seed >= 0) else None
    assert resolved_seed is None, "seed < 0 must resolve to None (unseeded random generator)"


def _check_seed_upper() -> None:
    # 상한 -> 같은 seed 두 번이 같은 결과
    from serve.server import MAX_SEED
    def make_model() -> FakeModel:
        def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
            arr = np.full(100, -10.0, dtype=np.float32)
            for i in range(10):
                arr[i] = 2.0 - 0.1 * i
            return arr
        return FakeModel(vocab_size=100, logits_fn=logits_fn)

    engine1 = Engine(None, FakeTokenizer(), model=make_model())
    engine2 = Engine(None, FakeTokenizer(), model=make_model())
    opts = GenOptions(max_tokens=15, temperature=1.0, top_k=0, top_p=1.0, seed=MAX_SEED)
    out1 = [c.text for c in engine1.generate("hi", raw=True, options=opts)]
    out2 = [c.text for c in engine2.generate("hi", raw=True, options=opts)]
    assert out1 == out2, f"seed={MAX_SEED} must produce identical output on both runs"


BOUNDARY_SEMANTICS_TABLE: dict[tuple[str, str], dict[str, Any]] = {
    ("temperature", "아래"): {"desc": "0 → argmax", "check": _check_temperature_lower},
    ("temperature", "위"): {"desc": "2 → 유한한 확률로 표본", "check": _check_temperature_upper},
    ("top_p", "아래"): {"desc": "0 → argmax", "check": _check_top_p_lower},
    ("top_p", "위"): {"desc": "1 → 거르지 않음(여러 토큰이 나올 수 있음)", "check": _check_top_p_upper},
    ("top_k", "아래"): {"desc": "0 → 거르지 않음", "check": _check_top_k_lower},
    ("top_k", "위"): {"desc": "어휘 크기 이상 → 거르지 않음", "check": _check_top_k_upper},
    ("presence_penalty", "아래"): {"desc": "-2 → 이미 나온 토큰 점수에 그만큼 더하고 뺌", "check": _check_presence_penalty_lower},
    ("presence_penalty", "위"): {"desc": "2 → 이미 나온 토큰 점수에서 그만큼 뺌", "check": _check_presence_penalty_upper},
    ("max_tokens", "아래"): {"desc": "0 → 답 토큰 0개, finish_reason 은 지금 규칙대로", "check": _check_max_tokens_lower},
    ("max_tokens", "위"): {"desc": "상한값 → 받아서 생성 시작(가짜 모델이 곧 EOS)", "check": _check_max_tokens_upper},
    ("max_completion_tokens", "아래"): {"desc": "0 → 답 토큰 0개, finish_reason 은 지금 규칙대로", "check": _check_max_completion_tokens_lower},
    ("max_completion_tokens", "위"): {"desc": "상한값 → 받아서 생성 시작(가짜 모델이 곧 EOS)", "check": _check_max_completion_tokens_upper},
    ("num_predict", "아래"): {"desc": "0 → 답 토큰 0개, finish_reason 은 지금 규칙대로", "check": _check_num_predict_lower},
    ("num_predict", "위"): {"desc": "상한값 → 받아서 생성 시작(가짜 모델이 곧 EOS)", "check": _check_num_predict_upper},
    ("seed", "아래"): {"desc": "음수 하한 → 고정 안 함", "check": _check_seed_lower},
    ("seed", "위"): {"desc": "상한 → 같은 seed 두 번이 같은 결과", "check": _check_seed_upper},
}


def test_numeric_options_boundary_semantics() -> None:
    """단언 2a: 표의 열쇠 집합 = NUMERIC_OPTION_SPECS의 모든 칸 × {아래, 위} 임을 단언하고,
    각 경계값의 엔진 동작을 확인한다.
    """
    from serve.server import NUMERIC_OPTION_SPECS
    expected_keys = {(field, bound) for field in NUMERIC_OPTION_SPECS for bound in ("아래", "위")}
    assert set(BOUNDARY_SEMANTICS_TABLE.keys()) == expected_keys, (
        f"Missing or extra boundary semantics keys: {set(BOUNDARY_SEMANTICS_TABLE.keys()) ^ expected_keys}"
    )

    for (field, bound), spec in sorted(BOUNDARY_SEMANTICS_TABLE.items()):
        check_fn = spec["check"]
        check_fn()


class _SmallPrefillWeights:
    def __init__(self, hidden_size: int = 4, vocab_size: int = 260000) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._norm = np.ones(hidden_size, dtype=np.float32)

    def get(self, name: str, to_float: bool = False) -> Any:
        if name == "model.norm.weight":
            return self._norm
        if name == "model.embed_tokens.weight":
            hidden_size = self.hidden_size

            class _Embed:
                def __getitem__(_self, tid: int) -> np.ndarray:
                    arr = np.full(hidden_size, float((int(tid) % 97) + 1) / 10.0, dtype=np.float32)
                    from host.raon import f32_to_bf16_rtne
                    return f32_to_bf16_rtne(arr)

            return _Embed()
        raise KeyError(name)


class SmallPrefillBackend:
    def __init__(self, vocab_size: int = 260000) -> None:
        self.vocab_size = vocab_size
        self.lm_head_calls: int = 0
        self.linear_calls: list[str] = []
        self.rda_states: dict[int, np.ndarray] = {}
        self.mla_k: dict[int, np.ndarray] = {}
        self.mla_v: dict[int, np.ndarray] = {}
        self.step_id: int = 0

    def set_step(self, step: int) -> None:
        self.step_id = step

    def reset(self) -> None:
        self.lm_head_calls = 0
        self.linear_calls.clear()
        self.rda_states.clear()
        self.mla_k.clear()
        self.mla_v.clear()
        self.step_id = 0

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        self.linear_calls.append(name)
        if name == "lm_head.weight":
            self.lm_head_calls += 1
            from host.raon import f32_to_bf16_rtne
            logits = np.full(self.vocab_size, -20.0, dtype=np.float32)
            logits[DEFAULT_EOS_IDS[0]] = 10.0
            return f32_to_bf16_rtne(logits)
        raise NotImplementedError(name)

    def get_rda_state(self, layer: int) -> np.ndarray | None:
        st = self.rda_states.get(layer)
        return None if st is None else st.copy()[np.newaxis, ...]

    def set_rda_state(self, layer: int, state: np.ndarray) -> None:
        st = state.squeeze(0) if state.ndim == 4 else state
        self.rda_states[layer] = np.array(st, dtype=np.float32, copy=True)

    def get_mla_cache(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        k = self.mla_k.get(layer)
        v = self.mla_v.get(layer)
        if k is None or v is None:
            return None, None
        return k[np.newaxis, ...], v[np.newaxis, ...]

    def set_mla_cache(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        k = key.squeeze(0) if key.ndim == 4 else key
        v = value.squeeze(0) if value.ndim == 4 else value
        self.mla_k[layer] = np.array(k, dtype=np.uint16, copy=True)
        self.mla_v[layer] = np.array(v, dtype=np.uint16, copy=True)


from host.raon import RaonConfig, RaonModel


class SmallPrefillModel(RaonModel):
    def _route_mhc(self, stream, site_prefix, stream_f32=None):
        return (
            np.ones(self.config.raon_mhc_streams, dtype=np.float32),
            np.ones(self.config.raon_mhc_streams, dtype=np.float32),
            np.eye(self.config.raon_mhc_streams, dtype=np.float32),
        )

    def _mhc_aggregate(self, stream, h_pre, stream_f32=None):
        return stream[0].copy()

    def _mhc_mix(self, stream, branch_out, h_post, h_res, stream_f32=None):
        from host.raon import bf16_to_f32, f32_to_bf16_rtne
        return f32_to_bf16_rtne(bf16_to_f32(stream) + np.float32(1.0))

    def _layer_constants(self, layer_idx):
        return {
            "input_norm": np.ones(self.config.hidden_size, np.float32),
            "post_norm": np.ones(self.config.hidden_size, np.float32),
        }

    def _forward_rda(self, layer_idx, hidden):
        prev = self.backend.rda_states.get(layer_idx)
        if prev is None:
            prev = np.zeros((2, 4, 4), dtype=np.float32)
        self.backend.rda_states[layer_idx] = prev * 0.9 + np.float32(1.0)
        return hidden

    def _forward_dense_mlp(self, layer_idx, hidden):
        return hidden

    def reset_state(self) -> None:
        self.backend.reset()


def test_prefill_p0_logits_and_greedy_tokens_parity() -> None:
    """P0 prefill(중간 토큰 lm_head 생략) 방식과 구 방식(모든 토큰 step)이
    질문 처리 후 마지막 logits 가 바이트 단위로 동일하고,
    이어지는 greedy 생성 토큰 시퀀스도 완전히 동일함을 검증한다.
    """
    fake_tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "사과 바나나 오렌지 포도 딸기"}]
    opts = GenOptions(max_tokens=3, temperature=0.0)

    cfg = RaonConfig(
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=260000,
        raon_mhc_streams=2,
        num_experts=0,
        rda_num_heads=1,
        rda_head_dim=4,
        rda_layers=[0],
        mla_layers=[],
    )

    # 1. 새 P0 prefill 엔진
    w_new = _SmallPrefillWeights(4, 260000)
    be_new = SmallPrefillBackend(260000)
    mdl_new = SmallPrefillModel(cfg, w_new, be_new)
    eng_new = Engine(None, fake_tok, model=mdl_new, backend=be_new)
    chunks_new = list(eng_new.chat(msgs, options=opts))
    logits_new = eng_new._last_logits
    tokens_new = [c.text for c in chunks_new]

    # 2. 구 방식 엔진 (모든 토큰 need_logits=True 강제)
    class LegacyAllStepEngine(Engine):
        def _step_token(self, token_id: int, *, need_logits: bool = True) -> np.ndarray | None:
            return super()._step_token(token_id, need_logits=True)

    w_old = _SmallPrefillWeights(4, 260000)
    be_old = SmallPrefillBackend(260000)
    mdl_old = SmallPrefillModel(cfg, w_old, be_old)
    eng_old = LegacyAllStepEngine(None, fake_tok, model=mdl_old, backend=be_old)
    chunks_old = list(eng_old.chat(msgs, options=opts))
    logits_old = eng_old._last_logits
    tokens_old = [c.text for c in chunks_old]

    # 단언: 마지막 logits 바이트 단위 완전 일치
    assert logits_new is not None and logits_old is not None
    assert bytes(logits_new.data) == bytes(logits_old.data)
    assert np.array_equal(logits_new, logits_old)

    # 단언: 이어지는 greedy 생성 토큰 동일
    assert tokens_new == tokens_old


def test_prefill_p0_lm_head_call_count() -> None:
    """질문 N 토큰 처리 시 lm_head 호출 수가 (경계 토큰 + 마지막 토큰)만큼만
    호출됨을 가짜 백엔드 호출 기록으로 확인한다.
    """
    fake_tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "프롬프트 토큰 수 테스트입니다"}]
    prompt_ids = fake_tok.chat_ids(msgs, add_generation_prompt=True)
    base_ids = fake_tok.chat_ids(msgs, add_generation_prompt=False)
    n_tokens = len(prompt_ids)
    checkpoint_len = len(base_ids)
    assert n_tokens > checkpoint_len > 0

    cfg = RaonConfig(
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=260000,
        raon_mhc_streams=2,
        num_experts=0,
        rda_num_heads=1,
        rda_head_dim=4,
        rda_layers=[0],
        mla_layers=[],
    )
    weights = _SmallPrefillWeights(4, 260000)
    backend = SmallPrefillBackend(260000)
    model = SmallPrefillModel(cfg, weights, backend)
    eng = Engine(None, fake_tok, model=model, backend=backend)

    # 1. max_tokens=0 으로 실행하여 순수 prefill 단계 호출 횟수 측정
    chunks = list(eng.chat(msgs, options=GenOptions(max_tokens=0)))
    assert len(chunks) == 1
    assert chunks[0].finish_reason == "length"

    # 전체 N 토큰 중 경계 토큰(checkpoint_len)과 마지막 토큰(n_tokens)에서만 lm_head 호출!
    assert backend.lm_head_calls == 2
    assert eng._checkpoint is not None
    assert len(eng._checkpoint.tokens) == checkpoint_len
    assert eng._checkpoint.last_logits is not None

    # 2. 접두사가 전부 캐시된 경우 (남은 질문 토큰 0개)
    calls_before = backend.lm_head_calls
    chunks2 = list(eng.chat(msgs, options=GenOptions(max_tokens=0)))
    assert chunks2[0].reused_prefix_tokens == n_tokens
    # to_prefill 이 빈 리스트이므로 prefill 동안 추가 lm_head 호출은 0회
    assert backend.lm_head_calls == calls_before

    # 3. 체크포인트 복원 후 남은 토큰만 처리하는 경우
    tail = [1001, 1002, 1003, 1004]
    prompt2_ids = base_ids + tail
    calls_before2 = backend.lm_head_calls
    chunks3 = list(eng._run_stream(prompt2_ids, GenOptions(max_tokens=0), None, base_ids=base_ids))
    assert chunks3[0].reused_prefix_tokens == checkpoint_len
    # 남은 4개 토큰 중 마지막 토큰 1회만 호출됨
    assert backend.lm_head_calls - calls_before2 == 1


def test_prefill_p0_checkpoint_and_last_logits_contract() -> None:
    """중간 토큰 뒤 _last_logits 가 None 으로 남아 낡은 값이 오염되지 않는지,
    경계 토큰에서는 logits 가 채워져 체크포인트에 정상 저장되고 복원되는지 검증한다.
    """
    fake_tok = FakeTokenizer()
    cfg = RaonConfig(
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=260000,
        raon_mhc_streams=2,
        num_experts=0,
        rda_num_heads=1,
        rda_head_dim=4,
        rda_layers=[0],
        mla_layers=[],
    )
    weights = _SmallPrefillWeights(4, 260000)
    backend = SmallPrefillBackend(260000)
    model = SmallPrefillModel(cfg, weights, backend)
    eng = Engine(None, fake_tok, model=model, backend=backend)

    # 1. 단일 토큰 need_logits=False 진행 시 _last_logits 가 None 이 됨을 확인
    eng._step_token(10, need_logits=False)
    assert eng._last_logits is None
    assert eng._cached_tokens == [10]
    assert backend.lm_head_calls == 0

    # 2. need_logits=True 진행 시 _last_logits 가 유효한 ndarray 로 채워짐
    eng._step_token(20, need_logits=True)
    assert eng._last_logits is not None
    assert eng._cached_tokens == [10, 20]
    assert backend.lm_head_calls == 1
    logits_saved = np.array(eng._last_logits, copy=True)

    # 3. _save_checkpoint() 계약: _last_logits 가 있으면 정상 저장
    eng._save_checkpoint()
    assert eng._checkpoint is not None
    assert np.array_equal(eng._checkpoint.last_logits, logits_saved)

    # 4. _last_logits 가 None 일 때는 _save_checkpoint() 가 저장을 건너뜀
    eng._reset_state(clear_checkpoint=True)
    eng._step_token(30, need_logits=False)
    assert eng._last_logits is None
    eng._save_checkpoint()
    assert eng._checkpoint is None
