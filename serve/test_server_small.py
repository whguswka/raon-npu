# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon NPU 서버 소켓 기반 스모크 및 기능 테스트.

`serve.engine`은 별도 워커가 만든다고 가정하므로, 테스트 전용 가짜 엔진을
`sys.modules`에 주입한 뒤 서버를 실제 소켓 위에 띄운다.  HTTP 클라이언트는
표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import ast
import copy
import http.client
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import types
from collections import namedtuple
from typing import Any
import pytest

# --------------------------------------------------------------------------- #
# 가짜 엔진 / serve.engine 모듈 주입
# --------------------------------------------------------------------------- #

GenOptions = namedtuple(
    "GenOptions",
    ["max_tokens", "temperature", "top_p", "top_k",
     "presence_penalty", "stop", "seed"],
    defaults=[None, None, None, None, None, None, None],
)

Chunk = namedtuple(
    "Chunk",
    ["text", "done", "finish_reason", "prompt_tokens", "completion_tokens",
     "prefill_s", "decode_s", "reused_prefix_tokens", "thinking"],
    defaults=["", False, None, 0, 0, 0.0, 0.0, 0, ""],
)


class FakeEngine:
    model_name = "raon-1-mini"

    def __init__(self, weights_dir: str = "", model_dir: str = "", preload: bool = True) -> None:
        self.weights_dir = weights_dir
        self.model_dir = model_dir
        self.preload = preload
        self.loaded = True
        self.cancel_log: list[bool] = []
        self.cancel_callbacks: list[Any] = []
        self.last_enable_thinking: bool | None = None
        self.sleep_per_chunk = 0.01
        self.thinking_text = "질문을 분석합니다.\n핵심 요점을 정리합니다."
        self.chat_text = (
            "안녕하세요. 저는 Raon NPU 언어 모델입니다. "
            "질문에 답변하고 글을 쓰거나 요약을 도와드릴 수 있습니다. "
            "무엇을 도와드릴까요?"
        )
        self.generate_text = "생성된 텍스트입니다."
        self.hold_seconds = 0.0  # busy/503 및 keep-alive 생성 중 보호 테스트용 점유 시간
        self.chat_calls = 0
        self.generate_calls = 0
        self.raise_after_chunks: int | None = None
        self.raise_exception: Exception | None = None
        self.last_messages: Any = None
        self.last_tools: Any = None
        self.last_prompt: Any = None
        self.last_raw: bool = False

    def unload(self) -> None:
        self.loaded = False

    def _word_pieces(self, text: str) -> list[str]:
        if not text:
            return []
        words = text.split(" ")
        if len(words) == 1:
            return list(text)
        return [w + ("" if i == len(words) - 1 else " ") for i, w in enumerate(words)]

    def _chat_chunks(
        self,
        *,
        enable_thinking: bool | None,
        max_tokens: int | None,
        prompt_tokens: int,
    ) -> list[Chunk]:
        think_pieces: list[str] = []
        if enable_thinking is not False and self.thinking_text:
            think_pieces = self._word_pieces(self.thinking_text)
        ans_pieces = self._word_pieces(self.chat_text)

        # (is_thinking, piece_str) 목록 구성
        steps: list[tuple[bool, str]] = [(True, p) for p in think_pieces] + [
            (False, p) for p in ans_pieces
        ]
        if not steps:
            steps = [(False, "")]

        finish_reason = "stop"
        if max_tokens is not None and max_tokens > 0 and len(steps) > max_tokens:
            steps = steps[:max_tokens]
            finish_reason = "length"

        out: list[Chunk] = []
        for i, (is_think, piece) in enumerate(steps):
            is_last = i == len(steps) - 1
            out.append(
                Chunk(
                    text="" if is_think else piece,
                    thinking=piece if is_think else "",
                    done=is_last,
                    finish_reason=finish_reason if is_last else None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=i + 1,
                    prefill_s=0.05,
                    decode_s=0.01 * (i + 1),
                    reused_prefix_tokens=0,
                )
            )
        return out

    def _gen_chunks(self, full_text: str, prompt_tokens: int) -> list[Chunk]:
        pieces = self._word_pieces(full_text) or [""]
        out: list[Chunk] = []
        for i, p in enumerate(pieces):
            is_last = i == len(pieces) - 1
            out.append(
                Chunk(
                    text=p,
                    thinking="",
                    done=is_last,
                    finish_reason="stop" if is_last else None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=i + 1,
                    prefill_s=0.05,
                    decode_s=0.01 * (i + 1),
                    reused_prefix_tokens=0,
                )
            )
        return out

    def chat(self, messages, tools=None, enable_thinking=None,
             options=None, cancel=None):
        self.chat_calls += 1
        self.last_messages = messages
        self.last_tools = tools
        self.last_enable_thinking = enable_thinking
        self.last_options = options
        max_tokens = getattr(options, "max_tokens", None) if options is not None else None
        ptoks = 7
        for index, ch in enumerate(self._chat_chunks(
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            prompt_tokens=ptoks,
        )):
            if self.raise_after_chunks is not None and index >= self.raise_after_chunks:
                raise self.raise_exception or RuntimeError("synthetic engine failure")
            if cancel is not None:
                self.cancel_callbacks.append(cancel)
                self.cancel_log.append(cancel())
                if cancel():
                    return
            if self.sleep_per_chunk:
                time.sleep(self.sleep_per_chunk)
            if self.hold_seconds:
                time.sleep(self.hold_seconds)
            yield ch

    def generate(self, prompt, raw=False, options=None, cancel=None):
        self.generate_calls += 1
        self.last_prompt = prompt
        self.last_raw = raw
        self.last_options = options
        ptoks = 3
        for index, ch in enumerate(self._gen_chunks(self.generate_text, ptoks)):
            if self.raise_after_chunks is not None and index >= self.raise_after_chunks:
                raise self.raise_exception or RuntimeError("synthetic engine failure")
            if cancel is not None:
                self.cancel_callbacks.append(cancel)
                self.cancel_log.append(cancel())
                if cancel():
                    return
            if self.sleep_per_chunk:
                time.sleep(self.sleep_per_chunk)
            yield ch


_fake_engine_module = types.ModuleType("serve.engine")
_fake_engine_module.Engine = FakeEngine
_fake_engine_module.GenOptions = GenOptions
_fake_engine_module.Chunk = Chunk
sys.modules["serve.engine"] = _fake_engine_module

from host.strict_json import loads_strict
from serve.server import (
    Handler,
    _LogWriter,
    Server,
    _build_parser,
    parse_keep_alive,
    OPENAI_CHAT_SUPPORTED,
    OPENAI_CHAT_IGNORED,
    OPENAI_CHAT_UNSUPPORTED,
    OPENAI_COMPLETIONS_SUPPORTED,
    OPENAI_COMPLETIONS_IGNORED,
    OPENAI_COMPLETIONS_UNSUPPORTED,
    STREAM_OPTIONS_SUPPORTED,
    STREAM_OPTIONS_IGNORED,
    STREAM_OPTIONS_UNSUPPORTED,
    OLLAMA_CHAT_SUPPORTED,
    OLLAMA_CHAT_IGNORED,
    OLLAMA_CHAT_UNSUPPORTED,
    OLLAMA_GENERATE_SUPPORTED,
    OLLAMA_GENERATE_IGNORED,
    OLLAMA_GENERATE_UNSUPPORTED,
    OLLAMA_OPTIONS_SUPPORTED,
    OLLAMA_OPTIONS_IGNORED,
    OLLAMA_OPTIONS_UNSUPPORTED,
    OPENAI_MESSAGE_ALLOWED_FIELDS,
    OLLAMA_MESSAGE_ALLOWED_FIELDS,
    OPENAI_CONTENT_PART_ALLOWED_FIELDS,
    OLLAMA_CONTENT_PART_ALLOWED_FIELDS,
    OPENAI_TOOL_CALL_ALLOWED_FIELDS,
    OLLAMA_TOOL_CALL_ALLOWED_FIELDS,
    OPENAI_TOOL_FUNCTION_ALLOWED_FIELDS,
    OLLAMA_TOOL_FUNCTION_ALLOWED_FIELDS,
)


# --------------------------------------------------------------------------- #
# pytest fixture: 실제 소켓 서버
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def server_addr():
    engine = FakeEngine("", "")
    Handler.engine = engine
    Handler.engine_factory = lambda: FakeEngine("", "")
    Handler.GenOptions = GenOptions
    Handler.Chunk = Chunk
    Handler.max_wait = 600.0
    Handler.keep_alive = 300.0
    Handler.logger = _LogWriter(None)
    Handler.model_name = engine.model_name
    Handler.load_count = 0
    Handler.unload_count = 0

    Handler._busy_event.clear()
    with Handler._busy_lock:
        Handler._sync_engine_tracking_locked()
        Handler._set_expiry_locked(Handler.keep_alive)

    server = Server(("127.0.0.1", 0), Handler)
    server.sndbuf = 1  # 클라이언트 끊김 감지를 빠르게 하기 위해 송신 버퍼 최소화
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ("127.0.0.1", port, engine)
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# HTTP 헬퍼
# --------------------------------------------------------------------------- #

def _request(host_port, method: str, path: str, body: dict | None = None,
             headers: dict | None = None, stream: bool = False):
    host, port = host_port
    conn = http.client.HTTPConnection(host, port, timeout=10)
    all_headers = headers or {}
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        all_headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=payload, headers=all_headers)
    resp = conn.getresponse()
    if stream:
        return conn, resp
    data = resp.read()
    status = resp.status
    ctype = resp.getheader("Content-Type", "")
    conn.close()
    return status, data, ctype


def _json(host_port, method: str, path: str, body: dict | None = None):
    status, data, _ = _request(host_port, method, path, body)
    return status, json.loads(data.decode("utf-8"))


def _wait_engine_released(timeout: float = 2.0) -> None:
    """서버 스레드가 요청 끝의 `_release_engine` 을 마칠 때까지 기다린다.

    서버는 응답을 다 쓴 뒤 finally 에서 만료 시각을 정하고(keep_alive 0 이면 엔진을 내려놓고)
    잠금을 푼다. 그래서 응답을 받은 직후에는 아직 잠겨 있을 수 있다. 잠금을 짧은 시한 안에
    잡았다 놓아 해제를 기다리고, 잡은 동안 바쁨 표시·소유자가 비었는지 본다.
    """
    assert Handler._busy_lock.acquire(timeout=timeout), "서버가 엔진 잠금을 풀지 않음"
    try:
        assert not Handler._busy_event.is_set()
        assert Handler._busy_holder is None
    finally:
        Handler._busy_lock.release()


def _read_sse(resp) -> tuple[list[dict], dict | None]:
    """SSE 응답을 파싱해 chunk 목록과 마지막 usage chunk를 반환."""
    chunks: list[dict] = []
    usage: dict | None = None
    buffer = b""
    while True:
        chunk = resp.read(256)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                return chunks, usage
            obj = json.loads(payload.decode("utf-8"))
            if obj.get("usage") is not None:
                usage = obj["usage"]
                if obj.get("choices") == []:
                    continue
            chunks.append(obj)
    return chunks, usage


def _read_ndjson(resp) -> list[dict]:
    """NDJSON 응답을 파싱."""
    out: list[dict] = []
    for line in resp.read().decode("utf-8").strip().split("\n"):
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# 기본 엔드포인트 및 모델 메타데이터 테스트
# --------------------------------------------------------------------------- #

def test_root_get_ollama_is_running(server_addr):
    status, data, ctype = _request(server_addr[:2], "GET", "/")
    assert status == 200
    assert data == b"Ollama is running"
    assert ctype.startswith("text/plain")


def test_root_head_no_body(server_addr):
    host, port = server_addr[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("HEAD", "/")
    resp = conn.getresponse()
    data = resp.read()
    assert resp.status == 200
    assert data == b""
    assert resp.getheader("Content-Type", "").startswith("text/plain")
    assert resp.getheader("Content-Length") == str(len(b"Ollama is running"))
    conn.close()
    status, _, _ = _request(server_addr[:2], "GET", "/nope")
    assert status == 404


def test_healthz(server_addr):
    status, data = _json(server_addr[:2], "GET", "/healthz")
    assert status == 200
    assert data["status"] == "ok"
    assert data["loaded"] is True


def test_api_version(server_addr):
    status, data = _json(server_addr[:2], "GET", "/api/version")
    assert status == 200
    assert "version" in data


def test_v1_models(server_addr):
    status, data = _json(server_addr[:2], "GET", "/v1/models")
    assert status == 200
    assert data["object"] == "list"
    assert any(m["id"] == "raon-1-mini" for m in data["data"])


def test_api_tags_and_show_model_details(server_addr, tmp_path):
    # manifest.json 의 total_size_bytes 및 실제 양자화/포맷 정보 확인
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"total_size_bytes": 26131993360, "bundles": []}),
        encoding="utf-8",
    )
    old_wdir = Handler.weights_dir
    Handler.weights_dir = str(tmp_path)
    Handler._cached_manifest_info = None
    try:
        status, data = _json(server_addr[:2], "GET", "/api/tags")
        assert status == 200
        models = data["models"]
        assert any(m["name"] == "raon-1-mini:latest" for m in models)
        m0 = models[0]
        assert m0["size"] == 26131993360
        assert m0["details"]["format"] == "raon-npu"
        assert m0["details"]["quantization_level"] == "D: q4nx + 9-bit bfp16"
        assert m0["details"]["parameter_size"] == "28.8B"

        status_show, show_data = _json(
            server_addr[:2], "POST", "/api/show", {"name": "raon-1-mini:latest"}
        )
        assert status_show == 200
        assert "template" in show_data
        assert show_data["size"] == 26131993360
        assert show_data["details"]["format"] == "raon-npu"
        assert show_data["details"]["quantization_level"] == "D: q4nx + 9-bit bfp16"
        assert show_data["details"]["parameter_size"] == "28.8B"
    finally:
        Handler.weights_dir = old_wdir
        Handler._cached_manifest_info = None


def test_unknown_model_404(server_addr):
    status, _ = _json(server_addr[:2], "POST", "/v1/chat/completions",
                      {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 404
    status, _ = _json(server_addr[:2], "POST", "/api/chat",
                      {"model": "unknown", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 404


# --------------------------------------------------------------------------- #
# 생각과 답 나누기 (OpenAI / Ollama, 스트림 / 비스트림 / 생각 중 max_tokens / 생각 끔)
# --------------------------------------------------------------------------- #

def test_v1_chat_completions_non_stream_thinking_split(server_addr):
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    })
    assert status == 200
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    msg = choice["message"]
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "질문을 분석합니다.\n핵심 요점을 정리합니다."
    assert msg["content"].startswith("안녕하세요.")
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@pytest.mark.parametrize("include_usage", [False, True])
def test_v1_chat_completions_stream_thinking_split(server_addr, include_usage):
    conn, resp = _request(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": include_usage},
    }, stream=True)
    assert resp.status == 200
    assert "text/event-stream" in resp.getheader("Content-Type", "")
    chunks, usage = _read_sse(resp)
    conn.close()

    reasonings = [c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks]
    texts = [c["choices"][0]["delta"].get("content", "") for c in chunks]
    full_reasoning = "".join(reasonings)
    full_content = "".join(texts)

    assert full_reasoning == "질문을 분석합니다.\n핵심 요점을 정리합니다."
    assert "안녕" in full_content
    if include_usage:
        assert usage is not None
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert all(c.get("usage") is None for c in chunks)
    else:
        assert usage is None
        assert all("usage" not in c for c in chunks)
    ids = {c["id"] for c in chunks}
    created = {c["created"] for c in chunks}
    assert len(ids) == 1
    assert len(created) == 1
    assert sum(c["choices"][0].get("finish_reason") == "stop" for c in chunks) == 1


def test_thinking_cutoff_by_max_tokens_leaves_content_empty(server_addr):
    """max_tokens가 생각 안에서 끝나면 content는 비고 finish_reason은 'length'여야 한다."""
    # 1. OpenAI non-stream (max_completion_tokens=2 -> 생각 조각 2개에서 끝남)
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 2,
        "stream": False,
    })
    assert status == 200
    choice = data["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["reasoning_content"] != ""
    assert choice["message"]["content"] == ""

    # 2. OpenAI stream
    conn, resp = _request(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "stream": True,
    }, stream=True)
    chunks, _ = _read_sse(resp)
    conn.close()
    assert "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks) != ""
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == ""
    assert any(c["choices"][0].get("finish_reason") == "length" for c in chunks)

    # 3. Ollama /api/chat non-stream (num_predict=2)
    status_o, data_o = _json(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"num_predict": 2},
        "stream": False,
    })
    assert status_o == 200
    assert data_o["done_reason"] == "length"
    assert data_o["message"]["thinking"] != ""
    assert data_o["message"]["content"] == ""


def test_disable_thinking_openai_and_ollama(server_addr):
    """chat_template_kwargs.enable_thinking=false, reasoning_effort='none', think=false 테스트."""
    # 1. OpenAI chat_template_kwargs.enable_thinking = False
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    })
    assert status == 200
    assert Handler.engine.last_enable_thinking is False
    assert data["choices"][0]["message"]["reasoning_content"] == ""
    assert data["choices"][0]["message"]["content"].startswith("안녕하세요.")

    # 2. OpenAI reasoning_effort = "none"
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
        "stream": False,
    })
    assert status == 200
    assert Handler.engine.last_enable_thinking is False
    assert data["choices"][0]["message"]["reasoning_content"] == ""
    assert data["choices"][0]["message"]["content"].startswith("안녕하세요.")

    # 3. Ollama think = False (non-stream & stream)
    status, data = _json(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "think": False,
        "stream": False,
    })
    assert status == 200
    assert Handler.engine.last_enable_thinking is False
    assert data["message"]["thinking"] == ""
    assert data["message"]["content"].startswith("안녕하세요.")

    conn, resp = _request(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "think": False,
        "stream": True,
    }, stream=True)
    nd_chunks = _read_ndjson(resp)
    conn.close()
    assert "".join(c["message"].get("thinking", "") for c in nd_chunks) == ""
    assert "".join(c["message"]["content"] for c in nd_chunks).startswith("안녕하세요.")


def test_api_chat_stream_and_non_stream_thinking(server_addr):
    # Non-stream
    status, data = _json(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert status == 200
    assert data["message"]["thinking"] == "질문을 분석합니다.\n핵심 요점을 정리합니다."
    assert data["message"]["content"].startswith("안녕하세요.")

    # Stream
    conn, resp = _request(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }, stream=True)
    assert resp.status == 200
    assert "ndjson" in resp.getheader("Content-Type", "")
    chunks = _read_ndjson(resp)
    conn.close()

    full_thinking = "".join(c["message"].get("thinking", "") for c in chunks)
    full_content = "".join(c["message"]["content"] for c in chunks)
    assert full_thinking == "질문을 분석합니다.\n핵심 요점을 정리합니다."
    assert full_content.startswith("안녕하세요.")
    last = chunks[-1]
    assert last["done"] is True
    assert last.get("done_reason") == "stop"
    assert last.get("prompt_eval_count", 0) > 0


def test_api_generate(server_addr):
    status, data = _json(server_addr[:2], "POST", "/api/generate", {
        "model": "raon-1-mini",
        "prompt": "hello",
        "stream": False,
    })
    assert status == 200
    assert data["done"] is True
    assert data.get("done_reason") == "stop"
    assert "response" in data


# --------------------------------------------------------------------------- #
# Keep-alive: 짧은 시간 내려놓기, 다시 적재, 생성 중 내려놓지 않음
# --------------------------------------------------------------------------- #

def test_keep_alive_unload_and_reload(server_addr):
    host, port, _ = server_addr
    old_ka = Handler.keep_alive
    try:
        Handler.keep_alive = 0.12

        # 1. 요청 보내서 keep_alive 타이머(0.12초) 시작
        status, _ = _json((host, port), "POST", "/api/chat", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        })
        assert status == 200

        # 직후에는 적재되어 있고 /api/ps 에 모델과 만료 시각이 나옴
        _, hz1 = _json((host, port), "GET", "/healthz")
        assert hz1 == {"status": "ok", "loaded": True}
        _, ps1 = _json((host, port), "GET", "/api/ps")
        assert len(ps1["models"]) == 1
        assert "expires_at" in ps1["models"][0]

        # 2. 0.25초 대기 후 자동으로 내려놓아졌는지 확인
        time.sleep(0.25)
        status_hz2, hz2 = _json((host, port), "GET", "/healthz")
        assert status_hz2 == 200
        assert hz2 == {"status": "ok", "loaded": False}
        _, ps2 = _json((host, port), "GET", "/api/ps")
        assert ps2 == {"models": []}
        assert Handler.engine is None

        # 3. 다음 요청이 오면 다시 적재하고 정상 응답
        loads_before = Handler.load_count
        status3, data3 = _json((host, port), "POST", "/v1/chat/completions", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "reload me"}],
            "stream": False,
        })
        assert status3 == 200
        assert data3["choices"][0]["message"]["content"].startswith("안녕하세요.")
        assert Handler.load_count == loads_before + 1
        _, hz3 = _json((host, port), "GET", "/healthz")
        assert hz3 == {"status": "ok", "loaded": True}

        # 4. Ollama 요청별 keep_alive=0 으로 즉시 내려놓기 확인
        status4, _ = _json((host, port), "POST", "/api/chat", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "unload immediately"}],
            "keep_alive": 0,
            "stream": False,
        })
        assert status4 == 200
        _wait_engine_released()
        _, hz4 = _json((host, port), "GET", "/healthz")
        assert hz4 == {"status": "ok", "loaded": False}

        # 다시 적재해 둠
        _json((host, port), "POST", "/api/chat", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "restore"}],
            "keep_alive": 300,
            "stream": False,
        })
    finally:
        Handler.keep_alive = old_ka
        if not Handler._is_loaded():
            with Handler._busy_lock:
                Handler._ensure_loaded_locked()
                Handler._set_expiry_locked(old_ka)


def test_keep_alive_never_unloads_during_generation(server_addr):
    """생성 중에는 만료 시각이 지나도 절대 내려놓지 않고, 생성이 끝난 뒤에만 내려놓는다."""
    host, port, _ = server_addr
    old_ka = Handler.keep_alive
    try:
        Handler.keep_alive = 0.08
        with Handler._busy_lock:
            eng = Handler._ensure_loaded_locked()
            eng.sleep_per_chunk = 0.03  # 약 15~20 청크 -> 총 0.45초 이상 소요
            Handler._set_expiry_locked(0.05)

        unloaded_mid_gen = False
        gen_done = threading.Event()
        result_holder: dict[str, Any] = {}

        def run_long_request() -> None:
            status, data = _json((host, port), "POST", "/v1/chat/completions", {
                "model": "raon-1-mini",
                "messages": [{"role": "user", "content": "long generation"}],
                "stream": False,
            })
            result_holder["status"] = status
            result_holder["data"] = data
            gen_done.set()

        t = threading.Thread(target=run_long_request)
        t.start()

        # 생성이 진행 중인 0.15초 시점(만료 시간 0.08초 경과 후)에 확인
        time.sleep(0.15)
        assert not gen_done.is_set(), "generation finished too quickly"
        _, hz_mid = _json((host, port), "GET", "/healthz")
        if not hz_mid["loaded"] or Handler.engine is None:
            unloaded_mid_gen = True

        t.join(timeout=5.0)
        assert not unloaded_mid_gen, "Engine was unloaded while generation was in progress!"
        assert result_holder.get("status") == 200

        # 이제 생성이 끝났으므로 0.2초 뒤에는 내려놓아져야 함
        time.sleep(0.20)
        _, hz_after = _json((host, port), "GET", "/healthz")
        assert hz_after == {"status": "ok", "loaded": False}
    finally:
        Handler.keep_alive = old_ka
        with Handler._busy_lock:
            eng = Handler._ensure_loaded_locked()
            eng.sleep_per_chunk = 0.01
            Handler._set_expiry_locked(old_ka)


# --------------------------------------------------------------------------- #
# 바쁠 때 503 및 클라이언트 끊김 cancel 테스트
# --------------------------------------------------------------------------- #

def test_busy_timeout_503(server_addr):
    """엔진이 바쁠 때 두 번째 요청은 대기 상한을 넘어 503을 받는다."""
    host, port, _ = server_addr
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    old_hold = engine.hold_seconds
    old_max_wait = Handler.max_wait
    try:
        engine.hold_seconds = 1.0  # 첫 요청을 1초간 붙잡음
        Handler.max_wait = 0.05    # 대기 상한을 짧게

        conn1, resp1 = _request((host, port), "POST", "/v1/chat/completions", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "first"}],
            "stream": True,
        }, stream=True)
        assert resp1.status == 200

        status2, data2, _ = _request((host, port), "POST", "/v1/chat/completions", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "second"}],
            "stream": False,
        })
        assert status2 == 503, f"expected 503, got {status2}: {data2!r}"

        _read_sse(resp1)
        conn1.close()
    finally:
        engine.hold_seconds = old_hold
        Handler.max_wait = old_max_wait
        Handler._busy_event.clear()
        try:
            Handler._busy_lock.release()
        except RuntimeError:
            pass


def test_client_disconnect_cancels(server_addr):
    """스트리밍 중 클라이언트가 연결을 끊으면 cancel 콜백이 True가 된다."""
    host, port, _ = server_addr
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    engine.cancel_log.clear()
    engine.cancel_callbacks.clear()
    old_sleep = engine.sleep_per_chunk
    try:
        engine.sleep_per_chunk = 0.1  # chunk를 천천히 보내 cancel을 유도

        body = json.dumps({
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        }, ensure_ascii=False).encode("utf-8")
        req = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1:" + str(port).encode() + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n" + body
        )

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((host, port))
        s.sendall(req)
        s.recv(64)  # 헤더 일부만 읽고 강제 종료
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()

        time.sleep(0.6)

        assert any(cb() for cb in engine.cancel_callbacks if cb is not None), \
            f"cancel_callbacks={engine.cancel_callbacks}"
    finally:
        engine.sleep_per_chunk = old_sleep


@pytest.mark.parametrize("body", [[], "text", 7])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
def test_non_object_json_is_400_without_engine_call(server_addr, path, body):
    engine = Handler.engine
    before = engine.chat_calls
    status, data = _json(server_addr[:2], "POST", path, body)
    assert status == 400
    if path.startswith("/v1/"):
        assert data["error"]["type"] == "invalid_request_error"
        assert "message" in data["error"]
    else:
        assert isinstance(data["error"], str)
    assert engine.chat_calls == before


def _connection_aborted_10053():
    exc = ConnectionAbortedError(10053, "aborted")
    exc.winerror = 10053
    return exc


@pytest.mark.parametrize(
    "path,body,param",
    [
        ("/v1/chat/completions", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "stop": 3,
        }, "stop"),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "options": {"stop": 3},
        }, "options.stop"),
        ("/v1/chat/completions", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": "3",
        }, "max_tokens"),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "options": {"num_predict": "3"},
        }, "options.num_predict"),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "options": "bad",
        }, "options"),
        ("/v1/chat/completions", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "temperature": 3,
        }, "temperature"),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "x"}],
            "options": {"temperature": 3},
        }, "options.temperature"),
    ],
)
def test_invalid_generation_options_are_400_before_engine_lock(
    server_addr, path, body, param
):
    engine = Handler.engine
    before = engine.chat_calls
    status, data = _json(server_addr[:2], "POST", path, body)
    assert status == 400
    if path.startswith("/v1/"):
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == param
    else:
        assert isinstance(data["error"], str)
        assert param in data["error"]
    assert engine.chat_calls == before


def test_openai_stream_reports_engine_exception_in_band(server_addr):
    engine = Handler.engine
    engine.raise_after_chunks = 1
    try:
        conn, resp = _request(server_addr[:2], "POST", "/v1/chat/completions", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "fail"}],
            "stream": True,
        }, stream=True)
        assert resp.status == 200
        chunks, usage = _read_sse(resp)
        conn.close()
        errors = [chunk["error"] for chunk in chunks if "error" in chunk]
        assert usage is None
        assert len(errors) == 1
        assert errors[0]["type"] == "server_error"
        assert errors[0]["code"] == "generation_error"
    finally:
        engine.raise_after_chunks = None


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "fail"}],
            "stream": False,
        }),
        ("/v1/completions", {
            "model": "raon-1-mini", "prompt": "fail", "stream": False,
        }),
        ("/api/chat", {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "fail"}],
            "stream": False,
        }),
        ("/api/generate", {
            "model": "raon-1-mini", "prompt": "fail", "stream": False,
        }),
    ],
)
def test_non_stream_engine_exception_returns_500_and_server_recovers(
    server_addr, path, body, raise_after_chunks
):
    engine = Handler.engine
    engine.raise_after_chunks = raise_after_chunks
    try:
        status, data = _json(server_addr[:2], "POST", path, body)
        assert status == 500
        if path.startswith("/v1/"):
            assert set(data["error"]) == {"message", "type", "param", "code"}
            assert data["error"]["type"] == "server_error"
            assert data["error"]["code"] == "generation_error"
        else:
            assert data == {"error": "synthetic engine failure"}
    finally:
        engine.raise_after_chunks = None

    status, recovered = _json(server_addr[:2], "POST", "/api/generate", {
        "model": "raon-1-mini", "prompt": "recovery", "stream": False,
    })
    assert status == 200
    assert recovered["done"] is True


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "fail"}],
            "stream": False,
        }),
        ("/v1/completions", {
            "model": "raon-1-mini", "prompt": "fail", "stream": False,
        }),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "fail"}],
            "stream": False,
        }),
        ("/api/generate", {
            "model": "raon-1-mini", "prompt": "fail", "stream": False,
        }),
    ],
)
def test_non_stream_engine_oserror_returns_500_and_server_recovers(
    server_addr, path, body, raise_after_chunks
):
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    engine.raise_after_chunks = raise_after_chunks
    engine.raise_exception = OSError("npu driver failed")
    try:
        status, data = _json(server_addr[:2], "POST", path, body)
        assert status == 500
        if path.startswith("/v1/"):
            assert data["error"] == {
                "message": "npu driver failed",
                "type": "server_error",
                "param": None,
                "code": "generation_error",
            }
        else:
            assert data == {"error": "npu driver failed"}
    finally:
        engine.raise_after_chunks = None
        engine.raise_exception = None

    status, recovered = _json(server_addr[:2], "POST", "/api/generate", {
        "model": "raon-1-mini", "prompt": "recovery", "stream": False,
    })
    assert status == 200
    assert recovered["done"] is True


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
@pytest.mark.parametrize(
    "path,body,is_openai",
    [
        ("/v1/chat/completions", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "fail"}],
            "stream": True,
        }, True),
        ("/v1/completions", {
            "model": "raon-1-mini", "prompt": "fail", "stream": True,
        }, True),
        ("/api/chat", {
            "model": "raon-1-mini", "messages": [{"role": "user", "content": "fail"}],
            "stream": True,
        }, False),
        ("/api/generate", {
            "model": "raon-1-mini", "prompt": "fail", "stream": True,
        }, False),
    ],
)
def test_stream_engine_oserror_is_reported_in_band_and_server_recovers(
    server_addr, path, body, is_openai, raise_after_chunks
):
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    engine.raise_after_chunks = raise_after_chunks
    engine.raise_exception = OSError("npu driver failed")
    try:
        conn, resp = _request(server_addr[:2], "POST", path, body, stream=True)
        assert resp.status == 200
        if is_openai:
            chunks, _ = _read_sse(resp)
            errors = [chunk["error"] for chunk in chunks if "error" in chunk]
            assert errors == [{
                "message": "npu driver failed",
                "type": "server_error",
                "param": None,
                "code": "generation_error",
            }]
        else:
            chunks = _read_ndjson(resp)
            assert [chunk["error"] for chunk in chunks if "error" in chunk] == [
                "npu driver failed"
            ]
        conn.close()
    finally:
        engine.raise_after_chunks = None
        engine.raise_exception = None

    status, recovered = _json(server_addr[:2], "POST", "/api/generate", {
        "model": "raon-1-mini", "prompt": "recovery", "stream": False,
    })
    assert status == 200
    assert recovered["done"] is True


@pytest.mark.parametrize("error_type", [BrokenPipeError, ConnectionResetError, OSError])
@pytest.mark.parametrize("writer", ["json", "sse", "ndjson"])
def test_socket_write_errors_set_generation_cancel(error_type, writer):
    handler = object.__new__(Handler)
    cancel = handler._make_cancel()
    if writer == "json":
        handler._write_json = lambda status, obj: (_ for _ in ()).throw(error_type("gone"))
        assert handler._write_json_or_cancel(200, {}, cancel) is False
    else:
        handler.connection = types.SimpleNamespace(
            sendall=lambda payload: (_ for _ in ()).throw(error_type("gone"))
        )
        write = handler._write_sse if writer == "sse" else handler._write_ndjson
        assert write({}, cancel) is False
    assert cancel() is True


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
@pytest.mark.parametrize(
    "method,args",
    [
        ("_single_openai_chat", ([{"role": "user", "content": "x"}], None, GenOptions(), None)),
        ("_single_openai_completion", ("x", GenOptions())),
        ("_single_ollama_chat", ([{"role": "user", "content": "x"}], None, GenOptions(), None)),
        ("_single_ollama_generate", ("x", False, GenOptions())),
    ],
)
def test_all_non_stream_methods_report_engine_oserror_without_socket(
    method, args, raise_after_chunks
):
    handler = object.__new__(Handler)
    engine = FakeEngine("", "")
    engine.sleep_per_chunk = 0
    engine.raise_after_chunks = raise_after_chunks
    engine.raise_exception = OSError("npu driver failed")
    handler.engine = engine
    handler._check_cancel = lambda cancel: None
    errors = []
    handler._send_error = lambda status, message, **kwargs: errors.append(
        (status, message, kwargs)
    )

    getattr(handler, method)(*args)

    assert errors == [(500, "npu driver failed", {"code": "generation_error"})]


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
@pytest.mark.parametrize(
    "method,args,protocol",
    [
        ("_stream_openai_chat", ([{"role": "user", "content": "x"}], None, GenOptions(), None), "sse"),
        ("_stream_openai_completion", ("x", GenOptions()), "sse"),
        ("_stream_ollama_chat", ([{"role": "user", "content": "x"}], None, GenOptions(), None), "ndjson"),
        ("_stream_ollama_generate", ("x", False, GenOptions()), "ndjson"),
    ],
)
def test_all_stream_methods_report_engine_oserror_without_socket(
    method, args, protocol, raise_after_chunks
):
    handler = object.__new__(Handler)
    engine = FakeEngine("", "")
    engine.sleep_per_chunk = 0
    engine.raise_after_chunks = raise_after_chunks
    engine.raise_exception = OSError("npu driver failed")
    handler.engine = engine
    handler._start_stream = lambda content_type, cancel: True
    handler._log = lambda *args, **kwargs: None
    chunks = []
    handler._write_sse = lambda obj, cancel: chunks.append(obj) or True
    handler._write_ndjson = lambda obj, cancel: chunks.append(obj) or True
    handler._write_done = lambda cancel: None

    getattr(handler, method)(*args)

    errors = [chunk["error"] for chunk in chunks if "error" in chunk]
    if protocol == "sse":
        assert errors == [{
            "message": "npu driver failed",
            "type": "server_error",
            "param": None,
            "code": "generation_error",
        }]
    else:
        assert errors == ["npu driver failed"]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "bad", "content": "x"}],
        [{"role": "user", "content": [{}]}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        [{"role": "user", "content": 7}],
        [{"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": "lookup", "arguments": "{}"}
        }]}],
    ],
)
def test_openai_invalid_messages_are_400_before_engine_call(server_addr, messages):
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    before = engine.chat_calls
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini", "messages": messages, "stream": False,
    })
    assert status == 400
    assert data["error"]["type"] == "invalid_request_error"
    assert engine.chat_calls == before


@pytest.mark.parametrize(
    "api_type,path",
    [("openai", "/v1/chat/completions"), ("ollama", "/api/chat")],
)
def test_unknown_field_at_every_chat_object_is_400_before_engine_call(
    server_addr, api_type, path
):
    from serve.chat_contract_test_utils import (
        build_all_fields_messages,
        copy_with_unknown_field,
        walk_chat_objects,
    )
    from serve.server import CHAT_FREE_FORM_VALUE_PATHS

    messages = build_all_fields_messages(api_type)
    object_paths = [
        object_path
        for object_path, _, _ in walk_chat_objects(
            messages, CHAT_FREE_FORM_VALUE_PATHS[api_type]
        )
    ]
    assert len(object_paths) == 8

    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    before = engine.chat_calls
    for object_path in object_paths:
        mutated, expected_param = copy_with_unknown_field(messages, object_path)
        status, data = _json(server_addr[:2], "POST", path, {
            "model": "raon-1-mini", "messages": mutated, "stream": False,
        })
        assert status == 400, (object_path, data)
        if api_type == "openai":
            assert data["error"]["param"] == expected_param
        else:
            assert expected_param in data["error"]
    assert engine.chat_calls == before


@pytest.mark.parametrize("index", [-1, True, 1.5, "0"])
def test_openai_tool_call_index_must_be_non_negative_integer(
    server_addr, index
):
    messages = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "index": index,
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }],
    }]
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    before = engine.chat_calls
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini", "messages": messages, "stream": False,
    })
    assert status == 400
    assert data["error"]["param"] == "messages.0.tool_calls.0.index"
    assert engine.chat_calls == before


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
@pytest.mark.parametrize(
    "tools,param",
    [
        ("lookup", "tools"),
        ({"type": "function"}, "tools"),
        ([{"type": "command", "function": {"name": "lookup"}}], "tools.0.type"),
        ([{"type": "function", "function": {}}], "tools.0.function.name"),
        ([{"type": "function", "function": {
            "name": "lookup", "parameters": []
        }}], "tools.0.function.parameters"),
    ],
)
def test_invalid_tools_are_400_before_engine_call(server_addr, path, tools, param):
    with Handler._busy_lock:
        engine = Handler._ensure_loaded_locked()
    before = engine.chat_calls
    status, data = _json(server_addr[:2], "POST", path, {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": tools,
        "stream": False,
    })
    assert status == 400
    if path.startswith("/v1/"):
        assert data["error"]["param"] == param
    else:
        assert param in data["error"]
    assert engine.chat_calls == before


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
def test_valid_tools_reach_engine(server_addr, path):
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    status, data = _json(server_addr[:2], "POST", path, {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": tools,
        "stream": False,
    })
    assert status == 200, data
    assert Handler.engine.last_tools == tools


def test_bifrost_openai_tool_history_is_accepted_and_normalized_for_template(server_addr):
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
    status, data = _json(server_addr[:2], "POST", "/v1/chat/completions", {
        "model": "raon-1-mini", "messages": messages, "stream": False,
    })
    assert status == 200, data
    passed = Handler.engine.last_messages
    assert passed[1]["reasoning_content"] == "checking"
    assert passed[1]["tool_calls"][0]["index"] == 0
    assert passed[1]["tool_calls"][0]["id"] == "call_1"
    assert passed[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}
    assert passed[2]["tool_call_id"] == "call_1"


def test_ollama_tool_history_fields_and_tools_reach_engine(server_addr):
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
    status, data = _json(server_addr[:2], "POST", "/api/chat", {
        "model": "raon-1-mini", "messages": messages, "tools": tools, "stream": False,
    })
    assert status == 200, data
    passed = Handler.engine.last_messages
    assert passed[1]["reasoning_content"] == "checking"
    assert "thinking" not in passed[1]
    assert passed[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}
    assert passed[2]["tool_name"] == "lookup"
    assert Handler.engine.last_tools == tools


@pytest.mark.parametrize("path,payload_key", [("/api/chat", "messages"), ("/api/generate", "prompt")])
@pytest.mark.parametrize("keep_alive,done_reason", [(300, "load"), (0, "unload")])
def test_ollama_empty_request_controls_model_lifecycle(
    server_addr, path, payload_key, keep_alive, done_reason
):
    host_port = server_addr[:2]
    payload = {"model": "raon-1-mini", payload_key: [] if payload_key == "messages" else "",
               "keep_alive": keep_alive, "stream": False}
    status, data = _json(host_port, "POST", path, payload)
    assert status == 200
    assert data["done"] is True
    assert data["done_reason"] == done_reason
    _wait_engine_released()
    _, health = _json(host_port, "GET", "/healthz")
    assert health["loaded"] is (done_reason == "load")
    if done_reason == "unload":
        with Handler._busy_lock:
            Handler._ensure_loaded_locked()
            Handler._set_expiry_locked(Handler.keep_alive)


def _raw_http_response(host: str, port: int, request: bytes) -> bytes:
    sock = socket.create_connection((host, port), timeout=3)
    try:
        sock.sendall(request)
        return sock.recv(4096)
    finally:
        sock.close()


def test_request_body_limit_and_bad_content_length(server_addr):
    host, port = server_addr[:2]
    oversized = _raw_http_response(host, port, (
        b"POST /api/chat HTTP/1.1\r\nHost: local\r\n"
        b"Content-Length: 16777217\r\nConnection: close\r\n\r\n"
    ))
    assert b" 413 " in oversized.split(b"\r\n", 1)[0]

    invalid = _raw_http_response(host, port, (
        b"POST /api/chat HTTP/1.1\r\nHost: local\r\n"
        b"Content-Length: nope\r\nConnection: close\r\n\r\n"
    ))
    assert b" 400 " in invalid.split(b"\r\n", 1)[0]


def test_non_stream_disconnect_cancels_engine(server_addr):
    host, port = server_addr[:2]
    engine = Handler.engine
    engine.cancel_log.clear()
    engine.cancel_callbacks.clear()
    old_sleep = engine.sleep_per_chunk
    try:
        engine.sleep_per_chunk = 0.15
        body = json.dumps({
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "disconnect"}],
            "stream": False,
        }).encode("utf-8")
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
            + body
        )
        sock = socket.create_connection((host, port), timeout=3)
        sock.sendall(request)
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not any(
            cb() for cb in engine.cancel_callbacks if cb is not None
        ):
            time.sleep(0.02)
        assert any(cb() for cb in engine.cancel_callbacks if cb is not None)
    finally:
        engine.sleep_per_chunk = old_sleep


@pytest.mark.parametrize("raise_after_chunks", [0, 1])
def test_single_openai_chat_engine_exception_without_socket(raise_after_chunks):
    """The sandbox blocks sockets, so also exercise the exception boundary directly."""
    handler = object.__new__(Handler)
    engine = FakeEngine("", "")
    engine.sleep_per_chunk = 0
    engine.raise_after_chunks = raise_after_chunks
    handler.engine = engine
    handler._check_cancel = lambda cancel: None
    errors = []
    handler._send_error = lambda status, message, **kwargs: errors.append(
        (status, message, kwargs)
    )

    handler._single_openai_chat(
        [{"role": "user", "content": "fail"}], None, GenOptions(), None
    )

    assert errors == [(500, "synthetic engine failure", {"code": "generation_error"})]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "bad", "content": "x"}],
        [{"role": "user", "content": [{}]}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        [{"role": "user", "content": 7}],
        [{"role": "tool", "content": "result", "tool_call_id": 1}],
    ],
)
def test_message_contract_validator_without_socket(messages):
    handler = object.__new__(Handler)
    failures = []
    handler._invalid = lambda message, param: failures.append((message, param)) or False

    assert handler._validate_messages(messages) is False
    assert len(failures) == 1


@pytest.mark.parametrize("api_type", ["openai", "ollama"])
def test_unknown_field_at_every_chat_object_validator_without_socket(api_type):
    from serve.chat_contract_test_utils import (
        build_all_fields_messages,
        copy_with_unknown_field,
        walk_chat_objects,
    )
    from serve.server import CHAT_FREE_FORM_VALUE_PATHS

    messages = build_all_fields_messages(api_type)
    object_paths = [
        object_path
        for object_path, _, _ in walk_chat_objects(
            messages, CHAT_FREE_FORM_VALUE_PATHS[api_type]
        )
    ]
    assert len(object_paths) == 8
    for object_path in object_paths:
        mutated, expected_param = copy_with_unknown_field(messages, object_path)
        failures = []
        handler = object.__new__(Handler)
        handler._invalid = lambda message, param: failures.append((message, param)) or False
        assert handler._validate_messages(
            mutated, ollama=(api_type == "ollama")
        ) is False
        assert failures[0][1] == expected_param


def test_message_contract_accepts_text_content_without_socket():
    handler = object.__new__(Handler)
    handler._invalid = lambda message, param: False

    assert handler._validate_messages([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": None, "reasoning_content": "thinking",
         "tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "call-1"},
    ]) is True


@pytest.mark.parametrize(
    "handler_name,payload_key,empty_value",
    [
        ("_handle_api_chat", "messages", []),
        ("_handle_api_generate", "prompt", ""),
    ],
)
@pytest.mark.parametrize("keep_alive,expected_reason", [(300, "load"), (0, "unload")])
def test_ollama_empty_lifecycle_without_socket(
    handler_name, payload_key, empty_value, keep_alive, expected_reason
):
    handler = object.__new__(Handler)
    handler.path = "/api/chat" if payload_key == "messages" else "/api/generate"
    handler.model_name = "raon-1-mini"
    handler.keep_alive = 300
    handler.GenOptions = GenOptions
    body = {
        "model": "raon-1-mini",
        payload_key: empty_value,
        "keep_alive": keep_alive,
        "stream": False,
    }
    waits = []
    releases = []
    responses = []
    handler._read_json = lambda: body
    handler._check_model = lambda model: True
    handler._wait_engine = lambda req_id, load=True: waits.append(load) or True
    handler._release_engine = lambda keep_alive_s: releases.append(keep_alive_s)
    handler._write_json = lambda status, response: responses.append((status, response))

    getattr(handler, handler_name)()

    assert waits == [expected_reason == "load"]
    assert releases == [float(keep_alive)]
    assert responses[0][0] == 200
    assert responses[0][1]["done"] is True
    assert responses[0][1]["done_reason"] == expected_reason


def test_ollama_normalize_messages_preserves_originals_and_moves_thinking():
    raw_messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "calling tool",
            "thinking": "need to lookup",
            "tool_calls": [{
                "function": {"name": "lookup", "arguments": {"city": "Seoul"}},
            }],
        },
        {"role": "tool", "tool_name": "lookup", "content": "result"},
    ]
    raw_copy = [dict(m) for m in raw_messages]

    normalized = Handler._normalize_ollama_messages(raw_messages)

    # 1. 원본 메시지 불변
    assert raw_messages[1]["thinking"] == "need to lookup"
    assert "reasoning_content" not in raw_messages[1]
    assert raw_messages[1] == raw_copy[1]

    # 2. 정규화 복사본에서 thinking -> reasoning_content 이동
    assert "thinking" not in normalized[1]
    assert normalized[1]["reasoning_content"] == "need to lookup"
    assert normalized[1]["content"] == "calling tool"

    # 3. 객체형 tool arguments 및 tool_name 유지
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}
    assert normalized[2]["tool_name"] == "lookup"


# --------------------------------------------------------------------------- #
# 적대적 필드 검증 시험 (AST 기반 자동 추출)
# --------------------------------------------------------------------------- #

def _collect_ast_request_fields_and_endpoints() -> tuple[set[str], list[str]]:
    server_py = Path(__file__).resolve().parent / "server.py"
    tree = ast.parse(server_py.read_text(encoding="utf-8"))

    var_to_prefix = {
        "body": "",
        "opts": "options.",
        "options": "options.",
        "stream_options": "stream_options.",
        "template_kwargs": "chat_template_kwargs.",
        "ct_kwargs": "chat_template_kwargs.",
    }
    non_request_callers = {
        "os.environ",
        "self.headers",
        "data",
        "b",
        "message",
        "copied",
        "part",
        "tool_call",
        "tool",
        "function",
        "record",
        "cur",
        "res",
        "parsed_opts",
        "filtered_opts",
        "delta",
        "choice",
        "usage",
        "err",
        "model_info",
        "details",
        "info",
        "item",
        "r",
        "d",
        "os.path",
    }

    # 선언된 표 상수를 순회하는 for 루프의 범위 식별
    allowed_table_names = {
        "OPENAI_CHAT_UNSUPPORTED",
        "OPENAI_COMPLETIONS_UNSUPPORTED",
        "OLLAMA_CHAT_UNSUPPORTED",
        "OLLAMA_GENERATE_UNSUPPORTED",
        "OLLAMA_OPTIONS_UNSUPPORTED",
        "unsupported",
    }
    table_loop_subnodes: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            iter_src = ast.unparse(node.iter)
            is_table_loop = (
                iter_src in allowed_table_names
                or any(t in iter_src for t in allowed_table_names)
            )
            if is_table_loop and isinstance(node.target, ast.Name):
                for child in ast.walk(node):
                    table_loop_subnodes[id(child)] = node.target.id

    collected_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            caller = ast.unparse(node.func.value)
            assert caller in var_to_prefix or caller in non_request_callers, (
                f"Unexpected .get() caller {caller!r} at line {node.lineno}. "
                f"Must be registered in var_to_prefix or non_request_callers."
            )
            if caller in var_to_prefix:
                assert node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str), (
                    f"Dynamic .get({ast.unparse(node.args[0])}) not allowed on request dict {caller!r} at line {node.lineno}!"
                )
                field_name = node.args[0].value
                collected_fields.add(f"{var_to_prefix[caller]}{field_name}")

        elif isinstance(node, ast.Subscript):
            caller = ast.unparse(node.value)
            if caller in var_to_prefix:
                loop_var = table_loop_subnodes.get(id(node))
                if loop_var is not None and isinstance(node.slice, ast.Name) and node.slice.id == loop_var:
                    # 선언한 표 상수를 순회하는 subscript 허용
                    pass
                else:
                    assert isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str), (
                        f"Dynamic subscript {caller}[{ast.unparse(node.slice)}] not allowed at line {node.lineno}!"
                    )
                    field_name = node.slice.value
                    collected_fields.add(f"{var_to_prefix[caller]}{field_name}")

        elif isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    caller = ast.unparse(comp)
                    if caller in var_to_prefix:
                        loop_var = table_loop_subnodes.get(id(node))
                        if loop_var is not None and isinstance(node.left, ast.Name) and node.left.id == loop_var:
                            # 선언한 표 상수를 순회하는 membership 허용
                            pass
                        else:
                            assert isinstance(node.left, ast.Constant) and isinstance(node.left.value, str), (
                                f"Dynamic membership {ast.unparse(node.left)} in {caller} not allowed at line {node.lineno}!"
                            )
                            field_name = node.left.value
                            collected_fields.add(f"{var_to_prefix[caller]}{field_name}")

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_validate_bool":
            caller = ast.unparse(node.args[0])
            if caller in var_to_prefix:
                assert len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str), (
                    f"Dynamic _validate_bool({caller}, {ast.unparse(node.args[1])}) not allowed at line {node.lineno}!"
                )
                field_name = node.args[1].value
                collected_fields.add(f"{var_to_prefix[caller]}{field_name}")

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys", "values"):
            caller = ast.unparse(node.func.value)
            if caller in var_to_prefix:
                assert False, (
                    f"Dynamic iteration {caller}.{node.func.attr}() not allowed on request dict at line {node.lineno}!"
                )

    collected_fields.update(OPENAI_CHAT_UNSUPPORTED)
    collected_fields.update(OPENAI_COMPLETIONS_UNSUPPORTED)
    collected_fields.update(OLLAMA_CHAT_UNSUPPORTED)
    collected_fields.update(OLLAMA_GENERATE_UNSUPPORTED)
    for opt_k in OLLAMA_OPTIONS_UNSUPPORTED:
        collected_fields.add(f"options.{opt_k}")

    endpoints: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("do_POST", "_route_post"):
            for child in ast.walk(node):
                if isinstance(child, ast.Compare):
                    for comp in child.comparators:
                        if isinstance(comp, (ast.Tuple, ast.List)):
                            for elt in comp.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    endpoints.append(elt.value)
                        elif isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                            endpoints.append(comp.value)

    distinct_endpoints = list(dict.fromkeys(endpoints))
    return collected_fields, distinct_endpoints


def test_request_reading_style_rules():
    """요청 본문 읽기 방식 규칙(C2) 검증:
    1. 수집기 실행 시 비상수 접근이나 dict 순회가 있으면 실패.
    2. 새로 모이는 필드에 최상위 enable_thinking, think, raw가 반드시 포함되어야 함.
    """
    fields, endpoints = _collect_ast_request_fields_and_endpoints()
    assert "enable_thinking" in fields, "enable_thinking must be in collected fields"
    assert "think" in fields, "think must be in collected fields"
    assert "raw" in fields, "raw must be in collected fields"


def _base_req_for_endpoint(path: str) -> dict[str, Any]:
    norm = path.rstrip("/")
    if norm == "/v1/chat/completions":
        return {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}]}
    if norm == "/v1/completions":
        return {"model": "raon-1-mini", "prompt": "hi"}
    if norm == "/api/chat":
        return {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    if norm == "/api/generate":
        return {"model": "raon-1-mini", "prompt": "hi", "stream": False}
    if norm == "/api/show":
        return {"model": "raon-1-mini"}
    raise ValueError(f"Unknown POST endpoint: {path}")


def _build_adversarial_payload(base: dict[str, Any], field: str, val: Any) -> bytes:
    b = copy.deepcopy(base)
    parts = field.split(".")
    cur = b
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    k = parts[-1]
    if val == "__RAW_NAN__":
        cur[k] = "__SENTINEL__"
        return json.dumps(b).replace('"__SENTINEL__"', "NaN").encode("utf-8")
    if val == "__RAW_INF__":
        cur[k] = "__SENTINEL__"
        return json.dumps(b).replace('"__SENTINEL__"', "Infinity").encode("utf-8")
    if val == "__RAW_NEGINF__":
        cur[k] = "__SENTINEL__"
        return json.dumps(b).replace('"__SENTINEL__"', "-Infinity").encode("utf-8")
    cur[k] = val
    return json.dumps(b).encode("utf-8")


ADVERSARIAL_VALUES: list[tuple[str, Any]] = [
    ("raw_nan", "__RAW_NAN__"),
    ("raw_inf", "__RAW_INF__"),
    ("raw_neginf", "__RAW_NEGINF__"),
    ("1e308", 1e308),
    ("-1e308", -1e308),
    ("10**30", 10**30),
    ("10**400", 10**400),
    ("-10**400", -10**400),
    ("str_nan", "nan"),
    ("str_inf", "inf"),
    ("empty_str", ""),
    ("str_abc", "abc"),
    ("empty_list", []),
    ("empty_dict", {}),
    ("bool_true", True),
    ("null", None),
]


NON_FINITE_JSON_TOKENS = ("NaN", "Infinity", "-Infinity", "1e400")


def _all_fields_request_bodies() -> dict[str, dict[str, Any]]:
    """Build valid requests covering every field consumed by each generation path."""
    from serve.chat_contract_test_utils import build_all_fields_messages

    tool = {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
    openai_options = {
        "keep_alive": "5m",
        "max_tokens": 32,
        "max_completion_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "presence_penalty": 0.1,
        "seed": 7,
        "stop": ["done", "stop"],
        "stream": False,
        "enable_thinking": True,
        "reasoning_effort": "high",
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    ollama_options = {
        "num_predict": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "presence_penalty": 0.1,
        "seed": 7,
        "stop": ["done", "stop"],
    }
    return {
        "/v1/chat/completions": {
            "model": "raon-1-mini",
            "messages": build_all_fields_messages("openai"),
            "tools": [copy.deepcopy(tool)],
            **openai_options,
        },
        "/v1/completions": {
            "model": "raon-1-mini",
            "prompt": ["hello", " world"],
            **openai_options,
        },
        "/api/chat": {
            "model": "raon-1-mini",
            "messages": build_all_fields_messages("ollama"),
            "tools": [copy.deepcopy(tool)],
            "keep_alive": "5m",
            "stream": False,
            "think": True,
            "raw": False,
            "options": copy.deepcopy(ollama_options),
        },
        "/api/generate": {
            "model": "raon-1-mini",
            "prompt": "hello",
            "keep_alive": "5m",
            "stream": False,
            "think": True,
            "raw": False,
            "options": copy.deepcopy(ollama_options),
        },
    }


def _walk_json_value_paths(value: Any, path: tuple[str | int, ...] = ()):
    """Yield every dict value and list element path, recursively."""
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, list):
        children = enumerate(value)
    else:
        return
    for component, child in children:
        child_path = path + (component,)
        yield child_path
        yield from _walk_json_value_paths(child, child_path)


def _replace_path_with_raw_token(
    body: dict[str, Any], path: tuple[str | int, ...], token: str
) -> bytes:
    copied = copy.deepcopy(body)
    target: Any = copied
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = "__NON_FINITE_JSON_SENTINEL__"
    encoded = json.dumps(copied, ensure_ascii=False, allow_nan=False)
    quoted_sentinel = '"__NON_FINITE_JSON_SENTINEL__"'
    assert encoded.count(quoted_sentinel) == 1
    return encoded.replace(quoted_sentinel, token).encode("utf-8")


def _raw_json_request(host_port, path: str, payload: bytes) -> tuple[int, dict[str, Any]]:
    host, port = host_port
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request(
        "POST", path, body=payload, headers={"Content-Type": "application/json"}
    )
    response = conn.getresponse()
    response_body = response.read()
    status = response.status
    conn.close()
    return status, json.loads(response_body.decode("utf-8"))


def test_non_finite_json_number_at_every_tree_position_http(server_addr):
    """Every JSON value position rejects all non-finite spellings before the engine."""
    engine = Handler.engine
    before_chat = engine.chat_calls
    before_generate = engine.generate_calls
    position_counts: dict[str, int] = {}

    for path, body in _all_fields_request_bodies().items():
        paths = list(_walk_json_value_paths(body))
        position_counts[path] = len(paths)
        for value_path in paths:
            for token in NON_FINITE_JSON_TOKENS:
                status, response = _raw_json_request(
                    server_addr[:2], path, _replace_path_with_raw_token(body, value_path, token)
                )
                assert status == 400, (path, value_path, token, response)
                message = response["error"]["message"] if path.startswith("/v1/") else response["error"]
                assert "invalid JSON" in message
                assert "non-finite JSON number" in message

    assert position_counts
    assert engine.chat_calls == before_chat
    assert engine.generate_calls == before_generate


def test_non_finite_json_number_at_every_tree_position_without_socket():
    """The isolated request decoder rejects the same complete tree mutation matrix."""
    position_counts: dict[str, int] = {}
    for endpoint, body in _all_fields_request_bodies().items():
        paths = list(_walk_json_value_paths(body))
        position_counts[endpoint] = len(paths)
        for value_path in paths:
            for token in NON_FINITE_JSON_TOKENS:
                payload = _replace_path_with_raw_token(body, value_path, token)
                with pytest.raises(ValueError, match="non-finite JSON number"):
                    loads_strict(payload)
    assert position_counts


def assert_engine_options_and_handler_state(options: Any, handler_cls: Any) -> None:
    """단언 C3 & 1d:
    1. 엔진에 닿은 GenOptions의 모든 숫자 칸이 범위 표의 한계 안이고 float 칸이 유한하다.
       칸 이름 대응(max_tokens <- max_tokens, max_completion_tokens, num_predict)은 표에서 읽는다.
    2. 처리기의 keep-alive 상태(만료 시각, 설정값 등)가 유한하거나 영구(<0 or None)이다.
    """
    if options is not None:
        from serve.server import NUMERIC_OPTION_SPECS
        gen_to_specs: dict[str, tuple[str, ...]] = {
            "max_tokens": ("max_tokens", "max_completion_tokens", "num_predict"),
            "temperature": ("temperature",),
            "top_p": ("top_p",),
            "top_k": ("top_k",),
            "presence_penalty": ("presence_penalty",),
            "seed": ("seed",),
        }
        for gen_field, spec_keys in gen_to_specs.items():
            val = getattr(options, gen_field, None)
            if val is not None:
                assert isinstance(val, (int, float)), f"GenOptions.{gen_field} must be numeric, got {val!r}"
                if isinstance(val, float):
                    assert math.isfinite(val), f"GenOptions.{gen_field} must be finite, got {val}"
                for sk in spec_keys:
                    _kind, lower, upper = NUMERIC_OPTION_SPECS[sk]
                    if lower is not None:
                        assert val >= lower, f"GenOptions.{gen_field}={val} must be >= {lower} (from {sk})"
                    if upper is not None:
                        assert val <= upper, f"GenOptions.{gen_field}={val} must be <= {upper} (from {sk})"

    # keep_alive 검증
    with handler_cls._busy_lock:
        expiry = handler_cls._expires_at
        ka_sec = handler_cls.keep_alive
    if expiry is not None:
        assert math.isfinite(expiry), f"Handler._expires_at must be finite, got {expiry}"
    if ka_sec is not None and ka_sec >= 0:
        assert math.isfinite(ka_sec), f"Handler.keep_alive must be finite, got {ka_sec}"


def test_adversarial_request_fields_on_http_endpoints():
    """모든 AST 추출 필드와 적대값 조합을 실제 HTTP 소켓 경로로 전송하여 4대 단언 검증."""
    fields, endpoints = _collect_ast_request_fields_and_endpoints()
    assert len(fields) > 0
    assert len(endpoints) > 0

    orig_engine = Handler.engine
    orig_factory = Handler.engine_factory
    orig_keep_alive = Handler.keep_alive
    orig_max_wait = Handler.max_wait
    orig_logger = Handler.logger
    orig_model_name = Handler.model_name

    engine = FakeEngine("", "")
    engine.sleep_per_chunk = 0.0
    engine.thinking_text = ""
    engine.chat_text = "hi"
    engine.generate_text = "hi"
    Handler.engine = engine
    Handler.engine_factory = lambda: engine
    Handler.GenOptions = GenOptions
    Handler.Chunk = Chunk
    Handler.max_wait = 10.0
    Handler.keep_alive = 300.0
    Handler.logger = _LogWriter(None)
    Handler.model_name = engine.model_name

    class InterceptingServer(Server):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.handler_errors: list[Any] = []

        def handle_error(self, request, client_address):
            self.handler_errors.append(sys.exc_info())

    server = InterceptingServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        total_requests = 0
        for ep in endpoints:
            base = _base_req_for_endpoint(ep)
            for f in sorted(fields):
                for val_name, val in ADVERSARIAL_VALUES:
                    payload = _build_adversarial_payload(base, f, val)
                    before_chat = engine.chat_calls
                    before_gen = engine.generate_calls

                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    conn.request("POST", ep, body=payload, headers={"Content-Type": "application/json"})
                    resp = conn.getresponse()
                    response_body = resp.read()
                    conn.close()
                    total_requests += 1

                    if val in ("__RAW_NAN__", "__RAW_INF__", "__RAW_NEGINF__"):
                        assert resp.status == 400
                        error_body = json.loads(response_body.decode("utf-8"))
                        message = (
                            error_body["error"]["message"]
                            if ep.startswith("/v1/")
                            else error_body["error"]
                        )
                        assert "invalid JSON" in message
                        assert "non-finite JSON number" in message

                    # 단언 1: 응답은 200 또는 400 이다 (500·연결 끊김 없음; model/name 식별자가 변경되어 404가 되는 경우 제외)
                    assert resp.status in (200, 400) or (f in ("model", "name") and resp.status == 404), (
                        f"Unexpected status {resp.status} on endpoint={ep}, field={f}, value={val_name}"
                    )

                    # 단언 2: 400 이면 엔진이 불리지 않았다
                    if resp.status == 400:
                        assert engine.chat_calls == before_chat, (
                            f"Engine chat called on 400 error! endpoint={ep}, field={f}, value={val_name}"
                        )
                        assert engine.generate_calls == before_gen, (
                            f"Engine generate called on 400 error! endpoint={ep}, field={f}, value={val_name}"
                        )

                    # 단언 2b (C3): 200 이면 엔진 옵션 및 keep-alive 상태 유한성 검증
                    if resp.status == 200 and not ep.startswith("/api/show"):
                        assert_engine_options_and_handler_state(engine.last_options, Handler)

        # 단언 3: 처리기 스레드 예외가 0 건이다 (handle_error 가로채 기록)
        assert len(server.handler_errors) == 0, (
            f"Handler thread exceptions observed: {server.handler_errors}"
        )

        # 단언 4: 모두 보낸 뒤 서버가 정상 요청에 답한다
        for ep in endpoints:
            base = _base_req_for_endpoint(ep)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("POST", ep, body=json.dumps(base).encode("utf-8"), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200, f"Server failed to respond 200 to valid request on {ep} after adversarial run"
            _ = resp.read()
            conn.close()

    finally:
        server.shutdown()
        server.server_close()
        Handler.engine = orig_engine
        Handler.engine_factory = orig_factory
        Handler.keep_alive = orig_keep_alive
        Handler.max_wait = orig_max_wait
        Handler.logger = orig_logger
        Handler.model_name = orig_model_name


def test_c3_engine_options_and_handler_keep_alive():
    """C3 시험: 가짜 엔진 옵션 및 처리기 keep-alive 상태 검증 계약 단언.
    가짜 엔진 쪽 검사에서 keep_alive 검증을 빼면 이 시험이 실패한다.
    """
    valid_opts = GenOptions(temperature=0.7, top_p=0.9, presence_penalty=0.0)
    assert_engine_options_and_handler_state(valid_opts, Handler)

    bad_opts = GenOptions(temperature=float("nan"))
    with pytest.raises(AssertionError):
        assert_engine_options_and_handler_state(bad_opts, Handler)

    with Handler._busy_lock:
        old_expiry = Handler._expires_at
        Handler._expires_at = float("nan")
    try:
        with pytest.raises(AssertionError):
            assert_engine_options_and_handler_state(valid_opts, Handler)
    finally:
        with Handler._busy_lock:
            Handler._expires_at = old_expiry


def test_adversarial_request_fields_without_socket():
    """소켓 없이 검증 함수·해석 함수에 동일한 AST 필드와 적대값을 직접 주입하여 검증."""
    fields, endpoints = _collect_ast_request_fields_and_endpoints()

    from serve.server import parse_keep_alive

    # 1. keep_alive 해석 함수 직접 검증
    # 유효하지 않은 적대값들은 반드시 ValueError를 발생시켜야 함
    invalid_keep_alive_values = [
        "__RAW_NAN__", "__RAW_INF__", "__RAW_NEGINF__",
        1e308, -1e308, 10**30, 10**400, -10**400, "nan", "inf", "", "abc", [], {}
    ]
    for bad in invalid_keep_alive_values:
        val = float("nan") if bad == "__RAW_NAN__" else (
            float("inf") if bad == "__RAW_INF__" else (
                float("-inf") if bad == "__RAW_NEGINF__" else bad
            )
        )
        try:
            parse_keep_alive(val, 300.0)
            assert False, f"parse_keep_alive expected ValueError for {bad!r}"
        except ValueError:
            pass

    # 2. 유효한 keep_alive 값은 정상 통과
    assert parse_keep_alive(None, 300.0) == 300.0
    assert parse_keep_alive(True, 300.0) == 300.0
    assert parse_keep_alive(False, 300.0) == 0.0
    assert parse_keep_alive(0, 300.0) == 0.0
    assert parse_keep_alive(-1, 300.0) == -1.0
    assert parse_keep_alive("300", 300.0) == 300.0
    assert parse_keep_alive("300s", 300.0) == 300.0
    assert parse_keep_alive("5m", 300.0) == 300.0
    assert parse_keep_alive("1h", 300.0) == 3600.0
    assert parse_keep_alive("500ms", 300.0) == 0.5

    # 3. DummyHandler를 통한 OpenAI / Ollama body 검증 함수 직접 호출
    class DummyHandler(Handler):
        def __init__(self):
            self.errors: list[tuple[int, str, Any, Any]] = []
            self.keep_alive = 300.0
            self.model_name = "raon-1-mini"

        def _send_error(self, status: int, message: str, param: str | None = None, code: str | None = None) -> None:
            self.errors.append((status, message, param, code))

    handler = DummyHandler()

    for ep in endpoints:
        base = _base_req_for_endpoint(ep)
        for f in sorted(fields):
            for val_name, val in ADVERSARIAL_VALUES:
                py_val = float("nan") if val == "__RAW_NAN__" else (
                    float("inf") if val == "__RAW_INF__" else (
                        float("-inf") if val == "__RAW_NEGINF__" else val
                    )
                )
                b = copy.deepcopy(base)
                parts = f.split(".")
                cur = b
                for p in parts[:-1]:
                    if not isinstance(cur.get(p), dict):
                        cur[p] = {}
                    cur = cur[p]
                cur[parts[-1]] = py_val

                handler.errors.clear()
                is_openai = ep.startswith("/v1/")
                is_chat = "chat" in ep
                if is_openai:
                    ok = handler._validate_openai_body(b, chat=is_chat)
                elif "chat" in ep or "generate" in ep:
                    ok = handler._validate_ollama_body(b, chat=is_chat)
                else:
                    ok = True

                assert isinstance(ok, bool)
                if not ok:
                    assert len(handler.errors) > 0
                    assert handler.errors[-1][0] == 400


def test_validated_numeric_options_accepted_by_engine():
    """검증기가 받는 값이면 실제 엔진 코드도 예외 없이 통과함을 검증."""
    import numpy as np
    from serve.server import NUMERIC_OPTION_SPECS, Handler
    from serve.test_engine_small import Engine, FakeModel, FakeTokenizer, DEFAULT_EOS_IDS
    from serve.test_engine_small import GenOptions as RealGenOptions

    fake_tok = FakeTokenizer()

    def make_fake_model():
        tok_a = 1000
        tok_b = 1001
        eos = DEFAULT_EOS_IDS[0]
        tokens = [tok_a, tok_b, eos]
        cursor = 0

        def logits_fn(_tid: int, _hist: list[int]) -> np.ndarray:
            nonlocal cursor
            arr = np.full(260000, -50.0, dtype=np.float32)
            if cursor < len(tokens):
                target = tokens[cursor]
                cursor += 1
            else:
                target = eos
            arr[target] = 20.0
            return arr

        return FakeModel(logits_fn=logits_fn)

    class DummyHandler(Handler):
        def __init__(self):
            self.errors: list[Any] = []
            self.keep_alive = 300.0
            self.model_name = "raon-1-mini"
            self.GenOptions = RealGenOptions

        def _send_error(self, status: int, message: str, *, param: str | None = None, code: str | None = None) -> None:
            self.errors.append((status, message, param, code))

    handler = DummyHandler()

    tested_combinations = 0
    total_engine_runs = 0

    for field, (kind, lower, upper) in NUMERIC_OPTION_SPECS.items():
        # 극단값 후보 생성:
        # - 하한·상한(있으면).
        # - 열린 쪽은 int면 ±10**400, float면 ±1e308.
        # - 0과 -1.
        cands: list[Any] = []
        if lower is not None:
            cands.append(lower)
        else:
            cands.append(-10**400 if kind is int else -1e308)
        if upper is not None:
            cands.append(upper)
        else:
            cands.append(10**400 if kind is int else 1e308)
        cands.append(0 if kind is int else 0.0)
        cands.append(-1 if kind is int else -1.0)

        unique_cands = list(dict.fromkeys(cands))

        for val in unique_cands:
            # 소켓 없는 검증기로 받는지 먼저 묻는다
            if not handler._validate_numbers({field: val}):
                continue

            tested_combinations += 1

            # 1. 서버의 실제 옵션 해석 함수(_parse_openai_options·_parse_ollama_options)로 GenOptions 생성
            # OpenAI는 최상위, Ollama는 options. 아래
            runs = []
            if field != "num_predict":
                opts_openai = handler._parse_openai_options({field: val, "stop": ["STOP"]})
                runs.append(("openai", opts_openai))
            if field != "max_completion_tokens":
                opts_ollama = handler._parse_ollama_options({"options": {field: val, "stop": ["STOP"]}})
                runs.append(("ollama", opts_ollama))

            for _proto, opts in runs:
                # 2. 실제 Engine 생성 경로를 가짜 모델로 몇 토큰 돌린다.
                # Engine._run_stream이 seed·표본추출·최대 토큰·정지 조건을 실제로 쓰는 경로를 지난다.
                # 3. 예외 없이 끝나야 한다.
                engine = Engine(None, fake_tok, model=make_fake_model())
                chunks = list(engine.generate("hello", raw=True, options=opts))
                assert len(chunks) > 0
                total_engine_runs += 1

    assert tested_combinations == 20
    assert total_engine_runs == 36


def test_numeric_option_specs_have_both_bounds():
    """단언 1a: NUMERIC_OPTION_SPECS의 모든 칸에 아래·위 한계가 둘 다 있다(None 없음)."""
    from serve.server import NUMERIC_OPTION_SPECS
    for field, (kind, lower, upper) in NUMERIC_OPTION_SPECS.items():
        assert lower is not None, f"NUMERIC_OPTION_SPECS[{field!r}] lower bound must not be None"
        assert upper is not None, f"NUMERIC_OPTION_SPECS[{field!r}] upper bound must not be None"
        assert lower <= upper, f"NUMERIC_OPTION_SPECS[{field!r}] lower ({lower}) must be <= upper ({upper})"


def test_server_cli_numeric_arguments_use_ranged_parser():
    """서버 CLI 동작(_actions) 중 숫자 기본값 또는 int/float 타입인 동작은 모두 범위를 가진 파서여야 한다 (B1 & 1b)."""
    parser = _build_parser()
    allowed_exceptions = {parse_keep_alive}
    for action in parser._actions:
        if not action.option_strings:
            continue
        opt_name = action.option_strings[0]
        if action.type in allowed_exceptions:
            continue
        is_num_default = isinstance(action.default, (int, float)) and not isinstance(action.default, bool)
        is_num_type = action.type in (int, float)

        if is_num_default or is_num_type:
            assert getattr(action.type, "is_ranged", False) is True, (
                f"Action {opt_name} must use a ranged parser, got type={action.type!r}, default={action.default!r}"
            )
            assert hasattr(action.type, "min_val") and action.type.min_val is not None, (
                f"Action {opt_name} ranged parser min_val must not be None"
            )
            assert hasattr(action.type, "max_val") and action.type.max_val is not None, (
                f"Action {opt_name} ranged parser max_val must not be None"
            )


def test_server_cli_ranged_parser_boundary_and_non_finite():
    """서버 CLI의 모든 ranged parser에 nan, inf, -inf 및 범위 밖 값을 주면 SystemExit(2) 검증 (B2 & 1c)."""
    parser = _build_parser()
    for action in parser._actions:
        if not getattr(action.type, "is_ranged", False):
            continue
        opt = action.option_strings[0]
        ptype = action.type

        # 1. nan, inf, -inf
        for bad_str in ("nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, bad_str])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={bad_str}"

        # 2. 범위 밖 값 (위 한계 바로 위, 10**400, 아래 한계 바로 아래)
        min_v = ptype.min_val
        max_v = ptype.max_val
        type_cls = getattr(ptype, "type_cls", float)

        if min_v is not None:
            if getattr(ptype, "min_inclusive", True):
                below = min_v - 1 if type_cls is int else min_v - 0.001
            else:
                below = min_v
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, str(below)])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={below} (below min {min_v})"

        if max_v is not None:
            if getattr(ptype, "max_inclusive", True):
                above = max_v + 1 if type_cls is int else max_v + 0.001
            else:
                above = max_v
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, str(above)])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={above} (above max {max_v})"

        # 2-b. 거대 수 10**400, -10**400, 1e308, -1e308
        huge_int_str = "1" + "0" * 400
        neg_huge_int_str = "-1" + "0" * 400
        for huge_str in (huge_int_str, neg_huge_int_str, "1e308", "-1e308"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, huge_str])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={huge_str}"

        # 3. 유효한 기본값은 통과
        if action.default is not None:
            args = parser.parse_args([opt, str(action.default)])
            dest = action.dest
            assert getattr(args, dest) == action.default


def test_numeric_options_boundary_and_adversarial_rejected():
    """단언 1c: 표마다 위 한계 바로 위, 10**400, 아래 한계 바로 아래가 HTTP 400이다.
    HTTP는 OpenAI 최상위와 Ollama options. 둘 다 본다.
    """
    from serve.server import NUMERIC_OPTION_SPECS, Handler

    class DummyHandler(Handler):
        def __init__(self):
            self.errors: list[Any] = []

        def _send_error(self, status: int, message: str, *, param: str | None = None, code: str | None = None) -> None:
            self.errors.append((status, message, param, code))

    handler = DummyHandler()

    for field, (kind, lower, upper) in NUMERIC_OPTION_SPECS.items():
        assert lower is not None
        assert upper is not None
        if kind is int:
            bad_values = [
                lower - 1,
                upper + 1,
                10**400,
                -(10**400),
            ]
        else:
            bad_values = [
                lower - 0.001,
                upper + 0.001,
                10**400,
                -(10**400),
                float("inf"),
                float("-inf"),
                float("nan"),
            ]

        for bad_val in bad_values:
            # 1. _validate_numbers 자체 검증
            assert not handler._validate_numbers({field: bad_val}), (
                f"Field {field}={bad_val} must be rejected by _validate_numbers"
            )

            # 2. OpenAI 최상위 경로 검증 (num_predict는 Ollama 전용이므로 제외)
            if field != "num_predict":
                openai_body = {
                    "model": "raon-1-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    field: bad_val,
                }
                handler.errors.clear()
                ok = handler._validate_openai_body(openai_body, chat=True)
                assert ok is False, f"OpenAI top-level {field}={bad_val} must fail validation"
                assert len(handler.errors) > 0
                assert handler.errors[-1][0] == 400, f"Expected HTTP 400 for {field}={bad_val}"

            # 3. Ollama options. 경로 검증 (max_completion_tokens는 OpenAI 전용이므로 제외)
            if field != "max_completion_tokens":
                ollama_body = {
                    "model": "raon-1-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {field: bad_val},
                }
                handler.errors.clear()
                ok = handler._validate_ollama_body(ollama_body, chat=True)
                assert ok is False, f"Ollama options.{field}={bad_val} must fail validation"
                assert len(handler.errors) > 0
                assert handler.errors[-1][0] == 400, f"Expected HTTP 400 for options.{field}={bad_val}"


def test_ast_all_json_dumps_allow_nan_false():
    """단언 2b: ast로 serve/의 비시험 파일과 npu/run_golden.py를 읽어,
    모든 json.dump·json.dumps 호출에 allow_nan=False가 있음을 단언한다.
    """
    import ast
    repo_root = Path(__file__).resolve().parent.parent

    target_files: list[Path] = []
    # serve/ 내 비시험 파일들
    serve_dir = repo_root / "serve"
    for py_file in sorted(serve_dir.glob("*.py")):
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
            continue
        target_files.append(py_file)

    # npu/run_golden.py
    run_golden_file = repo_root / "npu" / "run_golden.py"
    assert run_golden_file.is_file()
    target_files.append(run_golden_file)

    violations: list[str] = []

    class JsonDumpVisitor(ast.NodeVisitor):
        def __init__(self, filepath: Path):
            self.filepath = filepath

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            is_json_dump = False
            fn_name = ""
            if isinstance(func, ast.Attribute) and func.attr in ("dump", "dumps"):
                if isinstance(func.value, ast.Name) and func.value.id == "json":
                    is_json_dump = True
                    fn_name = f"json.{func.attr}"

            if is_json_dump:
                has_allow_nan_false = False
                for kw in node.keywords:
                    if kw.arg == "allow_nan":
                        if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            has_allow_nan_false = True
                            break
                if not has_allow_nan_false:
                    violations.append(
                        f"{self.filepath.name}:{node.lineno}: call to {fn_name}() without allow_nan=False"
                    )

            self.generic_visit(node)

    for target_path in target_files:
        tree = ast.parse(target_path.read_text(encoding="utf-8"), filename=str(target_path))
        visitor = JsonDumpVisitor(target_path)
        visitor.visit(tree)

    assert not violations, "Found json.dump/dumps calls without allow_nan=False:\n" + "\n".join(violations)


def _sample_tool_call_arguments_object() -> dict[str, Any]:
    return {
        "location": "Seoul",
        "days": 3,
        "details": {
            "include_alerts": True,
            "metrics": ["temp", "rain"],
            "threshold": 0.5,
        },
    }


def _make_tool_call_openai_payload(arguments_str: str) -> dict[str, Any]:
    return {
        "model": "raon-1-mini",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": arguments_str,
                        },
                    }
                ],
            },
        ],
    }


def test_tool_call_arguments_non_finite_json_tree_http(server_addr):
    """단언 1b (HTTP): OpenAI chat tool call arguments 내부 JSON 나무의 모든 값 자리에
    NaN, Infinity, -Infinity, 1e400을 넣으면 각각 400이고 엔진이 호출되지 않아야 한다.
    깊은 중첩(예: 10만 겹 배열) arguments도 400이어야 한다.
    """
    engine = Handler.engine
    before_chat = engine.chat_calls
    before_generate = engine.generate_calls

    inner_obj = _sample_tool_call_arguments_object()
    paths = list(_walk_json_value_paths(inner_obj))
    assert len(paths) >= 5

    for value_path in paths:
        for token in NON_FINITE_JSON_TOKENS:
            bad_args_bytes = _replace_path_with_raw_token(inner_obj, value_path, token)
            bad_args_str = bad_args_bytes.decode("utf-8")
            body = _make_tool_call_openai_payload(bad_args_str)
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")

            status, response = _raw_json_request(server_addr[:2], "/v1/chat/completions", payload)
            assert status == 400, (value_path, token, response)
            assert "error" in response
            assert engine.chat_calls == before_chat
            assert engine.generate_calls == before_generate

    # 깊은 중첩(10만 겹 배열) arguments도 400
    deep_nested_args = "[" * 100000 + "]" * 100000
    body = _make_tool_call_openai_payload(deep_nested_args)
    payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    status, response = _raw_json_request(server_addr[:2], "/v1/chat/completions", payload)
    assert status == 400
    assert engine.chat_calls == before_chat
    assert engine.generate_calls == before_generate


def test_tool_call_arguments_non_finite_json_tree_without_socket():
    """단언 1b (소켓 없는 짝): OpenAI chat tool call arguments 내부 JSON 나무의 모든 값 자리에
    NaN, Infinity, -Infinity, 1e400이 있거나 10만 겹 중첩일 때 Handler 검증에서 400으로 실패해야 한다.
    """
    from serve.server import Handler

    class DummyHandler(Handler):
        def __init__(self):
            self.errors: list[Any] = []

        def _send_error(self, status: int, message: str, *, param: str | None = None, code: str | None = None) -> None:
            self.errors.append((status, message, param, code))

    handler = DummyHandler()
    inner_obj = _sample_tool_call_arguments_object()
    paths = list(_walk_json_value_paths(inner_obj))
    assert len(paths) >= 5

    for value_path in paths:
        for token in NON_FINITE_JSON_TOKENS:
            bad_args_bytes = _replace_path_with_raw_token(inner_obj, value_path, token)
            bad_args_str = bad_args_bytes.decode("utf-8")
            body = _make_tool_call_openai_payload(bad_args_str)

            handler.errors.clear()
            ok = handler._validate_openai_body(body, chat=True)
            assert ok is False, (value_path, token)
            assert len(handler.errors) > 0
            assert handler.errors[-1][0] == 400

    # 깊은 중첩(10만 겹 배열) arguments도 400
    deep_nested_args = "[" * 100000 + "]" * 100000
    body = _make_tool_call_openai_payload(deep_nested_args)
    handler.errors.clear()
    ok = handler._validate_openai_body(body, chat=True)
    assert ok is False
    assert len(handler.errors) > 0
    assert handler.errors[-1][0] == 400


def test_ast_engine_and_tokenize_do_not_import_server():
    """단언 1c: ast로 serve/engine.py와 serve/tokenize.py를 읽어,
    serve.server 모듈을 import하지 않음을 단언한다.
    """
    import ast
    repo_root = Path(__file__).resolve().parent.parent

    target_files = [
        repo_root / "serve" / "engine.py",
        repo_root / "serve" / "tokenize.py",
    ]

    class ServerImportVisitor(ast.NodeVisitor):
        def __init__(self, filepath: Path):
            self.filepath = filepath
            self.violations: list[str] = []

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name == "serve.server" or alias.name.startswith("serve.server."):
                    self.violations.append(
                        f"{self.filepath.name}:{node.lineno}: import {alias.name}"
                    )
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            mod = node.module or ""
            if mod == "serve.server" or mod.startswith("serve.server."):
                self.violations.append(
                    f"{self.filepath.name}:{node.lineno}: from {mod} import ..."
                )
            elif mod == "serve":
                for alias in node.names:
                    if alias.name == "server":
                        self.violations.append(
                            f"{self.filepath.name}:{node.lineno}: from serve import server"
                        )
            self.generic_visit(node)

    all_violations: list[str] = []
    for target_path in target_files:
        assert target_path.is_file(), f"Target file not found: {target_path}"
        tree = ast.parse(target_path.read_text(encoding="utf-8"), filename=str(target_path))
        visitor = ServerImportVisitor(target_path)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, (
        "Found serve.server imports in engine/tokenize:\n" + "\n".join(all_violations)
    )


# --------------------------------------------------------------------------- #
# fix14: 2절 (엔진 재적재 실패 500 및 처리기 밖 예외 감싸기)
# --------------------------------------------------------------------------- #

def test_engine_reload_failure_table(server_addr):
    """표 시험: 엔진을 쓰는 모든 경로에서 keep-alive 후 재적재 실패 시 500 반환 및 복구.

    대상:
    - OpenAI chat (stream / non-stream)
    - OpenAI completions (stream / non-stream)
    - Ollama chat (stream / non-stream)
    - Ollama generate (stream / non-stream)
    - Ollama load-only (chat 빈 messages / generate 빈 prompt)
    """
    host, port, _ = server_addr
    orig_factory = Handler.engine_factory
    target_endpoints = [
        ("POST", "/v1/chat/completions", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": False}, True),
        ("POST", "/v1/chat/completions", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True}, True),
        ("POST", "/v1/completions", {"model": "raon-1-mini", "prompt": "hi", "stream": False}, True),
        ("POST", "/v1/completions", {"model": "raon-1-mini", "prompt": "hi", "stream": True}, True),
        ("POST", "/api/chat", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": False}, False),
        ("POST", "/api/chat", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True}, False),
        ("POST", "/api/generate", {"model": "raon-1-mini", "prompt": "hi", "stream": False}, False),
        ("POST", "/api/generate", {"model": "raon-1-mini", "prompt": "hi", "stream": True}, False),
        ("POST", "/api/chat", {"model": "raon-1-mini", "messages": []}, False),
        ("POST", "/api/generate", {"model": "raon-1-mini", "prompt": ""}, False),
    ]

    try:
        for method, path, body, is_openai in target_endpoints:
            # 1. 엔진을 유휴 상태로 언로드
            with Handler._busy_lock:
                Handler._unload_locked()
            assert not Handler._is_loaded()

            # 2. 재적재 시 예외를 던지는 팩토리 설정
            def failing_factory():
                raise RuntimeError("synthetic NPU reload kernel load error")

            Handler.engine_factory = failing_factory

            # 3. 요청 전송
            status, data, _ = _request((host, port), method, path, body)
            assert status == 500, f"Expected 500 on {path}, got {status}: {data.decode('utf-8')}"
            resp_json = json.loads(data.decode("utf-8"))

            if is_openai:
                assert "error" in resp_json
                err = resp_json["error"]
                assert err["type"] == "server_error"
                assert err["code"] == "internal_error"
                assert "model load failed" in err["message"]
            else:
                assert "error" in resp_json
                assert "model load failed" in resp_json["error"]

            # 4. 잠금 해제 상태 검증
            _wait_engine_released()

            # 5. 팩토리를 정상으로 복구하고 다음 요청이 200으로 성공함을 확인
            def good_factory():
                return FakeEngine("", "")

            Handler.engine_factory = good_factory
            # 정상 요청 테스트
            recover_path = "/v1/models" if is_openai else "/api/tags"
            status_rec, data_rec, _ = _request((host, port), "GET", recover_path)
            assert status_rec == 200

            # 엔진 적재가 수반되는 요청으로 200 및 잠금 해제 확인
            norm_body = dict(body)
            if "stream" in norm_body:
                norm_body["stream"] = False
            status_gen, data_gen, _ = _request((host, port), method, path, norm_body)
            assert status_gen == 200
            _wait_engine_released()
    finally:
        Handler.engine_factory = orig_factory


def test_unexpected_handler_exception_returns_500(server_addr, monkeypatch):
    """처리기 하나가 임의 예외를 던지게 바꾼 시험 (GET·POST·HEAD 각각) -> 500과 맞는 형식."""
    host, port, _ = server_addr

    # 1. GET 예외 (Ollama 형식 엔드포인트)
    def bad_tags(self):
        raise ValueError("synthetic GET error")

    monkeypatch.setattr(Handler, "_handle_api_tags", bad_tags)
    status, data, _ = _request((host, port), "GET", "/api/tags")
    assert status == 500
    res = json.loads(data.decode("utf-8"))
    assert res == {"error": "synthetic GET error"}

    # 2. GET 예외 (OpenAI 형식 엔드포인트)
    def bad_v1_models(self):
        raise RuntimeError("synthetic OpenAI GET error")

    monkeypatch.setattr(Handler, "_handle_v1_models", bad_v1_models)
    status, data, _ = _request((host, port), "GET", "/v1/models")
    assert status == 500
    res = json.loads(data.decode("utf-8"))
    assert res["error"]["message"] == "synthetic OpenAI GET error"
    assert res["error"]["code"] == "internal_error"
    assert res["error"]["type"] == "server_error"

    # 3. POST 예외 (Ollama show)
    def bad_show(self):
        raise TypeError("synthetic POST error")

    monkeypatch.setattr(Handler, "_handle_api_show", bad_show)
    status, data, _ = _request((host, port), "POST", "/api/show", {"name": "raon-1-mini"})
    assert status == 500
    res = json.loads(data.decode("utf-8"))
    assert res == {"error": "synthetic POST error"}

    # 4. HEAD 예외
    def bad_head(self):
        raise OSError("synthetic HEAD error")

    monkeypatch.setattr(Handler, "_route_head", bad_head)
    status, data, _ = _request((host, port), "HEAD", "/")
    assert status == 500
    assert len(data) == 0  # HEAD는 본문 없이 500


def test_stream_chunk_exception_sends_error_payload_without_second_header(server_addr):
    """스트림 생성 중 예외: 스트림 처리기가 오류 조각을 보낸 뒤 두 번째 응답 머리를 쓰지 않고(HTTP/1. 줄 1회) 연결 종료."""
    host, port, engine = server_addr

    # 첫 청크 이후 예외를 던지도록 가짜 엔진 설정
    engine.raise_after_chunks = 1
    engine.raise_exception = RuntimeError("stream chunk failure")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        body = json.dumps({"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode("utf-8")
        req = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        s.sendall(req)

        raw_received = b""
        s.settimeout(5.0)
        while True:
            try:
                data = s.recv(4096)
                if not data:
                    break
                raw_received += data
            except (socket.timeout, ConnectionResetError, OSError):
                break
        s.close()

        # 검증: HTTP/1. 상태 줄이 한 번만 나와야 함 (헤더 덧쓰기 방지)
        assert raw_received.count(b"HTTP/1.") == 1
        assert b"200 OK" in raw_received
        assert b" 500 " not in raw_received
    finally:
        engine.raise_after_chunks = None
        engine.raise_exception = None


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_exception_after_response_started_does_not_send_second_header(
    server_addr, monkeypatch, method
):
    """응답 시작 후(send_response + end_headers + 본문 일부) 처리기 밖으로 나온 예외:
    확인 1: 받은 바이트에 HTTP/1. 상태 줄이 정확히 1개다.
    확인 2: 500 상태 줄이 없다.
    확인 3: 서버가 연결을 닫는다 (읽기가 EOF로 끝난다).
    확인 4: 그다음 새 연결의 요청은 정상 처리된다.
    """
    host, port, _ = server_addr

    if method == "GET":
        def bad_route_get(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"partial-get-response")
            self.wfile.flush()
            raise RuntimeError("synthetic GET handler crash after response started")

        monkeypatch.setattr(Handler, "_route_get", bad_route_get)
        req = b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    else:
        def bad_route_post(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"partial":')
            self.wfile.flush()
            raise RuntimeError("synthetic POST handler crash after response started")

        monkeypatch.setattr(Handler, "_route_post", bad_route_post)
        body = b'{"name":"raon-1-mini"}'
        req = (
            b"POST /api/show HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.settimeout(5.0)
    s.sendall(req)

    raw_received = b""
    clean_eof = False
    while True:
        try:
            chunk = s.recv(4096)
            if not chunk:
                clean_eof = True
                break
            raw_received += chunk
        except (socket.timeout, ConnectionResetError, OSError):
            break
    s.close()

    # 확인 1: 받은 바이트에 HTTP/1. 상태 줄이 정확히 1개다.
    assert raw_received.count(b"HTTP/1.") == 1

    # 확인 2: 500 상태 줄이 없다.
    assert b" 500 " not in raw_received
    assert b"200 OK" in raw_received

    # 확인 3: 서버가 연결을 닫는다 (읽기가 EOF로 끝난다).
    assert clean_eof is True

    # 확인 4: 그다음 새 연결의 요청은 정상 처리된다.
    monkeypatch.undo()
    status, data = _json((host, port), "GET", "/healthz")
    assert status == 200
    assert data["status"] == "ok"


def test_ast_do_methods_only_call_safe_dispatch():
    """정적 규칙(ast): do_GET·do_POST·do_HEAD 의 본문은 감싸기 함수 호출 하나뿐이다."""
    server_path = Path(__file__).resolve().parent / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Handler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in ("do_GET", "do_POST", "do_HEAD"):
                    # docstring을 제외한 문장 목록
                    stmts = [
                        stmt for stmt in item.body
                        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
                    ]
                    assert len(stmts) == 1, f"{item.name} must have exactly one statement, got {len(stmts)}"
                    stmt = stmts[0]
                    assert isinstance(stmt, ast.Expr), f"{item.name} statement must be an Expr"
                    assert isinstance(stmt.value, ast.Call), f"{item.name} Expr must be a Call"
                    call_func = stmt.value.func
                    assert isinstance(call_func, ast.Attribute) and call_func.attr == "_safe_dispatch", (
                        f"{item.name} must call self._safe_dispatch, got {ast.dump(call_func)}"
                    )


# --------------------------------------------------------------------------- #
# fix14: 3절 (지원하지 않는 요청 칸 검증 및 prompt 배열 처리)
# --------------------------------------------------------------------------- #

OFFICIAL_OPENAI_CHAT_FIELDS = frozenset({
    "messages", "model", "audio", "frequency_penalty", "function_call", "functions",
    "logit_bias", "logprobs", "max_completion_tokens", "max_tokens", "metadata",
    "modalities", "n", "parallel_tool_calls", "prediction", "presence_penalty",
    "prompt_cache_key", "reasoning_effort", "response_format", "safety_identifier",
    "seed", "service_tier", "stop", "store", "stream", "stream_options", "temperature",
    "tool_choice", "tools", "top_logprobs", "top_p", "user", "verbosity", "web_search_options",
    # 서버 확장:
    "top_k", "enable_thinking", "chat_template_kwargs", "keep_alive",
})

OFFICIAL_OPENAI_COMPLETIONS_FIELDS = frozenset({
    "model", "prompt", "best_of", "echo", "frequency_penalty", "logit_bias", "logprobs",
    "max_tokens", "n", "presence_penalty", "seed", "stop", "stream", "stream_options",
    "suffix", "temperature", "top_p", "user",
    # 서버 확장:
    "top_k", "enable_thinking", "chat_template_kwargs", "keep_alive", "max_completion_tokens",
})

OFFICIAL_STREAM_OPTIONS_FIELDS = frozenset({
    "include_usage", "include_obfuscation",
})

OFFICIAL_OLLAMA_CHAT_FIELDS = frozenset({
    "model", "messages", "tools", "think", "format", "options", "stream", "keep_alive",
})

OFFICIAL_OLLAMA_GENERATE_FIELDS = frozenset({
    "model", "prompt", "suffix", "images", "think", "format", "options", "system",
    "template", "stream", "raw", "keep_alive", "context",
})

OFFICIAL_OLLAMA_OPTIONS_FIELDS = frozenset({
    "num_keep", "seed", "num_predict", "top_k", "top_p", "min_p", "typical_p",
    "repeat_last_n", "temperature", "repeat_penalty", "presence_penalty",
    "frequency_penalty", "penalize_newline", "stop", "numa", "num_ctx", "num_batch",
    "num_gpu", "main_gpu", "use_mmap", "num_thread", "mirostat", "mirostat_tau",
    "mirostat_eta", "tfs_z", "low_vram", "f16_kv", "vocab_only", "use_mlock",
})


def test_endpoint_parameter_tables_completeness():
    """표 완전성:
    (가) 공식 목록의 모든 칸이 정확히 한 분류에 있다.
    (나) 서버가 읽는 요청 칸 전부가 어느 끝점이든 '지원' 표나 메시지 칸 표에 있다.
    """
    # (가) 표 분류 분할 완전성
    endpoints = [
        ("OpenAI chat", OFFICIAL_OPENAI_CHAT_FIELDS, OPENAI_CHAT_SUPPORTED, OPENAI_CHAT_IGNORED, OPENAI_CHAT_UNSUPPORTED),
        ("OpenAI completions", OFFICIAL_OPENAI_COMPLETIONS_FIELDS, OPENAI_COMPLETIONS_SUPPORTED, OPENAI_COMPLETIONS_IGNORED, OPENAI_COMPLETIONS_UNSUPPORTED),
        ("stream_options", OFFICIAL_STREAM_OPTIONS_FIELDS, STREAM_OPTIONS_SUPPORTED, STREAM_OPTIONS_IGNORED, STREAM_OPTIONS_UNSUPPORTED),
        ("Ollama chat", OFFICIAL_OLLAMA_CHAT_FIELDS, OLLAMA_CHAT_SUPPORTED, OLLAMA_CHAT_IGNORED, OLLAMA_CHAT_UNSUPPORTED),
        ("Ollama generate", OFFICIAL_OLLAMA_GENERATE_FIELDS, OLLAMA_GENERATE_SUPPORTED, OLLAMA_GENERATE_IGNORED, OLLAMA_GENERATE_UNSUPPORTED),
        ("Ollama options", OFFICIAL_OLLAMA_OPTIONS_FIELDS, OLLAMA_OPTIONS_SUPPORTED, OLLAMA_OPTIONS_IGNORED, OLLAMA_OPTIONS_UNSUPPORTED),
    ]

    for name, official, supported, ignored, unsupported in endpoints:
        union = supported | ignored | unsupported
        assert union == official, f"{name} table missing or extra fields: diff={union ^ official}"
        assert supported.isdisjoint(ignored), f"{name} supported & ignored overlap: {supported & ignored}"
        assert supported.isdisjoint(unsupported), f"{name} supported & unsupported overlap: {supported & unsupported}"
        assert ignored.isdisjoint(unsupported), f"{name} ignored & unsupported overlap: {ignored & unsupported}"

    # (나) 서버가 읽는 요청 칸 전부가 어느 끝점이든 지원 표나 메시지 칸 표에 속하는지 AST로 수집/검증
    server_path = Path(__file__).resolve().parent / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"))

    req_dict_vars = {"body", "opts", "options", "stream_options", "template_kwargs", "ct_kwargs"}
    server_read_fields: set[str] = set()

    for node in ast.walk(tree):
        # .get("...")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            caller = ast.unparse(node.func.value)
            if caller in req_dict_vars and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                server_read_fields.add(node.args[0].value)
        # [...]
        elif isinstance(node, ast.Subscript):
            caller = ast.unparse(node.value)
            if caller in req_dict_vars and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                server_read_fields.add(node.slice.value)
        # "..." in ...
        elif isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    caller = ast.unparse(comp)
                    if caller in req_dict_vars and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                        server_read_fields.add(node.left.value)
        # _validate_bool(caller, "...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_validate_bool":
            caller = ast.unparse(node.args[0])
            if caller in req_dict_vars and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                server_read_fields.add(node.args[1].value)

    message_fields = set()
    for s in OPENAI_MESSAGE_ALLOWED_FIELDS.values():
        message_fields.update(s)
    for s in OLLAMA_MESSAGE_ALLOWED_FIELDS.values():
        message_fields.update(s)
    message_fields.update(OPENAI_CONTENT_PART_ALLOWED_FIELDS)
    message_fields.update(OLLAMA_CONTENT_PART_ALLOWED_FIELDS)
    message_fields.update(OPENAI_TOOL_CALL_ALLOWED_FIELDS)
    message_fields.update(OLLAMA_TOOL_CALL_ALLOWED_FIELDS)
    message_fields.update(OPENAI_TOOL_FUNCTION_ALLOWED_FIELDS)
    message_fields.update(OLLAMA_TOOL_FUNCTION_ALLOWED_FIELDS)

    all_supported_or_message_fields = (
        OPENAI_CHAT_SUPPORTED
        | OPENAI_COMPLETIONS_SUPPORTED
        | STREAM_OPTIONS_SUPPORTED
        | OLLAMA_CHAT_SUPPORTED
        | OLLAMA_GENERATE_SUPPORTED
        | OLLAMA_OPTIONS_SUPPORTED
        | message_fields
    )

    for field in server_read_fields:
        assert field in all_supported_or_message_fields, (
            f"Server read request field {field!r} is neither in any SUPPORTED table nor in message field tables!"
        )


def test_supported_parameters_enforced(server_addr):
    """지침 4(다): '지원은 지킨다' 표 시험.
    끝점마다, '지원' 칸마다 뜻 있는 값을 보낸 것과 뺀 것을 비교하여,
    가짜 엔진이 받은 값이나 응답에 반드시 차이가 발생함을 검증한다.
    """
    host, port, _ = server_addr
    sample_tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]

    # ------------------------------------------------------------------ #
    # 1. OpenAI Chat: /v1/chat/completions
    # 제외: "model", "messages"
    # ------------------------------------------------------------------ #
    chat_base = {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }

    # 1-1. stream: 스트림 SSE vs 비스트림 JSON
    st_off, d_off, _ = _request((host, port), "POST", "/v1/chat/completions", chat_base)
    assert st_off == 200
    st_on, d_on, _ = _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "stream": True})
    assert st_on == 200
    assert d_off.startswith(b"{")
    assert d_on.startswith(b"data: ")

    # 1-2. stream_options: include_usage=True vs False (스트림 환경)
    _, d_so_off, _ = _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "stream": True})
    _, d_so_on, _ = _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "stream": True, "stream_options": {"include_usage": True}})
    assert b'"usage"' not in d_so_off
    assert b'"usage"' in d_so_on

    # 1-3. stop: 엔진 옵션 차이
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    opt_base_stop = Handler.engine.last_options.stop if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "stop": ["custom_stop"]})
    assert Handler.engine.last_options.stop == ["custom_stop"]
    assert Handler.engine.last_options.stop != opt_base_stop

    # 1-4. tools: 엔진 도구 차이
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    assert Handler.engine.last_tools is None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "tools": sample_tools})
    assert Handler.engine.last_tools == sample_tools

    # 1-5. keep_alive: 처리기 만료 시각 차이
    with Handler._busy_lock:
        Handler._set_expiry_locked(100.0)
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "keep_alive": 9999})
    _wait_engine_released()
    assert Handler.keep_alive == 9999 or Handler._expires_at > time.monotonic() + 5000

    # 1-6. max_tokens / max_completion_tokens
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "max_tokens": 42})
    assert Handler.engine.last_options.max_tokens == 42
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "max_completion_tokens": 84})
    assert Handler.engine.last_options.max_tokens == 84

    # 1-7. temperature
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    base_temp = Handler.engine.last_options.temperature if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "temperature": 0.42})
    assert Handler.engine.last_options.temperature == 0.42
    assert Handler.engine.last_options.temperature != base_temp

    # 1-8. top_p
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    base_tp = Handler.engine.last_options.top_p if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "top_p": 0.88})
    assert Handler.engine.last_options.top_p == 0.88
    assert Handler.engine.last_options.top_p != base_tp

    # 1-9. top_k
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    base_tk = Handler.engine.last_options.top_k if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "top_k": 33})
    assert Handler.engine.last_options.top_k == 33
    assert Handler.engine.last_options.top_k != base_tk

    # 1-10. seed
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    base_seed = Handler.engine.last_options.seed if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "seed": 777})
    assert Handler.engine.last_options.seed == 777
    assert Handler.engine.last_options.seed != base_seed

    # 1-11. presence_penalty
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    base_pp = Handler.engine.last_options.presence_penalty if Handler.engine.last_options else None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "presence_penalty": 0.35})
    assert Handler.engine.last_options.presence_penalty == 0.35
    assert Handler.engine.last_options.presence_penalty != base_pp

    # 1-12. enable_thinking
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    assert Handler.engine.last_enable_thinking is None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "enable_thinking": False})
    assert Handler.engine.last_enable_thinking is False

    # 1-13. chat_template_kwargs
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    assert Handler.engine.last_enable_thinking is None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "chat_template_kwargs": {"enable_thinking": False}})
    assert Handler.engine.last_enable_thinking is False

    # 1-14. reasoning_effort
    _request((host, port), "POST", "/v1/chat/completions", chat_base)
    assert Handler.engine.last_enable_thinking is None
    _request((host, port), "POST", "/v1/chat/completions", {**chat_base, "reasoning_effort": "none"})
    assert Handler.engine.last_enable_thinking is False

    # ------------------------------------------------------------------ #
    # 2. OpenAI Completions: /v1/completions
    # 제외: "model", "prompt"
    # ------------------------------------------------------------------ #
    compl_base = {"model": "raon-1-mini", "prompt": "hello"}

    # 2-1. stream
    st_off, d_off, _ = _request((host, port), "POST", "/v1/completions", compl_base)
    assert st_off == 200
    st_on, d_on, _ = _request((host, port), "POST", "/v1/completions", {**compl_base, "stream": True})
    assert st_on == 200
    assert d_off.startswith(b"{")
    assert d_on.startswith(b"data: ")

    # 2-2. stream_options
    _, d_so_off, _ = _request((host, port), "POST", "/v1/completions", {**compl_base, "stream": True})
    _, d_so_on, _ = _request((host, port), "POST", "/v1/completions", {**compl_base, "stream": True, "stream_options": {"include_usage": True}})
    assert b'"usage"' not in d_so_off
    assert b'"usage"' in d_so_on

    # 2-3. stop
    _request((host, port), "POST", "/v1/completions", {**compl_base, "stop": ["compl_stop"]})
    assert Handler.engine.last_options.stop == ["compl_stop"]

    # 2-4. keep_alive
    _request((host, port), "POST", "/v1/completions", {**compl_base, "keep_alive": 8888})
    _wait_engine_released()
    assert Handler.keep_alive == 8888 or Handler._expires_at > time.monotonic() + 4000

    # 2-5. max_tokens / max_completion_tokens
    _request((host, port), "POST", "/v1/completions", {**compl_base, "max_tokens": 51})
    assert Handler.engine.last_options.max_tokens == 51
    _request((host, port), "POST", "/v1/completions", {**compl_base, "max_completion_tokens": 91})
    assert Handler.engine.last_options.max_tokens == 91

    # 2-6. temperature, top_p, top_k, seed, presence_penalty
    _request((host, port), "POST", "/v1/completions", {**compl_base, "temperature": 0.45})
    assert Handler.engine.last_options.temperature == 0.45
    _request((host, port), "POST", "/v1/completions", {**compl_base, "top_p": 0.81})
    assert Handler.engine.last_options.top_p == 0.81
    _request((host, port), "POST", "/v1/completions", {**compl_base, "top_k": 25})
    assert Handler.engine.last_options.top_k == 25
    _request((host, port), "POST", "/v1/completions", {**compl_base, "seed": 456})
    assert Handler.engine.last_options.seed == 456
    _request((host, port), "POST", "/v1/completions", {**compl_base, "presence_penalty": 0.15})
    assert Handler.engine.last_options.presence_penalty == 0.15

    # ------------------------------------------------------------------ #
    # 3. Ollama Chat: /api/chat
    # 제외: "model", "messages"
    # ------------------------------------------------------------------ #
    ollama_chat_base = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}]}

    # 3-1. tools
    _request((host, port), "POST", "/api/chat", {**ollama_chat_base, "tools": sample_tools})
    assert Handler.engine.last_tools == sample_tools

    # 3-2. think
    _request((host, port), "POST", "/api/chat", {**ollama_chat_base, "think": False})
    assert Handler.engine.last_enable_thinking is False

    # 3-3. options
    _request((host, port), "POST", "/api/chat", {**ollama_chat_base, "options": {"temperature": 0.55}})
    assert Handler.engine.last_options.temperature == 0.55

    # 3-4. stream
    _, d_stream_default, _ = _request((host, port), "POST", "/api/chat", ollama_chat_base)
    _, d_stream_false, _ = _request((host, port), "POST", "/api/chat", {**ollama_chat_base, "stream": False})
    assert d_stream_default.count(b"\n") > 1
    assert d_stream_false.count(b"\n") <= 1

    # 3-5. keep_alive
    _request((host, port), "POST", "/api/chat", {**ollama_chat_base, "keep_alive": 7777})
    _wait_engine_released()
    assert Handler.keep_alive == 7777 or Handler._expires_at > time.monotonic() + 3000

    # ------------------------------------------------------------------ #
    # 4. Ollama Generate: /api/generate
    # 제외: "model", "prompt"
    # ------------------------------------------------------------------ #
    ollama_gen_base = {"model": "raon-1-mini", "prompt": "hi"}

    # 4-1. options
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"temperature": 0.65}})
    assert Handler.engine.last_options.temperature == 0.65

    # 4-2. stream
    _, d_gstream_default, _ = _request((host, port), "POST", "/api/generate", ollama_gen_base)
    _, d_gstream_false, _ = _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "stream": False})
    assert d_gstream_default.count(b"\n") > 1
    assert d_gstream_false.count(b"\n") <= 1

    # 4-3. raw
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "raw": True})
    assert Handler.engine.last_raw is True
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "raw": False})
    assert Handler.engine.last_raw is False

    # 4-4. keep_alive
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "keep_alive": 6666})
    _wait_engine_released()
    assert Handler.keep_alive == 6666 or Handler._expires_at > time.monotonic() + 3000

    # ------------------------------------------------------------------ #
    # 5. Ollama options: OLLAMA_OPTIONS_SUPPORTED
    # ------------------------------------------------------------------ #
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"num_predict": 37}})
    assert Handler.engine.last_options.max_tokens == 37
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"temperature": 0.22}})
    assert Handler.engine.last_options.temperature == 0.22
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"top_p": 0.66}})
    assert Handler.engine.last_options.top_p == 0.66
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"top_k": 19}})
    assert Handler.engine.last_options.top_k == 19
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"seed": 888}})
    assert Handler.engine.last_options.seed == 888
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"presence_penalty": 0.77}})
    assert Handler.engine.last_options.presence_penalty == 0.77
    _request((host, port), "POST", "/api/generate", {**ollama_gen_base, "options": {"stop": ["gen_stop"]}})
    assert Handler.engine.last_options.stop == ["gen_stop"]


def test_unsupported_parameters_table(server_addr):
    """미지원 칸 표: 뜻 있는 값 -> 400(param=칸이름), 뜻 없는 값 -> 통과."""
    host, port, _ = server_addr

    # OpenAI Chat 테스트
    openai_chat_cases = [
        ("n", 2, 1),
        ("frequency_penalty", 0.5, 0),
        ("logit_bias", {"50256": -100}, {}),
        ("logprobs", True, False),
        ("top_logprobs", 2, 0),
        ("response_format", {"type": "json_object"}, {"type": "text"}),
        ("modalities", ["text", "audio"], ["text"]),
        ("audio", {"voice": "alloy"}, None),
        ("prediction", {"type": "content"}, None),
        ("web_search_options", {"search": True}, None),
        ("functions", [{"name": "f"}], None),
        ("function_call", "auto", None),
        ("tool_choice", "required", "auto"),
        ("parallel_tool_calls", False, True),
    ]

    for param, bad_val, good_val in openai_chat_cases:
        # 뜻 있는 값 -> 400
        bad_body = {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "hi"}],
            param: bad_val,
        }
        status, data, _ = _request((host, port), "POST", "/v1/chat/completions", bad_body)
        assert status == 400, f"Expected 400 for bad {param}={bad_val!r}, got {status}: {data.decode('utf-8')}"
        res = json.loads(data.decode("utf-8"))
        assert res["error"]["param"] == param
        assert res["error"]["code"] == "unsupported_parameter"

        # 뜻 없는 값 -> 통과 (200)
        good_body = {
            "model": "raon-1-mini",
            "messages": [{"role": "user", "content": "hi"}],
            param: good_val,
        }
        status_ok, data_ok, _ = _request((host, port), "POST", "/v1/chat/completions", good_body)
        assert status_ok == 200, f"Expected 200 for good {param}={good_val!r}, got {status_ok}: {data_ok.decode('utf-8')}"

    # OpenAI Completions 테스트
    openai_compl_cases = [
        ("best_of", 2, 1),
        ("echo", True, False),
        ("suffix", "world", ""),
    ]
    for param, bad_val, good_val in openai_compl_cases:
        bad_body = {"model": "raon-1-mini", "prompt": "hi", param: bad_val}
        status, data, _ = _request((host, port), "POST", "/v1/completions", bad_body)
        assert status == 400, f"Expected 400 for bad completions {param}={bad_val!r}, got {status}"
        res = json.loads(data.decode("utf-8"))
        assert res["error"]["param"] == param
        assert res["error"]["code"] == "unsupported_parameter"

        good_body = {"model": "raon-1-mini", "prompt": "hi", param: good_val}
        status_ok, _, _ = _request((host, port), "POST", "/v1/completions", good_body)
        assert status_ok == 200, f"Expected 200 for good completions {param}={good_val!r}, got {status_ok}"

    # Ollama Chat format 테스트
    status, data, _ = _request((host, port), "POST", "/api/chat", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "format": "json"})
    assert status == 400
    res = json.loads(data.decode("utf-8"))
    assert "format" in res["error"]

    status_ok, _, _ = _request((host, port), "POST", "/api/chat", {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "format": ""})
    assert status_ok == 200

    # Ollama Generate 테스트 (system, suffix, images, template, context)
    ollama_gen_cases = [
        ("system", "You are an assistant", ""),
        ("suffix", "end", ""),
        ("images", ["abc"], []),
        ("template", "{{ .Prompt }}", ""),
        ("context", [1, 2, 3], []),
    ]
    for param, bad_val, good_val in ollama_gen_cases:
        bad_body = {"model": "raon-1-mini", "prompt": "hi", param: bad_val}
        status, data, _ = _request((host, port), "POST", "/api/generate", bad_body)
        assert status == 400, f"Expected 400 for bad generate {param}={bad_val!r}, got {status}"

        good_body = {"model": "raon-1-mini", "prompt": "hi", param: good_val}
        status_ok, _, _ = _request((host, port), "POST", "/api/generate", good_body)
        assert status_ok == 200, f"Expected 200 for good generate {param}={good_val!r}, got {status_ok}"

    # Ollama options 테스트
    ollama_opt_cases = [
        ("repeat_penalty", 1.5, 1.0),
        ("typical_p", 0.8, 1.0),
        ("tfs_z", 0.9, 1.0),
        ("min_p", 0.05, 0.0),
        ("mirostat", 1, 0),
    ]
    for opt, bad_val, good_val in ollama_opt_cases:
        bad_body = {"model": "raon-1-mini", "prompt": "hi", "options": {opt: bad_val}}
        status, data, _ = _request((host, port), "POST", "/api/generate", bad_body)
        assert status == 400, f"Expected 400 for bad option {opt}={bad_val!r}, got {status}"

        good_body = {"model": "raon-1-mini", "prompt": "hi", "options": {opt: good_val}}
        status_ok, _, _ = _request((host, port), "POST", "/api/generate", good_body)
        assert status_ok == 200, f"Expected 200 for good option {opt}={good_val!r}, got {status_ok}"

    # stream_options.include_obfuscation 테스트: true/false 모두 거부되지 않고 200
    so_true = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream_options": {"include_obfuscation": True}}
    status_t, _, _ = _request((host, port), "POST", "/v1/chat/completions", so_true)
    assert status_t == 200

    so_false = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "stream_options": {"include_obfuscation": False}}
    status_f, _, _ = _request((host, port), "POST", "/v1/chat/completions", so_false)
    assert status_f == 200

    # verbosity (chat): medium(또는 null)은 200, low/high 등은 400 unsupported_parameter
    v_med = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "verbosity": "medium"}
    status_vm, _, _ = _request((host, port), "POST", "/v1/chat/completions", v_med)
    assert status_vm == 200

    v_bad = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "verbosity": "low"}
    status_vb, data_vb, _ = _request((host, port), "POST", "/v1/chat/completions", v_bad)
    assert status_vb == 400
    res_vb = json.loads(data_vb.decode("utf-8"))
    assert res_vb["error"]["code"] == "unsupported_parameter"

    # completions에서 enable_thinking 및 chat_template_kwargs 미지원
    c_think = {"model": "raon-1-mini", "prompt": "hi", "enable_thinking": True}
    status_ct, data_ct, _ = _request((host, port), "POST", "/v1/completions", c_think)
    assert status_ct == 400
    res_ct = json.loads(data_ct.decode("utf-8"))
    assert res_ct["error"]["code"] == "unsupported_parameter"

    c_think_none = {"model": "raon-1-mini", "prompt": "hi", "enable_thinking": None}
    status_ctn, _, _ = _request((host, port), "POST", "/v1/completions", c_think_none)
    assert status_ctn == 200

    # reasoning_effort (chat): none/medium은 200, low/high/minimal은 400 unsupported_parameter, 그 외는 400 invalid_value
    re_none = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}
    status_rn, _, _ = _request((host, port), "POST", "/v1/chat/completions", re_none)
    assert status_rn == 200

    re_med = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "medium"}
    status_rm, _, _ = _request((host, port), "POST", "/v1/chat/completions", re_med)
    assert status_rm == 200

    re_high = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"}
    status_rh, data_rh, _ = _request((host, port), "POST", "/v1/chat/completions", re_high)
    assert status_rh == 400
    res_rh = json.loads(data_rh.decode("utf-8"))
    assert res_rh["error"]["code"] == "unsupported_parameter"

    re_invalid = {"model": "raon-1-mini", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "extreme"}
    status_ri, data_ri, _ = _request((host, port), "POST", "/v1/chat/completions", re_invalid)
    assert status_ri == 400
    res_ri = json.loads(data_ri.decode("utf-8"))
    assert res_ri["error"]["code"] == "invalid_value"

    # 공식 목록 밖 확장 칸은 조용히 무시되어 통과
    ext_body = {
        "model": "raon-1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "custom_gateway_trace_id": "xyz-123",
        "bifrost_routing_cluster": "cluster-a",
    }
    status_ext, _, _ = _request((host, port), "POST", "/v1/chat/completions", ext_body)
    assert status_ext == 200


def test_openai_completions_prompt_list_handling(server_addr):
    """prompt 목록: []·['a','b'] -> 400. ['a'] 는 'a' 와 같은 결과다(가짜 엔진에 들어간 입력을 비교)."""
    host, port, _ = server_addr

    # 1. [] -> 400
    status, data, _ = _request((host, port), "POST", "/v1/completions", {"model": "raon-1-mini", "prompt": []})
    assert status == 400
    res = json.loads(data.decode("utf-8"))
    assert res["error"]["param"] == "prompt"
    assert "batch prompts are not supported" in res["error"]["message"]

    # 2. ["a", "b"] -> 400
    status, data, _ = _request((host, port), "POST", "/v1/completions", {"model": "raon-1-mini", "prompt": ["a", "b"]})
    assert status == 400
    res = json.loads(data.decode("utf-8"))
    assert res["error"]["param"] == "prompt"
    assert "batch prompts are not supported" in res["error"]["message"]

    # 3. ["single prompt test"] -> 200, 엔진에 들어간 입력이 'single prompt test'
    status, data, _ = _request((host, port), "POST", "/v1/completions", {"model": "raon-1-mini", "prompt": ["single prompt test"]})
    assert status == 200
    active_engine = Handler.engine
    assert active_engine.last_prompt == "single prompt test"

    # 4. "single prompt test" (문자열) -> 200, 엔진에 들어간 입력이 동일
    status, data, _ = _request((host, port), "POST", "/v1/completions", {"model": "raon-1-mini", "prompt": "single prompt test"})
    assert status == 200
    assert Handler.engine.last_prompt == "single prompt test"


def test_ollama_chat_messages_images_rejected(server_addr):
    """Ollama 메시지의 images가 스키마 표에서 거부(400)되는지 확인."""
    host, port, _ = server_addr
    body = {
        "model": "raon-1-mini",
        "messages": [
            {"role": "user", "content": "hi", "images": ["data:image/png;base64,..."]}
        ]
    }
    status, data, _ = _request((host, port), "POST", "/api/chat", body)
    assert status == 400
    res = json.loads(data.decode("utf-8"))
    assert "error" in res
    assert "images" in res["error"]


@pytest.mark.parametrize(
    "exc",
    [
        _connection_aborted_10053(),
        ConnectionResetError(104, "reset"),
        BrokenPipeError(32, "broken pipe"),
    ],
)
@pytest.mark.parametrize("response_started", [False, True])
def test_safe_dispatch_connection_errors_are_single_line_without_500(
    exc, response_started, capsys
):
    handler = Handler.__new__(Handler)
    handler.command = "POST"
    handler.path = "/api/chat?stream=true"
    handler.close_connection = False
    sent = []
    handler._send_error = lambda *args, **kwargs: sent.append((args, kwargs))

    def route():
        handler._response_started = response_started
        raise exc

    handler._safe_dispatch(route)
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "Traceback" not in lines[0]
    assert "method=POST" in lines[0]
    assert "path=/api/chat?stream=true" in lines[0]
    assert f"exception={type(exc).__name__}" in lines[0]
    assert "errno=" in lines[0]
    if isinstance(exc, ConnectionAbortedError):
        assert "errno=10053" in lines[0]
    assert sent == []
    assert handler.close_connection is True


def test_safe_dispatch_connection_error_while_writing_500_is_single_line(capsys):
    handler = Handler.__new__(Handler)
    handler.command = "POST"
    handler.path = "/api/chat"
    handler.close_connection = False

    def broken_500(*args, **kwargs):
        raise BrokenPipeError(32, "broken pipe")

    handler._send_error = broken_500
    handler._safe_dispatch(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    stderr = capsys.readouterr().err
    assert stderr.count("Traceback") == 1
    disconnect_lines = [
        line for line in stderr.splitlines() if line.startswith("client disconnected:")
    ]
    assert len(disconnect_lines) == 1
    assert "exception=BrokenPipeError" in disconnect_lines[0]
