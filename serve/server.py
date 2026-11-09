# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon NPU 서비스 — Ollama·OpenAI 호환 HTTP 서버.

`serve.engine`은 별도 워커가 제공한다고 가정한다.  이 모듈은
`Handler`가 주입받은 엔진 인스턴스만 사용하며, 실제 `Engine` import는
`main()`에서 한 번 수행한다.  테스트용 가짜 엔진을 주입하기 쉽도록
엔진 클래스 자체는 모듈 최상단에서 import하지 않는다.

수치 옵션 중 `seed`는 0 이상 정수일 때 고정 시드로 사용하며, 음수(`seed < 0`)는
llama.cpp·Ollama 관례에 따라 고정하지 않고 무작위(random seed)로 처리한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

from host.strict_json import loads_strict

DEFAULT_PORT = 11434
DEFAULT_MAX_WAIT = 600.0
DEFAULT_KEEP_ALIVE = 300.0
MAX_KEEP_ALIVE_SECONDS = 315360000.0  # 10 years (10 * 365 * 86400 seconds)
MAX_WAIT_SECONDS = 3600.0
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
REQUEST_SOCKET_TIMEOUT = 60.0

MAX_POSITION_EMBEDDINGS = 262144
MAX_VOCAB_SIZE = 259840
MIN_SEED = -(2**63)
MAX_SEED = 2**63 - 1

# Boundary semantics: top_p in [0, 1] where 0.0 forces greedy argmax (retains top-1 token)
# and 1.0 disables nucleus filtering. temperature in [0, 2] where 0.0 forces greedy argmax.
NUMERIC_OPTION_SPECS: dict[str, tuple[type, int | float, int | float]] = {
    "max_tokens": (int, 0, MAX_POSITION_EMBEDDINGS),
    "max_completion_tokens": (int, 0, MAX_POSITION_EMBEDDINGS),
    "num_predict": (int, 0, MAX_POSITION_EMBEDDINGS),
    "temperature": (float, 0, 2),
    "top_p": (float, 0, 1),
    "top_k": (int, 0, MAX_VOCAB_SIZE),
    "seed": (int, MIN_SEED, MAX_SEED),
    "presence_penalty": (float, -2, 2),
}

DEFAULT_PARAMETER_SIZE = "28.8B"
DEFAULT_QUANTIZATION_LEVEL = "D: q4nx + 9-bit bfp16"
DEFAULT_MODEL_FORMAT = "raon-npu"


def _fallback_manifest_paths() -> tuple[Path, ...]:
    """Extra manifest.json candidates taken from $RAON_WEIGHTS_D (D-mode weights dir)."""
    env_dir = os.environ.get("RAON_WEIGHTS_D")
    if not env_dir:
        return ()
    p = Path(env_dir)
    return (p if p.name == "manifest.json" else p / "manifest.json",)


OLLAMA_ROOT_TEXT = "Ollama is running"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_utc(ts_dt: dt.datetime) -> str:
    return ts_dt.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")


# Accepted chat fields by API and object level.  Keep these as the single source
# of truth for both request validation and the public accepted-field summaries.
OPENAI_MESSAGE_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "system": frozenset({"role", "content"}),
    "user": frozenset({"role", "content"}),
    "assistant": frozenset(
        {"role", "content", "reasoning_content", "tool_calls", "loss"}
    ),
    "tool": frozenset({"role", "content", "tool_call_id"}),
}

OLLAMA_MESSAGE_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "system": frozenset({"role", "content"}),
    "user": frozenset({"role", "content"}),
    "assistant": frozenset({"role", "content", "thinking", "tool_calls"}),
    "tool": frozenset({"role", "content", "tool_name"}),
}

OPENAI_CONTENT_PART_ALLOWED_FIELDS = frozenset({"type", "text"})
OLLAMA_CONTENT_PART_ALLOWED_FIELDS = frozenset({"type", "text"})
OPENAI_TOOL_CALL_ALLOWED_FIELDS = frozenset({"index", "id", "type", "function"})
OLLAMA_TOOL_CALL_ALLOWED_FIELDS = frozenset({"function"})
OPENAI_TOOL_FUNCTION_ALLOWED_FIELDS = frozenset({"name", "arguments"})
OLLAMA_TOOL_FUNCTION_ALLOWED_FIELDS = frozenset({"name", "arguments"})

# Values at these schema paths are free-form mappings.  Their keys are user
# arguments rather than protocol fields and must not be rejected.
CHAT_FREE_FORM_VALUE_PATHS: dict[str, frozenset[str]] = {
    "openai": frozenset(),
    "ollama": frozenset({"tool_call.function.arguments"}),
}


def _accepted_chat_fields(
    message_fields: dict[str, frozenset[str]],
    content_fields: frozenset[str],
    tool_call_fields: frozenset[str],
    function_fields: frozenset[str],
) -> frozenset[str]:
    """Build accepted-field names while preserving the existing path convention."""
    fields = set().union(*message_fields.values())
    fields.update(f"content.{field}" for field in content_fields)
    fields.update(
        f"tool_call.{field}" for field in tool_call_fields if field != "function"
    )
    fields.update(f"tool_call.function.{field}" for field in function_fields)
    return frozenset(fields)


OPENAI_ACCEPTED_FIELDS = _accepted_chat_fields(
    OPENAI_MESSAGE_ALLOWED_FIELDS,
    OPENAI_CONTENT_PART_ALLOWED_FIELDS,
    OPENAI_TOOL_CALL_ALLOWED_FIELDS,
    OPENAI_TOOL_FUNCTION_ALLOWED_FIELDS,
)
OLLAMA_ACCEPTED_FIELDS = _accepted_chat_fields(
    OLLAMA_MESSAGE_ALLOWED_FIELDS,
    OLLAMA_CONTENT_PART_ALLOWED_FIELDS,
    OLLAMA_TOOL_CALL_ALLOWED_FIELDS,
    OLLAMA_TOOL_FUNCTION_ALLOWED_FIELDS,
)

# --------------------------------------------------------------------------- #
# Endpoint Parameter Registry (Classification: SUPPORTED, IGNORED, UNSUPPORTED)
# --------------------------------------------------------------------------- #

OPENAI_CHAT_SUPPORTED = frozenset({
    "messages",
    "model",
    "stream",
    "stream_options",
    "stop",
    "tools",
    "keep_alive",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "presence_penalty",
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_effort",
})

OPENAI_CHAT_IGNORED = frozenset({
    "user",
    "metadata",
    "store",
    "service_tier",
    "prompt_cache_key",
    "safety_identifier",
})

OPENAI_CHAT_UNSUPPORTED = frozenset({
    "n",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
    "functions",
    "function_call",
    "tool_choice",
    "parallel_tool_calls",
    "verbosity",
})

OPENAI_COMPLETIONS_SUPPORTED = frozenset({
    "model",
    "prompt",
    "stream",
    "stream_options",
    "stop",
    "keep_alive",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "presence_penalty",
})

OPENAI_COMPLETIONS_IGNORED = frozenset({
    "user",
})

OPENAI_COMPLETIONS_UNSUPPORTED = frozenset({
    "best_of",
    "echo",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "suffix",
    "enable_thinking",
    "chat_template_kwargs",
})

STREAM_OPTIONS_SUPPORTED = frozenset({
    "include_usage",
})

STREAM_OPTIONS_IGNORED = frozenset({
    "include_obfuscation",
})

STREAM_OPTIONS_UNSUPPORTED = frozenset()

OLLAMA_CHAT_SUPPORTED = frozenset({
    "model",
    "messages",
    "tools",
    "think",
    "options",
    "stream",
    "keep_alive",
})

OLLAMA_CHAT_IGNORED = frozenset()

OLLAMA_CHAT_UNSUPPORTED = frozenset({
    "format",
})

OLLAMA_GENERATE_SUPPORTED = frozenset({
    "model",
    "prompt",
    "options",
    "stream",
    "raw",
    "keep_alive",
})

OLLAMA_GENERATE_IGNORED = frozenset()

OLLAMA_GENERATE_UNSUPPORTED = frozenset({
    "suffix",
    "images",
    "format",
    "system",
    "template",
    "context",
    "think",
})

OLLAMA_OPTIONS_SUPPORTED = frozenset({
    "num_predict",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "presence_penalty",
    "stop",
})

OLLAMA_OPTIONS_IGNORED = frozenset({
    "num_keep",
    "numa",
    "num_ctx",
    "num_batch",
    "num_gpu",
    "main_gpu",
    "use_mmap",
    "num_thread",
    "low_vram",
    "f16_kv",
    "vocab_only",
    "use_mlock",
    "repeat_last_n",
    "penalize_newline",
    "mirostat_tau",
    "mirostat_eta",
})

OLLAMA_OPTIONS_UNSUPPORTED = frozenset({
    "repeat_penalty",
    "typical_p",
    "tfs_z",
    "min_p",
    "frequency_penalty",
    "mirostat",
})


def is_meaningless_value(param: str, val: Any) -> bool:
    """Return True if val is a known no-op value for an unsupported parameter."""
    if val is None:
        return True
    if param in ("n", "best_of"):
        return val == 1 and not isinstance(val, bool)
    if param == "echo":
        return val is False
    if param == "logprobs":
        return (val is False) or (val == 0 and not isinstance(val, bool))
    if param == "top_logprobs":
        return val == 0 and not isinstance(val, bool)
    if param == "logit_bias":
        return val == {}
    if param == "frequency_penalty":
        return (val == 0 or val == 0.0) and not isinstance(val, bool)
    if param == "response_format":
        return val == {"type": "text"}
    if param == "modalities":
        return val == ["text"]
    if param in ("audio", "prediction", "web_search_options", "functions", "function_call", "enable_thinking", "chat_template_kwargs"):
        return False
    if param == "verbosity":
        return val == "medium"
    if param in ("suffix", "format", "template", "system"):
        return val == ""
    if param in ("images", "context"):
        return val == []
    if param == "tool_choice":
        return val == "auto"
    if param == "parallel_tool_calls":
        return val is True
    if param in ("repeat_penalty", "typical_p", "tfs_z"):
        return (val == 1 or val == 1.0) and not isinstance(val, bool)
    if param in ("min_p", "mirostat"):
        return (val == 0 or val == 0.0) and not isinstance(val, bool)
    return False


def parse_keep_alive(val: Any, default: float = DEFAULT_KEEP_ALIVE) -> float:
    """Ollama/CLI keep_alive 값을 초(float) 단위로 변환 및 유효성 검증.

    - None -> default
    - bool -> default if val else 0.0
    - int -> abs(val) <= MAX_KEEP_ALIVE_SECONDS 검사 후 float 변환
    - float -> 유한한 숫자 (abs <= MAX_KEEP_ALIVE_SECONDS)
    - str -> '300', '300s', '5m', '1h', '500ms', '0', '-1' 등 지원 (abs <= MAX_KEEP_ALIVE_SECONDS)
    그 밖(비유한 수, 해석 못 하는 문자열, 배열/객체, 상한 초과)은 ValueError를 발생시킨다.
    어떤 입력이든 parse_keep_alive는 유효한 float를 반환하거나 ValueError만 발생시킨다.
    """
    try:
        if val is None:
            return float(default)
        if isinstance(val, bool):
            return float(default) if val else 0.0
        if isinstance(val, int):
            if abs(val) > MAX_KEEP_ALIVE_SECONDS:
                raise ValueError(
                    f"keep_alive exceeds maximum allowed duration ({MAX_KEEP_ALIVE_SECONDS}s)"
                )
            return float(val)
        if isinstance(val, float):
            if not math.isfinite(val):
                raise ValueError("keep_alive must be finite")
            if abs(val) > MAX_KEEP_ALIVE_SECONDS:
                raise ValueError(
                    f"keep_alive exceeds maximum allowed duration ({MAX_KEEP_ALIVE_SECONDS}s)"
                )
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if not s:
                raise ValueError("keep_alive cannot be empty")
            if s in ("nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                raise ValueError("keep_alive must be finite")
            factor = 1.0
            num_str = s
            if s.endswith("ms"):
                num_str = s[:-2].strip()
                factor = 0.001
            elif s.endswith("s"):
                num_str = s[:-1].strip()
                factor = 1.0
            elif s.endswith("m"):
                num_str = s[:-1].strip()
                factor = 60.0
            elif s.endswith("h"):
                num_str = s[:-1].strip()
                factor = 3600.0
            try:
                f = float(num_str)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"unparseable keep_alive string {val!r}") from exc
            if not math.isfinite(f):
                raise ValueError("keep_alive must be finite")
            res = f * factor
            if not math.isfinite(res):
                raise ValueError("keep_alive must be finite")
            if abs(res) > MAX_KEEP_ALIVE_SECONDS:
                raise ValueError(
                    f"keep_alive exceeds maximum allowed duration ({MAX_KEEP_ALIVE_SECONDS}s)"
                )
            return res
        raise ValueError(f"invalid keep_alive type: {type(val).__name__}")
    except OverflowError as exc:
        raise ValueError(f"keep_alive numeric overflow: {exc}") from exc


def _strip_leading_blank_lines(text: str, *, final: bool = False) -> tuple[str, bool]:
    """`</think>` 뒤의 앞쪽 공백 줄(`[ \\t]*\\r?\\n`)을 제거한다."""
    while True:
        m = re.match(r"^[ \t]*\r?\n", text)
        if m:
            text = text[m.end() :]
        else:
            break
    if not text:
        return "", not final
    if not final and re.fullmatch(r"[ \t\r]+", text):
        return text, True
    if final and re.fullmatch(r"[ \t\r]+", text):
        return "", False
    return text, False


def _split_accumulated_think(
    thinking_parts: list[str],
    text_parts: list[str],
    *,
    thinking_enabled: bool,
    finish_reason: str | None,
) -> tuple[str, str]:
    """Join the engine's already-separated thinking and answer fields."""
    return "".join(thinking_parts), "".join(text_parts)


class _LogWriter:
    """스레드 안전한 단순 파일 로거."""

    def __init__(self, path: str | None) -> None:
        self._lock = threading.Lock()
        self._path = path

    def write(self, line: str) -> None:
        if not self._path:
            return
        with self._lock:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


class Handler(BaseHTTPRequestHandler):
    """Ollama/OpenAI 호환 요청 핸들러.

    아래 class attributes는 `main()` 또는 테스트 셋업에서 채운다:
      engine         : Engine 인스턴스 (내려놓아진 상태면 None)
      engine_factory : () -> Engine 재생성 콜백 (없으면 기존 엔진 생성 인자로 자동 복구)
      GenOptions     : GenOptions 클래스
      Chunk          : Chunk 클래스
      max_wait       : 엔진 사용 대기 상한(초)
      keep_alive     : 기본 keep-alive 시간(초; 기본 300, 0=즉시 해제, <0=영구 유지)
      logger         : _LogWriter 인스턴스
      model_name     : 노출할 모델 이름
      weights_dir    : 가중치 디렉터리 경로
      model_dir      : 토크나이저/모델 디렉터리 경로
    """

    engine: Any | None = None
    engine_factory: Callable[[], Any] | None = None
    GenOptions: Any | None = None
    Chunk: Any | None = None
    max_wait: float = DEFAULT_MAX_WAIT
    keep_alive: float = DEFAULT_KEEP_ALIVE
    logger: _LogWriter | None = None
    timeout = REQUEST_SOCKET_TIMEOUT

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.timeout)
    model_name: str = "raon-1-mini"
    weights_dir: str = ""
    model_dir: str = ""

    # 모델 메타데이터 캐시
    _cached_manifest_info: dict[str, Any] | None = None
    _cached_manifest_key: str | None = None

    # 엔진은 동시에 하나만 사용. busy_lock은 엔진 점유/대기 및 적재/해제 조정용.
    _busy_lock = threading.Lock()
    _busy_event = threading.Event()  # True면 누군가 생성/요청 처리 중
    _busy_holder: str | None = None  # 디버깅용 현재 요청 식별자

    # Keep-alive 만료 추적
    _expires_at: float | None = None  # time.monotonic() 기준 만료 시각
    _expires_at_iso: str | None = None  # /api/ps 노출용 UTC ISO-8601 문자열
    _last_engine_Ref_id: int | None = None
    load_count: int = 0
    unload_count: int = 0

    server_version = "RaonNPU/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler 기본 stderr 로깅 대신 조용히 유지.
        pass

    def _log(self, method: str, path: str, prompt_tokens: int | None,
             completion_tokens: int | None, prefill_s: float | None,
             decode_s: float | None, extra: str | None = None) -> None:
        if not self.logger:
            return
        parts = [f"{_now()} {method} {path}"]
        if prompt_tokens is not None:
            parts.append(f"prompt={prompt_tokens}")
        if completion_tokens is not None:
            parts.append(f"completion={completion_tokens}")
        if prefill_s is not None:
            parts.append(f"prefill={prefill_s:.3f}s")
        if decode_s is not None:
            parts.append(f"decode={decode_s:.3f}s")
        if extra:
            parts.append(extra)
        self.logger.write(" ".join(parts))

    @classmethod
    def _is_loaded(cls) -> bool:
        eng = cls.engine
        if eng is None:
            return False
        return bool(getattr(eng, "loaded", True))

    @classmethod
    def _sync_engine_tracking_locked(cls) -> None:
        """외부(테스트 픽스처 등)에서 Handler.engine을 직접 주입한 경우 타이머/팩토리 동기화."""
        eng = cls.engine
        if eng is not None and id(eng) != cls._last_engine_Ref_id:
            cls._last_engine_Ref_id = id(eng)
            eng_cls = type(eng)
            w_dir = getattr(eng, "weights_dir", cls.weights_dir) or cls.weights_dir
            m_dir = getattr(eng, "model_dir", cls.model_dir) or cls.model_dir
            preload = getattr(eng, "preload", getattr(eng, "_preload", True))
            if cls.engine_factory is None:
                cls.engine_factory = lambda: eng_cls(w_dir, m_dir, preload=preload)
            if cls._expires_at is None and cls.keep_alive > 0:
                cls._set_expiry_locked(cls.keep_alive)
            elif cls.keep_alive < 0:
                cls._set_expiry_locked(cls.keep_alive)

    @classmethod
    def _set_expiry_locked(cls, keep_alive_s: float) -> None:
        if keep_alive_s < 0:
            cls._expires_at = None
            cls._expires_at_iso = _format_utc(
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=36500)
            )
        elif keep_alive_s == 0:
            cls._expires_at = time.monotonic()
            cls._expires_at_iso = _now()
        else:
            cls._expires_at = time.monotonic() + keep_alive_s
            cls._expires_at_iso = _format_utc(
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=keep_alive_s)
            )

    @classmethod
    def _unload_locked(cls) -> None:
        """잠금(`_busy_lock`) 안에서 엔진의 NPU 텐서와 가중치를 내려놓고 메모리를 회수한다."""
        eng = cls.engine
        if eng is not None:
            if cls.engine_factory is None:
                eng_cls = type(eng)
                w_dir = getattr(eng, "weights_dir", cls.weights_dir) or cls.weights_dir
                m_dir = getattr(eng, "model_dir", cls.model_dir) or cls.model_dir
                preload = getattr(eng, "preload", getattr(eng, "_preload", True))
                cls.engine_factory = lambda: eng_cls(w_dir, m_dir, preload=preload)
            if hasattr(eng, "unload") and callable(eng.unload):
                try:
                    eng.unload()
                except Exception:
                    pass
            cls.engine = None
            cls._last_engine_Ref_id = None
            cls.unload_count += 1
            if cls.logger:
                cls.logger.write(f"{_now()} engine unloaded")
        cls._expires_at = None
        cls._expires_at_iso = None
        gc.collect()

    @classmethod
    def _ensure_loaded_locked(cls) -> Any:
        """잠금(`_busy_lock`) 안에서 엔진이 내려가 있으면 다시 적재한다."""
        if cls._is_loaded():
            cls._sync_engine_tracking_locked()
            return cls.engine
        t0 = time.perf_counter()
        try:
            if cls.engine_factory is not None:
                cls.engine = cls.engine_factory()
            else:
                from serve.engine import Engine
                cls.engine = Engine(cls.weights_dir, cls.model_dir, preload=True)
        except Exception as exc:
            raise RuntimeError(f"model load failed: {exc}") from exc
        cls._last_engine_Ref_id = id(cls.engine)
        cls.load_count += 1
        dt_s = time.perf_counter() - t0
        if cls.logger:
            cls.logger.write(f"{_now()} engine loaded in {dt_s:.3f}s")
        return cls.engine

    @classmethod
    def check_keep_alive_expired(cls) -> bool:
        """유휴 상태에서 keep-alive 만료 시각이 지났으면 잠금 안에서 엔진을 내려놓는다."""
        if cls._busy_event.is_set():
            return False
        if not cls._busy_lock.acquire(blocking=False):
            return False
        try:
            cls._sync_engine_tracking_locked()
            if (
                not cls._busy_event.is_set()
                and cls.engine is not None
                and cls._expires_at is not None
                and time.monotonic() >= cls._expires_at
            ):
                cls._unload_locked()
                return True
            return False
        finally:
            cls._busy_lock.release()

    def _write_json(self, status: int, obj: Any) -> None:
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json_or_cancel(
        self, status: int, obj: Any, cancel: Callable[[], bool]
    ) -> bool:
        """Write one JSON response, treating only socket I/O errors as cancellation."""
        try:
            self._write_json(status, obj)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]
            return False

    def _send_generation_error(
        self, exc: Exception, cancel: Callable[[], bool]
    ) -> None:
        """Send an engine failure unless the client socket is already gone."""
        try:
            self._send_error(500, str(exc), code="generation_error")
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]

    def _start_stream(
        self, content_type: str, cancel: Callable[[], bool]
    ) -> bool:
        """Send streaming headers and classify failures at this socket boundary."""
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]
            return False

    def _write_text(self, status: int, text: str, *, head_only: bool = False) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_error(
        self, status: int, message: str, *, param: str | None = None, code: str | None = None
    ) -> None:
        if self.path.split("?", 1)[0].startswith("/v1/"):
            self._write_json(status, {
                "error": {
                    "message": message,
                    "type": (
                        "invalid_request_error" if status == 400
                        else "server_error" if status >= 500
                        else "request_error"
                    ),
                    "param": param,
                    "code": code,
                }
            })
        else:
            self._write_json(status, {"error": message})

    def _read_json(self) -> dict[str, Any] | None:
        length = self.headers.get("Content-Length")
        if length is None:
            self._send_error(400, "missing Content-Length", param="Content-Length", code="invalid_content_length")
            return None
        try:
            n = int(length)
        except ValueError:
            self._send_error(400, "bad Content-Length")
            return None
        if n < 0:
            self._send_error(400, "bad Content-Length")
            return None
        if n > MAX_REQUEST_BODY_BYTES:
            self._send_error(413, "request body too large", param="Content-Length", code="request_too_large")
            return None
        try:
            data = self.rfile.read(n)
        except Exception:
            return None
        try:
            body = loads_strict(data)
        except Exception as e:
            self._send_error(400, f"invalid JSON: {e}")
            return None
        if not isinstance(body, dict):
            self._send_error(400, "JSON body must be an object", param=None, code="invalid_json_type")
            return None
        return body

    def _check_model(self, name: Any) -> bool:
        if not isinstance(name, str):
            return False
        base = self.model_name
        allowed = {base, f"{base}:latest"}
        return name in allowed

    def _wait_engine(self, req_id: str, *, load: bool = True) -> bool:
        """엔진이 비어질 때까지 대기하고, 내려가 있으면 잠금 안에서 다시 적재한다."""
        deadline = time.monotonic() + self.max_wait
        acquired = False
        while not acquired:
            acquired = self._busy_lock.acquire(blocking=False)
            if acquired:
                self._busy_event.set()
                type(self)._busy_holder = req_id
                try:
                    self._sync_engine_tracking_locked()
                    if load and not self._is_loaded():
                        self._ensure_loaded_locked()
                except BaseException:
                    self._busy_event.clear()
                    type(self)._busy_holder = None
                    self._busy_lock.release()
                    raise
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return False

    def _release_engine(self, keep_alive_s: float | None = None) -> None:
        eff_ka = self.keep_alive if keep_alive_s is None else float(keep_alive_s)
        try:
            if eff_ka == 0.0:
                self._unload_locked()
            elif self._is_loaded():
                self._set_expiry_locked(eff_ka)
        finally:
            self._busy_event.clear()
            type(self)._busy_holder = None
            try:
                self._busy_lock.release()
            except RuntimeError:
                pass

    def _make_cancel(self) -> Callable[[], bool]:
        """쓰기 실패/연결 끊김을 감지하는 cancel 콜백."""
        cancelled = threading.Event()

        def cancel() -> bool:
            return cancelled.is_set()

        cancel._set = cancelled.set  # type: ignore[attr-defined]
        return cancel

    def _check_cancel(self, cancel: Callable[[], bool]) -> None:
        """Set cancellation only when a readable socket has reached EOF."""
        if cancel():
            return
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                cancel._set()  # type: ignore[attr-defined]
        except (BlockingIOError, InterruptedError, TimeoutError):
            return
        except (ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Manifest & Model Details
    # ------------------------------------------------------------------ #

    @classmethod
    def _get_manifest_info(cls) -> dict[str, Any]:
        w_dir = cls.weights_dir or getattr(cls.engine, "weights_dir", "") or ""
        m_dir = cls.model_dir or getattr(cls.engine, "model_dir", "") or ""
        cache_key = f"{w_dir}|{m_dir}"
        if cls._cached_manifest_info is not None and cls._cached_manifest_key == cache_key:
            return cls._cached_manifest_info

        candidates: list[Path] = []
        for d in (w_dir, m_dir):
            if d:
                p = Path(d)
                candidates.append(p if p.name == "manifest.json" else (p / "manifest.json"))
        candidates.extend(_fallback_manifest_paths())

        total_size_bytes = 0
        parameter_size = DEFAULT_PARAMETER_SIZE

        for cand in candidates:
            try:
                if cand.is_file():
                    data = loads_strict(cand.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        if "total_size_bytes" in data and data["total_size_bytes"] is not None:
                            total_size_bytes = int(data["total_size_bytes"])
                        elif isinstance(data.get("bundles"), list):
                            total_size_bytes = sum(
                                int(b.get("size_bytes", 0))
                                for b in data["bundles"]
                                if isinstance(b, dict)
                            )
                        break
            except Exception:
                continue

        info = {
            "total_size_bytes": total_size_bytes,
            "parameter_size": parameter_size,
            "quantization_level": DEFAULT_QUANTIZATION_LEVEL,
            "format": DEFAULT_MODEL_FORMAT,
        }
        cls._cached_manifest_info = info
        cls._cached_manifest_key = cache_key
        return info

    @classmethod
    def _model_details(cls) -> dict[str, Any]:
        info = cls._get_manifest_info()
        return {
            "parent_model": "",
            "format": info["format"],
            "family": "raon",
            "families": ["raon"],
            "parameter_size": info["parameter_size"],
            "quantization_level": info["quantization_level"],
        }

    @classmethod
    def _model_size_bytes(cls) -> int:
        eng = cls.engine
        if eng is not None and getattr(eng, "total_size_bytes", 0):
            return int(eng.total_size_bytes)
        return int(cls._get_manifest_info()["total_size_bytes"])

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def send_response(self, code: int, message: str | None = None) -> None:
        self._response_started = True
        super().send_response(code, message)

    def _safe_dispatch(self, route_fn: Callable[[], None]) -> None:
        self._response_started = False
        try:
            route_fn()
        except ConnectionError as exc:
            self._log_disconnect(exc)
            self.close_connection = True
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.close_connection = True
            if not getattr(self, "_response_started", False):
                try:
                    method = getattr(self, "command", "")
                    if method == "HEAD":
                        self.send_response(500)
                        self.send_header("Connection", "close")
                        self.end_headers()
                    else:
                        self._send_error(500, str(exc), code="internal_error")
                except ConnectionError as write_exc:
                    self._log_disconnect(write_exc)

    def _log_disconnect(self, exc: ConnectionError) -> None:
        method = getattr(self, "command", "") or "-"
        path = getattr(self, "path", "") or "-"
        error_number = getattr(exc, "winerror", None)
        if error_number is None:
            error_number = getattr(exc, "errno", None)
        print(
            f"client disconnected: method={method} path={path} "
            f"exception={type(exc).__name__} errno={error_number}",
            file=sys.stderr,
        )

    def do_HEAD(self) -> None:
        self._safe_dispatch(self._route_head)

    def do_GET(self) -> None:
        self._safe_dispatch(self._route_get)

    def do_POST(self) -> None:
        self._safe_dispatch(self._route_post)

    def _route_head(self) -> None:
        # Ollama 와 동일: HEAD / 는 GET / 과 같은 머리, 본문 없음.
        # 알 수 없는 경로는 기존과 같이 404 (본문 없이 머리만).
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._write_text(200, OLLAMA_ROOT_TEXT, head_only=True)
            return
        body = _json_bytes({"error": f"unknown endpoint {path}"})
        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def _route_get(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            # Ollama 호환 생존 확인 (Open WebUI·API gateway 등이 사용).
            self._write_text(200, OLLAMA_ROOT_TEXT)
            return
        if path == "/healthz":
            self.check_keep_alive_expired()
            self._write_json(200, {"status": "ok", "loaded": self._is_loaded()})
            return
        if path in ("/api/version", "/api/version/"):
            self._write_json(200, {"version": "0.5.13"})
            return
        if path in ("/api/tags", "/api/tags/"):
            self._handle_api_tags()
            return
        if path in ("/api/ps", "/api/ps/"):
            self._handle_api_ps()
            return
        if path in ("/v1/models", "/v1/models/"):
            self._handle_v1_models()
            return
        self._send_error(404, f"unknown endpoint {path}")

    def _route_post(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._handle_v1_chat_completions()
            return
        if path in ("/v1/completions", "/v1/completions/"):
            self._handle_v1_completions()
            return
        if path in ("/api/chat", "/api/chat/"):
            self._handle_api_chat()
            return
        if path in ("/api/generate", "/api/generate/"):
            self._handle_api_generate()
            return
        if path in ("/api/show", "/api/show/"):
            self._handle_api_show()
            return
        self._send_error(404, f"unknown endpoint {path}")

    # ------------------------------------------------------------------ #
    # OpenAI endpoints
    # ------------------------------------------------------------------ #

    def _handle_v1_models(self) -> None:
        ts = int(time.time())
        self._write_json(200, {
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": ts,
                "owned_by": "raon",
            }],
        })

    def _invalid(self, message: str, param: str) -> bool:
        self._send_error(400, message, param=param, code="invalid_value")
        return False

    def _validate_keep_alive(self, value: Any) -> bool:
        if value is None:
            return True
        try:
            parse_keep_alive(value, self.keep_alive)
            return True
        except (ValueError, OverflowError) as exc:
            return self._invalid(f"keep_alive: {exc}", "keep_alive")

    def _validate_numbers(self, values: dict[str, Any], prefix: str = "") -> bool:
        for name, (kind, lower, upper) in NUMERIC_OPTION_SPECS.items():
            if name not in values or values[name] is None:
                continue
            value = values[name]
            param = f"{prefix}{name}"
            if kind is int:
                valid_type = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid_type:
                return self._invalid(f"{param} has an invalid type", param)
            if isinstance(value, float) and not math.isfinite(value):
                return self._invalid(f"{param} must be finite", param)
            if lower is not None and value < lower:
                return self._invalid(f"{param} must be at least {lower}", param)
            if upper is not None and value > upper:
                return self._invalid(f"{param} must be at most {upper}", param)
        return True

    def _validate_stop(self, value: Any, param: str) -> bool:
        if value is None or isinstance(value, str):
            return True
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return True
        return self._invalid(f"{param} must be a string or a list of strings", param)

    def _reject_unsupported_fields(
        self, value: dict[str, Any], allowed_fields: frozenset[str], param: str
    ) -> bool:
        """Reject the first unknown object key using its full request path."""
        unsupported = sorted(set(value) - allowed_fields)
        if not unsupported:
            return True
        field = unsupported[0]
        full_param = f"{param}.{field}"
        return self._invalid(
            f"{full_param} is not supported by the chat template", full_param
        )

    def _validate_messages(
        self,
        value: Any,
        *,
        allow_empty: bool = False,
        ollama: bool = False,
    ) -> bool:
        if not isinstance(value, list):
            return self._invalid("messages must be an array", "messages")
        if not value and not allow_empty:
            return self._invalid("messages must not be empty", "messages")
        for index, message in enumerate(value):
            param = f"messages.{index}"
            if not isinstance(message, dict):
                return self._invalid(f"{param} must be an object", param)
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                return self._invalid(
                    f"{param}.role must be one of system, user, assistant, or tool",
                    f"{param}.role",
                )
            schema = OLLAMA_MESSAGE_ALLOWED_FIELDS if ollama else OPENAI_MESSAGE_ALLOWED_FIELDS
            allowed_fields = schema[role]
            if not self._reject_unsupported_fields(message, allowed_fields, param):
                return False
            content = message.get("content")
            content_may_be_empty = role == "assistant" and "tool_calls" in message
            if content is None and content_may_be_empty:
                pass
            elif not isinstance(content, (str, list)):
                return self._invalid(
                    f"{param}.content must be a string or content array", f"{param}.content"
                )
            if isinstance(content, list):
                for part_index, part in enumerate(content):
                    part_param = f"{param}.content.{part_index}"
                    if not isinstance(part, dict):
                        return self._invalid(f"{part_param} must be an object", part_param)
                    part_fields = (
                        OLLAMA_CONTENT_PART_ALLOWED_FIELDS
                        if ollama
                        else OPENAI_CONTENT_PART_ALLOWED_FIELDS
                    )
                    if not self._reject_unsupported_fields(part, part_fields, part_param):
                        return False
                    if part.get("type") != "text" or not isinstance(part.get("text"), str):
                        return self._invalid(
                            f"{part_param} must be a text content object with string text",
                            part_param,
                        )
            if "reasoning_content" in message and not isinstance(
                message["reasoning_content"], str
            ):
                return self._invalid(
                    f"{param}.reasoning_content must be a string",
                    f"{param}.reasoning_content",
                )
            if "loss" in message and not isinstance(message["loss"], bool):
                return self._invalid(f"{param}.loss must be a boolean", f"{param}.loss")
            metadata_field = "tool_name" if ollama else "tool_call_id"
            if metadata_field in message and not isinstance(message[metadata_field], str):
                return self._invalid(
                    f"{param}.{metadata_field} must be a string",
                    f"{param}.{metadata_field}",
                )
            if "thinking" in message and not isinstance(message["thinking"], str):
                return self._invalid(
                    f"{param}.thinking must be a string", f"{param}.thinking"
                )
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                if not isinstance(tool_calls, list):
                    return self._invalid(
                        f"{param}.tool_calls must be an array", f"{param}.tool_calls"
                    )
                for call_index, tool_call in enumerate(tool_calls):
                    call_param = f"{param}.tool_calls.{call_index}"
                    if not isinstance(tool_call, dict):
                        return self._invalid(f"{call_param} must be an object", call_param)
                    call_fields = (
                        OLLAMA_TOOL_CALL_ALLOWED_FIELDS
                        if ollama
                        else OPENAI_TOOL_CALL_ALLOWED_FIELDS
                    )
                    if not self._reject_unsupported_fields(
                        tool_call, call_fields, call_param
                    ):
                        return False
                    if not ollama:
                        index_value = tool_call.get("index")
                        if "index" in tool_call and (
                            not isinstance(index_value, int)
                            or isinstance(index_value, bool)
                            or index_value < 0
                        ):
                            return self._invalid(
                                f"{call_param}.index must be a non-negative integer",
                                f"{call_param}.index",
                            )
                        if not isinstance(tool_call.get("id"), str):
                            return self._invalid(
                                f"{call_param}.id must be a string", f"{call_param}.id"
                            )
                        if tool_call.get("type") != "function":
                            return self._invalid(
                                f"{call_param}.type must be function", f"{call_param}.type"
                            )
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        return self._invalid(
                            f"{call_param}.function must be an object",
                            f"{call_param}.function",
                        )
                    function_fields = (
                        OLLAMA_TOOL_FUNCTION_ALLOWED_FIELDS
                        if ollama
                        else OPENAI_TOOL_FUNCTION_ALLOWED_FIELDS
                    )
                    if not self._reject_unsupported_fields(
                        function, function_fields, f"{call_param}.function"
                    ):
                        return False
                    if not isinstance(function.get("name"), str) or not function["name"].strip():
                        return self._invalid(
                            f"{call_param}.function.name must be a non-empty string",
                            f"{call_param}.function.name",
                        )
                    arguments = function.get("arguments")
                    expected_type = dict if ollama else str
                    if not isinstance(arguments, expected_type):
                        expected_name = "an object" if ollama else "a string"
                        return self._invalid(
                            f"{call_param}.function.arguments must be {expected_name}",
                            f"{call_param}.function.arguments",
                        )
                    if not ollama:
                        try:
                            parsed_arguments = loads_strict(arguments)
                        except (TypeError, ValueError, RecursionError):
                            parsed_arguments = None
                        if not isinstance(parsed_arguments, dict):
                            return self._invalid(
                                f"{call_param}.function.arguments must encode an object",
                                f"{call_param}.function.arguments",
                            )
        return True

    def _validate_tools(self, value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, list):
            return self._invalid("tools must be an array", "tools")
        for index, tool in enumerate(value):
            param = f"tools.{index}"
            if not isinstance(tool, dict):
                return self._invalid(f"{param} must be an object", param)
            if tool.get("type") != "function":
                return self._invalid(f"{param}.type must be function", f"{param}.type")
            function = tool.get("function")
            if not isinstance(function, dict):
                return self._invalid(
                    f"{param}.function must be an object", f"{param}.function"
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                return self._invalid(
                    f"{param}.function.name must be a non-empty string",
                    f"{param}.function.name",
                )
            if "description" in function and not isinstance(function["description"], str):
                return self._invalid(
                    f"{param}.function.description must be a string",
                    f"{param}.function.description",
                )
            if "parameters" in function and not isinstance(function["parameters"], dict):
                return self._invalid(
                    f"{param}.function.parameters must be an object",
                    f"{param}.function.parameters",
                )
        return True

    @staticmethod
    def _normalize_openai_messages(messages: list[dict]) -> list[dict]:
        """Convert validated OpenAI JSON argument strings for the model template."""
        normalized: list[dict] = []
        for message in messages:
            copied = dict(message)
            if "tool_calls" in copied:
                calls: list[dict] = []
                for tool_call in copied["tool_calls"]:
                    call = dict(tool_call)
                    function = dict(call["function"])
                    function["arguments"] = loads_strict(function["arguments"])
                    call["function"] = function
                    calls.append(call)
                copied["tool_calls"] = calls
            normalized.append(copied)
        return normalized

    @staticmethod
    def _normalize_ollama_messages(messages: list[dict]) -> list[dict]:
        """Convert validated Ollama messages for the model template."""
        normalized: list[dict] = []
        for message in messages:
            copied = dict(message)
            if copied.get("role") == "assistant" and "thinking" in copied:
                copied["reasoning_content"] = copied.pop("thinking")
            normalized.append(copied)
        return normalized

    def _validate_bool(self, d: dict[str, Any], key: str, prefix: str = "") -> bool:
        val = d.get(key)
        if val is not None and not isinstance(val, bool):
            return self._invalid(f"{prefix}{key} must be a boolean", f"{prefix}{key}")
        return True

    def _validate_openai_body(self, body: dict[str, Any], *, chat: bool) -> bool:
        unsupported = OPENAI_CHAT_UNSUPPORTED if chat else OPENAI_COMPLETIONS_UNSUPPORTED
        for k in unsupported:
            if k in body:
                v = body[k]
                if not is_meaningless_value(k, v):
                    self._send_error(
                        400,
                        f"unsupported parameter: {k}",
                        param=k,
                        code="unsupported_parameter",
                    )
                    return False

        if not self._validate_keep_alive(body.get("keep_alive")):
            return False
        if chat and not self._validate_messages(body.get("messages")):
            return False
        if chat and not self._validate_tools(body.get("tools")):
            return False
        if chat and "reasoning_effort" in body:
            effort = body["reasoning_effort"]
            if effort is not None:
                if not isinstance(effort, str):
                    return self._invalid("reasoning_effort must be a string", "reasoning_effort")
                effort_clean = effort.strip().lower()
                if effort_clean in ("minimal", "low", "high"):
                    self._send_error(
                        400,
                        f"unsupported reasoning_effort: {effort}",
                        param="reasoning_effort",
                        code="unsupported_parameter",
                    )
                    return False
                elif effort_clean not in ("none", "medium"):
                    return self._invalid(
                        f"invalid reasoning_effort: {effort}",
                        "reasoning_effort",
                    )
        if not chat:
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                if not all(isinstance(item, str) for item in prompt):
                    return self._invalid("prompt must be a string or a list of strings", "prompt")
                if len(prompt) != 1:
                    self._send_error(
                        400,
                        "batch prompts are not supported",
                        param="prompt",
                        code="unsupported_parameter",
                    )
                    return False
            elif not isinstance(prompt, str):
                return self._invalid("prompt must be a string or a list of strings", "prompt")
        if not self._validate_stop(body.get("stop"), "stop"):
            return False
        if not self._validate_numbers(body):
            return False
        if not self._validate_bool(body, "stream"):
            return False
        if chat:
            if not self._validate_bool(body, "enable_thinking"):
                return False
            template_kwargs = body.get("chat_template_kwargs")
            if template_kwargs is not None:
                if not isinstance(template_kwargs, dict):
                    return self._invalid("chat_template_kwargs must be an object", "chat_template_kwargs")
                enabled = template_kwargs.get("enable_thinking")
                if enabled is not None and not isinstance(enabled, bool):
                    return self._invalid(
                        "chat_template_kwargs.enable_thinking must be a boolean",
                        "chat_template_kwargs.enable_thinking",
                    )
        stream_options = body.get("stream_options")
        if stream_options is not None:
            if not isinstance(stream_options, dict):
                return self._invalid("stream_options must be an object", "stream_options")
            include_usage = stream_options.get("include_usage")
            if include_usage is not None and not isinstance(include_usage, bool):
                return self._invalid(
                    "stream_options.include_usage must be a boolean",
                    "stream_options.include_usage",
                )
        return True

    def _validate_ollama_body(self, body: dict[str, Any], *, chat: bool) -> bool:
        unsupported = OLLAMA_CHAT_UNSUPPORTED if chat else OLLAMA_GENERATE_UNSUPPORTED
        for k in unsupported:
            if k in body:
                v = body[k]
                if not is_meaningless_value(k, v):
                    self._send_error(
                        400,
                        f"unsupported parameter: {k}",
                        param=k,
                        code="unsupported_parameter",
                    )
                    return False

        if not self._validate_keep_alive(body.get("keep_alive")):
            return False
        if chat and not self._validate_messages(
            body.get("messages"), allow_empty=True, ollama=True
        ):
            return False
        if chat and not self._validate_tools(body.get("tools")):
            return False
        if not chat and not isinstance(body.get("prompt", ""), str):
            return self._invalid("prompt must be a string", "prompt")
        if not self._validate_bool(body, "stream"):
            return False
        if chat and not self._validate_bool(body, "think"):
            return False
        if not chat and not self._validate_bool(body, "raw"):
            return False
        options = body.get("options")
        if options is not None and not isinstance(options, dict):
            return self._invalid("options must be an object", "options")
        opts = options or {}
        for k in OLLAMA_OPTIONS_UNSUPPORTED:
            if k in opts:
                v = opts[k]
                if not is_meaningless_value(k, v):
                    self._send_error(
                        400,
                        f"unsupported parameter in options: {k}",
                        param=f"options.{k}",
                        code="unsupported_parameter",
                    )
                    return False
        if not self._validate_stop(opts.get("stop"), "options.stop"):
            return False
        return self._validate_numbers(opts, "options.")

    def _parse_openai_options(self, body: dict[str, Any]) -> Any:
        assert self.GenOptions is not None
        max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        parsed_opts = {
            "max_tokens": max_tokens,
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "presence_penalty": body.get("presence_penalty"),
            "stop": stop,
            "seed": body.get("seed"),
        }
        filtered_opts = {k: v for k, v in parsed_opts.items() if v is not None}
        return self.GenOptions(**filtered_opts)

    @staticmethod
    def _parse_openai_enable_thinking(body: dict[str, Any]) -> bool | None:
        """OpenAI 요청에서 생각 끄기/켜기 설정 추출.

        - `chat_template_kwargs.enable_thinking=false` -> False
        - `reasoning_effort="none"` -> False
        - 최상위 `enable_thinking` 지원
        - 기본값: None (모델 기본 = 생각 켬)
        """
        effort = body.get("reasoning_effort")
        if isinstance(effort, str) and effort.strip().lower() == "none":
            return False

        ct_kwargs = body.get("chat_template_kwargs")
        if isinstance(ct_kwargs, dict) and "enable_thinking" in ct_kwargs:
            val = ct_kwargs["enable_thinking"]
            if val is not None:
                return bool(val)

        if "enable_thinking" in body and body["enable_thinking"] is not None:
            return bool(body["enable_thinking"])

        return None

    @staticmethod
    def _parse_ollama_enable_thinking(body: dict[str, Any]) -> bool | None:
        """Ollama 요청의 `think` 칸에서 enable_thinking 추출 (기본 None)."""
        if "think" not in body or body["think"] is None:
            return None
        val = body["think"]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("false", "0", "none", "off", "no"):
                return False
            if s in ("true", "1", "on", "yes"):
                return True
        return bool(val)

    def _finish_reason(self, ch: Any) -> str | None:
        if not getattr(ch, "done", False):
            return None
        return getattr(ch, "finish_reason", None) or "stop"

    def _iter_normalized_chunks(
        self,
        raw_iter: Iterator[Any],
        *,
        thinking_enabled: bool,
    ) -> Iterator[tuple[Any, str, str]]:
        """Forward the engine's explicit thinking/text split unchanged."""
        for ch in raw_iter:
            raw_think = getattr(ch, "thinking", "") or ""
            raw_text = getattr(ch, "text", "") or ""
            yield ch, raw_think, raw_text

    def _handle_v1_chat_completions(self) -> None:
        body = self._read_json()
        if body is None:
            return
        if not self._validate_openai_body(body, chat=True):
            return
        model = body.get("model")
        if not self._check_model(model):
            self._send_error(404, f"model {model!r} not found")
            return
        messages = self._normalize_openai_messages(body.get("messages", []))
        tools = body.get("tools")
        stream = body.get("stream", False)
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))
        options = self._parse_openai_options(body)
        enable_thinking = self._parse_openai_enable_thinking(body)
        keep_alive_s = parse_keep_alive(body.get("keep_alive"), self.keep_alive)

        req_id = f"chat-{threading.current_thread().ident}"
        if not self._wait_engine(req_id):
            self._send_error(503, "engine busy, timeout waiting")
            return
        try:
            if stream:
                self._stream_openai_chat(
                    messages, tools, options, enable_thinking, include_usage=include_usage
                )
            else:
                self._single_openai_chat(messages, tools, options, enable_thinking)
        finally:
            self._release_engine(keep_alive_s)

    def _single_openai_chat(
        self,
        messages: list[dict],
        tools: Any,
        options: Any,
        enable_thinking: bool | None = None,
    ) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason: str | None = "stop"
        thinking_enabled = enable_thinking is not False
        try:
            raw_iter = self.engine.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                options=options,
                cancel=cancel,
            )
            for ch, think_piece, text_piece in self._iter_normalized_chunks(
                raw_iter, thinking_enabled=thinking_enabled
            ):
                self._check_cancel(cancel)
                if cancel():
                    break
                if think_piece:
                    thinking_parts.append(think_piece)
                if text_piece:
                    text_parts.append(text_piece)
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                if getattr(ch, "done", False):
                    finish_reason = self._finish_reason(ch)
        except Exception as exc:
            self._send_generation_error(exc, cancel)
            return
        if cancel():
            return

        thinking_text, full_text = _split_accumulated_think(
            thinking_parts,
            text_parts,
            thinking_enabled=thinking_enabled,
            finish_reason=finish_reason,
        )
        message_obj: dict[str, Any] = {
            "role": "assistant",
            "content": full_text,
            "reasoning_content": thinking_text,
        }
        resp = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": message_obj,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if not self._write_json_or_cancel(200, resp, cancel):
            return
        self._log("POST", "/v1/chat/completions", prompt_tokens,
                  completion_tokens, prefill_s, decode_s)

    def _stream_openai_chat(
        self,
        messages: list[dict],
        tools: Any,
        options: Any,
        enable_thinking: bool | None = None,
        *,
        include_usage: bool = False,
    ) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        if not self._start_stream("text/event-stream; charset=utf-8", cancel):
            return

        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason: str | None = "stop"
        idx = 0
        stream_id = f"chatcmpl-{int(time.time()*1000)}"
        created = int(time.time())
        thinking_enabled = enable_thinking is not False
        try:
            raw_iter = self.engine.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                options=options,
                cancel=cancel,
            )
            for ch, think_piece, text_piece in self._iter_normalized_chunks(
                raw_iter, thinking_enabled=thinking_enabled
            ):
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                done = bool(getattr(ch, "done", False))
                if done:
                    finish_reason = self._finish_reason(ch)

                delta: dict[str, Any] = {}
                if think_piece:
                    delta["reasoning_content"] = think_piece
                if text_piece:
                    delta["content"] = text_piece

                chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason if done else None,
                    }],
                }
                if include_usage:
                    chunk["usage"] = None
                if not self._write_sse(chunk, cancel):
                    break
                idx += 1
            else:
                if include_usage:
                    usage_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    self._write_sse(usage_chunk, cancel)
                self._write_done(cancel)
        except Exception as exc:
            error = {
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "param": None,
                    "code": "generation_error",
                }
            }
            self._write_sse(error, cancel)
            self._write_done(cancel)
        self._log("POST", "/v1/chat/completions (stream)", prompt_tokens,
                  completion_tokens, prefill_s, decode_s,
                  extra=f"chunks={idx}")

    def _write_sse(self, obj: Any, cancel: Callable[[], bool]) -> bool:
        try:
            payload = f"data: {json.dumps(obj, ensure_ascii=False, allow_nan=False)}\n\n".encode("utf-8")
            self.connection.sendall(payload)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]
            return False

    def _write_done(self, cancel: Callable[[], bool]) -> None:
        try:
            self.connection.sendall(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]

    def _handle_v1_completions(self) -> None:
        body = self._read_json()
        if body is None:
            return
        if not self._validate_openai_body(body, chat=False):
            return
        model = body.get("model")
        if not self._check_model(model):
            self._send_error(404, f"model {model!r} not found")
            return
        prompt_val = body.get("prompt", "")
        if isinstance(prompt_val, list):
            if len(prompt_val) != 1:
                self._send_error(
                    400,
                    "batch prompts are not supported",
                    param="prompt",
                    code="unsupported_parameter",
                )
                return
            prompt = prompt_val[0]
        else:
            prompt = prompt_val
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))
        options = self._parse_openai_options(body)
        keep_alive_s = parse_keep_alive(body.get("keep_alive"), self.keep_alive)
        req_id = f"compl-{threading.current_thread().ident}"
        if not self._wait_engine(req_id):
            self._send_error(503, "engine busy, timeout waiting")
            return
        try:
            if stream:
                self._stream_openai_completion(
                    prompt, options, include_usage=include_usage
                )
            else:
                self._single_openai_completion(prompt, options)
        finally:
            self._release_engine(keep_alive_s)

    def _single_openai_completion(self, prompt: str, options: Any) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        try:
            for ch in self.engine.generate(prompt, raw=False,
                                           options=options, cancel=cancel):
                self._check_cancel(cancel)
                if cancel():
                    break
                text_parts.append(getattr(ch, "text", "") or "")
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                if getattr(ch, "done", False):
                    finish_reason = self._finish_reason(ch)
        except Exception as exc:
            self._send_generation_error(exc, cancel)
            return
        if cancel():
            return
        full_text = "".join(text_parts)
        resp = {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "text": full_text,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if not self._write_json_or_cancel(200, resp, cancel):
            return
        self._log("POST", "/v1/completions", prompt_tokens,
                  completion_tokens, prefill_s, decode_s)

    def _stream_openai_completion(
        self, prompt: str, options: Any, *, include_usage: bool = False
    ) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        if not self._start_stream("text/event-stream; charset=utf-8", cancel):
            return
        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        idx = 0
        stream_id = f"cmpl-{int(time.time()*1000)}"
        created = int(time.time())
        try:
            for ch in self.engine.generate(prompt, raw=False,
                                           options=options, cancel=cancel):
                text = getattr(ch, "text", "") or ""
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                if getattr(ch, "done", False):
                    finish_reason = self._finish_reason(ch)
                chunk = {
                    "id": stream_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason if getattr(ch, "done", False) else None,
                    }],
                }
                if include_usage:
                    chunk["usage"] = None
                if not self._write_sse(chunk, cancel):
                    break
                idx += 1
            else:
                if include_usage:
                    self._write_sse({
                        "id": stream_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }, cancel)
                self._write_done(cancel)
        except Exception as exc:
            self._write_sse({
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "param": None,
                    "code": "generation_error",
                }
            }, cancel)
            self._write_done(cancel)
        self._log("POST", "/v1/completions (stream)", prompt_tokens,
                  completion_tokens, prefill_s, decode_s,
                  extra=f"chunks={idx}")

    # ------------------------------------------------------------------ #
    # Ollama endpoints
    # ------------------------------------------------------------------ #

    def _handle_api_tags(self) -> None:
        ts = _now()
        size_bytes = self._model_size_bytes()
        details = self._model_details()
        self._write_json(200, {
            "models": [{
                "name": f"{self.model_name}:latest",
                "model": f"{self.model_name}:latest",
                "modified_at": ts,
                "size": size_bytes,
                "digest": "raon-1-mini-digest",
                "details": details,
            }],
        })

    def _handle_api_show(self) -> None:
        body = self._read_json()
        if body is None:
            return
        name = body.get("name", body.get("model"))
        if not self._check_model(name):
            self._send_error(404, f"model {name!r} not found")
            return
        size_bytes = self._model_size_bytes()
        details = self._model_details()
        self._write_json(200, {
            "license": "Apache-2.0",
            "modelfile": f"FROM {self.model_name}\n",
            "parameters": "",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "size": size_bytes,
            "details": details,
        })

    def _handle_api_ps(self) -> None:
        self.check_keep_alive_expired()
        if self._is_loaded():
            with self._busy_lock:
                self._sync_engine_tracking_locked()
                expires_iso = self._expires_at_iso or _now()
            size_bytes = self._model_size_bytes()
            details = self._model_details()
            self._write_json(200, {
                "models": [{
                    "name": f"{self.model_name}:latest",
                    "model": f"{self.model_name}:latest",
                    "size": size_bytes,
                    "size_vram": size_bytes,
                    "digest": "raon-1-mini-digest",
                    "details": details,
                    "expires_at": expires_iso,
                }],
            })
        else:
            self._write_json(200, {"models": []})

    def _parse_ollama_options(self, body: dict[str, Any]) -> Any:
        assert self.GenOptions is not None
        opts = body.get("options") or {}
        max_tokens = opts.get("num_predict")
        stop = opts.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": opts.get("temperature"),
            "top_p": opts.get("top_p"),
            "top_k": opts.get("top_k"),
            "presence_penalty": opts.get("presence_penalty"),
            "stop": stop,
            "seed": opts.get("seed"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.GenOptions(**kwargs)

    def _handle_api_chat(self) -> None:
        body = self._read_json()
        if body is None:
            return
        if not self._validate_ollama_body(body, chat=True):
            return
        model = body.get("model")
        if not self._check_model(model):
            self._send_error(404, f"model {model!r} not found")
            return
        messages = self._normalize_ollama_messages(body.get("messages", []))
        tools = body.get("tools")
        stream = body.get("stream", True)
        options = self._parse_ollama_options(body)
        enable_thinking = self._parse_ollama_enable_thinking(body)
        keep_alive_s = parse_keep_alive(body.get("keep_alive"), self.keep_alive)

        if not messages:
            unloading = keep_alive_s == 0.0
            req_id = f"ollama-{'unload' if unloading else 'load'}-{threading.current_thread().ident}"
            if not self._wait_engine(req_id, load=not unloading):
                self._send_error(503, "engine busy, timeout waiting")
                return
            try:
                self._write_json(200, {
                    "model": self.model_name,
                    "created_at": _now(),
                    "message": {"role": "assistant", "content": "", "thinking": ""},
                    "done": True,
                    "done_reason": "unload" if unloading else "load",
                })
            finally:
                self._release_engine(keep_alive_s)
            return

        req_id = f"ollama-chat-{threading.current_thread().ident}"
        if not self._wait_engine(req_id):
            self._send_error(503, "engine busy, timeout waiting")
            return
        try:
            if stream:
                self._stream_ollama_chat(messages, tools, options, enable_thinking)
            else:
                self._single_ollama_chat(messages, tools, options, enable_thinking)
        finally:
            self._release_engine(keep_alive_s)

    def _single_ollama_chat(self, messages: list[dict], tools: Any, options: Any,
                            enable_thinking: bool | None) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        thinking_enabled = enable_thinking is not False
        try:
            raw_iter = self.engine.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                options=options,
                cancel=cancel,
            )
            for ch, think_piece, text_piece in self._iter_normalized_chunks(
                raw_iter, thinking_enabled=thinking_enabled
            ):
                self._check_cancel(cancel)
                if cancel():
                    break
                if think_piece:
                    thinking_parts.append(think_piece)
                if text_piece:
                    text_parts.append(text_piece)
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                if getattr(ch, "done", False):
                    finish_reason = self._finish_reason(ch)
        except Exception as exc:
            self._send_generation_error(exc, cancel)
            return
        if cancel():
            return

        thinking_text, full_text = _split_accumulated_think(
            thinking_parts,
            text_parts,
            thinking_enabled=thinking_enabled,
            finish_reason=finish_reason,
        )
        resp = {
            "model": self.model_name,
            "created_at": _now(),
            "message": {
                "role": "assistant",
                "content": full_text,
                "thinking": thinking_text,
            },
            "done": True,
            "done_reason": finish_reason,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
            "total_duration": int((prefill_s + decode_s) * 1e9),
        }
        if not self._write_json_or_cancel(200, resp, cancel):
            return
        self._log("POST", "/api/chat", prompt_tokens, completion_tokens,
                  prefill_s, decode_s)

    def _stream_ollama_chat(self, messages: list[dict], tools: Any, options: Any,
                            enable_thinking: bool | None) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        if not self._start_stream("application/x-ndjson; charset=utf-8", cancel):
            return

        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        idx = 0
        thinking_enabled = enable_thinking is not False
        try:
            raw_iter = self.engine.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                options=options,
                cancel=cancel,
            )
            for ch, think_piece, text_piece in self._iter_normalized_chunks(
                raw_iter, thinking_enabled=thinking_enabled
            ):
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                done = bool(getattr(ch, "done", False))
                if done:
                    finish_reason = self._finish_reason(ch)
                chunk = {
                    "model": self.model_name,
                    "created_at": _now(),
                    "message": {
                        "role": "assistant",
                        "content": text_piece,
                        "thinking": think_piece,
                    },
                    "done": done,
                }
                if done:
                    chunk["done_reason"] = finish_reason
                    chunk["prompt_eval_count"] = prompt_tokens
                    chunk["eval_count"] = completion_tokens
                if not self._write_ndjson(chunk, cancel):
                    break
                idx += 1
        except Exception as exc:
            self._write_ndjson({"error": str(exc)}, cancel)
        self._log("POST", "/api/chat (stream)", prompt_tokens,
                  completion_tokens, prefill_s, decode_s,
                  extra=f"chunks={idx}")

    def _write_ndjson(self, obj: Any, cancel: Callable[[], bool]) -> bool:
        try:
            line = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            self.connection.sendall(line)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancel._set()  # type: ignore[attr-defined]
            return False

    def _handle_api_generate(self) -> None:
        body = self._read_json()
        if body is None:
            return
        if not self._validate_ollama_body(body, chat=False):
            return
        model = body.get("model")
        if not self._check_model(model):
            self._send_error(404, f"model {model!r} not found")
            return
        prompt = body.get("prompt", "")
        raw = body.get("raw", False)
        stream = body.get("stream", True)
        options = self._parse_ollama_options(body)
        keep_alive_s = parse_keep_alive(body.get("keep_alive"), self.keep_alive)

        if not prompt:
            unloading = keep_alive_s == 0.0
            req_id = f"ollama-gen-{'unload' if unloading else 'load'}-{threading.current_thread().ident}"
            if not self._wait_engine(req_id, load=not unloading):
                self._send_error(503, "engine busy, timeout waiting")
                return
            try:
                self._write_json(200, {
                    "model": self.model_name,
                    "created_at": _now(),
                    "response": "",
                    "done": True,
                    "done_reason": "unload" if unloading else "load",
                })
            finally:
                self._release_engine(keep_alive_s)
            return

        req_id = f"ollama-gen-{threading.current_thread().ident}"
        if not self._wait_engine(req_id):
            self._send_error(503, "engine busy, timeout waiting")
            return
        try:
            if stream:
                self._stream_ollama_generate(prompt, raw, options)
            else:
                self._single_ollama_generate(prompt, raw, options)
        finally:
            self._release_engine(keep_alive_s)

    def _single_ollama_generate(self, prompt: str, raw: bool,
                                 options: Any) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        try:
            for ch in self.engine.generate(prompt, raw=raw,
                                           options=options, cancel=cancel):
                self._check_cancel(cancel)
                if cancel():
                    break
                text_parts.append(getattr(ch, "text", "") or "")
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                if getattr(ch, "done", False):
                    finish_reason = self._finish_reason(ch)
        except Exception as exc:
            self._send_generation_error(exc, cancel)
            return
        if cancel():
            return
        full_text = "".join(text_parts)
        resp = {
            "model": self.model_name,
            "created_at": _now(),
            "response": full_text,
            "done": True,
            "done_reason": finish_reason,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
            "total_duration": int((prefill_s + decode_s) * 1e9),
        }
        if not self._write_json_or_cancel(200, resp, cancel):
            return
        self._log("POST", "/api/generate", prompt_tokens, completion_tokens,
                  prefill_s, decode_s)

    def _stream_ollama_generate(self, prompt: str, raw: bool,
                                options: Any) -> None:
        assert self.engine is not None
        cancel = self._make_cancel()
        if not self._start_stream("application/x-ndjson; charset=utf-8", cancel):
            return

        prompt_tokens = 0
        completion_tokens = 0
        prefill_s = 0.0
        decode_s = 0.0
        finish_reason = "stop"
        idx = 0
        try:
            for ch in self.engine.generate(prompt, raw=raw,
                                           options=options, cancel=cancel):
                text = getattr(ch, "text", "") or ""
                if getattr(ch, "prompt_tokens", 0):
                    prompt_tokens = ch.prompt_tokens
                if getattr(ch, "completion_tokens", 0):
                    completion_tokens = ch.completion_tokens
                if getattr(ch, "prefill_s", 0.0):
                    prefill_s = ch.prefill_s
                if getattr(ch, "decode_s", 0.0):
                    decode_s = ch.decode_s
                done = bool(getattr(ch, "done", False))
                if done:
                    finish_reason = self._finish_reason(ch)
                chunk = {
                    "model": self.model_name,
                    "created_at": _now(),
                    "response": text,
                    "done": done,
                }
                if done:
                    chunk["done_reason"] = finish_reason
                    chunk["prompt_eval_count"] = prompt_tokens
                    chunk["eval_count"] = completion_tokens
                if not self._write_ndjson(chunk, cancel):
                    break
                idx += 1
        except Exception as exc:
            self._write_ndjson({"error": str(exc)}, cancel)
        self._log("POST", "/api/generate (stream)", prompt_tokens,
                  completion_tokens, prefill_s, decode_s,
                  extra=f"chunks={idx}")


class Server(ThreadingHTTPServer):
    """Nagle을 끄고 즉시 전송해 클라이언트 끊김을 빨리 감지하며 keep-alive 만료를 감시한다."""

    sndbuf: int | None = None
    if os.name == "nt":
        allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt":
            self.allow_reuse_address = False
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()
        try:
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.sndbuf is not None:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
        except (OSError, AttributeError):
            pass
        self._stop_reaper = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="raon-keepalive-reaper", daemon=True
        )
        self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        while not self._stop_reaper.wait(0.02):
            h = self.RequestHandlerClass
            if hasattr(h, "check_keep_alive_expired"):
                try:
                    h.check_keep_alive_expired()
                except Exception:
                    pass

    def server_close(self) -> None:
        if hasattr(self, "_stop_reaper"):
            self._stop_reaper.set()
        super().server_close()


def ranged_int(
    min_val: int | None = None,
    max_val: int | None = None,
    *,
    name: str = "integer",
) -> Any:
    def _parse(val_str: str) -> int:
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(f"{name} must be an integer, got {val_str!r}")
        if min_val is not None and val < min_val:
            raise argparse.ArgumentTypeError(f"{name} must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            raise argparse.ArgumentTypeError(f"{name} must be <= {max_val}, got {val}")
        return val

    _parse.__name__ = f"ranged_int_{name}"
    _parse.is_ranged = True
    _parse.type_cls = int
    _parse.min_val = min_val
    _parse.max_val = max_val
    _parse.min_inclusive = True
    _parse.max_inclusive = True
    _parse.name = name
    return _parse


def ranged_float(
    min_val: float | None = None,
    max_val: float | None = None,
    *,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    name: str = "float",
) -> Any:
    import math

    def _parse(val_str: str) -> float:
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(f"{name} must be a number, got {val_str!r}")
        if not math.isfinite(val):
            raise argparse.ArgumentTypeError(f"{name} must be finite, got {val_str!r}")
        if min_val is not None:
            if min_inclusive and val < min_val:
                raise argparse.ArgumentTypeError(f"{name} must be >= {min_val}, got {val}")
            if not min_inclusive and val <= min_val:
                raise argparse.ArgumentTypeError(f"{name} must be > {min_val}, got {val}")
        if max_val is not None:
            if max_inclusive and val > max_val:
                raise argparse.ArgumentTypeError(f"{name} must be <= {max_val}, got {val}")
            if not max_inclusive and val >= max_val:
                raise argparse.ArgumentTypeError(f"{name} must be < {max_val}, got {val}")
        return val

    _parse.__name__ = f"ranged_float_{name}"
    _parse.is_ranged = True
    _parse.type_cls = float
    _parse.min_val = min_val
    _parse.max_val = max_val
    _parse.min_inclusive = min_inclusive
    _parse.max_inclusive = max_inclusive
    _parse.name = name
    return _parse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Raon NPU HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--port",
        type=ranged_int(min_val=1, max_val=65535, name="port"),
        default=DEFAULT_PORT,
    )
    p.add_argument("--weights-d", default="")
    p.add_argument("--model-dir", default="")
    p.add_argument("--log", default="")
    p.add_argument(
        "--max-wait",
        type=ranged_float(
            min_val=0.0,
            max_val=MAX_WAIT_SECONDS,
            min_inclusive=True,
            max_inclusive=True,
            name="max_wait",
        ),
        default=DEFAULT_MAX_WAIT,
    )
    p.add_argument(
        "--keep-alive",
        type=parse_keep_alive,
        default=DEFAULT_KEEP_ALIVE,
        help="Keep-alive duration after last request (seconds or e.g. '5m'; 0=immediate unload, <0=never unload)",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    keep_alive_s = parse_keep_alive(args.keep_alive, DEFAULT_KEEP_ALIVE)

    logger = _LogWriter(args.log or None)
    Handler.logger = logger
    Handler.max_wait = args.max_wait
    Handler.keep_alive = keep_alive_s
    Handler.weights_dir = args.weights_d
    Handler.model_dir = args.model_dir

    try:
        server = Server((args.host, args.port), Handler)
    except OSError as exc:
        msg = f"Failed to bind {args.host}:{args.port}: {exc}"
        logger.write(f"{_now()} {msg}")
        sys.stderr.write(f"{msg}\n")
        sys.stderr.flush()
        sys.exit(1)

    from serve.engine import Engine, GenOptions, Chunk

    def _make_engine() -> Any:
        return Engine(args.weights_d, args.model_dir, preload=True)

    try:
        engine = _make_engine()
    except Exception:
        server.server_close()
        raise

    Handler.engine_factory = _make_engine
    Handler.engine = engine
    Handler.GenOptions = GenOptions
    Handler.Chunk = Chunk
    Handler.model_name = getattr(engine, "model_name", "raon-1-mini")
    with Handler._busy_lock:
        Handler._sync_engine_tracking_locked()
        if keep_alive_s == 0.0:
            Handler._unload_locked()
        else:
            Handler._set_expiry_locked(keep_alive_s)

    logger.write(
        f"{_now()} server started at {args.host}:{args.port} (keep_alive={keep_alive_s}s)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.write(f"{_now()} server stopped")


if __name__ == "__main__":
    main()
