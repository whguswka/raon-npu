# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon-1-mini streaming generation engine with prefix state reuse."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import numpy as np

from host.raon import RaonConfig, RaonModel
from host.weights import Weights
from host.strict_json import loads_strict
from serve.prefill_plan import prefill_segments
from serve.tokenize import DEFAULT_EOS_IDS, SPECIAL_TOKENS, IncrementalDecoder, RaonTokenizer

IM_START_TOKEN_ID: int = SPECIAL_TOKENS["<|im_start|>"]
IM_END_TOKEN_ID: int = SPECIAL_TOKENS["<|im_end|>"]
THINK_START_TOKEN_ID: int = SPECIAL_TOKENS["<think>"]
THINK_END_TOKEN_ID: int = SPECIAL_TOKENS["</think>"]


@dataclass
class GenOptions:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    stop: list[str] = field(default_factory=list)
    seed: int | None = None  # >= 0: 고정 seed, < 0: 고정하지 않음(무작위 seed)

    def __post_init__(self) -> None:
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop] if self.stop else []
        else:
            self.stop = [str(s) for s in self.stop if s]


@dataclass
class Chunk:  # 스트리밍 단위
    text: str  # 새로 완성된 답 글자(빈 문자열 가능)
    done: bool = False
    finish_reason: str | None = None  # "stop" | "length" | "cancelled"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    reused_prefix_tokens: int = 0
    thinking: str = ""  # 새로 완성된 생각 글자(빈 문자열 가능)


@dataclass
class _PromptCheckpoint:
    tokens: list[int]
    last_logits: np.ndarray
    rda_states: dict[int, np.ndarray]
    conv_states: dict[int, dict[str, np.ndarray]]
    mla_lengths: dict[int, int]
    save_s: float = 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x)
    if denom <= 0.0 or not np.isfinite(denom):
        probs = np.zeros_like(x, dtype=np.float64)
        probs[int(np.argmax(x))] = 1.0
        return probs
    return exp_x / denom


def sample_token(
    logits: np.ndarray,
    seen_tokens: Sequence[int] | set[int],
    options: GenOptions,
    rng: np.random.Generator,
) -> int:
    """Sample next token ID following HF/vLLM conventions.

    Order:
    1. presence_penalty: subtract presence_penalty from logits of already generated tokens.
    2. If temperature <= 0: greedy argmax.
    3. Otherwise: top_k -> top_p (0.0 keeps argmax, 1.0 does not filter) -> temperature scaling -> categorical sample with rng.
    """
    scores = np.array(logits, dtype=np.float64, copy=True).ravel()
    vocab_size = scores.size
    if vocab_size == 0:
        raise ValueError("Cannot sample from empty logits")

    # 1. Presence penalty (subtract once per unique token present in generated output so far)
    if options.presence_penalty != 0.0 and seen_tokens:
        unique_seen = [int(t) for t in set(seen_tokens) if 0 <= int(t) < vocab_size]
        if unique_seen:
            scores[unique_seen] -= float(options.presence_penalty)

    # 2. Greedy decoding when temperature == 0 (or < 0)
    if options.temperature <= 0.0:
        return int(np.argmax(scores))

    # 3. Top-k filtering
    top_k = int(options.top_k) if options.top_k is not None else 0
    if 0 < top_k < vocab_size:
        kth_val = np.partition(scores, -top_k)[-top_k]
        scores[scores < kth_val] = -np.inf

    # 4. Top-p (nucleus) filtering (matching HF TopPLogitsWarper)
    top_p = float(options.top_p) if options.top_p is not None else 1.0
    if 0.0 <= top_p < 1.0:
        sorted_indices = np.argsort(-scores, kind="stable")
        sorted_logits = scores[sorted_indices]
        probs = _softmax(sorted_logits)
        cum_probs = np.cumsum(probs)
        # Remove tokens where cumulative probability before including this token already >= top_p
        # (equivalently: shifted cum_probs > top_p, always keeping at least the top-1 token)
        to_remove = (cum_probs - probs) >= top_p
        to_remove[0] = False
        scores[sorted_indices[to_remove]] = -np.inf

    # 5. Temperature scaling
    if options.temperature != 1.0:
        scores = scores / float(options.temperature)

    probs = _softmax(scores)
    return int(rng.choice(vocab_size, p=probs))


def _find_stop_match(text: str, stop_list: list[str]) -> tuple[int, str] | None:
    """Return earliest (index, stop_str) if any stop string appears in text."""
    best: tuple[int, str] | None = None
    for s in stop_list:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            if best is None or idx < best[0] or (idx == best[0] and len(s) > len(best[1])):
                best = (idx, s)
    return best


def _stop_holdback_len(text: str, stop_list: list[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of any stop string."""
    if not text or not stop_list:
        return 0
    max_hold = 0
    for s in stop_list:
        if len(s) <= 1:
            continue
        max_k = min(len(text), len(s) - 1)
        for k in range(max_k, max_hold, -1):
            if text.endswith(s[:k]):
                max_hold = k
                break
    return max_hold


def _strip_leading_blank_lines(text: str, *, final: bool = False) -> tuple[str, bool]:
    """Strip leading blank lines (`[ \\t]*\\r?\\n`) after `</think>`.

    Returns `(remaining_text, still_stripping)`.
    When `final=False`, if `remaining_text` consists only of spaces/tabs or CR without LF yet,
    holds it back (`still_stripping=True`) until the next character decides whether the line is blank.
    """
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


class Engine:
    """Thread-safe streaming generation engine wrapping RaonModel and RaonTokenizer."""

    model_name: str = "raon-1-mini"

    def __init__(
        self,
        weights_dir: str | Path | None,
        model_dir: str | Path | Any | None,
        *,
        backend: Any = None,
        model: Any = None,
        preload: bool = True,
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.model_dir = Path(model_dir) if isinstance(model_dir, (str, Path)) and model_dir else None
        self._preload = preload

        # 1. Initialize tokenizer
        if model_dir is not None and hasattr(model_dir, "encode") and hasattr(model_dir, "decode"):
            self.tokenizer: RaonTokenizer | Any = model_dir
        else:
            tok_dir = self._resolve_tokenizer_dir()
            self.tokenizer = RaonTokenizer(tok_dir)

        self.eos_ids: list[int] = list(getattr(self.tokenizer, "eos_ids", DEFAULT_EOS_IDS))

        # 2. Initialize backend & model (skip heavy loading when `model` is injected for tests)
        if model is not None:
            self.model = model
            self.backend = backend if backend is not None else getattr(model, "backend", None)
        else:
            self.backend, self.model = self._build_model(preload=preload, backend=backend)

        self._lock = threading.Lock()
        self._cached_tokens: list[int] = []
        self._last_logits: np.ndarray | None = None
        self._checkpoint: _PromptCheckpoint | None = None
        self.last_save_state_s: float = 0.0
        self.last_restore_state_s: float = 0.0
        self._reset_state(clear_checkpoint=True)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        """Release NPU weight buffers, host weights, and recurrent caches."""
        with self._lock:
            self._reset_state(clear_checkpoint=True)
            be = self.backend
            if be is not None:
                for attr in ("_npu_weights", "_work_buffers", "_compiled_ops", "mock_weight_arrays"):
                    d = getattr(be, attr, None)
                    if isinstance(d, dict):
                        d.clear()
                if hasattr(be, "_mock_d_weights"):
                    be._mock_d_weights = None
                if hasattr(be, "host_weights"):
                    be.host_weights = None
            mdl = self.model
            if mdl is not None:
                w = getattr(mdl, "weights", None)
                if w is not None:
                    for attr in ("tensors", "_bundle_cache", "_expert_cache"):
                        d = getattr(w, attr, None)
                        if isinstance(d, dict):
                            d.clear()
                    if hasattr(w, "host_weights"):
                        w.host_weights = None
                if hasattr(mdl, "backend"):
                    mdl.backend = None
                if hasattr(mdl, "weights"):
                    mdl.weights = None
            self.backend = None
            self.model = None
        gc.collect()

    def _resolve_tokenizer_dir(self) -> Path:
        candidates: list[Path] = []
        if self.model_dir is not None:
            candidates.append(self.model_dir)
        if self.weights_dir is not None:
            candidates.append(self.weights_dir)
        for cand in candidates:
            if (cand / "tokenizer.json").is_file():
                return cand
        if candidates:
            return candidates[0]
        raise ValueError("Either model_dir or weights_dir must be provided to load RaonTokenizer")

    def _build_model(self, *, preload: bool, backend: Any) -> tuple[Any, RaonModel]:
        from npu.backend import DWeights, NpuBackend

        w_dir = self.weights_dir or self.model_dir
        if w_dir is None:
            raise ValueError("weights_dir or model_dir is required to initialize RaonModel")

        cfg_path = (
            (self.model_dir / "config.json")
            if (self.model_dir and (self.model_dir / "config.json").is_file())
            else ((w_dir / "config.json") if (w_dir / "config.json").is_file() else None)
        )
        if cfg_path and cfg_path.is_file():
            cfg_dict = loads_strict(cfg_path.read_text(encoding="utf-8"))
            config = RaonConfig.from_dict(cfg_dict)
        else:
            config = RaonConfig.from_dict({})

        num_layers = config.num_hidden_layers

        host_source = w_dir
        if not any(host_source.glob("*.safetensors")):
            if (w_dir / "host").is_dir() and any((w_dir / "host").glob("*.safetensors")):
                host_source = w_dir / "host"
            elif self.model_dir and self.model_dir.is_dir() and any(self.model_dir.glob("*.safetensors")):
                host_source = self.model_dir

        host_weights: Weights | None = None
        if host_source.is_dir() and any(host_source.glob("*.safetensors")):
            host_weights = Weights(
                host_source,
                selected_layers=list(range(num_layers)),
                skip_moe_experts=True,
            )

        if backend is None:
            backend = NpuBackend(
                weights_dir=w_dir,
                host_weights=host_weights,
            )
            if preload:
                backend.preload(num_layers=num_layers)

        model_weights: Weights
        if host_weights is not None:
            model_weights = host_weights
        else:
            model_weights = DWeights(
                w_dir,
                host_weights=None,
                selected_layers=list(range(num_layers)),
            )

        model = RaonModel(config, model_weights, backend=backend)
        return backend, model

    def _get_num_layers(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            return int(cfg.num_hidden_layers)
        return 25

    def _reset_state(self, *, clear_checkpoint: bool = False) -> None:
        """Reset model state and clear prefix token cache together."""
        if self.model is not None and hasattr(self.model, "reset_state"):
            self.model.reset_state()
        elif self.backend is not None and hasattr(self.backend, "reset"):
            self.backend.reset()
        self._cached_tokens.clear()
        self._last_logits = None
        if clear_checkpoint:
            self._checkpoint = None

    def _save_checkpoint(self) -> None:
        """Save prompt state (RDA states, conv states, MLA cache lengths) via backend get/set contract."""
        if self._last_logits is None or not self._cached_tokens:
            return
        t0 = time.perf_counter()
        rda_states: dict[int, np.ndarray] = {}
        conv_states: dict[int, dict[str, np.ndarray]] = {}
        mla_lengths: dict[int, int] = {}

        num_layers = self._get_num_layers()
        be = self.backend
        if be is not None:
            if hasattr(be, "get_rda_state"):
                for layer_idx in range(num_layers):
                    st = be.get_rda_state(layer_idx)
                    if st is not None:
                        rda_states[layer_idx] = np.array(st, dtype=np.float32, copy=True)
            if hasattr(be, "get_mla_cache"):
                for layer_idx in range(num_layers):
                    k_cache, v_cache = be.get_mla_cache(layer_idx)
                    if k_cache is not None and v_cache is not None:
                        t_axis = 2 if k_cache.ndim == 4 else 1
                        mla_lengths[layer_idx] = int(k_cache.shape[t_axis])

        model_conv = getattr(self.model, "conv_states", None)
        if isinstance(model_conv, dict):
            for layer_idx, ldict in model_conv.items():
                if isinstance(ldict, dict):
                    conv_states[int(layer_idx)] = {
                        k: np.array(v, copy=True) for k, v in ldict.items() if v is not None
                    }

        save_s = time.perf_counter() - t0
        self.last_save_state_s = save_s
        self._checkpoint = _PromptCheckpoint(
            tokens=list(self._cached_tokens),
            last_logits=np.array(self._last_logits, copy=True),
            rda_states=rda_states,
            conv_states=conv_states,
            mla_lengths=mla_lengths,
            save_s=save_s,
        )

    def _restore_checkpoint(self, cp: _PromptCheckpoint) -> None:
        """Restore saved prompt state (RDA states, conv states, MLA cache lengths)."""
        t0 = time.perf_counter()
        num_layers = self._get_num_layers()
        be = self.backend
        if be is not None:
            if hasattr(be, "set_rda_state"):
                for layer_idx, st in cp.rda_states.items():
                    be.set_rda_state(layer_idx, np.array(st, dtype=np.float32, copy=True))
            if hasattr(be, "get_mla_cache") and hasattr(be, "set_mla_cache"):
                for layer_idx in range(num_layers):
                    target_len = cp.mla_lengths.get(layer_idx)
                    if target_len is None:
                        continue
                    k_cache, v_cache = be.get_mla_cache(layer_idx)
                    if k_cache is not None and v_cache is not None:
                        if k_cache.ndim == 4:
                            k_sliced = np.ascontiguousarray(k_cache[:, :, :target_len, :])
                            v_sliced = np.ascontiguousarray(v_cache[:, :, :target_len, :])
                        else:
                            k_sliced = np.ascontiguousarray(k_cache[:, :target_len, :])
                            v_sliced = np.ascontiguousarray(v_cache[:, :target_len, :])
                        be.set_mla_cache(layer_idx, k_sliced, v_sliced)

        model_conv = getattr(self.model, "conv_states", None)
        if isinstance(model_conv, dict):
            model_conv.clear()
            for layer_idx, ldict in cp.conv_states.items():
                model_conv[int(layer_idx)] = {
                    k: np.array(v, copy=True) for k, v in ldict.items()
                }

        self._cached_tokens = list(cp.tokens)
        self._last_logits = np.array(cp.last_logits, copy=True)
        self.last_restore_state_s = time.perf_counter() - t0

    def _find_checkpoint_len(
        self,
        prompt_ids: list[int],
        *,
        base_ids: list[int] | None = None,
    ) -> int:
        """Find prefix length excluding trailing generation prompt (`<|im_start|>assistant\\n<think>\\n`)."""
        n = len(prompt_ids)
        if base_ids is not None and 0 < len(base_ids) <= n and prompt_ids[: len(base_ids)] == base_ids:
            return len(base_ids)

        # Look for trailing unclosed <|im_start|> (259201) without a following <|im_end|> (259202)
        for i in range(n - 1, -1, -1):
            tid = int(prompt_ids[i])
            if tid == IM_END_TOKEN_ID:
                break
            if tid == IM_START_TOKEN_ID:
                if i > 0:
                    return i
                break

        # Fallback for fake tokenizers that encode the generation prompt into arbitrary token IDs
        for suffix in (
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think></think>\n\n",
        ):
            try:
                sfx_ids = [int(x) for x in self.tokenizer.encode(suffix)]
            except Exception:
                sfx_ids = []
            if 0 < len(sfx_ids) < n and prompt_ids[-len(sfx_ids) :] == sfx_ids:
                return n - len(sfx_ids)

        return n

    def _prompt_ends_with_open_think(self, prompt_ids: Sequence[int]) -> bool:
        """Return True if prompt ends with an unclosed `<think>` (plus optional whitespace/newline)."""
        for i in range(len(prompt_ids) - 1, -1, -1):
            tid = int(prompt_ids[i])
            if tid == THINK_END_TOKEN_ID:
                return False
            if tid == THINK_START_TOKEN_ID:
                tail_ids = [int(x) for x in prompt_ids[i + 1 :]]
                if not tail_ids:
                    return True
                try:
                    tail_str = self.tokenizer.decode(tail_ids)
                except Exception:
                    tail_str = ""
                return isinstance(tail_str, str) and tail_str.strip() == ""

        try:
            decoded = self.tokenizer.decode(list(prompt_ids))
            if isinstance(decoded, str) and decoded.rstrip().endswith("<think>"):
                return True
        except Exception:
            pass
        return False

    def _step_token(self, token_id: int, *, need_logits: bool = True) -> np.ndarray | None:
        """Advance model by one token while keeping _cached_tokens strictly synchronized."""
        tid = int(token_id)
        try:
            if self.backend is not None and hasattr(self.backend, "set_step"):
                self.backend.set_step(len(self._cached_tokens) + 1)
            if hasattr(self.model, "advance"):
                logits = self.model.advance(tid, need_logits=need_logits)
            else:
                logits = self.model.step(tid)
                if not need_logits:
                    logits = None
        except BaseException:
            # If step fails mid-way, state may be partially mutated -> reset to prevent desync.
            self._reset_state(clear_checkpoint=True)
            raise
        self._cached_tokens.append(tid)
        self._last_logits = logits
        return logits

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        options: GenOptions | None = None,
        cancel: Callable[[], bool] = lambda: False,
    ) -> Iterator[Chunk]:
        """Render messages with ChatML template and stream generated Chunks."""
        opts = options if options is not None else GenOptions()
        prompt_ids = self.tokenizer.chat_ids(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        base_ids: list[int] | None = None
        try:
            base_ids = self.tokenizer.chat_ids(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
            )
        except Exception:
            base_ids = None
        return self._run_stream(prompt_ids, opts, cancel, base_ids=base_ids)

    def generate(
        self,
        prompt: str,
        *,
        raw: bool = False,
        options: GenOptions | None = None,
        cancel: Callable[[], bool] = lambda: False,
    ) -> Iterator[Chunk]:
        """Generate from a text prompt (wrapped as a single user chat message if raw=False)."""
        opts = options if options is not None else GenOptions()
        base_ids: list[int] | None = None
        if raw:
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            msgs = [{"role": "user", "content": prompt}]
            prompt_ids = self.tokenizer.chat_ids(
                msgs,
                add_generation_prompt=True,
            )
            try:
                base_ids = self.tokenizer.chat_ids(
                    msgs,
                    add_generation_prompt=False,
                )
            except Exception:
                base_ids = None
        return self._run_stream(prompt_ids, opts, cancel, base_ids=base_ids)

    def _run_stream(
        self,
        prompt_ids: list[int],
        options: GenOptions,
        cancel: Callable[[], bool] | None,
        *,
        base_ids: list[int] | None = None,
    ) -> Iterator[Chunk]:
        if not prompt_ids:
            raise ValueError("Prompt produced 0 tokens")

        is_cancelled = cancel if cancel is not None else (lambda: False)

        with self._lock:
            prompt_tokens = len(prompt_ids)
            checkpoint_len = self._find_checkpoint_len(prompt_ids, base_ids=base_ids)
            seed = options.seed if (options.seed is not None and options.seed >= 0) else None
            rng = np.random.default_rng(seed)

            # -----------------------------------------------------------------
            # 1. Prefill (reuse full _cached_tokens, or saved prompt checkpoint, else reset)
            # -----------------------------------------------------------------
            t_prefill0 = time.perf_counter()
            n_cached = len(self._cached_tokens)
            if (
                n_cached > 0
                and self._last_logits is not None
                and prompt_tokens >= n_cached
                and prompt_ids[:n_cached] == self._cached_tokens
            ):
                reused_prefix_tokens = n_cached
                to_prefill = prompt_ids[n_cached:]
            elif (
                self._checkpoint is not None
                and len(self._checkpoint.tokens) > 0
                and prompt_tokens >= len(self._checkpoint.tokens)
                and prompt_ids[: len(self._checkpoint.tokens)] == self._checkpoint.tokens
            ):
                self._restore_checkpoint(self._checkpoint)
                reused_prefix_tokens = len(self._checkpoint.tokens)
                to_prefill = prompt_ids[reused_prefix_tokens:]
            else:
                self._reset_state(clear_checkpoint=True)
                reused_prefix_tokens = 0
                to_prefill = prompt_ids

            saved_this_turn = (
                self._checkpoint is not None
                and self._checkpoint.tokens == prompt_ids[:checkpoint_len]
            )

            prefill_chunk_str = os.environ.get("RAON_PREFILL_CHUNK", "0")
            try:
                prefill_chunk = int(prefill_chunk_str)
            except ValueError:
                prefill_chunk = 0

            use_batched_prefill = prefill_chunk > 0 and hasattr(self.model, "prefill")

            def _make_prefill_cancel_chunk() -> Chunk:
                return Chunk(
                    text="",
                    done=True,
                    finish_reason="cancelled",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    prefill_s=time.perf_counter() - t_prefill0,
                    decode_s=0.0,
                    reused_prefix_tokens=reused_prefix_tokens,
                    thinking="",
                )

            if use_batched_prefill:
                segments = prefill_segments(
                    cached=len(self._cached_tokens),
                    n=len(to_prefill),
                    chunk=prefill_chunk,
                    checkpoint_len=checkpoint_len,
                    prompt_tokens=prompt_tokens,
                )
                p_idx = 0
                for step_size, logits_positions in segments:
                    if is_cancelled():
                        yield _make_prefill_cancel_chunk()
                        return

                    chunk_tokens = to_prefill[p_idx : p_idx + step_size]

                    try:
                        if self.backend is not None and hasattr(self.backend, "set_step"):
                            self.backend.set_step(len(self._cached_tokens) + 1)
                        logits_map = self.model.prefill(
                            chunk_tokens,
                            logits_positions=logits_positions,
                            chunk=step_size,
                        )
                    except BaseException:
                        self._reset_state(clear_checkpoint=True)
                        raise

                    self._cached_tokens.extend(chunk_tokens)
                    p_idx += step_size

                    last_pos = len(chunk_tokens) - 1
                    if last_pos in logits_map:
                        self._last_logits = logits_map[last_pos]
                    else:
                        self._last_logits = None

                    if len(self._cached_tokens) == checkpoint_len:
                        self._save_checkpoint()
                        saved_this_turn = True
            else:
                for tid in to_prefill:
                    if is_cancelled():
                        yield _make_prefill_cancel_chunk()
                        return
                    next_pos = len(self._cached_tokens) + 1
                    need_logits = (next_pos == checkpoint_len or next_pos == prompt_tokens)
                    self._step_token(tid, need_logits=need_logits)
                    if len(self._cached_tokens) == checkpoint_len:
                        self._save_checkpoint()
                        saved_this_turn = True

            if not saved_this_turn and len(self._cached_tokens) == checkpoint_len:
                self._save_checkpoint()

            prefill_s = time.perf_counter() - t_prefill0
            assert self._last_logits is not None

            # -----------------------------------------------------------------
            # 2. Decode loop (with <think> ... </think> splitting)
            # -----------------------------------------------------------------
            t_decode0 = time.perf_counter()
            max_tokens = max(0, int(options.max_tokens))
            if max_tokens == 0:
                yield Chunk(
                    text="",
                    done=True,
                    finish_reason="length",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    prefill_s=prefill_s,
                    decode_s=0.0,
                    reused_prefix_tokens=reused_prefix_tokens,
                    thinking="",
                )
                return

            inc_decoder = IncrementalDecoder(self.tokenizer, skip_special_tokens=False)
            stop_list = list(options.stop) if options.stop else []
            generated_ids: list[int] = []
            pending_text = ""

            in_thinking = self._prompt_ends_with_open_think(prompt_ids)
            stripping_leading_blank = False

            for _ in range(max_tokens):
                if is_cancelled():
                    decode_s = time.perf_counter() - t_decode0
                    tail = pending_text + inc_decoder.flush()
                    if in_thinking:
                        yield Chunk(
                            text="",
                            done=True,
                            finish_reason="cancelled",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=tail,
                        )
                    else:
                        if stripping_leading_blank:
                            tail, stripping_leading_blank = _strip_leading_blank_lines(
                                tail, final=True
                            )
                        yield Chunk(
                            text=tail,
                            done=True,
                            finish_reason="cancelled",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking="",
                        )
                    return

                next_token = sample_token(self._last_logits, generated_ids, options, rng)

                # Check EOS token
                if next_token in self.eos_ids:
                    decode_s = time.perf_counter() - t_decode0
                    tail = pending_text + inc_decoder.flush()
                    if in_thinking:
                        yield Chunk(
                            text="",
                            done=True,
                            finish_reason="stop",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=tail,
                        )
                    else:
                        if stripping_leading_blank:
                            tail, stripping_leading_blank = _strip_leading_blank_lines(
                                tail, final=True
                            )
                        yield Chunk(
                            text=tail,
                            done=True,
                            finish_reason="stop",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking="",
                        )
                    return

                generated_ids.append(next_token)

                # Explicit </think> token (259214) transitions from thinking -> content
                if in_thinking and int(next_token) == THINK_END_TOKEN_ID:
                    thinking_tail = pending_text + inc_decoder.flush()
                    inc_decoder.reset()
                    pending_text = ""
                    in_thinking = False
                    stripping_leading_blank = True

                    self._step_token(next_token)
                    decode_s = time.perf_counter() - t_decode0

                    if len(generated_ids) >= max_tokens:
                        yield Chunk(
                            text="",
                            done=True,
                            finish_reason="length",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=thinking_tail,
                        )
                        return

                    yield Chunk(
                        text="",
                        done=False,
                        finish_reason=None,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(generated_ids),
                        prefill_s=prefill_s,
                        decode_s=decode_s,
                        reused_prefix_tokens=reused_prefix_tokens,
                        thinking=thinking_tail,
                    )
                    continue

                piece = inc_decoder.step(next_token)
                pending_text += piece

                # Advance model state with the newly generated token so _cached_tokens
                # always equals prompt_ids + generated_ids even if stopped/cancelled later.
                self._step_token(next_token)
                decode_s = time.perf_counter() - t_decode0

                # Fallback if </think> was decoded from non-259214 token(s) in a fake tokenizer
                split_thinking = ""
                if in_thinking and "</think>" in pending_text:
                    split_thinking, after_think = pending_text.split("</think>", 1)
                    in_thinking = False
                    stripping_leading_blank = True
                    pending_text = after_think

                if in_thinking:
                    # Check stop strings inside thinking
                    if stop_list:
                        match = _find_stop_match(pending_text, stop_list)
                        if match is not None:
                            cut_idx, _ = match
                            emit_thinking = pending_text[:cut_idx]
                            yield Chunk(
                                text="",
                                done=True,
                                finish_reason="stop",
                                prompt_tokens=prompt_tokens,
                                completion_tokens=len(generated_ids),
                                prefill_s=prefill_s,
                                decode_s=decode_s,
                                reused_prefix_tokens=reused_prefix_tokens,
                                thinking=emit_thinking,
                            )
                            return

                    # Check max_tokens limit inside thinking -> content is empty, finish_reason="length"
                    if len(generated_ids) >= max_tokens:
                        emit_thinking = pending_text + inc_decoder.flush()
                        yield Chunk(
                            text="",
                            done=True,
                            finish_reason="length",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=emit_thinking,
                        )
                        return

                    hold = _stop_holdback_len(pending_text, stop_list + ["</think>"])
                    if hold > 0:
                        emit_thinking = pending_text[:-hold]
                        pending_text = pending_text[-hold:]
                    else:
                        emit_thinking = pending_text
                        pending_text = ""

                    yield Chunk(
                        text="",
                        done=False,
                        finish_reason=None,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(generated_ids),
                        prefill_s=prefill_s,
                        decode_s=decode_s,
                        reused_prefix_tokens=reused_prefix_tokens,
                        thinking=emit_thinking,
                    )
                    continue

                # Content phase (after </think> or when thinking was disabled)
                is_last_step = len(generated_ids) >= max_tokens
                if is_last_step:
                    pending_text += inc_decoder.flush()

                if stripping_leading_blank:
                    pending_text, stripping_leading_blank = _strip_leading_blank_lines(
                        pending_text, final=is_last_step
                    )
                    if stripping_leading_blank and not is_last_step:
                        yield Chunk(
                            text="",
                            done=False,
                            finish_reason=None,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=split_thinking,
                        )
                        continue

                # Check stop strings
                if stop_list:
                    match = _find_stop_match(pending_text, stop_list)
                    if match is not None:
                        cut_idx, _ = match
                        emit_text = pending_text[:cut_idx]
                        yield Chunk(
                            text=emit_text,
                            done=True,
                            finish_reason="stop",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=len(generated_ids),
                            prefill_s=prefill_s,
                            decode_s=decode_s,
                            reused_prefix_tokens=reused_prefix_tokens,
                            thinking=split_thinking,
                        )
                        return

                # Check max_tokens limit
                if is_last_step:
                    yield Chunk(
                        text=pending_text,
                        done=True,
                        finish_reason="length",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(generated_ids),
                        prefill_s=prefill_s,
                        decode_s=decode_s,
                        reused_prefix_tokens=reused_prefix_tokens,
                        thinking=split_thinking,
                    )
                    return

                # Hold back any suffix that could be a partial stop string prefix
                hold = _stop_holdback_len(pending_text, stop_list)
                if hold > 0:
                    emit_text = pending_text[:-hold]
                    pending_text = pending_text[-hold:]
                else:
                    emit_text = pending_text
                    pending_text = ""

                yield Chunk(
                    text=emit_text,
                    done=False,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(generated_ids),
                    prefill_s=prefill_s,
                    decode_s=decode_s,
                    reused_prefix_tokens=reused_prefix_tokens,
                    thinking=split_thinking,
                )
