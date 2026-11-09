# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon-1-mini tokenizer, ChatML template renderer, and streaming incremental decoder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from host.strict_json import loads_strict

DEFAULT_EOS_IDS: list[int] = [259202, 259200]

SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 259200,
    "<|im_start|>": 259201,
    "<|im_end|>": 259202,
    "<think>": 259213,
    "</think>": 259214,
}


class _GenerationExtension(jinja2.ext.Extension):
    """Jinja2 extension supporting HuggingFace {% generation %}...{% endgeneration %} tags."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> list[jinja2.nodes.Node]:
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(
    x: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _compile_chat_template(template_str: str) -> jinja2.Template:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationExtension, jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env.from_string(template_str)


class IncrementalDecoder:
    """Streaming token decoder for byte-level BPE tokenizers.

    Holds back incomplete multi-byte UTF-8 sequences (which decode to trailing
    U+FFFD replacement characters in isolation) and only emits completed characters.
    """

    def __init__(
        self,
        tokenizer: "RaonTokenizer | Tokenizer | Any",
        *,
        skip_special_tokens: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._pending: list[int] = []
        self._emitted: int = 0
        self._text_parts: list[str] = []

    def _raw_decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        if isinstance(self._tokenizer, RaonTokenizer):
            return self._tokenizer.decode(ids, skip_special_tokens=self._skip_special_tokens)
        if isinstance(self._tokenizer, Tokenizer):
            return self._tokenizer.decode(ids, skip_special_tokens=self._skip_special_tokens)
        try:
            return self._tokenizer.decode(ids, skip_special_tokens=self._skip_special_tokens)
        except TypeError:
            return self._tokenizer.decode(ids)

    def step(self, token_id: int) -> str:
        """Feed one token ID and return newly completed characters (or '' if still incomplete)."""
        self._pending.append(int(token_id))
        decoded = self._raw_decode(self._pending)
        if not decoded.endswith("\ufffd"):
            piece = decoded[self._emitted :]
            self._pending.clear()
            self._emitted = 0
        else:
            clean = decoded.rstrip("\ufffd")
            piece = clean[self._emitted :]
            self._emitted = len(clean)
        if piece:
            self._text_parts.append(piece)
        return piece

    def add(self, token_or_ids: int | Iterable[int]) -> str:
        """Feed a single token ID or an iterable of token IDs and return newly completed text."""
        if isinstance(token_or_ids, (int,)):
            return self.step(int(token_or_ids))
        return "".join(self.step(int(tid)) for tid in token_or_ids)

    push = step
    feed = add
    decode = add
    __call__ = step

    def flush(self) -> str:
        """Flush any remaining buffered tokens."""
        if not self._pending:
            return ""
        decoded = self._raw_decode(self._pending)
        piece = decoded[self._emitted :]
        self._pending.clear()
        self._emitted = 0
        if piece:
            self._text_parts.append(piece)
        return piece

    def reset(self) -> None:
        """Clear internal state."""
        self._pending.clear()
        self._emitted = 0
        self._text_parts.clear()

    @property
    def text(self) -> str:
        """All completed text emitted so far."""
        return "".join(self._text_parts)


class RaonTokenizer:
    """Tokenizer and ChatML formatter for Raon-1-mini."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.model_dir}")
        tokenizer_text = tok_path.read_text(encoding="utf-8")
        loads_strict(tokenizer_text)
        self._tokenizer = Tokenizer.from_str(tokenizer_text)

        # Load EOS token IDs from generation_config.json or tokenizer_config.json
        self.eos_ids: list[int] = list(DEFAULT_EOS_IDS)
        gen_cfg_path = self.model_dir / "generation_config.json"
        if gen_cfg_path.is_file():
            gen_cfg = loads_strict(gen_cfg_path.read_text(encoding="utf-8"))
            eos_val = gen_cfg.get("eos_token_id")
            if isinstance(eos_val, int):
                self.eos_ids = [eos_val]
            elif isinstance(eos_val, list) and eos_val:
                self.eos_ids = [int(x) for x in eos_val]

        # Load ChatML Jinja2 template
        tpl_path = self.model_dir / "chat_template.jinja"
        if tpl_path.is_file():
            self._chat_template_str = tpl_path.read_text(encoding="utf-8")
        else:
            tok_cfg_path = self.model_dir / "tokenizer_config.json"
            tpl_from_cfg = None
            if tok_cfg_path.is_file():
                tok_cfg = loads_strict(tok_cfg_path.read_text(encoding="utf-8"))
                tpl_from_cfg = tok_cfg.get("chat_template")
            if not tpl_from_cfg:
                raise FileNotFoundError(f"Missing chat_template.jinja in {self.model_dir}")
            self._chat_template_str = str(tpl_from_cfg)

        self._template = _compile_chat_template(self._chat_template_str)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs (special tokens in text are handled by added_tokens)."""
        if not text:
            return []
        return [int(tid) for tid in self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids]

    def decode(self, ids: Iterable[int] | int, *, skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into a string."""
        if isinstance(ids, int):
            id_list = [int(ids)]
        else:
            id_list = [int(tid) for tid in ids]
        if not id_list:
            return ""
        return self._tokenizer.decode(id_list, skip_special_tokens=skip_special_tokens)

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
        *,
        truncate_history_thinking: bool = False,
    ) -> str:
        """Render chat messages into a ChatML string using chat_template.jinja."""
        kwargs: dict[str, Any] = {
            "messages": list(messages),
            "add_generation_prompt": bool(add_generation_prompt),
            "tools": tools,
            "truncate_history_thinking": bool(truncate_history_thinking),
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        return self._template.render(**kwargs)

    def chat_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
        *,
        truncate_history_thinking: bool = False,
    ) -> list[int]:
        """Render chat messages and encode into token IDs."""
        rendered = self.render_chat(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
            truncate_history_thinking=truncate_history_thinking,
        )
        return self.encode(rendered, add_special_tokens=False)

    def incremental_decoder(self, *, skip_special_tokens: bool = False) -> IncrementalDecoder:
        """Create a new streaming IncrementalDecoder bound to this tokenizer."""
        return IncrementalDecoder(self, skip_special_tokens=skip_special_tokens)
