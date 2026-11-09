# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Shared JSON-tree helpers for chat request contract tests."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterator


JsonPath = tuple[str | int, ...]


def build_all_fields_messages(api_type: str) -> list[dict[str, Any]]:
    """Build one history that exercises every accepted message field."""
    is_ollama = api_type == "ollama"
    tool_args = {"location": "Seoul", "format": "celsius"}
    tool_call: dict[str, Any] = {
        "function": {
            "name": "check_weather",
            "arguments": tool_args if is_ollama else json.dumps(tool_args, allow_nan=False),
        }
    }
    if not is_ollama:
        tool_call.update({"index": 0, "id": "call_test_001", "type": "function"})

    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "서울 날씨를 조회하겠습니다.",
        "tool_calls": [tool_call],
    }
    if is_ollama:
        assistant["thinking"] = "서울 날씨 확인을 위해 check_weather 도구를 먼저 호출해야 한다."
    else:
        assistant["reasoning_content"] = (
            "서울 날씨 확인을 위해 check_weather 도구를 먼저 호출해야 한다."
        )
        assistant["loss"] = False

    tool: dict[str, Any] = {
        "role": "tool",
        "content": '{"status": "ok", "temp": 18}',
    }
    if is_ollama:
        tool["tool_name"] = "check_weather"
    else:
        tool["tool_call_id"] = "call_test_001"

    return [
        {"role": "system", "content": "지침: 안전하고 정확하게 답하십시오."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "현재 서울 날씨와 시간 알려줘."}],
        },
        assistant,
        tool,
        {"role": "user", "content": "결과 고마워."},
    ]


def _child_schema_path(parent: str, key: str) -> str:
    component = "tool_call" if key == "tool_calls" else key
    return f"{parent}.{component}" if parent else component


def walk_chat_objects(
    messages: list[dict[str, Any]], free_form_value_paths: frozenset[str]
) -> Iterator[tuple[JsonPath, str, dict[str, Any]]]:
    """Yield every protocol object, excluding mappings declared as free-form values."""

    def walk(value: Any, path: JsonPath, schema_path: str):
        if isinstance(value, dict):
            yield path, schema_path, value
            for key, child in value.items():
                child_schema = _child_schema_path(schema_path, key)
                if child_schema in free_form_value_paths:
                    continue
                yield from walk(child, path + (key,), child_schema)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, path + (index,), schema_path)

    for message_index, message in enumerate(messages):
        yield from walk(message, (message_index,), "")


def chat_used_fields(
    messages: list[dict[str, Any]], free_form_value_paths: frozenset[str]
) -> set[str]:
    """Collect accepted-field names with the server's existing path convention."""
    fields: set[str] = set()
    for _, schema_path, value in walk_chat_objects(messages, free_form_value_paths):
        for key in value:
            if schema_path == "tool_call" and key == "function":
                continue
            fields.add(key if not schema_path else _child_schema_path(schema_path, key))
    return fields


def copy_with_unknown_field(
    messages: list[dict[str, Any]], object_path: JsonPath
) -> tuple[list[dict[str, Any]], str]:
    """Copy a history and inject an unknown key into the selected object."""
    copied = deepcopy(messages)
    target: Any = copied
    for component in object_path:
        target = target[component]
    target["unknown_field"] = "rejected"
    param = "messages" + "".join(f".{component}" for component in object_path)
    return copied, f"{param}.unknown_field"
