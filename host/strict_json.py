# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Strict JSON decoding helpers rejecting non-finite numbers and excessive recursion."""

from __future__ import annotations

import json
import math
from typing import Any


def _reject_non_finite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")
    return parsed


def loads_strict(data: bytes | str) -> Any:
    """Decode JSON while rejecting every non-finite JSON number and excessive recursion."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    try:
        return json.loads(
            data,
            parse_constant=_reject_non_finite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except RecursionError as e:
        raise ValueError(f"JSON recursion limit exceeded: {e}") from e
