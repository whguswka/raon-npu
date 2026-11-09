# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Prefill segmentation planning utilities."""

from __future__ import annotations

from collections.abc import Collection


def prefill_segments(
    cached: int,
    n: int,
    chunk: int,
    checkpoint_len: int,
    prompt_tokens: int,
    *,
    target_logits_positions: Collection[int] | None = None,
) -> list[tuple[int, list[int]]]:
    """Calculate prefill chunk boundaries and in-chunk logits positions.

    Splits `n` tokens into chunks of up to `chunk` tokens, cutting at
    `checkpoint_len` and prompt end (`prompt_tokens`).

    Returns list of (step_size, in_chunk_logits_positions).
    """
    segments: list[tuple[int, list[int]]] = []
    if n <= 0:
        return segments

    targets: set[int]
    if target_logits_positions is None:
        targets = set()
        if checkpoint_len is not None and 0 < checkpoint_len <= prompt_tokens:
            targets.add(checkpoint_len - 1)
        if prompt_tokens > 0:
            targets.add(prompt_tokens - 1)
    else:
        targets = set(target_logits_positions)

    effective_chunk = max(1, chunk)
    cp_len = checkpoint_len if checkpoint_len is not None else 0
    p_idx = 0
    while p_idx < n:
        cur_cached = cached + p_idx
        step_size = min(effective_chunk, n - p_idx)
        if cur_cached < cp_len:
            rem = cp_len - cur_cached
            if rem > 0:
                step_size = min(step_size, rem)

        chunk_logits = [
            t for t in range(step_size)
            if (cur_cached + t) in targets
        ]
        segments.append((step_size, chunk_logits))
        p_idx += step_size

    return segments
