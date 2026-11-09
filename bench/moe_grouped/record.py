# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Record real routed-MoE calls while the unchanged engine performs prefill."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from host.strict_json import loads_strict
from serve.engine import Engine
from serve.tokenize import RaonTokenizer
from .data import Record, save_record, write_manifest


def parse_layers(value: str, count: int) -> set[int]:
    if value.strip().lower() == "all":
        return set(range(count))
    result: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(piece) for piece in item.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"layers must be within 0..{count - 1}")
    return result


def load_tokens(args: argparse.Namespace) -> list[int]:
    if args.tokens_file is not None:
        raw = loads_strict(args.tokens_file.read_text(encoding="utf-8"))
        values = raw.get("tokens") if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError("tokens file must contain a list or {'tokens': [...]} object")
        return [int(value) for value in values]
    tokenizer = RaonTokenizer(args.model_dir or args.weights_dir)
    return [int(value) for value in tokenizer.encode(args.prompt)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record real MoE prefill calls for grouped replay")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokens-file", type=Path)
    source.add_argument("--prompt")
    parser.add_argument("--layers", default="all", help="all, comma list, or ranges")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")

    tokens = load_tokens(args)
    if not tokens:
        parser.error("the input produced no tokens")
    engine = Engine(args.weights_dir, args.model_dir, preload=True)
    model = engine.model
    backend = engine.backend
    if model is None or backend is None:
        raise RuntimeError("Engine did not build a model/backend")
    layers = parse_layers(args.layers, int(model.config.num_hidden_layers))
    args.out.mkdir(parents=True, exist_ok=True)
    counters: defaultdict[int, int] = defaultdict(int)
    original = backend.moe_experts

    def wrapped(
        self: Any,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
    ) -> np.ndarray:
        output = original(
            layer, x, top_ids, top_weights,
            gate_beta=gate_beta, value_beta=value_beta,
        )
        token = counters[layer]
        counters[layer] += 1
        if layer not in layers:
            return output
        bundle = self._bundle_cache.get(f"L{layer:02d}/shared_gu")
        cached = self._cached_shared.get(layer)
        if bundle is None or cached is None:
            raise RuntimeError(f"layer {layer}: shared expert caches missing after moe_experts")
        shared_gate_up = np.asarray(bundle[2]).view(np.uint16).reshape(-1).copy()
        shared_down = np.asarray(cached["sh_down"], dtype=np.uint16).reshape(-1).copy()
        save_record(
            args.out,
            Record(
                layer, token, np.asarray(x, dtype=np.uint16).reshape(-1).copy(),
                np.asarray(top_ids, dtype=np.int64).reshape(-1).copy(),
                np.asarray(top_weights, dtype=np.float32).reshape(-1).copy(),
                shared_gate_up, np.asarray(output, dtype=np.uint16).reshape(-1).copy(),
                shared_down, float(gate_beta), float(value_beta),
            ),
        )
        return output

    backend.moe_experts = MethodType(wrapped, backend)
    try:
        model.reset_state()
        backend.reset()
        for start in range(0, len(tokens), args.chunk):
            token_chunk = tokens[start : start + args.chunk]
            if hasattr(backend, "set_step"):
                backend.set_step(start + 1)
            logits = {len(token_chunk) - 1} if start + len(token_chunk) == len(tokens) else set()
            model.prefill(token_chunk, logits_positions=logits, chunk=len(token_chunk))
    finally:
        backend.moe_experts = original
        engine.unload()
    counts = {str(layer): counters[layer] for layer in sorted(layers)}
    write_manifest(
        args.out,
        {"tokens": len(tokens), "layers": sorted(layers), "records_by_layer": counts},
    )
    print(f"recorded {sum(counts.values())} layer-token calls in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
