# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for prefill replay comparison harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from host.strict_json import loads_strict
from serve.engine import Engine
from serve.tokenize import RaonTokenizer
from .harness import measure_timing_repeats, run_prefill_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefill byte-level comparison replay harness")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Path to weights directory")
    parser.add_argument("--model-dir", type=Path, default=None, help="Path to model directory")
    parser.add_argument("--tokens-file", type=Path, default=None, help="JSON file containing token IDs list")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string to tokenize")
    parser.add_argument("--chunk", type=int, default=16, help="Prefill chunk size (default: 16)")
    parser.add_argument("--checkpoint-len", type=int, default=None, help="Checkpoint token length boundary")
    parser.add_argument("--logits-positions", type=str, default=None, help="Comma-separated logits positions")
    parser.add_argument("--timing-repeats", type=int, default=0, help="Number of timing repeat pairs (default: 0)")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (prints to stdout if omitted)")

    args = parser.parse_args()

    # Load tokens
    token_ids: list[int] = []
    if args.tokens_file is not None:
        data = loads_strict(args.tokens_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            token_ids = [int(x) for x in data]
        elif isinstance(data, dict) and "tokens" in data:
            token_ids = [int(x) for x in data["tokens"]]
        else:
            raise ValueError(f"Unexpected format in tokens file {args.tokens_file}")
    elif args.prompt is not None:
        tokenizer_dir = args.model_dir or args.weights_dir
        tokenizer = RaonTokenizer(tokenizer_dir)
        token_ids = [int(x) for x in tokenizer.encode(args.prompt)]
    else:
        # Default fallback synthetic tokens if none provided
        token_ids = [42, 161, 88, 204, 512, 1024, 2048, 4096]

    logits_positions = None
    if args.logits_positions is not None:
        logits_positions = [int(x.strip()) for x in args.logits_positions.split(",") if x.strip()]

    engine = Engine(args.weights_dir, args.model_dir, preload=True)
    if engine.model is None:
        raise RuntimeError("Failed to build model via Engine")
    model = engine.model

    # 1. Byte-level comparison replay (1 run)
    result = run_prefill_replay(
        model,
        token_ids,
        chunk=args.chunk,
        checkpoint_len=args.checkpoint_len,
        logits_positions=logits_positions,
    )

    # 2. Timing repeats if requested
    if args.timing_repeats > 0:
        timing_res = measure_timing_repeats(
            model,
            token_ids,
            args.timing_repeats,
            chunk=args.chunk,
            checkpoint_len=args.checkpoint_len,
            logits_positions=logits_positions,
        )
        result["timing_repeats"] = timing_res["pairs"]
        result["timing_summary"] = timing_res["summary"]

        for p in timing_res["pairs"]:
            tag = f"준비 실행 (쌍 {p['pair']})" if p["warmup"] else f"반복 (쌍 {p['pair']})"
            print(
                f"[{tag}] advance: {p['ms_per_token_a']:.3f} ms/tok, "
                f"prefill: {p['ms_per_token_b']:.3f} ms/tok "
                f"(speedup: {p['speedup']:.2f}x)",
                file=sys.stderr,
            )
        sm = timing_res["summary"]
        if sm and sm.get("count", 0) > 0:
            print(
                f"[요약 (중앙값 {sm['count']}쌍)] advance: {sm['median_ms_per_token_a']:.3f} ms/tok, "
                f"prefill: {sm['median_ms_per_token_b']:.3f} ms/tok "
                f"(speedup: {sm['median_speedup']:.2f}x)",
                file=sys.stderr,
            )

    out_json = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.write_text(out_json, encoding="utf-8")
    else:
        print(out_json)

    return 0 if result["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
