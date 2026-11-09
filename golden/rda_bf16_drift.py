#!/usr/bin/env python3
# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Compare Raon RDA fp32 state with per-token BF16-rounded state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F


def _decode_array(array: np.ndarray, dtype: str) -> torch.Tensor:
    value = torch.from_numpy(np.array(array, copy=True))
    if dtype == "bfloat16":
        return value.view(torch.bfloat16)
    torch_dtype = getattr(torch, dtype, None)
    return value.to(torch_dtype) if torch_dtype is not None else value


def load_capture(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        dtypes = json.loads(str(archive["__dtypes_json__"].item()))
        return {
            name: _decode_array(archive[name], dtypes[name])
            for name in ("q", "k", "v", "log_alpha", "beta")
        }


def synthetic_inputs(tokens: int, heads: int, dim: int, seed: int) -> dict[str, torch.Tensor]:
    """Generate inputs at the same rounding boundary as Raon1MiniRDA._core."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, tokens, heads, dim, generator=generator)
    k = torch.randn(1, tokens, heads, dim, generator=generator)
    v = torch.randn(1, tokens, heads, dim, generator=generator)
    # modeling_raon1_mini.py: q/k normalized in fp32 then q/k/v rounded to activation dtype.
    q = F.normalize(q, dim=-1, eps=1e-6).to(torch.bfloat16) * (dim ** -0.5)
    k = F.normalize(k, dim=-1, eps=1e-6).to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    raw_decay = torch.randn(1, tokens, heads, dim, generator=generator) * 0.25
    log_alpha = -5.0 * torch.sigmoid(raw_decay)  # remains fp32 in the original path
    beta_raw = torch.randn(1, tokens, heads, generator=generator)
    beta = torch.sigmoid(beta_raw).to(torch.bfloat16)
    return {"q": q, "k": k, "v": v, "log_alpha": log_alpha, "beta": beta}


def checkpoint_positions(spec: str, tokens: int) -> list[int]:
    values: set[int] = {tokens}
    suffixes = {"k": 1024, "m": 1024 * 1024}
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        scale = suffixes.get(item[-1], 1)
        number = item[:-1] if scale != 1 else item
        values.add(int(float(number) * scale))
    return sorted(value for value in values if 1 <= value <= tokens)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f, right_f = left.float().flatten(), right.float().flatten()
    denom = torch.linalg.vector_norm(left_f) * torch.linalg.vector_norm(right_f)
    if denom.item() == 0:
        return 1.0 if torch.equal(left_f, right_f) else 0.0
    return float(torch.dot(left_f, right_f).div(denom).item())


def compare(inputs: dict[str, torch.Tensor], positions: list[int]) -> list[dict[str, Any]]:
    output_dtype = inputs["v"].dtype
    q, k, v = (inputs[name].float() for name in ("q", "k", "v"))
    log_alpha = inputs["log_alpha"].float()
    beta = inputs["beta"].float()
    batch, tokens, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    reference = torch.zeros(batch, heads, value_dim, key_dim, dtype=torch.float32)
    rounded = torch.zeros_like(reference)
    wanted = set(positions)
    records: list[dict[str, Any]] = []
    for idx in range(tokens):
        decay = torch.exp(log_alpha[:, idx]).unsqueeze(-2)
        kt = k[:, idx]
        qt = q[:, idx]
        vt = v[:, idx]
        bt = beta[:, idx][..., None, None]

        reference = reference * decay
        prediction = torch.einsum("bhvk,bhk->bhv", reference, kt)
        reference = reference + bt * torch.einsum("bhv,bhk->bhvk", vt - prediction, kt)
        output_reference = torch.einsum("bhvk,bhk->bhv", reference, qt)

        rounded = rounded * decay
        prediction_rounded = torch.einsum("bhvk,bhk->bhv", rounded, kt)
        rounded = rounded + bt * torch.einsum("bhv,bhk->bhvk", vt - prediction_rounded, kt)
        # This is the experiment: state storage rounds after every token update.
        rounded = rounded.to(torch.bfloat16).float()
        output_rounded = torch.einsum("bhvk,bhk->bhv", rounded, qt)

        position = idx + 1
        if position in wanted:
            # torch_recurrent_rda returns outputs rounded to v.dtype, while its
            # recurrent state remains fp32.
            output_reference = output_reference.to(output_dtype).float()
            output_rounded = output_rounded.to(output_dtype).float()
            output_delta = output_rounded - output_reference
            state_delta = rounded - reference
            output_norm = torch.linalg.vector_norm(output_reference)
            state_rms = torch.sqrt(torch.mean(reference.square()))
            records.append({
                "token_position": position,
                "output_relative_error": float(
                    torch.linalg.vector_norm(output_delta).div(output_norm.clamp_min(1e-30)).item()
                ),
                "output_cosine": _cosine(output_reference, output_rounded),
                "state_nrmse": float(
                    torch.sqrt(torch.mean(state_delta.square())).div(state_rms.clamp_min(1e-30)).item()
                ),
                "reference_state_norm": float(torch.linalg.vector_norm(reference).item()),
                "rounded_state_norm": float(torch.linalg.vector_norm(rounded).item()),
            })
    return records


def input_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("rda_inputs_layer_*.npz"))
    else:
        raise FileNotFoundError(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    results: list[dict[str, Any]] = []
    if args.input:
        files = list(input_files(Path(args.input)))
        if not files:
            raise ValueError(f"no rda_inputs_layer_*.npz files under {args.input}")
        sources = ((path.name, load_capture(path)) for path in files)
        mode = "capture"
    else:
        sources = iter([("synthetic", synthetic_inputs(args.tokens, args.heads, args.dim, args.seed))])
        mode = "synthetic"
    for name, inputs in sources:
        tokens = int(inputs["q"].shape[1])
        positions = checkpoint_positions(args.positions, tokens)
        results.append({
            "source": name,
            "tokens": tokens,
            "heads": int(inputs["q"].shape[2]),
            "key_dim": int(inputs["q"].shape[3]),
            "input_dtypes": {key: str(value.dtype).removeprefix("torch.") for key, value in inputs.items()},
            "metrics": compare(inputs, positions),
        })
    report = {
        "format_version": 1,
        "mode": mode,
        "experiment": "state rounded to bfloat16 after every token update",
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(rendered, end="")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="captured RDA .npz or dump directory")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tokens", type=int, default=32768, help="synthetic token count")
    parser.add_argument("--heads", type=int, default=4, help="synthetic head count")
    parser.add_argument("--dim", type=int, default=32, help="synthetic key/value dimension")
    parser.add_argument("--positions", default="128,512,2k,8k,32k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tokens <= 0 or args.heads <= 0 or args.dim <= 0:
        raise SystemExit("--tokens, --heads, and --dim must be > 0")
    run(args)


if __name__ == "__main__":
    main()
