#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors
"""End-to-end Raon quality comparison for original and replaced BF16 weights.

Exactly one CPU model is resident.  The original metrics are collected first,
then eligible matrices are replaced in place, one tensor at a time, and the
same inputs are evaluated again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golden.dump import load_model  # noqa: E402
from quant.gguf_q4k import GGUFReader, GGUFWeightReader  # noqa: E402
from quant.q4nx import dequantize_simple, quantize  # noqa: E402

MODES = (
    "orig",
    "minmax",
    "gguf_q4km",
    "gguf_q4nx",
    "mixed_full",
    "mixed_lite",
    "mixed_attn",
)

from quant.bfp16 import (
    BLOCK,
    BLOCK_BYTES,
    _BLOCK_DTYPE,
    _MANTISSA_SHIFT,
    bfp16ebs8_decode,
    bfp16ebs8_encode,
    quantize_bfp16ebs8,
    round_to_bfp16ebs8,
)

_ATTN_PROJECTIONS = (
    "q_proj|k_proj|v_proj|g_proj|o_proj|f_a_proj|f_b_proj|b_proj|rda_w_proj|"
    "q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj"
)
_TARGET_PATTERNS = (
    re.compile(rf"^model\.layers\.\d+\.self_attn\.({_ATTN_PROJECTIONS})\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$"),
    re.compile(
        r"^model\.layers\.\d+\.moe\.(experts\.\d+|shared_experts)\."
        r"(gate|up|down)_proj\.weight$"
    ),
    re.compile(r"^lm_head\.weight$"),
)


class GGUFSource(Protocol):
    def matrix(self, name: str) -> np.ndarray: ...
    def q4_restore(self, name: str) -> np.ndarray: ...


def is_quantized_weight_name(name: str) -> bool:
    """True only for matrices read as 4-bit weights by the NPU design."""
    return any(pattern.fullmatch(name) for pattern in _TARGET_PATTERNS)


def is_mixed_lite_bfp_target(name: str) -> bool:
    """True for expert down (ffn_down_exps), shared expert down (ffn_down_shexp), and output (output.weight)."""
    if name == "lm_head.weight":
        return True
    if ".moe.shared_experts.down_proj.weight" in name:
        return True
    if ".moe.experts." in name and ".down_proj.weight" in name:
        return True
    return False


def is_mixed_attn_bfp_target(name: str) -> bool:
    """True for attention projections (``model.layers.N.self_attn.*``).

    ``mixed_attn`` replaces these with bfp16ebs8, but only when the GGUF tensor is
    not natively Q4_K; natively Q4_K tensors keep their preserved 4-bit codes.
    """
    return ".self_attn." in name


def is_gguf_q4k(gguf: Any, name: str) -> bool:
    """True if the GGUF tensor backing `name` is natively Q4_K."""
    if hasattr(gguf, "is_q4_k"):
        return bool(gguf.is_q4_k(name))
    if hasattr(gguf, "source"):
        try:
            _, types = gguf.source(name)
            return types == ["Q4_K"]
        except Exception:
            return False
    return False


def _minmax_restore(weight: torch.Tensor) -> np.ndarray:
    source = weight.detach().float().cpu().numpy()
    codes, scales, mins = quantize(source)
    return dequantize_simple(codes, scales, mins)


def replace_target_weights(
    model: torch.nn.Module,
    mode: str,
    gguf: GGUFSource | None = None,
    *,
    bfp_rounding: str = "conv_even",
) -> dict[str, Any]:
    """Replace eligible matrices in place and return an auditable replacement manifest.

    ``bfp_rounding`` selects the rounding used by ``quantize_bfp16ebs8`` for the
    mixed modes; it is offline pre-quantization we control, so the default is
    round-to-nearest-even ("conv_even") instead of the hardware-default "floor".
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if (mode.startswith("gguf_") or mode.startswith("mixed_")) and gguf is None:
        raise ValueError(f"{mode} requires a GGUF source")
    changed: list[str] = []
    families: Counter[str] = Counter()
    quant_types: Counter[str] = Counter()
    if mode == "orig":
        return {"count": 0, "names": [], "families": {}, "quant_types": {}}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not is_quantized_weight_name(name):
                continue
            if parameter.ndim != 2:
                raise ValueError(f"target matrix is not 2D: {name} {tuple(parameter.shape)}")
            if mode == "minmax":
                restored = _minmax_restore(parameter)
                qtype = "q4nx_minmax"
            elif mode == "gguf_q4km":
                assert gguf is not None
                restored = gguf.matrix(name)
                qtype = "gguf_q4km"
            elif mode == "gguf_q4nx":
                assert gguf is not None
                restored = gguf.q4_restore(name)
                qtype = "q4nx"
            elif mode == "mixed_full":
                assert gguf is not None
                if is_gguf_q4k(gguf, name):
                    restored = gguf.q4_restore(name)
                    qtype = "q4nx"
                else:
                    matrix_f32 = gguf.matrix(name)
                    restored = quantize_bfp16ebs8(matrix_f32, rounding=bfp_rounding)
                    del matrix_f32
                    qtype = "bfp16ebs8"
            elif mode == "mixed_lite":
                assert gguf is not None
                if is_mixed_lite_bfp_target(name):
                    matrix_f32 = gguf.matrix(name)
                    restored = quantize_bfp16ebs8(matrix_f32, rounding=bfp_rounding)
                    del matrix_f32
                    qtype = "bfp16ebs8"
                else:
                    restored = gguf.q4_restore(name)
                    qtype = "q4nx"
            elif mode == "mixed_attn":
                assert gguf is not None
                # Attention projections only: a non-Q4_K attention projection gets
                # bfp16ebs8 (same treatment as mixed_full); everything else, Q4_K or
                # not, keeps the 4-bit q4nx path (same as gguf_q4nx).
                if is_mixed_attn_bfp_target(name) and not is_gguf_q4k(gguf, name):
                    matrix_f32 = gguf.matrix(name)
                    restored = quantize_bfp16ebs8(matrix_f32, rounding=bfp_rounding)
                    del matrix_f32
                    qtype = "bfp16ebs8"
                else:
                    restored = gguf.q4_restore(name)
                    qtype = "q4nx"
            else:
                assert gguf is not None
                restored = gguf.q4_restore(name)
                qtype = "q4nx"
            if tuple(restored.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"replacement shape mismatch for {name}: {restored.shape} != {tuple(parameter.shape)}"
                )
            parameter.copy_(torch.from_numpy(np.ascontiguousarray(restored)).to(parameter.dtype))
            changed.append(name)
            quant_types[qtype] += 1
            if ".self_attn." in name:
                family = "attention"
            elif ".moe.experts." in name:
                family = "routed_expert"
            elif ".moe.shared_experts." in name:
                family = "shared_expert"
            elif ".mlp." in name:
                family = "dense_mlp"
            else:
                family = "lm_head"
            families[family] += 1
            del restored
    return {
        "count": len(changed),
        "names": changed,
        "families": dict(sorted(families.items())),
        "quant_types": dict(sorted(quant_types.items())),
    }


def estimate_npu_read_bytes_per_token(
    model: torch.nn.Module,
    mode: str,
    gguf: Any | None = None,
) -> dict[str, Any]:
    """Estimate bytes read per token by NPU across target matrices.

    Routing: routed experts (.moe.experts.) have 6/128 activation ratio.
    Other target matrices are 1.0.
    Bit allocation per mode:
      - orig: 16 bits (BF16)
      - minmax, gguf_q4nx: 5 bits per weight (q4nx packed: 4-bit codes + 1-bit scale/min)
        [also records 4-bit nominal code count]
      - gguf_q4km: 5 bits per weight
      - mixed_full: Q4_K -> 5 bits (q4nx), non-Q4_K -> 9 bits (bfp16ebs8)
      - mixed_lite: down_proj / lm_head -> 9 bits (bfp16ebs8), others -> 5 bits (q4nx)
      - mixed_attn: non-Q4_K self_attn -> 9 bits (bfp16ebs8), everything else -> 5 bits
        (q4nx: Q4_K code preservation and non-Q4_K min-max q4nx alike)
    """
    total_elements = 0
    effective_elements = 0.0
    total_bytes_5bit = 0.0
    total_bytes_4bit = 0.0
    breakdown_by_family: dict[str, float] = Counter()

    for name, parameter in model.named_parameters():
        if not is_quantized_weight_name(name):
            continue
        numel = parameter.numel()
        is_expert = ".moe.experts." in name
        ratio = (6.0 / 128.0) if is_expert else 1.0
        eff_numel = numel * ratio
        total_elements += numel
        effective_elements += eff_numel

        if mode == "orig":
            bits_packed = 16.0
            bits_code = 16.0
        elif mode in ("minmax", "gguf_q4nx"):
            bits_packed = 5.0
            bits_code = 4.0
        elif mode == "mixed_full":
            if gguf is not None and is_gguf_q4k(gguf, name):
                bits_packed = 5.0
                bits_code = 4.0
            else:
                bits_packed = 9.0
                bits_code = 9.0
        elif mode == "mixed_lite":
            if is_mixed_lite_bfp_target(name):
                bits_packed = 9.0
                bits_code = 9.0
            else:
                bits_packed = 5.0
                bits_code = 4.0
        elif mode == "mixed_attn":
            if (
                gguf is not None
                and is_mixed_attn_bfp_target(name)
                and not is_gguf_q4k(gguf, name)
            ):
                bits_packed = 9.0
                bits_code = 9.0
            else:
                bits_packed = 5.0
                bits_code = 4.0
        elif mode == "gguf_q4km":
            bits_packed = 5.0
            bits_code = 4.0
        else:
            bits_packed = 5.0
            bits_code = 4.0

        b_packed = eff_numel * (bits_packed / 8.0)
        b_code = eff_numel * (bits_code / 8.0)
        total_bytes_5bit += b_packed
        total_bytes_4bit += b_code

        if ".self_attn." in name:
            fam = "attention"
        elif ".moe.experts." in name:
            fam = "routed_expert"
        elif ".moe.shared_experts." in name:
            fam = "shared_expert"
        elif ".mlp." in name:
            fam = "dense_mlp"
        else:
            fam = "lm_head"
        breakdown_by_family[fam] += b_packed

    return {
        "bytes_per_token": int(round(total_bytes_5bit)),
        "bytes_per_token_4bit_nominal": int(round(total_bytes_4bit)),
        "gib_per_token": total_bytes_5bit / (1024.0**3),
        "total_target_elements": total_elements,
        "effective_target_elements_per_token": int(round(effective_elements)),
        "breakdown_by_family_bytes": dict(sorted(breakdown_by_family.items())),
    }


def load_text_chunks(tokenizer: Any, path: Path, chunk_len: int, chunks: int) -> list[torch.Tensor]:
    text = path.read_text(encoding="utf-8")
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    result = []
    for start in range(0, max(0, len(token_ids) - 1), chunk_len):
        piece = token_ids[start : start + chunk_len]
        if piece.numel() >= 2:
            result.append(piece.unsqueeze(0).long())
        if len(result) == chunks:
            break
    if not result:
        raise ValueError(f"{path} does not contain at least two tokenizer tokens")
    return result


def load_prompts(tokenizer: Any, path: Path) -> list[torch.Tensor]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [line for line in raw.splitlines() if line.strip()]
    if not isinstance(payload, list) or not payload:
        raise ValueError("--prompts must be a non-empty JSON list or a non-empty line-delimited file")
    result = []
    for item in payload:
        if isinstance(item, dict) and "input_ids" in item:
            ids = torch.tensor([item["input_ids"]], dtype=torch.long)
        else:
            ids = tokenizer(str(item), add_special_tokens=True, return_tensors="pt")["input_ids"].long()
        if ids.shape[1] == 0:
            raise ValueError("empty prompt after tokenization")
        result.append(ids)
    return result


def _forward(model: torch.nn.Module, ids: torch.Tensor, **kwargs: Any) -> Any:
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True, **kwargs)


def greedy_tokens(model: torch.nn.Module, prompt: torch.Tensor, length: int) -> tuple[list[int], np.ndarray]:
    generated: list[int] = []
    cache = None
    current = prompt
    final_logits: np.ndarray | None = None
    attention_length = prompt.shape[1]
    for step in range(length):
        attention = torch.ones((1, attention_length), dtype=torch.long)
        output = model(
            input_ids=current,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        logits = output.logits[:, -1, :].float()
        token = int(torch.argmax(logits, dim=-1).item())
        generated.append(token)
        final_logits = logits[0].cpu().numpy().copy()
        cache = output.past_key_values
        current = torch.tensor([[token]], dtype=torch.long)
        attention_length += 1
    if final_logits is None:
        output = _forward(model, prompt, use_cache=False, logits_to_keep=1)
        final_logits = output.logits[0, -1].float().cpu().numpy().copy()
    return generated, final_logits


def evaluate_once(
    model: torch.nn.Module,
    text_chunks: Sequence[torch.Tensor],
    prompts: Sequence[torch.Tensor],
    gen_len: int,
) -> tuple[dict[str, Any], list[np.ndarray], list[list[int]]]:
    total_nll = 0.0
    predicted = 0
    final_logits: list[np.ndarray] = []
    greedy: list[list[int]] = []
    with torch.inference_mode():
        for ids in text_chunks:
            output = _forward(model, ids, use_cache=False)
            logits = output.logits[:, :-1, :].float()
            labels = ids[:, 1:]
            total_nll += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"))
            predicted += labels.numel()
            final_logits.append(output.logits[0, -1].float().cpu().numpy().copy())
            del output, logits
        for prompt in prompts:
            tokens, logits = greedy_tokens(model, prompt, gen_len)
            greedy.append(tokens)
            final_logits.append(logits)
    mean_nll = total_nll / predicted
    return (
        {"nll": mean_nll, "perplexity": math.exp(mean_nll), "predicted_tokens": predicted},
        final_logits,
        greedy,
    )


def compare_logits(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("logit sample count mismatch")
    kls, top1 = [], []
    for ref, got in zip(reference, candidate):
        p = torch.softmax(torch.from_numpy(ref), dim=-1)
        log_p = torch.log_softmax(torch.from_numpy(ref), dim=-1)
        log_q = torch.log_softmax(torch.from_numpy(got), dim=-1)
        kls.append(float(torch.sum(p * (log_p - log_q))))
        top1.append(int(np.argmax(ref) == np.argmax(got)))
    return {
        "samples": len(kls),
        "kl_mean": float(np.mean(kls)),
        "kl_max": float(np.max(kls)),
        "top1_match_rate": float(np.mean(top1)),
    }


def compare_greedy(reference: Sequence[Sequence[int]], candidate: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    rows = []
    for idx, (ref, got) in enumerate(zip(reference, candidate)):
        mismatches = [position for position, pair in enumerate(zip(ref, got)) if pair[0] != pair[1]]
        row: dict[str, Any] = {
            "prompt_index": idx,
            "first_mismatch": mismatches[0] if mismatches else None,
            "generated_tokens": min(len(ref), len(got)),
        }
        for limit in (64, 256):
            count = min(limit, len(ref), len(got))
            row[f"first_{limit}_match_rate"] = (
                float(sum(a == b for a, b in zip(ref[:count], got[:count])) / count) if count else 1.0
            )
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.chunk_len < 2 or args.chunks < 1 or args.gen_len < 0:
        raise ValueError("--chunk-len >= 2, --chunks >= 1 and --gen-len >= 0 are required")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    text_chunks = load_text_chunks(tokenizer, args.text, args.chunk_len, args.chunks)
    prompts = load_prompts(tokenizer, args.prompts)
    started = time.perf_counter()
    model, config = load_model(args.model_dir, args.code_dir or args.model_dir)
    model.eval()
    baseline, baseline_logits, baseline_greedy = evaluate_once(model, text_chunks, prompts, args.gen_len)

    gguf = None
    if args.mode.startswith("gguf_") or args.mode.startswith("mixed_"):
        reader = GGUFReader(args.gguf)
        one_based = config.rda_config.get("rda_layers", [])
        gguf = GGUFWeightReader(reader, {int(layer) - 1 for layer in one_based})
    replacements = replace_target_weights(
        model, args.mode, gguf, bfp_rounding=args.bfp_rounding
    )
    npu_read = estimate_npu_read_bytes_per_token(model, args.mode, gguf)
    replacements["npu_read_bytes_per_token"] = npu_read["bytes_per_token"]
    selected, selected_logits, selected_greedy = evaluate_once(model, text_chunks, prompts, args.gen_len)
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "bfp_rounding": args.bfp_rounding,
        "inputs": {
            "model_dir": str(args.model_dir),
            "gguf": (
                str(args.gguf)
                if (args.mode.startswith("gguf_") or args.mode.startswith("mixed_"))
                else None
            ),
            "text": str(args.text),
            "prompts": str(args.prompts),
            "chunk_len": args.chunk_len,
            "chunks_requested": args.chunks,
            "chunks_used": len(text_chunks),
            "gen_len": args.gen_len,
            "threads": args.threads,
        },
        "original": baseline,
        "selected": selected,
        "perplexity_ratio": selected["perplexity"] / baseline["perplexity"],
        "npu_read_bytes_per_token": npu_read["bytes_per_token"],
        "npu_read_bytes_per_token_details": npu_read,
        "final_logits": compare_logits(baseline_logits, selected_logits),
        "greedy": compare_greedy(baseline_greedy, selected_greedy),
        "replacements": replacements,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--bfp-rounding",
        choices=("conv_even", "floor"),
        default="conv_even",
        help="rounding for quantize_bfp16ebs8 in the mixed modes (default: conv_even)",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.threads <= 64:
        raise ValueError("--threads must be between 1 and 64")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "mode": args.mode, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
