#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors
"""Evaluate q4nx against the released Raon-1-mini weights and golden activations.

The evaluator deliberately loads one safetensors tensor (or one MXFP8 weight/scale
pair) at a time.  It never constructs the full model.  Results are written below
``quant/eval_out`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant.q4nx import dequantize_simple, quantize  # noqa: E402
from quant.gguf_q4k import GGUFReader, GGUFWeightReader  # noqa: E402

THRESHOLD = 0.995
EXPERT_SEED = 20261028
EXPERT_COUNT = 4
LM_CHUNK_ROWS = 4096


def bf16_bits_to_f32(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom else float(av.size == 0 or np.array_equal(av, bv))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64)
    denom = np.linalg.norm(av.reshape(-1))
    return float(np.linalg.norm((av - np.asarray(b, dtype=np.float64)).reshape(-1)) / denom) if denom else 0.0


def tensor_metrics(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(restored, dtype=np.float32)
    dot = np.einsum("ij,ij->i", a.astype(np.float64), b.astype(np.float64))
    an = np.linalg.norm(a.astype(np.float64), axis=1)
    bn = np.linalg.norm(b.astype(np.float64), axis=1)
    row = np.divide(dot, an * bn, out=np.ones_like(dot), where=(an * bn) != 0)
    return {
        "cosine": cosine(a, b),
        "relative_frobenius": relative_l2(a, b),
        "max_abs_error": float(np.max(np.abs(a - b))),
        "worst_output_channel_cosine": float(np.min(row)),
        "worst_output_channel": int(np.argmin(row)),
    }


class WeightReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    def _tensor(self, name: str, row_slice: slice | None = None) -> torch.Tensor:
        with safe_open(self.model_dir / self.weight_map[name], framework="pt", device="cpu") as handle:
            if row_slice is None:
                return handle.get_tensor(name)
            value = handle.get_slice(name)
            return value[row_slice]

    def shape_dtype(self, name: str) -> tuple[tuple[int, ...], str]:
        with safe_open(self.model_dir / self.weight_map[name], framework="pt", device="cpu") as handle:
            value = handle.get_slice(name)
            return tuple(value.get_shape()), str(value.get_dtype())

    def matrix(self, name: str, row_slice: slice | None = None) -> np.ndarray:
        weight = self._tensor(name, row_slice)
        if weight.dtype == torch.bfloat16:
            return weight.float().numpy()
        if weight.dtype != torch.float8_e4m3fn:
            return weight.float().numpy()
        scale_name = name + "_scale_inv"
        scale = self._tensor(scale_name, row_slice)
        rows, cols = weight.shape
        blocks = weight.float().reshape(rows, cols // 32, 32)
        raw_scale = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
        factor = torch.exp2(raw_scale.float() - 127.0).unsqueeze(-1)
        return (blocks * factor).reshape(rows, cols).to(torch.bfloat16).float().numpy()

    def vector(self, name: str) -> np.ndarray:
        return self._tensor(name).float().numpy()


def q4_restore(weight: np.ndarray) -> np.ndarray:
    codes, scales, mins = quantize(weight)
    return dequantize_simple(codes, scales, mins)


def bf16_round(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(
        torch.bfloat16
    ).float().numpy()


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Reproduce a model-dtype BF16 linear and expose its rounded output as FP32."""
    hidden = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16)
    matrix = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32)).to(torch.bfloat16)
    return torch.nn.functional.linear(hidden, matrix).float().numpy()


def linear_fp32(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Router-only FP32 linear, matching ``Raon1MiniRouter.forward``."""
    hidden = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    matrix = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32))
    return torch.nn.functional.linear(hidden, matrix).numpy()


def functional_metrics(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    return {"cosine": cosine(original, restored), "relative_l2": relative_l2(original, restored)}


def situ_glu(gate: np.ndarray, value: np.ndarray, gate_beta: float, value_beta: float) -> np.ndarray:
    g = torch.from_numpy(np.ascontiguousarray(gate, dtype=np.float32))
    v = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    return ((gate_beta * torch.tanh(g / gate_beta) * torch.sigmoid(g)) *
            (value_beta * torch.tanh(v / value_beta))).numpy()


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    value = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    gamma = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32))
    return (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps) * gamma).numpy()


def load_activations(golden_dir: Path, layers: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
    wanted = set(layers)
    parts: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {key: [] for key in ("streams_in", "mixer_in", "mixer_out", "streams_out")}
        for layer in wanted
    }
    for path in sorted(golden_dir.glob("step_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for layer in wanted:
                prefix = f"layer_{layer:03d}."
                for key in parts[layer]:
                    value = data[prefix + key]
                    if value.dtype == np.uint16:
                        value = bf16_bits_to_f32(value)
                    parts[layer][key].append(value.reshape(-1, *value.shape[2:]))
    return {
        layer: {key: np.concatenate(values, axis=0) for key, values in fields.items()}
        for layer, fields in parts.items()
    }


def helmert_basis(n: int) -> np.ndarray:
    q = np.zeros((n, n - 1), dtype=np.float32)
    for k in range(1, n):
        q[:k, k - 1] = 1.0
        q[k, k - 1] = -float(k)
        q[:, k - 1] /= math.sqrt(k * (k + 1.0))
    return q


def mhc_route(reader: WeightReader, prefix: str, streams: np.ndarray, eps: float,
              post_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = streams.shape[1]
    flat = torch.from_numpy(
        np.ascontiguousarray(streams.reshape(streams.shape[0], -1), dtype=np.float32)
    ).to(torch.bfloat16)
    names = ["phi_pre", "phi_post", "res.phi_z"]
    weight = torch.from_numpy(np.ascontiguousarray(
        np.concatenate([reader.vector(prefix + name) for name in names], axis=0),
        dtype=np.float32,
    )).to(torch.bfloat16)
    inv = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + eps)
    z = torch.nn.functional.linear(flat, weight).float() * inv
    alpha_pre = torch.tensor(float(reader.vector(prefix + "alpha_pre")), dtype=torch.float32)
    alpha_post = torch.tensor(float(reader.vector(prefix + "alpha_post")), dtype=torch.float32)
    alpha_z = torch.tensor(float(reader.vector(prefix + "res.alpha_z")), dtype=torch.float32)
    b_pre = torch.from_numpy(reader.vector(prefix + "b_pre")).float()
    b_post = torch.from_numpy(reader.vector(prefix + "b_post")).float()
    b_z = torch.from_numpy(reader.vector(prefix + "res.b_z")).float()
    h_pre = torch.sigmoid(alpha_pre * z[:, :n] + b_pre)
    h_post = post_factor * torch.sigmoid(alpha_post * z[:, n:2*n] + b_post)
    t_z = alpha_z * z[:, 2*n:] + b_z
    q = torch.from_numpy(helmert_basis(n))
    e = torch.einsum("im,bmk,jk->bij", q, t_z.reshape(-1, n - 1, n - 1), q)
    eye = torch.eye(n, dtype=torch.float32)
    chart_b = 0.5 * (eye + torch.full_like(eye, 1.0 / n))
    rho = (-e / chart_b).amax(dim=(-2, -1)).clamp_min(0.0)
    phi = chart_b + e / (1.0 + rho)[:, None, None]
    eta = torch.tensor(float(reader.vector(prefix + "res.eta")), dtype=torch.float32)
    gamma = torch.sigmoid(eta)
    h_res = (1.0 - gamma) * eye + gamma * phi
    return h_pre.numpy(), h_post.numpy(), h_res.numpy()


def reconstruct_mlp_input(reader: WeightReader, layer: int, act: dict[str, np.ndarray],
                          rms_eps: float, selector_eps: float, post_factor: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    streams = act["streams_in"].astype(np.float32)
    attn_prefix = f"model.layers.{layer}.self_attn_raon_mhc."
    h_pre, h_post, h_res = mhc_route(reader, attn_prefix, streams, selector_eps, post_factor)
    mixed = torch.einsum("bij,bjc->bic", torch.from_numpy(h_res), torch.from_numpy(streams))
    mixed += torch.from_numpy(h_post).unsqueeze(-1) * torch.from_numpy(
        act["mixer_out"].astype(np.float32)
    ).unsqueeze(1)
    # The model rounds this boundary to BF16.
    mixed = mixed.to(torch.bfloat16).float().numpy()
    mlp_prefix = f"model.layers.{layer}.mlp_raon_mhc."
    m_pre, m_post, m_res = mhc_route(reader, mlp_prefix, mixed, selector_eps, post_factor)
    aggregate = torch.einsum(
        "bn,bnc->bc", torch.from_numpy(m_pre), torch.from_numpy(mixed)
    ).to(torch.bfloat16).float().numpy()
    norm_weight = reader.vector(f"model.layers.{layer}.post_attention_layernorm.weight")
    branch_in = rmsnorm(aggregate, norm_weight, rms_eps)
    branch_in = torch.from_numpy(branch_in).to(torch.bfloat16).float().numpy()
    return branch_in, {"streams": mixed, "h_post": m_post, "h_res": m_res}


def projection_names(layer: int, kind: str) -> list[str]:
    base = f"model.layers.{layer}.self_attn."
    if kind == "rda":
        suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight",
                    "o_proj.weight", "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight",
                    "rda_w_proj.weight"]
    else:
        suffixes = ["q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
                    "kv_b_proj.weight", "g_proj.weight", "o_proj.weight"]
    return [base + suffix for suffix in suffixes]


def matrix_plan(experts: dict[int, list[int]]) -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = []
    for layer, kind in ((0, "rda"), (1, "rda"), (3, "mla"), (24, "mla")):
        for name in projection_names(layer, kind):
            plan.append((name, f"layer{layer}/{kind}"))
        if layer == 0:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                plan.append((f"model.layers.0.mlp.{proj}.weight", "layer0/dense_mlp"))
        else:
            plan.append((f"model.layers.{layer}.moe.gate.weight", f"layer{layer}/router"))
            for expert in experts[layer]:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    plan.append((f"model.layers.{layer}.moe.experts.{expert}.{proj}.weight",
                                 f"layer{layer}/expert{expert}"))
            for proj in ("gate_proj", "up_proj", "down_proj"):
                plan.append((f"model.layers.{layer}.moe.shared_experts.{proj}.weight",
                             f"layer{layer}/shared_expert"))
    return plan


def evaluate_matrix(reader: WeightReader, gguf: GGUFWeightReader, name: str, category: str) -> dict[str, Any]:
    started = time.perf_counter()
    weight = reader.matrix(name)
    restored = q4_restore(weight)
    restored_gguf = gguf.q4_restore(name)
    metrics = {"minmax": tensor_metrics(weight, restored),
               "gguf": tensor_metrics(weight, restored_gguf)}
    method, types = gguf.source(name)
    shape, dtype = reader.shape_dtype(name)
    del weight, restored, restored_gguf
    return {"name": name, "category": category, "shape": list(shape), "source_dtype": dtype,
            "gguf_types": types, "gguf_method": method,
            "metrics": metrics, "seconds": time.perf_counter() - started}


def evaluate_projection(reader: WeightReader, gguf: GGUFWeightReader,
                        name: str, x: np.ndarray) -> dict[str, Any] | None:
    weight = reader.matrix(name)
    if x.shape[-1] != weight.shape[1]:
        return None
    restored = q4_restore(weight)
    restored_gguf = gguf.q4_restore(name)
    result = {"name": name, "tokens": int(x.shape[0]),
              "metrics": {"minmax": functional_metrics(linear(x, weight), linear(x, restored)),
                          "gguf": functional_metrics(linear(x, weight), linear(x, restored_gguf))}}
    del weight, restored, restored_gguf
    return result


def evaluate_router(reader: WeightReader, gguf: GGUFWeightReader, golden_dir: Path, layer: int,
                    x: np.ndarray) -> dict[str, Any]:
    name = f"model.layers.{layer}.moe.gate.weight"
    weight = reader.matrix(name)
    restored = q4_restore(weight)
    restored_gguf = gguf.q4_restore(name)
    logits = linear_fp32(x, weight)
    logits_q = linear_fp32(x, restored)
    logits_g = linear_fp32(x, restored_gguf)
    bias = reader.vector(f"model.layers.{layer}.moe.gate.e_score_correction_bias")
    scores = 1.0 / (1.0 + np.exp(-logits)) + bias
    scores_q = 1.0 / (1.0 + np.exp(-logits_q)) + bias
    scores_g = 1.0 / (1.0 + np.exp(-logits_g)) + bias
    top = np.argsort(scores, axis=1)[:, -6:]
    top_q = np.argsort(scores_q, axis=1)[:, -6:]
    top_g = np.argsort(scores_g, axis=1)[:, -6:]
    reference_logits, reference_top = [], []
    for path in sorted(golden_dir.glob("step_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            reference_logits.append(data[f"layer_{layer:03d}.router_logits"].reshape(-1, logits.shape[1]))
            reference_top.append(data[f"layer_{layer:03d}.router_top"].reshape(-1, 6))
    reference_logits_arr = np.concatenate(reference_logits)
    reference_top_arr = np.concatenate(reference_top)
    modes = {}
    for mode, candidate_logits, candidate_top in (("minmax", logits_q, top_q), ("gguf", logits_g, top_g)):
        overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top, candidate_top)]
        modes[mode] = {
            **functional_metrics(logits, candidate_logits),
            "top6_exact_match_rate": float(np.mean(np.all(
                np.sort(top, axis=1) == np.sort(candidate_top, axis=1), axis=1))),
            "top6_overlap_mean": float(np.mean(overlap)),
        }
    return {
        "name": name, "tokens": int(x.shape[0]), "metrics": modes,
        "original_logits_vs_dump_cosine": cosine(logits, reference_logits_arr),
        "original_top6_vs_dump_exact_match_rate": float(
            np.mean(np.all(np.sort(top, axis=1) == np.sort(reference_top_arr, axis=1), axis=1))
        ),
    }


def evaluate_mlp(reader: WeightReader, gguf: GGUFWeightReader, prefix: str, x: np.ndarray, label: str,
                 gate_beta: float, value_beta: float
                 ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    weights = {proj: reader.matrix(prefix + proj + ".weight")
               for proj in ("gate_proj", "up_proj", "down_proj")}
    restored = {proj: q4_restore(value) for proj, value in weights.items()}
    restored_gguf = {proj: gguf.q4_restore(prefix + proj + ".weight")
                     for proj in ("gate_proj", "up_proj", "down_proj")}
    gate = linear(x, weights["gate_proj"])
    up = linear(x, weights["up_proj"])
    gate_q = linear(x, restored["gate_proj"])
    up_q = linear(x, restored["up_proj"])
    gate_g = linear(x, restored_gguf["gate_proj"])
    up_g = linear(x, restored_gguf["up_proj"])
    act = bf16_round(situ_glu(gate, up, gate_beta, value_beta))
    act_q = bf16_round(situ_glu(gate_q, up_q, gate_beta, value_beta))
    act_g = bf16_round(situ_glu(gate_g, up_g, gate_beta, value_beta))
    out = linear(act, weights["down_proj"])
    out_q = linear(act_q, restored["down_proj"])
    out_g = linear(act_g, restored_gguf["down_proj"])
    result = {
        "name": label,
        "tokens": int(x.shape[0]),
        "gate_pre_activation": {"minmax": functional_metrics(gate, gate_q),
                                "gguf": functional_metrics(gate, gate_g)},
        "up_pre_activation": {"minmax": functional_metrics(up, up_q),
                              "gguf": functional_metrics(up, up_g)},
        "situ_glu": {"minmax": functional_metrics(act, act_q),
                     "gguf": functional_metrics(act, act_g)},
        "final_output": {"minmax": functional_metrics(out, out_q),
                         "gguf": functional_metrics(out, out_g)},
    }
    return result, out, out_q, out_g


def evaluate_lm_head(reader: WeightReader, gguf: GGUFWeightReader,
                     final_streams: np.ndarray, golden_dir: Path,
                     chunk_rows: int, rms_eps: float) -> dict[str, Any]:
    # Each golden step records logits for the final token of that invocation:
    # 20-token prefill, followed by fifteen one-token decode calls.
    step_lengths = []
    golden_top = []
    for path in sorted(golden_dir.glob("step_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            step_lengths.append(int(data["layer_024.streams_out"].shape[1]))
            golden_top.append(data["logits_top20_ids"].reshape(-1, 20)[-1])
    cursor = 0
    selected = []
    for length in step_lengths:
        cursor += length
        selected.append(cursor - 1)
    # Match Raon1MiniModel exactly: fp32 stream reduction, BF16 boundary, then
    # fp32-statistics RMSNorm rounded back to BF16 before the BF16 lm_head.
    hidden_t = torch.from_numpy(
        np.ascontiguousarray(final_streams[np.asarray(selected)], dtype=np.float32)
    ).mean(dim=1).to(torch.bfloat16)
    norm_weight = torch.from_numpy(reader.vector("model.norm.weight")).float()
    normalized = (
        hidden_t.float()
        * torch.rsqrt(hidden_t.float().pow(2).mean(-1, keepdim=True) + rms_eps)
        * norm_weight
    )
    hidden = normalized.to(torch.bfloat16).float().numpy()
    shape, _ = reader.shape_dtype("lm_head.weight")
    vocab = shape[0]
    logits = np.empty((hidden.shape[0], vocab), dtype=np.float32)
    logits_q = np.empty_like(logits)
    logits_g = np.empty_like(logits)
    accum = {mode: {key: 0.0 for key in ("dot", "a2", "b2", "err2")}
             for mode in ("minmax", "gguf")}
    max_abs = {"minmax": 0.0, "gguf": 0.0}
    worst_row_cos = {"minmax": 1.0, "gguf": 1.0}
    worst_row = {"minmax": -1, "gguf": -1}
    for start in range(0, vocab, chunk_rows):
        stop = min(vocab, start + chunk_rows)
        weight = reader.matrix("lm_head.weight", slice(start, stop))
        restored = q4_restore(weight)
        restored_g = gguf.q4_restore("lm_head.weight", slice(start, stop))
        logits[:, start:stop] = linear(hidden, weight)
        logits_q[:, start:stop] = linear(hidden, restored)
        logits_g[:, start:stop] = linear(hidden, restored_g)
        a = weight.astype(np.float64)
        for mode, candidate in (("minmax", restored), ("gguf", restored_g)):
            b = candidate.astype(np.float64)
            accum[mode]["dot"] += float(np.sum(a * b))
            accum[mode]["a2"] += float(np.sum(a * a))
            accum[mode]["b2"] += float(np.sum(b * b))
            accum[mode]["err2"] += float(np.sum((a - b) ** 2))
            max_abs[mode] = max(max_abs[mode], float(np.max(np.abs(weight - candidate))))
            dot = np.einsum("ij,ij->i", a, b)
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            rc = np.divide(dot, denom, out=np.ones_like(dot), where=denom != 0)
            local = int(np.argmin(rc))
            if float(rc[local]) < worst_row_cos[mode]:
                worst_row_cos[mode], worst_row[mode] = float(rc[local]), start + local
        del weight, restored, restored_g, a, b
    p = torch.softmax(torch.from_numpy(logits), dim=-1)
    logp = torch.log_softmax(torch.from_numpy(logits), dim=-1)
    # Golden capture uses torch.topk.  BF16 logits can tie at the cutoff, so
    # NumPy argpartition may choose a different equal-valued id set.
    top = torch.topk(torch.from_numpy(logits), 20, dim=-1).indices.numpy()
    golden_overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top, golden_top)]
    matrix_modes, functional_modes = {}, {}
    for mode, candidate in (("minmax", logits_q), ("gguf", logits_g)):
        ac = accum[mode]
        matrix_modes[mode] = {
            "cosine": ac["dot"] / math.sqrt(ac["a2"] * ac["b2"]),
            "relative_frobenius": math.sqrt(ac["err2"] / ac["a2"]),
            "max_abs_error": max_abs[mode], "worst_output_channel_cosine": worst_row_cos[mode],
            "worst_output_channel": worst_row[mode],
        }
        log_candidate = torch.log_softmax(torch.from_numpy(candidate), dim=-1)
        kl = torch.sum(p * (logp - log_candidate), dim=-1)
        top_candidate = torch.topk(torch.from_numpy(candidate), 20, dim=-1).indices.numpy()
        overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top, top_candidate)]
        functional_modes[mode] = {
            **functional_metrics(logits, candidate),
            "argmax_match_rate": float(np.mean(np.argmax(logits, axis=1) == np.argmax(candidate, axis=1))),
            "top20_overlap_mean": float(np.mean(overlap)), "top20_overlap_min": int(np.min(overlap)),
            "kl_mean": float(torch.mean(kl)), "kl_max": float(torch.max(kl)),
        }
    return {
        "matrix": {
            "name": "lm_head.weight", "shape": [vocab, hidden.shape[1]], "source_dtype": "BF16",
            "gguf_types": gguf.source("lm_head.weight")[1], "metrics": matrix_modes,
        },
        "functional": {
            "name": "lm_head", "tokens": int(hidden.shape[0]),
            "metrics": functional_modes,
            "original_vs_golden_top20_overlap_mean": float(np.mean(golden_overlap)),
        },
    }


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Raon-1-mini q4nx 실제 품질 측정", "",
             f"- 실행 시간: {result['runtime']['elapsed_seconds']:.1f}초",
             f"- 최대 RSS: {result['runtime']['max_rss_gib']:.2f} GiB",
             f"- 실제 활성 토큰: {result['inputs']['activation_tokens']}개",
             f"- 전문가 표본(seed={EXPERT_SEED}): {result['inputs']['experts']}", "",
             "## 행렬 단위", "",
             "| 텐서 | GGUF type | min-max cosine | GGUF→q4nx cosine | min-max worst row | GGUF worst row |",
             "|---|---:|---:|---:|---:|---:|"]
    matrices = result["matrices"] + [result["lm_head"]["matrix"]]
    for item in matrices:
        mm, gg = item["metrics"]["minmax"], item["metrics"]["gguf"]
        lines.append(f"| `{item['name']}` | {'+'.join(item['gguf_types'])} | {mm['cosine']:.8f} | "
                     f"{gg['cosine']:.8f} | {mm['worst_output_channel_cosine']:.8f} | "
                     f"{gg['worst_output_channel_cosine']:.8f} |")
    lines += ["", "## 기능 단위", "",
              "| 경로 | 토큰 | min-max cosine | GGUF→q4nx cosine |", "|---|---:|---:|---:|"]
    for item in result["projections"]:
        mm, gg = item["metrics"]["minmax"], item["metrics"]["gguf"]
        lines.append(f"| `{item['name']}` | {item['tokens']} | {mm['cosine']:.8f} | {gg['cosine']:.8f} |")
    for item in result["mlps"]:
        for stage in ("gate_pre_activation", "up_pre_activation", "situ_glu", "final_output"):
            metric = item[stage]
            lines.append(f"| `{item['name']}/{stage}` | {item['tokens']} | "
                         f"{metric['minmax']['cosine']:.8f} | {metric['gguf']['cosine']:.8f} |")
    lm = result["lm_head"]["functional"]
    lines.append(f"| `lm_head/logits` | {lm['tokens']} | {lm['metrics']['minmax']['cosine']:.8f} | "
                 f"{lm['metrics']['gguf']['cosine']:.8f} |")
    lines += ["", "## lm_head 토큰 품질", "",
              "| 방식 | argmax | top-20 평균/최소 | KL 평균/최대 |", "|---|---:|---:|---:|"]
    for mode in ("minmax", "gguf"):
        m = lm["metrics"][mode]
        lines.append(f"| {mode} | {m['argmax_match_rate']:.4f} | {m['top20_overlap_mean']:.2f}/{m['top20_overlap_min']} | "
                     f"{m['kl_mean']:.6g}/{m['kl_max']:.6g} |")
    lines += [
              f"- 재계산 원본과 golden top-20 겹침: 평균 {lm['original_vs_golden_top20_overlap_mean']:.2f}/20", "",
              "## 판정", "",
              f"- min-max: **{result['verdict']['minmax']['status']}**, 최약 "
              f"`{result['verdict']['minmax']['weakest']['name']}` "
              f"({result['verdict']['minmax']['weakest']['cosine']:.8f})",
              f"- GGUF→q4nx: **{result['verdict']['gguf']['status']}**, 최약 "
              f"`{result['verdict']['gguf']['weakest']['name']}` "
              f"({result['verdict']['gguf']['weakest']['cosine']:.8f})", "",
              "### 범위", "",
              "행렬 품질은 지정된 RDA/MLA/라우터/밀집 MLP와 각 MoE 층의 고정 난수 전문가 4개 및 공유 전문가를 포함한다. "
              "기능 품질은 golden-run의 35개 실제 토큰을 사용한다. MoE 기능 평가는 표본 전문가 각각을 동일 실제 MoE 입력에 통과시킨 경로이며, "
              "전체 128전문가 혼합 층의 end-to-end 재실행은 아니다. layer 0의 `layer_output`은 원본 attention 출력을 고정하고 dense MLP만 q4nx로 바꾼 민감도이다.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "quant/eval_out")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--golden-threads", type=int, default=1,
        help="thread count used to create the golden dump; applied to BF16 lm_head reproduction",
    )
    parser.add_argument("--lm-chunk-rows", type=int, default=LM_CHUNK_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.threads <= 16:
        raise ValueError("--threads must be between 1 and 16")
    if not 1 <= args.golden_threads <= 16:
        raise ValueError("--golden-threads must be between 1 and 16")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    started = time.perf_counter()
    reader = WeightReader(args.model_dir)
    gguf_reader = GGUFReader(args.gguf)
    gguf = GGUFWeightReader(gguf_reader)
    config = json.loads((args.model_dir / "config.json").read_text())
    rms_eps = float(config["rms_norm_eps"])
    selector_eps = float(config.get("raon_mhc_selector_eps") or 1e-5)
    post_factor = float(config.get("raon_mhc_post_factor") or 2.0)
    gate_beta = float(config["activation_situ_beta"])
    value_beta = float(config["activation_situ_linear_beta"])
    rng = np.random.default_rng(EXPERT_SEED)
    experts = {layer: sorted(map(int, rng.choice(int(config["num_experts"]), EXPERT_COUNT, replace=False)))
               for layer in (1, 3, 24)}
    activations = load_activations(args.golden_dir, (0, 1, 3, 24))

    matrices = []
    for name, category in matrix_plan(experts):
        print(f"matrix {name}", flush=True)
        matrices.append(evaluate_matrix(reader, gguf, name, category))

    projections = []
    for layer, kind in ((0, "rda"), (1, "rda"), (3, "mla"), (24, "mla")):
        x = activations[layer]["mixer_in"].reshape(-1, activations[layer]["mixer_in"].shape[-1])
        for name in projection_names(layer, kind):
            item = evaluate_projection(reader, gguf, name, x)
            if item is not None:
                projections.append(item)
        # Evaluate low-rank second projections with their true, separately quantized first-stage input.
        if kind == "rda":
            first, second = (f"model.layers.{layer}.self_attn.f_a_proj.weight",
                             f"model.layers.{layer}.self_attn.f_b_proj.weight")
            w = reader.matrix(first)
            wq = q4_restore(w)
            wg = gguf.q4_restore(first)
            mid, midq, midg = linear(x, w), linear(x, wq), linear(x, wg)
            w2 = reader.matrix(second)
            w2q = q4_restore(w2)
            w2g = gguf.q4_restore(second)
            projections.append({"name": second + " (chained)", "tokens": int(x.shape[0]),
                                "metrics": {"minmax": functional_metrics(linear(mid, w2), linear(midq, w2q)),
                                            "gguf": functional_metrics(linear(mid, w2), linear(midg, w2g))}})
        else:
            for first_suffix, second_suffix, norm_suffix in (
                ("q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight"),
                ("kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "kv_a_layernorm.weight")):
                first = f"model.layers.{layer}.self_attn.{first_suffix}"
                second = f"model.layers.{layer}.self_attn.{second_suffix}"
                w = reader.matrix(first)
                wq = q4_restore(w)
                wg = gguf.q4_restore(first)
                mid, midq, midg = linear(x, w), linear(x, wq), linear(x, wg)
                if first_suffix.startswith("kv_a"):
                    rank = int(config["kv_lora_rank"])
                    mid, midq, midg = mid[:, :rank], midq[:, :rank], midg[:, :rank]
                norm = reader.vector(f"model.layers.{layer}.self_attn.{norm_suffix}")
                latent_eps = float(config.get("mla_latent_norm_eps") or 1e-6)
                mid, midq, midg = (
                    bf16_round(rmsnorm(mid, norm, latent_eps)),
                    bf16_round(rmsnorm(midq, norm, latent_eps)),
                    bf16_round(rmsnorm(midg, norm, latent_eps)),
                )
                w2 = reader.matrix(second)
                w2q = q4_restore(w2)
                w2g = gguf.q4_restore(second)
                projections.append({"name": second + " (chained)", "tokens": int(x.shape[0]),
                                    "metrics": {"minmax": functional_metrics(linear(mid, w2), linear(midq, w2q)),
                                                "gguf": functional_metrics(linear(mid, w2), linear(midg, w2g))}})

    mlps = []
    layer0_input, layer0_mix = reconstruct_mlp_input(reader, 0, activations[0], rms_eps, selector_eps, post_factor)
    dense, dense_out, dense_out_q, dense_out_g = evaluate_mlp(reader, gguf, "model.layers.0.mlp.", layer0_input,
                                                 "layer0/dense_mlp", gate_beta, value_beta)
    original_layer = np.einsum("bij,bjc->bic", layer0_mix["h_res"], layer0_mix["streams"]) + layer0_mix["h_post"][:, :, None] * dense_out[:, None, :]
    quant_layer = np.einsum("bij,bjc->bic", layer0_mix["h_res"], layer0_mix["streams"]) + layer0_mix["h_post"][:, :, None] * dense_out_q[:, None, :]
    gguf_layer = np.einsum("bij,bjc->bic", layer0_mix["h_res"], layer0_mix["streams"]) + layer0_mix["h_post"][:, :, None] * dense_out_g[:, None, :]
    dense["layer_output"] = {"minmax": functional_metrics(original_layer, quant_layer),
                             "gguf": functional_metrics(original_layer, gguf_layer)}
    dense["original_reconstruction_vs_dump"] = functional_metrics(original_layer, activations[0]["streams_out"])
    mlps.append(dense)
    for layer in (1, 3, 24):
        mlp_input, _ = reconstruct_mlp_input(reader, layer, activations[layer], rms_eps, selector_eps, post_factor)
        projections.append(evaluate_router(reader, gguf, args.golden_dir, layer, mlp_input))
        for expert in experts[layer]:
            prefix = f"model.layers.{layer}.moe.experts.{expert}."
            item, _, _, _ = evaluate_mlp(reader, gguf, prefix, mlp_input, f"layer{layer}/expert{expert}", gate_beta, value_beta)
            mlps.append(item)
        prefix = f"model.layers.{layer}.moe.shared_experts."
        item, _, _, _ = evaluate_mlp(reader, gguf, prefix, mlp_input, f"layer{layer}/shared_expert", gate_beta, value_beta)
        mlps.append(item)

    print("lm_head.weight", flush=True)
    torch.set_num_threads(args.golden_threads)
    lm_head = evaluate_lm_head(reader, gguf, activations[24]["streams_out"], args.golden_dir,
                               args.lm_chunk_rows, rms_eps)
    torch.set_num_threads(args.threads)

    verdict = {}
    for mode in ("minmax", "gguf"):
        candidates = [{"name": item["name"], "cosine": item["metrics"][mode]["cosine"]} for item in matrices]
        candidates.append({"name": "lm_head.weight", "cosine": lm_head["matrix"]["metrics"][mode]["cosine"]})
        candidates.extend({"name": item["name"], "cosine": item["metrics"][mode]["cosine"]} for item in projections)
        for item in mlps:
            for stage in ("situ_glu", "final_output"):
                candidates.append({"name": f"{item['name']}/{stage}", "cosine": item[stage][mode]["cosine"]})
            if "layer_output" in item:
                candidates.append({"name": f"{item['name']}/layer_output", "cosine": item["layer_output"][mode]["cosine"]})
        candidates.append({"name": "lm_head/logits", "cosine": lm_head["functional"]["metrics"][mode]["cosine"]})
        below = sorted((item for item in candidates if item["cosine"] < THRESHOLD), key=lambda x: x["cosine"])
        verdict[mode] = {"status": "FAIL" if below else "PASS (측정 범위)", "below_threshold": below,
                         "weakest": min(candidates, key=lambda x: x["cosine"])}
    elapsed = time.perf_counter() - started
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "schema_version": 2,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "threads": args.threads, "golden_threads": args.golden_threads},
        "inputs": {"model_dir": str(args.model_dir), "golden_dir": str(args.golden_dir), "gguf": str(args.gguf),
                   "activation_tokens": int(activations[0]["mixer_in"].shape[0]), "experts": experts,
                   "expert_seed": EXPERT_SEED},
        "runtime": {"elapsed_seconds": elapsed, "max_rss_gib": rss_kib / 1024**2},
        "threshold": THRESHOLD, "matrices": matrices, "projections": projections,
        "mlps": mlps, "lm_head": lm_head,
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "results.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "elapsed_seconds": elapsed,
                      "max_rss_gib": result["runtime"]["max_rss_gib"],
                      "status": {mode: verdict[mode]["status"] for mode in verdict}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
