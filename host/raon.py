# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from KRAFTON Raon-1-mini-preview, modeling_raon1_mini.py (Helmert/mHC) and configuration_raon1_mini.py, Apache-2.0.
# Modified by the raon-npu authors, 2026.
"""Raon-1-mini host-side reference model execution in NumPy."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backend import _MLA_POOL, Backend, NumpyBackend, situ_glu
from .weights import Weights, bf16_to_f32, f32_to_bf16_rtne

# Tokens per prefill RDA call: the T of the plate's RDA streams (bench/rda_state/rda_state_multi.py RDA_TOKEN_CHOICES).
RDA_BLOCK_SIZES = (1, 2, 4, 8)


def rda_block_plan(n: int, block: int) -> list[tuple[int, int]]:
    """(start, size) of the prefill RDA calls for n tokens (DESIGN-prefill.md section 36): calls of ``block`` tokens,
    then the rest in descending powers of two, so every size is a T of the RDA streams (61 tokens, block 8: seven
    calls of 8, then 4 and 1)."""
    plan = []
    start, size = 0, block
    while start < n:
        while size > n - start:
            size //= 2
        plan.append((start, size))
        start += size
    return plan


# Prefill mHC row split (DESIGN-prefill.md section 43): a chunk of T tokens runs the batch mHC functions on
# min(RAON_PREFILL_MHC_THREADS, T // MHC_MIN_ROWS) contiguous row ranges in threads (fewer than two ranges: one call
# as before); the mix goes MHC_MIX_ROWS rows at a time inside each range.
MHC_MIN_ROWS = 16
MHC_MIX_ROWS = 8


def _mhc_split(rows: int, threads: int) -> int:
    """Threads for the batch mHC of a chunk of ``rows`` tokens: 1 (the whole chunk in one call) below two ranges."""
    k = min(threads, rows // MHC_MIN_ROWS)
    return k if k >= 2 else 1


def _mhc_run_rows(fn: Callable[[int, int], None], rows: int, k: int) -> None:
    """fn(lo, hi) over k contiguous row ranges: ranges 1..k-1 on _MLA_POOL, range 0 on the calling thread.

    Waits for every submitted range before returning or raising - also when the calling thread's range or a submit
    fails - then raises the calling thread's error, else the first failed range's. Callers: RaonModel.prefill's mHC
    only, on its own thread, one prefill at a time (the server serializes generation); MLA uses the same pool but
    never at the same time, and fn must not submit to the pool itself.
    """
    edges = [i * rows // k for i in range(k + 1)]
    futures = []
    try:
        for i in range(1, k):
            futures.append(_MLA_POOL.submit(fn, edges[i], edges[i + 1]))
        fn(edges[0], edges[1])
    finally:
        wait(futures)
    for future in futures:
        future.result()


# 1-based layer indices from the released Raon1MiniConfig defaults.
_RELEASED_RDA_LAYERS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
_RELEASED_MLA_LAYERS = [4, 8, 12, 16, 20, 24, 25]


def _default_if_none(value: Any, default: Any) -> Any:
    """Match the reference configuration's explicit-null fallback semantics."""
    return default if value is None else value


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
    keep_fp32: bool = False,
) -> np.ndarray:
    """RMSNorm with FP32 statistics and plain gamma."""
    # These arrays are read-only below; copy=False safely preserves an existing
    # float32 input instead of allocating a same-dtype copy on every call.
    val = bf16_to_f32(x) if x.dtype == np.uint16 else x.astype(np.float32, copy=False)
    w = bf16_to_f32(weight) if weight.dtype == np.uint16 else weight.astype(np.float32, copy=False)

    mean_sq = np.mean(val ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)
    normed = val * rsqrt * w
    if keep_fp32:
        return normed.astype(np.float32, copy=False)
    return f32_to_bf16_rtne(normed)


def helmert_basis(n: int) -> np.ndarray:
    """Orthonormal Q in R^{n x (n-1)} with Q^T Q = I and Q^T 1 = 0."""
    q = np.zeros((n, n - 1), dtype=np.float32)
    for k in range(1, n):
        q[:k, k - 1] = 1.0
        q[k, k - 1] = -float(k)
        q[:, k - 1] /= math.sqrt(k * (k + 1.0))
    return q


_MHC_CONSTANTS_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def get_mhc_constants(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n not in _MHC_CONSTANTS_CACHE:
        eye = np.eye(n, dtype=np.float32)
        chart_b = 0.5 * (eye + np.full_like(eye, 1.0 / n))
        helmert = helmert_basis(n)
        _MHC_CONSTANTS_CACHE[n] = (helmert, eye, chart_b)
    return _MHC_CONSTANTS_CACHE[n]


@dataclass
class RaonConfig:
    hidden_size: int = 2560
    num_hidden_layers: int = 25
    vocab_size: int = 259840
    rms_norm_eps: float = 1e-5
    raon_mhc_streams: int = 4
    raon_mhc_selector_eps: float = 1e-5
    raon_mhc_post_factor: float = 2.0
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    intermediate_size: int = 12032
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_experts: int = 128
    num_experts_per_token: int = 6
    moe_intermediate_size: int = 1088
    num_shared_experts: int = 1
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    # MLA
    num_attention_heads: int = 40
    q_lora_rank: int = 576
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_latent_norm_eps: float = 1e-6
    mla_use_output_gate: bool = True
    # RDA
    rda_num_heads: int = 40
    rda_head_dim: int = 128
    rda_conv_kernel_size: int = 4
    rda_decay_rank: int = 128
    rda_gate_lower_bound: float = -5.0
    rda_layers: list[int] = field(default_factory=list)  # 0-based
    mla_layers: list[int] = field(default_factory=list)  # 0-based

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaonConfig":
        num_attention_heads = data.get("num_attention_heads", 40)
        head_dim = data.get("head_dim", 128)
        if head_dim is None:
            head_dim = data.get("hidden_size", 2560) // num_attention_heads
        num_hidden_layers = data.get("num_hidden_layers", 25)
        rda_cfg = data.get("rda_config")
        rda_config_provided = rda_cfg is not None
        if rda_cfg is None:
            rda_cfg = {
                "num_heads": num_attention_heads,
                "head_dim": head_dim,
                "conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "rda_layers": _RELEASED_RDA_LAYERS,
                "mla_layers": _RELEASED_MLA_LAYERS,
            }
        # The reference config keeps the provided rda_config dict as-is and a
        # missing "rda_layers" surfaces as a hard key error at model-config time;
        # a provided rda_config without rda_layers must never be silently
        # accepted here as "no RDA layers" (which would run every layer as MLA).
        if rda_config_provided and rda_cfg.get("rda_layers") is None:
            raise ValueError(
                "rda_config was provided without 'rda_layers': MLA layers are the complement"
                " of rda_layers, so a partial rda_config is ambiguous — provide 'rda_layers'"
                " (1-based layer indices) or omit rda_config to use the released defaults"
            )
        # Note: 1-based in config -> 0-based in RaonConfig. The reference model
        # never reads mla_layers — every non-RDA layer is MLA — so MLA layers are
        # computed as the complement of rda_layers; an explicitly provided
        # mla_layers must agree with that complement.
        raw_rda_layers = rda_cfg.get("rda_layers")
        rda_layers = sorted(x - 1 for x in raw_rda_layers)
        mla_complement = sorted(set(range(num_hidden_layers)) - set(rda_layers))
        raw_mla_layers = rda_cfg.get("mla_layers")
        if raw_mla_layers is None:
            mla_layers = mla_complement
        else:
            mla_layers = sorted(x - 1 for x in raw_mla_layers)
            if mla_layers != mla_complement:
                raise ValueError(
                    f"rda_config 'mla_layers' {sorted(raw_mla_layers)} (0-based {mla_layers}) does not"
                    f" match the complement of 'rda_layers' {sorted(raw_rda_layers)}"
                    f" (0-based {mla_complement}) over {num_hidden_layers} hidden layers"
                )

        router_activation = data.get("moe_router_activation_func", "sigmoid")
        if router_activation != "sigmoid":
            raise NotImplementedError("raon-1-mini uses the sigmoid router.")

        return cls(
            hidden_size=data.get("hidden_size", 2560),
            num_hidden_layers=num_hidden_layers,
            vocab_size=data.get("vocab_size", 259840),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            raon_mhc_streams=_default_if_none(data.get("raon_mhc_streams"), 4),
            raon_mhc_selector_eps=_default_if_none(data.get("raon_mhc_selector_eps"), 1e-5),
            raon_mhc_post_factor=_default_if_none(data.get("raon_mhc_post_factor"), 2.0),
            activation_situ_beta=data.get("activation_situ_beta", 4.0),
            activation_situ_linear_beta=data.get("activation_situ_linear_beta", 25.0),
            intermediate_size=data.get("intermediate_size", 12032),
            first_k_dense_replace=data.get("first_k_dense_replace", 1),
            moe_layer_freq=data.get("moe_layer_freq", 1),
            num_experts=data.get("num_experts", 128),
            num_experts_per_token=data.get("num_experts_per_token", 6),
            moe_intermediate_size=data.get("moe_intermediate_size", 1088),
            num_shared_experts=data.get("num_shared_experts", 1),
            moe_router_activation_func=router_activation,
            moe_renormalize=data.get("moe_renormalize", True),
            routed_scaling_factor=data.get("routed_scaling_factor", 1.0),
            num_attention_heads=num_attention_heads,
            q_lora_rank=data.get("q_lora_rank", 576),
            kv_lora_rank=data.get("kv_lora_rank", 512),
            qk_nope_head_dim=data.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=data.get("qk_rope_head_dim", 64),
            v_head_dim=data.get("v_head_dim", 128),
            mla_latent_norm_eps=data.get("mla_latent_norm_eps", 1e-6),
            mla_use_output_gate=data.get("mla_use_output_gate", True),
            rda_num_heads=rda_cfg.get("num_heads", num_attention_heads),
            rda_head_dim=rda_cfg.get("head_dim", head_dim),
            rda_conv_kernel_size=rda_cfg.get("conv_kernel_size", 4),
            rda_decay_rank=data.get("rda_decay_rank", 128),
            rda_gate_lower_bound=rda_cfg.get("gate_lower_bound", -5.0),
            rda_layers=rda_layers,
            mla_layers=mla_layers,
        )

    def is_rda_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.rda_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.num_experts:
            return False
        return (
            layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )


class RaonModel:
    """Raon-1-mini decode engine."""

    def __init__(
        self,
        config: RaonConfig,
        weights: Weights,
        backend: Backend | None = None,
    ):
        self.config = config
        self.weights = weights
        self.backend = backend or NumpyBackend(weights)

        # Conv states for RDA: layer_idx -> { 'q': [channels, K-1], 'k': ..., 'v': ... }
        self.conv_states: dict[int, dict[str, np.ndarray]] = {}

        # Last captured intermediates for inspection / verification
        self.last_intermediates: dict[str, np.ndarray] = {}
        self._mhc_site_cache: dict[str, tuple[Any, ...]] = {}
        self._layer_host_cache: dict[int, dict[str, np.ndarray]] = {}
        # Prefill mHC in chunk-wide batches (RAON_PREFILL_MHC_BATCH, on by default); None = read the environment.
        self._prefill_mhc_batch: bool | None = None
        # Prefill MLA attention per chunk (RAON_PREFILL_MLA_BATCH, on by default); None = read the environment.
        self._prefill_mla_batch: bool | None = None
        # Prefill mHC threads (RAON_PREFILL_MHC_THREADS: 4 by default, 1 = one call per chunk as before -
        # DESIGN-prefill.md section 43); None = read the environment.
        self._prefill_mhc_threads: int | None = None
        # Fast RDA conv (DESIGN-prefill.md sections 33/36): RAON_RDA_CONV_FAST (on by default) outside the prefill
        # overlap loop, RAON_RDA_CONV_FAST_LOOP (on by default since the block loop, R3-2) inside it too; None = read
        # the environment.
        # _conv_fast: RDA layer -> (time-major window, the history views it put into conv_states); _conv_wt: RDA layer
        # -> (conv weight, its transpose); _in_rda_overlap is set while _rda_tokens_overlapped runs.
        self._rda_conv_fast: bool | None = None
        self._rda_conv_fast_loop: bool | None = None
        self._conv_fast: dict[int, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._conv_wt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._in_rda_overlap = False
        # Prefill RDA tokens per overlapped call (RAON_PREFILL_RDA_BLOCK: 8 by default, or 1, 2, 4 - DESIGN-prefill.md
        # section 36); None = read the environment.
        self._prefill_rda_block: int | None = None

    def reset_state(self) -> None:
        self.conv_states.clear()
        self._conv_fast.clear()
        self.backend.reset()
        self.last_intermediates.clear()

    def load_snapshot(self, snapshot_file_or_dict: str | Path | dict[str, np.ndarray]) -> None:
        """Load recurrent states, MLA cache, and conv states from a golden snapshot."""
        if isinstance(snapshot_file_or_dict, (str, Path)):
            npz = np.load(str(snapshot_file_or_dict))
            data = {k: npz[k] for k in npz.files}
        else:
            data = snapshot_file_or_dict

        self.reset_state()

        for idx in range(self.config.num_hidden_layers):
            pfx = f"layer_{idx:03d}"
            # RDA recurrent state
            rda_key = f"{pfx}.rda_state"
            if rda_key in data:
                self.backend.set_rda_state(idx, data[rda_key])

            # MLA expanded KV cache; a half pair (only one side present) is a
            # corrupt snapshot, never a silently-skipped cache.
            mk_key = f"{pfx}.mla_key_cache"
            mv_key = f"{pfx}.mla_value_cache"
            if (mk_key in data) != (mv_key in data):
                present = mk_key if mk_key in data else mv_key
                missing = mv_key if mk_key in data else mk_key
                raise KeyError(
                    f"Incomplete MLA expanded cache pair in snapshot for layer {idx}: "
                    f"'{present}' present but '{missing}' missing"
                )
            if mk_key in data and mv_key in data:
                self.backend.set_mla_cache(idx, data[mk_key], data[mv_key])

            # MLA latent KV cache -> expand to full KV cache if expanded KV is not provided
            m_lat_key = f"{pfx}.mla_kv_lat_cache"
            m_pe_key = f"{pfx}.mla_k_pe_cache"
            if (m_lat_key in data) != (m_pe_key in data):
                present = m_lat_key if m_lat_key in data else m_pe_key
                missing = m_pe_key if m_lat_key in data else m_lat_key
                raise KeyError(
                    f"Incomplete MLA latent/PE cache pair in snapshot for layer {idx}: "
                    f"'{present}' present but '{missing}' missing"
                )
            if (mk_key not in data or mv_key not in data) and (m_lat_key in data and m_pe_key in data):
                raw_lat = data[m_lat_key].squeeze(0) if data[m_lat_key].ndim == 3 else data[m_lat_key]
                raw_pe = data[m_pe_key].squeeze(0) if data[m_pe_key].ndim == 3 else data[m_pe_key]
                pe_f32 = bf16_to_f32(raw_pe) if raw_pe.dtype == np.uint16 else raw_pe.astype(np.float32)

                # Expand the latent bits (uint16 BF16 [T, K]) through the backend so
                # backends that own kv_b themselves (e.g. an NPU k_b/v_b bundle) work
                # too; a missing kv_b raises instead of silently skipping expansion.
                kv_b_name = f"model.layers.{idx}.self_attn.kv_b_proj.weight"
                kv_u16 = self.backend.linear(kv_b_name, raw_lat)

                if kv_u16.dtype != np.uint16:
                    kv_u16 = f32_to_bf16_rtne(kv_u16)

                heads = self.config.num_attention_heads
                qk_nope = self.config.qk_nope_head_dim
                v_dim = self.config.v_head_dim
                t_len = raw_lat.shape[0]

                kv_f32 = bf16_to_f32(kv_u16).reshape(t_len, heads, qk_nope + v_dim)
                k_nope = kv_f32[:, :, :qk_nope]
                value = kv_f32[:, :, qk_nope:]

                pe_exp = np.tile(pe_f32[:, np.newaxis, :], (1, heads, 1))
                key = np.concatenate([k_nope, pe_exp], axis=-1)

                key_cache = f32_to_bf16_rtne(key.transpose(1, 0, 2))
                val_cache = f32_to_bf16_rtne(value.transpose(1, 0, 2))
                self.backend.set_mla_cache(idx, key_cache, val_cache)

            # RDA conv states
            conv_key = f"{pfx}.rda_conv_state"
            if conv_key in data:
                # Shape: [3 * proj_size, K-1]
                raw_conv = data[conv_key].squeeze(0) if data[conv_key].ndim == 3 else data[conv_key]
                if raw_conv.dtype == np.uint16:
                    raw_conv = bf16_to_f32(raw_conv)
                elif raw_conv.dtype != np.float32:
                    raw_conv = raw_conv.astype(np.float32)
                proj_size = self.config.rda_num_heads * self.config.rda_head_dim
                q_hist = raw_conv[:proj_size]
                k_hist = raw_conv[proj_size : 2 * proj_size]
                v_hist = raw_conv[2 * proj_size :]
                self.conv_states[idx] = {"q": q_hist, "k": k_hist, "v": v_hist}

    def _conv_step(
        self,
        layer_idx: int,
        kind: str,
        x: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """1-step 1D depthwise short causal conv with SiLU in FP32."""
        # x is [proj_size] uint16 or float32
        # weight is [proj_size, kernel_size] float32
        # x_f32 is only read while constructing `window`; aliasing x is safe.
        x_f32 = bf16_to_f32(x) if x.dtype == np.uint16 else x.astype(np.float32, copy=False)
        k_size = self.config.rda_conv_kernel_size
        hist_size = k_size - 1

        layer_conv = self.conv_states.setdefault(layer_idx, {})
        hist = layer_conv.get(kind)
        if hist is None:
            hist = np.zeros((x_f32.shape[0], hist_size), dtype=np.float32)
        elif hist.dtype == np.uint16:
            hist = bf16_to_f32(hist)
        else:
            hist = hist.astype(np.float32, copy=False)

        # Window: [hist, x] -> [channels, k_size]
        window = np.concatenate([hist, x_f32[:, np.newaxis]], axis=-1)

        # New history is the rightmost hist_size tokens
        layer_conv[kind] = window[:, -hist_size:]

        # Convolve: sum over kernel dimension
        conv_out = np.sum(window * weight, axis=-1)  # [channels]
        # SiLU in FP32: out * sigmoid(out)
        sig = 1.0 / (1.0 + np.exp(-conv_out))
        silu_out = conv_out * sig
        return f32_to_bf16_rtne(silu_out)

    def _conv_step_qkv(
        self,
        layer_idx: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the three independent depthwise convolutions as one row batch."""
        if self.rda_conv_fast and (self.rda_conv_fast_loop or not self._in_rda_overlap):
            return self._conv_step_qkv_fast(layer_idx, q, k, v, weight)
        proj_size = q.shape[0]
        x_f32 = bf16_to_f32(np.concatenate([q, k, v]))
        hist_size = self.config.rda_conv_kernel_size - 1
        layer_conv = self.conv_states.setdefault(layer_idx, {})
        histories = []
        for kind in ("q", "k", "v"):
            hist = layer_conv.get(kind)
            if hist is None:
                hist = np.zeros((proj_size, hist_size), dtype=np.float32)
            elif hist.dtype == np.uint16:
                hist = bf16_to_f32(hist)
            else:
                hist = hist.astype(np.float32, copy=False)
            histories.append(hist)
        window = np.concatenate(
            [np.concatenate(histories, axis=0), x_f32[:, np.newaxis]], axis=-1
        )
        next_hist = window[:, -hist_size:]
        layer_conv["q"] = next_hist[:proj_size]
        layer_conv["k"] = next_hist[proj_size : 2 * proj_size]
        layer_conv["v"] = next_hist[2 * proj_size :]
        conv_out = np.sum(window * weight, axis=-1)
        silu_out = conv_out * (1.0 / (1.0 + np.exp(-conv_out)))
        out = f32_to_bf16_rtne(silu_out)
        return out[:proj_size], out[proj_size : 2 * proj_size], out[2 * proj_size :]

    def _conv_step_qkv_fast(
        self,
        layer_idx: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_conv_step_qkv with a time-major window, same bytes (DESIGN-prefill.md section 33).

        The window is [K, 3P] FP32, a new buffer per step (rows 0..K-2 copied from the previous window, row K-1 = this
        token), so an array already handed out is never written again; conv_states holds views of it. A history that
        was replaced from outside (checkpoint restore, load_snapshot, reset_state, the reference path) is no longer
        the cached view, and the window is rebuilt from conv_states - conv history arrays are replaced, never modified
        in place. The products are the same elementwise multiplies; the sum is +0 + p0 + p1 + ... in window order,
        which is what np.sum(axis=-1) does on a C-contiguous [N, K] float32 array (host/test_rda_conv_fast.py checks
        it). SiLU and the BF16 rounding are the same expressions.
        """
        proj = q.shape[0]
        n = 3 * proj
        k_size = self.config.rda_conv_kernel_size
        layer_conv = self.conv_states.setdefault(layer_idx, {})
        cached = self._conv_fast.get(layer_idx)
        if cached is not None and all(layer_conv.get(kind) is view for kind, view in zip(("q", "k", "v"), cached[1])):
            prev = cached[0]
        else:  # first step, or histories replaced / cleared from outside: rebuild from conv_states
            prev = np.zeros((k_size, n), dtype=np.float32)
            for j, kind in enumerate(("q", "k", "v")):
                hist = layer_conv.get(kind)
                if hist is not None:
                    hist = bf16_to_f32(hist) if hist.dtype == np.uint16 else hist.astype(np.float32, copy=False)
                    prev[1:, j * proj : (j + 1) * proj] = hist.T
        window = np.empty((k_size, n), dtype=np.float32)
        window[:-1] = prev[1:]
        window[-1, :proj] = bf16_to_f32(q)
        window[-1, proj : 2 * proj] = bf16_to_f32(k)
        window[-1, 2 * proj :] = bf16_to_f32(v)
        views = tuple(window[1:, j * proj : (j + 1) * proj].T for j in range(3))
        layer_conv["q"], layer_conv["k"], layer_conv["v"] = views
        self._conv_fast[layer_idx] = (window, views)
        cached_wt = self._conv_wt.get(layer_idx)
        if cached_wt is None or cached_wt[0] is not weight:
            cached_wt = (weight, np.ascontiguousarray(weight.T, dtype=np.float32))
            self._conv_wt[layer_idx] = cached_wt
        prod = window * cached_wt[1]
        acc = np.float32(0.0) + prod[0]  # np.sum(axis=-1) starts from +0 and adds the window in order
        for j in range(1, k_size):
            acc += prod[j]
        out = f32_to_bf16_rtne(acc * (1.0 / (1.0 + np.exp(-acc))))
        return out[:proj], out[proj : 2 * proj], out[2 * proj :]

    def _mhc_site_constants(self, site_prefix: str) -> tuple[Any, ...]:
        cached = self._mhc_site_cache.get(site_prefix)
        if cached is None:
            weight = np.concatenate(
                [
                    self.weights.get(f"{site_prefix}.phi_pre", to_float=True),
                    self.weights.get(f"{site_prefix}.phi_post", to_float=True),
                    self.weights.get(f"{site_prefix}.res.phi_z", to_float=True),
                ],
                axis=0,
            )
            cached = (
                weight,
                float(self.weights.get(f"{site_prefix}.alpha_pre", to_float=True)),
                float(self.weights.get(f"{site_prefix}.alpha_post", to_float=True)),
                float(self.weights.get(f"{site_prefix}.res.alpha_z", to_float=True)),
                self.weights.get(f"{site_prefix}.b_pre", to_float=True),
                self.weights.get(f"{site_prefix}.b_post", to_float=True),
                self.weights.get(f"{site_prefix}.res.b_z", to_float=True),
                float(self.weights.get(f"{site_prefix}.res.eta", to_float=True)),
            )
            self._mhc_site_cache[site_prefix] = cached
        return cached

    def _layer_constants(self, layer_idx: int) -> dict[str, np.ndarray]:
        cached = self._layer_host_cache.get(layer_idx)
        if cached is not None:
            return cached
        layer_pfx = f"model.layers.{layer_idx}"
        cached = {
            "input_norm": self.weights.get(f"{layer_pfx}.input_layernorm.weight", to_float=True),
            "post_norm": self.weights.get(
                f"{layer_pfx}.post_attention_layernorm.weight", to_float=True
            ),
        }
        if self.config.is_rda_layer(layer_idx):
            pfx = f"{layer_pfx}.self_attn"
            cached.update(
                {
                    "conv_weight": np.concatenate(
                        [
                            self.weights.get(f"{pfx}.q_conv1d.weight", to_float=True),
                            self.weights.get(f"{pfx}.k_conv1d.weight", to_float=True),
                            self.weights.get(f"{pfx}.v_conv1d.weight", to_float=True),
                        ],
                        axis=0,
                    ),
                    "dt_bias": self.weights.get(f"{pfx}.dt_bias", to_float=True).reshape(
                        self.config.rda_num_heads, self.config.rda_head_dim
                    ),
                    "exp_a_log": np.exp(self.weights.get(f"{pfx}.A_log", to_float=True)),
                    "o_norm": self.weights.get(f"{pfx}.o_norm.weight", to_float=True),
                }
            )
        else:
            pfx = f"{layer_pfx}.self_attn"
            cached.update(
                {
                    "q_a_norm": self.weights.get(f"{pfx}.q_a_layernorm.weight", to_float=True),
                    "kv_a_norm": self.weights.get(f"{pfx}.kv_a_layernorm.weight", to_float=True),
                }
            )
        if self.config.is_moe_layer(layer_idx):
            pfx = f"{layer_pfx}.moe"
            cached.update(
                {
                    "router_weight": self.weights.get(f"{pfx}.gate.weight", to_float=True),
                    "router_bias": self.weights.get(
                        f"{pfx}.gate.e_score_correction_bias", to_float=True
                    ),
                }
            )
        self._layer_host_cache[layer_idx] = cached
        return cached

    def _route_mhc(
        self,
        stream: np.ndarray,
        site_prefix: str,
        stream_f32: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Raon_mHC routing coefficients (h_pre, h_post, h_res) in FP32."""
        n = self.config.raon_mhc_streams
        c = self.config.hidden_size
        eps = self.config.raon_mhc_selector_eps
        post_factor = self.config.raon_mhc_post_factor

        weight, alpha_pre, alpha_post, alpha_z, b_pre, b_post, b_z, eta = (
            self._mhc_site_constants(site_prefix)
        )

        # flat stream: [n * c] uint16
        flat_f32 = (
            bf16_to_f32(stream.reshape(n * c))
            if stream_f32 is None
            else stream_f32.reshape(n * c)
        )
        # inv = rsqrt(mean(flat^2) + eps)
        inv = 1.0 / np.sqrt(np.mean(flat_f32 ** 2) + eps)

        # F.linear(flat, weight.to(flat.dtype)).float() * inv
        # Notice: GEMM in model dtype (BF16), then float * inv
        z_gemm = np.dot(flat_f32, weight.T)
        z = f32_to_bf16_rtne(z_gemm)
        z_f32 = bf16_to_f32(z) * inv

        t_pre = alpha_pre * z_f32[:n] + b_pre
        t_post = alpha_post * z_f32[n : 2 * n] + b_post
        t_z = alpha_z * z_f32[2 * n :] + b_z

        h_pre = 1.0 / (1.0 + np.exp(-t_pre))
        h_post = post_factor * (1.0 / (1.0 + np.exp(-t_post)))

        nm = n - 1
        helmert, eye, chart_b = get_mhc_constants(n)
        t_z_mat = t_z.reshape(nm, nm)
        # e = helmert @ t_z_mat @ helmert.T
        e = np.dot(helmert, np.dot(t_z_mat, helmert.T))
        rho = max(0.0, float(np.max(-e / chart_b)))
        phi = chart_b + e / (1.0 + rho)
        gamma = 1.0 / (1.0 + math.exp(-eta))
        h_res = (1.0 - gamma) * eye + gamma * phi
        return h_pre, h_post, h_res

    def _mhc_aggregate(
        self, stream: np.ndarray, h_pre: np.ndarray, stream_f32: np.ndarray | None = None
    ) -> np.ndarray:
        """stream [n, C] -> branch input [C] in BF16 uint16."""
        if stream_f32 is None:
            stream_f32 = bf16_to_f32(stream)
        agg = np.dot(h_pre, stream_f32)  # [C]
        return f32_to_bf16_rtne(agg)

    def _mhc_mix(
        self,
        stream: np.ndarray,
        branch_out: np.ndarray,
        h_post: np.ndarray,
        h_res: np.ndarray,
        stream_f32: np.ndarray | None = None,
    ) -> np.ndarray:
        """Residual mixing plus branch write-back, rounded once to BF16."""
        if stream_f32 is None:
            stream_f32 = bf16_to_f32(stream)  # [n, C]
        branch_f32 = bf16_to_f32(branch_out)  # [C]

        mixed = np.dot(h_res, stream_f32)  # [n, C]
        res = mixed + h_post[:, np.newaxis] * branch_f32[np.newaxis, :]
        return f32_to_bf16_rtne(res)

    @property
    def prefill_mhc_batch(self) -> bool:
        if self._prefill_mhc_batch is not None:
            return bool(self._prefill_mhc_batch)
        return os.environ.get("RAON_PREFILL_MHC_BATCH", "1") not in ("0", "false", "False", "no")

    @prefill_mhc_batch.setter
    def prefill_mhc_batch(self, value: bool | None) -> None:
        self._prefill_mhc_batch = value

    @property
    def prefill_mhc_threads(self) -> int:
        value = self._prefill_mhc_threads
        if value is None:
            raw = os.environ.get("RAON_PREFILL_MHC_THREADS", "4")
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"RAON_PREFILL_MHC_THREADS must be an integer >= 1, got {raw!r}") from None
        if value < 1:
            raise ValueError(f"RAON_PREFILL_MHC_THREADS must be an integer >= 1, got {value}")
        return value

    @prefill_mhc_threads.setter
    def prefill_mhc_threads(self, value: int | None) -> None:
        self._prefill_mhc_threads = value

    @property
    def prefill_mla_batch(self) -> bool:
        if self._prefill_mla_batch is not None:
            return bool(self._prefill_mla_batch)
        return os.environ.get("RAON_PREFILL_MLA_BATCH", "1") not in ("0", "false", "False", "no")

    @prefill_mla_batch.setter
    def prefill_mla_batch(self, value: bool | None) -> None:
        self._prefill_mla_batch = value

    @property
    def rda_conv_fast(self) -> bool:
        if self._rda_conv_fast is not None:
            return bool(self._rda_conv_fast)
        return os.environ.get("RAON_RDA_CONV_FAST", "1") not in ("0", "false", "False", "no")

    @rda_conv_fast.setter
    def rda_conv_fast(self, value: bool | None) -> None:
        self._rda_conv_fast = value

    @property
    def rda_conv_fast_loop(self) -> bool:
        if self._rda_conv_fast_loop is not None:
            return bool(self._rda_conv_fast_loop)
        return os.environ.get("RAON_RDA_CONV_FAST_LOOP", "1") not in ("0", "false", "False", "no")

    @rda_conv_fast_loop.setter
    def rda_conv_fast_loop(self, value: bool | None) -> None:
        self._rda_conv_fast_loop = value

    @property
    def prefill_rda_block(self) -> int:
        value = self._prefill_rda_block
        if value is None:
            value = int(os.environ.get("RAON_PREFILL_RDA_BLOCK", "8"))
        if value not in RDA_BLOCK_SIZES:
            raise ValueError(f"RAON_PREFILL_RDA_BLOCK must be one of {RDA_BLOCK_SIZES}, got {value}")
        return value

    @prefill_rda_block.setter
    def prefill_rda_block(self, value: int | None) -> None:
        self._prefill_rda_block = value

    def _route_mhc_batch(
        self, stream_f32: np.ndarray, site_prefix: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_route_mhc for a whole prefill chunk ([T, n, C] FP32), byte-identical to calling it token by token."""
        n = self.config.raon_mhc_streams
        c = self.config.hidden_size
        eps = self.config.raon_mhc_selector_eps
        post_factor = self.config.raon_mhc_post_factor

        weight, alpha_pre, alpha_post, alpha_z, b_pre, b_post, b_z, eta = (
            self._mhc_site_constants(site_prefix)
        )
        T = stream_f32.shape[0]
        flat_f32 = stream_f32.reshape(T, n * c)
        # _route_mhc adds eps to the np.float32 scalar np.mean returns (numpy 1.x promotes that sum to float64, numpy 2
        # keeps float32): the same scalar expression per row, then float32 as its array x scalar product rounds it.
        inv = np.array([1.0 / np.sqrt(m + eps) for m in np.mean(flat_f32 ** 2, axis=1)]).astype(np.float32)
        # Stacked vectors keep numpy on the per-row GEMV path of np.dot(flat, weight.T); a 2-D np.dot would use GEMM,
        # whose summation order differs.
        z_gemm = np.matmul(flat_f32[:, np.newaxis, :], weight.T)[:, 0, :]
        z_f32 = bf16_to_f32(f32_to_bf16_rtne(z_gemm)) * inv[:, np.newaxis]

        t_pre = alpha_pre * z_f32[:, :n] + b_pre
        t_post = alpha_post * z_f32[:, n : 2 * n] + b_post
        t_z = alpha_z * z_f32[:, 2 * n :] + b_z

        h_pre = 1.0 / (1.0 + np.exp(-t_pre))
        h_post = post_factor * (1.0 / (1.0 + np.exp(-t_post)))

        nm = n - 1
        helmert, eye, chart_b = get_mhc_constants(n)
        e = np.matmul(helmert, np.matmul(t_z.reshape(T, nm, nm), helmert.T))
        # _route_mhc: rho = max(0.0, float(np.max(-e / chart_b))) - a Python float (0.0 for NaN) - and e / (1.0 + rho)
        # rounds 1.0 + rho to float32.
        m = np.max(-e / chart_b, axis=(1, 2)).astype(np.float64)
        rho = np.where(m > 0.0, m, 0.0)
        phi = chart_b + e / (1.0 + rho).astype(np.float32)[:, np.newaxis, np.newaxis]
        gamma = 1.0 / (1.0 + math.exp(-eta))
        h_res = (1.0 - gamma) * eye + gamma * phi
        return h_pre, h_post, h_res

    def _mhc_aggregate_batch(self, stream_f32: np.ndarray, h_pre: np.ndarray) -> np.ndarray:
        """_mhc_aggregate for a chunk: [T, n, C] FP32 and h_pre [T, n] -> branch inputs [T, C] BF16 uint16."""
        agg = np.matmul(h_pre[:, np.newaxis, :], stream_f32)[:, 0, :]
        return f32_to_bf16_rtne(agg)

    def _mhc_mix_batch(
        self, stream_f32: np.ndarray, branch_out: np.ndarray, h_post: np.ndarray, h_res: np.ndarray
    ) -> np.ndarray:
        """_mhc_mix for a chunk: returns the new stream [T, n, C] BF16 uint16."""
        branch_f32 = bf16_to_f32(branch_out)  # [T, C]
        mixed = np.matmul(h_res, stream_f32)  # [T, n, C]
        res = mixed + h_post[:, :, np.newaxis] * branch_f32[:, np.newaxis, :]
        return f32_to_bf16_rtne(res)

    def _mhc_front_batch(
        self, stream_f32: np.ndarray, site_prefix: str, norm_w: np.ndarray, threads: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Route, aggregate and rms_norm of a prefill chunk -> (h_pre, h_post, h_res, branch input [T, C] BF16).

        Rows split over threads (_mhc_split): the batch functions work row by row, so every row keeps the bytes of
        the single call (DESIGN-prefill.md section 43)."""
        rows = stream_f32.shape[0]
        k = _mhc_split(rows, threads)
        eps = self.config.rms_norm_eps
        if k == 1:
            h_pre, h_post, h_res = self._route_mhc_batch(stream_f32, site_prefix)
            return h_pre, h_post, h_res, rms_norm(self._mhc_aggregate_batch(stream_f32, h_pre), norm_w, eps=eps)
        n = self.config.raon_mhc_streams
        # Fill the caches on this thread, so the workers only read them (no concurrent weight loader calls).
        self._mhc_site_constants(site_prefix)
        get_mhc_constants(n)
        h_pre = np.empty((rows, n), dtype=np.float32)
        h_post = np.empty((rows, n), dtype=np.float32)
        h_res = np.empty((rows, n, n), dtype=np.float32)
        branch_in = np.empty((rows, stream_f32.shape[2]), dtype=np.uint16)

        def run(lo: int, hi: int) -> None:
            part = stream_f32[lo:hi]
            p, po, rs = self._route_mhc_batch(part, site_prefix)
            h_pre[lo:hi] = p
            h_post[lo:hi] = po
            h_res[lo:hi] = rs
            branch_in[lo:hi] = rms_norm(self._mhc_aggregate_batch(part, p), norm_w, eps=eps)

        _mhc_run_rows(run, rows, k)
        return h_pre, h_post, h_res, branch_in

    def _mhc_back_batch(
        self,
        stream: np.ndarray,
        stream_f32: np.ndarray,
        branch_out: np.ndarray,
        h_post: np.ndarray,
        h_res: np.ndarray,
        threads: int,
    ) -> None:
        """Mix a prefill chunk into stream (BF16) and stream_f32 in place, rows split like _mhc_front_batch and
        MHC_MIX_ROWS rows at a time inside a range (smaller temporaries - DESIGN-prefill.md section 43)."""
        rows = stream.shape[0]
        k = _mhc_split(rows, threads)
        if k == 1:
            stream[...] = self._mhc_mix_batch(stream_f32, branch_out, h_post, h_res)
            stream_f32[...] = bf16_to_f32(stream)
            return

        def run(lo: int, hi: int) -> None:
            for r in range(lo, hi, MHC_MIX_ROWS):
                e = min(r + MHC_MIX_ROWS, hi)
                stream[r:e] = self._mhc_mix_batch(stream_f32[r:e], branch_out[r:e], h_post[r:e], h_res[r:e])
                stream_f32[r:e] = bf16_to_f32(stream[r:e])

        _mhc_run_rows(run, rows, k)

    def step(self, token_id: int) -> np.ndarray:
        """Decode exactly one token -> logits [vocab_size] in FP32."""
        logits = self.advance(token_id, need_logits=True)
        assert logits is not None
        return logits

    def advance(self, token_id: int, need_logits: bool = True) -> np.ndarray | None:
        """Advance model by one token. If need_logits is False, skip final norm & lm_head."""
        self.last_intermediates.clear()
        embed_w = self.weights.get("model.embed_tokens.weight", to_float=False)
        tok_embed = embed_w[token_id]  # [hidden_size] uint16

        n = self.config.raon_mhc_streams
        # stream: [n, hidden_size]
        stream = np.tile(tok_embed, (n, 1))
        stream_f32 = bf16_to_f32(stream)

        for idx in range(self.config.num_hidden_layers):
            layer_pfx = f"model.layers.{idx}"
            dump_pfx = f"layer_{idx:03d}"

            # Capture streams_in: [1, 1, n, c]
            self.last_intermediates[f"{dump_pfx}.streams_in"] = stream[np.newaxis, np.newaxis, ...]

            # --- Attention Site ---
            attn_mhc_pfx = f"{layer_pfx}.self_attn_raon_mhc"
            h_pre, h_post, h_res = self._route_mhc(stream, attn_mhc_pfx, stream_f32)
            branch_in_raw = self._mhc_aggregate(stream, h_pre, stream_f32)

            layer_constants = self._layer_constants(idx)
            in_norm_w = layer_constants["input_norm"]
            branch_in = rms_norm(branch_in_raw, in_norm_w, eps=self.config.rms_norm_eps)

            # Capture mixer_in: [1, 1, c]
            self.last_intermediates[f"{dump_pfx}.mixer_in"] = branch_in[np.newaxis, np.newaxis, ...]

            if self.config.is_rda_layer(idx):
                branch_out = self._forward_rda(idx, branch_in)
            else:
                branch_out = self._forward_mla(idx, branch_in)

            # Capture mixer_out: [1, 1, c]
            self.last_intermediates[f"{dump_pfx}.mixer_out"] = branch_out[np.newaxis, np.newaxis, ...]

            stream = self._mhc_mix(stream, branch_out, h_post, h_res, stream_f32)
            stream_f32 = bf16_to_f32(stream)

            # --- MLP Site ---
            mlp_mhc_pfx = f"{layer_pfx}.mlp_raon_mhc"
            h_pre, h_post, h_res = self._route_mhc(stream, mlp_mhc_pfx, stream_f32)
            branch_in_raw = self._mhc_aggregate(stream, h_pre, stream_f32)

            post_norm_w = layer_constants["post_norm"]
            branch_in = rms_norm(branch_in_raw, post_norm_w, eps=self.config.rms_norm_eps)

            if self.config.is_moe_layer(idx):
                branch_out = self._forward_moe(idx, branch_in, dump_pfx)
            else:
                branch_out = self._forward_dense_mlp(idx, branch_in)

            stream = self._mhc_mix(stream, branch_out, h_post, h_res, stream_f32)
            stream_f32 = bf16_to_f32(stream)

            # Capture streams_out: [1, 1, n, c]
            self.last_intermediates[f"{dump_pfx}.streams_out"] = stream[np.newaxis, np.newaxis, ...]

        if not need_logits:
            return None

        # Final norm & head
        mean_stream = np.mean(stream_f32, axis=0)  # [c] FP32
        mean_stream_u16 = f32_to_bf16_rtne(mean_stream)
        norm_w = self.weights.get("model.norm.weight", to_float=True)
        final_norm = rms_norm(mean_stream_u16, norm_w, eps=self.config.rms_norm_eps)

        # The reference lm_head consumes and returns BF16; only the public API is FP32.
        logits = self.backend.linear("lm_head.weight", final_norm)
        return bf16_to_f32(logits)

    def _rda_inputs(
        self,
        layer_idx: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        beta_raw: np.ndarray,
        rda_delta: np.ndarray,
        decay_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Short convolutions
        constants = self._layer_constants(layer_idx)
        q, k, v = self._conv_step_qkv(layer_idx, q, k, v, constants["conv_weight"])
        return self._rda_inputs_after_conv(layer_idx, q, k, v, beta_raw, rda_delta, decay_raw)

    def _rda_inputs_after_conv(
        self,
        layer_idx: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        beta_raw: np.ndarray,
        rda_delta: np.ndarray,
        decay_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``_rda_inputs`` after the short convolution: q, k, v are already convolved (DESIGN-prefill.md section 48)."""
        num_heads = self.config.rda_num_heads
        head_dim = self.config.rda_head_dim
        constants = self._layer_constants(layer_idx)

        # Reshape to [heads, head_dim]
        q_h = bf16_to_f32(q).reshape(num_heads, head_dim)
        k_h = bf16_to_f32(k).reshape(num_heads, head_dim)
        v_h = bf16_to_f32(v).reshape(num_heads, head_dim)

        # Reference boundary: normalize in FP32, cast to BF16, then scale q in BF16.
        q_norm = np.linalg.norm(q_h, axis=-1, keepdims=True)
        q_normalized_u16 = f32_to_bf16_rtne(q_h / np.maximum(q_norm, 1e-6))
        q_u16 = f32_to_bf16_rtne(bf16_to_f32(q_normalized_u16) * (head_dim ** -0.5))

        k_norm = np.linalg.norm(k_h, axis=-1, keepdims=True)
        k_h = k_h / np.maximum(k_norm, 1e-6)

        k_u16 = f32_to_bf16_rtne(k_h)

        # Decay computation in FP32
        dt_bias = constants["dt_bias"]
        decay = bf16_to_f32(decay_raw).reshape(num_heads, head_dim) + dt_bias
        scaled = constants["exp_a_log"][:, np.newaxis] * decay
        if self.config.rda_gate_lower_bound is None:
            # -softplus(scaled)
            log_alpha = -np.log1p(np.exp(-np.abs(scaled))) - np.maximum(scaled, 0.0)
        else:
            log_alpha = self.config.rda_gate_lower_bound * (1.0 / (1.0 + np.exp(-scaled)))

        b_f32 = bf16_to_f32(beta_raw)
        delta_f32 = bf16_to_f32(rda_delta)

        # beta = sigmoid(beta_raw)
        beta = 1.0 / (1.0 + np.exp(-b_f32))
        beta_u16 = f32_to_bf16_rtne(beta)

        # ratio = exp(logsigmoid(b + delta) - logsigmoid(b))
        def logsigmoid(x: np.ndarray) -> np.ndarray:
            return -np.log1p(np.exp(-np.abs(x))) - np.maximum(-x, 0.0)

        ratio = np.exp(logsigmoid(b_f32 + delta_f32) - logsigmoid(b_f32))
        v_h = (v_h * ratio[:, np.newaxis])
        v_u16 = f32_to_bf16_rtne(v_h)

        return q_u16, k_u16, v_u16, log_alpha, beta_u16

    def _rda_output(
        self,
        layer_idx: int,
        out_u16: np.ndarray,
        gate: np.ndarray,
    ) -> np.ndarray:
        num_heads = self.config.rda_num_heads
        head_dim = self.config.rda_head_dim
        proj_size = num_heads * head_dim
        constants = self._layer_constants(layer_idx)

        out_h = bf16_to_f32(out_u16).reshape(num_heads, head_dim)

        # Per-head RMSNorm in FP32
        o_norm_w = constants["o_norm"]
        normed = rms_norm(out_h, o_norm_w, eps=self.config.rms_norm_eps, keep_fp32=True)

        # Sigmoid output gate in FP32, then round once to BF16
        gate_f32 = bf16_to_f32(gate).reshape(num_heads, head_dim)
        sig_gate = 1.0 / (1.0 + np.exp(-gate_f32))
        return f32_to_bf16_rtne((normed * sig_gate).reshape(proj_size))

    def _step_rda_core(
        self,
        layer_idx: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gate: np.ndarray,
        beta_raw: np.ndarray,
        rda_delta: np.ndarray,
        decay_raw: np.ndarray,
    ) -> np.ndarray:
        """Token-serial RDA state update and output gating. Returns gated uint16 [proj_size]."""
        q_u16, k_u16, v_u16, log_alpha, beta_u16 = self._rda_inputs(
            layer_idx, q, k, v, beta_raw, rda_delta, decay_raw
        )
        out_u16 = self.backend.rda_update(layer_idx, q_u16, k_u16, v_u16, log_alpha, beta_u16)
        return self._rda_output(layer_idx, out_u16, gate)

    def _rda_tokens_overlapped(
        self,
        idx: int,
        q_batch: np.ndarray,
        k_batch: np.ndarray,
        v_batch: np.ndarray,
        gate_batch: np.ndarray,
        beta_batch: np.ndarray,
        rda_w_batch: np.ndarray,
        f_b_batch: np.ndarray,
    ) -> np.ndarray:
        """Prefill RDA of one chunk in calls of up to RAON_PREFILL_RDA_BLOCK tokens (DESIGN-prefill.md section 36).

        NPU(call b) overlaps the host staging call b+1 and finishing call b-1; conv and RDA inputs are still computed
        token by token in order. Any exception aborts the live calls and leaves this request's conv and RDA state
        undefined: the caller resets the model (serve/engine.py does on every prefill error).
        """
        be = self.backend
        T = q_batch.shape[0]
        gated = [None] * T
        blocks = rda_block_plan(T, self.prefill_rda_block)

        def stage(start: int, size: int) -> Any:
            return be.rda_stage(
                idx,
                [
                    self._rda_inputs(
                        idx, q_batch[t], k_batch[t], v_batch[t], beta_batch[t], rda_w_batch[t], f_b_batch[t]
                    )
                    for t in range(start, start + size)
                ],
            )

        def finish(start: int, ticket: Any) -> None:
            out = be.rda_read(ticket)  # [size, heads, head_dim] BF16
            for j in range(out.shape[0]):
                gated[start + j] = self._rda_output(idx, out[j], gate_batch[start + j])

        self._in_rda_overlap = True
        try:
            cur = stage(*blocks[0])
            be.rda_launch(cur)
            prev = None
            for b in range(len(blocks)):
                nxt = stage(*blocks[b + 1]) if b + 1 < len(blocks) else None  # NPU(b) 와 겹침
                if prev is not None:
                    finish(blocks[b - 1][0], prev)  # NPU(b) 와 겹침
                be.rda_wait(cur)  # NPU(b) 끝 — 상태 슬롯은 여기서 확정
                prev = cur
                if nxt is not None:
                    be.rda_launch(nxt)  # NPU(b+1) 를 바로 시작
                cur = nxt
            finish(blocks[-1][0], prev)
        except BaseException:
            be.rda_abort_all()
            raise
        finally:
            self._in_rda_overlap = False
        return np.stack(gated, axis=0)

    def _forward_rda(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        pfx = f"model.layers.{layer_idx}.self_attn"

        q = self.backend.linear(f"{pfx}.q_proj.weight", hidden)
        k = self.backend.linear(f"{pfx}.k_proj.weight", hidden)
        v = self.backend.linear(f"{pfx}.v_proj.weight", hidden)
        gate = self.backend.linear(f"{pfx}.g_proj.weight", hidden)
        beta_raw = self.backend.linear(f"{pfx}.b_proj.weight", hidden)
        rda_delta = self.backend.linear(f"{pfx}.rda_w_proj.weight", hidden)

        f_a = self.backend.linear(f"{pfx}.f_a_proj.weight", hidden)
        # f_b depends only on f_a and the short convolution only on q, k, v: the backend may run the f_b GEMV on the
        # NPU while the host convolves (DESIGN-prefill.md section 48). The steps are _step_rda_core's, in its order.
        pending = self.backend.linear_start(f"{pfx}.f_b_proj.weight", f_a)
        try:
            constants = self._layer_constants(layer_idx)
            q, k, v = self._conv_step_qkv(layer_idx, q, k, v, constants["conv_weight"])
        except BaseException:
            self.backend.linear_abort(pending)
            raise
        decay_raw = self.backend.linear_finish(pending)

        q_u16, k_u16, v_u16, log_alpha, beta_u16 = self._rda_inputs_after_conv(
            layer_idx, q, k, v, beta_raw, rda_delta, decay_raw
        )
        out_u16 = self.backend.rda_update(layer_idx, q_u16, k_u16, v_u16, log_alpha, beta_u16)
        gated = self._rda_output(layer_idx, out_u16, gate)
        return self.backend.linear(f"{pfx}.o_proj.weight", gated)

    def _step_mla_attend(
        self,
        layer_idx: int,
        q: np.ndarray,
        k_nope: np.ndarray,
        k_pe: np.ndarray,
        value: np.ndarray,
        gate: np.ndarray | None = None,
    ) -> np.ndarray:
        """Token-serial MLA attention computation and output gate for 1 token."""
        heads = self.config.num_attention_heads
        qk_nope = self.config.qk_nope_head_dim
        qk_pe = self.config.qk_rope_head_dim
        q_head_dim = qk_nope + qk_pe

        # Expand k_pe across heads: [heads, qk_pe]
        k_pe_expanded = np.tile(k_pe, (heads, 1))
        key = np.concatenate([k_nope, k_pe_expanded], axis=-1)  # [heads, 192]

        scale = q_head_dim ** -0.5
        context = self.backend.mla_attend(layer_idx, q, key, value, scale=scale)

        if self.config.mla_use_output_gate and gate is not None:
            # Both sigmoid(g_proj) and the product execute in the BF16 activation dtype.
            sig_g = f32_to_bf16_rtne(1.0 / (1.0 + np.exp(-bf16_to_f32(gate))))
            context = f32_to_bf16_rtne(bf16_to_f32(context) * bf16_to_f32(sig_g))

        return context

    def _forward_mla(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        pfx = f"model.layers.{layer_idx}.self_attn"
        heads = self.config.num_attention_heads
        qk_nope = self.config.qk_nope_head_dim
        qk_pe = self.config.qk_rope_head_dim
        v_dim = self.config.v_head_dim
        q_head_dim = qk_nope + qk_pe

        # The latent projections and the output gate depend only on the token: the backend may run their GEMVs back
        # to back now and serve the linear calls below from its cache (DESIGN-prefill.md section 47).
        first = [(f"{pfx}.q_a_proj.weight", hidden)]
        if self.config.mla_use_output_gate:
            first.append((f"{pfx}.g_proj.weight", hidden))
        self.backend.prefetch_linears(tuple(first))

        # Query projection & norm
        q_lat_raw = self.backend.linear(f"{pfx}.q_a_proj.weight", hidden)
        constants = self._layer_constants(layer_idx)
        q_a_norm_w = constants["q_a_norm"]
        q_lat = rms_norm(q_lat_raw, q_a_norm_w, eps=self.config.mla_latent_norm_eps)

        # Key-Value projection with MQA
        kv_full = self.backend.linear(f"{pfx}.kv_a_proj_with_mqa.weight", hidden)
        kv_lora_rank = self.config.kv_lora_rank
        kv_lat_raw = kv_full[:kv_lora_rank]
        k_pe = kv_full[kv_lora_rank:]  # [qk_pe]

        kv_a_norm_w = constants["kv_a_norm"]
        kv_lat = rms_norm(kv_lat_raw, kv_a_norm_w, eps=self.config.mla_latent_norm_eps)

        # q_b and kv_b depend only on the two latents (DESIGN-prefill.md section 47).
        self.backend.prefetch_linears(
            ((f"{pfx}.q_b_proj.weight", q_lat), (f"{pfx}.kv_b_proj.weight", kv_lat))
        )
        q = self.backend.linear(f"{pfx}.q_b_proj.weight", q_lat).reshape(heads, q_head_dim)
        kv = self.backend.linear(f"{pfx}.kv_b_proj.weight", kv_lat).reshape(heads, qk_nope + v_dim)
        k_nope = kv[:, :qk_nope]
        value = kv[:, qk_nope:]

        gate = (
            self.backend.linear(f"{pfx}.g_proj.weight", hidden)
            if self.config.mla_use_output_gate
            else None
        )
        context = self._step_mla_attend(layer_idx, q, k_nope, k_pe, value, gate)
        return self.backend.linear(f"{pfx}.o_proj.weight", context)

    def _forward_dense_mlp(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        pfx = f"model.layers.{layer_idx}.mlp"
        gate = self.backend.linear(f"{pfx}.gate_proj.weight", hidden)
        up = self.backend.linear(f"{pfx}.up_proj.weight", hidden)
        act = situ_glu(
            gate,
            up,
            gate_beta=self.config.activation_situ_beta,
            value_beta=self.config.activation_situ_linear_beta,
        )
        return self.backend.linear(f"{pfx}.down_proj.weight", f32_to_bf16_rtne(act))

    def _route_token(
        self, layer_idx: int, hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute one token's top-k expert IDs, router weights, and logits."""
        hidden_f32 = bf16_to_f32(hidden)
        constants = self._layer_constants(layer_idx)
        router_w = constants["router_weight"]
        bias = constants["router_bias"]

        logits = np.dot(hidden_f32, router_w.T)  # [num_experts]
        scores = 1.0 / (1.0 + np.exp(-logits))

        # Selection using score + e_score_correction_bias
        choice_scores = scores + bias
        top_k = self.config.num_experts_per_token
        sorted_indices = np.argsort(-choice_scores)
        top = sorted_indices[:top_k]

        chosen = scores[top]
        if self.config.moe_renormalize:
            chosen = chosen / (np.sum(chosen) + 1e-20)
        router_weights = (chosen * self.config.routed_scaling_factor).astype(np.float32)

        return top, router_weights, logits

    def _step_moe_router(
        self,
        layer_idx: int,
        hidden: np.ndarray,
        shared_gu: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Route and execute the routed experts for one token."""
        try:
            if shared_gu is None:
                # The shared gate/up depends only on the token: the backend may start it now, so the NPU computes it
                # while the host routes (DESIGN-prefill.md section 46).
                self.backend.moe_prestart_shared(layer_idx, hidden)
            top, router_weights, logits = self._route_token(layer_idx, hidden)

            if shared_gu is None:
                routed_out = self.backend.moe_experts(
                    layer_idx,
                    hidden,
                    top,
                    router_weights,
                    gate_beta=self.config.activation_situ_beta,
                    value_beta=self.config.activation_situ_linear_beta,
                )
            else:
                routed_out = self.backend.moe_experts(
                    layer_idx,
                    hidden,
                    top,
                    router_weights,
                    gate_beta=self.config.activation_situ_beta,
                    value_beta=self.config.activation_situ_linear_beta,
                    shared_gu=shared_gu,
                )
        except BaseException:
            # A prestart, router or MoE failure (or an interrupt) must not leave the prestarted shared run on the NPU.
            self.backend.moe_abort_prestart_shared()
            raise
        return routed_out, top, router_weights, logits

    def _moe_token(
        self,
        layer_idx: int,
        hidden: np.ndarray,
        shared_gate_up: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute router, routed experts, shared experts, and combined output for 1 token."""
        pfx = f"model.layers.{layer_idx}.moe"
        shared_gu = None
        if shared_gate_up is not None:
            gate_row, up_row = shared_gate_up
            shared_gu = np.concatenate([
                np.asarray(gate_row, dtype=np.uint16).reshape(-1),
                np.asarray(up_row, dtype=np.uint16).reshape(-1),
            ])

        routed_out, top, router_weights, logits = self._step_moe_router(
            layer_idx, hidden, shared_gu=shared_gu
        )

        if self.config.num_shared_experts > 0:
            if shared_gate_up is not None:
                sh_gate_row, sh_up_row = shared_gate_up
                sh_gate = np.asarray(sh_gate_row, dtype=np.uint16)
                sh_up = np.asarray(sh_up_row, dtype=np.uint16)
            else:
                sh_gate = self.backend.linear(f"{pfx}.shared_experts.gate_proj.weight", hidden)
                sh_up = self.backend.linear(f"{pfx}.shared_experts.up_proj.weight", hidden)
            sh_act = situ_glu(
                sh_gate,
                sh_up,
                gate_beta=self.config.activation_situ_beta,
                value_beta=self.config.activation_situ_linear_beta,
            )
            shared_out = self.backend.linear(
                f"{pfx}.shared_experts.down_proj.weight", f32_to_bf16_rtne(sh_act)
            )
            comb = bf16_to_f32(routed_out) + bf16_to_f32(shared_out)
            out = f32_to_bf16_rtne(comb)
        else:
            out = routed_out

        return out, top, router_weights, logits

    def _forward_moe(self, layer_idx: int, hidden: np.ndarray, dump_pfx: str) -> np.ndarray:
        out, top, router_weights, logits = self._moe_token(layer_idx, hidden)

        # Record router intermediate outputs
        self.last_intermediates[f"{dump_pfx}.router_logits"] = logits[np.newaxis, ...]
        self.last_intermediates[f"{dump_pfx}.router_top"] = top[np.newaxis, ...]
        self.last_intermediates[f"{dump_pfx}.router_weights"] = router_weights[np.newaxis, ...]

        return out

    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        logits_positions: Collection[int] = (),
        chunk: int = 16,
        record_intermediates: bool = False,
    ) -> dict[int, np.ndarray]:
        """Layer-wise batched prefill for a sequence of tokens.

        Processes token_ids chunk by chunk (up to chunk tokens per batch) across layers.
        State updates for conv, RDA, MLA KV cache appends, and MoE experts remain strictly token-serial;
        MLA attention runs per chunk (RAON_PREFILL_MLA_BATCH).
        Returns a dict mapping selected positions (0-based in token_ids) to FP32 logits.
        """
        self.last_intermediates.clear()
        if not token_ids:
            return {}

        logits_pos_set = set(logits_positions)
        result: dict[int, np.ndarray] = {}
        chunk_size = max(1, int(chunk))
        total_tokens = len(token_ids)
        n = self.config.raon_mhc_streams
        embed_w = self.weights.get("model.embed_tokens.weight", to_float=False)

        streams_in_dict: dict[int, list[np.ndarray]] = {}
        streams_out_dict: dict[int, list[np.ndarray]] = {}
        router_top_dict: dict[int, list[np.ndarray]] = {}
        router_weights_dict: dict[int, list[np.ndarray]] = {}
        if record_intermediates:
            for idx in range(self.config.num_hidden_layers):
                streams_in_dict[idx] = []
                streams_out_dict[idx] = []
                if self.config.is_moe_layer(idx):
                    router_top_dict[idx] = []
                    router_weights_dict[idx] = []

        mhc_batch = self.prefill_mhc_batch
        # Threads for the batch mHC, read once so a bad RAON_PREFILL_MHC_THREADS fails before any work.
        mhc_threads = self.prefill_mhc_threads if mhc_batch else 1
        mla_batch = self.prefill_mla_batch
        for chunk_start in range(0, total_tokens, chunk_size):
            chunk_tokens = token_ids[chunk_start : chunk_start + chunk_size]
            T = len(chunk_tokens)

            # stream: [T, n, c]
            stream = np.stack([np.tile(embed_w[tid], (n, 1)) for tid in chunk_tokens], axis=0)
            stream_f32 = bf16_to_f32(stream)

            for idx in range(self.config.num_hidden_layers):
                layer_pfx = f"model.layers.{idx}"
                dump_pfx = f"layer_{idx:03d}"

                if record_intermediates:
                    streams_in_dict[idx].append(stream.copy())

                # --- Attention Site ---
                attn_mhc_pfx = f"{layer_pfx}.self_attn_raon_mhc"
                layer_constants = self._layer_constants(idx)
                in_norm_w = layer_constants["input_norm"]

                if mhc_batch:
                    h_pre_b, h_post_b, h_res_b, branch_in = self._mhc_front_batch(
                        stream_f32, attn_mhc_pfx, in_norm_w, mhc_threads
                    )
                else:
                    h_pre_list = []
                    h_post_list = []
                    h_res_list = []
                    branch_in_list = []
                    for t in range(T):
                        hp, hpost, hres = self._route_mhc(stream[t], attn_mhc_pfx, stream_f32[t])
                        h_pre_list.append(hp)
                        h_post_list.append(hpost)
                        h_res_list.append(hres)
                        b_raw = self._mhc_aggregate(stream[t], hp, stream_f32[t])
                        b_in = rms_norm(b_raw, in_norm_w, eps=self.config.rms_norm_eps)
                        branch_in_list.append(b_in)

                    branch_in = np.stack(branch_in_list, axis=0)

                if self.config.is_rda_layer(idx):
                    pfx = f"{layer_pfx}.self_attn"
                    q_batch = self.backend.linear_batch(f"{pfx}.q_proj.weight", branch_in)
                    k_batch = self.backend.linear_batch(f"{pfx}.k_proj.weight", branch_in)
                    v_batch = self.backend.linear_batch(f"{pfx}.v_proj.weight", branch_in)
                    gate_batch = self.backend.linear_batch(f"{pfx}.g_proj.weight", branch_in)
                    beta_batch = self.backend.linear_batch(f"{pfx}.b_proj.weight", branch_in)
                    rda_w_batch = self.backend.linear_batch(f"{pfx}.rda_w_proj.weight", branch_in)
                    f_a_batch = self.backend.linear_batch(f"{pfx}.f_a_proj.weight", branch_in)
                    f_b_batch = self.backend.linear_batch(f"{pfx}.f_b_proj.weight", f_a_batch)

                    ready = getattr(self.backend, "rda_overlap_ready", None)
                    if callable(ready) and ready():
                        gated_batch = self._rda_tokens_overlapped(
                            idx,
                            q_batch,
                            k_batch,
                            v_batch,
                            gate_batch,
                            beta_batch,
                            rda_w_batch,
                            f_b_batch,
                        )
                    else:
                        gated_list = []
                        for t in range(T):
                            gated = self._step_rda_core(
                                idx,
                                q_batch[t],
                                k_batch[t],
                                v_batch[t],
                                gate_batch[t],
                                beta_batch[t],
                                rda_w_batch[t],
                                f_b_batch[t],
                            )
                            gated_list.append(gated)
                        gated_batch = np.stack(gated_list, axis=0)
                    branch_out = self.backend.linear_batch(f"{pfx}.o_proj.weight", gated_batch)
                else:
                    pfx = f"{layer_pfx}.self_attn"
                    heads = self.config.num_attention_heads
                    qk_nope = self.config.qk_nope_head_dim
                    qk_pe = self.config.qk_rope_head_dim
                    v_dim = self.config.v_head_dim
                    q_head_dim = qk_nope + qk_pe

                    q_lat_raw = self.backend.linear_batch(f"{pfx}.q_a_proj.weight", branch_in)
                    kv_full = self.backend.linear_batch(f"{pfx}.kv_a_proj_with_mqa.weight", branch_in)
                    gate_batch = (
                        self.backend.linear_batch(f"{pfx}.g_proj.weight", branch_in)
                        if self.config.mla_use_output_gate
                        else None
                    )

                    q_a_norm_w = layer_constants["q_a_norm"]
                    q_lat_batch = np.stack(
                        [
                            rms_norm(q_lat_raw[t], q_a_norm_w, eps=self.config.mla_latent_norm_eps)
                            for t in range(T)
                        ],
                        axis=0,
                    )
                    kv_lora_rank = self.config.kv_lora_rank
                    kv_lat_raw = kv_full[:, :kv_lora_rank]
                    k_pe_batch = kv_full[:, kv_lora_rank:]

                    kv_a_norm_w = layer_constants["kv_a_norm"]
                    kv_lat_batch = np.stack(
                        [
                            rms_norm(kv_lat_raw[t], kv_a_norm_w, eps=self.config.mla_latent_norm_eps)
                            for t in range(T)
                        ],
                        axis=0,
                    )

                    q_batch = self.backend.linear_batch(f"{pfx}.q_b_proj.weight", q_lat_batch).reshape(
                        T, heads, q_head_dim
                    )
                    kv_batch = self.backend.linear_batch(f"{pfx}.kv_b_proj.weight", kv_lat_batch).reshape(
                        T, heads, qk_nope + v_dim
                    )

                    if mla_batch:
                        # _step_mla_attend for the whole chunk: the same keys and scale, one backend call, the same
                        # gate element by element.
                        k_pe_heads = np.broadcast_to(k_pe_batch[:, np.newaxis, :], (T, heads, qk_pe))
                        key_batch = np.concatenate([kv_batch[:, :, :qk_nope], k_pe_heads], axis=-1)
                        context_batch = self.backend.mla_attend_chunk(
                            idx, q_batch, key_batch, kv_batch[:, :, qk_nope:], q_head_dim ** -0.5
                        )
                        if self.config.mla_use_output_gate and gate_batch is not None:
                            sig_g = f32_to_bf16_rtne(1.0 / (1.0 + np.exp(-bf16_to_f32(gate_batch))))
                            context_batch = f32_to_bf16_rtne(bf16_to_f32(context_batch) * bf16_to_f32(sig_g))
                    else:
                        context_list = []
                        for t in range(T):
                            k_nope = kv_batch[t, :, :qk_nope]
                            val = kv_batch[t, :, qk_nope:]
                            gt = gate_batch[t] if gate_batch is not None else None
                            ctx = self._step_mla_attend(
                                idx, q_batch[t], k_nope, k_pe_batch[t], val, gt
                            )
                            context_list.append(ctx)
                        context_batch = np.stack(context_list, axis=0)
                    branch_out = self.backend.linear_batch(f"{pfx}.o_proj.weight", context_batch)

                # Attention mix
                if mhc_batch:
                    self._mhc_back_batch(stream, stream_f32, branch_out, h_post_b, h_res_b, mhc_threads)
                else:
                    for t in range(T):
                        stream[t] = self._mhc_mix(
                            stream[t], branch_out[t], h_post_list[t], h_res_list[t], stream_f32[t]
                        )
                        stream_f32[t] = bf16_to_f32(stream[t])

                # --- MLP Site ---
                mlp_mhc_pfx = f"{layer_pfx}.mlp_raon_mhc"
                post_norm_w = layer_constants["post_norm"]

                if mhc_batch:
                    mlp_h_pre_b, mlp_h_post_b, mlp_h_res_b, mlp_branch_in = self._mhc_front_batch(
                        stream_f32, mlp_mhc_pfx, post_norm_w, mhc_threads
                    )
                else:
                    mlp_h_pre_list = []
                    mlp_h_post_list = []
                    mlp_h_res_list = []
                    mlp_branch_in_list = []
                    for t in range(T):
                        hp, hpost, hres = self._route_mhc(stream[t], mlp_mhc_pfx, stream_f32[t])
                        mlp_h_pre_list.append(hp)
                        mlp_h_post_list.append(hpost)
                        mlp_h_res_list.append(hres)
                        b_raw = self._mhc_aggregate(stream[t], hp, stream_f32[t])
                        b_in = rms_norm(b_raw, post_norm_w, eps=self.config.rms_norm_eps)
                        mlp_branch_in_list.append(b_in)

                    mlp_branch_in = np.stack(mlp_branch_in_list, axis=0)

                if self.config.is_moe_layer(idx):
                    use_shared_precompute = (
                        getattr(self.backend, "prefill_shared_precompute", False)
                        and self.config.num_shared_experts > 0
                    )
                    if use_shared_precompute:
                        pfx_sh = f"{layer_pfx}.moe.shared_experts"
                        sh_gate_all = self.backend.linear_batch(
                            f"{pfx_sh}.gate_proj.weight", mlp_branch_in
                        )
                        sh_up_all = self.backend.linear_batch(
                            f"{pfx_sh}.up_proj.weight", mlp_branch_in
                        )

                    use_grouped = use_shared_precompute and getattr(
                        self.backend, "prefill_grouped_moe", False
                    )
                    top_list: list[np.ndarray] = []
                    rweights_list: list[np.ndarray] = []
                    if use_grouped:
                        for t in range(T):
                            top, r_weights, _ = self._route_token(idx, mlp_branch_in[t])
                            top_list.append(top)
                            rweights_list.append(r_weights)
                        top_all = np.stack(top_list, axis=0)
                        w_all = np.stack(rweights_list, axis=0).astype(np.float32, copy=False)
                        routed, shared_down = self.backend.moe_experts_grouped(
                            idx,
                            mlp_branch_in,
                            top_all,
                            w_all,
                            np.concatenate([sh_gate_all, sh_up_all], axis=1),
                            gate_beta=self.config.activation_situ_beta,
                            value_beta=self.config.activation_situ_linear_beta,
                        )
                        branch_out = f32_to_bf16_rtne(
                            bf16_to_f32(routed) + bf16_to_f32(shared_down)
                        )
                    else:
                        branch_out_list = []
                        for t in range(T):
                            h_t = mlp_branch_in[t]
                            if use_shared_precompute:
                                shared_gate_up = (sh_gate_all[t], sh_up_all[t])
                                out_t, top, r_weights, _ = self._moe_token(
                                    idx, h_t, shared_gate_up=shared_gate_up
                                )
                            else:
                                out_t, top, r_weights, _ = self._moe_token(idx, h_t)
                            branch_out_list.append(out_t)
                            if record_intermediates:
                                top_list.append(top)
                                rweights_list.append(r_weights)
                        branch_out = np.stack(branch_out_list, axis=0)
                    if record_intermediates:
                        router_top_dict[idx].append(np.stack(top_list, axis=0))
                        router_weights_dict[idx].append(np.stack(rweights_list, axis=0))
                else:
                    pfx = f"{layer_pfx}.mlp"
                    gate = self.backend.linear_batch(f"{pfx}.gate_proj.weight", mlp_branch_in)
                    up = self.backend.linear_batch(f"{pfx}.up_proj.weight", mlp_branch_in)
                    act = situ_glu(
                        gate,
                        up,
                        gate_beta=self.config.activation_situ_beta,
                        value_beta=self.config.activation_situ_linear_beta,
                    )
                    branch_out = self.backend.linear_batch(
                        f"{pfx}.down_proj.weight", f32_to_bf16_rtne(act)
                    )

                # MLP mix
                if mhc_batch:
                    self._mhc_back_batch(stream, stream_f32, branch_out, mlp_h_post_b, mlp_h_res_b, mhc_threads)
                else:
                    for t in range(T):
                        stream[t] = self._mhc_mix(
                            stream[t], branch_out[t], mlp_h_post_list[t], mlp_h_res_list[t], stream_f32[t]
                        )
                        stream_f32[t] = bf16_to_f32(stream[t])

                if record_intermediates:
                    streams_out_dict[idx].append(stream.copy())

            # Final norm & head for requested positions within this chunk
            norm_w = self.weights.get("model.norm.weight", to_float=True)
            for t in range(T):
                global_pos = chunk_start + t
                if global_pos in logits_pos_set:
                    mean_stream = np.mean(stream_f32[t], axis=0)
                    mean_stream_u16 = f32_to_bf16_rtne(mean_stream)
                    final_norm = rms_norm(mean_stream_u16, norm_w, eps=self.config.rms_norm_eps)
                    logits = self.backend.linear("lm_head.weight", final_norm)
                    result[global_pos] = bf16_to_f32(logits)

        if record_intermediates:
            for idx in range(self.config.num_hidden_layers):
                dump_pfx = f"layer_{idx:03d}"
                if streams_in_dict[idx]:
                    self.last_intermediates[f"{dump_pfx}.streams_in"] = np.concatenate(
                        streams_in_dict[idx], axis=0
                    )
                if streams_out_dict[idx]:
                    self.last_intermediates[f"{dump_pfx}.streams_out"] = np.concatenate(
                        streams_out_dict[idx], axis=0
                    )
                if self.config.is_moe_layer(idx):
                    if router_top_dict[idx]:
                        self.last_intermediates[f"{dump_pfx}.router_top"] = np.concatenate(
                            router_top_dict[idx], axis=0
                        )
                    if router_weights_dict[idx]:
                        self.last_intermediates[f"{dump_pfx}.router_weights"] = np.concatenate(
                            router_weights_dict[idx], axis=0
                        )

        return result
