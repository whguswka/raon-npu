# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from KRAFTON Raon-1-mini-preview, modeling_raon1_mini.py (SiTU-GLU), Apache-2.0.
# Modified by the raon-npu authors, 2026.
"""Execution backend interface and reference NumPy backend.

Backend contract
----------------
``linear`` accepts a final-dimension activation and returns the same shape with
the weight's output dimension substituted. Multi-dimensional ``[..., K]`` input
is accepted: snapshot latent-cache expansion passes ``[T, K]`` (uint16 BF16
token latents) and relies on the same per-row FP32 GEMM + RTNE semantics as the
1-token path. BF16 activations are represented by ``uint16`` bit patterns and
produce BF16 ``uint16`` output; FP32 input produces FP32 output. ``rda_update`` consumes ``q/k/v`` BF16 arrays shaped ``[H, K]``,
``[H, K]``, ``[H, V]``, FP32 ``log_alpha [H, K]``, and BF16 ``beta [H]``. It
updates the backend-owned FP32 recurrent state ``S [H, V, K]`` during the call
and returns BF16 ``[H, V]``. The model, not the backend, advances the RDA raw
q/k/v convolution history before this call.

``mla_attend`` consumes current-token BF16 ``q/k [H, QK]`` and ``v [H, V]``.
It appends k/v to the backend-owned BF16 caches ``[H, T, QK]`` and
``[H, T, V]`` before computing and returning BF16 flattened context ``[H*V]``.
``mla_attend_chunk`` does the same for ``T`` consecutive tokens at once (``q/k [T, H, QK]``, ``v [T, H, V]`` ->
``[T, H*V]``), byte-identical to ``T`` ``mla_attend`` calls.
``set_*`` replaces state during snapshot restore (accepting unbatched ``[H, ...]``
or leading-batch ``[1, H, ...]`` arrays); ``get_*`` returns a leading batch
dimension ``[1, H, ...]`` without advancing it. ``reset`` clears only backend-owned
RDA and MLA state. ``moe_experts`` consumes BF16 ``x [C]``, expert IDs, FP32
weights ``[top_k]``, and the SiTU activation betas ``gate_beta``/``value_beta``
(the model passes the config's ``activation_situ_beta``/
``activation_situ_linear_beta``; the defaults are the released-checkpoint
4.0/25.0 pair for callers that do not pass config betas), and returns a BF16
``[C]`` weighted sum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import numpy as np

from .weights import Weights, bf16_to_f32, f32_to_bf16_rtne


def situ_glu(
    gate: np.ndarray,
    value: np.ndarray,
    gate_beta: float = 4.0,
    value_beta: float = 25.0,
) -> np.ndarray:
    """SiTU-GLU in FP32: gate_beta*tanh(g/gate_beta)*sigmoid(g) * value_beta*tanh(v/value_beta)."""
    # g/v are only read by out-of-place ufuncs, so float32 inputs may alias.
    g = bf16_to_f32(gate) if gate.dtype == np.uint16 else gate.astype(np.float32, copy=False)
    v = bf16_to_f32(value) if value.dtype == np.uint16 else value.astype(np.float32, copy=False)

    # Sigmoid(g) = 1 / (1 + exp(-g))
    sig_g = 1.0 / (1.0 + np.exp(-g))
    term1 = gate_beta * np.tanh(g / gate_beta) * sig_g
    term2 = value_beta * np.tanh(v / value_beta)
    return term1 * term2


class Backend(ABC):
    """Abstract base class for Raon computation backends."""

    prefill_shared_precompute: bool = False
    prefill_grouped_moe: bool = False

    def linear_batch(self, name: str, x: np.ndarray) -> np.ndarray:
        """x: [T, K] uint16(bf16 비트), T>=1 -> [T, M] uint16.
        np.stack([self.linear(name, x[t]) for t in range(T)]) 와 바이트 단위로 같아야 한다."""
        return np.stack([self.linear(name, x[t]) for t in range(x.shape[0])])

    @abstractmethod
    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        """Matrix multiplication: y = x @ W^T (+ bias if present).

        Accepts multi-dimensional ``[..., K]`` input; snapshot latent-cache
        expansion passes uint16 BF16 ``[T, K]`` token latents.
        """
        pass

    @abstractmethod
    def rda_update(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        log_alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """V-first delta rule recurrence for 1 decode token."""
        pass

    @abstractmethod
    def moe_experts(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
        shared_gu: np.ndarray | None = None,
    ) -> np.ndarray:
        """Weighted sum of routed MoE experts for 1 token.

        ``gate_beta``/``value_beta`` are the SiTU activation betas; the model
        passes the config's ``activation_situ_beta``/``activation_situ_linear_beta``.
        The defaults are the released-checkpoint 4.0/25.0 pair.
        ``shared_gu``: optional precomputed uint16 BF16 [2176] gate+up array
        (gate [:1088], up [1088:]). When provided, skipping separate shared gate+up GEMV.
        """
        pass

    def moe_prestart_shared(self, layer: int, x: np.ndarray) -> None:
        """Start one token's shared-expert gate/up before its router runs.

        Backends that overlap NPU work with the host router (NpuBackend's decode direct path, DESIGN-prefill.md
        section 46) override this; the result of ``moe_experts`` must not depend on whether it was called.
        """
        return None

    def moe_abort_prestart_shared(self) -> None:
        """Drop work started by moe_prestart_shared when the token does not finish its MoE step. No-op here."""
        return None

    def prefetch_linears(self, items: tuple[tuple[str, np.ndarray], ...]) -> None:
        """Hint that ``linear(name, x)`` follows for each ``(name, x)`` pair; the pairs do not depend on each other.

        Backends that can run them back to back (NpuBackend's decode direct path, DESIGN-prefill.md section 47)
        override this; the results of the following ``linear`` calls must not depend on whether it was called.
        """
        return None

    def linear_start(self, name: str, x: np.ndarray) -> object:
        """Begin ``linear(name, x)``; ``linear_finish`` returns its result and ``linear_abort`` drops it.

        Backends that can run the GEMV while the caller does host work (NpuBackend's decode direct path,
        DESIGN-prefill.md section 48) override these; ``linear_finish`` must return exactly ``linear(name, x)``.
        Every start is ended by one ``linear_finish`` or one ``linear_abort``, and ``x`` must not change in between.
        """
        return (name, x)

    def linear_finish(self, pending: object) -> np.ndarray:
        """Result of a ``linear_start``: here ``linear`` itself."""
        name, x = pending
        return self.linear(name, x)

    def linear_abort(self, pending: object) -> None:
        """Drop a ``linear_start`` whose result is not needed. No-op here."""
        return None

    @abstractmethod
    def mla_attend(
        self,
        layer: int,
        q: np.ndarray,
        k_cur: np.ndarray,
        v_cur: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """MLA attention computation and KV-cache update for 1 token."""
        pass

    @abstractmethod
    def mla_attend_chunk(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """MLA attention and KV-cache update for T consecutive tokens: q/k [T, H, QK], v [T, H, V] -> [T, H*V]."""
        pass

    @abstractmethod
    def set_rda_state(self, layer: int, state: np.ndarray) -> None:
        """Set RDA recurrent state from ``[H, V, K]`` or ``[1, H, V, K]`` (FP32)."""
        pass

    @abstractmethod
    def get_rda_state(self, layer: int) -> np.ndarray | None:
        """Get RDA recurrent state as ``[1, H, V, K]`` (FP32), or ``None`` if unset."""
        pass

    @abstractmethod
    def set_mla_cache(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        """Set MLA expanded KV cache from ``key [H, T, QK]`` or ``[1, H, T, QK]`` and ``value [H, T, V]`` or ``[1, H, T, V]`` (uint16 BF16)."""
        pass

    @abstractmethod
    def get_mla_cache(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get MLA expanded KV cache as ``(key [1, H, T, QK], value [1, H, T, V])`` (uint16 BF16), or ``(None, None)`` if unset."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recurrent and cache states."""
        pass


# Head-group threads for the host MLA attention (numpy's einsum and ufuncs release the GIL); threads start on first use.
_MLA_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 8, thread_name_prefix="raon-mla")
_MLA_TB = 64  # positions per block of the [t, q] <-> [q, t] transposes


def mla_threads(n: int, nq: int = 1) -> int:
    """Head-group threads for MLA attention of nq queries over n cached positions: RAON_MLA_THREADS as a count
    (1 = one thread, query by query); anything else (default "auto") follows the laptop measurements in
    DESIGN-prefill.md section 30."""
    value = os.environ.get("RAON_MLA_THREADS", "auto").strip()
    if value.isdigit():
        return max(1, int(value))
    if n < (2048 if nq == 1 else 256):
        return 1
    return 4 if nq == 1 or n < 512 else 8


def _mla_map(fn, groups) -> None:
    """fn over head groups: one group on this thread, more on the pool."""
    if len(groups) == 1:
        fn(groups[0])
    else:
        list(_MLA_POOL.map(fn, groups))


def _mla_attend_queries(q_f32, k_f32, v_f32, scale, out, hs) -> None:
    """mla_attend's arithmetic, one query at a time, for heads hs: queries q [T, H, QK] at the last T of the n cached
    positions (k/v [H, n, QK/V]) -> out[:, hs] ([T, H, V])."""
    nq, n = q_f32.shape[0], k_f32.shape[1]
    for i in range(nq):
        t = n - nq + i + 1
        scores = np.einsum("hd,htd->ht", q_f32[i, hs], k_f32[hs, :t]) * scale
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out[i, hs] = np.einsum("ht,htd->hd", attn, v_f32[hs, :t])


def _mla_attend_chunk_rows(qt, k_f32, v_f32, scale, out, hs) -> None:
    """Heads hs of a chunk: queries qt [H, T, QK] at the last T of the n cached positions (k/v [H, n, QK/V]) ->
    out[:, hs] ([T, H, V]), byte-identical to _mla_attend_queries for a finite V.

    Per head, 2-D einsums in [t, q] order run numpy's inner kernels of the per-token einsums (the dot product over QK
    for each score, the multiply-add over V for the context) in the same order per value and read each K/V row once
    for all queries. Each query's softmax runs on its own rows with the per-token shapes. Later positions get weight
    zero: einsum multiplies and adds in separate roundings, so a sum started at +0.0 never becomes -0.0 and adding
    0 * v leaves it unchanged - except 0 * inf = nan, which mla_context keeps off this path."""
    nq, n = qt.shape[1], k_f32.shape[1]
    p0 = n - nq
    heads = range(qt.shape[0])[hs]
    s_tq = np.empty((len(heads), n, nq), np.float32)
    for j, h in enumerate(heads):
        np.einsum("qd,td->tq", qt[h], k_f32[h], out=s_tq[j])
    s = np.empty((len(heads), nq, n), np.float32)
    for t0 in range(0, n, _MLA_TB):
        s[:, :, t0 : t0 + _MLA_TB] = s_tq[:, t0 : t0 + _MLA_TB].transpose(0, 2, 1)
    for i in range(nq):
        t = p0 + i + 1
        scores = s[:, i, :t] * scale
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        s[:, i, :t] = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        s[:, i, t:] = 0.0
    for t0 in range(0, n, _MLA_TB):
        s_tq[:, t0 : t0 + _MLA_TB] = s[:, :, t0 : t0 + _MLA_TB].transpose(0, 2, 1)
    for j, h in enumerate(heads):
        np.einsum("tq,td->qd", s_tq[j], v_f32[h], out=out[:, h])


def mla_context(q_f32: np.ndarray, k_f32: np.ndarray, v_f32: np.ndarray, scale: float) -> np.ndarray:
    """FP32 MLA context [T, H, V] of queries q [T, H, QK] at the last T of the n cached positions (k/v [H, n, QK/V]),
    byte-identical to mla_attend's arithmetic query by query; head groups run on mla_threads(n, T) threads. More than
    one thread and query take the chunk path when V at the chunk's positions is finite."""
    nq, h = q_f32.shape[:2]
    n = k_f32.shape[1]
    g = mla_threads(n, nq)
    out = np.empty((nq, h, v_f32.shape[2]), np.float32)
    groups = [slice(int(a[0]), int(a[-1]) + 1) for a in np.array_split(np.arange(h), min(g, h))]
    if g > 1 and nq > 1 and np.isfinite(v_f32[:, n - nq :]).all():
        qt = np.ascontiguousarray(q_f32.transpose(1, 0, 2))
        _mla_map(partial(_mla_attend_chunk_rows, qt, k_f32, v_f32, scale, out), groups)
    else:
        _mla_map(partial(_mla_attend_queries, q_f32, k_f32, v_f32, scale, out), groups)
    return out


class _MlaLayerCache:
    """Preallocated growing buffer for MLA KV cache (uint16 and FP32 mirror)."""

    def __init__(self, h: int, dk: int, dv: int, init_cap: int):
        self.length = 0
        self.capacity = max(1, init_cap)
        self.k_u16 = np.empty((h, self.capacity, dk), dtype=np.uint16)
        self.v_u16 = np.empty((h, self.capacity, dv), dtype=np.uint16)
        self.k_f32 = np.empty((h, self.capacity, dk), dtype=np.float32)
        self.v_f32 = np.empty((h, self.capacity, dv), dtype=np.float32)

    @classmethod
    def from_existing(cls, k: np.ndarray, v: np.ndarray) -> "_MlaLayerCache":
        """Initialize cache from existing [H, T, DK] and [H, T, DV] arrays."""
        h, t, dk = k.shape
        _, _, dv = v.shape
        margin = max(256, t // 4)
        cap = t + margin
        cache = cls(h, dk, dv, cap)
        if t > 0:
            cache.k_u16[:, :t, :] = k
            cache.v_u16[:, :t, :] = v
            cache.k_f32[:, :t, :] = bf16_to_f32(k)
            cache.v_f32[:, :t, :] = bf16_to_f32(v)
            cache.length = t
        return cache

    def _grow(self) -> None:
        delta = max(256, self.capacity // 4)
        new_cap = self.capacity + delta
        h, _, dk = self.k_u16.shape
        _, _, dv = self.v_u16.shape

        new_k_u16 = np.empty((h, new_cap, dk), dtype=np.uint16)
        new_v_u16 = np.empty((h, new_cap, dv), dtype=np.uint16)
        new_k_f32 = np.empty((h, new_cap, dk), dtype=np.float32)
        new_v_f32 = np.empty((h, new_cap, dv), dtype=np.float32)

        t = self.length
        if t > 0:
            new_k_u16[:, :t, :] = self.k_u16[:, :t, :]
            new_v_u16[:, :t, :] = self.v_u16[:, :t, :]
            new_k_f32[:, :t, :] = self.k_f32[:, :t, :]
            new_v_f32[:, :t, :] = self.v_f32[:, :t, :]

        self.capacity = new_cap
        self.k_u16 = new_k_u16
        self.v_u16 = new_v_u16
        self.k_f32 = new_k_f32
        self.v_f32 = new_v_f32

    def append(self, k_u16: np.ndarray, v_u16: np.ndarray) -> None:
        """Append one token's k and v ([H, DK] and [H, DV]) to the preallocated buffers."""
        if self.length >= self.capacity:
            self._grow()
        idx = self.length
        self.k_u16[:, idx, :] = k_u16
        self.v_u16[:, idx, :] = v_u16
        self.k_f32[:, idx, :] = bf16_to_f32(k_u16)
        self.v_f32[:, idx, :] = bf16_to_f32(v_u16)
        self.length += 1


class NumpyBackend(Backend):
    """Pure NumPy reference computation backend."""

    def __init__(self, weights: Weights):
        self.weights = weights
        self.prefill_shared_precompute: bool = False
        self.rda_states: dict[int, np.ndarray] = {}  # layer -> float32 [H, V, K]
        self.mla_key_cache: dict[int, np.ndarray] = {}  # layer -> uint16 [H, T, 192] (views)
        self.mla_value_cache: dict[int, np.ndarray] = {}  # layer -> uint16 [H, T, 128] (views)
        self._mla_caches: dict[int, _MlaLayerCache] = {}

    def reset(self) -> None:
        self.rda_states.clear()
        self.mla_key_cache.clear()
        self.mla_value_cache.clear()
        self._mla_caches.clear()

    def set_rda_state(self, layer: int, state: np.ndarray) -> None:
        # Squeeze batch dimension if present: [1, H, V, K] -> [H, V, K]
        st = state.squeeze(0) if state.ndim == 4 else state
        self.rda_states[layer] = np.ascontiguousarray(st, dtype=np.float32)

    def get_rda_state(self, layer: int) -> np.ndarray | None:
        st = self.rda_states.get(layer)
        if st is None:
            return None
        return st[np.newaxis, ...]  # [1, H, V, K]

    def set_mla_cache(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        k = key.squeeze(0) if key.ndim == 4 else key
        v = value.squeeze(0) if value.ndim == 4 else value
        k_u16 = np.ascontiguousarray(k, dtype=np.uint16)
        v_u16 = np.ascontiguousarray(v, dtype=np.uint16)
        cache = _MlaLayerCache.from_existing(k_u16, v_u16)
        self._mla_caches[layer] = cache
        t = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :t, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :t, :]

    def get_mla_cache(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        cache = self._mla_caches.get(layer)
        if cache is None or cache.length == 0:
            return None, None
        t = cache.length
        return cache.k_u16[:, :t, :][np.newaxis, ...], cache.v_u16[:, :t, :][np.newaxis, ...]

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        w_f32 = self.weights.get(name, to_float=True)
        is_x_f32 = (x.dtype == np.float32)
        x_f32 = x if is_x_f32 else bf16_to_f32(x)

        # y = x @ W.T
        y_f32 = np.dot(x_f32, w_f32.T)

        bias_name = name[:-6] + "bias" if name.endswith("weight") else name + ".bias"
        if bias_name in self.weights:
            b_f32 = self.weights.get(bias_name, to_float=True)
            y_f32 = y_f32 + b_f32

        if is_x_f32:
            return y_f32.astype(np.float32)
        return f32_to_bf16_rtne(y_f32)

    def rda_update(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        log_alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """V-first delta rule recurrence for 1 decode token (token serial, FP32 state)."""
        # q: [H, K], k: [H, K], v: [H, V]
        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        k_f32 = bf16_to_f32(k) if k.dtype == np.uint16 else k.astype(np.float32)
        v_f32 = bf16_to_f32(v) if v.dtype == np.uint16 else v.astype(np.float32)
        log_alpha_f32 = log_alpha.astype(np.float32)
        beta_f32 = bf16_to_f32(beta) if beta.dtype == np.uint16 else beta.astype(np.float32)

        H, key_dim = k_f32.shape[-2], k_f32.shape[-1]
        val_dim = v_f32.shape[-1]

        state = self.rda_states.get(layer)
        if state is None:
            state = np.zeros((H, val_dim, key_dim), dtype=np.float32)
        else:
            state = state.astype(np.float32)

        # 1. Decay state: state * exp(log_alpha) [H, V, K]
        # In PyTorch: state * torch.exp(log_alpha[:, t]).unsqueeze(-2)
        decay = np.exp(log_alpha_f32)[:, np.newaxis, :]
        state = state * decay

        # 2. Predicted: einsum('hvk, hk -> hv', state, k)
        predicted = np.einsum("hvk,hk->hv", state, k_f32)

        # 3. Error: v - predicted
        error = v_f32 - predicted

        # 4. State update: state + beta * (error (x) k)
        # beta is [H] -> beta[:, None, None]
        update = np.einsum("hv,hk->hvk", error, k_f32)
        state = state + beta_f32[:, np.newaxis, np.newaxis] * update
        self.rda_states[layer] = state

        # 5. Output: einsum('hvk, hk -> hv', state, q)
        out_f32 = np.einsum("hvk,hk->hv", state, q_f32)
        return f32_to_bf16_rtne(out_f32)

    def moe_experts(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
        shared_gu: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the weighted sum of 6 routed experts for 1 decode token.

        The SiTU betas come from the caller (the model passes the config's
        ``activation_situ_beta``/``activation_situ_linear_beta``); the defaults
        are the released-checkpoint 4.0/25.0 pair.

        ``shared_gu`` is accepted for signature compatibility with Backend and
        ignored here because NumpyBackend evaluates shared experts separately
        inside _moe_token.
        """
        # x is [C] (uint16)
        dim = x.shape[-1]
        weighted_out = np.zeros(dim, dtype=np.float32)

        # The dequantized reference path visits selected experts by module ID,
        # irrespective of router top-slot order, and accumulates in FP32.
        selected = sorted(
            ((int(expert_id), float(weight)) for expert_id, weight in zip(top_ids, top_weights)),
            key=lambda item: item[0],
        )
        for expert_id, weight in selected:
            gate_name = f"model.layers.{layer}.moe.experts.{expert_id}.gate_proj.weight"
            up_name = f"model.layers.{layer}.moe.experts.{expert_id}.up_proj.weight"
            down_name = f"model.layers.{layer}.moe.experts.{expert_id}.down_proj.weight"

            gate_out = self.linear(gate_name, x)  # uint16
            up_out = self.linear(up_name, x)  # uint16

            act_f32 = situ_glu(gate_out, up_out, gate_beta=gate_beta, value_beta=value_beta)
            act_u16 = f32_to_bf16_rtne(act_f32)

            expert_out = self.linear(down_name, act_u16)  # uint16
            weighted_out += bf16_to_f32(expert_out) * float(weight)

        return f32_to_bf16_rtne(weighted_out)

    def mla_attend(
        self,
        layer: int,
        q: np.ndarray,
        k_cur: np.ndarray,
        v_cur: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """MLA attention computation and cache update for 1 token."""
        # q: [H, 192], k_cur: [H, 192], v_cur: [H, 128]
        # Store in cache as uint16
        k_cur_u16 = f32_to_bf16_rtne(k_cur) if k_cur.dtype == np.float32 else k_cur
        v_cur_u16 = f32_to_bf16_rtne(v_cur) if v_cur.dtype == np.float32 else v_cur
        if k_cur_u16.ndim == 3:
            k_cur_u16 = k_cur_u16.squeeze(1)
        if v_cur_u16.ndim == 3:
            v_cur_u16 = v_cur_u16.squeeze(1)

        cache = self._mla_caches.get(layer)
        if cache is None:
            h, dk = k_cur_u16.shape[0], k_cur_u16.shape[-1]
            _, dv = v_cur_u16.shape[0], v_cur_u16.shape[-1]
            margin = max(256, 1 // 4)
            cache = _MlaLayerCache(h, dk, dv, 1 + margin)
            self._mla_caches[layer] = cache

        cache.append(k_cur_u16, v_cur_u16)
        t = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :t, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :t, :]

        # Attend in FP32
        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        k_f32 = cache.k_f32[:, :t, :]
        v_f32 = cache.v_f32[:, :t, :]

        # Context [H, 128] per head group (mla_context, RAON_MLA_THREADS)
        context_f32 = mla_context(q_f32[np.newaxis], k_f32, v_f32, scale)[0]
        flat_context = context_f32.reshape(-1)  # [H * 128]
        return f32_to_bf16_rtne(flat_context)

    def mla_attend_chunk(
        self,
        layer: int,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """mla_attend for T consecutive tokens at once (q/k [T, H, QK], v [T, H, V]) -> [T, H*V] BF16 uint16.

        The cache gets the same appends token by token; mla_context gives the bytes of T mla_attend calls."""
        k_u16 = f32_to_bf16_rtne(k) if k.dtype == np.float32 else k
        v_u16 = f32_to_bf16_rtne(v) if v.dtype == np.float32 else v
        cache = self._mla_caches.get(layer)
        if cache is None:  # the capacity mla_attend starts with
            cache = _MlaLayerCache(k_u16.shape[1], k_u16.shape[2], v_u16.shape[2], 1 + max(256, 1 // 4))
            self._mla_caches[layer] = cache
        for i in range(k_u16.shape[0]):
            cache.append(k_u16[i], v_u16[i])
        n = cache.length
        self.mla_key_cache[layer] = cache.k_u16[:, :n, :]
        self.mla_value_cache[layer] = cache.v_u16[:, :n, :]

        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        context_f32 = mla_context(q_f32, cache.k_f32[:, :n, :], cache.v_f32[:, :n, :], scale)
        return f32_to_bf16_rtne(context_f32.reshape(k_u16.shape[0], -1))
