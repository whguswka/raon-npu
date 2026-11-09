# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for layer-wise prefill byte parity against token-serial advance."""

from __future__ import annotations

import numpy as np
import pytest

from bench.prefill_replay.harness import run_prefill_replay
from host.backend import NumpyBackend, situ_glu
from host.raon import RaonModel, rms_norm
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.weights import Weights, f32_to_bf16_rtne, bf16_to_f32

_ = tiny_raon_model


class NpuCoupledMockBackend(NumpyBackend):
    """NumpyBackend mocking NPU hardware shared-expert cache coupling rules.

    Coupling rules:
    (가) moe_experts(layer, x, ...) stores the token's shared expert activation bits
         into a 1-slot-per-layer cache (_cached_shared_act).
    (나) linear(shared_experts.down_proj, x) checks that input activation bits match
         the layer's 1-slot cache; raises ValueError if mismatched.
    (다) linear(shared_experts.gate_proj/up_proj, x) counts instances where x is not
         the exact same object passed to moe_experts.
    """

    prefill_grouped_moe = True

    def __init__(self, weights: Weights):
        super().__init__(weights)
        # Existing 2b-1 tests opt into shared precompute without implicitly
        # changing their token-serial MoE path; grouped tests enable this.
        self.prefill_grouped_moe = False
        self._cached_shared_act: dict[int, np.ndarray] = {}
        self._last_moe_x_id: dict[int, int] = {}
        self.shared_gu_obj_mismatches: int = 0
        self.shared_down_calls: int = 0
        self.shared_gemv_calls: int = 0
        self.shared_gu_batch_calls: int = 0
        self.grouped_moe_calls: int = 0
        self.token_moe_calls: int = 0
        self._batch_bundle_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        super().reset()
        self._cached_shared_act.clear()
        self._last_moe_x_id.clear()
        self._batch_bundle_cache.clear()
        self.shared_gu_obj_mismatches = 0
        self.shared_down_calls = 0
        self.shared_gemv_calls = 0
        self.shared_gu_batch_calls = 0
        self.grouped_moe_calls = 0
        self.token_moe_calls = 0

    def linear_batch(self, name: str, x: np.ndarray) -> np.ndarray:
        if "moe.shared_experts." in name and ("gate_proj" in name or "up_proj" in name):
            parts = name.split(".")
            layer_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            bundle_key = f"L{layer_idx:02d}/shared_gu"
            gate_w_name = f"model.layers.{layer_idx}.moe.shared_experts.gate_proj.weight"
            up_w_name = f"model.layers.{layer_idx}.moe.shared_experts.up_proj.weight"

            cached = self._batch_bundle_cache.get(bundle_key)
            if cached is not None and cached[0] is x:
                y_full = cached[1]
            else:
                self.shared_gu_batch_calls += 1
                gate_batch = np.stack([NumpyBackend.linear(self, gate_w_name, x[t]) for t in range(x.shape[0])])
                up_batch = np.stack([NumpyBackend.linear(self, up_w_name, x[t]) for t in range(x.shape[0])])
                y_full = np.concatenate([gate_batch, up_batch], axis=-1)
                self._batch_bundle_cache[bundle_key] = (x, y_full)

            half = y_full.shape[-1] // 2
            if "gate_proj" in name:
                return y_full[:, :half].copy()
            elif "up_proj" in name:
                return y_full[:, half:].copy()
        return super().linear_batch(name, x)

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
        self.token_moe_calls += 1
        pfx = f"model.layers.{layer}.moe.shared_experts"
        gate_w_name = f"{pfx}.gate_proj.weight"
        up_w_name = f"{pfx}.up_proj.weight"
        if shared_gu is not None:
            if not isinstance(shared_gu, np.ndarray) or shared_gu.dtype != np.uint16 or shared_gu.ndim != 1:
                raise ValueError("shared_gu must be a 1D uint16 ndarray")
            if gate_w_name in self.weights and up_w_name in self.weights:
                ref_gate = super().linear(gate_w_name, x)
                ref_up = super().linear(up_w_name, x)
                ref_gu = np.concatenate([ref_gate, ref_up])
                if not np.array_equal(shared_gu, ref_gu):
                    raise ValueError(
                        f"shared_gu mismatch in moe_experts at layer {layer}: "
                        "supplied shared_gu bits do not match reference calculation"
                    )
            half = len(shared_gu) // 2
            sh_gate = shared_gu[:half]
            sh_up = shared_gu[half:]
            sh_act = situ_glu(sh_gate, sh_up, gate_beta=gate_beta, value_beta=value_beta)
            sh_act_u16 = f32_to_bf16_rtne(sh_act)
            self._cached_shared_act[layer] = sh_act_u16.reshape(-1)
        else:
            if gate_w_name in self.weights and up_w_name in self.weights:
                self.shared_gemv_calls += 1
                sh_gate = super().linear(gate_w_name, x)
                sh_up = super().linear(up_w_name, x)
                sh_act = situ_glu(sh_gate, sh_up, gate_beta=gate_beta, value_beta=value_beta)
                sh_act_u16 = f32_to_bf16_rtne(sh_act)
                self._cached_shared_act[layer] = sh_act_u16.reshape(-1)

        self._last_moe_x_id[layer] = id(x)
        if shared_gu is None:
            return super().moe_experts(
                layer,
                x,
                top_ids,
                top_weights,
                gate_beta=gate_beta,
                value_beta=value_beta,
            )
        return super().moe_experts(
            layer,
            x,
            top_ids,
            top_weights,
            gate_beta=gate_beta,
            value_beta=value_beta,
            shared_gu=shared_gu,
        )

    def moe_experts_grouped(
        self,
        layer: int,
        x: np.ndarray,
        top_ids: np.ndarray,
        top_weights: np.ndarray,
        shared_gu: np.ndarray,
        gate_beta: float = 4.0,
        value_beta: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small-model grouped contract built from token-serial NumPy operations."""
        self.grouped_moe_calls += 1
        self._cached_shared_act.pop(layer, None)
        routed_rows = []
        shared_rows = []
        down_name = f"model.layers.{layer}.moe.shared_experts.down_proj.weight"
        for token in range(x.shape[0]):
            routed_rows.append(
                NumpyBackend.moe_experts(
                    self,
                    layer,
                    x[token],
                    top_ids[token],
                    top_weights[token],
                    gate_beta=gate_beta,
                    value_beta=value_beta,
                    shared_gu=shared_gu[token],
                )
            )
            half = shared_gu.shape[1] // 2
            shared_act = situ_glu(
                shared_gu[token, :half],
                shared_gu[token, half:],
                gate_beta=gate_beta,
                value_beta=value_beta,
            )
            shared_rows.append(
                NumpyBackend.linear(self, down_name, f32_to_bf16_rtne(shared_act))
            )
        return np.stack(routed_rows), np.stack(shared_rows)

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        parts = name.split(".")
        layer_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

        if layer_idx is not None and "moe.shared_experts." in name:
            if "gate_proj" in name or "up_proj" in name:
                expected_id = self._last_moe_x_id.get(layer_idx)
                if expected_id is None or id(x) != expected_id:
                    self.shared_gu_obj_mismatches += 1
            elif "down_proj" in name:
                self.shared_down_calls += 1
                x_flat = np.asarray(x, dtype=np.uint16).reshape(-1)
                cached = self._cached_shared_act.get(layer_idx)
                if (
                    cached is None
                    or x_flat.shape != cached.shape
                    or not np.array_equal(x_flat, cached)
                ):
                    raise ValueError(
                        f"Shared down cache mismatch at layer {layer_idx}: input activation bits "
                        "do not match B2 slot-6 cached shared activation"
                    )

        return super().linear(name, x)


@pytest.mark.parametrize("n_tokens", [1, 2, 3, 16, 17, 33])
def test_prefill_parity_token_counts(tiny_raon_model, n_tokens):
    """(가) Verify prefill == advance bit-for-bit parity across token counts 1, 2, 3, 16, 17, 33."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(1234 + n_tokens)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=n_tokens).tolist()

    res = run_prefill_replay(model, token_ids, chunk=16)
    assert res["match"] is True, f"Mismatch on {n_tokens} tokens: {res['first_mismatch']}"


def test_prefill_parity_checkpoint_in_chunk(tiny_raon_model):
    """(가) Verify prefill == advance byte parity when checkpoint is mid-chunk (5th token)."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(5555)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=24).tolist()

    res = run_prefill_replay(model, token_ids, chunk=16, checkpoint_len=5)
    assert res["match"] is True, f"Mismatch with checkpoint_len=5: {res['first_mismatch']}"


def test_prefill_parity_logits_positions_mid_chunk(tiny_raon_model):
    """(가) Verify prefill == advance byte parity when logits_positions are in the middle of chunks."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(7777)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=20).tolist()
    # middle positions: 2, 7, 15
    res = run_prefill_replay(model, token_ids, chunk=16, logits_positions=[2, 7, 15])
    assert res["match"] is True, f"Mismatch with mid-chunk logits: {res['first_mismatch']}"


def test_advance_deterministic(tiny_raon_model):
    """(나) Verify advance() output is bit-for-bit identical across runs after reset (reproducibility)."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    # Use fixed token IDs [42, 161, 88] as in test_host_small
    tokens = [42, 161, 88]
    model.reset_state()
    logits_list = []
    for tok in tokens:
        lg = model.advance(tok, need_logits=True)
        assert lg is not None
        logits_list.append(lg)

    # Re-run after reset to confirm deterministic reproducibility
    model.reset_state()
    for idx, tok in enumerate(tokens):
        lg_new = model.advance(tok, need_logits=True)
        assert lg_new is not None
        assert np.array_equal(logits_list[idx], lg_new), f"Non-deterministic advance at step {idx}"


def test_harness_catches_intentional_order_mutation(tiny_raon_model, monkeypatch):
    """(라) Verify harness reliably catches mismatches if token order inside prefill is intentionally mutated."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    orig_prefill = model.prefill

    def buggy_prefill(token_ids, *, logits_positions=(), chunk=16, record_intermediates=False):
        # Intentionally reverse token order if more than 1 token
        if len(token_ids) > 1:
            mutated = list(reversed(token_ids))
        else:
            mutated = token_ids
        return orig_prefill(
            mutated,
            logits_positions=logits_positions,
            chunk=chunk,
            record_intermediates=record_intermediates,
        )

    monkeypatch.setattr(model, "prefill", buggy_prefill)

    tokens = [10, 20, 30, 40]
    res = run_prefill_replay(model, tokens, chunk=16)
    assert res["match"] is False, "Harness failed to detect intentional token reversal"
    assert res["first_mismatch"] is not None
    assert res["first_mismatch"]["max_abs_diff"] > 0


# =============================================================================
# NPU Shared Expert Coupling Tests (Task Section 2)
# =============================================================================


@pytest.mark.parametrize("n_tokens", [2, 17])
def test_npu_coupled_prefill_parity_token_counts(tiny_raon_model, n_tokens):
    """Verify prefill == advance byte parity and NPU shared-expert cache coupling with tokens 2 and 17."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(2026 + n_tokens)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=n_tokens).tolist()

    res = run_prefill_replay(model, token_ids, chunk=16)
    assert res["match"] is True, f"Mismatch on {n_tokens} tokens: {res['first_mismatch']}"
    assert backend.shared_down_calls > 0, "No shared down calls were executed"
    assert backend.shared_gu_obj_mismatches == 0, (
        f"Found {backend.shared_gu_obj_mismatches} shared gate/up object mismatches"
    )


def test_npu_coupled_prefill_parity_checkpoint_mid_chunk(tiny_raon_model):
    """Verify prefill == advance byte parity with checkpoint in the middle of a chunk under NPU coupling."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(9999)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=24).tolist()

    res = run_prefill_replay(model, token_ids, chunk=16, checkpoint_len=5)
    assert res["match"] is True, f"Mismatch with checkpoint_len=5: {res['first_mismatch']}"
    assert backend.shared_down_calls > 0, "No shared down calls were executed"
    assert backend.shared_gu_obj_mismatches == 0, (
        f"Found {backend.shared_gu_obj_mismatches} shared gate/up object mismatches"
    )


def test_npu_coupled_revert_batched_shared_experts_fails(tiny_raon_model, monkeypatch):
    """Verify that batching shared experts across tokens (round 1 bug) fails on NpuCoupledMockBackend."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)

    orig_prefill = model.prefill

    def round1_batched_prefill(token_ids, *, logits_positions=(), chunk=16, record_intermediates=False):
        # Simulate round 1 MoE prefill: call router+moe_experts for all tokens, then batch shared experts
        total_tokens = len(token_ids)
        chunk_size = max(1, int(chunk))
        embed_w = model.weights.get("model.embed_tokens.weight", to_float=False)
        n = model.config.raon_mhc_streams

        for chunk_start in range(0, total_tokens, chunk_size):
            chunk_tokens = token_ids[chunk_start : chunk_start + chunk_size]
            T = len(chunk_tokens)
            stream = np.stack([np.tile(embed_w[tid], (n, 1)) for tid in chunk_tokens], axis=0)
            stream_f32 = bf16_to_f32(stream)

            for idx in range(model.config.num_hidden_layers):
                layer_pfx = f"model.layers.{idx}"
                layer_constants = model._layer_constants(idx)
                mlp_mhc_pfx = f"{layer_pfx}.mlp_raon_mhc"
                post_norm_w = layer_constants["post_norm"]

                mlp_branch_in_list = []
                for t in range(T):
                    hp, _, _ = model._route_mhc(stream[t], mlp_mhc_pfx, stream_f32[t])
                    b_raw = model._mhc_aggregate(stream[t], hp, stream_f32[t])
                    b_in = rms_norm(b_raw, post_norm_w, eps=model.config.rms_norm_eps)
                    mlp_branch_in_list.append(b_in)
                mlp_branch_in = np.stack(mlp_branch_in_list, axis=0)

                if model.config.is_moe_layer(idx):
                    pfx = f"{layer_pfx}.moe"
                    # Round 1 flaw: 1. call router & moe_experts for all tokens
                    routed_out_list = []
                    for t in range(T):
                        routed_out, _, _, _ = model._step_moe_router(idx, mlp_branch_in[t])
                        routed_out_list.append(routed_out)

                    # 2. call shared experts gate, up, down via linear_batch
                    if model.config.num_shared_experts > 0:
                        sh_gate = model.backend.linear_batch(
                            f"{pfx}.shared_experts.gate_proj.weight", mlp_branch_in
                        )
                        sh_up = model.backend.linear_batch(
                            f"{pfx}.shared_experts.up_proj.weight", mlp_branch_in
                        )
                        sh_act = situ_glu(
                            sh_gate,
                            sh_up,
                            gate_beta=model.config.activation_situ_beta,
                            value_beta=model.config.activation_situ_linear_beta,
                        )
                        # This must raise ValueError on NpuCoupledMockBackend!
                        model.backend.linear_batch(
                            f"{pfx}.shared_experts.down_proj.weight", f32_to_bf16_rtne(sh_act)
                        )

        return orig_prefill(
            token_ids,
            logits_positions=logits_positions,
            chunk=chunk,
            record_intermediates=record_intermediates,
        )

    monkeypatch.setattr(model, "prefill", round1_batched_prefill)

    token_ids = [42, 161]
    with pytest.raises(ValueError, match="Shared down cache mismatch"):
        model.prefill(token_ids, chunk=16)
