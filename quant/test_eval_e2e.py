# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors
"""Small random-model and unit tests for all end-to-end replacement modes."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from quant.eval_e2e import (
    MODES,
    compare_greedy,
    compare_logits,
    estimate_npu_read_bytes_per_token,
    evaluate_once,
    is_gguf_q4k,
    is_mixed_attn_bfp_target,
    is_mixed_lite_bfp_target,
    is_quantized_weight_name,
    parse_args,
    quantize_bfp16ebs8,
    replace_target_weights,
)


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 32)
        self.model.norm = torch.nn.LayerNorm(32)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(32, 32, bias=False)
        layer.self_attn.selector = torch.nn.Linear(32, 32, bias=False)
        layer.moe = torch.nn.Module()
        layer.moe.gate = torch.nn.Linear(32, 4, bias=False)
        layer.moe.shared_experts = torch.nn.Module()
        layer.moe.shared_experts.gate_proj = torch.nn.Linear(32, 32, bias=False)
        layer.moe.shared_experts.up_proj = torch.nn.Linear(32, 32, bias=False)
        layer.moe.shared_experts.down_proj = torch.nn.Linear(32, 32, bias=False)
        layer.moe.experts = torch.nn.ModuleList([torch.nn.Module()])
        layer.moe.experts[0].gate_proj = torch.nn.Linear(32, 32, bias=False)
        layer.moe.experts[0].up_proj = torch.nn.Linear(32, 32, bias=False)
        layer.moe.experts[0].down_proj = torch.nn.Linear(32, 32, bias=False)
        self.model.layers = torch.nn.ModuleList([layer])
        self.lm_head = torch.nn.Linear(32, 32, bias=False)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        hidden = torch.tanh(self.model.layers[0].self_attn.q_proj(hidden))
        hidden = self.model.layers[0].moe.experts[0].down_proj(
            torch.tanh(self.model.layers[0].moe.experts[0].gate_proj(hidden))
        )
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=())


class FakeGGUF:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state

    def matrix(self, name: str) -> np.ndarray:
        return self.state[name].float().numpy().copy()

    def q4_restore(self, name: str) -> np.ndarray:
        return self.matrix(name) * np.float32(0.98)

    def is_q4_k(self, name: str) -> bool:
        # In this mock: self_attn.q_proj and expert gate_proj are Q4_K.
        # self_attn.k_proj, down projections, up projections and lm_head are
        # non-Q4_K (e.g. Q6_K/Q8_0).
        return "q_proj" in name or "gate_proj" in name

    def source(self, name: str) -> tuple[str, list[str]]:
        types = ["Q4_K"] if self.is_q4_k(name) else ["Q6_K"]
        method = "direct Q4_K→q4nx" if types == ["Q4_K"] else "dequantize→min-max q4nx"
        return method, types


def _run_mode(mode: str, *, bfp_rounding: str = "conv_even"):
    torch.manual_seed(7)
    model = TinyCausalLM().to(torch.bfloat16).eval()
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    chunks = [torch.tensor([[1, 2, 3, 4, 5]])]
    prompts = [torch.tensor([[2, 4, 6]])]
    baseline, baseline_logits, baseline_greedy = evaluate_once(model, chunks, prompts, 4)
    source = FakeGGUF(original) if (mode.startswith("gguf_") or mode.startswith("mixed_")) else None
    manifest = replace_target_weights(model, mode, source, bfp_rounding=bfp_rounding)
    selected, selected_logits, selected_greedy = evaluate_once(model, chunks, prompts, 4)
    return model, original, manifest, baseline, selected, compare_logits(
        baseline_logits, selected_logits
    ), compare_greedy(baseline_greedy, selected_greedy)


def test_all_modes_run_and_orig_is_repeatable() -> None:
    for mode in MODES:
        _model, _before, manifest, baseline, selected, logits, greedy = _run_mode(mode)
        assert baseline["predicted_tokens"] == 4
        assert selected["predicted_tokens"] == 4
        assert logits["samples"] == 2
        assert len(greedy) == 1
        if mode == "orig":
            assert manifest["count"] == 0
            assert baseline == selected
            assert logits["kl_max"] == 0.0
            assert logits["top1_match_rate"] == 1.0


def test_replacement_changes_only_designated_matrices() -> None:
    model, before, manifest, *_ = _run_mode("gguf_q4nx")
    assert manifest["count"] > 0
    changed = set(manifest["names"])
    for name, value in model.named_parameters():
        if name in changed:
            assert is_quantized_weight_name(name)
            assert not torch.equal(value, before[name])
        else:
            assert torch.equal(value, before[name]), name
    assert "model.layers.0.moe.gate.weight" not in changed
    assert "model.layers.0.self_attn.selector.weight" not in changed
    assert "model.embed_tokens.weight" not in changed
    assert "model.norm.weight" not in changed


def test_mixed_modes_replacement_rules() -> None:
    # Test mixed_full: Q4_K -> q4nx, non-Q4_K -> bfp16ebs8
    model_full, _, manifest_full, *_ = _run_mode("mixed_full")
    assert manifest_full["quant_types"]["q4nx"] > 0
    assert manifest_full["quant_types"]["bfp16ebs8"] > 0

    # Test mixed_lite: only expert down, shared down, and lm_head are bfp16ebs8; others q4nx
    model_lite, _, manifest_lite, *_ = _run_mode("mixed_lite")
    assert "bfp16ebs8" in manifest_lite["quant_types"]
    assert "q4nx" in manifest_lite["quant_types"]
    # In TinyCausalLM, the lite targets are:
    # 1. model.layers.0.moe.experts.0.down_proj.weight
    # 2. model.layers.0.moe.shared_experts.down_proj.weight
    # 3. lm_head.weight
    assert manifest_lite["quant_types"]["bfp16ebs8"] == 3


def _mixed_attn_qtype(name: str, fake_gguf: FakeGGUF) -> str:
    """The quant type mixed_attn must apply to target matrix `name`."""
    if is_mixed_attn_bfp_target(name) and not is_gguf_q4k(fake_gguf, name):
        return "bfp16ebs8"
    return "q4nx"


def test_mixed_attn_replaces_only_non_q4k_attention_with_bfp16() -> None:
    """Q4_K -> q4nx, non-Q4_K self_attn -> bfp16ebs8, other non-Q4_K -> q4nx."""
    model, original, manifest, *_ = _run_mode("mixed_attn")
    fake_gguf = FakeGGUF(original)
    params = dict(model.named_parameters())
    expected = {name: _mixed_attn_qtype(name, fake_gguf) for name in manifest["names"]}
    # The tiny model has one Q4_K and one non-Q4_K attention projection.
    assert expected["model.layers.0.self_attn.q_proj.weight"] == "q4nx"
    assert expected["model.layers.0.self_attn.k_proj.weight"] == "bfp16ebs8"
    # Non-attention non-Q4_K matrices stay q4nx (unlike mixed_lite).
    for name in (
        "model.layers.0.moe.experts.0.down_proj.weight",
        "model.layers.0.moe.shared_experts.down_proj.weight",
        "lm_head.weight",
    ):
        assert expected[name] == "q4nx"
    assert manifest["quant_types"] == dict(sorted(Counter(expected.values()).items()))
    assert manifest["quant_types"]["bfp16ebs8"] == 1
    for name, qtype in expected.items():
        if qtype == "bfp16ebs8":
            want = quantize_bfp16ebs8(
                original[name].float().numpy(), rounding="conv_even"
            )
        else:
            want = fake_gguf.q4_restore(name)
        assert torch.equal(
            params[name], torch.from_numpy(np.ascontiguousarray(want)).to(params[name].dtype)
        ), f"{name}: expected {qtype} replacement"


def test_mixed_attn_npu_read_bytes_bit_allocation() -> None:
    """mixed_attn counts 9 bits only for non-Q4_K attention; everything else is 5 bits."""
    torch.manual_seed(7)
    model = TinyCausalLM().to(torch.bfloat16).eval()
    fake_gguf = FakeGGUF(
        {name: value.detach().clone() for name, value in model.named_parameters()}
    )
    est = estimate_npu_read_bytes_per_token(model, "mixed_attn", fake_gguf)
    packed = code = attention = 0.0
    for name, parameter in model.named_parameters():
        if not is_quantized_weight_name(name):
            continue
        ratio = 6.0 / 128.0 if ".moe.experts." in name else 1.0
        bfp = _mixed_attn_qtype(name, fake_gguf) == "bfp16ebs8"
        bits_packed, bits_code = (9.0, 9.0) if bfp else (5.0, 4.0)
        packed += parameter.numel() * ratio * bits_packed / 8.0
        code += parameter.numel() * ratio * bits_code / 8.0
        if is_mixed_attn_bfp_target(name):
            attention += parameter.numel() * ratio * bits_packed / 8.0
    assert est["bytes_per_token"] == round(packed)
    assert est["bytes_per_token_4bit_nominal"] == round(code)
    assert est["breakdown_by_family_bytes"]["attention"] == pytest.approx(attention)
    # Non-attention families are all-q4nx, exactly as in gguf_q4nx.
    q4nx = estimate_npu_read_bytes_per_token(model, "gguf_q4nx", fake_gguf)
    for family, value in q4nx["breakdown_by_family_bytes"].items():
        if family == "attention":
            continue
        assert est["breakdown_by_family_bytes"][family] == pytest.approx(value)


def _bfp_replaced_names(mode: str, names: list[str], fake_gguf: FakeGGUF) -> list[str]:
    """Names replaced via quantize_bfp16ebs8 under `mode` (the rest go through q4nx)."""
    if mode == "mixed_full":
        return [n for n in names if not is_gguf_q4k(fake_gguf, n)]
    if mode == "mixed_attn":
        return [
            n
            for n in names
            if is_mixed_attn_bfp_target(n) and not is_gguf_q4k(fake_gguf, n)
        ]
    return [n for n in names if is_mixed_lite_bfp_target(n)]


@pytest.mark.parametrize("mode", ["mixed_full", "mixed_lite", "mixed_attn"])
def test_mixed_modes_default_to_conv_even_rounding(tmp_path, mode) -> None:
    """mixed modes must round bfp16ebs8 weights to nearest-even by default."""
    model, original, manifest, *_ = _run_mode(mode)
    fake_gguf = FakeGGUF(original)
    bfp_names = set(_bfp_replaced_names(mode, manifest["names"], fake_gguf))
    assert bfp_names, f"no bfp16ebs8 targets for {mode}"
    differs = False
    for name in manifest["names"]:
        if name not in bfp_names:
            continue
        got = dict(model.named_parameters())[name]
        orig = original[name].float().numpy()
        conv = quantize_bfp16ebs8(orig, rounding="conv_even")
        floored = quantize_bfp16ebs8(orig, rounding="floor")
        differs |= not np.array_equal(conv, floored)
        assert torch.equal(
            got, torch.from_numpy(np.ascontiguousarray(conv)).to(got.dtype)
        ), f"{name}: default bfp rounding is not conv_even"
    assert differs, "conv_even and floor agree everywhere: test cannot distinguish them"
    # An explicit floor request must reproduce the old behaviour exactly.
    model_f, original_f, manifest_f, *_ = _run_mode(mode, bfp_rounding="floor")
    for name in manifest_f["names"]:
        if name not in _bfp_replaced_names(mode, manifest_f["names"], FakeGGUF(original_f)):
            continue
        got = dict(model_f.named_parameters())[name]
        expect = quantize_bfp16ebs8(original_f[name].float().numpy(), rounding="floor")
        assert torch.equal(
            got, torch.from_numpy(np.ascontiguousarray(expect)).to(got.dtype)
        ), f"{name}: bfp_rounding='floor' not honored"


def test_conv_even_weight_error_bias_is_smaller_than_floor() -> None:
    """On random weights the mean signed error (quantized − original) must be closer
    to zero for conv_even than for floor (floor always biases weights downward)."""
    rng = np.random.RandomState(4242)
    x = (rng.randn(64, 128) * np.exp2(rng.randint(-10, 10, size=(64, 128)))).astype(
        np.float32
    )
    conv = quantize_bfp16ebs8(x, rounding="conv_even")
    floored = quantize_bfp16ebs8(x, rounding="floor")
    conv_bias = abs(float((conv - x).mean()))
    floor_bias = abs(float((floored - x).mean()))
    assert conv_bias < floor_bias


def test_bfp_rounding_cli_option(monkeypatch) -> None:
    base = ["eval_e2e.py", "--mode", "mixed_full", "--text", "t", "--prompts", "p",
            "--model-dir", "fixture-model", "--gguf", "fixture.gguf"]
    monkeypatch.setattr(
        sys, "argv", base + ["--out", str(Path("/tmp/unused.json"))]
    )
    assert parse_args().bfp_rounding == "conv_even"
    monkeypatch.setattr(
        sys, "argv", base + ["--out", "/tmp/unused.json", "--bfp-rounding", "floor"]
    )
    assert parse_args().bfp_rounding == "floor"


def test_npu_read_bytes_estimation() -> None:
    torch.manual_seed(7)
    model = TinyCausalLM().to(torch.bfloat16).eval()
    state = {name: value.detach().clone() for name, value in model.named_parameters()}
    fake_gguf = FakeGGUF(state)

    for mode in MODES:
        est = estimate_npu_read_bytes_per_token(model, mode, fake_gguf)
        assert est["bytes_per_token"] > 0
        assert est["total_target_elements"] > 0
        assert est["effective_target_elements_per_token"] > 0
        # orig (16-bit) should be larger than 4/5-bit modes
        if mode == "orig":
            assert est["bytes_per_token"] > estimate_npu_read_bytes_per_token(model, "minmax", fake_gguf)["bytes_per_token"]


def test_bfp16ebs8_matches_mlir_aie_reference() -> None:
    """Bit-for-bit test against original mlir-aie/python/utils/bfp.py quantize definition."""
    import os
    if not os.environ.get("RAON_MLIR_AIE_DIR"):
        pytest.skip("RAON_MLIR_AIE_DIR is not set")
    orig_path = Path(os.environ["RAON_MLIR_AIE_DIR"]) / "python/utils/bfp.py"
    if not orig_path.exists():
        pytest.skip("mlir-aie bfp.py not found on filesystem")

    # Mock minimal dependencies to load mlir-aie bfp.py without ml_dtypes
    sys_modules_backup = dict(sys.modules)
    try:
        m_aie = ModuleType("aie")
        m_aie_helpers = ModuleType("aie.helpers")
        m_aie_helpers_npdtypes = ModuleType("aie.helpers.npdtypes")

        class DummyV8:
            block_size = 8
            block_bytes = 9

        m_aie_helpers_npdtypes.v8bfp16ebs8 = DummyV8
        m_aie_helpers_npdtypes.is_block_float = lambda dt: dt is DummyV8
        sys.modules["aie"] = m_aie
        sys.modules["aie.helpers"] = m_aie_helpers
        sys.modules["aie.helpers.npdtypes"] = m_aie_helpers_npdtypes

        spec = importlib.util.spec_from_file_location("ref_bfp", orig_path)
        ref_bfp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_bfp)

        # Test vectors
        rng = np.random.RandomState(42)
        test_inputs = [
            rng.randn(4, 16).astype(np.float32),
            rng.uniform(-100.0, 100.0, size=(8, 64)).astype(np.float32),
            np.array([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 127.0] * 2, dtype=np.float32).reshape(2, 8),
            np.array([-1e-4, 1e-4, -1e-6, 1e-6, -10.0, 10.0, -0.01, 0.01], dtype=np.float32).reshape(1, 8),
        ]

        for x in test_inputs:
            ref_floor = ref_bfp.quantize(x, rounding="floor")
            my_floor = quantize_bfp16ebs8(x, rounding="floor")
            np.testing.assert_array_equal(ref_floor, my_floor)

            ref_conv = ref_bfp.quantize(x, rounding="conv_even")
            my_conv = quantize_bfp16ebs8(x, rounding="conv_even")
            np.testing.assert_array_equal(ref_conv, my_conv)
    finally:
        # Restore sys.modules
        for k in list(sys.modules):
            if k not in sys_modules_backup:
                del sys.modules[k]
        sys.modules.update(sys_modules_backup)


def test_quantize_bfp16ebs8_row_chunking_is_bitwise_identical() -> None:
    """Chunked (rows_per_chunk < n_rows) must be bit-identical to one-shot processing.

    bfp16 blocks are 8 elements along the last axis only, so splitting rows must
    never change the encoded block exponent or mantissas.
    """
    rng = np.random.RandomState(2026)
    cases = []
    for rounding in ("floor", "conv_even"):
        for shape, rows_per_chunk in (
            ((10, 8), 3),          # 2D, rows not divisible by chunk
            ((9, 16), 7),          # 2D, two uneven chunks, wider last axis
            ((4, 3, 8), 2),        # 3D: 12 flattened rows, chunk splits mid-flatten
            ((5, 4, 16), 7),       # 3D with a wider last axis
        ):
            x = rng.randn(*shape).astype(np.float32) * np.exp2(
                rng.randint(-20, 20, size=shape)
            ).astype(np.float32)
            cases.append((x, rows_per_chunk, rounding))

    for x, rows_per_chunk, rounding in cases:
        assert x.reshape(-1, x.shape[-1]).shape[0] > rows_per_chunk
        whole = quantize_bfp16ebs8(x, rounding=rounding, rows_per_chunk=x.shape[0])
        chunked = quantize_bfp16ebs8(x, rounding=rounding, rows_per_chunk=rows_per_chunk)
        assert chunked.dtype == whole.dtype == np.float32
        assert chunked.shape == x.shape
        np.testing.assert_array_equal(chunked, whole)

    # single-row and exact-divisor chunks too
    x = np.array([[1024.0, 0.1, -0.2, 0.3, -1e-6, 5.0, -0.01, 0.7]], np.float32)
    for rounding in ("floor", "conv_even"):
        np.testing.assert_array_equal(
            quantize_bfp16ebs8(x, rounding=rounding, rows_per_chunk=3),
            quantize_bfp16ebs8(x, rounding=rounding, rows_per_chunk=1),
        )


def test_quantize_bfp16ebs8_argument_validation() -> None:
    x = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="rounding"):
        quantize_bfp16ebs8(x, rounding="round")
    with pytest.raises(ValueError, match="rows_per_chunk"):
        quantize_bfp16ebs8(x, rounding="floor", rows_per_chunk=0)
    with pytest.raises(ValueError, match="multiple of 8"):
        quantize_bfp16ebs8(np.ones((2, 6), dtype=np.float32), rows_per_chunk=1)


def test_bfp16ebs8_block_orientation_last_axis() -> None:
    """Verify that bfp16ebs8 blocks along the last axis (input k direction)."""
    # A (2, 8) matrix: each row is a separate block of 8 values
    # Row 0 has max exponent corresponding to 1024.0
    # Row 1 has max exponent corresponding to 1.0
    x = np.array([
        [1024.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ], dtype=np.float32)

    q = quantize_bfp16ebs8(x, rounding="floor")
    # Row 0: 0.1 is 10 binades below 1024.0, will be heavily truncated/coarsened
    # Row 1: 0.1 is close to 1.0, preserved with fine resolution
    assert q[0, 0] == 1024.0
    assert q[1, 0] == 1.0
    assert abs(q[1, 1] - 0.1) < abs(q[0, 1] - 0.1)
