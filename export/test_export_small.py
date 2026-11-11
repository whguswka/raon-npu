# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit and integration tests for Raon-1-mini D mode weight export and verification."""

from __future__ import annotations

import os
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from export.export_d import (
    DModeExporter,
    _save_tensors_verified,
    _tensor_bits,
    compute_sha256,
    fix_host_only,
    host_tensor_to_numpy,
    parse_layers_arg,
    write_and_record,
    write_host_index,
)
from export.verify_d import DModeVerifier
from quant.bfp16 import (
    pad_bfp16_gemv,
    pack_bfp16_gemv,
    parse_bfp16_gemv,
    quantize_bfp16ebs8,
)
from quant.q4nx import (
    dequantize_simple,
    pack_gemv,
    pad_gemv,
    parse_gemv,
    quantize,
)


DEFAULT_GGUF = Path(os.environ.get("RAON_GGUF", "__RAON_GGUF_NOT_SET__"))


def test_parse_layers_arg():
    assert parse_layers_arg(None) is None
    assert parse_layers_arg("all") is None
    assert parse_layers_arg("0,1,3") == [0, 1, 3]
    assert parse_layers_arg("0-3") == [0, 1, 2, 3]
    assert parse_layers_arg("0, 2-4, 7") == [0, 2, 3, 4, 7]


def test_q4nx_bundle_pack_parse_roundtrip():
    """Verify q4nx pack_gemv -> parse_gemv roundtrip preserves exact bit patterns."""
    np.random.seed(42)
    M, K = 128, 256
    w = (np.random.randn(M, K) * 0.1).astype(np.float32)

    codes, scales, mins = quantize(w)
    packed = pack_gemv(codes, scales, mins, rows=4)
    assert len(packed) == M * K * 5 // 8

    p_codes, p_scales, p_mins = parse_gemv(packed, K=K, M=M, rows=4)
    assert np.array_equal(codes, p_codes)
    assert np.array_equal(scales.view(np.uint32), p_scales.view(np.uint32))
    assert np.array_equal(mins.view(np.uint32), p_mins.view(np.uint32))

    restored = dequantize_simple(p_codes, p_scales, p_mins)
    expected = dequantize_simple(codes, scales, mins)
    assert np.array_equal(restored.view(np.uint32), expected.view(np.uint32))


def test_bfp16_bundle_pack_parse_roundtrip():
    """Verify 9-bit bfp16 pack_bfp16_gemv -> parse_bfp16_gemv bit-exact parity with quantize_bfp16ebs8."""
    np.random.seed(123)
    M, K = 128, 256
    w = (np.random.randn(M, K) * 0.5).astype(np.float32)

    expected = quantize_bfp16ebs8(w, rounding="conv_even")
    packed = pack_bfp16_gemv(w, rows=4, flush_tiny=True)
    assert len(packed) == M * K * 9 // 8

    restored = parse_bfp16_gemv(packed, K=K, M=M, rows=4)
    assert np.array_equal(restored.view(np.uint32), expected.view(np.uint32))


def test_bfp16_padding_and_zero_verification():
    """Verify that pad_bfp16_gemv zero-pads rows and columns and packed representation retains exact 0s."""
    np.random.seed(999)
    orig_m, orig_k = 100, 200
    w = np.random.randn(orig_m, orig_k).astype(np.float32)

    padded, (m, k) = pad_bfp16_gemv(w, m_align=128, k_align=256)
    assert padded.shape == (128, 256)
    assert m == orig_m and k == orig_k

    # Unpadded area matches
    assert np.array_equal(padded[:orig_m, :orig_k], w)
    # Padded areas are 0.0
    assert np.all(padded[orig_m:, :] == 0.0)
    assert np.all(padded[:, orig_k:] == 0.0)

    packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
    restored = parse_bfp16_gemv(packed, K=256, M=128, rows=4)

    expected_orig = quantize_bfp16ebs8(w, rounding="conv_even")
    assert np.array_equal(restored[:orig_m, :orig_k].view(np.uint32), expected_orig.view(np.uint32))
    assert np.all(restored[orig_m:, :] == 0.0)
    assert np.all(restored[:, orig_k:] == 0.0)


def test_rda_in_stacking_and_slicing():
    """Simulate RDA rda_in bundle stacking, padding, packing, and slicing verification."""
    np.random.seed(777)
    K = 256
    # Synthesize slices resembling [q, k, v, g, f_a, b, rda_w]
    shapes = [
        ("q", 64),
        ("k", 64),
        ("v", 64),
        ("g", 64),
        ("f_a", 16),
        ("b", 8),
        ("rda_w", 8),
    ]
    slices = [np.random.randn(rows, K).astype(np.float32) for _, rows in shapes]
    total_orig_rows = sum(rows for _, rows in shapes)  # 288
    stacked = np.vstack(slices)

    # Pad to multiple of 128 -> 384
    padded, (orig_m, orig_k) = pad_bfp16_gemv(stacked, m_align=128, k_align=256)
    assert padded.shape == (384, K)
    assert np.all(padded[total_orig_rows:, :] == 0.0)

    packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
    restored = parse_bfp16_gemv(packed, K=K, M=384, rows=4)

    # Verify each slice independently matches quantize_bfp16ebs8
    offset = 0
    for i, (_, rows) in enumerate(shapes):
        orig_slice = slices[i]
        expected_slice = quantize_bfp16ebs8(orig_slice, rounding="conv_even")
        restored_slice = restored[offset : offset + rows, :]
        assert np.array_equal(restored_slice.view(np.uint32), expected_slice.view(np.uint32))
        offset += rows

    # Verify padding rows [288:384] are all 0.0
    assert np.all(restored[total_orig_rows:, :] == 0.0)


def test_mla_kb_vb_transposition_and_correspondence():
    """Verify MLA k_b transpose + flatten and head correspondence with HF kv_b_proj."""
    np.random.seed(888)
    heads = 4
    input_dim = 16  # K in GGUF attn_k_b
    output_dim = 8  # V in GGUF attn_k_b
    # GGUF attn_k_b: [heads, input_dim, output_dim] = [4, 16, 8]
    raw_k = np.random.randn(heads, input_dim, output_dim).astype(np.float32)
    # GGUF attn_v_b: [heads, output_dim, input_dim] = [4, 8, 16]
    raw_v = np.random.randn(heads, output_dim, input_dim).astype(np.float32)

    # k_transposed: [4, 8, 16] -> [32, 16]
    k_trans = raw_k.transpose(0, 2, 1)
    k_flat = k_trans.reshape(-1, input_dim)
    # v_flat: [32, 16]
    v_flat = raw_v.reshape(-1, input_dim)

    # Synthesize HF kv_b_proj: interleaved head-wise [heads, 2 * output_dim, input_dim]
    hf_kv_b = np.concatenate([k_trans, raw_v], axis=1).reshape(-1, input_dim)
    assert hf_kv_b.shape == (heads * 2 * output_dim, input_dim)  # [64, 16]

    # Verify correspondence rule
    for h in range(heads):
        k_hf = hf_kv_b[h * 2 * output_dim : h * 2 * output_dim + output_dim, :]
        v_hf = hf_kv_b[h * 2 * output_dim + output_dim : (h + 1) * 2 * output_dim, :]
        assert np.array_equal(k_hf, k_flat[h * output_dim : (h + 1) * output_dim, :])
        assert np.array_equal(v_hf, v_flat[h * output_dim : (h + 1) * output_dim, :])


def test_manifest_schema_and_verifier_error_detection():
    """Test manifest creation, sha256 calculation, and verifier failure detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bin_file = tmp_path / "test_bundle.bin"
        data = b"0" * 9216  # exactly one bfp16 GEMV block (32x256)
        entry = write_and_record(
            out_path=bin_file,
            data=data,
            bundle_name="test_bundle",
            layer_idx=0,
            fmt="bfp16_gemv",
            orig_shape=(32, 256),
            padded_shape=(32, 256),
            components=[],
            gguf_types=["TEST"],
        )

        manifest_file = tmp_path / "manifest.json"
        manifest_data = {
            "version": "1.0",
            "model": "Raon-1-mini-preview",
            "mode": "mixed_full_d",
            "bundles": [entry],
        }
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

        # Check sha256 match
        assert compute_sha256(bin_file) == entry["sha256"]

        # Tamper with file to verify corruption detection
        with open(bin_file, "wb") as f:
            f.write(b"1" * 9216)

        assert compute_sha256(bin_file) != entry["sha256"]


@pytest.mark.skipif(not DEFAULT_GGUF.is_file(), reason="RAON_GGUF is not set or file is unavailable")
def test_export_single_bundle_integration():
    """Integration test: export layer 0 dense_down and verify bit equality against GGUF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        exporter = DModeExporter(
            gguf_path=DEFAULT_GGUF,
            hf_dir=None,
            out_dir=tmp_path,
            layers=[0],
            only_bundles=["dense_down"],
            export_host=False,
        )
        manifest_data = exporter.run()
        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.is_file()

        bundles = manifest_data["bundles"]
        assert len(bundles) == 1
        bundle = bundles[0]
        assert bundle["bundle"] == "L00/dense_down"
        assert bundle["format"] == "bfp16_gemv"
        assert bundle["size_bytes"] == 34652160


def _make_host_only_exporter(hf_dir: Path, out_dir: Path, layers: list[int]) -> DModeExporter:
    """Build a DModeExporter without GGUF loading, for host-tensor-only tests."""
    exporter = object.__new__(DModeExporter)
    exporter.export_host = True
    exporter.hf_dir = Path(hf_dir)
    exporter.out_dir = Path(out_dir)
    exporter.layers = list(layers)
    exporter.only_bundles = {"host"}
    exporter.manifest_entries = []
    return exporter


def test_host_tensor_to_numpy_bf16_bit_exact_and_passthrough():
    """bf16 -> uint16 bit patterns (matches host/weights.py BF16-as-uint16); fp32/fp16/int as-is."""
    torch = pytest.importorskip("torch")

    bf = torch.tensor([0.1, -2.3, 3.75, 1e-8], dtype=torch.bfloat16)
    arr, src_dtype = host_tensor_to_numpy(bf)
    assert src_dtype == "bfloat16"
    assert arr.dtype == np.uint16
    # Bit-exact: stored bits equal the bf16 tensor bits (same uint16 view the host reads)
    assert np.array_equal(arr, bf.view(torch.int16).numpy().view(np.uint16))
    # host/weights.py bf16_to_f32 conversion of stored bits reproduces the original values
    restored_f32 = (arr.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(restored_f32, bf.float().numpy())

    for t in (
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float16),
        torch.arange(4, dtype=torch.int64),
    ):
        arr_p, src_p = host_tensor_to_numpy(t)
        assert arr_p.dtype == t.numpy().dtype
        assert src_p == str(t.numpy().dtype)
        assert np.array_equal(arr_p, t.numpy())


def test_export_host_tensors_bf16_bf16_header_bit_exact_and_manifest():
    """Synthetic bf16+fp32 checkpoint -> host safetensors with BF16 header dtype, bit-exact."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        orig = {
            "model.layers.0.input_layernorm.weight": torch.tensor([0.1, -2.3, 3.75, 1e-8], dtype=torch.bfloat16),
            "model.layers.0.mlp_raon_mhc.alpha_pre": torch.tensor(0.015625, dtype=torch.bfloat16),
            "model.layers.0.mlp_raon_mhc.alpha_post": torch.tensor(-0.3, dtype=torch.bfloat16),
            "model.layers.0.mlp_raon_mhc.res.alpha_z": torch.tensor(0.5, dtype=torch.bfloat16),
            "model.layers.0.mlp_raon_mhc.res.eta": torch.tensor(0.125, dtype=torch.float32),
            "model.layers.0.mlp_raon_mhc.res.phi_z": torch.arange(6, dtype=torch.float32) * 0.25,
            "model.norm.weight": torch.arange(4, dtype=torch.float32) * 0.5,
            "model.embed_tokens.weight": torch.tensor([1e-9, 2.0, -3.5], dtype=torch.bfloat16),
        }
        shard = "model-00001-of-00001.safetensors"
        st.save_file(orig, str(hf_dir / shard))
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: shard for k in orig}})
        )

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        manifest_data = exporter.run()

        bundles = {b["bundle"]: b for b in manifest_data["bundles"]}
        assert set(bundles) == {"L00/host", "host_global"}
        comps = {c["name"]: c for b in bundles.values() for c in b["components"]}
        # bf16 tensors: original dtype recorded, header dtype stored as BF16
        assert comps["model.layers.0.input_layernorm.weight"]["dtype"] == "bfloat16"
        assert comps["model.layers.0.input_layernorm.weight"]["stored_dtype"] == "BF16"
        assert comps["model.embed_tokens.weight"]["dtype"] == "bfloat16"
        assert comps["model.embed_tokens.weight"]["stored_dtype"] == "BF16"
        # fp32 tensors stay F32
        assert comps["model.layers.0.mlp_raon_mhc.res.phi_z"]["dtype"] == "float32"
        assert comps["model.layers.0.mlp_raon_mhc.res.phi_z"]["stored_dtype"] == "F32"
        assert comps["model.norm.weight"]["dtype"] == "float32"
        assert comps["model.layers.0.mlp_raon_mhc.res.eta"]["dtype"] == "float32"
        assert comps["model.layers.0.mlp_raon_mhc.res.eta"]["stored_dtype"] == "F32"
        assert comps["model.layers.0.mlp_raon_mhc.res.eta"]["shape"] == []
        for name in (
            "model.layers.0.mlp_raon_mhc.alpha_pre",
            "model.layers.0.mlp_raon_mhc.alpha_post",
            "model.layers.0.mlp_raon_mhc.res.alpha_z",
        ):
            assert comps[name]["dtype"] == "bfloat16"
            assert comps[name]["stored_dtype"] == "BF16"
            assert comps[name]["shape"] == []
        for name, t in orig.items():
            assert comps[name]["shape"] == list(t.shape)

        # Saved files are readable and bit-exact; bf16 comes back as torch.bfloat16
        for bundle in bundles.values():
            with safe_open(str(out_dir / bundle["file"]), framework="pt") as f:
                for name in comps:
                    if name not in {c["name"] for c in bundle["components"]}:
                        continue
                    stored = f.get_tensor(name)
                    if orig[name].dtype == torch.bfloat16:
                        assert stored.dtype == torch.bfloat16
                        assert torch.equal(
                            stored.view(torch.int16),
                            orig[name].view(torch.int16),
                        )
                    else:
                        assert np.array_equal(stored.numpy(), orig[name].numpy())


def test_host_export_failure_writes_manifest():
    """Invalid HF input must fail before creating a manifest or output bundle."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()
        # index references a shard file that does not exist -> load fails
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.norm.weight": "missing.safetensors"}})
        )

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[])
        with pytest.raises(FileNotFoundError):
            exporter.run()

        assert not (out_dir / "manifest.json").exists()


@pytest.mark.skipif(not DEFAULT_GGUF.is_file(), reason="RAON_GGUF is not set or file is unavailable")
def test_host_export_failure_after_bundle_written_keeps_bundle_in_manifest():
    """A bad HF shard must be detected before the selected NPU bundle is written."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()
        # Layer-0 host shard file referenced by the index does not exist -> the host
        # stage raises FileNotFoundError after L00/dense_down has already been written.
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"model.layers.0.input_layernorm.weight": "missing.safetensors"}}
            )
        )

        exporter = DModeExporter(
            gguf_path=DEFAULT_GGUF,
            hf_dir=hf_dir,
            out_dir=out_dir,
            layers=[0],
            only_bundles=["dense_down", "host"],
            export_host=True,
        )
        with pytest.raises(FileNotFoundError):
            exporter.run()

        assert not (out_dir / "L00" / "dense_down.bin").exists()
        assert not (out_dir / "manifest.json").exists()


def test_host_safetensors_header_dtype_is_bf16_and_bits_unchanged():
    """Raw header dtype string is 'BF16' for bf16 tensors and bit patterns are preserved."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        orig = {
            "w_bf16": torch.tensor([0.1, -2.3, 3.75, 1e-8], dtype=torch.bfloat16),
            "w_fp32": torch.arange(4, dtype=torch.float32) * 0.5,
        }
        shard = "model.safetensors"
        st.save_file(orig, str(hf_dir / shard))
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: shard for k in orig}})
        )

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[])
        exporter.run()

        header_path = out_dir / "host_global.safetensors"
        with open(header_path, "rb") as f:
            import struct
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

        assert header["w_bf16"]["dtype"] == "BF16"
        assert header["w_fp32"]["dtype"] == "F32"

        from safetensors import safe_open
        with safe_open(str(header_path), framework="pt") as f:
            bf16_stored = f.get_tensor("w_bf16")
            assert bf16_stored.dtype == torch.bfloat16
            assert torch.equal(bf16_stored.view(torch.int16), orig["w_bf16"].view(torch.int16))


def test_host_index_weight_map_covers_all_host_tensors():
    """model.safetensors.index.json points every host tensor to the correct file."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        orig = {
            "model.layers.0.input_layernorm.weight": torch.tensor([0.1, -2.3], dtype=torch.bfloat16),
            "model.norm.weight": torch.arange(4, dtype=torch.float32) * 0.5,
        }
        shard = "model.safetensors"
        st.save_file(orig, str(hf_dir / shard))
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: shard for k in orig}})
        )

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        exporter.run()

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        for bundle_name, file_path in (
            ("model.layers.0.input_layernorm.weight", "L00/host.safetensors"),
            ("model.norm.weight", "host_global.safetensors"),
        ):
            assert weight_map[bundle_name] == file_path


def test_fix_host_only_converts_u16_to_bf16_and_preserves_bits():
    """--fix-host-only rewrites U16 bf16 headers to BF16 without changing data bits."""
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        orig_bf16 = torch.tensor([0.1, -2.3, 3.75, 1e-8], dtype=torch.bfloat16)
        u16_bits = orig_bf16.view(torch.int16).numpy().view(np.uint16)
        file_path = out_dir / "host_global.safetensors"
        snp.save_file({"w": u16_bits}, str(file_path))

        def tensor_data_sha256(path: Path) -> str:
            from safetensors import safe_open
            h = hashlib.sha256()
            with safe_open(str(path), framework="pt") as f:
                for name in sorted(f.keys()):
                    t = f.get_tensor(name)
                    if t.dtype == torch.bfloat16:
                        h.update(t.view(torch.int16).numpy().tobytes())
                    else:
                        h.update(t.numpy().tobytes())
            return h.hexdigest()

        old_data_sha = tensor_data_sha256(file_path)

        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "bundles": [
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": file_path.stat().st_size,
                    "components": [
                        {"name": "w", "shape": [4], "dtype": "bfloat16", "stored_dtype": "uint16"}
                    ],
                    "sha256": compute_sha256(file_path),
                    "description": "test",
                }
            ],
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = fix_host_only(out_dir)
        assert result["updated"]
        assert result["host_bundles_checked"] == 1

        new_data_sha = tensor_data_sha256(file_path)
        assert new_data_sha == old_data_sha, "tensor data bits must be unchanged"

        with open(file_path, "rb") as f:
            import struct
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        assert header["w"]["dtype"] == "BF16"

        from safetensors import safe_open
        with safe_open(str(file_path), framework="pt") as f:
            t = f.get_tensor("w")
            assert t.dtype == torch.bfloat16
            assert torch.equal(t.view(torch.int16), orig_bf16.view(torch.int16))

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert index["weight_map"]["w"] == "host_global.safetensors"

        mf = json.loads((out_dir / "manifest.json").read_text())
        assert mf["bundles"][0]["components"][0]["stored_dtype"] == "BF16"
        assert mf["bundles"][0]["sha256"] == compute_sha256(file_path)


def test_fix_host_only_handles_0d_tensors():
    """--fix-host-only rewrites 0d U16 bf16 alongside an unchanged 0d F32 tensor."""
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        orig_bf16 = torch.tensor(0.015625, dtype=torch.bfloat16)
        alpha_bits = orig_bf16.view(torch.int16).numpy().view(np.uint16)
        file_path = out_dir / "host_global.safetensors"
        snp.save_file(
            {"alpha": alpha_bits, "eta": np.array(0.125, dtype=np.float32)},
            str(file_path),
        )

        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "bundles": [
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": file_path.stat().st_size,
                    "components": [
                        {"name": "alpha", "shape": [], "dtype": "bfloat16", "stored_dtype": "uint16"},
                        {"name": "eta", "shape": [], "dtype": "float32", "stored_dtype": "F32"},
                    ],
                    "sha256": compute_sha256(file_path),
                    "description": "test",
                }
            ],
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = fix_host_only(out_dir)
        assert result["updated"]
        assert result["host_bundles_checked"] == 1

        with open(file_path, "rb") as f:
            import struct
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        assert header["alpha"]["dtype"] == "BF16"
        assert header["eta"]["dtype"] == "F32"

        with safe_open(str(file_path), framework="pt") as f:
            alpha = f.get_tensor("alpha")
            eta = f.get_tensor("eta")
        assert alpha.shape == ()
        assert eta.shape == ()
        assert torch.equal(alpha.view(torch.int16), orig_bf16.view(torch.int16))
        assert eta.item() == 0.125


def test_fix_host_only_recomputes_manifest_total_sizes():
    """--fix-host-only recomputes total_size_bytes/mb/gb as the sum of bundle size_bytes."""
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        # Non-host bundle keeps its size untouched by the fix and still counts in the sum.
        bin_file = out_dir / "L00" / "dense_down.bin"
        bin_file.parent.mkdir(parents=True)
        bin_file.write_bytes(b"0" * 9216)

        # Host safetensors written with a U16 header; after the BF16 rewrite the header
        # length changes, so its size_bytes (and hence the totals) must be recomputed.
        orig_bf16 = torch.tensor([0.1, -2.3, 3.75, 1e-8] * 8, dtype=torch.bfloat16)
        u16_bits = orig_bf16.view(torch.int16).numpy().view(np.uint16)
        host_file = out_dir / "host_global.safetensors"
        snp.save_file({"w": u16_bits}, str(host_file))
        stale_size = host_file.stat().st_size  # pre-fix size, stale until recomputed

        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            # Stale top-level totals left over from a manifest written before the fix.
            "total_size_bytes": 9216 + stale_size - 584,
            "total_size_mb": 0.01,
            "total_size_gb": 0.0,
            "bundles": [
                {
                    "bundle": "L00/dense_down",
                    "layer": 0,
                    "format": "bfp16_gemv",
                    "file": "L00/dense_down.bin",
                    "size_bytes": 9216,
                    "components": [],
                    "sha256": compute_sha256(bin_file),
                },
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": stale_size,
                    "components": [
                        {"name": "w", "shape": [32], "dtype": "bfloat16", "stored_dtype": "uint16"}
                    ],
                    "sha256": "stale",
                },
            ],
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = fix_host_only(out_dir)
        assert result["updated"]

        mf = json.loads((out_dir / "manifest.json").read_text())
        bundles = {b["bundle"]: b for b in mf["bundles"]}
        # Repaired host bundle carries its real post-rewrite on-disk size; the untouched
        # bin is unchanged, and the rewritten header changed the host file's size.
        fixed_size = host_file.stat().st_size
        assert fixed_size != stale_size
        assert bundles["host_global"]["size_bytes"] == fixed_size
        assert bundles["L00/dense_down"]["size_bytes"] == 9216

        # Top-level totals are recomputed as the bundle size_bytes sum.
        total = sum(b["size_bytes"] for b in mf["bundles"])
        assert mf["total_size_bytes"] == total
        assert mf["total_size_bytes"] == 9216 + fixed_size
        assert mf["total_size_mb"] == round(total / (1024 * 1024), 2)
        assert mf["total_size_gb"] == round(total / (1024 * 1024 * 1024), 3)


def test_save_tensors_verified_preserves_original_on_bit_mismatch():
    """Verification failure in _save_tensors_verified must leave the old file untouched."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_file = tmp_path / "host.safetensors"
        original = {"a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        st.save_file(original, str(out_file))
        original_sha = compute_sha256(out_file)

        # Same name but different tensor; expected bits come from the original tensor,
        # so the post-write verification must fail before os.replace() swaps anything.
        bad_dict = {"a": torch.tensor([3.0, 4.0], dtype=torch.float32)}
        expected_bits = {"a": _tensor_bits(original["a"])}

        with pytest.raises(RuntimeError, match="Bit pattern changed"):
            _save_tensors_verified(bad_dict, out_file, expected_bits)

        assert compute_sha256(out_file) == original_sha, "original file must be untouched"


def test_save_tensors_verified_accepts_0d_tensors():
    """Verified safetensors writes preserve dtype, shape, and bits for 0d tensors."""
    torch = pytest.importorskip("torch")
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "host.safetensors"
        tensors = {
            "eta": torch.tensor(0.125, dtype=torch.float32),
            "nan": torch.tensor(float("nan"), dtype=torch.float32),
            "alpha": torch.tensor(0.015625, dtype=torch.bfloat16),
            "step": torch.tensor(7, dtype=torch.int64),
        }
        expected_bits = {k: _tensor_bits(v) for k, v in tensors.items()}

        _save_tensors_verified(tensors, out_file, expected_bits)

        with safe_open(str(out_file), framework="pt") as f:
            for name, original in tensors.items():
                stored = f.get_tensor(name)
                assert stored.dtype == original.dtype
                assert stored.shape == ()
                assert _tensor_bits(stored).tobytes() == _tensor_bits(original).tobytes()


def test_save_tensors_verified_detects_0d_bit_mismatch():
    """0d value and signed-zero bit mismatches fail without replacing the original."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "host.safetensors"
        original = {
            "eta": torch.tensor(0.125, dtype=torch.float32),
            "zero": torch.tensor(0.0, dtype=torch.float32),
        }
        st.save_file(original, str(out_file))
        original_sha = compute_sha256(out_file)
        expected_bits = {k: _tensor_bits(v) for k, v in original.items()}

        bad_value = {
            "eta": torch.tensor(0.25, dtype=torch.float32),
            "zero": torch.tensor(0.0, dtype=torch.float32),
        }
        with pytest.raises(RuntimeError, match="Bit pattern changed"):
            _save_tensors_verified(bad_value, out_file, expected_bits)
        assert compute_sha256(out_file) == original_sha, "original file must be untouched"

        bad_bits = {
            "eta": torch.tensor(0.125, dtype=torch.float32),
            "zero": torch.tensor(-0.0, dtype=torch.float32),
        }
        with pytest.raises(RuntimeError, match="Bit pattern changed"):
            _save_tensors_verified(bad_bits, out_file, expected_bits)
        assert compute_sha256(out_file) == original_sha, "original file must be untouched"


def test_fix_host_only_preserves_original_on_rewrite_failure():
    """If the atomic rewrite inside fix_host_only fails, the original host file stays as-is."""
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        orig_bf16 = torch.tensor([0.1, -2.3, 3.75, 1e-8], dtype=torch.bfloat16)
        u16_bits = orig_bf16.view(torch.int16).numpy().view(np.uint16)
        file_path = out_dir / "host_global.safetensors"
        snp.save_file({"w": u16_bits}, str(file_path))
        original_sha = compute_sha256(file_path)

        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "bundles": [
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": file_path.stat().st_size,
                    "components": [
                        {"name": "w", "shape": [4], "dtype": "bfloat16", "stored_dtype": "uint16"}
                    ],
                    "sha256": compute_sha256(file_path),
                    "description": "test",
                }
            ],
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with patch("export.export_d._save_tensors_verified", side_effect=RuntimeError("verification failed")):
            with pytest.raises(RuntimeError, match="verification failed"):
                fix_host_only(out_dir)

        assert compute_sha256(file_path) == original_sha, "original host file must be preserved"


def test_host_export_missing_tensor_from_shard_fails_and_writes_manifest():
    """A required host tensor present in the index but missing from its shard is a hard failure."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        # Shard contains only one of two tensors referenced by the index.
        orig = {
            "model.layers.0.input_layernorm.weight": torch.tensor([0.1, -2.3], dtype=torch.bfloat16),
        }
        shard = "model.safetensors"
        st.save_file(orig, str(hf_dir / shard))
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.0.input_layernorm.weight": shard,
                        # Declared in the index but absent from the shard -> hard failure.
                        "model.layers.0.post_attention_layernorm.weight": shard,
                    }
                }
            )
        )

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        with pytest.raises(KeyError, match="post_attention_layernorm"):
            exporter.run()

        assert not (out_dir / "manifest.json").exists()


@pytest.mark.skipif(not DEFAULT_GGUF.is_file(), reason="RAON_GGUF is not set or file is unavailable")
def test_verify_d_missing_bundle_name_is_an_error():
    """Requesting a bundle name absent from the manifest must raise, not silently pass 0/0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(
            json.dumps({"version": "1.0", "model": "test", "mode": "test", "bundles": []})
        )

        verifier = DModeVerifier(
            manifest_path=manifest_file,
            weights_dir=tmp_path,
            gguf_path=DEFAULT_GGUF,
        )
        with pytest.raises(ValueError, match="not present in manifest"):
            verifier.verify(bundle_names=["L00/dense_down"])


def test_export_host_tensors_missing_index_requires_single_file_or_fails():
    """No index.json and no model.safetensors -> FileNotFoundError; single model.safetensors works."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        # Neither index nor single file: hard failure.
        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            exporter.run()

        # Single-file model: every tensor is in model.safetensors.
        st.save_file(
            {
                "model.layers.0.input_layernorm.weight": torch.tensor([0.1, -2.3], dtype=torch.bfloat16),
                "model.norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            },
            str(hf_dir / "model.safetensors"),
        )
        exporter2 = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        manifest_data = exporter2.run()
        bundles = {b["bundle"]: b for b in manifest_data["bundles"]}
        assert "L00/host" in bundles
        assert "host_global" in bundles


@pytest.mark.skipif(not DEFAULT_GGUF.is_file(), reason="RAON_GGUF is not set or file is unavailable")
def test_partial_run_merge_recalculates_exported_layers():
    """Running --layers 1 on top of an existing layer-0 manifest reports [0, 1]."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        st.save_file(
            {"model.norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
            str(hf_dir / "model.safetensors"),
        )

        # Pre-existing manifest: layer 0 already exported.
        (out_dir / "L00").mkdir(parents=True, exist_ok=True)
        (out_dir / "L00" / "dense_down.bin").write_bytes(b"\x00" * 128)
        manifest0 = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "num_layers_total": 25,
            "exported_layers": [0],
            "total_size_bytes": 128,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
            "bundles": [
                {
                    "bundle": "L00/dense_down",
                    "layer": 0,
                    "format": "bfp16_gemv",
                    "file": "L00/dense_down.bin",
                    "size_bytes": 128,
                    "components": [],
                    "sha256": compute_sha256(out_dir / "L00" / "dense_down.bin"),
                }
            ],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest0))

        # Second partial run: layer 1 only, exporting a bundle valid for that layer.
        (out_dir / "L01").mkdir(parents=True, exist_ok=True)
        (out_dir / "L01" / "o_proj.bin").write_bytes(b"\x00" * 128)

        exporter1 = DModeExporter(
            gguf_path=DEFAULT_GGUF,
            hf_dir=hf_dir,
            out_dir=out_dir,
            layers=[1],
            only_bundles=["o_proj"],
            export_host=False,
        )
        exporter1.export_lm_head = lambda: None
        exporter1._should_export = lambda name: name == "o_proj"
        exporter1.export_layer = lambda layer: [
            {
                "bundle": f"L{layer:02d}/o_proj",
                "layer": layer,
                "format": "bfp16_gemv",
                "file": f"L{layer:02d}/o_proj.bin",
                "size_bytes": 128,
                "components": [],
                "sha256": compute_sha256(out_dir / f"L{layer:02d}/o_proj.bin"),
            }
        ]
        manifest_data = exporter1.run()

        assert manifest_data["exported_layers"] == [0, 1]
        bundle_names = {b["bundle"] for b in manifest_data["bundles"]}
        assert bundle_names == {"L00/dense_down", "L01/o_proj"}


def test_fix_host_only_refreshes_manifest_when_header_already_bf16():
    """If the actual header is already BF16, --fix-host-only still refreshes manifest metadata."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        # File already has BF16 header.
        file_path = out_dir / "host_global.safetensors"
        st.save_file({"w": torch.tensor([0.1, -2.3], dtype=torch.bfloat16)}, str(file_path))
        real_sha = compute_sha256(file_path)

        # Manifest is stale: wrong stored_dtype and wrong sha256.
        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "total_size_bytes": file_path.stat().st_size,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
            "bundles": [
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": file_path.stat().st_size,
                    "components": [
                        {"name": "w", "shape": [2], "dtype": "bfloat16", "stored_dtype": "U16"}
                    ],
                    "sha256": "stale-sha256",
                }
            ],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest))

        result = fix_host_only(out_dir)
        assert result["updated"]
        mf = json.loads((out_dir / "manifest.json").read_text())
        comp = mf["bundles"][0]["components"][0]
        assert comp["stored_dtype"] == "BF16"
        assert mf["bundles"][0]["sha256"] == real_sha


def test_partial_run_preserves_host_failure_when_host_skipped():
    """A partial run with --skip-host semantics must keep a previous host_tensors_status failure.

    This uses the real _should_export("host") path (no --only override) so the
    host-skipped decision comes from export_host=False, not from a mocked filter.
    """
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "hf"
        out_dir = tmp_path / "out"
        hf_dir.mkdir()
        out_dir.mkdir()

        st.save_file(
            {"model.norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
            str(hf_dir / "model.safetensors"),
        )

        # Pre-existing manifest: layer 0 bundle plus a host failure record.
        (out_dir / "L00").mkdir(parents=True, exist_ok=True)
        (out_dir / "L00" / "dense_down.bin").write_bytes(b"\x00" * 64)
        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "num_layers_total": 25,
            "exported_layers": [0],
            "total_size_bytes": 64,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
            "bundles": [
                {
                    "bundle": "L00/dense_down",
                    "layer": 0,
                    "format": "bfp16_gemv",
                    "file": "L00/dense_down.bin",
                    "size_bytes": 64,
                    "components": [],
                    "sha256": compute_sha256(out_dir / "L00" / "dense_down.bin"),
                }
            ],
            "host_tensors_status": {
                "status": "failed",
                "error": "FileNotFoundError: missing shard",
                "note": "host failed",
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest))

        # Partial run with export_host=False and no --only override.
        # _should_export("host") returns True, but host is not attempted, so the
        # old host_tensors_status must survive.
        (out_dir / "L01").mkdir(parents=True, exist_ok=True)
        (out_dir / "L01" / "dense_down.bin").write_bytes(b"\x00" * 64)
        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[1])
        exporter.export_host = False
        # Host is requested in only_bundles, but export_host=False means it is skipped.
        exporter.only_bundles = {"dense_down", "host"}
        exporter.export_layer = lambda layer: [
            {
                "bundle": f"L{layer:02d}/dense_down",
                "layer": layer,
                "format": "bfp16_gemv",
                "file": f"L{layer:02d}/dense_down.bin",
                "size_bytes": 64,
                "components": [],
                "sha256": compute_sha256(out_dir / f"L{layer:02d}/dense_down.bin"),
            }
        ]
        exporter.export_lm_head = lambda: None
        manifest_data = exporter.run()

        assert manifest_data["exported_layers"] == [0, 1]
        assert manifest_data["host_tensors_status"]["status"] == "failed"


def test_host_export_requested_but_hf_dir_missing_fails_manifest():
    """Host export requested with a missing HF dir must fail and record host_tensors_status."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_dir = tmp_path / "missing_hf"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[])
        # _make_host_only_exporter already sets only_bundles={"host"} and export_host=True.
        with pytest.raises(FileNotFoundError, match="HF directory"):
            exporter.run()

        assert not (out_dir / "manifest.json").exists()


def test_write_host_index_drops_stale_mappings():
    """write_host_index rebuilds weight_map from the manifest, removing stale entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        stale_index = {
            "metadata": {"foo": "bar"},
            "weight_map": {
                "stale.tensor": "old_host.safetensors",
                "model.layers.0.input_layernorm.weight": "L00/host.safetensors",
            },
        }
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(stale_index))
        entries = [
            {
                "bundle": "L00/host",
                "format": "host_safetensors",
                "file": "L00/host.safetensors",
                "components": [
                    {"name": "model.layers.0.input_layernorm.weight"},
                    {"name": "model.layers.0.post_attention_layernorm.weight"},
                ],
            }
        ]
        write_host_index(out_dir, entries, merge=True)
        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        }
        assert index["metadata"] == {"foo": "bar"}


def test_fix_host_only_recovers_after_partial_failure():
    """First file is rewritten, second fails, then rerun recovers manifest/SHA/totals."""
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp
    from unittest.mock import patch
    from export.export_d import _save_tensors_verified as real_save_tensors_verified

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        (out_dir / "L00").mkdir()

        orig1 = torch.tensor([0.1, -2.3, 3.75, 1e-8] * 4, dtype=torch.bfloat16)
        orig2 = torch.tensor([1.0, -1.0, 2.0, -2.0] * 4, dtype=torch.bfloat16)
        u16_1 = orig1.view(torch.int16).numpy().view(np.uint16)
        u16_2 = orig2.view(torch.int16).numpy().view(np.uint16)

        file1 = out_dir / "host_global.safetensors"
        file2 = out_dir / "L00" / "host.safetensors"
        snp.save_file({"w1": u16_1}, str(file1))
        snp.save_file({"w2": u16_2}, str(file2))

        manifest = {
            "version": "1.0",
            "model": "test",
            "mode": "test",
            "total_size_bytes": file1.stat().st_size + file2.stat().st_size,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
            "bundles": [
                {
                    "bundle": "host_global",
                    "layer": None,
                    "format": "host_safetensors",
                    "file": "host_global.safetensors",
                    "size_bytes": file1.stat().st_size,
                    "components": [
                        {"name": "w1", "shape": [16], "dtype": "bfloat16", "stored_dtype": "uint16"}
                    ],
                    "sha256": compute_sha256(file1),
                },
                {
                    "bundle": "L00/host",
                    "layer": 0,
                    "format": "host_safetensors",
                    "file": "L00/host.safetensors",
                    "size_bytes": file2.stat().st_size,
                    "components": [
                        {"name": "w2", "shape": [16], "dtype": "bfloat16", "stored_dtype": "uint16"}
                    ],
                    "sha256": compute_sha256(file2),
                },
            ],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest))

        def fake_save(torch_dict, out_file, expected_bits):
            if Path(out_file).name == "host.safetensors":
                raise RuntimeError("second file rewrite fails")
            return real_save_tensors_verified(torch_dict, out_file, expected_bits)

        with patch("export.export_d._save_tensors_verified", side_effect=fake_save):
            with pytest.raises(RuntimeError, match="second file rewrite fails"):
                fix_host_only(out_dir)

        # Re-run unpatched: first file is already BF16, second is rewritten now.
        result = fix_host_only(out_dir)
        assert result["updated"]

        mf = json.loads((out_dir / "manifest.json").read_text())
        total = sum(b["size_bytes"] for b in mf["bundles"])
        assert mf["total_size_bytes"] == total
        for b in mf["bundles"]:
            assert b["sha256"] == compute_sha256(out_dir / b["file"])
            comp = b["components"][0]
            assert comp["stored_dtype"] == "BF16"
            assert comp["dtype"] == "bfloat16"

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == {"w1", "w2"}
        assert index["weight_map"]["w1"] == "host_global.safetensors"
        assert index["weight_map"]["w2"] == "L00/host.safetensors"


def test_verify_d_host_bundle_is_integrity_only():
    """Host safetensors bundles must be reported as integrity-only, not bit-exact checked."""
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_file = tmp_path / "host_global.safetensors"
        snp.save_file({"w": np.array([1, 2, 3], dtype=np.float32)}, str(host_file))

        entry = {
            "bundle": "host_global",
            "layer": None,
            "format": "host_safetensors",
            "file": "host_global.safetensors",
            "size_bytes": host_file.stat().st_size,
            "components": [
                {"name": "w", "shape": [3], "dtype": "float32", "stored_dtype": "F32"}
            ],
            "sha256": compute_sha256(host_file),
        }
        verifier = object.__new__(DModeVerifier)
        verifier.weights_dir = tmp_path
        verifier.manifest = {"bundles": [entry]}

        res = verifier.verify_bundle(entry)
        assert res["status"] == "PASS"
        assert res["check_type"] == "host_integrity"
        assert res.get("bit_exact") is None
        assert res.get("integrity_only") is True
        assert res["checked_tensors"] == []


def test_verify_d_summary_separates_host_and_npu_counts():
    """Summary must bucket host integrity and NPU bit-exact results separately."""
    verifier = object.__new__(DModeVerifier)
    verifier.weights_dir = Path(tempfile.gettempdir())
    verifier.manifest = {
        "bundles": [
            {"bundle": "host_global", "format": "host_safetensors"},
            {"bundle": "L00/host", "format": "host_safetensors"},
            {"bundle": "L00/rda_in", "format": "bfp16_gemv"},
            {"bundle": "L00/dense_down", "format": "bfp16_gemv"},
            {"bundle": "L00/dense_gu", "format": "q4nx_gemv"},
        ]
    }

    def fake_verify_bundle(entry):
        name = entry["bundle"]
        if name == "host_global":
            return {"bundle": name, "check_type": "host_integrity", "status": "PASS", "errors": [], "checked_tensors": []}
        if name == "L00/host":
            return {"bundle": name, "check_type": "host_integrity", "status": "FAIL", "errors": ["sha mismatch"], "checked_tensors": []}
        if name == "L00/rda_in":
            return {"bundle": name, "check_type": "npu_bit_exact", "status": "PASS", "errors": [], "checked_tensors": ["t1"]}
        if name == "L00/dense_down":
            return {"bundle": name, "check_type": "npu_bit_exact", "status": "FAIL", "errors": ["bit mismatch"], "checked_tensors": []}
        return {"bundle": name, "check_type": "npu_bit_exact", "status": "PASS", "errors": [], "checked_tensors": ["t2"]}

    verifier.verify_bundle = fake_verify_bundle
    summary = verifier.verify(verify_all=True)

    assert summary["integrity_checked"] == 2
    assert summary["integrity_passed"] == 1
    assert summary["integrity_failed"] == 1
    assert summary["bit_exact_checked"] == 3
    assert summary["bit_exact_passed"] == 2
    assert summary["bit_exact_failed"] == 1
    assert summary["failed"] == 2
    assert "passed" not in summary


def test_verify_d_host_sha_failure_counts_as_integrity():
    """A host bundle that fails SHA256 early must be counted as an integrity failure."""
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_file = tmp_path / "host_global.safetensors"
        snp.save_file({"w": np.array([1, 2, 3], dtype=np.uint16)}, str(host_file))

        entry = {
            "bundle": "host_global",
            "layer": None,
            "format": "host_safetensors",
            "file": "host_global.safetensors",
            "size_bytes": host_file.stat().st_size,
            "components": [{"name": "w", "shape": [3], "dtype": "bfloat16", "stored_dtype": "U16"}],
            "sha256": "0" * 64,
        }
        verifier = object.__new__(DModeVerifier)
        verifier.weights_dir = tmp_path
        verifier.manifest = {"bundles": [entry]}

        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert res["check_type"] == "host_integrity"
        assert "SHA256 mismatch" in res["errors"][0]

        verifier.verify_bundle = lambda _e: res
        summary = verifier.verify(bundle_names=["host_global"])
        assert summary["integrity_failed"] == 1
        assert summary["integrity_passed"] == 0
        assert summary["bit_exact_checked"] == 0


def test_verify_d_host_header_dtype_shape_pass():
    """Host verification passes when manifest stored_dtype/shape match the header."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_file = tmp_path / "host_global.safetensors"
        tensors = {
            "w_f32": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "w_bf16": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        }
        st.save_file(tensors, str(host_file))

        entry = {
            "bundle": "host_global",
            "layer": None,
            "format": "host_safetensors",
            "file": "host_global.safetensors",
            "size_bytes": host_file.stat().st_size,
            "components": [
                {"name": "w_f32", "shape": [2], "dtype": "float32", "stored_dtype": "F32"},
                {"name": "w_bf16", "shape": [3], "dtype": "bfloat16", "stored_dtype": "BF16"},
            ],
            "sha256": compute_sha256(host_file),
        }
        verifier = object.__new__(DModeVerifier)
        verifier.weights_dir = tmp_path
        verifier.manifest = {"bundles": [entry]}

        res = verifier.verify_bundle(entry)
        assert res["status"] == "PASS"
        assert res["check_type"] == "host_integrity"


def test_verify_d_host_header_dtype_shape_fail():
    """Host verification fails when manifest dtype/shape disagree with the real header.

    A bfloat16 tensor must have a BF16 header regardless of what stored_dtype says.
    """
    torch = pytest.importorskip("torch")
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_file = tmp_path / "host_global.safetensors"
        bf16_bits = (
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
        )
        tensors = {
            "w_f32": np.array([1.0, 2.0], dtype=np.float32),
            "w_bf16": bf16_bits,
        }
        snp.save_file(tensors, str(host_file))

        entry = {
            "bundle": "host_global",
            "layer": None,
            "format": "host_safetensors",
            "file": "host_global.safetensors",
            "size_bytes": host_file.stat().st_size,
            "components": [
                {"name": "w_f32", "shape": [2], "dtype": "float32", "stored_dtype": "float32"},
                {"name": "w_bf16", "shape": [3], "dtype": "bfloat16", "stored_dtype": "uint16"},
                {"name": "w_f32", "shape": [5], "dtype": "float32", "stored_dtype": "F32"},
            ],
            "sha256": compute_sha256(host_file),
        }
        verifier = object.__new__(DModeVerifier)
        verifier.weights_dir = tmp_path
        verifier.manifest = {"bundles": [entry]}

        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert res["check_type"] == "host_integrity"
        errors = " ".join(res["errors"])
        assert "dtype mismatch" in errors
        assert "shape mismatch" in errors
        # The bfloat16 tensor is recorded as stored_dtype uint16 but the real header is U16,
        # which must be rejected because the original dtype requires BF16.
        assert "BF16" in errors


def test_verify_d_host_header_unknown_or_missing_dtype_fails():
    """Host verification fails when manifest dtype/stored_dtype is unknown or missing."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_file = tmp_path / "host_global.safetensors"
        st.save_file(
            {
                "w_unknown": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "w_missing": torch.tensor([3.0, 4.0], dtype=torch.float32),
            },
            str(host_file),
        )

        entry = {
            "bundle": "host_global",
            "layer": None,
            "format": "host_safetensors",
            "file": "host_global.safetensors",
            "size_bytes": host_file.stat().st_size,
            "components": [
                {"name": "w_unknown", "shape": [2], "dtype": "complex64"},
                {"name": "w_missing", "shape": [2]},
            ],
            "sha256": compute_sha256(host_file),
        }
        verifier = object.__new__(DModeVerifier)
        verifier.weights_dir = tmp_path
        verifier.manifest = {"bundles": [entry]}

        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert res["check_type"] == "host_integrity"
        errors = " ".join(res["errors"])
        assert "w_unknown" in errors
        assert "w_missing" in errors
        assert "complex64" in errors
        assert "not a recognized dtype" in errors


def test_export_d_moe_b1_keeps_existing_file_on_mid_failure(monkeypatch):
    """If an expert conversion fails mid-write, the previous moe_b1.bin must stay intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        layer_dir = out_dir / "L00"
        layer_dir.mkdir()
        existing_file = layer_dir / "moe_b1.bin"
        existing_file.write_text("ORIGINAL")

        class FakeInfo:
            type_name = "Q4_K"

        call_count = 0

        class FakeReader:
            tensors = {
                "blk.0.ffn_gate_exps.weight": FakeInfo(),
                "blk.0.ffn_up_exps.weight": FakeInfo(),
            }

            def q4nx_components(self, name, outer_index=None):
                return (
                    np.zeros((2, 2), dtype=np.uint8),
                    np.zeros((2, 1), dtype=np.float32),
                    np.zeros((2, 1), dtype=np.float32),
                )

        def fake_pack(codes, scales, mins, rows):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("simulated mid-write expert failure")
            return b"\x00" * 3481600

        monkeypatch.setattr("export.export_d.pack_gemv", fake_pack)

        exporter = object.__new__(DModeExporter)
        exporter.out_dir = out_dir
        exporter.reader = FakeReader()

        with pytest.raises(RuntimeError, match="simulated mid-write expert failure"):
            exporter.export_moe_b1(0)

        assert existing_file.read_text() == "ORIGINAL"
        assert not any(layer_dir.glob(".*.tmp"))
        assert call_count >= 3


def _make_two_layer_hf(out_dir: Path) -> Path:
    """Create a tiny HF directory with two layers + global host tensors."""
    torch = pytest.importorskip("torch")
    import safetensors.torch as st

    hf_dir = out_dir.parent / "hf"
    hf_dir.mkdir()
    st.save_file(
        {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "model.layers.1.input_layernorm.weight": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            "model.norm.weight": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
        },
        str(hf_dir / "model.safetensors"),
    )
    return hf_dir


def _write_two_layer_base_manifest(out_dir: Path) -> dict:
    """Write a manifest with NPU bundles for layers 0 and 1 and a prior host failure."""
    (out_dir / "L00").mkdir()
    (out_dir / "L01").mkdir()
    (out_dir / "L00" / "dense_down.bin").write_bytes(b"\x00" * 64)
    (out_dir / "L01" / "dense_down.bin").write_bytes(b"\x00" * 64)
    manifest = {
        "version": "1.0",
        "model": "test",
        "mode": "test",
        "num_layers_total": 25,
        "exported_layers": [0, 1],
        "total_size_bytes": 128,
        "total_size_mb": 0.0,
        "total_size_gb": 0.0,
        "bundles": [
            {
                "bundle": "L00/dense_down",
                "layer": 0,
                "format": "bfp16_gemv",
                "file": "L00/dense_down.bin",
                "size_bytes": 64,
                "components": [],
                "sha256": compute_sha256(out_dir / "L00" / "dense_down.bin"),
            },
            {
                "bundle": "L01/dense_down",
                "layer": 1,
                "format": "bfp16_gemv",
                "file": "L01/dense_down.bin",
                "size_bytes": 64,
                "components": [],
                "sha256": compute_sha256(out_dir / "L01" / "dense_down.bin"),
            },
        ],
        "host_tensors_status": {
            "status": "failed",
            "error": "RuntimeError: earlier host export failed before L01/host",
            "note": "host failed before L01/host",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_partial_host_success_keeps_failure_status_when_other_layers_missing():
    """Exporting host for only layer 0 must not erase a prior failure for layer 1."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        hf_dir = _make_two_layer_hf(out_dir)
        _write_two_layer_base_manifest(out_dir)

        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        exporter.export_layer = lambda layer: []
        exporter.export_lm_head = lambda: None
        manifest_data = exporter.run()

        assert manifest_data["exported_layers"] == [0, 1]
        assert manifest_data["host_tensors_status"]["status"] == "failed"
        note = manifest_data["host_tensors_status"]["note"]
        assert "L01/host" in note
        assert (out_dir / "L00" / "host.safetensors").is_file()
        assert (out_dir / "host_global.safetensors").is_file()
        assert not (out_dir / "L01" / "host.safetensors").is_file()


def test_completing_missing_host_layer_clears_failure_status():
    """After the missing layer 1 host is exported, the failure status must be removed."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        hf_dir = _make_two_layer_hf(out_dir)
        _write_two_layer_base_manifest(out_dir)

        exporter0 = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        exporter0.export_layer = lambda layer: []
        exporter0.export_lm_head = lambda: None
        exporter0.run()

        exporter1 = _make_host_only_exporter(hf_dir, out_dir, layers=[1])
        exporter1.export_layer = lambda layer: []
        exporter1.export_lm_head = lambda: None
        manifest_data = exporter1.run()

        assert manifest_data["exported_layers"] == [0, 1]
        assert "host_tensors_status" not in manifest_data
        assert (out_dir / "L00" / "host.safetensors").is_file()
        assert (out_dir / "L01" / "host.safetensors").is_file()
        assert (out_dir / "host_global.safetensors").is_file()


def test_corrupt_host_file_sha_keeps_failure_status():
    """A sha256 mismatch in an existing host file must prevent clearing the failure status."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        hf_dir = _make_two_layer_hf(out_dir)

        # First export both layer hosts successfully.
        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0, 1])
        exporter.export_layer = lambda layer: []
        exporter.export_lm_head = lambda: None
        manifest_ok = exporter.run()
        assert "host_tensors_status" not in manifest_ok

        # Inject a prior failure status and corrupt L00/host.safetensors.
        mf = json.loads((out_dir / "manifest.json").read_text())
        mf["host_tensors_status"] = {
            "status": "failed",
            "error": "RuntimeError: simulated earlier host failure",
            "note": "host failed earlier",
        }
        (out_dir / "manifest.json").write_text(json.dumps(mf))
        l00_host = out_dir / "L00" / "host.safetensors"
        l00_host.write_bytes(l00_host.read_bytes() + b"\xff")

        # Re-exporting layer 1 host leaves L00/host corrupted, so status must survive.
        exporter1 = _make_host_only_exporter(hf_dir, out_dir, layers=[1])
        exporter1.export_layer = lambda layer: []
        exporter1.export_lm_head = lambda: None
        manifest_data = exporter1.run()

        assert manifest_data["host_tensors_status"]["status"] == "failed"
        assert "RuntimeError: simulated earlier host failure" in manifest_data["host_tensors_status"]["error"]
        note = manifest_data["host_tensors_status"]["note"]
        assert "L00/host" in note



# ---------------------------------------------------------------------------
# 10th round: "must fail but passes" defaults
# ---------------------------------------------------------------------------

def _host_manifest(out_dir: Path, bundles: list[dict]) -> Path:
    manifest = {"version": "1.0", "model": "test", "mode": "test", "bundles": bundles}
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return mp


def _host_bundle(name: str, file_path: Path, rel: str, components: list[dict]) -> dict:
    return {
        "bundle": name,
        "layer": None,
        "format": "host_safetensors",
        "file": rel,
        "size_bytes": file_path.stat().st_size,
        "components": components,
        "sha256": compute_sha256(file_path),
    }


@pytest.mark.parametrize(
    "header_tensor, manifest_shape, expected_error",
    [
        # F32 header (same element count) for a tensor declared bfloat16: dtype rejected.
        ("f32_same_count", [4], "header dtype"),
        # F32 header holding half as many elements as a [4] bfloat16 tensor: shape rejected.
        ("f32_wrong_shape", [4], "Shape mismatch"),
        # U16 header but different shape from the manifest: shape rejected.
        ("u16_wrong_shape", [8], "Shape mismatch"),
    ],
)
def test_fix_host_only_rejects_bad_header_and_leaves_files_untouched(
    header_tensor, manifest_shape, expected_error
):
    """F32 / wrong-shape headers fail BEFORE any file (even an earlier valid U16 one) changes."""
    pytest.importorskip("torch")
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        good = out_dir / "a_good.safetensors"  # legacy U16, same shape: would be rewritten
        snp.save_file({"g": np.array([1, 2, 3, 4], dtype=np.uint16)}, str(good))

        bad = out_dir / "b_bad.safetensors"
        if header_tensor == "f32_same_count":
            snp.save_file({"w": np.array([1, 2, 3, 4], dtype=np.float32)}, str(bad))
        elif header_tensor == "f32_wrong_shape":
            snp.save_file({"w": np.array([1, 2], dtype=np.float32)}, str(bad))
        else:
            snp.save_file({"w": np.array([1, 2, 3, 4], dtype=np.uint16)}, str(bad))

        bundles = [
            _host_bundle(
                "a_good", good, "a_good.safetensors",
                [{"name": "g", "shape": [4], "dtype": "bfloat16", "stored_dtype": "uint16"}],
            ),
            _host_bundle(
                "b_bad", bad, "b_bad.safetensors",
                [{"name": "w", "shape": manifest_shape, "dtype": "bfloat16", "stored_dtype": "uint16"}],
            ),
        ]
        mp = _host_manifest(out_dir, bundles)
        before = {p.name: p.read_bytes() for p in (good, bad, mp)}

        with pytest.raises(RuntimeError, match=expected_error):
            fix_host_only(out_dir)

        after = {p.name: p.read_bytes() for p in (good, bad, mp)}
        assert after == before, "no file (nor the manifest) may change when preflight fails"
        assert not list(out_dir.glob(".*.tmp")), "no leftover temp files"


def test_fix_host_only_non_bfloat16_dtype_mismatch_fails_before_touching_files():
    pytest.importorskip("torch")
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        f = out_dir / "host_global.safetensors"
        snp.save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, str(f))
        bundles = [
            _host_bundle(
                "host_global", f, "host_global.safetensors",
                [{"name": "w", "shape": [2], "dtype": "float16", "stored_dtype": "F16"}],
            )
        ]
        mp = _host_manifest(out_dir, bundles)
        before = (f.read_bytes(), mp.read_bytes())
        with pytest.raises(RuntimeError, match="does not match header dtype"):
            fix_host_only(out_dir)
        assert (f.read_bytes(), mp.read_bytes()) == before


def _make_verifier_without_gguf(manifest_path: Path, entries: list[dict]) -> DModeVerifier:
    verifier = object.__new__(DModeVerifier)
    verifier.manifest_path = manifest_path
    verifier.weights_dir = manifest_path.parent
    verifier.manifest = {"bundles": entries}
    return verifier


def _unknown_format_bundle(tmp_path: Path) -> dict:
    bin_file = tmp_path / "typo.bin"
    bin_file.write_bytes(b"\x00" * 64)
    return {
        "bundle": "L00/typo",
        "layer": 0,
        "format": "bfp16_gemv_typo",
        "file": "typo.bin",
        "size_bytes": 64,
        "original_shape": [8, 8],
        "padded_shape": [8, 8],
        "components": [],
        "gguf_types": [],
        "sha256": compute_sha256(bin_file),
    }


def test_verify_d_unknown_npu_format_fails_with_format_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        entry = _unknown_format_bundle(tmp_path)
        verifier = _make_verifier_without_gguf(tmp_path / "manifest.json", [entry])

        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert res["check_type"] == "npu_bit_exact"
        assert res["bit_exact"] is False
        assert any("bfp16_gemv_typo" in e for e in res["errors"])

        summary = verifier.verify(sample_count=None)
        assert summary["failed"] == 1
        assert summary["bit_exact_failed"] == 1


def test_verify_d_cli_exits_nonzero_for_unknown_npu_format(monkeypatch):
    import sys

    import export.verify_d as vd

    class _Dummy:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(vd, "GGUFReader", _Dummy)
    monkeypatch.setattr(vd, "GGUFWeightReader", _Dummy)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        entry = _unknown_format_bundle(tmp_path)
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps({"bundles": [entry]}))
        monkeypatch.setattr(sys, "argv", ["verify_d", "--manifest", str(mp), "--all"])
        with pytest.raises(SystemExit) as exc:
            vd.main()
        assert exc.value.code not in (0, None)


def test_verify_d_npu_bundle_with_nothing_compared_fails():
    """A known format with no components compares nothing and must not PASS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bin_file = tmp_path / "x.bin"
        bin_file.write_bytes(b"\x00" * 36864)  # one 128x256 bfp16 GEMV tile group
        entry = {
            "bundle": "x", "layer": 0, "format": "bfp16_gemv", "file": "x.bin",
            "size_bytes": 36864, "original_shape": [128, 256], "padded_shape": [128, 256],
            "components": [], "gguf_types": [], "sha256": compute_sha256(bin_file),
        }
        verifier = _make_verifier_without_gguf(tmp_path / "manifest.json", [entry])
        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert any("No tensors were bit-compared" in e for e in res["errors"])


def test_verify_d_empty_or_host_only_selection_and_failed_host_status_fail(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        verifier = _make_verifier_without_gguf(tmp_path / "manifest.json", [])
        with pytest.raises(ValueError, match="No bundles selected"):
            verifier.verify(sample_count=None)

        import safetensors.numpy as snp

        f = tmp_path / "host_global.safetensors"
        snp.save_file({"w": np.array([1.0], dtype=np.float32)}, str(f))
        entry = _host_bundle(
            "host_global", f, "host_global.safetensors",
            [{"name": "w", "shape": [1], "dtype": "float32", "stored_dtype": "F32"}],
        )
        verifier = _make_verifier_without_gguf(tmp_path / "manifest.json", [entry])
        # Host integrity alone cannot establish NPU bit-exactness.
        assert verifier.verify(sample_count=0)["failed"] == 1
        assert "FAIL: no NPU bundles were bit-exact checked" in capsys.readouterr().out
        verifier.manifest["host_tensors_status"] = {"status": "failed", "error": "x"}
        assert verifier.verify(sample_count=None)["failed"] == 2
        # Explicit selection cannot suppress either whole-run failure condition.
        assert verifier.verify(bundle_names=["host_global"])["failed"] == 2
        assert "FAIL: manifest records host_tensors_status=" in capsys.readouterr().out


def test_verify_d_host_component_without_shape_fails():
    import safetensors.numpy as snp

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        f = tmp_path / "host_global.safetensors"
        snp.save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, str(f))
        entry = _host_bundle(
            "host_global", f, "host_global.safetensors",
            [{"name": "w", "dtype": "float32", "stored_dtype": "F32"}],
        )
        verifier = _make_verifier_without_gguf(tmp_path / "manifest.json", [entry])
        res = verifier.verify_bundle(entry)
        assert res["status"] == "FAIL"
        assert any("shape" in e for e in res["errors"])


def test_host_completeness_requires_host_global_bundle():
    """A global tensor smuggled into another bundle must not satisfy completeness."""
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        hf_dir = _make_two_layer_hf(out_dir)
        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        entries = exporter.export_host_tensors([0])
        assert {e["bundle"] for e in entries} == {"L00/host", "host_global"}

        # Complete set: nothing missing.
        assert exporter._verify_host_tensors_complete([0], entries) == ([], [], [])

        # Move the global tensor's component into L00/host and drop host_global.
        global_entry = next(e for e in entries if e["bundle"] == "host_global")
        l00 = next(e for e in entries if e["bundle"] == "L00/host")
        tampered_l00 = dict(l00, components=l00["components"] + global_entry["components"])
        merged = [tampered_l00]
        missing_bundles, invalid_bundles, missing_tensors = exporter._verify_host_tensors_complete([0], merged)
        assert missing_bundles == ["host_global"]
        assert invalid_bundles == []


def test_host_completeness_without_hf_dir_is_incomplete_not_complete():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        exporter = _make_host_only_exporter(Path(tmpdir) / "no_such_hf", out_dir, layers=[0])
        missing_bundles, invalid_bundles, missing_tensors = exporter._verify_host_tensors_complete([0], [])
        assert missing_bundles or invalid_bundles or missing_tensors


def test_hf_index_without_weight_map_is_an_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dir = Path(tmpdir) / "hf"
        hf_dir.mkdir()
        (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
        exporter = _make_host_only_exporter(hf_dir, Path(tmpdir) / "out", layers=[0])
        with pytest.raises(ValueError, match="weight_map"):
            exporter._load_hf_weight_map()
        (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="weight_map"):
            exporter._load_hf_weight_map()


def test_exporter_rejects_unknown_only_bundle_name():
    with pytest.raises(ValueError, match="Unknown bundle"):
        DModeExporter(gguf_path="/nonexistent.gguf", hf_dir=None, out_dir="unused-output", only_bundles=["dense_dwon"])


@pytest.mark.parametrize(
    ("layers", "only_bundles", "export_host", "message"),
    [
        ([], ["host"], True, "zero-layer"),
        ([25], ["o_proj"], False, "out of range"),
        ([1], ["dense_down"], False, "produces no bundles"),
        ([0], ["host"], False, "produces no bundles"),
        ([0], [], False, "empty"),
    ],
)
def test_exporter_rejects_empty_or_inapplicable_selection_before_gguf_open(
    layers, only_bundles, export_host, message
):
    with pytest.raises(ValueError, match=message):
        DModeExporter(
            gguf_path="/nonexistent.gguf",
            hf_dir=None,
            out_dir="unused-output",
            layers=layers,
            only_bundles=only_bundles,
            export_host=export_host,
        )


def test_run_refuses_to_overwrite_unreadable_existing_manifest():
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        hf_dir = _make_two_layer_hf(out_dir)
        manifest_path = out_dir / "manifest.json"
        npu_path = out_dir / "L00" / "dense_down.bin"
        host_path = out_dir / "host_global.safetensors"
        index_path = out_dir / "model.safetensors.index.json"
        npu_path.parent.mkdir()
        manifest_path.write_bytes(b"{not json")
        npu_path.write_bytes(b"OLD-NPU")
        host_path.write_bytes(b"OLD-HOST")
        index_path.write_bytes(b'{"metadata":{},"weight_map":{}}')
        before = {p: p.read_bytes() for p in (manifest_path, npu_path, host_path, index_path)}
        exporter = _make_host_only_exporter(hf_dir, out_dir, layers=[0])
        exporter.export_layer = lambda layer: []
        exporter.export_lm_head = lambda: None
        with pytest.raises(RuntimeError, match="refusing to touch"):
            exporter.run()
        assert {p: p.read_bytes() for p in before} == before


def test_fix_host_only_rejects_manifest_without_host_bundles_and_cli_fails():
    """A zero-host repair must fail without changing the manifest or existing index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        manifest_path = out_dir / "manifest.json"
        index_path = out_dir / "model.safetensors.index.json"
        manifest_path.write_bytes(
            json.dumps({"bundles": [{
                "bundle": "L00/dense_down", "layer": 0, "format": "bfp16_gemv",
                "file": "L00/dense_down.bin", "size_bytes": 3, "components": [],
            }]}).encode()
        )
        index_path.write_bytes(b'{"metadata":{"keep":true},"weight_map":{"old":"old.safetensors"}}')
        before = (manifest_path.read_bytes(), index_path.read_bytes())

        with pytest.raises(RuntimeError, match="no host_safetensors bundles"):
            fix_host_only(out_dir)
        assert (manifest_path.read_bytes(), index_path.read_bytes()) == before

        proc = subprocess.run(
            [sys.executable, "-m", "export.export_d", "--fix-host-only", str(out_dir)],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode != 0
        assert "no host_safetensors bundles" in proc.stderr
        assert (manifest_path.read_bytes(), index_path.read_bytes()) == before


def test_write_host_index_requires_file_for_host_bundle():
    from export.export_d import write_host_index

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="no 'file'"):
            write_host_index(
                Path(tmpdir),
                [{"bundle": "host_global", "format": "host_safetensors",
                  "components": [{"name": "w"}]}],
            )
