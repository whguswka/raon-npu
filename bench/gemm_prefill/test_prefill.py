# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from bench.gemm_prefill.reference import bfp16_batch_reference, mmul_reference, unpack_bfp16
from bench.gemm_prefill import prototype
from bench.rda_state.rda_state_multi import check_rda_tokens
from kernels.gemm_prefill.budget import (
    CORE_PROGRAM_LIMIT_BYTES,
    resource_budget,
    supported_batches,
    validate_link_usage,
)
from kernels.gemm_prefill.kernel import compile_flags
from kernels.gemm_prefill.op import PrefillGEMM
from kernels.gemm_prefill.shapes import (
    MAX_BATCH_T,
    PREFILL_SHAPE_NAMES,
    SHAPES,
    rtp_table,
    validate_batch_t,
)
from npu.backend import PREFILL_GEMM_MAX_T
from quant.bfp16 import pack_bfp16_gemv
from quant.q4nx import round_to_bf16_rne


def _raw_block(seed=41):
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((128, 256), dtype=np.float32)
    return pack_bfp16_gemv(weights)[:9216]


@pytest.mark.parametrize("t", (1, 2, 3, 23, 24))
def test_exact_mmul_matches_decode_reference_bits(t):
    weights = unpack_bfp16(_raw_block())
    rng = np.random.default_rng(100 + t)
    x = round_to_bf16_rne(rng.standard_normal((t, 256), dtype=np.float32))
    expected = bfp16_batch_reference(weights, x)
    # mmul_reference is bf16-valued float32; take its upper BF16 bits.
    actual = (mmul_reference(weights, x).view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)


def test_wrong_unpack_scale_is_detected():
    weights = unpack_bfp16(_raw_block())
    bad = weights * np.float32(2.0)
    x = round_to_bf16_rne(np.ones((2, 256), np.float32))
    assert not np.array_equal(mmul_reference(weights, x), mmul_reference(bad, x))


def test_kernel_bfp16_scale_matches_packed_format_contract():
    # pack_bfp16_gemv stores scale bits as (max(exp, 7) - 6) << 7.  Pin the
    # kernel operations so NumPy-only parity cannot hide a changed device scale.
    source = (Path(__file__).resolve().parents[2] / "kernels/gemm_prefill/mmul.cc").read_text()
    compact = re.sub(r"\s+", "", source)
    assert "aie::max(aie::load_v<16>(p->exp),aie::broadcast<uint8,16>(7))" in compact
    assert "aie::sub(e.unpack(),aie::broadcast<uint16,16>(6))" in compact
    assert "aie::upshift(biased,7).templatecast_to<bfloat16>()" in compact


def test_stage1_shape_registry_and_rtp_contract():
    assert tuple(shape.name for shape in SHAPES) == PREFILL_SHAPE_NAMES
    for shape in SHAPES:
        table = rtp_table(shape)
        assert len(table) == 32
        assert {entry[0] for entry in table} == {shape.k_blocks}
        assert tuple(entry[1] for entry in table[::4]) == shape.rounds_by_col(8)


@pytest.mark.parametrize("shape", SHAPES)
def test_fixed_t24_budget(shape):
    budget = resource_budget(shape, 24, cols=7)
    assert budget.limits_ok
    assert budget.memtile_total_bytes == 4 * 2 * 9216 + 2 * 24 * 128 * 2 == 86016
    assert budget.core_total_bytes == 62472
    assert supported_batches(shape, cols=7) == (24,)


@pytest.mark.parametrize("bad", (0, 25, True, 1.5))
def test_actual_t_validation(bad):
    with pytest.raises(ValueError):
        validate_batch_t(bad)


def test_one_xclbin_and_one_instruction_name_per_shape():
    ops = [PrefillGEMM(name, t) for name in PREFILL_SHAPE_NAMES for t in (1, 24)]
    assert len({op.config_name for op in ops}) == 1
    for shape_name in PREFILL_SHAPE_NAMES:
        names = {op.name for op in ops if op.shape_name == shape_name}
        assert len(names) == 1
    assert "s2a" in ops[0].config_name and "s1k" not in ops[0].config_name


def test_7col_rda_prototype_has_distinct_plate_name_and_abi():
    gemm = PrefillGEMM("rda_in", 24, cols=7, rda_col=7)
    rda = PrefillGEMM("rda_in", 24, cols=7, rda_col=7, execution="rda")
    assert gemm.config_name == rda.config_name
    assert "c7_rda7t8" in gemm.config_name
    assert len(gemm.get_arg_spec()) == 3
    assert len(rda.get_arg_spec()) == 4


def test_rda_stream_names_share_plate_and_identify_token_count():
    gemm = PrefillGEMM("rda_in", 24, cols=7, rda_col=7)
    rda_ops = [
        PrefillGEMM("rda_in", 24, cols=7, rda_col=7, execution="rda", rda_tokens=t)
        for t in (1, 2, 4, 8)
    ]
    assert "c7_rda7t8" in gemm.config_name
    assert {op.config_name for op in rda_ops} == {gemm.config_name}
    assert [op.name.endswith(f"_rda_t{t}_mode0") for op, t in zip(rda_ops, (1, 2, 4, 8))] == [True] * 4
    assert len({op.name for op in rda_ops}) == 4


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_rda_argument_spec_tracks_token_count(tokens):
    op = PrefillGEMM("rda_in", 24, execution="rda", rda_tokens=tokens)
    specs = op.get_arg_spec()
    assert [(spec.direction, spec.shape, spec.dtype) for spec in specs] == [
        ("in", (40, 128, 128), np.float32),
        ("out", (40, 128, 128), np.float32),
        ("in", (40, tokens, 640), np.float32),
        ("out", (40, 4, tokens, 32), np.float32),
    ]


@pytest.mark.parametrize("bad", (0, 3, 16, True))
def test_rda_operator_rejects_invalid_token_count(bad):
    with pytest.raises(ValueError):
        PrefillGEMM("rda_in", 24, execution="rda", rda_tokens=bad)


def test_gemm_operator_rejects_rda_token_count_and_for_shape_forwards_it():
    with pytest.raises(ValueError):
        PrefillGEMM("rda_in", 24, rda_tokens=2)
    op = PrefillGEMM.for_shape(
        "rda_in", batch_t=24, execution="rda", rda_tokens=4
    )
    assert op.rda_tokens == 4


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_check_rda_tokens_accepts_supported_values(tokens):
    assert check_rda_tokens(tokens) == tokens


@pytest.mark.parametrize("bad", (0, 3, 16, 8.0, True, "8"))
def test_check_rda_tokens_rejects_other_values(bad):
    with pytest.raises(ValueError):
        check_rda_tokens(bad)


def test_default_operator_is_the_7col_rda_plate():
    op = PrefillGEMM("k_b_v_b", 24)
    assert (op.cols, op.rda_col, op.execution) == (7, 7, "gemm")
    assert "s2a" in op.config_name


def test_compile_flags_are_exact_bfp16_only():
    flags = compile_flags()
    assert "-DPREFILL_MAX_T=24" in flags
    assert "-UAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" in flags
    with pytest.raises(ValueError):
        compile_flags(16)
    with pytest.raises(ValueError):
        compile_flags(24, "emulated", 0)


def test_config_name_pins_physical_t24():
    config_name = PrefillGEMM.for_shape("rda_in", batch_t=24).config_name
    assert "tmax24" in config_name
    assert "tmax16" not in config_name


def test_backend_and_kernel_max_t_match():
    assert PREFILL_GEMM_MAX_T == MAX_BATCH_T


def test_mmul_source_pins_32_lane_unpack_and_required_max_t_flag():
    source = (Path(__file__).resolve().parents[2] / "kernels/gemm_prefill/mmul.cc").read_text()
    unpack = source[source.index("const Packed *w="):source.rindex("#endif")]
    assert "for (int k=0; k<8; k+=2)" in unpack
    assert "aie::concat(aie::load_v<16>(p->mant[k])," in unpack
    assert "aie::load_v<16>(p->mant[k]),0));" not in unpack
    assert "#define PREFILL_MAX_T 16" not in source
    assert '#error "PREFILL_MAX_T must come from kernels/gemm_prefill/kernel.py compile_flags' in source


def test_link_budget_gate():
    validate_link_usage(core_bytes=65536, memtile_bytes=524288,
                        text_bytes=CORE_PROGRAM_LIMIT_BYTES, stack_bytes=12288)
    with pytest.raises(ValueError):
        validate_link_usage(core_bytes=65537, memtile_bytes=1,
                            text_bytes=1, stack_bytes=1)
    with pytest.raises(ValueError):
        validate_link_usage(core_bytes=1, memtile_bytes=1,
                            text_bytes=CORE_PROGRAM_LIMIT_BYTES + 1, stack_bytes=1)


def test_aie2p_program_memory_limit_is_16_kib():
    assert CORE_PROGRAM_LIMIT_BYTES == 0x4000
    assert resource_budget("rda_in", 24).program_limit_bytes == 0x4000


def test_artifact_collection_is_scoped_to_own_plate(tmp_path, monkeypatch):
    xclbin = tmp_path / "plate.xclbin"
    xclbin.touch()
    own = tmp_path / "plate.mlir.d" / "elfs_main_core_0_2"
    other = tmp_path / "other.mlir.d" / "elfs_main_core_7_5"
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    (own / "core.elf").touch()
    (other / "foreign.elf").touch()
    monkeypatch.setattr(prototype, "section_sizes", lambda path: {".text": 11})
    monkeypatch.setattr(prototype, "text_size", lambda path: 11)

    report = prototype._artifact_rows(
        "prefill-7col-rda", SimpleNamespace(filename=str(xclbin))
    )

    assert report["build_root"] == str(tmp_path / "plate.mlir.d")
    assert [(row["col"], row["row"], row["role"])
            for row in report["elf"]] == [(0, 2, "gemm")]
    assert "foreign.elf" not in repr(report)


def test_addressed_mlir_counts_allocated_resources_by_tile(tmp_path):
    mlir = tmp_path / "input_with_addresses.mlir"
    mlir.write_text(
        """
%shim = aie.tile(0, 0)
%memtile = aie.tile(0, 1)
%core = aie.tile(0, 2)
%buf = aie.buffer(%core) {address = 1024 : i32} : memref<256xi32>
%lock0 = aie.lock(%core, 3)
%lock1 = aie.lock(%core, 3)
%mem = aie.mem(%core) {
  ^bb0:
    aie.dma_bd(%buf : memref<256xi32>) {bd_id = 4 : i32}
    aie.dma_bd(%buf : memref<256xi32>) {bd_id = 5 : i32}
}
aie.shim_dma_allocation @host_in(%shim, MM2S, 0)
aiex.runtime_sequence() {
  %task = aiex.dma_configure_task_for @host_in {
    aie.dma_bd(%arg0 : memref<256xi32>) {bd_id = 7 : i32}
  }
}
""",
        encoding="utf-8",
    )

    rows = prototype._address_resources(mlir)
    by_coord = {(row["col"], row["row"]): row for row in rows}
    assert by_coord[(0, 2)]["buffer_end_bytes"] == 2048
    assert by_coord[(0, 2)]["bd_ids"] == [4, 5]
    assert by_coord[(0, 2)]["lock_ids"] == [3]
    assert by_coord[(0, 0)]["bd_ids"] == [7]


def test_addressed_mlir_counts_live_shim_bds_without_assigned_ids(tmp_path):
    mlir = tmp_path / "input_with_addresses.mlir"
    mlir.write_text(
        """
%shim = aie.tile(0, 0)
aie.shim_dma_allocation @host_in(%shim, MM2S, 0)
aie.runtime_sequence() {
  %first = aiex.dma_configure_task_for @host_in {
    aie.dma_bd(%arg0 : memref<256xi32>)
  }
  %second = aiex.dma_configure_task_for @host_in {
    aie.dma_bd(%arg0 : memref<256xi32>)
  }
  aiex.dma_free_task(%first)
  %third = aiex.dma_configure_task_for @host_in {
    aie.dma_bd(%arg0 : memref<256xi32>)
  }
  aiex.dma_free_task(%second)
  aiex.dma_free_task(%third)
}
""",
        encoding="utf-8",
    )

    rows = prototype._address_resources(mlir)

    assert rows[0]["kind"] == "shim"
    assert rows[0]["bd_ids"] == []
    assert rows[0]["bd_count"] == 2


def _audit_report(allocation, *, text_bytes=1):
    return {
        "elf": [{"text_bytes": text_bytes}],
        "addressed_mlir": [{"allocations": [allocation]}],
    }


@pytest.mark.parametrize(
    "kind,field,value",
    (
        ("core", "buffer_end_bytes", 65537),
        ("core", "bd_count", 17),
        ("core", "lock_count", 17),
        ("memtile", "buffer_end_bytes", 524289),
        ("memtile", "bd_count", 49),
        ("memtile", "lock_count", 65),
        ("shim", "bd_count", 17),
        ("shim", "bd_count", 0),
    ),
)
def test_link_audit_rejects_each_tile_limit_and_zero_shim_bd(kind, field, value):
    allocation = {
        "kind": kind,
        "buffer_end_bytes": 1,
        "bd_count": 1,
        "lock_count": 1,
    }
    allocation[field] = value
    assert not prototype._link_audit_passed([_audit_report(allocation)])


def test_link_audit_accepts_values_at_each_tile_limit():
    allocations = [
        {"kind": "core", "buffer_end_bytes": 65536, "bd_count": 16, "lock_count": 16},
        {"kind": "memtile", "buffer_end_bytes": 524288, "bd_count": 48, "lock_count": 64},
        {"kind": "shim", "buffer_end_bytes": 0, "bd_count": 16, "lock_count": 0},
    ]
    report = {
        "elf": [{"text_bytes": CORE_PROGRAM_LIMIT_BYTES}],
        "addressed_mlir": [{"allocations": allocations}],
    }
    assert prototype._link_audit_passed([report])


def test_prototype_tracks_itemized_first_failure():
    counts = {"gemm": 0, "rda_state": 0, "rda_output": 0}
    counts["rda_state"] += 1
    first = prototype._record_first_failure(None, 17, "rda_state")
    first = prototype._record_first_failure(first, 18, "rda_output")
    assert counts == {"gemm": 0, "rda_state": 1, "rda_output": 0}
    assert first == {"iteration": 17, "item": "rda_state"}


def test_decode_unified_generation_inputs_unchanged():
    root = Path(__file__).resolve().parents[2]
    # Fingerprints refreshed for public headers/prose only; executable AST and C++ tokens audited.
    expected = {
        "kernels/gemv_unified/design.py": "cd0c9cd7a3e755e2c828c4bc8e675c841a61040fba813a5955b611352211c155",
        "kernels/gemv_unified/shapes.py": "ef728426e456dc47a029fcc68937674fd9e4d6966bcb20424f894794cf6ac731",
        "kernels/gemv_unified/bfp16_gemv.cc": "c01facf24121737597bbd1561c6572611e33ff55aa7ea03111274ca30e4cbaa7",
        "kernels/gemv_unified/op.py": "d5d5c578206c094e4188772ca09a4ae93f3968bd502b19e000444032931396a4",
        "kernels/gemv_unified/rda.py": "191505906d5d19e12f4953a254b9f27a11520bd71e39d73323e042fbce640495",
        "bench/rda_state/rda_state_1col.py": "27ed29c65a5d2ad20a0ef98b0b5ec809d60dff267fb890fbabe9e10aa0d15c30",
        "kernels/rda_state.cc": "44c2980f108ad76cbcbe8da824ab745951b4e99e6eaacb9c6ef4b1844e481c56",
    }
    for rel, digest in expected.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest


def test_prototype_budgets_use_physical_rows():
    budgets = prototype._budgets()
    assert len(budgets) == 18
    for key, budget in budgets.items():
        assert budget["batch_t"] == MAX_BATCH_T
        assert budget["limits_ok"] is True


def test_prototype_source_has_no_16_row_buffers():
    source = Path(prototype.__file__).read_text(encoding="utf-8")
    assert "(16, bmeta.k)" not in source
    assert "16 * bmeta.m" not in source
    assert "reshape(16, bmeta.m)" not in source
    assert "resource_budget(shape, 16" not in source
    assert "PrefillGEMM(shape_name, 16" not in source
    assert '"rda_in", 16,' not in source
