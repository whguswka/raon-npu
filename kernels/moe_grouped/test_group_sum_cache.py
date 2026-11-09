# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Regression and contract checks for grouped MoE input group-sum cache (prefill 2b-4a).

This test suite runs on CPU host without IRON / AIE compiler tools by inspecting AST,
monkeypatching kernel signatures, and validating C++ source contracts against git main.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest

import kernels.moe_grouped.kernel as kernel_mod
from kernels.moe_grouped.layout import B1, B2, resource_budget, stream_groups


def test_design_ast_core_and_worker_alignment() -> None:
    source = Path(__file__).with_name("design.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # 1. Locate core function definition inside moe_grouped
    core_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "core":
            core_fn = node
            break
    assert core_fn is not None, "core function not found in design.py"

    params = [arg.arg for arg in core_fn.args.args]
    expected_params = [
        "x_in", "w_in", "y_out", "acc", "sums", "rtp",
        "barrier", "zero", "block", "epilogue",
    ]
    assert params == expected_params, f"core parameters mismatch: {params}"
    acc_idx = params.index("acc")
    assert params[acc_idx:acc_idx + 3] == ["acc", "sums", "rtp"]

    # 2. Locate Worker instantiation and check fn_args order
    worker_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Worker":
            worker_call = node
            break
    assert worker_call is not None, "Worker call not found in design.py"

    # fn_args is the second positional argument
    fn_args_node = worker_call.args[1]
    assert isinstance(fn_args_node, ast.List)
    arg_exprs = [ast.unparse(elt) for elt in fn_args_node.elts]
    assert "acc" in arg_exprs
    acc_arg_idx = arg_exprs.index("acc")
    assert arg_exprs[acc_arg_idx:acc_arg_idx + 3] == ["acc", "sums", "rtp"]
    assert len(arg_exprs) == len(params)

    # 3. Check both block() calls have the same 9-argument shape
    block_calls = [
        node for node in ast.walk(core_fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "block"
    ]
    assert len(block_calls) == 2, f"expected 2 block calls in core(), found {len(block_calls)}"
    arg_lists = {tuple(ast.unparse(arg) for arg in call.args) for call in block_calls}
    assert arg_lists == {
        ("w", "x", "acc", "sums", "rows", "mode", "_round", "_k", "_slot"),
        ("w", "x", "acc", "sums", "rows", "mode", "_e", "_k", "extra_slot"),
    }

    # 4. The two outer loops (main slot/round and extra round) have the same shape:
    #    zero(acc) -> for _k: acquire x/w, block, release x/w -> acquire y, epilogue, release y.
    def _body_has_call(loop: ast.For, name: str) -> bool:
        return any(
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "id", None) == name
            for stmt in loop.body
        )

    outer_loops = [
        node for node in ast.walk(core_fn)
        if isinstance(node, ast.For) and _body_has_call(node, "zero")
    ]
    assert len(outer_loops) == 2, f"expected 2 outer block loops, found {len(outer_loops)}"
    for loop in outer_loops:
        body = [ast.unparse(stmt) for stmt in loop.body]
        assert any("zero(acc, TMAX)" in stmt for stmt in body)
        assert any("block(" in stmt for stmt in body)
        assert any("epilogue(y, acc, TMAX)" in stmt for stmt in body)
        assert any("y = y_out.acquire(1)" in stmt for stmt in body)
        assert any("y_out.release(1)" in stmt for stmt in body)
        for stmt in body:
            if "block(" in stmt:
                assert "x = x_in.acquire(1)" in stmt
                assert "w = w_in.acquire(1)" in stmt
                assert "w_in.release(1)" in stmt
                assert "x_in.release(1)" in stmt

    # 4. Check group_sums_ Buffer creation per worker
    buffer_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Buffer":
            for kw in node.keywords:
                if kw.arg == "name":
                    buffer_names.append(ast.unparse(kw.value))
    assert any('group_sums_{col}_{row}' in name for name in buffer_names)


def test_design_ast_extra_round_loop_uses_rtp5_and_rtp6_exactly() -> None:
    """The extra round must read rtp[5]/rtp[6] and iterate range_(extra) with the right block call."""
    source = Path(__file__).with_name("design.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    core_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "core"
    )

    assignments = {
        ast.unparse(t): ast.unparse(node.value)
        for node in core_fn.body
        if isinstance(node, ast.Assign)
        for t in node.targets
    }
    assert assignments.get("extra") == "rtp[5]", "extra must be assigned from rtp[5]"
    assert assignments.get("extra_slot") == "rtp[6]", "extra_slot must be assigned from rtp[6]"

    extra_loop = None
    for node in ast.walk(core_fn):
        if not isinstance(node, ast.For):
            continue
        iter_ = node.iter
        if (
            isinstance(iter_, ast.Call)
            and getattr(iter_.func, "id", None) == "range_"
            and len(iter_.args) == 1
            and ast.unparse(iter_.args[0]) == "extra"
        ):
            extra_loop = node
            break
    assert extra_loop is not None, "no for-loop over range_(extra) found"
    assert extra_loop.target.id == "_e", "extra loop target must be _e"

    block_calls = [
        node
        for node in ast.walk(extra_loop)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "block"
    ]
    assert len(block_calls) == 1, f"expected 1 block call in extra loop, got {len(block_calls)}"
    assert tuple(ast.unparse(arg) for arg in block_calls[0].args) == (
        "w", "x", "acc", "sums", "rows", "mode", "_e", "_k", "extra_slot",
    )


def test_design_dma_and_runtime_sequence_matches_new_contract() -> None:
    """Runtime sequence follows stream_groups; X is one contiguous line per slot."""
    source = Path(__file__).with_name("design.py").read_text(encoding="utf-8")

    marker = "def sequence(Y, W, X, x_ps, w_ps, y_cs):"
    assert marker in source
    seq = source[source.index(marker):]

    # RTP words 5 and 6 are written before word 0 for patch.py's find_rows_words.
    assert "rtps[col][row][5] = 1 if shape.balanced else 0" in seq
    assert "rtps[col][row][6] = col" in seq
    assert seq.index("rtps[col][row][5]") < seq.index("rtps[col][row][0]")
    assert seq.index("rtps[col][row][6]") < seq.index("rtps[col][row][0]")

    # The sequence iterates stream_groups(shape), not slot/round nested loops.
    assert "for records in stream_groups(shape):" in seq
    assert "for slot in range(SLOTS):" not in seq
    assert "for rnd in range(max(rounds)):" not in seq

    # X transfer is one contiguous run per slot: [1, 1, 1, TMAX * shape.k] with stride [0, 0, 0, 1].
    assert "[1, 1, 1, TMAX * shape.k]" in seq
    assert "[0, 0, 0, 1]" in seq

    # W and Y access patterns stay the same shape as before.
    assert "[1, 1, 1, shape.chunk_bytes // 4], [0, 0, 0, 1]" in seq
    assert "[1, ROWS, TMAX, 32], [0, 32, shape.m, 1]" in seq

    # Two-deep overlap window is unchanged.
    assert "if len(live) > 2:" in seq
    assert "live.pop(0).finish()" in seq


def test_kernel_contract_and_argument_types(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMarker:
        def __init__(self, name: str):
            self.name = name
        def __repr__(self) -> str:
            return self.name

    In = DummyMarker("In")
    InOut = DummyMarker("InOut")
    Out = DummyMarker("Out")
    Param = DummyMarker("Param")

    created_externs: list[tuple[str, list, object]] = []

    def mock_make_extern(
        func_name: str,
        source_path: object,
        arg_types: list,
        compile_flags: list[str] | None = None,
        include_dirs: list[str] | None = None,
        contract: object = None,
    ):
        ext = MagicMock()
        ext.func_name = func_name
        ext.source_path = source_path
        ext.arg_types = lambda: arg_types
        ext.contract = contract
        ext.object_file_name = "grouped_block.o"
        ext.source_file = "grouped_mmul.cc"
        ext.include_dirs = include_dirs
        ext.compile_flags = compile_flags
        ext.object_file = MagicMock(symbol_prefix="grouped_block_prefix_")
        created_externs.append((func_name, arg_types, contract))
        return ext

    class FakeKernelContract:
        def __init__(self, *, trace=None, roles=None, tolerance=None, ops_per_call=0, acc_dtype=None, reduction=None):
            self.trace = trace
            self.roles = roles
            self.tolerance = tolerance
            self.ops_per_call = ops_per_call
            self.acc_dtype = acc_dtype
            self.reduction = reduction

    class FakeTrace:
        @staticmethod
        def whole_call():
            return "whole_call"

    class FakeTolerance:
        @staticmethod
        def exact(*, note=""):
            return note

    fake_config = MagicMock()
    fake_config.aie_kernels_dir.return_value = "/tmp/aie_kernels"
    fake_config.cxx_header_path.return_value = "/tmp/cxx_headers"

    monkeypatch.setattr(kernel_mod, "NPU_AVAILABLE", True)
    monkeypatch.setattr(kernel_mod, "_make_extern", mock_make_extern, raising=False)
    monkeypatch.setattr(kernel_mod, "KernelContract", FakeKernelContract, raising=False)
    monkeypatch.setattr(kernel_mod, "Trace", FakeTrace, raising=False)
    monkeypatch.setattr(kernel_mod, "Tolerance", FakeTolerance, raising=False)
    monkeypatch.setattr(kernel_mod, "config", fake_config, raising=False)
    monkeypatch.setattr(kernel_mod, "ExternalFunction", MagicMock, raising=False)
    monkeypatch.setattr(kernel_mod, "In", In, raising=False)
    monkeypatch.setattr(kernel_mod, "InOut", InOut, raising=False)
    monkeypatch.setattr(kernel_mod, "Out", Out, raising=False)
    monkeypatch.setattr(kernel_mod, "Param", Param, raising=False)

    # Test mode 0 (9-bit) and mode 1 (q4nx)
    k0 = kernel_mod.grouped_kernel(0)
    k1 = kernel_mod.grouped_kernel(1)

    assert len(created_externs) == 2

    for mode, (func_name, arg_types, contract) in enumerate(created_externs):
        assert func_name == "grouped_block"
        assert len(arg_types) == 9, f"expected 9 arguments, got {len(arg_types)}"

        w, x, acc, sums, p0, p1, p2, p3, p4 = arg_types

        # sums must be (640,) bf16 (1,280 bytes: 10 k_blocks * 8 groups * 8 tokens)
        assert sums.__args__[0] == (640,)
        assert sums.__args__[1] == kernel_mod._BF16
        # 640 elements * 2 bytes = 1,280 bytes
        assert sums.__args__[0][0] * 2 == 1280

        # mode-dependent W shape check
        if mode == 0:
            assert w.__args__[0] == (4608,)
            assert w.__args__[0][0] * 2 == 9216
        else:
            assert w.__args__[0] == (2560,)
            assert w.__args__[0][0] * 2 == 5120

        # Param types
        assert p0 == np.int32
        assert p1 == np.int32
        assert p2 == np.int32
        assert p3 == np.int32
        assert p4 == np.int32

        # Contract roles
        assert contract.roles == (In, In, InOut, InOut, Param, Param, Param, Param, Param)

    # Sibling functions must exist
    assert hasattr(k0, "zero") and hasattr(k1, "zero")
    assert hasattr(k0, "epilogue") and hasattr(k1, "epilogue")


def test_grouped_mmul_source_contracts() -> None:
    cc_path = Path(__file__).with_name("grouped_mmul.cc")
    cc_text = cc_path.read_text(encoding="utf-8")

    # 1. group_sum_scalar matches original group_sum logic exactly
    assert "group_sum_scalar" in cc_text
    scalar_code = """static inline bfloat16 group_sum_scalar(const bfloat16 *x, int group, int token) {
  const bfloat16 *p = x + group * 4 * MAX_T * 8 + token * 8;
  auto values = aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + MAX_T * 8),
                            aie::load_v<8>(p + 2 * MAX_T * 8),
                            aie::load_v<8>(p + 3 * MAX_T * 8));
  aie::accum<accfloat, 16> sum(aie::broadcast<float, 16>(aie::reduce_add(values)));
  return sum.to_vector<bfloat16>().get(0);
}"""
    assert scalar_code in cc_text

    # 2. Old group_sum is removed, no group_sum(x calls left
    assert "group_sum(x" not in cc_text
    assert "group_sum(" not in cc_text

    # 3. ks[...] writes occur strictly inside round == 0
    assert "if (round == 0)" in cc_text
    round_0_idx = cc_text.index("if (round == 0)")
    ks_write_idx = cc_text.index("ks[group * MAX_T + token] =")
    assert round_0_idx < ks_write_idx
    # Ensure there are no other writes to ks[...]
    assert cc_text.count("ks[") == 1

    # 4. 9-bit mode arithmetic path matches main 173e24b verbatim
    git_cc = subprocess.check_output(
        ["git", "show", "173e24b:kernels/moe_grouped/grouped_mmul.cc"],
        encoding="utf-8",
    )
    # Extract mode 0 block arithmetic
    mode0_marker = "const Packed9 *w = static_cast<const Packed9 *>(packed);"
    return_marker = "return;\n  }"
    assert mode0_marker in git_cc and return_marker in git_cc
    git_mode0_arith = git_cc[git_cc.index(mode0_marker):git_cc.index(return_marker)]
    curr_mode0_arith = cc_text[cc_text.index(mode0_marker):cc_text.index(return_marker)]
    assert curr_mode0_arith == git_mode0_arith, "mode 0 arithmetic was modified from main!"

    # 5. Check (void) handling in mode 0
    mode0_start = cc_text.index("if (mode == 0) {")
    mode0_header = cc_text[mode0_start:cc_text.index(mode0_marker)]
    assert "(void)sums;" in mode0_header
    assert "(void)round;" in mode0_header
    assert "(void)k;" in mode0_header


def test_resource_budget_updated() -> None:
    budget = resource_budget()
    assert budget["l1_used_bytes"] == 52480
    assert budget["l1_banks"]["bank1"]["group_sums"] == (26624, 27904)
