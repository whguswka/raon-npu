#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors
"""Minimal GGUF v2/v3 reader and llama.cpp quantized-weight to q4nx bridge.

Only NumPy is required.  Tensor payloads are mapped lazily and a caller may
select leading rows (or one expert from a three-dimensional expert tensor), so
the 20 GB GGUF is never copied into memory as a whole.

The Q4_K conversion preserves the 4-bit codes.  For every 32-value sub-block it
maps ``scale = d * sc`` and ``min = -dmin * m`` to BF16, exactly matching the
q4nx equation ``min + scale * code`` apart from BF16 parameter rounding.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from quant.q4nx import dequantize_simple, quantize, round_to_bf16_rne


GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    6: ("Q5_0", 32, 22),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]  # GGML order: fastest dimension first
    ggml_type: int
    offset: int

    @property
    def shape(self) -> tuple[int, ...]:
        """NumPy/PyTorch logical shape."""
        return tuple(reversed(self.dimensions))

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"TYPE_{self.ggml_type}", 0, 0))[0]

    @property
    def elements(self) -> int:
        return math.prod(self.dimensions)

    @property
    def nbytes(self) -> int:
        _, block_elements, block_bytes = GGML_TYPES[self.ggml_type]
        if self.elements % block_elements:
            raise ValueError(f"{self.name}: {self.elements} is not divisible by {block_elements}")
        return self.elements // block_elements * block_bytes


class GGUFReader:
    """Read GGUF metadata and lazily map supported tensor payloads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ValueError(f"not a GGUF file: {self.path}")
            self.version = self._scalar(handle, "I")
            if self.version not in (2, 3):
                raise ValueError(f"unsupported GGUF version {self.version}")
            tensor_count = self._scalar(handle, "Q")
            metadata_count = self._scalar(handle, "Q")
            self.metadata: dict[str, Any] = {}
            for _ in range(metadata_count):
                key = self._string(handle)
                self.metadata[key] = self._value(handle, self._scalar(handle, "I"))
            tensors = []
            for _ in range(tensor_count):
                name = self._string(handle)
                dimensions = tuple(self._scalar(handle, "Q") for _ in range(self._scalar(handle, "I")))
                ggml_type = self._scalar(handle, "I")
                offset = self._scalar(handle, "Q")
                tensors.append(TensorInfo(name, dimensions, ggml_type, offset))
            alignment = int(self.metadata.get("general.alignment", 32))
            self.data_offset = (handle.tell() + alignment - 1) // alignment * alignment
        self.tensors = {tensor.name: tensor for tensor in tensors}

    @staticmethod
    def _scalar(handle: BinaryIO, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        data = handle.read(size)
        if len(data) != size:
            raise EOFError("truncated GGUF header")
        return struct.unpack("<" + fmt, data)[0]

    @classmethod
    def _string(cls, handle: BinaryIO) -> str:
        size = cls._scalar(handle, "Q")
        return handle.read(size).decode("utf-8")

    @classmethod
    def _value(cls, handle: BinaryIO, value_type: int) -> Any:
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                   7: "B", 10: "Q", 11: "q", 12: "d"}
        if value_type in formats:
            value = cls._scalar(handle, formats[value_type])
            return bool(value) if value_type == 7 else value
        if value_type == 8:
            return cls._string(handle)
        if value_type == 9:
            element_type = cls._scalar(handle, "I")
            return [cls._value(handle, element_type) for _ in range(cls._scalar(handle, "Q"))]
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")

    def _block_view(self, info: TensorInfo, start_block: int, block_count: int) -> np.ndarray:
        _, _, block_bytes = GGML_TYPES[info.ggml_type]
        return np.memmap(
            self.path, mode="r", dtype=np.uint8,
            offset=self.data_offset + info.offset + start_block * block_bytes,
            shape=(block_count, block_bytes),
        )

    def _leading_range(self, info: TensorInfo, row_slice: slice | None,
                       outer_index: int | None) -> tuple[int, int, tuple[int, ...]]:
        shape = info.shape
        if outer_index is not None:
            if len(shape) != 3 or not 0 <= outer_index < shape[0]:
                raise ValueError(f"{info.name}: outer_index requires a 3D tensor, got {shape}")
            per_outer = math.prod(shape[1:])
            return outer_index * per_outer, per_outer, shape[1:]
        if len(shape) == 1:
            if row_slice is not None:
                raise ValueError("row_slice is only valid for matrices")
            return 0, shape[0], shape
        if len(shape) > 2 and row_slice is None:
            return 0, math.prod(shape), shape
        rows = math.prod(shape[:-1])
        width = shape[-1]
        selected = range(rows)[row_slice or slice(None)]
        if selected.step != 1:
            raise ValueError("row_slice step must be 1")
        count = len(selected)
        return selected.start * width, count * width, (count, width)

    def dequantize(self, name: str, row_slice: slice | None = None,
                   outer_index: int | None = None) -> np.ndarray:
        info = self.tensors[name]
        if info.ggml_type not in GGML_TYPES:
            raise NotImplementedError(f"{name}: unsupported {info.type_name}")
        element_start, element_count, shape = self._leading_range(info, row_slice, outer_index)
        _, block_elements, _ = GGML_TYPES[info.ggml_type]
        if element_start % block_elements or element_count % block_elements:
            raise ValueError(f"{name}: requested range is not block aligned")
        raw = self._block_view(info, element_start // block_elements, element_count // block_elements)
        if info.ggml_type == 0:
            out = np.asarray(raw).reshape(-1).view("<f4").astype(np.float32, copy=True)
        elif info.ggml_type == 1:
            out = np.asarray(raw).reshape(-1).view("<f2").astype(np.float32)
        elif info.ggml_type == 30:
            words = np.asarray(raw).reshape(-1).view("<u2").astype(np.uint32)
            out = (words << 16).view(np.float32)
        elif info.ggml_type == 6:
            out = dequantize_q5_0_blocks(raw).reshape(-1)
        elif info.ggml_type == 8:
            out = dequantize_q8_0_blocks(raw).reshape(-1)
        elif info.ggml_type == 12:
            out = dequantize_q4_k_blocks(raw).reshape(-1)
        elif info.ggml_type == 14:
            out = dequantize_q6_k_blocks(raw).reshape(-1)
        else:  # pragma: no cover - guarded above
            raise AssertionError(info.ggml_type)
        return out.reshape(shape)

    def q4nx_components(self, name: str, row_slice: slice | None = None,
                        outer_index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Q4_K directly, or dequantize and min-max requantize another type."""
        info = self.tensors[name]
        element_start, element_count, shape = self._leading_range(info, row_slice, outer_index)
        if len(shape) != 2 or shape[1] % 32:
            raise ValueError(f"{name}: q4nx requires a 2D matrix with K divisible by 32, got {shape}")
        if info.ggml_type == 12:
            if element_start % 256 or element_count % 256:
                raise ValueError(f"{name}: Q4_K selection is not super-block aligned")
            raw = self._block_view(info, element_start // 256, element_count // 256)
            codes, scales, mins = q4_k_blocks_to_q4nx(raw)
            return (codes.reshape(shape), scales.reshape(shape[0], shape[1] // 32),
                    mins.reshape(shape[0], shape[1] // 32))
        return quantize(self.dequantize(name, row_slice=row_slice, outer_index=outer_index))


def unpack_q4_k_scale_min(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack llama.cpp's 12-byte K-scale encoding into eight 6-bit pairs."""
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape[-1] != 12:
        raise ValueError(f"Q4_K scale field must be 12 bytes, got {p.shape}")
    scale = np.empty(p.shape[:-1] + (8,), dtype=np.uint8)
    minimum = np.empty_like(scale)
    scale[..., :4] = p[..., :4] & 0x3F
    minimum[..., :4] = p[..., 4:8] & 0x3F
    scale[..., 4:] = (p[..., 8:12] & 0x0F) | ((p[..., :4] >> 6) << 4)
    minimum[..., 4:] = (p[..., 8:12] >> 4) | ((p[..., 4:8] >> 6) << 4)
    return scale, minimum


def unpack_q4_k_codes(qs: np.ndarray) -> np.ndarray:
    """Return natural 32-element-group order used by dequantize_row_q4_K."""
    q = np.asarray(qs, dtype=np.uint8)
    if q.shape[-1] != 128:
        raise ValueError(f"Q4_K code field must be 128 bytes, got {q.shape}")
    q = q.reshape(q.shape[:-1] + (4, 32))
    out = np.empty(q.shape[:-2] + (8, 32), dtype=np.uint8)
    out[..., 0::2, :] = q & 0x0F
    out[..., 1::2, :] = q >> 4
    return out


def _q4_k_fields(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(blocks, dtype=np.uint8)
    if b.ndim != 2 or b.shape[1] != 144:
        raise ValueError(f"Q4_K blocks must have shape [N, 144], got {b.shape}")
    d = b[:, 0:2].copy().reshape(-1).view("<f2").astype(np.float32)
    dmin = b[:, 2:4].copy().reshape(-1).view("<f2").astype(np.float32)
    sc, mn = unpack_q4_k_scale_min(b[:, 4:16])
    codes = unpack_q4_k_codes(b[:, 16:144])
    return d, dmin, sc, mn, codes


def q4_k_blocks_to_q4nx(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map block_q4_K to q4nx components, preserving every 4-bit code."""
    d, dmin, sc, mn, codes = _q4_k_fields(blocks)
    scales = round_to_bf16_rne(d[:, None] * sc.astype(np.float32))
    mins = round_to_bf16_rne(-dmin[:, None] * mn.astype(np.float32))
    return codes.reshape(-1, 256), scales, mins


def dequantize_q4_k_blocks(blocks: np.ndarray) -> np.ndarray:
    """NumPy equivalent of llama.cpp dequantize_row_q4_K (float32 output)."""
    d, dmin, sc, mn, codes = _q4_k_fields(blocks)
    values = (d[:, None, None] * sc[..., None].astype(np.float32) * codes.astype(np.float32)
              - dmin[:, None, None] * mn[..., None].astype(np.float32))
    return values.reshape(-1, 256).astype(np.float32)


def dequantize_q6_k_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint8)
    if b.ndim != 2 or b.shape[1] != 210:
        raise ValueError(f"Q6_K blocks must have shape [N, 210], got {b.shape}")
    ql = b[:, :128].reshape(-1, 2, 64)
    qh = b[:, 128:192].reshape(-1, 2, 32)
    scales = b[:, 192:208].view(np.int8).reshape(-1, 2, 8).astype(np.float32)
    d = b[:, 208:210].copy().reshape(-1).view("<f2").astype(np.float32)
    out = np.empty((len(b), 2, 128), dtype=np.float32)
    for half in range(2):
        lo, hi, scale = ql[:, half], qh[:, half], scales[:, half]
        for lane in range(32):
            si = lane // 16
            values = (
                (lo[:, lane] & 0x0F) | ((hi[:, lane] & 0x03) << 4),
                (lo[:, lane + 32] & 0x0F) | (((hi[:, lane] >> 2) & 0x03) << 4),
                (lo[:, lane] >> 4) | (((hi[:, lane] >> 4) & 0x03) << 4),
                (lo[:, lane + 32] >> 4) | (((hi[:, lane] >> 6) & 0x03) << 4),
            )
            for quarter, q in enumerate(values):
                out[:, half, lane + quarter * 32] = d * scale[:, si + quarter * 2] * (q.astype(np.float32) - 32.0)
    return out.reshape(-1, 256)


def dequantize_q8_0_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint8)
    d = b[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
    q = b[:, 2:34].view(np.int8).astype(np.float32)
    return d[:, None] * q


def dequantize_q5_0_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint8)
    d = b[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
    qh = b[:, 2:6].copy().reshape(-1).view("<u4")
    qs = b[:, 6:22]
    out = np.empty((len(b), 32), dtype=np.float32)
    bits = np.arange(16, dtype=np.uint32)
    out[:, :16] = ((qs & 0x0F) | (((qh[:, None] >> bits) & 1).astype(np.uint8) << 4))
    out[:, 16:] = ((qs >> 4) | (((qh[:, None] >> (bits + 16)) & 1).astype(np.uint8) << 4))
    return d[:, None] * (out - 16.0)


def hf_to_gguf(
    hf_name: str,
    *,
    rda_layers: set[int] | None = None,
    tensor_names: set[str] | None = None,
) -> tuple[str, int | None] | None:
    """Map released HF matrix names to the Raon llama.cpp-fork GGUF names."""
    if hf_name == "lm_head.weight":
        return "output.weight", None
    if hf_name == "model.norm.weight":
        return "output_norm.weight", None
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight", None
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", hf_name)
    if not match:
        return None
    layer, suffix = int(match.group(1)), match.group(2)
    direct = {
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.f_a_proj.weight": "ssm_f_a.weight",
        "self_attn.f_b_proj.weight": "ssm_f_b.weight",
        "self_attn.b_proj.weight": "ssm_beta.weight",
        "self_attn.rda_w_proj.weight": "ssm_rda.weight",
        "self_attn.q_a_proj.weight": "attn_q_a.weight",
        "self_attn.q_b_proj.weight": "attn_q_b.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
        "moe.gate.weight": "ffn_gate_inp.weight",
        "moe.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
        "moe.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
        "moe.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    if suffix == "self_attn.g_proj.weight":
        candidates = (f"blk.{layer}.ssm_g.weight", f"blk.{layer}.attn_gate.weight")
        if tensor_names is not None:
            present = [name for name in candidates if name in tensor_names]
            if len(present) == 1:
                return present[0], None
            if len(present) > 1:
                raise ValueError(f"ambiguous GGUF g_proj tensors for layer {layer}: {present}")
        if rda_layers is not None:
            return candidates[0 if layer in rda_layers else 1], None
        raise KeyError(
            f"g_proj mapping for layer {layer} needs GGUF tensor names or config rda_layers"
        )
    if suffix in direct:
        return f"blk.{layer}.{direct[suffix]}", None
    expert = re.fullmatch(r"moe\.experts\.(\d+)\.(gate|up|down)_proj\.weight", suffix)
    if expert:
        return f"blk.{layer}.ffn_{expert.group(2)}_exps.weight", int(expert.group(1))
    # kv_b is split into K and V tensors and is reconstructed by GGUFWeightReader.
    if suffix == "self_attn.kv_b_proj.weight":
        return f"blk.{layer}.attn_k_b.weight+blk.{layer}.attn_v_b.weight", None
    return None


class GGUFWeightReader:
    """Expose GGUF matrices in the released HF matrix naming/layout."""

    def __init__(self, gguf: GGUFReader, rda_layers: set[int] | None = None):
        self.gguf = gguf
        self.rda_layers = rda_layers

    def _mapped(self, hf_name: str) -> tuple[str, int | None] | None:
        return hf_to_gguf(
            hf_name, rda_layers=self.rda_layers, tensor_names=set(self.gguf.tensors)
        )

    def source(self, hf_name: str) -> tuple[str, list[str]]:
        mapped = self._mapped(hf_name)
        if mapped is None:
            raise KeyError(f"no GGUF mapping for {hf_name}")
        names = mapped[0].split("+")
        types = [self.gguf.tensors[name].type_name for name in names]
        method = "direct Q4_K→q4nx" if types == ["Q4_K"] else "dequantize→min-max q4nx"
        return method, types

    def matrix(self, hf_name: str, row_slice: slice | None = None) -> np.ndarray:
        mapped = self._mapped(hf_name)
        if mapped is None:
            raise KeyError(f"no GGUF mapping for {hf_name}")
        gguf_name, expert = mapped
        if "+" not in gguf_name:
            return self.gguf.dequantize(gguf_name, row_slice=row_slice, outer_index=expert)
        if row_slice is not None:
            raise ValueError("row slicing split kv_b is not supported")
        k_name, v_name = gguf_name.split("+")
        # Fork layout: K [head, input, output], V [head, output, input].
        k = self.gguf.dequantize(k_name).transpose(0, 2, 1)
        v = self.gguf.dequantize(v_name)
        return np.concatenate((k, v), axis=1).reshape(-1, k.shape[-1])

    def q4_restore(self, hf_name: str, row_slice: slice | None = None) -> np.ndarray:
        mapped = self._mapped(hf_name)
        if mapped is None:
            raise KeyError(f"no GGUF mapping for {hf_name}")
        gguf_name, expert = mapped
        if "+" in gguf_name:
            value = self.matrix(hf_name, row_slice=row_slice)
            codes, scales, mins = quantize(value)
        else:
            codes, scales, mins = self.gguf.q4nx_components(
                gguf_name, row_slice=row_slice, outer_index=expert)
        return dequantize_simple(codes, scales, mins)


def inventory_markdown(reader: GGUFReader) -> str:
    lines = ["# GGUF inventory", "",
             f"- file: `{reader.path}`", f"- GGUF version: {reader.version}",
             f"- architecture: `{reader.metadata.get('general.architecture')}`",
             f"- alignment: {reader.metadata.get('general.alignment', 32)} bytes",
             f"- data offset: {reader.data_offset}", f"- tensor count: {len(reader.tensors)}", "",
             "| GGUF tensor | type | GGUF dimensions | logical shape | bytes |",
             "|---|---:|---:|---:|---:|"]
    for info in reader.tensors.values():
        lines.append(f"| `{info.name}` | {info.type_name} | {'×'.join(map(str, info.dimensions))} | "
                     f"{'×'.join(map(str, info.shape))} | {info.nbytes} |")
    lines += ["", "## HF → GGUF conversion-name rules", "",
              "| HF pattern | GGUF pattern | note |", "|---|---|---|",
              "| `lm_head.weight` | `output.weight` | output projection |",
              "| `model.layers.N.self_attn.{q,k,v}_proj.weight` | `blk.N.attn_{q,k,v}.weight` | RDA |",
              "| `...{q_a,q_b,kv_a}_proj...` | `blk.N.attn_{q_a,q_b,kv_a_mqa}.weight` | MLA |",
              "| `...kv_b_proj.weight` | `blk.N.attn_k_b.weight` + `attn_v_b.weight` | head-wise K/V split |",
              "| `...moe.gate.weight` | `blk.N.ffn_gate_inp.weight` | router |",
              "| `...experts.E.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_exps.weight[E]` | 3D expert bundle |",
              "| `...shared_experts.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_shexp.weight` | shared expert |",
              "| layer 0 `mlp.{gate,up,down}_proj.weight` | `blk.0.ffn_{gate,up,down}.weight` | dense FFN |", "",
              "이 규칙은 파일 내부 `general.architecture=raon1`, 실제 tensor names/shapes, 그리고 HF shape의 "
              "동일성을 근거로 한다. 특히 `ffn_*_exps`의 logical shape는 `[128, out, in]`이다."]
    return "\n".join(lines) + "\n"


def q4_k_proof(reader: GGUFReader, tensor_name: str, block_count: int | None = None) -> dict[str, Any]:
    """Prove code preservation and BF16 parameter-rounding bounds on a real Q4_K tensor."""
    info = reader.tensors[tensor_name]
    if info.ggml_type != 12:
        raise ValueError(f"{tensor_name} is {info.type_name}, not Q4_K")
    total_blocks = info.elements // 256
    count = total_blocks if block_count is None else min(block_count, total_blocks)
    if count <= 0:
        raise ValueError("block_count must be positive")
    raw = reader._block_view(info, 0, count)
    codes, scales, mins = q4_k_blocks_to_q4nx(raw)
    reference = dequantize_q4_k_blocks(raw)
    converted = dequantize_simple(codes, scales, mins)
    d, dmin, packed_scale, packed_min, original_codes = _q4_k_fields(raw)
    exact_scale = d[:, None] * packed_scale.astype(np.float32)
    exact_min = -dmin[:, None] * packed_min.astype(np.float32)
    bound = (
        np.abs(exact_scale - scales)[..., None] * original_codes.astype(np.float32)
        + np.abs(exact_min - mins)[..., None]
    )
    difference = np.abs(reference - converted)
    return {
        "tensor": tensor_name,
        "blocks": count,
        "values": count * 256,
        "codes_bit_exact": bool(np.array_equal(codes.reshape(count, 8, 32), original_codes)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "max_bf16_rounding_bound": float(bound.max()),
        "violations_over_bound_plus_2e-6_float32_tolerance": int(
            np.count_nonzero(difference.reshape(count, 8, 32) > bound + 2e-6)
        ),
        "equations": {
            "q4_k": "d * sc * q - dmin * m",
            "q4nx": "bf16(-dmin*m) + bf16(d*sc) * q",
            "per_value_bound": (
                "abs(q * (bf16(d*sc)-d*sc)) + "
                "abs(bf16(-dmin*m)-(-dmin*m))"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--proof", type=Path, help="write a real Q4_K conversion proof as JSON")
    parser.add_argument("--proof-tensor", default="blk.24.ffn_gate_exps.weight")
    parser.add_argument("--proof-blocks", type=int, default=4096)
    args = parser.parse_args()
    reader = GGUFReader(args.gguf)
    if args.inventory:
        args.inventory.write_text(inventory_markdown(reader), encoding="utf-8")
    if args.json:
        payload = {
            "version": reader.version, "metadata": reader.metadata,
            "data_offset": reader.data_offset,
            "tensors": [{"name": x.name, "type": x.type_name, "dimensions": x.dimensions,
                         "shape": x.shape, "offset": x.offset, "bytes": x.nbytes}
                        for x in reader.tensors.values()],
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.proof:
        payload = q4_k_proof(reader, args.proof_tensor, args.proof_blocks)
        args.proof.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for info in reader.tensors.values():
        counts[info.type_name] = counts.get(info.type_name, 0) + 1
    print(json.dumps({"version": reader.version, "architecture": reader.metadata.get("general.architecture"),
                      "tensors": len(reader.tensors), "types": counts, "data_offset": reader.data_offset}))


if __name__ == "__main__":
    main()
