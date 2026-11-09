# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Host weights loader: BF16 storage as uint16, on-the-fly FP32 conversion."""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from host.strict_json import loads_strict

_BF16_IMPL_REQUEST = os.environ.get("RAON_BF16_IMPL", "auto").strip().lower()
if _BF16_IMPL_REQUEST not in {"auto", "numpy"}:
    raise ValueError("RAON_BF16_IMPL must be 'auto' or 'numpy'")

try:
    if _BF16_IMPL_REQUEST == "numpy":
        raise ImportError("NumPy BF16 conversion forced by RAON_BF16_IMPL")
    from ml_dtypes import bfloat16 as _ML_DTYPES_BFLOAT16
except (ImportError, ModuleNotFoundError):
    _ML_DTYPES_BFLOAT16 = None

BF16_IMPL = "ml_dtypes" if _ML_DTYPES_BFLOAT16 is not None else "numpy"

# Maximum tensor size kept in the per-Weights FP32 conversion cache.
# Tensors above this limit are converted on demand without caching.
F32_CACHE_MAX_TENSOR_BYTES = 64 * 1024 * 1024

# Lookup table for float8_e4m3fn to float32 (256 entries).
# Pre-computed to avoid any PyTorch dependence.
FP8_E4M3_LUT = np.array([
    0.00000000e+00, 1.95312500e-03, 3.90625000e-03, 5.85937500e-03,
    7.81250000e-03, 9.76562500e-03, 1.17187500e-02, 1.36718750e-02,
    1.56250000e-02, 1.75781250e-02, 1.95312500e-02, 2.14843750e-02,
    2.34375000e-02, 2.53906250e-02, 2.73437500e-02, 2.92968750e-02,
    3.12500000e-02, 3.51562500e-02, 3.90625000e-02, 4.29687500e-02,
    4.68750000e-02, 5.07812500e-02, 5.46875000e-02, 5.85937500e-02,
    6.25000000e-02, 7.03125000e-02, 7.81250000e-02, 8.59375000e-02,
    9.37500000e-02, 1.01562500e-01, 1.09375000e-01, 1.17187500e-01,
    1.25000000e-01, 1.40625000e-01, 1.56250000e-01, 1.71875000e-01,
    1.87500000e-01, 2.03125000e-01, 2.18750000e-01, 2.34375000e-01,
    2.50000000e-01, 2.81250000e-01, 3.12500000e-01, 3.43750000e-01,
    3.75000000e-01, 4.06250000e-01, 4.37500000e-01, 4.68750000e-01,
    5.00000000e-01, 5.62500000e-01, 6.25000000e-01, 6.87500000e-01,
    7.50000000e-01, 8.12500000e-01, 8.75000000e-01, 9.37500000e-01,
    1.00000000e+00, 1.12500000e+00, 1.25000000e+00, 1.37500000e+00,
    1.50000000e+00, 1.62500000e+00, 1.75000000e+00, 1.87500000e+00,
    2.00000000e+00, 2.25000000e+00, 2.50000000e+00, 2.75000000e+00,
    3.00000000e+00, 3.25000000e+00, 3.50000000e+00, 3.75000000e+00,
    4.00000000e+00, 4.50000000e+00, 5.00000000e+00, 5.50000000e+00,
    6.00000000e+00, 6.50000000e+00, 7.00000000e+00, 7.50000000e+00,
    8.00000000e+00, 9.00000000e+00, 1.00000000e+01, 1.10000000e+01,
    1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01,
    1.60000000e+01, 1.80000000e+01, 2.00000000e+01, 2.20000000e+01,
    2.40000000e+01, 2.60000000e+01, 2.80000000e+01, 3.00000000e+01,
    3.20000000e+01, 3.60000000e+01, 4.00000000e+01, 4.40000000e+01,
    4.80000000e+01, 5.20000000e+01, 5.60000000e+01, 6.00000000e+01,
    6.40000000e+01, 7.20000000e+01, 8.00000000e+01, 8.80000000e+01,
    9.60000000e+01, 1.04000000e+02, 1.12000000e+02, 1.20000000e+02,
    1.28000000e+02, 1.44000000e+02, 1.60000000e+02, 1.76000000e+02,
    1.92000000e+02, 2.08000000e+02, 2.24000000e+02, 2.40000000e+02,
    2.56000000e+02, 2.88000000e+02, 3.20000000e+02, 3.52000000e+02,
    3.84000000e+02, 4.16000000e+02, 4.48000000e+02, np.nan,
    -0.00000000e+00, -1.95312500e-03, -3.90625000e-03, -5.85937500e-03,
    -7.81250000e-03, -9.76562500e-03, -1.17187500e-02, -1.36718750e-02,
    -1.56250000e-02, -1.75781250e-02, -1.95312500e-02, -2.14843750e-02,
    -2.34375000e-02, -2.53906250e-02, -2.73437500e-02, -2.92968750e-02,
    -3.12500000e-02, -3.51562500e-02, -3.90625000e-02, -4.29687500e-02,
    -4.68750000e-02, -5.07812500e-02, -5.46875000e-02, -5.85937500e-02,
    -6.25000000e-02, -7.03125000e-02, -7.81250000e-02, -8.59375000e-02,
    -9.37500000e-02, -1.01562500e-01, -1.09375000e-01, -1.17187500e-01,
    -1.25000000e-01, -1.40625000e-01, -1.56250000e-01, -1.71875000e-01,
    -1.87500000e-01, -2.03125000e-01, -2.18750000e-01, -2.34375000e-01,
    -2.50000000e-01, -2.81250000e-01, -3.12500000e-01, -3.43750000e-01,
    -3.75000000e-01, -4.06250000e-01, -4.37500000e-01, -4.68750000e-01,
    -5.00000000e-01, -5.62500000e-01, -6.25000000e-01, -6.87500000e-01,
    -7.50000000e-01, -8.12500000e-01, -8.75000000e-01, -9.37500000e-01,
    -1.00000000e+00, -1.12500000e+00, -1.25000000e+00, -1.37500000e+00,
    -1.50000000e+00, -1.62500000e+00, -1.75000000e+00, -1.87500000e+00,
    -2.00000000e+00, -2.25000000e+00, -2.50000000e+00, -2.75000000e+00,
    -3.00000000e+00, -3.25000000e+00, -3.50000000e+00, -3.75000000e+00,
    -4.00000000e+00, -4.50000000e+00, -5.00000000e+00, -5.50000000e+00,
    -6.00000000e+00, -6.50000000e+00, -7.00000000e+00, -7.50000000e+00,
    -8.00000000e+00, -9.00000000e+00, -1.00000000e+01, -1.10000000e+01,
    -1.20000000e+01, -1.30000000e+01, -1.40000000e+01, -1.50000000e+01,
    -1.60000000e+01, -1.80000000e+01, -2.00000000e+01, -2.20000000e+01,
    -2.40000000e+01, -2.60000000e+01, -2.80000000e+01, -3.00000000e+01,
    -3.20000000e+01, -3.60000000e+01, -4.00000000e+01, -4.40000000e+01,
    -4.80000000e+01, -5.20000000e+01, -5.60000000e+01, -6.00000000e+01,
    -6.40000000e+01, -7.20000000e+01, -8.00000000e+01, -8.80000000e+01,
    -9.60000000e+01, -1.04000000e+02, -1.12000000e+02, -1.20000000e+02,
    -1.28000000e+02, -1.44000000e+02, -1.60000000e+02, -1.76000000e+02,
    -1.92000000e+02, -2.08000000e+02, -2.24000000e+02, -2.40000000e+02,
    -2.56000000e+02, -2.88000000e+02, -3.20000000e+02, -3.52000000e+02,
    -3.84000000e+02, -4.16000000e+02, -4.48000000e+02, np.nan,
], dtype=np.float32)


def _f32_to_bf16_numpy(arr: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of float32 -> BF16 RTNE."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    orig_shape = arr_f32.shape
    u32 = arr_f32.reshape(-1).view(np.uint32)
    lsb = (u32 >> 16) & 1
    bias = 0x7FFF + lsb
    nan_mask = np.isnan(arr_f32.reshape(-1))
    rounded = u32 + bias
    out = (rounded >> 16).astype(np.uint16)
    out[nan_mask] = 0x7FC0
    return out.reshape(orig_shape)


def _bf16_to_f32_numpy(arr: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of BF16 bits -> float32."""
    arr_u16 = np.asarray(arr, dtype=np.uint16)
    orig_shape = arr_u16.shape
    flat = (arr_u16.reshape(-1).astype(np.uint32) << 16).view(np.float32)
    return flat.reshape(orig_shape)


def _f32_to_bf16_ml_dtypes(arr: np.ndarray) -> np.ndarray:
    """C-backed float32 -> BF16 RTNE with the historical canonical NaN."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    if not arr_f32.flags.c_contiguous:
        arr_f32 = np.ascontiguousarray(arr_f32)
    out = arr_f32.astype(_ML_DTYPES_BFLOAT16).view(np.uint16)
    # ml_dtypes preserves/quietens NaN payloads.  The old converter instead
    # canonicalized every NaN, so repair only arrays containing non-finite data.
    if not np.isfinite(arr_f32).all():
        nan_mask = np.isnan(arr_f32)
        if np.any(nan_mask):
            out[nan_mask] = 0x7FC0
    return out


def _bf16_to_f32_ml_dtypes(arr: np.ndarray) -> np.ndarray:
    """C-backed BF16 bits -> float32, preserving shape and payload bits."""
    arr_u16 = np.asarray(arr, dtype=np.uint16)
    if not arr_u16.flags.c_contiguous:
        arr_u16 = np.ascontiguousarray(arr_u16)
    return arr_u16.view(_ML_DTYPES_BFLOAT16).astype(np.float32)


if BF16_IMPL == "ml_dtypes":
    f32_to_bf16_rtne = _f32_to_bf16_ml_dtypes
    bf16_to_f32 = _bf16_to_f32_ml_dtypes
else:
    f32_to_bf16_rtne = _f32_to_bf16_numpy
    bf16_to_f32 = _bf16_to_f32_numpy


def dequantize_mxfp8_to_bf16(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """E4M3 uint8 [N, K] and E8M0 scale uint8 [N, K/32] -> BF16 uint16 [N, K]."""
    rows, cols = weight_u8.shape
    w_f32 = FP8_E4M3_LUT[weight_u8].reshape(rows, cols // 32, 32)
    scale_factor = np.exp2(scale_u8.astype(np.float32) - 127.0)[:, :, None]
    prod_f32 = (w_f32 * scale_factor).reshape(rows, cols)
    return f32_to_bf16_rtne(prod_f32)


class SafeTensorsReader:
    """Zero-copy safetensors file reader using mmap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.header_len = struct.unpack("<Q", self.mm[:8])[0]
        self.header: dict[str, Any] = loads_strict(self.mm[8 : 8 + self.header_len])
        self.data_start = 8 + self.header_len

    def close(self) -> None:
        if hasattr(self, "mm") and self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass
        if hasattr(self, "f") and self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass

    def __enter__(self) -> "SafeTensorsReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def get_tensor_raw(self, name: str) -> tuple[np.ndarray, str, list[int]]:
        meta = self.header[name]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        abs_offset = self.data_start + start
        byte_len = end - start

        if dtype_str == "BF16":
            arr = np.frombuffer(self.mm, dtype=np.uint16, count=byte_len // 2, offset=abs_offset)
        elif dtype_str == "F32":
            arr = np.frombuffer(self.mm, dtype=np.float32, count=byte_len // 4, offset=abs_offset)
        elif dtype_str in {"U8", "F8_E4M3", "F8_E8M0"}:
            arr = np.frombuffer(self.mm, dtype=np.uint8, count=byte_len, offset=abs_offset)
        elif dtype_str == "I64":
            arr = np.frombuffer(self.mm, dtype=np.int64, count=byte_len // 8, offset=abs_offset)
        elif dtype_str == "I32":
            arr = np.frombuffer(self.mm, dtype=np.int32, count=byte_len // 4, offset=abs_offset)
        elif dtype_str == "F16":
            arr = np.frombuffer(self.mm, dtype=np.float16, count=byte_len // 2, offset=abs_offset)
        else:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")

        if shape:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())
        return arr, dtype_str, shape


class Weights:
    """Manages model weights in memory as BF16 (uint16) or FP32 (float32)."""

    def __init__(
        self,
        model_dir: str | Path,
        selected_layers: list[int] | None = None,
        load_global: bool = True,
        skip_moe_experts: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.selected_layers = set(selected_layers) if selected_layers is not None else None
        self.load_global = load_global
        self.skip_moe_experts = skip_moe_experts
        self.tensors: dict[str, np.ndarray] = {}
        self._f32_cache: dict[str, np.ndarray] = {}
        self._load()

    def _should_load_tensor(self, name: str) -> bool:
        if self.skip_moe_experts and ".moe.experts." in name:
            return False
        if self.selected_layers is None:
            return True
        # Global weights
        if not name.startswith("model.layers."):
            return self.load_global
        # Layer weights: model.layers.{idx}.xxx
        parts = name.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            idx = int(parts[2])
            return idx in self.selected_layers
        return True

    def _load(self) -> None:
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"

        if index_path.is_file():
            payload = loads_strict(index_path.read_bytes())
            weight_map: dict[str, str] = payload.get("weight_map", {})
            needed_shards = set()
            for name, shard in weight_map.items():
                if self._should_load_tensor(name):
                    needed_shards.add(shard)
            shards = sorted(needed_shards)
        elif single_path.is_file():
            with SafeTensorsReader(single_path) as reader:
                weight_map = {k: single_path.name for k in reader.keys()}
            shards = sorted(set(weight_map.values()))
        else:
            raise FileNotFoundError(f"No safetensors checkpoint found under {self.model_dir}")
        scale_names = {name for name in weight_map if name.endswith(".weight_scale_inv")}
        pairs = {name.removesuffix("_scale_inv"): name for name in scale_names}

        for shard_name in shards:
            shard_path = self.model_dir / shard_name
            with SafeTensorsReader(shard_path) as reader:
                shard_keys = set(reader.keys())
                for name in shard_keys:
                    if name in scale_names:
                        continue
                    if not self._should_load_tensor(name):
                        continue

                    scale_name = pairs.get(name)
                    if scale_name is not None and scale_name in shard_keys:
                        # MXFP8 expert pair
                        w_raw, w_dtype, _ = reader.get_tensor_raw(name)
                        s_raw, s_dtype, _ = reader.get_tensor_raw(scale_name)
                        # Dequantize to BF16 (uint16)
                        dequantized = dequantize_mxfp8_to_bf16(w_raw, s_raw)
                        self.tensors[name] = np.ascontiguousarray(dequantized)
                    else:
                        tensor_raw, dtype_str, shape = reader.get_tensor_raw(name)
                        if dtype_str == "BF16":
                            self.tensors[name] = np.array(tensor_raw, copy=True, dtype=np.uint16)
                        elif dtype_str == "F32":
                            self.tensors[name] = np.array(tensor_raw, copy=True, dtype=np.float32)
                        elif dtype_str == "F16":
                            f32 = tensor_raw.astype(np.float32)
                            self.tensors[name] = f32_to_bf16_rtne(f32)
                        else:
                            self.tensors[name] = np.array(tensor_raw, copy=True)

    def get(self, name: str, to_float: bool = True) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found in loaded weights.")
        arr = self.tensors[name]
        if not to_float:
            return arr
        if arr.dtype == np.float32:
            return arr

        cache = getattr(self, "_f32_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_f32_cache", cache)

        cached = cache.get(name)
        if cached is not None:
            return cached

        if arr.dtype == np.uint16:
            out = bf16_to_f32(arr)
        else:
            out = arr.astype(np.float32, order="C")

        if out.nbytes <= F32_CACHE_MAX_TENSOR_BYTES:
            # ``np.ascontiguousarray`` reshapes 0-dimensional arrays to (1,).
            # Use ndarray.copy() to preserve shape, dtype, and value for scalars
            # and tensors alike, then freeze the cached copy.
            cached = out.copy()
            cached.flags.writeable = False
            cache[name] = cached
            return cached
        return out

    def __contains__(self, name: str) -> bool:
        return name in self.tensors
