# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors

"""q4nx quantization and packing package for raon-npu."""

from quant.bfp16 import (
    bfp16ebs8_decode,
    bfp16ebs8_encode,
    pack_bfp16_gemv,
    pad_bfp16_gemv,
    parse_bfp16_gemv,
    quantize_bfp16ebs8,
)
from quant.q4nx import (
    BLOCK_BYTES,
    GROUP,
    K_TILE,
    M_TILE,
    bf16_ceil,
    bf16_floor,
    bf16_nextafter,
    dequantize_simple,
    mask_padded_logits,
    pad,
    pack_decode_proj,
    pack_lm_head,
    parse_decode_proj,
    parse_lm_head,
    quantize,
    round_to_bf16_rne,
    unpad,
)

__all__ = [
    "BLOCK_BYTES",
    "GROUP",
    "K_TILE",
    "M_TILE",
    "quantize",
    "dequantize_simple",
    "pad",
    "unpad",
    "pack_decode_proj",
    "parse_decode_proj",
    "pack_lm_head",
    "parse_lm_head",
    "mask_padded_logits",
    "round_to_bf16_rne",
    "bf16_floor",
    "bf16_ceil",
    "bf16_nextafter",
    "bfp16ebs8_encode",
    "bfp16ebs8_decode",
    "quantize_bfp16ebs8",
    "pack_bfp16_gemv",
    "parse_bfp16_gemv",
    "pad_bfp16_gemv",
]
