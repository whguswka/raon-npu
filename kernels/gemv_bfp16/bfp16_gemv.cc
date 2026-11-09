//===- bfp16_gemv.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified by the raon-npu authors, 2026.
// Portions modified for raon-npu (Raon-1-mini general 9-bit bfp16ebs8 GEMV)
//
// Differences from q4nx_gemv:
// 1. Weights stored in 9-bit bfp16ebs8 format (1 exponent byte + 8 mantissa bytes per 8 in-features).
//    Block size: 32 rows x 256 cols = 9216 bytes (1024 B exponents + 8192 B mantissas).
// 2. Prep step (b_group_sums / sum(x)) eliminated:
//    bfp16ebs8 has shared exponents and zero offset (no minimum term), completely removing
//    the group sum overhead and saving L1 buffer memory.
// 3. No intermediate bf16 narrowing of dot products:
//    Mantissa int8 (|m| <= 128) converts to bf16 exactly, and 2^e scale multiplication is exact.
//    Packing guarantees e >= 7 (pack_bfp16_gemv lowers e < 7 blocks to mantissa 0, e = 7), so the
//    scale is always the normal-bf16 bit pattern (e - 6) << 7 assembled with pure vector ops; the
//    kernel only applies a defensive vector max(e, 7) (raw e < 7 blocks decode to ~2^-126, not
//    subnormal garbage). Weights are exact in bf16, then multiplied by x (bf16) and accumulated
//    in float32. Zero narrowing loss during reduction across in-features K.
// 4. Rounding mode explicitly set to conv_even at entry points (bfp16_gemv_block and epilogue).
// 5. Parameterized K dimension via RAON_GEMV_K and generic K_TILE / M_TILE geometry.
//
//===----------------------------------------------------------------------===//

#if __has_include("flm_gemma4/utils.h")
#include "aie_kernel_utils.h"
#include "common/zero.h"
#include "flm_gemma4/utils.h"
#elif __has_include("../aie_kernel_utils.h")
#include "../aie_kernel_utils.h"
#include "../common/zero.h"
#include "utils.h"
#else
#include "aie_kernel_utils.h"
#include "zero.h"
#include "utils.h"
#endif

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(BFP16_M_TILE) || !defined(BFP16_K_TILE) || !defined(RAON_GEMV_K)
#define BFP16_M_TILE 32
#define BFP16_K_TILE 256
#ifndef RAON_GEMV_K
#define RAON_GEMV_K 2560
#endif
#endif

namespace {

constexpr int M_TILE = BFP16_M_TILE;
constexpr int K_TILE = BFP16_K_TILE;
constexpr int K_DIM = RAON_GEMV_K;
constexpr int K_BLOCKS = K_DIM / K_TILE;
constexpr int GROUP = 8;
constexpr int GROUPS_PER_TILE = K_TILE / GROUP; // 32
constexpr int HALF_ROWS = 16;

static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");
static_assert(K_TILE % GROUP == 0, "k tile must hold whole 8-element groups");
static_assert(K_DIM % K_TILE == 0, "K dimension must be a multiple of K_TILE");

} // namespace

// Layout of 1 group: 16 rows, 8 columns
// - exp: 16 bytes (1 exponent per row)
// - mant: 128 bytes (8 columns x 16 rows, column-major)
// Total: 144 bytes per group.
struct bfp16_group_half {
  uint8_t exp[HALF_ROWS];
  int8_t mant[GROUP][HALF_ROWS];
};

// Half: 32 groups for 16 rows = 32 * 144 = 4608 bytes.
struct bfp16_half {
  bfp16_group_half groups[GROUPS_PER_TILE];
};

// Full block: 2 halves = 32 rows x 256 cols = 2 * 4608 = 9216 bytes.
struct bfp16_block_t {
  bfp16_half half[2]; // half 0: rows 0..15, half 1: rows 16..31
};

static_assert(sizeof(bfp16_group_half) == 144, "bfp16_group_half must be 144 bytes");
static_assert(sizeof(bfp16_half) == 4608, "bfp16_half must be 4608 bytes");
static_assert(sizeof(bfp16_block_t) == 9216, "bfp16_block_t must be 9216 bytes");

extern "C" {

/// Initialize output accumulator tile to zero before accumulating over K blocks.
void bfp16_gemv_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

/// Accumulate one bfp16ebs8 weight block (32 rows x 256 cols):
/// y_acc += w * x[k * K_TILE : (k + 1) * K_TILE].
///
/// Arithmetic pipeline:
/// - int8 mantissas convert to bf16 exactly (7-bit fraction fits |m| <= 128).
/// - 2^(e - 133) scale factors convert to bf16 exactly via pure vector exponent assembly:
///   Packing guarantees e >= 7 (pack_bfp16_gemv lowers e < 7 blocks to mantissa 0, e = 7),
///   so scale is always a normal bf16 with biased exponent e - 6, assembled as (e - 6) << 7.
///   A defensive vector max(e, 7) guarantees zero-adjacent ~2^-126 scale even if un-packed
///   blocks with e < 7 are supplied. No branches, scalar loops, or memcpy are used.
/// - w = bf16(m) * scale is exact in bf16.
/// - acc += w * x (bf16 x bf16 -> accfloat MAC, 16-lane parallel).
/// No intermediate narrowing loss occurs across the K reduction.
void bfp16_gemv_block(const bfp16_block_t *w, const bfloat16 *x, float *y_acc, int k) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *x_tile = x + k * K_TILE;

  // Process rows 16 at a time (half 0: 0..15, half 1: 16..31)
  for (int h = 0; h < 2; h++) {
    const int row_offset = h * HALF_ROWS;
    aie::accum<accfloat, HALF_ROWS> acc;
    acc.from_vector(aie::load_v<HALF_ROWS>(y_acc + row_offset));

    const bfp16_half *cur_half = &w->half[h];

    for (int g = 0; g < GROUPS_PER_TILE; g++) {
      const bfp16_group_half *grp = &cur_half->groups[g];

      // Assemble 16 bf16 scale factors from 16 exponent bytes:
      // scale = 2^(e - 133).
      // Packing guarantees e >= 7 (pack_bfp16_gemv lowers e < 7 blocks to mantissa 0,
      // e = 7), so scale is always a normal bf16 value with biased exponent (e - 6),
      // bit pattern (e - 6) << 7.
      // Pure vector arithmetic without branches, scalar loops, or memcpy:
      // 1) Vector load 16 exponent bytes
      // 2) Defensive clamp max(e, 7) so un-packed blocks with e < 7 decode to ~2^-126
      // 3) Unpack uint8 -> uint16
      // 4) Subtract 6 to form the 8-bit biased exponent in bits [7:0]
      // 5) Shift left 7 (upshift) to place exponent in bits [14:7] of normal bf16
      // 6) Reinterpret bitcast to bfloat16 vector via cast_to<bfloat16>()
      aie::vector<uint8, HALF_ROWS> exp_u8 = aie::load_v<HALF_ROWS>(grp->exp);
      aie::vector<uint8, HALF_ROWS> exp_clamped =
          aie::max(exp_u8, aie::broadcast<uint8, HALF_ROWS>(7));
      aie::vector<uint16, HALF_ROWS> exp_u16 = exp_clamped.unpack();
      aie::vector<uint16, HALF_ROWS> exp_biased =
          aie::sub(exp_u16, aie::broadcast<uint16, HALF_ROWS>(6));
      aie::vector<uint16, HALF_ROWS> scale_bits = aie::upshift(exp_biased, 7);
      aie::vector<bfloat16, HALF_ROWS> scale_v =
          scale_bits.template cast_to<bfloat16>();

      // Unroll the 8 columns of this group
      const bfloat16 *x_grp = x_tile + g * GROUP;
#pragma clang loop unroll(full)
      for (int c = 0; c < GROUP; c++) {
        bfloat16 x_val = x_grp[c];
        aie::vector<int8, HALF_ROWS> m_vec = aie::load_v<HALF_ROWS>(grp->mant[c]);

        // Exact conversion: int8 -> accfloat -> bf16
        aie::accum<accfloat, HALF_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(m_vec, 0));
        aie::vector<bfloat16, HALF_ROWS> m_bf16 = m_acc.template to_vector<bfloat16>();

        // Exact weight: w = m * scale
        aie::vector<bfloat16, HALF_ROWS> w_col = aie::mul(m_bf16, scale_v);

        // MAC into float accumulator
        acc = aie::mac(acc, w_col, x_val);
      }
    }

    aie::store_v(y_acc + row_offset, acc.template to_vector<float>());
  }

  event1();
}

/// Narrow accumulated float32 output vector y_acc to bfloat16 with conv_even rounding.
void bfp16_gemv_epilogue(bfloat16 *y, const float *y_acc) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

} // extern "C"
