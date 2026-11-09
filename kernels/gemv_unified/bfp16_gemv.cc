//===- bfp16_gemv.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified by the raon-npu authors, 2026.
// Portions modified for raon-npu authors: the X buffer has a fixed maximum
// extent and the runtime k-block index selects the active 256-element slice.
//
//===----------------------------------------------------------------------===//

#if __has_include("flm_gemma4/utils.h")
#include "aie_kernel_utils.h"
#include "common/zero.h"
#include "flm_gemma4/q4nx.h"
#include "flm_gemma4/utils.h"
#elif __has_include("../aie_kernel_utils.h")
#include "../aie_kernel_utils.h"
#include "../common/zero.h"
#include "q4nx.h"
#include "utils.h"
#else
#include "aie_kernel_utils.h"
#include "zero.h"
#include "q4nx.h"
#include "utils.h"
#endif

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef BFP16_M_TILE
#define BFP16_M_TILE 32
#endif
#ifndef BFP16_K_TILE
#define BFP16_K_TILE 256
#endif

namespace {
constexpr int M_TILE = BFP16_M_TILE;
constexpr int K_TILE = BFP16_K_TILE;
constexpr int GROUP = 8;
constexpr int GROUPS_PER_TILE = K_TILE / GROUP;
constexpr int HALF_ROWS = 16;
static_assert(M_TILE == 32, "the packed block ABI is fixed at 32 output rows");
static_assert(K_TILE == 256, "the packed block ABI is fixed at 256 input values");
} // namespace

struct bfp16_group_half {
  uint8_t exp[HALF_ROWS];
  int8_t mant[GROUP][HALF_ROWS];
};
struct bfp16_half { bfp16_group_half groups[GROUPS_PER_TILE]; };
struct bfp16_block_t { bfp16_half half[2]; };
using q4nx_unified_block_t = q4nx_block<M_TILE, K_TILE>;

static_assert(sizeof(bfp16_group_half) == 144, "packed group ABI changed");
static_assert(sizeof(bfp16_half) == 4608, "packed half ABI changed");
static_assert(sizeof(bfp16_block_t) == 9216, "packed block ABI changed");

extern "C" {

void bfp16_gemv_unified_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

// Arithmetic is intentionally identical to kernels/gemv_bfp16/bfp16_gemv.cc.
// K is not a compile-time define: the IRON worker supplies k=0..k_blocks-1.
void bfp16_gemv_unified_block(const bfp16_block_t *w, const bfloat16 *x,
                              float *y_acc, int k) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *x_tile = x + k * K_TILE;

  for (int h = 0; h < 2; h++) {
    const int row_offset = h * HALF_ROWS;
    aie::accum<accfloat, HALF_ROWS> acc;
    acc.from_vector(aie::load_v<HALF_ROWS>(y_acc + row_offset));
    const bfp16_half *cur_half = &w->half[h];

    for (int g = 0; g < GROUPS_PER_TILE; g++) {
      const bfp16_group_half *grp = &cur_half->groups[g];
      aie::vector<uint8, HALF_ROWS> exp_u8 = aie::load_v<HALF_ROWS>(grp->exp);
      aie::vector<uint8, HALF_ROWS> exp_clamped =
          aie::max(exp_u8, aie::broadcast<uint8, HALF_ROWS>(7));
      aie::vector<uint16, HALF_ROWS> exp_u16 = exp_clamped.unpack();
      aie::vector<uint16, HALF_ROWS> exp_biased =
          aie::sub(exp_u16, aie::broadcast<uint16, HALF_ROWS>(6));
      aie::vector<uint16, HALF_ROWS> scale_bits = aie::upshift(exp_biased, 7);
      aie::vector<bfloat16, HALF_ROWS> scale_v =
          scale_bits.template cast_to<bfloat16>();

      const bfloat16 *x_grp = x_tile + g * GROUP;
#pragma clang loop unroll(full)
      for (int c = 0; c < GROUP; c++) {
        const bfloat16 x_val = x_grp[c];
        aie::vector<int8, HALF_ROWS> m_vec =
            aie::load_v<HALF_ROWS>(grp->mant[c]);
        aie::accum<accfloat, HALF_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(m_vec, 0));
        aie::vector<bfloat16, HALF_ROWS> m_bf16 =
            m_acc.template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF_ROWS> w_col = aie::mul(m_bf16, scale_v);
        acc = aie::mac(acc, w_col, x_val);
      }
    }
    aie::store_v(y_acc + row_offset, acc.template to_vector<float>());
  }
  event1();
}

void bfp16_gemv_unified_epilogue(bfloat16 *y, const float *y_acc) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

// q4nx and bfp16ebs8 live in this same object so one core ELF can select the
// arithmetic family from RTP format.  k_blocks is dynamic because the shared
// xclbin fixes only the maximum x allocation, not K.
void q4nx_gemv_unified_prep(const bfloat16 *x, bfloat16 *group_sums,
                            int k_blocks) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int k = 0; k < k_blocks; k++) {
    group_sums_32<K_TILE>(group_sums + k * (K_TILE / 32), x + k * K_TILE);
  }
  event1();
}

void q4nx_gemv_unified_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

void q4nx_gemv_unified_block(const q4nx_unified_block_t *w,
                             const bfloat16 *x, float *y_acc,
                             const bfloat16 *group_sums, int k) {
  event0();
  // Keep prep and block symmetric: both use conv_even, matching the bit model.
  aie::set_rounding(aie::rounding_mode::conv_even);
  q4nx_accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc,
                                  group_sums + k * (K_TILE / 32));
  event1();
}

void q4nx_gemv_unified_epilogue(bfloat16 *y, const float *y_acc) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

} // extern "C"
