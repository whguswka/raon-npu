//===- q4nx_gemv.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified by the raon-npu authors, 2026.
// Portions modified for raon-npu (KRAFTON Raon-1-mini general GEMV without RMS norm or softcap)
//
// Differences from AMD original (flm_gemma4_q4nx_lm_head):
// 1. Removed RMS normalization (pure general linear projection GEMV).
// 2. Removed tanh softcap epilogue (direct narrow_to_bf16).
// 3. Rounding mode explicitly set to conv_even in BOTH q4nx_gemv_prep and q4nx_gemv_block.
//    [CRITICAL RATIONALE FOR BOTH SITES]:
//    - q4nx_accumulate narrows D = sum(code * x) to bf16 (temp_acc.to_vector<bf16>()).
//    - q4nx_gemv_prep narrows S = sum(x) to bf16 (group_sums_32).
//    - Default AIE rounding mode is floor (truncation towards -infinity), biasing both
//      D and S downwards. In y = scale * D + min * S, since min ≈ -7.5 * scale, the downward
//      bias of D (positive term) and S (negative term) partially cancel out under floor.
//    - If conv_even is set ONLY in q4nx_gemv_block, D becomes unbiased (nearest-even) while
//      S remains floor-biased; this breaks the cancellation and severely degrades accuracy
//      (e.g., K=12032 relative L2 error worsens from 1.5e-2 to 8.4e-2).
//    - Setting conv_even at the entry of BOTH q4nx_gemv_prep (for S) and q4nx_gemv_block (for D)
//      symmetrically eliminates rounding bias in both terms, significantly improving accuracy
//      (K=12032, M=2560 rel L2 drops from 1.50e-2 to 4.79e-3; cosine 0.999887 -> 0.999989;
//       K=2560, M=20480 rel L2 drops from 1.23e-2 to 7.09e-3) with zero NPU runtime penalty.
// 4. Parameterized K dimension via RAON_GEMV_K and generic K_TILE / M_TILE geometry.
//
//===----------------------------------------------------------------------===//

// General matrix-vector multiplication against a q4nx-quantized matrix:
// Precompute the per-32-column sums of x, accumulate one weight block at a time,
// then narrow the accumulated float result to bf16.
// One entry point per step of the caller's streaming loop. The caller owns the
// synchronization, so none of these locks or double-buffers. Geometry arrives
// as -D flags; the Q4NX_ prefix avoids R, S and T, which collide with template
// parameters in the aie2p built-in headers.

#if __has_include("flm_gemma4/q4nx.h")
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

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) ||  \
    !defined(RAON_GEMV_K)
#error "q4nx_gemv.cc needs its geometry -D defined (Q4NX_M_TILE, Q4NX_K_TILE, Q4NX_GROUP, RAON_GEMV_K)"
#endif

// Named outside the anonymous namespace because it appears in an extern "C"
// signature.
using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int GROUP = Q4NX_GROUP;
constexpr int K_DIM = RAON_GEMV_K;

constexpr int K_BLOCKS = K_DIM / K_TILE;

static_assert(GROUP == 32, "a column sum covers exactly one group");
static_assert(K_TILE % GROUP == 0, "a k tile must hold whole groups");
static_assert(K_DIM % K_TILE == 0,
              "the block loop covers whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");

} // namespace

extern "C" {

/// Once per token / vector: precompute the per-32-column sums of x that every weight block reuses.
/// RMS normalization is omitted (pure linear projection).
/// Sets conv_even rounding mode so that column sum narrowing to bf16 is unbiased.
void q4nx_gemv_prep(bfloat16 *x, bfloat16 *b_group_sums) {
  event0();
  // Must match q4nx_gemv_block rounding to preserve symmetry between D and S
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int k = 0; k < K_BLOCKS; k++) {
    group_sums_32<K_TILE>(b_group_sums + k * (K_TILE / 32), x + k * K_TILE);
  }
  event1();
}

/// Once per output tile, before the k loop.
void q4nx_gemv_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

/// One weight block: y_acc += w * x[k]. k selects the slice of x and of the
/// column sums.
/// Sets conv_even rounding mode for D = dot(code, x) narrowing to bf16.
void q4nx_gemv_block(const q4nx_block_t *w, const bfloat16 *x, float *y_acc,
                     const bfloat16 *b_group_sums, int k) {
  event0();
  // Must match q4nx_gemv_prep rounding to prevent asymmetric floor/even cancellation breakdown
  aie::set_rounding(aie::rounding_mode::conv_even);
  q4nx_accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc,
                                  b_group_sums + k * (K_TILE / 32));
  event1();
}

/// Once per output tile, after the k loop: narrow y_acc to bf16 directly.
/// Softcap tanh is omitted.
void q4nx_gemv_epilogue(bfloat16 *y, const float *y_acc) {
  event0();
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

} // extern "C"
