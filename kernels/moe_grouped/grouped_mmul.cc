// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified by the raon-npu authors, 2026.
// One AIE2P program selects exact 9-bit or q4nx unpack/mmul by RTP mode.
// RTP word 4 carries each slot's real rows: an empty slot returns at once and a q4nx slot with at most 4 rows
// computes one 4-row tile. Rows are independent, so the real rows keep their bytes; padding rows are never read.
#include "aie_kernel_utils.h"
#include "common/zero.h"
#include "flm_gemma4/q4nx.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int MAX_T = 8;
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
constexpr int R = 8;
#else
constexpr int R = 4;
#endif
using MM = aie::mmul<R, 8, 8, bfloat16, bfloat16, accfloat>;
using Q4 = q4nx_block<32, 256>;
struct Group { uint8_t exp[16]; int8_t mant[8][16]; };
struct Half { Group groups[32]; };
struct Packed9 { Half half[2]; };
static_assert(sizeof(Packed9) == 9216, "9-bit block ABI");
static_assert(MAX_T % (2 * R) == 0, "whole 2x2 token tiles");

static inline aie::vector<bfloat16, 16> ones() {
  return aie::broadcast<uint16, 16>(0x3f80).cast_to<bfloat16>();
}
static inline int ci(int half, int token, int side = 0) {
  return half * MAX_T * 16 + (token / R) * R * 16 + side * R * 8;
}
static inline aie::vector<bfloat16, R * 8>
load_a(const bfloat16 *x, int k8, int token, int actual) {
  if (token + R <= actual)
    return aie::load_v<R * 8>(x + k8 * MAX_T * 8 + token * 8);
  auto value = aie::zeros<bfloat16, R * 8>();
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i)
    if (token + i < actual)
      value.insert<8>(i, aie::load_v<8>(x + k8 * MAX_T * 8 + (token + i) * 8));
  return value;
}
static inline void pack_column(bfloat16 *w, int k,
                               aie::vector<bfloat16, 16> value) {
  aie::store_v(w + (k / 8) * 128 + (k % 8) * 8, value.extract<8>(0));
  aie::store_v(w + (k / 8) * 128 + 64 + (k % 8) * 8,
               value.extract<8>(1));
}
static void exact_tile(const bfloat16 *w, const bfloat16 *x, float *c,
                       int half, int actual) {
  for (int t = 0; t < actual; t += 2 * R) {
    MM c00(aie::load_v<R * 8>(c + ci(half, t)));
    MM c01(aie::load_v<R * 8>(c + ci(half, t, 1)));
    MM c10(aie::load_v<R * 8>(c + ci(half, t + R)));
    MM c11(aie::load_v<R * 8>(c + ci(half, t + R, 1)));
    for (int k8 = 0; k8 < 32; ++k8) {
      auto b0 = aie::load_v<64>(w + k8 * 128);
      auto b1 = aie::load_v<64>(w + k8 * 128 + 64);
      auto a0 = load_a(x, k8, t, actual);
      auto a1 = load_a(x, k8, t + R, actual);
      c00.mac(a0, b0); c01.mac(a0, b1);
      c10.mac(a1, b0); c11.mac(a1, b1);
    }
    aie::store_v(c + ci(half, t), c00.template to_vector<float>());
    aie::store_v(c + ci(half, t, 1), c01.template to_vector<float>());
    aie::store_v(c + ci(half, t + R), c10.template to_vector<float>());
    aie::store_v(c + ci(half, t + R, 1), c11.template to_vector<float>());
  }
}
static inline bfloat16 group_sum_scalar(const bfloat16 *x, int group, int token) {
  const bfloat16 *p = x + group * 4 * MAX_T * 8 + token * 8;
  auto values = aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + MAX_T * 8),
                            aie::load_v<8>(p + 2 * MAX_T * 8),
                            aie::load_v<8>(p + 3 * MAX_T * 8));
  aie::accum<accfloat, 16> sum(aie::broadcast<float, 16>(aie::reduce_add(values)));
  return sum.to_vector<bfloat16>().get(0);
}
static void finish_group(const bfloat16 *gs, float *c, int half, int token,
                         int actual, const MM &lo, const MM &hi,
                         aie::vector<bfloat16, 16> scale,
                         aie::vector<bfloat16, 16> minimum) {
  auto d0 = lo.template to_vector<bfloat16>();
  auto d1 = hi.template to_vector<bfloat16>();
  auto c0 = aie::load_v<R * 8>(c + ci(half, token));
  auto c1 = aie::load_v<R * 8>(c + ci(half, token, 1));
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i) {
    auto dot = aie::concat(d0.template extract<8>(i),
                           d1.template extract<8>(i));
    aie::accum<accfloat, 16> acc(aie::concat(c0.template extract<8>(i),
                                             c1.template extract<8>(i)));
    acc = aie::mac(acc, dot, scale);
    auto gv = token + i < actual
                  ? aie::broadcast<bfloat16, 16>(gs[token + i])
                  : aie::zeros<bfloat16, 16>();
    acc = aie::mac(acc, minimum, gv);
    auto value = acc.to_vector<float>();
    c0.insert<8>(i, value.extract<8>(0));
    c1.insert<8>(i, value.extract<8>(1));
  }
  aie::store_v(c + ci(half, token), c0);
  aie::store_v(c + ci(half, token, 1), c1);
}
static void q4_tile(const bfloat16 *w, const bfloat16 *x, float *c,
                    const bfloat16 *gs, int half, int group, int actual,
                    const Q4 *packed) {
  auto scale = aie::load_v<16>(packed->scales + group * 32 + half * 16);
  auto minimum = aie::load_v<16>(packed->mins + group * 32 + half * 16);
  // Tile loads take whole 4-row vectors (load_a with MAX_T, as when every call passed TMAX): X padding rows are
  // zero (planner) and cannot reach a real row.
  for (int t = 0; t < actual; t += 2 * R) {
    if (t + R >= actual) {  // at most R real rows left: the first 4-row tile only
      MM c00(aie::zeros<float, R * 8>()), c01(aie::zeros<float, R * 8>());
      for (int k8 = 0; k8 < 4; ++k8) {
        auto b0 = aie::load_v<64>(w + k8 * 128);
        auto b1 = aie::load_v<64>(w + k8 * 128 + 64);
        auto a0 = load_a(x, group * 4 + k8, t, MAX_T);
        c00.mac(a0, b0); c01.mac(a0, b1);
      }
      finish_group(gs, c, half, t, actual, c00, c01, scale, minimum);
      continue;
    }
    MM c00(aie::zeros<float, R * 8>()), c01(aie::zeros<float, R * 8>());
    MM c10(aie::zeros<float, R * 8>()), c11(aie::zeros<float, R * 8>());
    for (int k8 = 0; k8 < 4; ++k8) {
      auto b0 = aie::load_v<64>(w + k8 * 128);
      auto b1 = aie::load_v<64>(w + k8 * 128 + 64);
      auto a0 = load_a(x, group * 4 + k8, t, MAX_T);
      auto a1 = load_a(x, group * 4 + k8, t + R, MAX_T);
      c00.mac(a0, b0); c01.mac(a0, b1);
      c10.mac(a1, b0); c11.mac(a1, b1);
    }
    finish_group(gs, c, half, t, actual, c00, c01, scale, minimum);
    finish_group(gs, c, half, t + R, actual, c10, c11, scale, minimum);
  }
}
} // namespace

extern "C" {
void grouped_zero(float *c, int actual) {
  (void)actual;
  zero_vectorized<float, MAX_T * 32, 1, false>(c);
}
void grouped_epilogue(bfloat16 *y, const float *c, int actual) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int t = 0; t < MAX_T; t += R) {
    auto c00 = aie::load_v<R * 8>(c + ci(0, t));
    auto c01 = aie::load_v<R * 8>(c + ci(0, t, 1));
    auto c10 = aie::load_v<R * 8>(c + ci(1, t));
    auto c11 = aie::load_v<R * 8>(c + ci(1, t, 1));
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) {
      aie::accum<accfloat, 32> acc(aie::concat(
          c00.template extract<8>(i), c01.template extract<8>(i),
          c10.template extract<8>(i), c11.template extract<8>(i)));
      auto value = acc.to_vector<bfloat16>();
      if (t + i >= actual) value = aie::zeros<bfloat16, 32>();
      aie::store_v(y + (t + i) * 32, value);
    }
  }
}

// mode: 0 = exact 9-bit bfp16ebs8, 1 = q4nx. rows: real rows of the 8 slots, 4 bits each (slot s in bits 4s..4s+3,
// 0..8; the B2 stream keeps 8 rows everywhere); slot: the slot of this block.
void grouped_block(const void *packed, const bfloat16 *x, float *c, bfloat16 *sums,
                   int rows, int mode, int round, int k, int slot) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  // actual within 1..MAX_T: together with the whole-vector tile loads and the at-most-4-rows branch of q4_tile this
  // made a q4nx call ~19% faster even with 8 rows in every slot (each change alone gave little; mlir-aie 1.4.4).
  int actual = (int)(((unsigned)rows >> (4 * slot)) & 0xFu);
  if (actual > MAX_T) actual = MAX_T;
  if (actual == 0) return;  // empty slot: its output rows stay zero (grouped_zero) and are never read
  if (mode == 0) {
    (void)sums;
    (void)round;
    (void)k;
    const Packed9 *w = static_cast<const Packed9 *>(packed);
    alignas(32) bfloat16 scratch[256 * 16];
    for (int half = 0; half < 2; ++half) {
      for (int group = 0; group < 32; ++group) {
        const Group *p = &w->half[half].groups[group];
        auto exponent = aie::max(aie::load_v<16>(p->exp),
                                 aie::broadcast<uint8, 16>(7));
        auto biased = aie::sub(exponent.unpack(), aie::broadcast<uint16, 16>(6));
        auto scale = aie::upshift(biased, 7).template cast_to<bfloat16>();
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; ++k) {
          aie::accum<accfloat, 16> mant;
          mant.from_vector(aie::to_float(aie::load_v<16>(p->mant[k]), 0));
          pack_column(scratch, group * 8 + k,
                      aie::mul(mant.template to_vector<bfloat16>(), scale));
        }
      }
      exact_tile(scratch, x, c, half, actual);
    }
    return;
  }
  bfloat16 *ks = sums + k * 8 * MAX_T;
  if (round == 0) {
    for (int group = 0; group < 8; ++group) {
      for (int token = 0; token < MAX_T; ++token) {
        ks[group * MAX_T + token] =
            token < actual ? group_sum_scalar(x, group, token) : bfloat16(0);
      }
    }
  }
  const Q4 *w = static_cast<const Q4 *>(packed);
  alignas(32) bfloat16 scratch[32 * 16];
  for (int half = 0; half < 2; ++half) {
    for (int group = 0; group < 8; ++group) {
      const uint4 *p = w->qs + half * 2048 + group * 256;
      for (int sv = 0; sv < 4; ++sv) {
        aie::accum<accfloat, 128> unpacked;
        unpacked.from_vector(aie::to_float(aie::load_v<128>(p + sv * 64), 0));
        auto values = unpacked.to_vector<bfloat16>();
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; ++k)
          pack_column(scratch, sv * 8 + k, values.template extract<16>(k));
      }
      q4_tile(scratch, x, c, ks + group * MAX_T, half, group, actual, w);
    }
  }
}
}
