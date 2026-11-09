// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified by the raon-npu authors, 2026.
// Packed W ABI unchanged. X: [32][MAX_T][8], C: [2][MAX_T/R][2][R][8].
#include "aie_kernel_utils.h"
#include "common/zero.h"
#include "flm_gemma4/q4nx.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef PREFILL_MAX_T
#error "PREFILL_MAX_T must come from kernels/gemm_prefill/kernel.py compile_flags (-DPREFILL_MAX_T=<MAX_BATCH_T>)"
#endif
#ifndef PREFILL_Q4
#define PREFILL_Q4 0
#endif
namespace {
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
constexpr int R = 8;
#else
constexpr int R = 4;
#endif
using MM = aie::mmul<R, 8, 8, bfloat16, bfloat16, accfloat>;
struct Group { uint8_t exp[16]; int8_t mant[8][16]; };
struct Half { Group groups[32]; };
struct Packed { Half half[2]; };
static_assert(sizeof(Packed) == 9216, "9-bit ABI");
static_assert(PREFILL_MAX_T % (2*R) == 0, "whole 2x2 token tiles");
using Q4 = q4nx_block<32, 256>;

// Bit constants avoid scalar C++ float/integer/bfloat16 conversions.
static inline aie::vector<bfloat16,16> ones() {
  return aie::broadcast<uint16,16>(0x3f80).cast_to<bfloat16>();
}
static inline int ci(int h, int t, int side=0) {
  return h*PREFILL_MAX_T*16 + (t/R)*R*16 + side*R*8;
}
static inline aie::vector<bfloat16,R*8>
load_a(const bfloat16 *x, int k8, int t, int actual) {
  if (t+R <= actual) return aie::load_v<R*8>(x+k8*PREFILL_MAX_T*8+t*8);
  auto a = aie::zeros<bfloat16,R*8>();
  // Only the partial token block takes this path; each operation is 8 lanes.
#pragma clang loop unroll(full)
  for (int i=0; i<R; ++i)
    if (t+i < actual)
      a.insert<8>(i,aie::load_v<8>(x+k8*PREFILL_MAX_T*8+(t+i)*8));
  return a;
}
static inline void pack_column(bfloat16 *w, int k, aie::vector<bfloat16,16> v) {
  // [K8][row-half8][K-lane][row8], so B0 and B1 are contiguous 64 lanes.
  aie::store_v(w+(k/8)*128+(k%8)*8,v.extract<8>(0));
  aie::store_v(w+(k/8)*128+64+(k%8)*8,v.extract<8>(1));
}
static inline void checksum(float *c, aie::vector<bfloat16,16> v) {
  aie::accum<accfloat,16> acc(aie::load_v<16>(c));
  acc=aie::mac(acc,v,ones());
  aie::store_v(c,acc.to_vector<float>());
}

// Four accumulators live across ALL 32 K8 blocks. No C traffic in this loop.
static void exact_tile(const bfloat16 *w, const bfloat16 *x,
                       float *c, int h, int actual) {
  for (int t=0; t<actual; t+=2*R) {
    MM c00(aie::load_v<R*8>(c+ci(h,t)));
    MM c01(aie::load_v<R*8>(c+ci(h,t,1)));
    MM c10(aie::load_v<R*8>(c+ci(h,t+R)));
    MM c11(aie::load_v<R*8>(c+ci(h,t+R,1)));
    for (int k8=0; k8<32; ++k8) {
      auto b0=aie::load_v<64>(w+k8*128);
      auto b1=aie::load_v<64>(w+k8*128+64);
      auto a0=load_a(x,k8,t,actual);
      auto a1=load_a(x,k8,t+R,actual);
      c00.mac(a0,b0); c01.mac(a0,b1);
      c10.mac(a1,b0); c11.mac(a1,b1);
    }
    aie::store_v(c+ci(h,t),c00.template to_vector<float>());
    aie::store_v(c+ci(h,t,1),c01.template to_vector<float>());
    aie::store_v(c+ci(h,t+R),c10.template to_vector<float>());
    aie::store_v(c+ci(h,t+R,1),c11.template to_vector<float>());
  }
}

static inline aie::vector<bfloat16,16>
group_sum(const bfloat16 *x, int g, int token, int actual) {
  if (token >= actual) return aie::zeros<bfloat16,16>();
  const bfloat16 *p=x+g*4*PREFILL_MAX_T*8+token*8;
  auto values=aie::concat(aie::load_v<8>(p),
                          aie::load_v<8>(p+PREFILL_MAX_T*8),
                          aie::load_v<8>(p+2*PREFILL_MAX_T*8),
                          aie::load_v<8>(p+3*PREFILL_MAX_T*8));
  // Vector narrowing replaces the scalar (bfloat16)reduce_add cast.
  aie::accum<accfloat,16> sum(aie::broadcast<float,16>(aie::reduce_add(values)));
  return sum.to_vector<bfloat16>();
}
static void finish_group(const bfloat16 *x, float *c, int h, int t, int g,
                         int actual, const MM &lo, const MM &hi,
                         aie::vector<bfloat16,16> scale,
                         aie::vector<bfloat16,16> minimum) {
  auto d0=lo.template to_vector<bfloat16>();
  auto d1=hi.template to_vector<bfloat16>();
  auto c0=aie::load_v<R*8>(c+ci(h,t));
  auto c1=aie::load_v<R*8>(c+ci(h,t,1));
#pragma clang loop unroll(full)
  for (int i=0; i<R; ++i) {
    auto dot=aie::concat(d0.template extract<8>(i),d1.template extract<8>(i));
    aie::accum<accfloat,16> acc(aie::concat(c0.template extract<8>(i),
                                                         c1.template extract<8>(i)));
    acc=aie::mac(acc,dot,scale);
    acc=aie::mac(acc,minimum,group_sum(x,g,t+i,actual));
    auto value=acc.to_vector<float>();
    c0.insert<8>(i,value.extract<8>(0));
    c1.insert<8>(i,value.extract<8>(1));
  }
  aie::store_v(c+ci(h,t),c0);
  aie::store_v(c+ci(h,t,1),c1);
}
static void q4_tile(const bfloat16 *w, const bfloat16 *x, float *c,
                    int h, int g, int actual, const Q4 *packed) {
  auto scale=aie::load_v<16>(packed->scales+g*32+h*16);
  auto minimum=aie::load_v<16>(packed->mins+g*32+h*16);
  for (int t=0; t<actual; t+=2*R) {
    MM c00(aie::zeros<float,R*8>()),c01(aie::zeros<float,R*8>());
    MM c10(aie::zeros<float,R*8>()),c11(aie::zeros<float,R*8>());
    for (int k8=0; k8<4; ++k8) {
      auto b0=aie::load_v<64>(w+k8*128);
      auto b1=aie::load_v<64>(w+k8*128+64);
      auto a0=load_a(x,g*4+k8,t,actual);
      auto a1=load_a(x,g*4+k8,t+R,actual);
      c00.mac(a0,b0); c01.mac(a0,b1);
      c10.mac(a1,b0); c11.mac(a1,b1);
    }
    finish_group(x,c,h,t,g,actual,c00,c01,scale,minimum);
    finish_group(x,c,h,t+R,g,actual,c10,c11,scale,minimum);
  }
}
}
extern "C" {
void prefill_zero(float *c, int actual) {
  (void)actual;
  zero_vectorized<float,PREFILL_MAX_T*32,1,false>(c);
}
void prefill_epilogue(bfloat16 *y,const float *c,int actual) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int t=0; t<PREFILL_MAX_T; t+=R) {
    auto c00=aie::load_v<R*8>(c+ci(0,t));
    auto c01=aie::load_v<R*8>(c+ci(0,t,1));
    auto c10=aie::load_v<R*8>(c+ci(1,t));
    auto c11=aie::load_v<R*8>(c+ci(1,t,1));
#pragma clang loop unroll(full)
    for (int i=0; i<R; ++i) {
      aie::accum<accfloat,32> acc(aie::concat(
          c00.template extract<8>(i),c01.template extract<8>(i),
          c10.template extract<8>(i),c11.template extract<8>(i)));
      auto value=acc.to_vector<bfloat16>();
      if (t+i >= actual) value=aie::zeros<bfloat16,32>();
      aie::store_v(y+(t+i)*32,value);
    }
  }
}
// mode 0 full; 1 transport; 2 observable unpack checksum; 3 synthetic W=1.
void prefill_block(const void *packed,const bfloat16 *x,float *c,
                   int actual,int mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (mode==1) return;
#if PREFILL_Q4
  const Q4 *w=static_cast<const Q4 *>(packed);
  alignas(32) bfloat16 scratch[32*16];
  for (int h=0; h<2; ++h) {
    for (int g=0; g<8; ++g) {
      // uint4 pointer offsets are bytes, matching q4nx_accumulate.
      const uint4 *p=w->qs+h*2048+g*256;
      for (int sv=0; sv<4; ++sv) {
        auto values=aie::broadcast<uint16,128>(0x3f80).cast_to<bfloat16>();
        if (mode!=3) {
          aie::accum<accfloat,128> a;
          a.from_vector(aie::to_float(aie::load_v<128>(p+sv*64),0));
          values=a.to_vector<bfloat16>();
        }
#pragma clang loop unroll(full)
        for (int k=0; k<8; ++k) {
          auto value=values.template extract<16>(k);
          pack_column(scratch,sv*8+k,value);
          if (mode==2) checksum(c,value);
        }
      }
      if (mode!=2) q4_tile(scratch,x,c,h,g,actual,w);
    }
  }
#else
  const Packed *w=static_cast<const Packed *>(packed);
  alignas(32) bfloat16 scratch[256*16];
  for (int h=0; h<2; ++h) {
    for (int g=0; g<32; ++g) {
      const Group *p=&w->half[h].groups[g];
      auto e=aie::max(aie::load_v<16>(p->exp),aie::broadcast<uint8,16>(7));
      auto biased=aie::sub(e.unpack(),aie::broadcast<uint16,16>(6));
      auto scale=aie::upshift(biased,7).template cast_to<bfloat16>();
      // Two K values (32 lanes) per conversion and multiply: the AIE2P int8 -> float conversion works on 32 lanes
      // even when asked for 16, and the bf16 multiply is 32 lanes wide. mant x 2^e stays exact, so the bytes are the
      // same (DESIGN-prefill.md section 45).
      auto scale2=aie::concat(scale,scale);
#pragma clang loop unroll(full)
      for (int k=0; k<8; k+=2) {
        auto value=aie::broadcast<uint16,32>(0x3f80).cast_to<bfloat16>();
        if (mode!=3) {
          aie::accum<accfloat,32> mant;
          mant.from_vector(aie::to_float(aie::concat(aie::load_v<16>(p->mant[k]),
                                                     aie::load_v<16>(p->mant[k+1])),0));
          value=aie::mul(mant.template to_vector<bfloat16>(),scale2);
        }
        pack_column(scratch,g*8+k,value.template extract<16>(0));
        pack_column(scratch,g*8+k+1,value.template extract<16>(1));
        if (mode==2) { checksum(c,value.template extract<16>(0)); checksum(c,value.template extract<16>(1)); }
      }
    }
    if (mode!=2) exact_tile(scratch,x,c,h,actual);
  }
#endif
}
}
