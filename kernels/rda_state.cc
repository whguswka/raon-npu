//===- rda_state.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 raon-npu authors
// SPDX-License-Identifier: Apache-2.0
//
// RDA (Raon Delta Attention) fp32 state tile update kernel for AMD XDNA 2 (AIE2P).
//
// Computes recurrent RDA state update for one V-tile of 32 rows × 128 keys (16 KiB):
// For each row v in 0..31:
//   1. Decay:        s[v, :] = s[v, :] * a[:]               (vectorized fp32 elementwise)
//   2. Prediction:   pred    = dot(s[v, :], k[:])           (vectorized fp32 reduction)
//   3. Error:        err     = v[v] - pred                  (scalar fp32)
//   4. Rank-1 update:s[v, :] = s[v, :] + beta * err * k[:]  (vectorized fp32 fma)
//   5. Output:       out[v]  = dot(s[v, :], q[:])           (vectorized fp32 reduction)
// The three-phase schedule runs steps 1-2 for all rows, then step 3 for all rows, then steps 4-5 for all rows; per-row operations and order are unchanged.
// The fp32 vector products run 32 keys at a time (AIE2P emulates an fp32 multiply 32 lanes at a time); the two 16-key halves of each product are added to the 16-lane dot accumulators in key order, so every lane keeps the same operations in the same order.
// Phases 1 and 3 take two rows per iteration with the fp32 multiply emulation written out (fp32_split, fp32_mul2), so the nine-product add chains of the two rows share the multiplier; every row keeps the same operations in the same order.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

// Safe loop and profiling annotations across toolchains
#ifndef AIE_LOOP_UNROLL_FULL
#if defined(__clang__)
#define AIE_LOOP_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define AIE_LOOP_UNROLL_FULL
#endif
#endif

#ifndef AIE_PREPARE_FOR_PIPELINING
#define AIE_PREPARE_FOR_PIPELINING
#endif

#ifndef event0
#define event0() ((void)0)
#endif
#ifndef event1
#define event1() ((void)0)
#endif

// AIE2P has no scalar float multiply instruction.
// A plain scalar float multiply (*) lowers to a soft-float __mulsf3 libcall
// that costs tens of cycles and blocks software pipelining.
// We perform the scalar multiply through a vector lane via broadcast and vector mul.
static inline float scalar_mul(float a, float b) {
    return ::aie::mul(::aie::broadcast<float, 16>(a), b).to_vector<float>()[0];
}

// The fp32 multiply emulation of the toolchain header aie2p_vmult.h (mul_elem_32_accuracy_safe_inner and
// mac_elem_32_accuracy_safe_inner with sub_mul 0, which ::aie::mul and ::aie::mac of two float vectors call) written
// out, so that the products of two rows can be interleaved: the compiler runs each emulated product as one chain and
// does not interleave independent chains by itself. Each product keeps the header's three bf16 parts per operand, its
// nine bf16 products in the same order and (mac) the same final add, so the bits are those of ::aie::mul / ::aie::mac.
// Checked with mlir-aie 1.4.4 / llvm-aie clang 22.0.0; after a toolchain update compare old and new kernels again.
struct fp32_parts {
    v32bfloat16 hi, mid, lo;  // a, b, c (first operand) or d, e, f (second operand) of the header
};

static inline fp32_parts fp32_split(v32float v) {
    v32bfloat16 one = ::broadcast_one_to_v32bfloat16();
    fp32_parts p;
    p.hi = ::to_v32bfloat16((v32accfloat)v);
    v32accfloat rest = ::msc_elem_32(p.hi, one, (v32accfloat)v);
    p.mid = ::to_v32bfloat16(rest);
    p.lo = ::to_v32bfloat16(::msc_elem_32(p.mid, one, rest));
    return p;
}

// The nine-product sums of x1 * y1 and x2 * y2 (x = first operand, y = second operand), step by step interleaved.
static inline void fp32_mul2(const fp32_parts &x1, const fp32_parts &y1, v32accfloat &t1,
                             const fp32_parts &x2, const fp32_parts &y2, v32accfloat &t2) {
    t1 = ::mul_elem_32_conf(x1.lo, y1.lo, 0);
    t2 = ::mul_elem_32_conf(x2.lo, y2.lo, 0);
    t1 = ::add(t1, ::mul_elem_32_conf(x1.lo, y1.mid, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.lo, y2.mid, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.mid, y1.lo, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.mid, y2.lo, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.hi, y1.lo, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.hi, y2.lo, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.mid, y1.mid, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.mid, y2.mid, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(y1.hi, x1.lo, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(y2.hi, x2.lo, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.mid, y1.hi, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.mid, y2.hi, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.hi, y1.mid, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.hi, y2.mid, 0));
    t1 = ::add(t1, ::mul_elem_32_conf(x1.hi, y1.hi, 0));
    t2 = ::add(t2, ::mul_elem_32_conf(x2.hi, y2.hi, 0));
}

extern "C" {

/// \brief Update one 32x128 fp32 state tile with RDA recurrent formulation.
///
/// Uses a packed head input bundle to satisfy AIE2P Core Tile DMA channel
/// limits (maximum 2 S2MM input channels: 1 for state tile, 1 for head bundle).
/// Local arrays pred and scale (256 B) and the bf16 parts of a, k and q (3,072 B) use the 12,288 B RDA core stack
/// configured in bench/rda_state/rda_state_1col.py (rda_update_tile frame 4,736 B with mlir-aie 1.4.4).
///
/// \param s_in     Input state tile [32][128] fp32 (16 KiB, 64-byte aligned: AIE2P loads and stores of 512-bit and wider vectors need 64 B)
/// \param s_out    Output state tile [32][128] fp32 (16 KiB, 64-byte aligned)
/// \param head_in  Packed head input bundle [640] fp32 (2,560 B, 64-byte aligned):
///                   - 0..127:   q [128] (512 B, 64-byte aligned)
///                   - 128..255: k [128] (512 B, 64-byte aligned)
///                   - 256..383: a [128] = exp(log_alpha) (512 B, 64-byte aligned)
///                   - 384..511: v [128] (512 B, 64-byte aligned; tile v32 at 384 + tile_idx * 32)
///                   - 512:      beta [1] (4 B, 64-byte aligned)
///                   - 513..639: padding (127 floats, zeroed for 64-byte vector alignment)
/// \param tile_idx V-dimension tile index (0, 1, 2, or 3)
/// \param out32    Output projection slice [32] fp32 (128 B, 32-byte aligned)
void rda_update_tile(
    const float *__restrict s_in,
    float *__restrict s_out,
    const float *__restrict head_in,
    int32_t tile_idx,
    float *__restrict out32
) {
    event0();

    const float *__restrict q = head_in;
    const float *__restrict k = head_in + 128;
    const float *__restrict a = head_in + 256;
    const float *__restrict v32 = head_in + 384 + tile_idx * 32;
    const float beta_val = head_in[512];

    // Three phases over the 32 value rows of this tile. Every row keeps exactly the same operations in the same order
    // as a row-by-row loop (so the bits are the same); only the schedule changes: the scalar step no longer sits
    // between the two vector passes of each row, so the vector passes of consecutive rows run back to back.
    alignas(64) float pred[32];
    alignas(64) float scale[32];

    // ========================================================================
    // Phase 1: decay every row and compute pred[row] = dot(s_decayed, k)
    // ========================================================================
    // Two rows per iteration: a and k are split into their bf16 parts once (the emulation would split them into the
    // same parts in every multiply), then the decay and key products of rows row and row + 1 run as interleaved
    // emulated multiplies.
    fp32_parts a_parts[4];
    fp32_parts k_parts1[4];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 4; ++c) {
        a_parts[c] = fp32_split(::aie::load_v<32>(a + 32 * c));
        k_parts1[c] = fp32_split(::aie::load_v<32>(k + 32 * c));
    }
    for (int row = 0; row < 32; row += 2) {
        const float *__restrict s_in_r0 = s_in + row * 128;
        const float *__restrict s_in_r1 = s_in + (row + 1) * 128;
        float *__restrict s_out_r0 = s_out + row * 128;
        float *__restrict s_out_r1 = s_out + (row + 1) * 128;
        ::aie::accum<accfloat, 16> dot_k0 = ::aie::zeros<accfloat, 16>();
        ::aie::accum<accfloat, 16> dot_k1 = ::aie::zeros<accfloat, 16>();

        AIE_LOOP_UNROLL_FULL
        for (int c = 0; c < 4; ++c) {
            const int i = 32 * c;
            fp32_parts s0 = fp32_split(::aie::load_v<32>(s_in_r0 + i));
            fp32_parts s1 = fp32_split(::aie::load_v<32>(s_in_r1 + i));
            v32accfloat t0, t1;
            fp32_mul2(s0, a_parts[c], t0, s1, a_parts[c], t1);  // s_decay = s * a
            ::aie::vector<float, 32> d0 = ::aie::accum<accfloat, 32>(t0).template to_vector<float>();
            ::aie::vector<float, 32> d1 = ::aie::accum<accfloat, 32>(t1).template to_vector<float>();
            ::aie::store_v(s_out_r0 + i, d0);
            ::aie::store_v(s_out_r1 + i, d1);

            fp32_parts e0 = fp32_split(d0);
            fp32_parts e1 = fp32_split(d1);
            v32accfloat p0, p1;
            fp32_mul2(e0, k_parts1[c], p0, e1, k_parts1[c], p1);  // dot_k += s_decay * k, halves in key order
            ::aie::accum<accfloat, 32> prod0(p0);
            ::aie::accum<accfloat, 32> prod1(p1);
            dot_k0 = ::aie::add(dot_k0, prod0.template extract<16>(0));
            dot_k1 = ::aie::add(dot_k1, prod1.template extract<16>(0));
            dot_k0 = ::aie::add(dot_k0, prod0.template extract<16>(1));
            dot_k1 = ::aie::add(dot_k1, prod1.template extract<16>(1));
        }
        pred[row] = ::aie::reduce_add(dot_k0.template to_vector<float>());
        pred[row + 1] = ::aie::reduce_add(dot_k1.template to_vector<float>());
    }

    // ========================================================================
    // Phase 2: scalar error and update factor for every row
    // ========================================================================
    for (int row = 0; row < 32; ++row) {
        float err = v32[row] - pred[row];
        scale[row] = scalar_mul(beta_val, err);
    }

    // ========================================================================
    // Phase 3: rank-1 update s_new = s_decay + scale * k and out[row] = dot(s_new, q)
    // ========================================================================
    // Two rows per iteration: k and q are split once and each row's broadcast scale once (the emulation would split
    // them into the same parts in every chunk), then the update mac and query products of rows row and row + 1 run as
    // interleaved emulated products.
    fp32_parts k_parts3[4];
    fp32_parts q_parts[4];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 4; ++c) {
        k_parts3[c] = fp32_split(::aie::load_v<32>(k + 32 * c));
        q_parts[c] = fp32_split(::aie::load_v<32>(q + 32 * c));
    }
    for (int row = 0; row < 32; row += 2) {
        float *__restrict s_out_r0 = s_out + row * 128;
        float *__restrict s_out_r1 = s_out + (row + 1) * 128;
        fp32_parts g0 = fp32_split(::aie::broadcast<float, 32>(scale[row]));
        fp32_parts g1 = fp32_split(::aie::broadcast<float, 32>(scale[row + 1]));
        ::aie::accum<accfloat, 16> dot_q0 = ::aie::zeros<accfloat, 16>();
        ::aie::accum<accfloat, 16> dot_q1 = ::aie::zeros<accfloat, 16>();

        AIE_LOOP_UNROLL_FULL
        for (int c = 0; c < 4; ++c) {
            const int i = 32 * c;
            v32accfloat t0, t1;
            fp32_mul2(g0, k_parts3[c], t0, g1, k_parts3[c], t1);  // scale * k
            ::aie::accum<accfloat, 32> acc0;
            ::aie::accum<accfloat, 32> acc1;
            acc0.from_vector(::aie::load_v<32>(s_out_r0 + i));
            acc1.from_vector(::aie::load_v<32>(s_out_r1 + i));
            v32accfloat n0 = ::add_conf((v32accfloat)acc0, t0, 0, 0, 0);  // the mac's final add: s_new = s + scale * k
            v32accfloat n1 = ::add_conf((v32accfloat)acc1, t1, 0, 0, 0);
            ::aie::vector<float, 32> s_new0 = ::aie::accum<accfloat, 32>(n0).template to_vector<float>();
            ::aie::vector<float, 32> s_new1 = ::aie::accum<accfloat, 32>(n1).template to_vector<float>();
            ::aie::store_v(s_out_r0 + i, s_new0);
            ::aie::store_v(s_out_r1 + i, s_new1);

            fp32_parts e0 = fp32_split(s_new0);
            fp32_parts e1 = fp32_split(s_new1);
            v32accfloat p0, p1;
            fp32_mul2(e0, q_parts[c], p0, e1, q_parts[c], p1);  // dot_q += s_new * q, halves in key order
            ::aie::accum<accfloat, 32> prod0(p0);
            ::aie::accum<accfloat, 32> prod1(p1);
            dot_q0 = ::aie::add(dot_q0, prod0.template extract<16>(0));
            dot_q1 = ::aie::add(dot_q1, prod1.template extract<16>(0));
            dot_q0 = ::aie::add(dot_q0, prod0.template extract<16>(1));
            dot_q1 = ::aie::add(dot_q1, prod1.template extract<16>(1));
        }
        out32[row] = ::aie::reduce_add(dot_q0.template to_vector<float>());
        out32[row + 1] = ::aie::reduce_add(dot_q1.template to_vector<float>());
    }

    event1();
}

/// \brief Packed variant: one pointer holds both state tile and output slice.
///
/// The packed layout places the updated 32x128 state tile (4096 floats) at
/// offset 0, followed immediately by the 32-element output slice at offset
/// 4096.  This lets a single core→memtile DMA transfer move both results in
/// one 4128-float element, which the memory tile then splits into the two
/// host BOs with different TAPs.
///
/// \param s_in           Input state tile [32][128] fp32 (16 KiB, 64-byte aligned)
/// \param s_out_packed   Output buffer [4128] fp32:
///                         - 0..4095: updated state tile
///                         - 4096..4127: output slice
/// \param head_in        Packed head input bundle [640] fp32
/// \param tile_idx       V-dimension tile index (0, 1, 2, or 3)
///
/// This wrapper calls the original rda_update_tile with s_out_packed as the
/// state output and s_out_packed + 4096 as the output slice.  The two pointers
/// are disjoint regions within the same object (4096 + 32 = 4128 floats),
/// so the restrict contract of rda_update_tile is preserved: s_out and out32
/// alias only through the single packed buffer object, and the kernel treats
/// them as independent arrays with no overlapping accesses.
void rda_update_tile_packed(
    const float *__restrict s_in,
    float *__restrict s_out_packed,
    const float *__restrict head_in,
    int32_t tile_idx
) {
    rda_update_tile(s_in, s_out_packed, head_in, tile_idx, s_out_packed + 4096);
}

} // extern "C"
