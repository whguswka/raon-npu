//===- rda_state_multi.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 raon-npu authors
// SPDX-License-Identifier: Apache-2.0
//
// Prefill RDA entry for T tokens per call (DESIGN-prefill.md section 36, stage R3-2): the decode kernel rda_state.cc
// itself, run once per token on the 32x128 state tile a core received.
// rda_update_tile decays s_in into s_out and updates s_out in place, so token t reads the state from one core buffer
// and writes it to the other: t = 0 reads the received tile S and writes packed, then the state alternates between
// packed and S. out32 always goes behind the state in packed (packed + 4096 + 32 t, disjoint from the state as in
// rda_update_tile_packed). With an even T the last state lands in S and is copied into packed, which is what leaves
// the core. Every token runs the same function on the same values, so the bytes equal T single-token calls, and T = 1
// is the rda_update_tile_packed call. rda_state.cc and its object stay as they are: the decode xclbin does not change.
//
//===----------------------------------------------------------------------===//

#include "rda_state.cc"

extern "C" {

/// \param S         State tile the core received [32][128] fp32; also the state buffer of odd tokens
/// \param packed    [4096 + 32 * 8] fp32: the state tile that leaves the core, then output slice t at 4096 + 32 t
/// \param head_in   Head bundle of token t [640] fp32 (layout in rda_state.cc)
/// \param tile_idx  V-dimension tile index (0, 1, 2, or 3)
/// \param t         Token index in this call, 0 <= t < tokens
/// \param tokens    Tokens in this call (1, 2, 4 or 8)
void rda_update_tile_multi(float *__restrict S, float *__restrict packed, const float *__restrict head_in,
                           int32_t tile_idx, int32_t t, int32_t tokens) {
    float *out32 = packed + 4096 + t * 32;
    if ((t & 1) == 0) {
        rda_update_tile(S, packed, head_in, tile_idx, out32);
    } else {
        rda_update_tile(packed, S, head_in, tile_idx, out32);
    }
    if ((tokens & 1) == 0 && t == tokens - 1) {  // the last state is in S: it leaves through packed
        for (int i = 0; i < 4096; i += 16)
            ::aie::store_v(packed + i, ::aie::load_v<16>(S + i));
    }
}

} // extern "C"
