# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""합성 자료용 NumPy 모델. 흉내 BFP 양자화의 하드웨어 모델은 아님."""
import numpy as np
from quant.q4nx import round_to_bf16_rne, to_bf16_u16
from kernels.gemv.reference import gemv_bit_reference


def unpack_bfp16(block):
    raw = np.frombuffer(bytes(block), np.uint8)
    if raw.size != 9216:
        raise ValueError('one 9216-byte block required')
    raw = raw.reshape(2,32,144)
    exp = raw[:,:,:16].astype(np.int32)
    scale_bits = ((np.maximum(exp,7)-6) << 7).astype(np.uint16)
    scales = (scale_bits.astype(np.uint32) << 16).view(np.float32)
    mant = raw[:,:,16:].copy().view(np.int8).reshape(2,32,8,16)
    w = mant.astype(np.float32) * scales[:,:,None,:]
    return round_to_bf16_rne(w.transpose(0,3,1,2).reshape(32,256))


def mmul_reference(weights, x):
    w, x = np.asarray(weights,np.float32), round_to_bf16_rne(x)
    if w.ndim != 2 or x.ndim != 2 or w.shape[1] != x.shape[1] or w.shape[1] % 8:
        raise ValueError('matching K divisible by 8 required')
    c = np.zeros((len(x),len(w)),np.float32)
    # NumPy matmul may select a different reduction tree.  This explicit K
    # order is the decode-GEMV bit oracle used for local regression tests.
    for k in range(w.shape[1]):
        c = (c + x[:, k, None] * w[None, :, k]).astype(np.float32)
    return round_to_bf16_rne(c)


def q4nx_mmul_reference(codes, scales, mins, x):
    codes = np.asarray(codes,np.float32)
    x = round_to_bf16_rne(x)
    scales, mins = round_to_bf16_rne(scales), round_to_bf16_rne(mins)
    c = np.zeros((len(x),len(codes)),np.float32)
    for g in range(codes.shape[1]//32):
        dot = np.zeros_like(c)
        for k in range(g*32,(g+1)*32,8):
            dot += x[:,k:k+8] @ codes[:,k:k+8].T
        dot = round_to_bf16_rne(dot)
        sums = round_to_bf16_rne(x[:,g*32:(g+1)*32].astype(np.float64).sum(axis=1))
        c += dot * scales[:,g]
        c += sums[:,None] * mins[:,g]
    return round_to_bf16_rne(c)


def bfp16_batch_reference(weights, x):
    rows = np.asarray(x, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"x must be [T,K], got {rows.shape}")
    return to_bf16_u16(mmul_reference(weights, rows))


def q4nx_batch_reference(codes, scales, mins, x):
    rows = np.asarray(x, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"x must be [T,K], got {rows.shape}")
    return np.stack([gemv_bit_reference(codes, scales, mins, row)[1] for row in rows])


def dma_x(x, max_t):
    """호스트 [T,K]에서 코어 [K256,K8,max_T,8]로 가는 TAP 모델."""
    x = round_to_bf16_rne(x)
    padded = np.zeros((max_t, x.shape[1]), np.float32)
    padded[:len(x)] = x
    return padded.reshape(max_t, -1, 32, 8).transpose(1, 2, 0, 3).copy()


def pack_b(weights):
    """16행 W의 연속 load_v<64> 배치: [K8,side8,K-lane,row8]."""
    w = np.asarray(weights, np.float32)
    return w.reshape(2, 8, -1, 8).transpose(2, 0, 3, 1).copy()


def blocked_mmul_reference(weights, x, max_t, r=4):
    """새 DMA/B/C 주소식을 직접 사용하는 합성 모델 (실제 AIE 누산 트리 아님)."""
    w = np.asarray(weights, np.float32)
    xs = dma_x(x, max_t)
    c = np.zeros((2, max_t // r, 2, r, 8), np.float32)
    for kb in range(xs.shape[0]):
        for h in range(2):
            bs = pack_b(w[h*16:(h+1)*16, kb*256:(kb+1)*256])
            for t in range(0, len(x), 2*r):
                # Copy represents four live accumulators, committed once per K256.
                live = c[h, t//r:t//r+2].copy()
                for k8 in range(32):
                    for a in range(2):
                        for b in range(2):
                            live[a, b] += xs[kb,k8,t+a*r:t+(a+1)*r] @ bs[k8,b]
                c[h, t//r:t//r+2] = live
    return round_to_bf16_rne(c.transpose(1,3,0,2,4).reshape(max_t,32))


def blocked_q4_reference(codes, scales, mins, x, max_t, r=4):
    """Q4 새 DMA/B 배치에서 그룹 dot을 유지한 뒤 벡터 끝맺음."""
    xs = dma_x(x, max_t)
    c = np.zeros((max_t,32), np.float32)
    for kb in range(xs.shape[0]):
        for h in range(2):
            for g in range(8):
                group = kb*8+g
                bs = pack_b(codes[h*16:(h+1)*16,group*32:(group+1)*32])
                for t in range(0,len(x),2*r):
                    dot = np.zeros((2*r,16),np.float32)
                    for k8 in range(4):
                        a = xs[kb,g*4+k8,t:t+2*r]
                        dot[:,:8] += a @ bs[k8,0]
                        dot[:,8:] += a @ bs[k8,1]
                    values=xs[kb,g*4:g*4+4,t:t+2*r].transpose(1,0,2).reshape(2*r,32)
                    sums=round_to_bf16_rne(values.astype(np.float64).sum(axis=1))
                    dst=c[t:t+2*r,h*16:(h+1)*16]
                    dst += round_to_bf16_rne(dot)*scales[h*16:(h+1)*16,group]
                    dst += sums[:,None]*mins[h*16:(h+1)*16,group]
    return round_to_bf16_rne(c)
