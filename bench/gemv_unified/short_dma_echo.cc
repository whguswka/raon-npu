// Copyright (C) 2026 raon-npu authors
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

extern "C" void short_dma_echo(const uint32_t *__restrict src,
                               uint32_t *__restrict dst, int32_t words) {
  for (int32_t i = 0; i < words; ++i)
    dst[i] = src[i];
}
