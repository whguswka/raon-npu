# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""깊이 2 FIFO의 보수적 배치 예산. 최종 ELF/map 검증은 별도 필수."""
from dataclasses import dataclass, asdict
from kernels.gemm_prefill.shapes import MAX_BATCH_T, get_shape, validate_batch_t


CORE_DATA_LIMIT_BYTES = 0x10000
CORE_PROGRAM_LIMIT_BYTES = 0x4000
MEMTILE_LIMIT_BYTES = 0x80000
CORE_BD_LIMIT = 16
CORE_LOCK_LIMIT = 16
MEMTILE_BD_LIMIT = 48
MEMTILE_LOCK_LIMIT = 64
SHIM_BD_LIMIT = 16


@dataclass(frozen=True)
class ResourceBudget:
    shape: str
    batch_t: int
    core_w_bytes: int
    core_x_bytes: int
    core_acc_bytes: int
    core_y_bytes: int
    stack_bytes: int
    rtp_alignment_bytes: int
    core_total_bytes: int
    memtile_total_bytes: int
    core_bd_peak: int = 6
    mem_bd_peak: int = 24
    shim_bd_peak: int = 9
    locks_per_core: int = 7
    locks_per_memtile: int = 24
    channel_queue_peak: int = 3
    program_reservation_bytes: int = CORE_PROGRAM_LIMIT_BYTES
    program_limit_bytes: int = CORE_PROGRAM_LIMIT_BYTES

    @property
    def core_fits(self):
        return self.core_total_bytes <= CORE_DATA_LIMIT_BYTES

    @property
    def memtile_fits(self):
        return self.memtile_total_bytes <= MEMTILE_LIMIT_BYTES

    @property
    def limits_ok(self):
        return (self.core_fits and self.memtile_fits
                and self.core_bd_peak <= CORE_BD_LIMIT
                and self.mem_bd_peak <= MEMTILE_BD_LIMIT
                and self.shim_bd_peak <= SHIM_BD_LIMIT
                and self.locks_per_core <= CORE_LOCK_LIMIT
                and self.locks_per_memtile <= MEMTILE_LOCK_LIMIT
                and self.channel_queue_peak <= 4
                and self.program_reservation_bytes <= self.program_limit_bytes)

    def as_dict(self):
        return dict(asdict(self), limits_ok=self.limits_ok,
                    program_measured=False)


def resource_budget(shape, batch_t, *, cols=8):
    s = get_shape(shape)
    validate_batch_t(batch_t, max_batch_t=batch_t)
    if cols not in range(1, 9):
        raise ValueError('cols must be 1..8')
    w = 2 * (9216 if s.format == 0 else 5120)
    x, acc, y = 2 * batch_t * 256 * 2, batch_t * 32 * 4, 2 * batch_t * 32 * 2
    # Full K256 x 16 BF16 scratch (8192 B), plus 4096 B reserve.
    # Q4 retains only one K32 group (1024 B) within its 4096 B stack.
    stack, overhead = (12288 if s.format == 0 else 4096), 8 + 1024
    # Memory tile: W split to the four rows and Y joined from them, both depth 2. X is not staged there - its FIFO
    # broadcasts from the shim straight to the column's four cores (lowered MLIR of the 16-row plate: the memtile
    # buffers are batch_w_*_cons_buff_0/1 and batch_y_*_buff_0/1 only, DESIGN-prefill.md section 45).
    mem = 4 * w + 2 * batch_t * 128 * 2
    # Three live runtime rounds before retiring the oldest; each has X/W/Y.
    return ResourceBudget(s.name, batch_t, w, x, acc, y, stack, overhead,
                          w + x + acc + y + stack + overhead, mem)


def supported_batches(shape, *, cols=8):
    return (MAX_BATCH_T,) if resource_budget(shape, MAX_BATCH_T, cols=cols).limits_ok else ()


def validate_link_usage(*, core_bytes, memtile_bytes, text_bytes, stack_bytes, fmt=0):
    """노트북 링크 맵의 실제 수치를 받아 초과를 거부한다."""
    if fmt not in (0, 1):
        raise ValueError("invalid format")
    limits = (CORE_DATA_LIMIT_BYTES, MEMTILE_LIMIT_BYTES,
              CORE_PROGRAM_LIMIT_BYTES, 12288 if fmt == 0 else 4096)
    values = (core_bytes, memtile_bytes, text_bytes, stack_bytes)
    if any(v < 0 or v > limit for v, limit in zip(values, limits)):
        raise ValueError('linked resource usage exceeds budget')
