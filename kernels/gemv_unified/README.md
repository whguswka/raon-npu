<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Unified GEMV (`bfp16ebs8` + `q4nx`)

9비트 GEMV 10개 모양과 단계 2의 `q4nx dense_gu`, MoE B1-6, 9비트
B2-9 명령 흐름을 하나의 configuration-only `xclbin`으로 묶습니다. C++
object에는 두 산술 계열이 함께 링크되고 코어는 RTP `format`으로 분기합니다.

---

## 1. 아키텍처 및 고정 장치 구성 (8열 × 4행 = 32코어)

장치 본체(`Tile`, `ObjectFifo`, `Buffer`, `Worker`, `core_fn`)는 `(K, M)` 모양에 무관하게 고정됩니다.  단, 열 개수는 `cols` 매개변수(기본값 `8`)로 조절할 수 있어, 동일한 설계 코드로 7열 변형을 생성할 수 있습니다.  `cols=8` 일 때는 기존 xclbin/insts.bin 과 **바이트 동일**하게 생성되며, 7열 빌드는 실험/검증용으로 사용합니다.

- **열별 `x` ObjectFifo (`x0..x7`)**:
  - 소비측(L1) depth `1`, 크기 `MAX_K = 12,032` `bf16` (`24,064 B`).
  - 공유 xclbin의 수신 BD도 이 고정 크기를 기다리므로, 호스트 X BO를 `MAX_K`까지 0으로 패딩하고 매 호출 `MAX_K * 2` 바이트를 전송합니다. 실제 계산은 `k_blocks = K // 256` 블록까지만 읽습니다. 실제 `K`만 보내면 Shim BD만 끝나고 수신 BD/lock이 완료되지 않아 첫 `x_in.acquire(1)`에서 교착합니다.
- **열별 저수준 `W` 경로**:
  - 명시적 `Flow`로 shim→Memory Tile→4개 코어를 연결하고, Memory Tile과 코어에 고정 크기 ping/pong `Buffer` 및 슬롯별 `Lock`을 둡니다.
  - 명령 흐름의 runtime DMA task가 수신 BD를 9비트는 `9,216 B`, q4nx는 `5,120 B`로 구성합니다. 코어 L1 할당은 `9,216 B × 2 = 18,432 B`로 고정입니다.
  - Memory Tile·코어의 W task는 completion token을 내지 않습니다(`issue_token=False`; `routing.py`의 `W_TASKS_ISSUE_TOKEN`). 비-shim 타일이 token을 내면 `aie-generate-column-control-overlay`가 TileControl→shim South 아래 방향 packet flow를 추가해 Memory Tile North source(한도 4)가 y join 4 + 1 = 5로 넘칩니다. token은 shim 작업(x·W 입력·Y drain)에서만 나며, W task BD는 그 task가 맡은 마지막 라운드의 Y drain await 뒤 `free()`만 합니다. 기본 모드(`continuous_mem_w=False`)에서는 Memory Tile W task도 라운드마다 ping부터 다시 시작합니다. 선택적 연속 모드(`continuous_mem_w=True`, xclbin 접미사 `_contw`)에서는 Memory Tile W task가 라운드를 넘어 이어지며(`w_plan.py`: slot = 누적 블록 번호 % 2, 255 pair 단위로 분할, 홀수 총 블록이면 끝에 1-BD tail), 코어 W task는 어떤 모드에서든 라운드마다 ping부터 시작합니다. 따라서 다음 라운드/호출은 ping과 초기 lock count에서 시작합니다(Y drain 완료가 그 라운드 모든 W task 완료를 보장).
- **열별 `y` ObjectFifo (`y0..y7`)**:
  - Memory Tile에서 4행 `join`, 코어당 `32` `bf16`(`64 B`) × depth `2` = `128 B`.
- **코어 로컬 버퍼**:
  - `y_acc`: `32 × float32` (`128 B`)
  - `rtp`: `3 × int32` (`12 B`, `use_write_rtp=True`)
  - `q4 group_sums`: `MAX_K / 32` bf16 (`752 B`)
  - `WorkerRuntimeBarrier`: 호스트 RTP 기록 완료 후 코어가 RTP를 읽도록 동기화하고 즉시 `release_with_value(1)`로 다음 호출을 위해 잠금을 재설정합니다.
- **동적 ObjectFifo Lowering**:
  - `Worker(..., stack_size=10240, dynamic_objfifo_lowering=True)` 및 `XclbinArtifact` / `InstsBinArtifact`의 `extra_flags=["--dynamic-objFifos"]`를 사용합니다.

---

## 2. L1 메모리 예산 (코어당 64 KiB = 65,536 B 한도)

| 항목 | 크기 (Bytes) | 비고 |
|---|---:|---|
| `x` 소비 버퍼 (`12,032` bf16 × depth 1) | 24,064 | 모든 모양의 최대 `K=12,032` 수용 |
| `W` 소비 버퍼 (`9,216 B` × depth 2) | 18,432 | `32 × 256` 9-bit 블록 ping/pong |
| `y` 생산 버퍼 (`64 B` × depth 2) | 128 | `32` bf16 출력 타일 ping/pong |
| `y_acc` (`32 × float32`) | 128 | `accfloat` 누산 버퍼 |
| q4 group sums (`12,032 / 32 × bf16`) | 752 | 최대 K 기준 |
| `rtp` (`3 × int32`) | 12 | `format`, `k_blocks`, `rounds` |
| **데이터 버퍼 소계** | **43,516** | **42.50 KiB** |
| 코어 스택 (`STACK_SIZE`) | 10,240 | 10.00 KiB |
| **데이터 + 스택 합계** | **53,756** | **52.50 KiB (여유 11,780 B)** |

### 노트북 빌드 후 `.data/.bss` 및 스택 실측 명령

```powershell
llvm-size build\**\*.elf
```
- 확인 기준: `.data + .bss + 10,240 + 43,516 <= 65,536 B` (여유 `>= 4 KiB`).

---

## 3. 코어 RTP 계약 (`RTP_WORDS = 3`)

각 호출(`sequence`) 시작 시 호스트 명령 흐름은 32개 코어의 RTP를 모두 기록한 뒤 32개 코어의 `WorkerRuntimeBarrier`를 `set(1)`로 해제합니다:

| 인덱스 | 상수명 | 값 (1단계) | 설명 |
|---:|---|---|---|
| `rtp[0]` | `RTP_FORMAT` | `0` 또는 `1` | `0 = bfp16ebs8`, `1 = q4nx` |
| `rtp[1]` | `RTP_K_BLOCKS` | `K // 256` (`1..47`) | K축 256-원소 블록 반복 횟수 |
| `rtp[2]` | `RTP_ROUNDS` | `rounds_by_col[col]` | 해당 열에 배정된 128-행 출력 덩어리(chunk) 수 |

---

## 4. 1단계 지원 10개 모양 및 열별 라운드 표

128-행 출력 덩어리 `c` (`0 <= c < M // 128`)는 열 `c % 8`, 라운드 `c // 8`에 배정됩니다.

| 모양 이름 | K | M | `k_blocks` | `chunks` (`M/128`) | 열별 `rounds` (`col 0..7`) | W 크기 (Words) |
|---|---:|---:|---:|---:|---|---:|
| `rda_in` | 2,560 | 20,736 | 10 | 162 | `(21, 21, 20, 20, 20, 20, 20, 20)` | 14,929,920 |
| `f_b` | 256 | 5,120 | 1 | 40 | `(5, 5, 5, 5, 5, 5, 5, 5)` | 368,640 |
| `o_proj` | 5,120 | 2,560 | 20 | 20 | `(3, 3, 3, 3, 2, 2, 2, 2)` | 3,686,400 |
| `mla_in` | 2,560 | 1,152 | 10 | 9 | `(2, 1, 1, 1, 1, 1, 1, 1)` | 829,440 |
| `q_b` | 768 | 7,680 | 3 | 60 | `(8, 8, 8, 8, 7, 7, 7, 7)` | 1,658,880 |
| `k_b_v_b` | 512 | 5,120 | 2 | 40 | `(5, 5, 5, 5, 5, 5, 5, 5)` | 737,280 |
| `g` | 2,560 | 5,120 | 10 | 40 | `(5, 5, 5, 5, 5, 5, 5, 5)` | 3,686,400 |
| `shared_gu` | 2,560 | 2,176 | 10 | 17 | `(3, 2, 2, 2, 2, 2, 2, 2)` | 1,566,720 |
| `dense_down` | 12,032 | 2,560 | 47 | 20 | `(3, 3, 3, 3, 2, 2, 2, 2)` | 8,663,040 |
| `lm_head` | 2,560 | 260,096 | 10 | 2,032 | `(254, 254, 254, 254, 254, 254, 254, 254)` | 187,269,120 |

---

## 5. 공유 `xclbin` 및 3-BO ABI 검증 (`op.py`)

- **Configuration-only `xclbin`**: `Raon_GEMV_Unified_c8_npu2.xclbin` 1개만 생성합니다.
- **형식/모양별 `insts.bin`**: `Raon_GEMV_Unified_c8_npu2_f{format}_k{K}_m{M}_a{active_cols}_x{x_slots}_c{chunks_per_col|auto}.bin`을 생성하며, 형식별 W 수신 BD 길이는 이 명령 바이너리에 들어갑니다.
- **3-BO ABI**: `(Y: out bf16 [M], W: in uint32 [weight_words], X: in bf16 [...])`입니다. B2의 X는 `[x_slots * MAX_K]`(7칸), 나머지는 `[MAX_K]`이며 각 칸의 `K..MAX_K` 구간을 0 패딩합니다. `diagnostic_small_x`에서는 X가 `[x_slots * K]`입니다.
- `UnifiedGEMV.get_callable()` 및 `assert_shared_xclbin()`은 시작 시 모든 모양 callable의 해결된 `xclbin` 경로(`Path.resolve()`), `st_mtime_ns`, `kernel_name == "MLIR_AIE"`, 3-BO ABI가 완전히 일치하는지 `assert`합니다.

### 엔진 설정: `continuous_mem_w=True`

`npu/backend.py`의 통합 GEMV 엔진은 `UNIFIED_CONTINUOUS_MEM_W = True`로 동작합니다. 이 설정을 켜면 xclbin 이름에 `_contw` 접미사가 붙고 Memory Tile의 W DMA 작업이 라운드를 넘어 이어집니다. 노트북 실측(10-31)에서 동일 25.8 MB 바이트를 가진 `K=1280, M=17920` 모양은 기본 1.879 ms에서 `contw` 0.940 ms로 줄었으며, 라운드 수가 많은 MoE B2(`moe_b2_9`, K=1280, 열당 20라운드)에서 이득이 큽니다. 공식 골든 2회 전체 시험은 통과했고, 3~16단계 지연은 319.5 ms/토큰에서 276.2 ms/토큰으로 줄었으며 골든 수치, 단계별 logit cos, NPU 1위 토큰은 동일하게 유지되었습니다. MoE B2는 50.0→27.6, B1은 25.9→20.1, `rda_in`은 25.3→21.5 ms/토큰이었습니다.

모든 통합 연산(dense 모양, `shared_gu`, MoE B1/B2 템플릿)은 반드시 같은 `_contw` xclbin을 사용해야 합니다. 서로 다른 `_contw` xclbin이 섞이면 decode 중 추가 xclbin 전환이 발생해 측정된 가속 효과가 사라집니다. `npu/backend.py`의 `_require_single_unified_xclbin()`이 모든 통합 연산의 해결된 xclbin 경로를 `Path.resolve()`로 비교하여 경로가 다르면 `RuntimeError`를 냅니다.

### 런타임 명령열 계획: `per_round` / `per_call`

`per_round`는 af44167 운영 기준 명령열을 그대로 보존합니다. `per_call`은 장치
본체와 xclbin 이름을 바꾸지 않고, 열별 shim W/Y 작업과 코어 W 수신 사슬을
호출 단위로 합칩니다. 짝수 K는 2-BD ping/pong 사슬, 16블록 이하 홀수 K는
`P,Q,P,...` 위치별 BD 사슬 전체를 열 라운드 수만큼 반복하므로 매 라운드가
항상 ping에서 시작합니다. 반복 수는 255 이하로, shim iteration은 64 이하로
분할합니다.

- **per_call 적용 모양 (`UNIFIED_PER_CALL_SHAPES`)**:
  `f_b`, `q_b`, `k_b_v_b`, `dense_gu`, `moe_b1_6` (B1), `moe_b2_9` (B2) 총 6개 모양입니다.
- **per_round 유지 모양 및 사유**:
  - `dense_down`: `K_blocks=47` (홀수). 코어 47-BD 사슬은 하드웨어 코어 BD 풀 한도(16개)를 초과하므로 per_round에 남습니다 (`test_dense_down_is_not_a_legal_per_call_core_chain`).
  - `lm_head`: 대형 출력 차원(`M=260,096`, `chunks=2,032`, 7열 엔진에서 열당 290~291 라운드). DDR 대역폭 바운드(DDR-bound)이며 Shim DMA iteration 하드웨어 한도(≤ 64) 분할 오버헤드 및 코어 BD 한도로 인해 per_round로 유지합니다. 위 1단계 표의 열당 254 라운드는 8열 배치 값입니다.
  - `rda_in`, `mla_in`, `o_proj`, `g`: DDR 대역폭 바운드 또는 소형/고정 라운드 특성 등으로 인해 baseline인 per_round를 유지합니다.
- **단일 되돌리기 스위치**: `UNIFIED_SEQ_PLAN = "per_round"`로 설정하면 모양 허용 목록과 무관하게 모든 연산자가 af44167 바이트 동일 기준 계획으로 즉시 되돌아갑니다. per-call insts에는 `_pc` 접미사가 붙고 xclbin 이름은 동일합니다. B1의 6열×17덩어리 배치는 per-call에서만 파생하며, 공용 `STAGE2_SHAPES`의 7열 기준 모양은 유지합니다.

#### 노트북 실측 수치 (2026-11-01)

- **모양별 실측 A/B (`lap/plan_ab.py`, 200회, 비트 일치·canary 정상)**:

| 모양 (엔진 이름) | K / M | p50 레이턴시 (per_round → per_call) | TXN 연산 수 (per_round → per_call) | DDR_PATCH (per_round → per_call) | 비고 |
|---|---|---:|---:|---:|---|
| `f_b` | 256 / 5,120 | 320 → 231 µs (-27.8%) | 1,198 → 478 (-60.1%) | - | K_blocks=1 (홀수) |
| `q_b` | 768 / 7,680 | 541 → 344 µs (-36.4%) | 3,168 → 620 (-80.4%) | - | K_blocks=3 (홀수) |
| `k_b_v_b` | 512 / 5,120 | 346 → 241 µs (-30.3%) | 1,488 → 504 (-66.1%) | - | K_blocks=2 (짝수) |
| `moe_b2_9` (B2) | 1,280 / 17,920 | 1,133 → 739 µs (-34.8%) | 7,028 → 672 (-90.4%) | 287 → 21 (-92.7%) | 7-slot down |
| `moe_b1_6` (B1) | 2,560 / 13,056 | 782 → 681 µs (-12.9%) | 3,534 → 453 (-87.2%) | 211 → 19 (-91.0%) | 6-expert gate/up |
| `dense_gu` | 2,560 / 24,064 | 1,051 → 937 µs (-10.8%) | 6,372 → 504 (-92.1%) | - | K_blocks=10 (짝수), 188 chunks |

- **엔진 공식 골든 (운영 판 af44167 대 per_call 트리, A·B 번갈아, 3~16단계 decode)**:
  - 전체 토큰 레이턴시: 207.9·205.8 → 192.3·191.0 ms/토큰
  - 순수 NPU 연산 시간: 156 → 140 ms
  - 16단계 logits 및 중간값 해시: 전부 동일.
- **산출물 비교 (`plan_artifact_check --compile-insts`)**:
  - `per_round` 13모양 + RDA의 커널 객체/심볼 8자리 접미사를 정규화한 MLIR, `_make_extern` 공개 입력 지문, `insts.bin` SHA-256을 기준 트리와 비교합니다. 저장소 안 include 폴더 지문은 C/C++ 소스·헤더(`.h`, `.hh`, `.hpp`, `.hxx`, `.inc`, `.ipp`, `.c`, `.cc`, `.cpp`, `.cxx`, 대소문자 무시)만 포함하며, 숨김 폴더와 `__pycache__` 하위는 제외합니다. MLIR의 device/아키텍처 줄은 정규화하지 않으므로 아키텍처 차이도 판정에 포함됩니다.
  - 현재 트리의 허용 목록 6모양에는 `per_call` 레코드가 반드시 있어야 하며, 오류가 없고 이름이 per-round 이름과 다르면서 `_pc`로 끝나야 합니다. `call_device_equal`은 현재 트리의 per-round/per-call 정규화 장치 본체 비교, `call_name_ok`는 이 이름 계약 판정입니다. per-round 전용 모양은 두 칸에 `-`를 출력합니다.
  - `round_equal`은 기준/현재 per-round 정규화 MLIR과 이름의 동시 일치, `round_name_equal`은 이름만의 일치입니다. `raw_sha_diff`, `norm_sha_equal`, `insts_sha_equal`은 각각 원시 MLIR 차이, 정규화 MLIR 일치, `--compile-insts` 사용 시 per-round 명령 바이너리 일치를 뜻합니다. 이 검사는 컴파일된 `.o`와 xclbin 바이트 동일성은 보장하지 않습니다.

### 5c 합친 설정: 7열 GEMV + 열 7 RDA 단일 xclbin 구동

Unified 엔진은 7열 GEMV(`cols=7`)와 열 7 RDA(`rda_col=7`)가 하나로 결합된 단일 xclbin 설정(`Raon_GEMV_Unified_c7_rda7_npu2_contw.xclbin`)으로 동작합니다.

- **전환 횟수 0**: RDA 순환 상태 갱신이 통합 xclbin 내에서 실행되므로 xclbin 전환 횟수가 토큰당 0회(기존 분리 설정에서는 토큰당 36회)로 감소합니다.
- **노트북 관문 검증 결과**:
  - 합친 설정에서 장치 본체(`Tile`, `ObjectFifo`, `Buffer`, `Lock`, `Flow`, `Worker`)가 설정(configuration), RDA, 모양 명령열 사이에서 비트 단위로 동일함을 확인하였습니다.
  - 모든 dense 모양(dense_gu 등) 출력 비트가 일치함을 확인하였습니다.
  - RDA 연산 결과가 기존 4열 별도 xclbin인 `rda_step`과 3개 시드 × 200토큰에 걸쳐 uint32 완전 일치함을 확인하였습니다.
  - GEMV와 RDA의 교대 실행 시 추가 지연(alternation overhead) p50 ≈ 0 ms를 달성하였습니다 (기존 분리 엔진은 쌍마다 약 2.3 ms 발생).
- **레거시(Legacy) 엔진 경로 보존**: `legacy` 엔진은 기존 개별 GEMV xclbin(`gemv_{fmt}_k{k}_m{m}`), MoE 템플릿, 그리고 4열 별도 `rda_step` xclbin 경로(`_prepare_rda_kernel`)를 그대로 유지하며 전환 횟수도 기존대로 계측합니다.

---

## 6. 실행 방법

### 진단·정확도 관문의 판정 범위

- `diagnose_stall.py`는 각 케이스를 별도 프로세스와 제한 시간으로 실행합니다.
  여기서 PASS는 **0 가중치와 0 입력의 호출이 제한 시간 안에 끝나고 출력 비트가
  모두 0**이라는 뜻입니다. 교착·멈춤을 좁히는 진단이며 일반 수치 정확도 판정은
  아닙니다.
- `diagnose.py`는 비영(nonzero) 결정론 입력을 독립 참조 구현과 비교하여
  `mismatches=0`, `max_ulp=0`인 비트 정확도를 판정합니다.
- `unified_bench.py`는 지원 9비트 10모양의 smoke, 두 모양 교대, 전체 모양,
  장시간 무작위 교대 호출에서 비트 정확도와 공유 xclbin 계약을 판정합니다.
- `stage2_bench.py`는 짧은 DMA 선행 관문 뒤 q4nx dense/B1, 9비트 B2 및 혼합
  반복의 참조 정확도와 공유 xclbin 계약을 판정합니다.

`unified_bench.py`는 개인 IRON 경로를 내장하지 않습니다. 별도 경로가 필요하면
`RAON_IRON_DIR` 환경 변수를 설정할 수 있으며, 설정하지 않으면 `sys.path`를
변경하지 않습니다. `run-unified.ps1`은 기존처럼 `PYTHONPATH`에 저장소와
`C:\dev\IRON`을 넣으므로 노트북 실행 동작은 유지됩니다.

### 변환용 Linux 기계 단위 테스트 (numpy + mock)
```bash
python3 -m pytest -q bench quant kernels -p no:cacheprovider
```

### 노트북 NPU 단계별 실행 (`C:\dev\raon\gemv-unified`)

저장소 루트에서 파일 실행과 모듈 실행을 모두 지원합니다. 먼저 전체 통합
설계를 컴파일해 routing을 확인하고, 성공한 뒤에만 NPU 단계를 실행합니다.

```powershell
# 파일 실행
python bench\gemv_unified\stage2_bench.py --compile-only
python bench\gemv_unified\unified_bench.py --compile-only

# 같은 명령의 모듈 실행형
python -m bench.gemv_unified.stage2_bench --compile-only
python -m bench.gemv_unified.unified_bench --compile-only
```

```powershell
# 1단계: 단일 모양(o_proj) 비트 대조 및 NPU 시간
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage smoke

# 2단계: 두 모양(o_proj <-> f_b) 교대 호출 검증
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage alt2

# 3단계: 10모양 전체 비트 대조 및 1,000회 p50 측정 (+ 정적 GEMV_BFP16 비교 시 -CompareStatic)
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage all10 -CompareStatic

# 4단계: 무작위 모양 교대 10,000회 스트레스 시험 (1,000회마다 비트 대조)
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage stress

# 단계 0 실험 2: 저수준 W 경로의 9,216/5,120 B 수신 BD를 10,000회 교대 시험
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage exp_short_dma

# 단계 2: q4 smoke → B1 5패턴 → B2 5패턴 → 혼합 10,000회
powershell -ExecutionPolicy Bypass -File bench\gemv_unified\run-unified.ps1 -Stage stage2
```

`stage2_bench.py`는 q4/B1/B2/혼합 단계를 시작하기 전에 단계 0을
첫 full+short 호출, 짧은 반복, 10,000회 soak의 별도 watchdog으로 실행합니다.
어느 단계든 실패하거나 timeout이면 이후 단계는 시작하지 않습니다.

### 노트북 직접 관문 명령

```powershell
python -m bench.gemv_unified.check_device_invariance
python -m bench.gemv_unified.check_device_invariance --continuous
python -m bench.gemv_unified.plan_artifact_check --baseline C:\dev\raon-af44167 --compile-insts
python -m bench.gemv_unified.plan_ab_bench --shape f_b --weights C:\dev\raon-weights --iterations 400 --timeout 300
python -m bench.gemv_unified.diagnose_stall --k 512 --timeout 120 --json diag.json
python -m bench.gemv_unified.unified_bench
python -m bench.gemv_unified.stage2_bench --stage all --timeout 900
```

불변성 명령의 기본 집합은 등록된 `ALL_SHAPES` 전체와 홀수·짝수 K 블록,
`MAX_K`, 1열 `M=128`, 8열 `M=5120` 경계 모양입니다. 등록 모양만 고를 때는
`--names dense_gu,moe_b1_6,moe_b2_9`, 임의 진단 모양은 기존처럼
`--shapes K:M[:cols]`를 사용합니다.
