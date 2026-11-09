<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon-1-mini MoE 커널 벤치마크 및 테스트 (B1 gate_up & B2 down)

AMD Ryzen AI 7 350 (XDNA 2 AIE2P, 32 코어: 8열 × 4행) NPU를 위한 Raon-1-mini MoE(Mixture-of-Experts) 연산자 벤치마크 스위트입니다.

---

## 1. 아키텍처 및 설계 원리

### 1.1 동적 디스패치(Dynamic Dispatch)를 쓰지 않는 이유 (Codex 분석 요약)

기존 런타임 C++ 동적 명령 합성 흐름(`runtime dispatch`)에서는 드라이버 TDR(NPU 멈춤/hang)이 지속적으로 발생했습니다. Codex 상세 분석(`local analysis notes`)에 따르면 원인은 다음과 같습니다:

1. **열 0의 BD0 수명 관리 오류 (조기 반환 및 교차 채널 재사용)**:
   - 동적 MoE 흐름은 첫 번째 TaskGroup 종료 직후 x 전송의 BD0을 `dma_free_task(%32)`로 즉시 반환합니다.
   - 반환 직후 다음 라운드에서 동일한 BD0을 S2MM 출력 채널(`c0/S2MM0/BD0`)로 다시 프로그래밍합니다.
   - `dma_free_task`는 하드웨어 DMA 완료를 대기하지 않으며, `mlir-aie` 컴파일러 내부에서도 이를 *"아직 실행 중인 BD를 밑에서 재프로그래밍하는 race"*로 규정하고 있습니다.
2. **정적 템플릿의 검증된 안정성**:
   - 반면 정적으로 컴파일된 연산자는 x 전송용 BD0을 전체 연산이 끝날 때까지 유지하며, 열 0의 Y/W 전송은 끝까지 BD1~6만 순환합니다.
   - 동적 경로와 정적 경로의 명령어(9,092 words, TXN ops 1,042개, sync 119회 등)는 동일하며, DDR 주소 패치나 sync 누락 문제는 없었습니다. 유일한 위험 요소는 **열 0 x 전송 BD0의 조기 반납 및 재사용 레이스**였습니다.

### 1.2 정적 틀(Static Template) + DDR_PATCH `arg_plus` 런타임 패치

본 구현은 검증된 정적 컴파일 산출물(BD0 수명 보존, 완벽한 토큰 동기화)을 템플릿으로 고정하고, 런타임에 호출마다 가중치 DMA 전송의 `DDR_PATCH` 주소 오프셋(`arg_plus`)만 동적으로 교체하는 `PatchedTemplateKernel` 아키텍처를 채택했습니다.

- **B1 (moe_gate_up)**:
  - 활성 7개 전문가(라우팅 6개 $e_0 \dots e_5$ + 공유 1개 $e_6=128$)의 gate+up 선형 투영.
  - 형상: 입력 $X$ [2560] (bf16, 32개 코어 브로드캐스트, ObjectFifo depth 1), 출력 $Y$ [15,232] (bf16).
  - 가중치 버퍼: 전체 129개 전문가 연속 버퍼 (`TOTAL_GU_WORDS` = 112,281,600 uint32 words, 449.13 MB).
  - 주소 패치: 119개 W DDR_PATCH (`w_arg_idx = 1`)의 `arg_plus = expert_id * GU_BYTES + l * 0x32000` (바이트).
- **B2 (moe_down)**:
  - 활성 7개 전문가의 down 선형 투영.
  - 형상: 입력 $X$ [7 × 1280] (bf16, 열 0..6 전용 ObjectFifo depth 1), 출력 $Y$ [17,920] (bf16 부분합).
  - 가중치 버퍼: 전체 129개 전문가 연속 버퍼 (`TOTAL_DN_WORDS` = 66,048,000 uint32 words, 264.19 MB).
  - 주소 패치: 140개 W DDR_PATCH (`w_arg_idx = 1`)의 `arg_plus = expert_id * DN_BYTES + r * 0x19000` (바이트).

### 1.3 하드웨어 리소스 예산 분석 (Shim 채널·Core L1·BD)

1. **Shim 채널 예산**:
   - AMD XDNA 2 AIE2P Shim 아키텍처는 컬럼당 MM2S(DDR $\to$ AIE) 최대 2채널, S2MM(AIE $\to$ DDR) 최대 2채널을 지원합니다.
   - **B1 (8 cols)**:
     - 열 0: MM2S 2채널 (MM2S1: $X$ 브로드캐스트, MM2S0: $W$ 스트리밍) $\le 2$, S2MM 1채널 (S2MM0: $Y$) $\le 2$.
     - 열 1~7: MM2S 1채널 (MM2S0: $W$) $\le 2$, S2MM 1채널 (S2MM0: $Y$) $\le 2$.
   - **B2 (7 cols)**:
     - 열 0~6: MM2S 2채널 (MM2S1: $X_j$ 전용 슬라이스, MM2S0: $W$) $\le 2$, S2MM 1채널 (S2MM0: $Y$) $\le 2$.
     - 열 7: 비활성 (0채널).
   - 모든 단계에서 Shim 채널 할당 한도($\le 2$)를 엄격히 준수합니다.

2. **Core L1 데이터 메모리 예산 (64 KiB = 65,536 Bytes Budget)**:
   - AIE2P 코어 타일의 L1 데이터 메모리는 64 KiB로 제한되며, 코어 실행 스택(`STACK_SIZE = 10 * 1024` = 10,240 Bytes)을 반드시 예산 내에 포함해야 합니다.
   - 기반 GEMV 산식(`kernels/gemv/design.py`)을 그대로 적용합니다 (스택은 총량에 **한 번만** 포함): 총 L1 점유 = $2K$ ($X$ 버퍼, depth 1) + $\frac{K}{16}$ (Sums) + $10{,}240$ ($W$ 버퍼, depth 2 ping-pong) + $128$ ($Y$ 버퍼) + $128$ ($Y_{acc}$ 누적기) + $10{,}240$ (실행 스택) = $2.0625 \times K + 20{,}736 \text{ Bytes}$.
   - **B1 (K=2560)**: $2.0625 \times 2560 + 20{,}736 = \mathbf{26{,}016 \text{ Bytes}}$ (실행 스택 10,240 B 포함; $< 65{,}536 \text{ B}$, 예산의 39.7%로 여유 확보).
   - **B2 (K=1280)**: $2.0625 \times 1280 + 20{,}736 = \mathbf{23{,}376 \text{ Bytes}}$ (실행 스택 10,240 B 포함; $< 65{,}536 \text{ B}$, 예산의 35.7%로 여유 확보).

3. **Shim Buffer Descriptors (BD) 예산**:
   - Shim 타일당 하드웨어 BD 한도는 최대 16개입니다.
   - **B1**: 열 0에서 $X$ 전송 BD0 예약 유지 + $W$(depth 2) 2 BDs + $Y$(depth 2) 2 BDs = 최대 5 BDs ($\le 16$). 열 1~7은 4 BDs ($\le 16$).
   - **B2**: 열 0~6에서 $X_j$ 전송 BD0 예약 유지 + $W$(depth 2) 2 BDs + $Y$(depth 2) 2 BDs = 최대 5 BDs ($\le 16$).
   - 모든 Shim 타일에서 최대 5개 BD만 사용하므로 하드웨어 BD 한도(16)를 안전하게 준수합니다.

---

## 2. 벤치마크 단계 (moe_bench.py)

모든 단계는 프로세스 격리(`multiprocessing.Process`) 및 타임아웃(기본 180초, CLI `--timeout 180` 기본값과 일치) 하에 순차 실행됩니다. 한 단계라도 실패하거나 시간 초과 시 다음 단계로 진행하지 않고 즉시 중단됩니다.

| 단계 | 명칭 | 대상 연산자 | 검증 내용 |
|:---|:---|:---|:---|
| **Stage 1** | **B1 Static Template** | `MoEGateUpTemplate` | 주소 패치 없는 순수 정적 insts.bin (전문가 0..6) 비트 대조 (비트 불일치 0 / bit-exact match) |
| **Stage 2** | **B1 Patched Dispatch** | `MoEGateUp` | 5종 패턴(fixed, random, adjacent, boundary, overlap_prev) 정답 대조 및 NPU/Host 레이턴시(평균·p50·p99), 대역폭 실측 |
| **Stage 3** | **B2 Static Template** | `MoEDownTemplate` | 열별 독립 x fifo 정적 insts.bin (전문가 0..6) 비트 대조 (비트 불일치 0 / bit-exact match) |
| **Stage 4** | **B2 Patched Dispatch** | `MoEDown` | 5종 패턴 정답 대조 및 NPU/Host 레이턴시(평균·p50·p99), 대역폭 실측 |
| **Stage 5** | **Stress Test** | `MoEGateUp` + `MoEDown` | 무작위 전문가 1,000회 연속 호출 (B1+B2 파이프라인, 100회마다 둘 다 정답 대조) |

---

## 3. 실측 벤치마크 결과 (노트북 AMD Ryzen AI 7 350 NPU)

> 메인 실측: 2026-10-28 13:54(리뷰 수정 반영판), 노트북(Ryzen AI 7 350, NPU 드라이버 32.0.203.314), `bench/moe/run-moe.ps1`(단계마다 자식 프로세스, 180초 제한, `--iters 30`), 무작위 129 전문가(시드 42). 비트 대조는 정수 표본에서 각 고르기 모양의 실제 선택으로 NPU 출력과 `gemv_bit_reference` 의 bf16 비트 비교. "Host Call" 은 명령 고치기·새 명령 BO 제출·동기화를 포함한 호출 전체 시간, "NPU" 는 NPU 실행 시간(평균·p50·p99). 대역폭은 평균 기준, B1 24.4 MB·B2 14.3 MB. 측정 당시 원본 로그를 기준으로 정리했습니다.

### 3.1 Stage 1 & Stage 3: 정적 템플릿 베이스라인

| 연산자 | 단계 | 출력 형상 | Mismatches | Max ULP | Call Time (ms) | 판정 |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| `MoEGateUpTemplate` | Stage 1 | [15,232] | 0 | 0 | 3.91 (첫 호출, 적재 포함) | PASS |
| `MoEDownTemplate` | Stage 3 | [17,920] | 0 | 0 | 3.12 (첫 호출, 적재 포함) | PASS |

### 3.2 Stage 2: B1 (Gate-Up) 패치 디스패치 (5개 패턴)

| 패턴(실제 선택) | Mismatches | Host Call 평균/p50/p99 (ms) | NPU 평균/p50/p99 (ms) | NPU 대역폭 (GB/s) | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `fixed` [0..5] | 0 | 0.79 / 0.78 / 1.01 | 0.50 / 0.50 / 0.55 | 48.6 | PASS |
| `random` [8,89,24,60,104,79] | 0 | 0.84 / 0.79 / 1.65 | 0.53 / 0.52 / 0.80 | 45.6 | PASS |
| `adjacent` [10..15] | 0 | 0.76 / 0.77 / 0.87 | 0.49 / 0.49 / 0.52 | 49.9 | PASS |
| `boundary` [122..127] | 0 | 0.83 / 0.77 / 1.73 | 0.54 / 0.50 / 0.99 | 45.1 | PASS |
| `overlap_prev` [0,1,2,24,80,6] (직전과 3개 겹침) | 0 | 0.88 / 0.79 / 1.67 | 0.56 / 0.51 / 1.15 | 43.3 | PASS |

### 3.3 Stage 4: B2 (Down) 패치 디스패치 (5개 패턴)

| 패턴(실제 선택) | Mismatches | Host Call 평균/p50/p99 (ms) | NPU 평균/p50/p99 (ms) | NPU 대역폭 (GB/s) | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `fixed` [0..5] | 0 | 0.69 / 0.67 / 1.19 | 0.41 / 0.37 / 0.89 | 35.3 | PASS |
| `random` [53,2,42,4,119,56] | 0 | 0.71 / 0.67 / 1.40 | 0.40 / 0.38 / 0.77 | 35.6 | PASS |
| `adjacent` [10..15] | 0 | 0.67 / 0.65 / 0.96 | 0.37 / 0.37 / 0.39 | 38.8 | PASS |
| `boundary` [122..127] | 0 | 0.65 / 0.64 / 0.79 | 0.37 / 0.37 / 0.42 | 38.7 | PASS |
| `overlap_prev` [0,1,2,33,43,58] (직전과 3개 겹침) | 0 | 0.81 / 0.71 / 2.14 | 0.45 / 0.39 / 0.93 | 32.1 | PASS |

### 3.4 Stage 5: 스트레스 테스트 (1,000회 연속 무작위 호출, B1+B2)

| 항목 | 측정값 | 비고 |
|:---|:---|:---|
| 총 반복 횟수 | 1,000 | 반복마다 같은 무작위 선택으로 B1·B2 를 차례로 호출 |
| 검증 주기 | 100회마다 | B1·B2 모두 정답 대조, 10회 모두 통과 |
| 총 소요 시간 | 8.71 s | |
| 반복당 시간 | B1 3.32 ms · B2 2.91 ms · 합계 6.23 ms | 단계 2·4 의 호출 전체(약 0.8·0.7 ms)보다 크다 — 반복 루프의 버퍼 동기화·읽기 등 측정 부대 비용으로 추정, 엔진 조립 때 분석 |
| 최종 판정 | PASS | 멈춤·드라이버 재설정 없음 |

호출 전체와 NPU 시간의 차이(B1 약 0.3 ms, B2 약 0.3 ms)는 호출마다 명령 BO 를 새로 만들어 보내는 비용이다 — 명령 BO 재사용이 다음 최적화 대상.

---

## 4. 실행 방법

### 4.1 변환용 Linux 기계 (NumPy 전용 단위 테스트)
```bash
python3 -m pytest -q bench quant -p no:cacheprovider
```

### 4.2 노트북 (AMD Ryzen AI 7 350 NPU 실측)

PowerShell 콘솔에서 `run-moe.ps1`을 실행합니다 (내부적으로 VS 빌드 도구 및 ironenv 환경을 활성화하는 `C:\dev\raon\run.cmd`를 통해 실행):
```powershell
cd C:\dev\raon\moe
.\bench\moe\run-moe.ps1
```

모의 테스트 (NPU 없이 구조 및 디스패치 로직 검증):
```powershell
python bench/moe/moe_bench.py --mock
```
