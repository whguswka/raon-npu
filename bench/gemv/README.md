<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon-1-mini 일반 q4nx GEMV 커널 및 NPU 벤치마크 명세

본 디렉토리는 AMD XDNA 2 AIE2P (Ryzen AI 7 350, 32 코어: 8열 × 4행) 상에서 KRAFTON Raon-1-mini 언어 모델의 투영 행렬(RDA q/k/v/g, MLA q_a/kv_a/q_b, o_proj, Expert gate/up/down, Dense MLP 등)을 수행하는 일반 q4nx GEMV(Matrix-Vector Multiplication)의 NPU 실행기 및 명세 문서를 담고 있습니다.

이미 노트북에서 정확도와 성능(6.50 ms, 약 64 GB/s)이 검증된 `lm_head` 설계를 바탕으로 RMSNorm과 Softcap epilogue를 제거하고, 일반 K 및 M 형상(M % 128 == 0, K % 256 == 0)을 지원하도록 확장되었습니다.

---

## 1. 노트북 실행 방법

노트북 AMD Ryzen AI 환경(XDNA 2 NPU)에서 실행합니다.
모든 모양(shape)은 **개별 독립 자식 프로세스**와 **모양당 300초 타임아웃**으로 격리되어 있어, 특정 형상에서 NPU 데드락이나 시간 초과가 발생하더라도 자식 프로세스를 강제 종료하여 NPU 리소스를 즉시 회수하고 다음 모양의 벤치마크를 계속 진행합니다.

### 1) PowerShell 스크립트를 통한 실행 (권장)
```powershell
bench\gemv\run-gemv.ps1
```

### 2) 직접 커맨드라인 실행
```powershell
Set-Location C:\dev\raon\gemv
$env:PYTHONPATH = 'C:\dev\raon\gemv;C:\dev\IRON'
cmd /c "C:\dev\raon\run.cmd -u bench\gemv\gemv_bench.py --preset raon --warmup 10 --iters 100 --timeout 300 --json-out C:\dev\raon\gemv\last-run.json > C:\dev\raon\gemv\last-run.log 2>&1"
```

### 3) 임의 형상 단독 실행
```powershell
cmd /c "C:\dev\raon\run.cmd -u bench\gemv\gemv_bench.py --shape 2560,20480 --cols 8 --timeout 300"
```

---

## 2. AIE2P 코어당 L1 메모리 예산 분석 (64 KiB Budget)

AMD XDNA 2 코어 타일의 L1 데이터 메모리는 **64 KiB (65,536 Bytes)** 로 엄격히 제한되어 있습니다.
본 설계는 입력 벡터 `x`를 `depth=1`로 유지하여(한 번 수신 후 전체 라운드가 끝날 때까지 acquire 유지) L1 공간을 극대화하고, K가 큰 모델 레이어(예: Dense MLP down-projection `K=12,032`)에서도 L1 한도 내에서 안전하게 동작하도록 최적화되었습니다.

### 1) 코어당 할당 요소별 크기 산정식
- **`w` 버퍼 (`w_cores[i]`)**:
  - 1개 q4nx 블록: $M_{tile} \times K_{tile} \times 5 / 8 = 32 \times 256 \times 5 / 8 = 5,120 \text{ Bytes}$
  - ObjectFifo `depth = 2` (Ping-Pong 이중 버퍼링으로 DMA 스트리밍과 연산 중첩)
  - 용량: $5,120 \times 2 = \mathbf{10,240 \text{ Bytes}}$ (10 KiB)
- **`x` 버퍼 (`of_x.cons()`)**:
  - $K$ 개의 `bfloat16` 원소 ($K \times 2 \text{ Bytes}$)
  - ObjectFifo `depth = 1` (단일 버퍼)
  - 용량: $\mathbf{2 \times K \text{ Bytes}}$
- **`y` 버퍼 (`y_cores[i]`)**:
  - $M_{tile} = 32$ 개의 `bfloat16` 원소 ($32 \times 2 = 64 \text{ Bytes}$)
  - ObjectFifo `depth = 2` (Ping-Pong)
  - 용량: $64 \times 2 = \mathbf{128 \text{ Bytes}}$
- **`y_acc` 누적 버퍼**:
  - $M_{tile} = 32$ 개의 `float32` 원소
  - 용량: $32 \times 4 = \mathbf{128 \text{ Bytes}}$
- **`sums` 열합 버퍼 (`b_group_sums`)**:
  - $K / 32$ 개 그룹의 32개 원소 합 (`bfloat16`)
  - 용량: $(K / 32) \times 2 = \mathbf{K / 16 \text{ Bytes}}$
- **스택 공간 (`STACK_SIZE`)**:
  - 컴파일러가 요구하는 최소 스택 + 안전 마진
  - 용량: $\mathbf{10,240 \text{ Bytes}}$ (10 KiB)

### 2) 총 L1 메모리 사용량 수식
$$\text{Total L1 Bytes} = 2K + 10,240 + 128 + 128 + \frac{K}{16} + 10,240 = \mathbf{2.0625 \times K + 20,736 \text{ Bytes}}$$

### 3) 주요 K 형상별 L1 메모리 사용량 및 여유 공간

| 입력 차원 $K$ | $x$ 버퍼 (B) | sums 버퍼 (B) | 고정 항목 (w+y+y_acc+스택) | 총 L1 사용량 | 잔여 L1 용량 | 사용률 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **256** (최소 타일) | 512 B | 16 B | 20,736 B | **21,264 B (20.77 KiB)** | 44,272 B (43.23 KiB) | 32.4% |
| **576 $\rightarrow$ 768** (MLA q_b) | 1,536 B | 48 B | 20,736 B | **22,320 B (21.80 KiB)** | 43,216 B (42.20 KiB) | 34.1% |
| **1,088 $\rightarrow$ 1,280** (Expert down) | 2,560 B | 80 B | 20,736 B | **23,376 B (22.83 KiB)** | 42,160 B (41.17 KiB) | 35.7% |
| **2,560** (Raon 표준 은닉) | 5,120 B | 160 B | 20,736 B | **26,016 B (25.41 KiB)** | 39,520 B (38.59 KiB) | 39.7% |
| **5,120** (Output projection) | 10,240 B | 320 B | 20,736 B | **31,296 B (30.56 KiB)** | 34,240 B (33.44 KiB) | 47.8% |
| **12,032** (Dense MLP down) | 24,064 B | 752 B | 20,736 B | **45,552 B (44.48 KiB)** | 19,984 B (19.52 KiB) | **69.5%** |
| **21,504** (L1 산식상 이론 상한, 컴파일·실측 미검증) | 43,008 B | 1,344 B | 20,736 B | **65,088 B (63.56 KiB)** | 448 B (0.44 KiB) | 99.3% |

> **지원 K 범위 확인 결과**:
> - 최소 지원 $K$: **256** (1개 $K_{tile}$, 사용률 32.4%)
> - 최대 $K$: L1 산식상 이론 상한(컴파일·실측 미검증), 실측 검증 최대 K=12,032 (Dense MLP down, 사용률 69.5%)
> - Raon-1-mini 모델에서 가장 큰 차원인 Dense MLP down-projection ($K=12,032$)에서도 L1 사용량은 **44.48 KiB (69.5%)** 로 약 19.5 KiB의 충분한 안전 여유 공간을 확보하고 있습니다.
> - 만약 $x$ 버퍼가 `depth=2`로 이중 버퍼링되었다면 $K=12,032$ 시 $x$만 47 KiB를 차지하여 총 69 KiB로 64 KiB L1을 초과(OOM/컴파일 에러)했을 것이나, `depth=1` 선택을 통해 이를 완벽히 방지했습니다.

---

## 3. Raon 투영 행렬 프리셋 형상 사양

Raon-1-mini의 전체 투영 계층을 모사하는 13개 벤치마크 구성입니다:

| 계층 구분 | 원래 형상 ($K \rightarrow M$) | 패딩 후 형상 ($K_{pad} \rightarrow M_{pad}$) | 활성 열 (`cols`) | 라운드 수 (`max_rounds`) | 패킹 크기 (KB/MB) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RDA q/k/v/g 묶음** | 2560 $\rightarrow$ 20480 | 2560 $\rightarrow$ 20480 | 8 | 20 | 32,000 KB (31.25 MB) |
| **RDA q/k/v/g 묶음 (cols=4)** | 2560 $\rightarrow$ 20480 | 2560 $\rightarrow$ 20480 | 4 | 40 | 32,000 KB (31.25 MB) |
| **출력 투영 (RDA/MLA o_proj)** | 5120 $\rightarrow$ 2560 | 5120 $\rightarrow$ 2560 | 8 | 3 (col 0~3: 3, col 4~7: 2) | 8,000 KB (7.81 MB) |
| **출력 투영 (cols=4)** | 5120 $\rightarrow$ 2560 | 5120 $\rightarrow$ 2560 | 4 | 5 | 8,000 KB (7.81 MB) |
| **MLA q_a+kv_a 묶음** | 2560 $\rightarrow$ 1152 | 2560 $\rightarrow$ 1152 | 8 | 2 (col 0: 2, col 1~7: 1) | 1,800 KB (1.76 MB) |
| **MLA q_b** | 576 $\rightarrow$ 7680 | 768 $\rightarrow$ 7680 | 8 | 8 (col 0~3: 8, col 4~7: 7) | 3,600 KB (3.52 MB) |
| **전문가 1 gate+up 묶음** | 2560 $\rightarrow$ 2176 | 2560 $\rightarrow$ 2176 | 8 | 3 (col 0: 3, col 1~7: 2) | 3,400 KB (3.32 MB) |
| **전문가 1 gate+up (cols=4)** | 2560 $\rightarrow$ 2176 | 2560 $\rightarrow$ 2176 | 4 | 5 (col 0: 5, col 1~3: 4) | 3,400 KB (3.32 MB) |
| **전문가 1 down** | 1088 $\rightarrow$ 2560 | 1280 $\rightarrow$ 2560 | 8 | 3 (col 0~3: 3, col 4~7: 2) | 2,000 KB (1.95 MB) |
| **밀집 MLP gate+up** | 2560 $\rightarrow$ 24064 | 2560 $\rightarrow$ 24064 | 8 | 24 (col 0~3: 24, col 4~7: 23) | 37,600 KB (36.72 MB) |
| **밀집 MLP down** | 12032 $\rightarrow$ 2560 | 12032 $\rightarrow$ 2560 | 8 | 3 (col 0~3: 3, col 4~7: 2) | 18,800 KB (18.36 MB) |
| **출력층 크기 대조 (lm_head)** | 2560 $\rightarrow$ 260096 | 2560 $\rightarrow$ 260096 | 8 | 254 | 406,400 KB (396.88 MB) |
| **고정 비용 측정용 (Minimal)** | 256 $\rightarrow$ 128 | 256 $\rightarrow$ 128 | 8 (실제 1) | 1 (col 0만 1) | 20 KB |

---

## 4. 커널 반올림 모드 및 정확도 검증 기준 명세

### 1) 커널 반올림 모드 (conv_even) 원리
- AIE2P 커널의 기본 부동소수점 반올림 모드는 `floor`(-무한대 방향 내림)입니다.
- $q4nx$ 행렬-벡터 연산은 $y = \text{scale} \cdot D + \min \cdot S$ 로 구성되며, $\min \approx -7.5 \cdot \text{scale}$ 관계를 갖습니다.
  - $D = \sum (\text{code} \cdot x)$ 는 `q4nx_gemv_block`에서 bf16으로 축소
  - $S = \sum x$ 는 `q4nx_gemv_prep`에서 bf16으로 축소
- 기본 `floor` 환경에서는 $D$와 $S$가 모두 아래로 편향(bias)되나, $\min$의 음수 부호로 인해 두 편향 오차가 부분적으로 상쇄(cancellation)됩니다.
- 만약 `q4nx_gemv_block`에만 `aie::set_rounding(aie::rounding_mode::conv_even)`을 적용하면, $D$만 비편향 최근접 짝수로 반올림되고 $S$는 계속 `floor`로 남아 상쇄가 깨지면서 오히려 상대 L2 오차가 급증합니다(K=12032 기준 1.5e-2 $\rightarrow$ 8.4e-2).
- 따라서 **`q4nx_gemv_prep`과 `q4nx_gemv_block` 두 진입점 모두 맨 앞**에 `aie::set_rounding(aie::rounding_mode::conv_even)`을 명시하여 $D$와 $S$ 양쪽의 편향을 대칭적으로 제거해야 정확도가 대폭 개선됩니다(K=12032 상대 L2: 1.50e-2 $\rightarrow$ 4.79e-3, Cosine: 0.999887 $\rightarrow$ 0.999989, NPU 런타임 불변).

### 2) 정확도 검증 2갈래 기준
벤치마크 실행기(`gemv_bench.py`)는 매 형상마다 다음 두 갈래 검증을 독립적으로 수행합니다:

1. **(가) 비트 대조 (Bit-exact Sample Contract)**:
   - **생성 방식**: AMD 계약 표본 규격(x는 $-4 \dots 4$의 작은 정수, codes는 $0 \dots 15$, scale과 min은 $2^{-e}$ 형태의 2의 거듭제곱)으로 행렬 전체 생성.
   - **참조 모형**: $D$와 $S$를 `conv_even`(RNE)으로 bf16 변환, float32로 $y_{\text{acc}}$ 누적 후 최종 narrow_to_bf16도 `conv_even`으로 변환하는 하드웨어 정밀 모형.
   - **판정 기준**: 불일치 개수(`mismatch_count`) 0개 시 **PASS** (1개 이상 불일치 시 FAIL).
2. **(나) 현실 표본 (Realistic Sample)**:
   - **생성 방식**: 연속 실수 정규분포 가중치 및 입력 벡터 대상.
   - **참조 모형**: float64 비양자화 역변환 행렬 곱($w_{\text{deq}} \times x$).
   - **판정 기준**: Cosine Similarity $\ge \mathbf{0.9999}$ 시 **PASS** (상대 L2 오차 및 최대 절대 오차를 정보로 함께 출력).
3. **(다) 비정수 표본 비트 대조 (Non-Integer Bit Check, 정보용)**:
   - **생성 방식**: 전용 bf16 비정수 x, 무작위 code, bf16 scale·min 대상 (최소 1개 묶음에 $D = \sum(\text{code} \cdot x)$가 bf16 RNE 동점 경계에 위치하는 값 주입).
   - **참조 모형**: `gemv_bit_reference` (커널 순차 누적 및 conv_even 모형).
   - **출력 정보**: 불일치 개수(`mismatch_count`) 및 최대 ULP(`max_ulp_err`)를 정보(INFO)로 출력 및 JSON에 기록.
   - **판정 미반영 사유**: 하드웨어 누적기의 내부 반올림 방식 및 `utils.h` 내 `aie::reduce_add`의 내부 가산 순서가 미확인(하드웨어 구현 의존) 상태이므로, 비정수 입력 시 순서 차이로 인한 비트 불일치가 발생할 수 있어 판정에는 사용하지 않고 참고 정보로만 제공합니다.

---

## 5. 노트북 NPU 실측 결과 (메인 작성 영역)

> **안내**: 아래 표는 노트북(Ryzen AI 7 350)에서 `bench\gemv\run-gemv.ps1`을 실행한 후 메인 세션(측정 담당자)이 채워 넣을 자리입니다.

| 형상 / 레이어 구분 | 패딩 후 형상 | Cols | 비트 대조 (불일치 / Max ULP) | Cosine 유사도 | Rel L2 오차 | NPU 평균 (ms) | p50 (ms) | p99 (ms) | 유효 대역폭 (GB/s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RDA q/k/v/g 묶음 | 2560 $\rightarrow$ 20480 | 8 | 0 / 0 | 0.999989 | 4.79e-03 | 0.661 | 0.661 | 0.749 | 49.5 |
| RDA q/k/v/g 묶음 (cols=4) | 2560 $\rightarrow$ 20480 | 4 | 0 / 0 | 0.999988 | 4.88e-03 | 0.875 | 0.875 | 0.995 | 37.4 |
| 출력 투영 (o_proj) | 5120 $\rightarrow$ 2560 | 8 | 0 / 0 | 0.999988 | 4.89e-03 | 0.304 | 0.316 | 0.386 | 26.9 |
| 출력 투영 (cols=4) | 5120 $\rightarrow$ 2560 | 4 | 0 / 0 | 0.999977 | 6.74e-03 | 0.295 | 0.289 | 0.334 | 27.8 |
| MLA q_a+kv_a 묶음 | 2560 $\rightarrow$ 1152 | 8 | 0 / 0 | 0.999987 | 5.06e-03 | 0.159 | 0.155 | 0.211 | 11.6 |
| MLA q_b | 768 $\rightarrow$ 7680 | 8 | 0 / 0 | 0.999990 | 4.58e-03 | 0.263 | 0.271 | 0.318 | 14.0 |
| 전문가 1 gate+up 묶음 | 2560 $\rightarrow$ 2176 | 8 | 0 / 0 | 0.999988 | 4.89e-03 | 0.182 | 0.178 | 0.226 | 19.2 |
| 전문가 1 gate+up (cols=4) | 2560 $\rightarrow$ 2176 | 4 | 0 / 0 | 0.999976 | 6.96e-03 | 0.200 | 0.192 | 0.272 | 17.4 |
| 전문가 1 down | 1280 $\rightarrow$ 2560 | 8 | 0 / 0 | 0.999985 | 5.41e-03 | 0.200 | 0.204 | 0.251 | 10.2 |
| 밀집 MLP gate+up | 2560 $\rightarrow$ 24064 | 8 | 0 / 0 | 0.999986 | 5.23e-03 | 0.751 | 0.754 | 0.874 | 51.2 |
| 밀집 MLP down | 12032 $\rightarrow$ 2560 | 8 | 0 / 0 | 0.999977 | 6.75e-03 | 0.611 | 0.633 | 0.759 | 31.5 |
| 출력층 크기 대조 (lm_head) | 2560 $\rightarrow$ 260096 | 8 | 0 / 0 | 0.999983 | 5.81e-03 | 6.404 | 6.398 | 6.793 | 65.0 |
| 고정 비용 측정용 | 256 $\rightarrow$ 128 | 8 (활성 1) | 0 / 0 | 0.999963 | 8.63e-03 | 0.164 | 0.169 | 0.274 | 0.1 |

측정: 2026-10-28 09:4x, 노트북(Ryzen AI 7 350, NPU 드라이버 32.0.203.314), 모양마다 자식 프로세스 하나, 예열 10·반복 100, 무작위 가중치(시드 고정). 비트 대조는 정수 표본(AMD 계약 방식)에서 NPU 출력과 최근접 짝수 반올림 모형의 bf16 비트 비교. 유효 대역폭은 패딩 포함 패킹 바이트 ÷ NPU 평균 시간. 같은 날 반올림 수정 전(AIE 기본 내림) 측정: 밀집 down 12032→2560 cosine 0.999887·상대 L2 1.50e-2, RDA 묶음 1.23e-2 — 수정 뒤 4.8e-3~6.7e-3. 시간 모형: NPU 시간 ≈ 0.16 ms(호출 고정 비용) + 코어당 블록 수 × 약 2.5 µs.

비정수 표본 비트 대조(정보, 2026-10-28 12:0x 재측정 — 순차 누적 참조·동점 경계 포함 표본): 13개 중 11개 모양 불일치 0, 밀집 MLP gate+up(24,064행) 1개·출력층 크기(260,096행) 3개가 1 ulp 차이. 하드웨어가 순차 fp32 누적·최근접 짝수 좁히기 모형과 비정수 값에서도 거의 비트 단위로 같다는 근거다(1 ulp 차이는 `reduce_add` 순서 또는 누적기 세부로 추정, 판정에는 쓰지 않음). 같은 실행에서 정수 표본 비트 대조는 13개 모두 0, 시간·GB/s 는 위 표와 같은 급.
