# Raon-1-mini RDA fp32 State Update Microbenchmark

AMD XDNA 2 (AIE2P, NPU2: 8열 × 코어 4행, 코어 L1 64 KiB, 메모리 타일 512 KiB/열) NPU 상의 RDA(Raon Delta Attention) 상태 갱신 마이크로커널 벤치마크.

KRAFTON `Raon-1-mini` 모델의 디코드(T=1) 시점 순환식 RDA 상태 갱신을 NPU 코어 및 메모리 타일을 활용하여 스트리밍 방식으로 가속하고 성능 및 정밀도를 측정합니다.

---

## 1. 개요 및 계산 수식

`torch_recurrent_rda` (`modeling_raon1_mini.py:241-274`) 기준:
- **차원**: 헤드 수 $H = 40$, 키 차원 $K = 128$, 값 차원 $V = 128$.
- **상태 $S$**: fp32 $[40, 128, 128]$ (2.5 MiB). 토큰마다 DDR에서 읽고 갱신 후 DDR에 다시 씁니다.
- **입력 (토큰마다)**:
  - $q[40, 128]$, $k[40, 128]$, $v[40, 128]$ (fp32)
  - $a[40, 128] = \exp(\text{log\_alpha})$ (호스트가 미리 계산하여 fp32로 공급)
  - $\beta[40]$ (fp32)
- **헤드 $h$, 값 행 $v$ 마다의 연산 (행끼리 완전 독립)**:
  1. 감쇠(Decay): $S[h, v, :] \leftarrow S[h, v, :] \odot a[h, :]$
  2. 예측(Predict): $\text{pred} \leftarrow \sum_{k} S[h, v, k] \cdot k[h, k]$
  3. 오차(Error): $\text{err} \leftarrow v[h, v] - \text{pred}$
  4. 갱신(Update): $S[h, v, :] \leftarrow S[h, v, :] + \beta[h] \cdot \text{err} \cdot k[h, :]$
  5. 출력(Output): $\text{out}[h, v] \leftarrow \sum_{k} S[h, v, k] \cdot q[h, k]$
- **타일 단위**:
  - $V$ 방향 32행 = $32 \times 128$ fp32 = **16 KiB**가 1개 타일 단위.
  - 헤드당 4개 타일 ($4 \times 32 = 128$행).
  - 총 $40 \times 4 = 160$ 타일/토큰.

---

## 2. 데이터 흐름 다이어그램 및 DMA 채널 예산 (Dataflow Architecture & DMA Channel Budget)

### 2.1 하드웨어 타일별 DMA 채널 예산 및 준수 현황

AIE2P (XDNA 2) 하드웨어 제약(Core Tile: 입력 2 / 출력 2, Shim Tile: 입력 2 / 출력 2, MemTile: 입력 6 / 출력 6)을 충족하기 위해, 헤드별 작은 파라미터 벡터들($q, k, a, v, \beta$)을 640 floats 크기의 단일 **"헤드 묶음(Head Bundle)"** 버퍼로 결합했습니다.

| 타일 유형 (Tile Type) | 위치 (Row) | 가용 S2MM (입력) | 사용 S2MM (입력) | 입력 스트림 연결 대상 | 가용 MM2S (출력) | 사용 MM2S (출력) | 출력 스트림 연결 대상 | 상태 |
|---|---|---|---|---|---|---|---|:---:|
| **Shim Tile** (NOC 인터페이스) | Row 0 | **2** | **2** | AIE $\rightarrow$ Host:<br>1. `s_out` (갱신 상태)<br>2. `out` (출력 벡터) | **2** | **2** | Host $\rightarrow$ AIE:<br>1. `s_in` (이전 상태)<br>2. `head_in` (헤드 묶음) | **PASS** (100%) |
| **Memory Tile** (`--use-memtile 1`) | Row 1 | **6** | **2** | 1. Shim $\rightarrow$ MemTile (`s_in_l3l2`)<br>2. Core $\rightarrow$ MemTile (`s_out_core`) | **6** | **2** | 1. MemTile $\rightarrow$ Core (`s_in_core`)<br>2. MemTile $\rightarrow$ Shim (`s_out_l2l3`) | **PASS** (33%) |
| **Compute Core Tile** (AIE2P 코어) | Row 2 | **2** | **2** | 1. `s_in_core` (상태 타일, 16 KiB)<br>2. `head_in` (헤드 묶음, 2.5 KiB) | **2** | **2** | 1. `s_out_core` (갱신 상태, 16 KiB)<br>2. `out` (출력 슬라이스, 128 B) | **PASS** (100%) |

### 2.2 AIE2P 코어 L1 (64 KiB) 메모리 예산 및 라우팅 경로별 비교

컴파일러 요구 스택 크기(`stack_size = 12288`, 10,560 B 이상 필요)를 반영한 코어 L1(65,536 Bytes) 예산 계산입니다:

| 경로 (Routing) | 상태 입력 (`s_in`) | 상태 출력 (`s_out`) | 헤드 묶음 (`head_in`) | 출력 슬라이스 (`out`) | 코어 스택 (`stack_size`) | 총 L1 점유량 | 잔여 여유 공간 | 판정 |
|---|---|---|---|---|---|---|---|:---:|
| **메모리 타일 경유**<br>(`--use-memtile 1`, 기본/단일화) | 16 KiB (깊이 1) | 16 KiB (깊이 1) | 5 KiB (2.5 KiB×2, 깊이 2) | 256 B (128 B×2, 깊이 2) | 12 KiB (12,288 B) | **50,432 B (~49.25 KiB)** | **+15,104 B (~14.75 KiB, 23.0%)** | **PASS** (안정 실행) |
| **직결 경로 (기존)**<br>(`--use-memtile 0`) | 32 KiB (깊이 2) | 32 KiB (깊이 2) | 5 KiB (깊이 2) | 256 B (깊이 2) | 12 KiB (12,288 B) | **83,200 B (~81.25 KiB)** | **-17,664 B (26.9% 초과)** | **FAIL** (L1 초과) |
| **직결 대안 A**<br>(상태 출력 깊이 1 축소) | 32 KiB (깊이 2) | 16 KiB (깊이 1) | 5 KiB (깊이 2) | 256 B (깊이 2) | 12 KiB (12,288 B) | **66,816 B (~65.25 KiB)** | **-1,280 B (1.95% 초과)** | **FAIL** (여전히 초과) |
| **직결 대안 B**<br>(상태 입·출력 모두 깊이 1) | 16 KiB (깊이 1) | 16 KiB (깊이 1) | 5 KiB (깊이 2) | 256 B (깊이 2) | 12 KiB (12,288 B) | **50,432 B (~49.25 KiB)** | **+15,104 B (23.0% 여유)** | **폐기** (더블버퍼링 상실/직렬화) |

> **직결 경로 폐기 및 MemTile 단일화 근거**:
> 1. 직결 경로에서 더블 버퍼링(깊이 2) 유지 시 L1 사용량이 81.25 KiB로 64 KiB를 초과합니다.
> 2. 상태 출력을 깊이 1로 줄여도 65.25 KiB로 여전히 64 KiB를 초과합니다.
> 3. 상태 입·출력을 모두 깊이 1로 줄이면 L1 용량 내에 들어오지만, 더블 버퍼링이 사라져 Shim DMA와 코어 연산이 락스텝(lock-step)으로 직렬화됩니다.
> 4. XDNA 2는 대용량 상태(2.5 MiB)의 L3-L1 스테이징을 위해 열당 512 KiB의 고대역 Memory Tile(Row 1)이 제공되므로, MemTile에서 깊이 2 더블 버퍼링을 하고 Core L1은 깊이 1 포워딩을 취하는 메모리 타일 경유 경로로 단일화했습니다.

### 2.3 데이터 흐름 다이어그램

```
================================ DDR (L3 External Memory) ================================
  [S_in: 2.5 MiB]     [head_in: 100 KiB (q+k+a+v+beta+pad)]      [S_out: 2.5 MiB]  [out: 20 KiB]
       │                                │                              ▲              ▲
       │ Shim DMA (Fill, Ch 0)          │ Shim DMA (Fill, Ch 1)        │ Drain (Ch 0) │ Drain (Ch 1)
       ▼                                ▼                              │ (wait=True)  │
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ TaskGroup Orchestration (8 Columns in Parallel)                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
       │                                │                              ▲              ▲
       ▼                                │                              │              │
┌───────────────────────────────────────┼──────────────────────────────┼──────────────┼─────────┐
│ Memory Tiles (Row 1) - 512 KiB / Col  │                              │              │         │
│   of_s_in_l3l2 (depth 2, 32 KiB) ──► forward(depth 1) ──► of_s_in_core              │         │
│   of_s_out_l2l3 (depth 2, 32 KiB) ◄── forward(depth 2) ◄── of_s_out_core             │         │
└───────────────────────────────────────┼──────────────────────────────┼──────────────┼─────────┘
       │                                │                              ▲              ▲
       ▼                                ▼                              │              │
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ AIE2P Compute Cores (Row 2) - L1 64 KiB / Core                                                │
│                                                                                               │
│   Worker w (w = 0 .. N-1, Column w):                                                          │
│   - of_s_in_core   (16 KiB, depth 1) [S2MM Channel 0]                                         │
│   - of_head_in     (2.5 KiB, depth 2 = 5 KiB) [S2MM Channel 1]                                │
│   - of_s_out_core  (16 KiB, depth 1) [MM2S Channel 0]                                         │
│   - of_out         (128 B, depth 2 = 256 B) [MM2S Channel 1]                                 │
│   - Core Stack     (12 KiB = 12,288 B)                                                        │
│   [Total L1 Buffer Footprint: ~49.25 KiB / 64 KiB — Safe Margin 14.75 KiB (23.0%)]            │
│   [DMA Channels Used: 2 S2MM in / 2 MM2S out (Exceeds 0% of budget!)]                         │
│                                                                                               │
│   Inner Core Loop:                                                                            │
│     for head in range(heads_per_worker):                                                      │
│       acquire(head_in)  # 640 floats (q: 0..127, k: 128..255, a: 256..383, v: 384..511, beta)│
│       for tile_idx in range(4):                                                               │
│         acquire(s_in, s_out, out)                                                             │
│         rda_update_tile(s_in, s_out, head_in, tile_idx, out)                                  │
│           v32 = head_in + 384 + tile_idx * 32                                                 │
│           Pass 1: decay s_in & dot(s, k) -> pred (AIE 16-lane vector mul/mac)                 │
│           Scalar: err = v32[row] - pred; scale = scalar_mul(beta, err)                        │
│           Pass 2: s_out = s_decay + scale*k & dot(s, q) -> out                                │
│         release(s_in, s_out, out)                                                             │
│       release(head_in)                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 실행 방법 (노트북 환경)

노트북(Windows 원생 Python 3.13, mlir-aie v1.4.4, XRT SDK 2.21.75):

```bash
# 기본 실행: Worker 4개, 64 토큰
python bench/rda_state/rda_state.py --workers 4 --tokens 64

# Worker 1개 (단일 코어 베이스라인)
python bench/rda_state/rda_state.py --workers 1 --tokens 64

# Worker 8개 (8열 병렬 최대 활용)
python bench/rda_state/rda_state.py --workers 8 --tokens 64

# 옵션 목록 확인
python bench/rda_state/rda_state.py --help
```

### 주요 커맨드라인 옵션
- `--workers {1, 4, 8}`: 사용할 AIE Worker 코어 개수 (기본값: 4).
- `--tokens T`: 순차 실행할 디코드 토큰 개수 (기본값: 64).
- `--heads H`: 총 RDA 헤드 수 (기본값: 40).
- `--use-memtile {0, 1}`: 메모리 타일 경유 버퍼링 여부 (기본값: 1).
- `--warmup K`: JIT 컴파일 및 웜업 횟수 (기본값: 5).
- `--seed S`: 난수 시드 (기본값: 42).

---

## 4. 판정 기준 (Verification & Acceptance Criteria)

1. **수치 정확도 (Accuracy)**:
   - 기준: **상태 NRMSE $\le 1.0 \times 10^{-4}$** 및 **토큰별 출력 상대 L2 $\le 1.0 \times 10^{-5}$**
   - 비교 대상: `torch_recurrent_rda`와 정확히 동일한 Numpy fp32 골든 참조.
   - **단계별 조기 검증 (Early Step Verification)**:
     - **처음 3 토큰(토큰 1, 2, 3)**은 매 토큰 연산 직후 NPU 상태 전체를 해당 시점의 골든 상태와 직접 비교하여 NRMSE $\le 1.0 \times 10^{-4}$를 만족하는지 단계별로 검증.
   - **최종 정확도 측정 항목**:
     - 토큰별 출력 최대 절대 오차 (`out_max_err`) 및 상대 L2 오차 (`out_rel_l2`)
     - 최종 상태 최대 절대 오차 (`s_max_err`), 상대 L2 오차 (`s_rel_l2`), NRMSE (`s_nrmse`)

2. **성능 및 지연 시간 (Performance)**:
   - `aie.utils.benchmark.run_iters`를 통한 NPU 순수 실행 시간 (`avg_us`, `min_us`, `max_us`).
   - NPU 유효 DDR 대역폭 (토큰당 입출력 트래픽 약 5.12 MiB 기준).
   - 호스트 파이썬 엔드투엔드 호출 지연 시간 p50, p99, 평균.

---

## 5. NPU 마이크로벤치 실측 결과 (메인 실측)

- **측정 환경**: 노트북(127.0.0.1, Ryzen AI 7 350, XDNA 2 AIE2P NPU, Windows 11 원생 환경)
- **실행 경로**: 기본 경로 (메모리 타일 경유, `--use-memtile 1`)

### 5.1 실측 요약표

| Workers (코어 수) | Tokens (토큰 수) | NPU 시간 평균 | 파이썬 호출 p50 | 토큰별 출력 상대 L2 | 최종 상태 상대 L2 | 최종 상태 NRMSE | 판정 / 상태 |
|---|---|---|---|---|---|---|:---:|
| **8 (메인 최종 실측, 10-28 01:4x, 판정 강화 후)** | 32 | **0.75 ms** | - | **1.4e-7** | **2.4e-8** | **8.6e-10** | **PASS** (상태·출력 두 조건) |
| 8 (메인 실측, 10-27 21:3x, 상태 비교 수정 후) | 32 | 0.71 ms | 1.29 ms | 1.4e-7 | 2.4e-8 | 8.6e-10 | PASS (상태 조건) |
| 1 (초기 측정 회차) | 8 | 4.44 ms | 5.08 ms | 1.42e-7 | - | 미측정* | 출력 정상 (수정 후 상태 미측정) |
| 4 (초기 측정 회차) | 32 | 1.23 ms | 1.78 ms | 1.40e-7 | - | 미측정* | 출력 정상 (수정 후 상태 미측정) |
| 8 (초기 측정 회차) | 32 | 0.69 ms | 1.18 ms | 1.40e-7 | - | 미측정* | 출력 정상 (버그 회차 0.69 ms) |

> **\*실행기 버그 및 상태 검증 정정 설명**:
> - 초기 측정 회차(1코어 4.44 ms, 4코어 1.23 ms, 8코어 0.69 ms)에서는 토큰별 출력이 이미 골든과 일치(상대 L2 ~1.4e-7)했으나, 타이밍 측정 도구(`run_iters`)가 순차 루프 후 동일한 NPU 상태 버퍼(`s_out_npu`)를 넘겨받아 추가 연산(warmup 2 + iters 20)을 수행하면서 상태를 덮어써 최종 상태 비교만 FAIL(1코어 9.07e-3, 4·8코어 1.01e-2)이 발생했던 실행기 버그였습니다.
> - 상태 스냅샷(`s_final_npu`) 즉시 보존 및 `run_iters` 전용 스크래치 버퍼 분리 수정 후 **메인 실측(10-27 21:3x, 8코어 32토큰)**에서 NPU 0.71 ms/층, 파이썬 p50 1.29 ms, 토큰별 출력 상대 L2 1.4e-7, 최종 상태 상대 L2 2.4e-8, 최종 상태 NRMSE 8.6e-10을 기록하여 판정 기준(상태 NRMSE $\le 1.0\times 10^{-4}$ 및 출력 상대 L2 $\le 1.0\times 10^{-5}$)을 만족하여 **PASS**되었습니다.
> - 수정 후 재측정하지 않은 1코어 및 4코어의 상태값은 PASS로 추정하지 않고 '미측정'으로 표기합니다.

### 5.2 주요 관찰: 코어 수 확장성 및 병목 분석
1. **코어 수 확장성**:
   - 1 Worker: 4.44 ms (기준 1.00x)
   - 4 Workers: 1.23 ms (3.61x 스피드업)
   - 8 Workers: 0.71~0.75 ms (5.9~6.3x 스피드업; 회차별 0.69·0.71·0.75 ms)
   - 코어 수 증가에 따라 NPU 연산 시간이 거의 선형으로 감소합니다.
2. **연산 병목 분석 (추정)**:
   - 토큰당 입출력 데이터량은 약 5.12 MiB (상태 2.5 MiB R/W + 헤드 묶음 100 KiB + 출력 20 KiB).
   - 토큰당 트래픽은 실행기 계산으로 5,365,760 B. 8 Workers 유효 대역폭은 0.75 ms 에서 약 7.15 GB/s, 0.71 ms 에서 약 7.56 GB/s, 0.69 ms 에서 약 7.78 GB/s(10⁹ B/s 기준)로, 같은 노트북에서 memcpy 로 잰 DDR 읽기+쓰기 최대(57~62 GB/s)의 약 12~14% 수준입니다.
   - 코어 수에 거의 비례해 줄어 연산 병목으로 보인다(추정). 다만 유효 DDR 트래픽 수치만으로 확정할 수는 없으며 compute-bound로 추정되나 추가 프로파일링이 필요합니다.

---

## 6. 1열 타일 단위 작업자 설계 (opt3 관문 ③·④)

`DESIGN-rda-merge.md` 메인 검토를 따르는 후속 설계.  RDA 상태 갱신을 통합 GEMV 설정의 남는 1열(예: 열 7)에 단독으로 배치한다.  핵심 아이디어는 **작업자를 헤드가 아니라 상태 타일 단위로 분할**하는 것이다.

### 6.1 타일 단위 작업자 할당

- 한 열 = 심 1 + 메모리 타일 1 + 코어 4(행 2~5).
- 작업자 `t`는 코어 `(col, 2+t)`에서 **모든 헤드의 V-타일 `t`**(행 `t*32:(t+1)*32`)를 맡는다.
- `rda_update_tile_packed(s_in, s_out_packed, head_bundle, tile_idx)`는 한 타일의 새 상태(4096 float)와 출력 슬라이스(32 float)를 하나의 4128-float 원소에 쓴다.

### 6.2 데이터 흐름 및 DMA 채널 예산

| 흐름 | 경로 | 심 | 메모리 타일 | 코어(각) |
|---|---|---|---|---|
| `s_in` (헤드마다 64 KiB = 타일 4개) | DDR → 심 MM2S → 메모리 타일(헤드 원소, 깊이 2) → split → 코어 t | MM2S 1 | S2MM 1 · MM2S 4 | S2MM 1 |
| `head_in` (헤드마다 2.5 KiB) | DDR → 심 MM2S → 4코어 방송(메모리 타일 거치지 않음) | MM2S 1 | — | S2MM 1 |
| `s_out + out` | 코어 t가 [4096+32] 원소 → 메모리 타일 집계 버퍼 `[4×4128]`(깊이 2) → 두 소비자: 상태 부분(보폭 4128 의 4096×4) → 심 S2MM → `s_out`, out 부분(오프셋 4096·보폭 4128 의 32×4) → 심 S2MM → `out` | S2MM 2 | S2MM 4 · MM2S 2 | MM2S 1 |
| **합계** | | **MM2S 2/2 · S2MM 2/2** | **S2MM 5/6 · MM2S 6/6** | **S2MM 2/2 · MM2S 1/2** |

- 메모리 타일 메모리: `s_in` 64 KiB×2 + `s_out/out` 64.5 KiB×2 ≈ **257 KiB / 512 KiB**.
- 코어 L1: `s_in` 타일 16 KiB + packed output 16.1 KiB + head bundle 2.5 KiB×2 + 스택 12 KiB ≈ **49.3 KiB / 64 KiB**.

### 6.3 명시적 출력 집계 버퍼 및 잠금 규약

메모리 타일에 `[4, 4128] fp32` ping/pong 버퍼를 두고, 같은 버퍼를 서로 다른 TAP로 읽는 두 개의 MM2S task로 `s_out`과 `out`을 갈라 쓴다.  ObjectFifo 하나에 소비자 둘을 달지 않고 명시적 `Buffer` + `Flow` + `Lock`으로 동기화한다.

잠금은 슬롯 `s(0·1)` × 코어 `t(0..3)`마다 두 개씩 둔다: `agg_empty[s][t]`(초기 2), `agg_full[s][t]`(초기 0).  총 16개(메모리 타일 한도 64).

- 생산자(코어 `t` → 메모리 타일 S2MM): 슬롯 0·1 BD를 핑퐁으로 두고, 각 BD가 자기 구간 `[t*4128:(t+1)*4128]`에 쓰기 전 `agg_empty[s][t]`를 값 2로 획득하고, 쓴 뒤 `agg_full[s][t]`를 값 2로 해제한다.
- 소비자 1(s_out drain): 슬롯 0의 4개 BD(코어 구간마다 하나, 오프셋 `t*4128`, 길이 4096) → 슬롯 1의 4개 BD → 반복.  각 BD는 `agg_full[s][t]`를 값 1로 획득하고 `agg_empty[s][t]`를 값 1로 해제한다.
- 소비자 2(out drain): 같은 꼴로 오프셋 `t*4128 + 4096`, 길이 32.

생산자는 `empty[s][t]`가 2가 될 때까지, 즉 두 소비자 모두 자기 구간을 읽고 각각 empty 1을 돌려준 뒤에야 같은 슬롯을 재사용할 수 있다.  따라서 한 소비자만 끝나도 덮어쓰이지 않는다.  BD 사슬 반복 횟수는 헤드 수에 맞춘다: 쓰기 2-BD 사슬은 `H/2`회, 읽기 8-BD 사슬도 `H/2`회(`H=40`이면 20회).  `runs`는 IRON `DmaEndpoint.task`에서 "전체 BD 사슬을 몇 번 반복할지"를 뜻한다.

| 항목 | 개수 |
|---|---|
| 메모리 타일 명시적 BD(쓰기) | 4코어 × 2슬롯 = 8 |
| 메모리 타일 명시적 BD(읽기) | 2채널 × 8 = 16 |
| 메모리 타일 잠금 | 16(agg) + 8(core) = 24 / 64 |
| 심 활성 BD(런타임 시퀀스) | `s_in`·`head_in` 채우기 + `s_out`·`out` 비우기 = 4 (+ `s_in` ObjectFifo producer 1개) |

비-심 DMA task는 완료 토큰을 내지 않는다.  마지막 심 S2MM drain 완료가 전체 호출 완료를 보장한다.

### 6.4 새 파일

- `bench/rda_state/rda_state_1col.py`: `@iron.jit rda_step_1col(s_in, s_out, head_in, out, *, num_heads=40, col=0)`.
- `bench/rda_state/compare_1col.py`: 노트북에서 두 설계를 같은 프로세스로 비교·벤치마크·soak.
- `bench/rda_state/test_rda_state_1col.py`: NPU 없이 돌아가는 순수 파이썬 TAP/수치 검증.
- `kernels/rda_state.cc`: `rda_update_tile_packed` 감싸기 함수 추가(기존 `rda_update_tile`은 그대로).

### 6.5 노트북 실행 명령

```bash
python -m bench.rda_state.compare_1col --seeds 1 2 3 --tokens 200 --iters 500 --soak 10000
```

## 7. 커널 3단 일정(R1)

`rda_update_tile`은 행마다 패스 1 → 스칼라 → 패스 2를 하던 일정을 모든 행 패스 1 → 모든 행 스칼라 → 모든 행 패스 2의 3단으로 바꿨다. 행마다 연산·순서는 같다.
단독 1열 설계, 헤드 40, 300호출 장치 시간 중앙값은 1.214 → 1.085 ms/호출(−10.6%)이었고, 96토큰 × 3벌에서 상태·출력이 바이트로 같았다.
행 엮기는 넣지 않았다. 2행은 1.138 ms로 느려졌고, 4행은 프로그램 메모리 16 KB를 넘쳐 빌드에 실패했다.
이 커널은 디코드 통합 GEMV xclbin과 prefill 판 xclbin의 RDA 열이 함께 쓰므로, 변경하면 두 xclbin을 다시 빌드한다.
컴파일러가 바뀌면 특수값을 포함해 옛·새 커널의 바이트 대조를 다시 한다.

## 8. FP32 곱 32칸 묶음(R4)

AIE2P 에는 FP32 곱셈기가 없어 FP32 벡터 곱을 bf16 곱으로 흉내 내는데(llvm-aie `aie2p_vmult.h` `mul_elem_32_accuracy_safe_inner`), 이 흉내는 32칸 단위로만 돈다. 16칸 float 벡터로 곱하면 곱마다 32칸 일을 하고 16칸만 쓴다.
`rda_update_tile` 의 1·3단계 안쪽 루프는 키 32개씩 32칸 곱을 하고, 곱 합의 앞·뒤 16칸을 키 순서대로 16칸 누산기에 더한다. 흉내 곱은 칸마다 따로 계산되므로 칸마다 연산·순서가 같고 비트도 같다.
단독 1열 설계, 헤드 40, 300호출 장치 시간 중앙값은 1.085~1.105 → 0.773~0.794 ms/호출(−28.6%)이었고, 96토큰 × 3벌에서 상태·출력이 바이트로 같았다(앞뒤 16칸 순서를 뒤집은 음성 대조는 96/96 다름). `.text` 는 8,064 → 5,328 B.
512비트 이상 벡터의 읽기·쓰기 정렬은 64 B 라(AIE API `detail/ld_st.hpp`) 32칸 읽기·쓰기도 지금 버퍼 배치 그대로 된다.
흉내 곱의 조각·곱 순서는 툴체인 헤더가 정하므로, 컴파일러가 바뀌면 특수값을 포함해 옛·새 커널의 바이트 대조를 다시 한다.

## 9. 두 행 엮기(R5)

흉내 FP32 곱 하나는 bf16 곱 9개를 덧셈 하나씩 차례로 더하는 사슬이다. 덧셈마다 앞 덧셈 결과를 기다리므로(간격 4묶음) 사슬 하나로는 곱셈기 자리(`vmul.f`·`vadd.f`·`vmsc.f` 가 나눠 쓰는 VLIW 묶음의 한 자리)의 절반이 빈다. 컴파일러는 서로 독립인 사슬도 스스로 겹치지 않았다(R4 한 행: 1단계 353·3단계 392묶음, 곱셈기 자리 사용률 41~46%).
`rda_update_tile` 의 1·3단계는 한 번에 두 행을 처리한다. 헤더의 흉내(피연산자마다 세 조각 쪼개기, 곱 9개와 그 순서, mac 의 마지막 덧셈)를 `fp32_split`·`fp32_mul2` 로 그대로 풀어 쓰고, 두 행의 같은 순번 곱을 번갈아 둔다. 행마다 연산·순서가 같아 비트가 같다. 헤드 상수 a·k·q 는 한 번만 쪼갠다(R1·R4 에서도 컴파일러가 행 루프 밖에서 한 번 쪼갰다).
단독 1열 설계, 헤드 40, 300호출 장치 시간 중앙값은 0.771 → 0.654~0.658 ms/호출(−15%)이었고, 96토큰 × 3벌에서 상태·출력이 바이트로 같았다(곱의 두 피연산자 역할을 뒤집어 합 순서를 바꾼 음성 대조는 96/96 다름). 한 행 묶음은 279·276, `.text` 는 5,328 → 7,888 B, 스택 프레임은 3,776 → 4,736 B 다(mlir-aie 1.4.4 에서 잰 값).
네 행은 −17.7% 였지만 `.text` 가 13,632 B(프로그램 메모리 16 KB 의 83%)라 넣지 않았다. AIE API 호출(`::aie::mul`·`::aie::mac`) 그대로 두 행만 묶으면 −8.1% 였다.
흉내 순서를 커널이 직접 가지므로, 컴파일러를 바꿀 때는 헤더 흉내의 차이를 먼저 보고, `fp32_split`·`fp32_mul2` 를 새 헤더와 맞춰 본 뒤, 두 xclbin 을 다시 빌드해 특수값을 포함한 옛·새 커널의 바이트 대조를 한다. 헤더의 흉내 순서가 바뀌었으면 새 순서를 따를지 옛 비트를 지킬지 정한다. 곱 순서가 같아도 내장 함수가 기계 명령으로 바뀌는 방식이나 반올림 규칙이 바뀌면 출력이 달라질 수 있다.
