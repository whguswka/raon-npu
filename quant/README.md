# q4nx 4비트 가중치 변환기 (raon-npu)

KRAFTON `Raon-1-mini` 모델을 AMD Ryzen AI 7 350 (XDNA 2 AIE2P NPU)에서 실행하기 위한 CPU 기반 q4nx 4비트 양자화, 패딩 및 패킹 라이브러리입니다.

---

## 1. q4nx 포맷 요약

q4nx는 FastFlowLM 및 AMD IRON 연산자에서 사용하는 블록 양자화 포맷입니다.

| 항목 | 규격 | 비고 |
|---|---|---|
| 블록 크기 | $M = 32$ (출력/행), $K = 256$ (입력/열) | NPU 코어 타일 단위 |
| 양자화 그룹 | $GROUP = 32$ (입력 방향) | 블록당 8개 그룹 |
| 원소당 비트 수 | 5 비트 | 4비트 코드 + (2개 bf16 parameter / 32) = 5비트 |
| 블록 바이트 크기 | 5,120 바이트 | scale/min 1,024B + 코드 4,096B |
| 복원 수식 | $\text{weight} = \text{min} + \text{scale} \times \text{code}$ | min은 오프셋, scale/min은 bf16 |

---

## 2. bf16 격자 선택 규칙 (Grid Selection Rules)

NPU 하드웨어 디양자화기가 bf16 파라미터(`scale`, `min`)를 읽어 복원하므로, 가중치 그룹의 실제 범위가 bf16 격자 내에 완전히 포함되도록 방향 반올림(Directed Rounding)을 수행합니다.

### 1) 입력 유효성 및 오버플로우 방지
- **비유한값 엄격 거부**: NaN, Inf, -Inf가 하나라도 포함되면 즉시 `ValueError`를 발생시킵니다.
- **float64 연산**: 한 그룹에 큰 음수와 양수(예: $[-3\times 10^{38}, 3\times 10^{38}]$)가 공존하는 경우 차이($6\times 10^{38}$) 및 나눗셈 계산 시 발생하는 float32 오버플로우를 방지하기 위해 모든 통계 및 정규화 계산을 float64로 수행합니다.
- **파라미터 유한성 검사**: bf16으로 변환된 `scale_q`와 `min_q`가 오버플로우되어 비유한값(Inf/NaN)이 될 경우 `ValueError`를 발생시킵니다.

### 2) 격자 커버리지 보장 알고리즘
1. **`min_q` 선택**: 그룹 최소값 이하인 가장 큰 bfloat16을 선택합니다 (음의 무한대 방향 반올림, `bf16_floor`).
   $$\text{min}_q = \text{bf16\_floor}(\min(w))$$
2. **`scale_q` 선택**: 그룹 전체 폭을 15단계로 나눈 값 이상인 가장 작은 bfloat16을 선택합니다 (양의 무한대 방향 반올림, `bf16_ceil`).
   $$\text{scale}_q = \text{bf16\_ceil}\left(\frac{\max(w) - \text{min}_q}{15}\right)$$
3. **상한 커버리지 검증 및 승격**: 부동소수점 오차로 인해 $\text{min}_q + 15 \times \text{scale}_q < \max(w)$가 되는 경우, `scale_q`를 바로 다음 bfloat16(`bf16_nextafter`)으로 1단계 승격합니다.
4. **코드 결정**:
   $$\text{code} = \text{clip}\left(\text{rint}\left(\frac{w - \text{min}_q}{\text{scale}_q}\right), 0, 15\right)$$
   (단, $\max(w) == \text{min}_q$인 정확한 상수 그룹은 $\text{scale}_q = 0.0, \text{code} = 0$).
5. **수학적 성질 보장**: 모든 가중치 원소에 대해 다음 오차 상계가 반드시 성립합니다:
   $$|w - (\text{min}_q + \text{scale}_q \times \text{code})| \le \frac{\text{scale}_q}{2} + \epsilon$$

---

## 3. 배치 형식 (Two Packing Layouts)

NPU 연산자 구조에 따라 두 종류의 패킹 레이아웃을 제공합니다.

### 1) DecodeLayer 투영 배치 (`pack_decode_proj`)
- **대상**: 어텐션/RDA 및 FFN/MoE 투영 행렬 (`w_q`, `w_k`, `w_v`, `w_o`, `gate_up`, `down`)
- **참조**: IRON `layer/reference.py:pack_q4nx` 및 `dequant/README.md`
- **구조**:
  - 행(N)은 64(2 × M_TILE)의 배수, 열(K)은 256(K_TILE)의 배수.
  - 32행 블록(M_TILE)이 2개씩 짝지어져 64행 밴드를 구성하고, 밴드 간 블록이 인터리빙됩니다.
  - `blk = blk.reshape(nb_r // 2, 2, nb_c, BLOCK_BYTES).transpose(0, 2, 1, 3)`
- **역함수**: `parse_decode_proj`를 통해 원본 codes, scales, mins를 비트 단위로 복원합니다.

### 2) LMHead 어휘 배치 (`pack_lm_head`)
- **대상**: 최종 언어 모델 헤드 행렬 (`lm_head`, vocab × dim)
- **참조**: IRON `lm_head/reference.py:dequantize`, `lm_head/design.py:vocab_per_round`
- **구조**:
  - 32코어 전체 어레이(8열 × 4행) 기준 1라운드 = $8 \times 4 \times 32 = 1,024$ out-features.
  - 행(어휘)은 1,024의 배수, 열(dim)은 256의 배수.
  - 블록 인덱스 순서: `((round * cols + col) * k_blocks + k) * rows + row`.
  - 블록 내부: scale 256개(8×32) u16 + min 256개(8×32) u16 + codes 4,096B.
- **검증**: `lm_head/reference.py`의 `dequantize` 함수와 비교 시 오차 0.0 (`diff == 0.0`) 일치.

---

## 4. 패딩 및 로짓 마스킹 규칙

NPU 하드웨어 타일링 제약에 따라 행렬 크기를 0으로 채우고, 추론 후 원래 차원으로 되돌리거나 유효하지 않은 어휘 로짓을 마스킹합니다.

| 행렬 | 원본 크기 | 패딩 크기 | 단위 (M, K) | 비고 |
|---|---|---|---|---|
| MoE Gate/Up | (1088, 2560) | (1088, 2560) | (32, 256) | 이미 배수 (1088=32×34, 2560=256×10) |
| MoE Down | (2560, 1088) | (2560, 1280) | (32, 256) | 입력 1088을 1280으로 패딩 (256×5) |
| RDA 저랭크 | (40, 128) | (64, 256) | (32, 256) | 출력 40→64, 입력 128→256 |
| MLA q_b | (512, 576) | (512, 768) | (32, 256) | 입력 576을 768로 패딩 (256×3) |
| 어휘 (LMHead) | (259840, 2560) | (260096, 2560) | (1024, 256) | 어휘 259,840을 260,096으로 패딩 (1024×254) |

- `pad(w, m_tile=32, k_tile=256)`: 0 채움 및 `(padded_w, orig_shape)` 반환.
- `unpad(padded_w, orig_shape)`: 원래 크기로 슬라이싱 복원.
- `mask_padded_logits(logits, true_vocab=259840)`: 259,840 이후 패딩된 어휘 인덱스의 로짓을 `-inf`로 마스킹.

---

## 5. 시험 방법 및 AMD 원본 대조 (KAT)

pytest를 통해 단위 및 통합 테스트를 실행합니다:

```bash
python3 -m pytest quant/test_q4nx.py -v
```

### 1) AMD IRON 원본 정답 해시 시험 (Known-Answer Test)
변환용 Linux 기계 환경에는 `ml_dtypes` 및 `aie` 패키지가 설치되어 있지 않으므로, 노트북 환경에서 IRON 원본 `pack_q4nx`로 실측한 바이트 바이너리의 SHA256 해시를 박아두고 바이트 단위 일치를 검증합니다:

```python
def kat_inputs(rows, cols):
    n = rows * cols
    codes = ((np.arange(n, dtype=np.int64) * 7 + 3) % 16).astype(np.uint8).reshape(rows, cols)
    g = rows * (cols // 32)
    i = np.arange(g, dtype=np.int64)
    scales = (((i % 97) - 48) * 1e-3).astype(np.float32).reshape(rows, cols // 32)
    mins = (((i % 89) - 44) * 3e-3).astype(np.float32).reshape(rows, cols // 32)
    return codes, scales, mins
```

- `(64, 256)`: 길이 10,240 B, SHA256 `2f934f13747e9f5e261c226655ea811bbae8e45ff6784374d6dd31bd91a394fa`
- `(1088, 2560)`: 길이 1,740,800 B, SHA256 `155407fbcfd7ba1b3ddd4a242ec7cb16d5725ad201fa001283180f11048af0bc`
- `(2560, 1280)`: 길이 2,048,000 B, SHA256 `12846cb24767c9f150311cba31b9186a87df6fdf12d8c517553493be08181fee`
- `(128, 768)`: 길이 61,440 B, SHA256 `7815e5a5b40aafb908cbb0c774914520fed87cbc09a91a72bd76adab59ca5d7f`

또한, `ml_dtypes`가 존재하는 환경에서는 `test_pack_decode_proj_iron_direct_comparison`이 자동으로 활성화되어 IRON 원본 함수와 직접 바이트 단위 비교를 수행합니다.

### 2) 전체 테스트 항목 (19개 수집, 18 통과, 1 스킵)
1. `test_bf16_direction_rounding_properties`: `bf16_floor`, `bf16_ceil`, `bf16_nextafter`의 IEEE-754 방향 반올림 및 격자 보증 검증.
2. `test_quantize_random_bounds`: 무작위 행렬 양자화 오차가 $0.5 \times \text{scale}$ 이하인지 검증.
3. `test_quantize_non_finite_rejected`: NaN, Inf, -Inf 입력 시 `ValueError` 검출 검증.
4. `test_quantize_large_magnitude_opposite_signs`: $[-3\times 10^{38}, 3\times 10^{38}]$ 양·음 극단값 공존 시 float32 오버플로우 방지 및 오차 상계 검증.
5. `test_quantize_narrow_range_far_from_zero`: 0에서 먼 좁은 범위($[1001.0, 1001.1]$, $[-1001.1, -1001.0]$)에서 코드 포화 방지 및 오차 상계 검증.
6. `test_quantize_inexact_constant_group`: bf16으로 정확히 표현되지 않는 상수 그룹($0.1$) 양자화 검증.
7. `test_quantize_exact_constant_groups`: max == min인 정확한 상수 그룹($0.0, 5.0, -12.5$) 처리 검증.
8. `test_quantize_validation_errors`: 2D 및 열 32배수 검증.
9. `test_pack_decode_proj_kat_sha256`: AMD 원본 실측 정답 SHA256 해시 4종 대조 검증.
10. `test_pack_decode_proj_iron_direct_comparison`: IRON `pack_q4nx` 직접 대조 (환경 부재 시 안전하게 skip).
11. `test_pack_decode_proj_roundtrip`: `pack_decode_proj`와 `parse_decode_proj` 왕복 비트 단위 일치.
12. `test_pack_decode_proj_shape_validation`: 64행 및 256열 배수 예외 검증.
13. `test_decode_proj_end_to_end`: quantize → pack → parse → dequantize 엔드-투-엔드 오차 $0.5 \times \text{scale}$ 검증.
14. `test_pack_lm_head_roundtrip_and_iron_dequantize`: IRON `lm_head.reference.dequantize`와 바이트 단위 복원 오차 0.0 확인.
15. `test_pack_lm_head_multiple_rounds`: 다중 라운드(2라운드=2048 어휘) 검증.
16. `test_pack_lm_head_shape_validation`: 1024행 및 256열 배수 예외 검증.
17. `test_pad_unpad_raon_shapes`: MoE gate/up, MoE down, RDA 저랭크, MLA q_b, LMHead 전체 어휘(259840×2560) 채움/되돌림.
18. `test_mask_padded_logits`: 1D/2D 로짓 마스킹 및 in-place 지원 검증.
19. `test_to_bf16_rne_precision`: IEEE-754 RNE tie-to-even 비트 수준 정확도 검증.

---

## 6. 아직 안 한 것 (Next Steps)

본 작업은 CPU 상의 q4nx 변환/패킹 기반 및 관문 검증이며, 후속 작업은 다음과 같습니다:

1. **MXFP8 원본 가중치 변환 파이프라인**:
   - Raon-1-mini 가중치 체크포인트 다운로드 후 전문가 MXFP8(E4M3 + E8M0 scale) 및 기타 레이어 bf16 가중치를 float32 거쳐 q4nx로 일괄 변환.
2. **양자화 품질 측정**:
   - 가중치 행렬별 cosine similarity 및 최대 오차 측정.
   - SiTU-GLU 활성화 전후 오차 곡선 대조 ($\beta=4.0$, 선형 $\beta=25.0$).
   - 층 출력 cosine $\ge 0.995$ 및 첫 32 토큰 greedy 일치 검증.
3. **NPU 하드웨어 연동**:
   - 노트북(Ryzen AI 7 350) 환경으로 패킹된 q4nx 바이너리 전송 및 AIE2P 커널 실행/마이크로벤치마크.
