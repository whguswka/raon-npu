<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 raon-npu authors -->

# Raon-1-mini 양자화 품질 평가

## 평가 경계와 정정 사항

기준은 HF 체크포인트의 MXFP8→BF16 복원값이다. `eval_real.py`는 원본 구현과 같은 다음 경계를 재현한다.

- 일반 projection과 LM head: BF16 입력·가중치·출력
- mHC selector: BF16 GEMM 뒤 FP32 chart 계산
- router: 원본과 같은 FP32 linear
- gate/up: BF16 linear, SiTU는 FP32, down 직전에 BF16 반올림
- RMSNorm: FP32 통계 뒤 원래 BF16 경계로 복귀

이전 문서의 기능 수치는 모든 GEMM을 FP32로 실행한 오류판이었다. 그 수치는 채택 근거로 사용하면 안 된다. 수정 전 재구성 원본은 layer 1·24 router top-6이 97.14%, LM top-20이 평균 19.625/20이었다. 수정 후 router 세 층은 모두 35/35(100%), logit cosine은 각각 0.9999999999999976, 0.9999999999999967, 0.9999999999999962다. LM은 golden 생성 조건과 같은 1 thread 및 `torch.topk` tie 처리에서 16/16 step 모두 20/20이다.

## 재현 명령

저장소 최상위에서 실행한다. `python -m` 형태는 package import를 보장한다.

```bash
/usr/bin/time -v python -m quant.eval_real \
  --model-dir ./hf/Raon-1-mini-preview \
  --golden-dir ./golden-run/ko16 \
  --gguf ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --output-dir quant/eval_out \
  --threads 16 \
  --golden-threads 1

python -m quant.gguf_q4k \
  ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --inventory quant/eval_out/gguf_inventory.md \
  --json quant/eval_out/gguf_inventory.json

python -m quant.gguf_q4k \
  ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --proof quant/eval_out/q4k_proof.json \
  --proof-tensor blk.24.ffn_gate_exps.weight \
  --proof-blocks 4096

python -m pytest -q quant/test_q4nx.py quant/test_gguf_q4k.py quant/test_eval_e2e.py
```

## GGUF 구조와 이름 대응

실제 파일은 GGUF v3, `general.architecture=raon1`, 32바이트 정렬, data offset 12,048,512, 총 731 tensors다. 타입은 F32 321, F16 122, Q4_K 51, Q5_0 12, Q6_K 143, Q8_0 82개다. `g_proj`의 RDA/MLA 이름은 층 번호 범위로 추정하지 않고 실제 tensor set의 `ssm_g`/`attn_gate`로 판별한다. config를 쓰는 호출자는 1-based `rda_config.rda_layers`를 0-based로 바꿔 보조 정보로 넘긴다. 따라서 뒤쪽 RDA layer 4와 MLA layer 7 이상도 올바르게 대응한다.

## 실제 Q4_K proof

`blk.24.ffn_gate_exps.weight`의 4,096 super-block(1,048,576 values)을 대조했다.

| 검증 | 결과 |
|---|---:|
| 4-bit code | bit-exact |
| 최대 절대 복원 차이 | 0.0013492107 |
| 평균 절대 복원 차이 | 0.0001647569 |
| BF16 parameter rounding 최대 상계 | 0.0013720989 |
| 상계 초과(+2e-6 허용) | 0 |

Q4_K의 `d*sc*q-dmin*m`을 q4nx의 `bf16(-dmin*m)+bf16(d*sc)*q`로 옮기며, 관측 차이는 scale/min의 BF16 반올림 상계 안에 모두 들어간다.

## 경계 수정 후 결과

실제 활성값은 `golden-run/ko16`의 35 tokens, 전문가는 seed 20261028로 각 MoE 층에서 4개를 고정 표본 추출했다. 판정 기준은 cosine 0.995다.

| 측정 | min-max q4nx | GGUF 유래 q4nx |
|---|---:|---:|
| 표본 Q4_K 행렬 평균 cosine(26개) | 0.99694430 | 0.99709245 |
| dense MLP SiTU-GLU | 0.99634807 | 0.99723184 |
| dense MLP 최종 출력 | 0.99748312 | 0.99772336 |
| 최악 전문가 SiTU-GLU | 0.98987954 | 0.99086399 |
| 최악 전문가 최종 출력 | 0.98688915 | 0.98703105 |
| layer 24 MLA q_a→norm→q_b | 0.99489510 | 0.99465342 |
| LM head 행렬 cosine | 0.99697493 | 0.99681030 |
| LM logits cosine | 0.99882821 | 0.99876300 |
| LM argmax 일치 | 93.75% | 93.75% |
| LM top-20 평균/최소 | 19.25/18 | 19.4375/18 |
| LM KL 평균/최대 | 0.005676/0.050737 | 0.015466/0.088424 |

Router 자체는 설계상 양자화 대상이 아니며 전체 모델 평가에서는 원본 FP32로 유지한다. `eval_real.py`의 router q4 결과는 민감도 비교를 위한 별도 실험이다. 그 실험의 top-6 완전 일치율은 layer 1/3/24에서 80.00%/77.14%/71.43%다.

최종 재실행의 evaluator wall time은 142.73초, 최대 RSS는 2.49 GiB였다. 결과는 `quant/eval_out/results.json`, 사람이 읽는 표는 `quant/eval_out/results.md`에 생성된다. 이 산출물 디렉터리는 재생성 가능하므로 gitignore 대상이다.

## 전체 모델 평가 도구

`eval_e2e.py`는 한 번에 모델 하나만 올린다. 먼저 같은 모델에서 원본 지표를 수집하고, 대상 행렬을 하나씩 제자리 교체한 뒤 같은 inputs를 재평가한다. Router, mHC selector, norm, embedding은 교체하지 않는다. 결과 JSON에는 perplexity, 원본 대비 ratio, greedy 첫 불일치 위치와 첫 64/256 tokens 일치율, final-logit KL/top-1, 실제 교체된 parameter manifest, 그리고 토큰당 NPU 읽기 바이트 추정(routed expert는 6/128 반영)이 들어간다.

### 모델 전체 품질 실측 결과 (메인, 변환용 기계, 2026-10-28, 4,092 토큰 · 프롬프트 3개)

perplexity 는 다음 토큰 예측의 헷갈림 정도(낮을수록 좋음), 원본(HF MXFP8→BF16)은 27.684. 토큰당 NPU 읽기는 결과 JSON 의 `npu_read_bytes_per_token`(대상 행렬만, routed expert 6/128 반영). greedy 는 프롬프트 3개(한국어·영어·채팅) 256토큰 생성의 첫 불일치 위치(없음 = 256토큰 완전 일치).

| mode | 9비트(bfp16ebs8)로 두는 것 | perplexity | 원본 대비 | 토큰당 NPU 읽기 | greedy 첫 불일치(한·영·채팅) |
|---|---|---:|---:|---:|---|
| `minmax` | 없음(원본에서 min-max q4nx) | 29.125 | +5.20% | 2.27 GB* | 2 · 2 · 1 |
| `gguf_q4nx` | 없음(Q4_K 코드 보존 + 나머지 min-max q4nx) | 29.278 | +5.76% | 2.27 GB* | 0 · 2 · 1 |
| `mixed_lite` | 전문가 down·공유 전문가 down·출력층 | 28.738 | +3.81% | 2.84 GB | 없음 · 2 · 18 |
| `mixed_attn` | attention 투영(self_attn.*)의 비Q4_K | 28.521 | +3.02% | 3.01 GB | 0 · 2 · 1 |
| `mixed_full` | Q4_K 가 아닌 대상 전부 | **27.892** | **+0.75%** | 3.66 GB | 없음 · 2 · 32 |
| `gguf_q4km` (참고: KRAFTON 공식 Q4_K_M 그대로, NPU 형식 아님) | — | 27.910 | +0.82% | 2.94 GB* | 없음 · 2 · 30 |

\* 이 세 줄은 JSON 에 읽기 바이트 필드가 생기기 전 실행이라 같은 식으로 손계산한 값(q4nx 5비트, Q4_K_M 은 형식별 비트).

- **반올림이 결정적이다**: 9비트 가중치를 mlir-aie `bfp.py` 기본값인 내림(floor)으로 만들면 `mixed_full` 이 29.876(+7.92%)으로 4비트 전부보다 나빠진다 — 모든 가중치가 아래로 치우치기 때문. 최근접 짝수(conv_even, `--bfp-rounding` 기본값)로 만들면 27.892. 9비트 가중치는 오프라인으로 만드는 값이라 반올림 방식은 우리가 고른다(`floor` 는 하드웨어 실시간 변환용 기본값). `mixed_lite` 는 내림 28.669·최근접 28.738 로 차이가 작다.
- **효과는 두 묶음이 더해진다**: `gguf_q4nx`(+5.76%) 대비 lite 묶음(전문가 down·출력층)이 약 1.95점, attention 묶음이 약 2.74점을 줄이고, 둘 다(`mixed_full`) 하면 +0.75% — 어느 한 묶음만으로는 공식 품질에 못 미친다.
- `mixed_full` 은 공식 Q4_K_M 과 같은 급(+0.75% vs +0.82%)이고, 한국어 256토큰 생성이 원본과 완전히 같다.

모드 정의:

1. `mixed_full`: GGUF 에서 Q4_K 인 텐서 → 코드 보존 q4nx, Q4_K 가 아닌 대상 텐서(Q6_K·Q8_0·Q5_0 등) → GGUF 역양자화 값을 `bfp16ebs8`(입력 k 방향 8개 지수 공유)로.
2. `mixed_lite`: 전문가 down(`ffn_down_exps`)·공유 전문가 down(`ffn_down_shexp`)·출력층만 `bfp16ebs8`, 나머지는 `gguf_q4nx` 와 같음.
3. `mixed_attn`: `self_attn.*` 의 비Q4_K 텐서만 `bfp16ebs8`, 나머지는 `gguf_q4nx` 와 같음.

- **bfp16ebs8 반올림 구현**: mlir-aie 원본 (`./mlir-aie/python/utils/bfp.py` 146행 `quantize(x, rounding=...)`)의 수학적 정의를 의존성(`ml_dtypes`) 없이 순수 NumPy로 이식했다(floor·conv_even 둘 다, 원본과 비트 대조 시험). 블록 방향은 IRON `flm/dequant/README.md` 출력 배치(입력 k 방향 8개, 즉 가중치 행렬의 마지막 축 `shape[-1]`)와 일치한다. 큰 행렬은 4,096행씩 나눠 처리해 임시 메모리를 행렬 크기 수준으로 묶는다.
- **NPU 읽기 바이트 추정**: 결과 JSON의 `npu_read_bytes_per_token`에 대상 텐서 원소 수 × 비트 (q4nx 패킹 5비트 / 순수 코드 4비트, bfp16ebs8 9비트)와 routed expert의 활성화 비율(6/128)을 적용한 토큰당 메모리 읽기량을 기록한다.

### 실행 명령

메인이 준비한 `eval_text.txt`와 `eval_prompts.json`을 사용한 실행 예시는 다음과 같다. 각 모드는 메모리 격리를 위해 반드시 별도 process로 실행한다. 변환용 기계 실측에서 mixed 계열은 cgroup 사용이 80 GiB 근처까지 올라 두 번 OOM 으로 죽었다 — 84 GiB 상한과 `MALLOC_MMAP_THRESHOLD_=1048576`(큰 할당을 mmap 으로 해 해제 즉시 반환)으로 통과했다.

```bash
for mode in orig minmax gguf_q4km gguf_q4nx mixed_full mixed_lite mixed_attn; do
  systemd-run --user --scope -p MemoryMax=84G -p MemorySwapMax=0 \
    /usr/bin/env MALLOC_MMAP_THRESHOLD_=1048576 \
    python -m quant.eval_e2e \
      --mode "$mode" \
      --model-dir ./hf/Raon-1-mini-preview \
      --gguf ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
      --text eval_text.txt \
      --chunk-len 1024 \
      --chunks 4 \
      --prompts eval_prompts.json \
      --gen-len 256 \
      --threads 16 \
      --out "quant/eval_out/e2e-${mode}.json"
done
```

전체 모델은 BF16 복원 후 약 58 GiB이고 tokenizer, model objects, temporary tensor 한 개, logits/cache가 더해진다. 변환용 기계 실측(32 threads): 모드당 18~48분, 최대 RSS 74~85 GB(파일 매핑 포함).

## 결론

BF16 원본 재현 오류는 해소됐다. 순수 4비트(`minmax` +5.20%, `gguf_q4nx` +5.76%)는 KRAFTON 공식 판(+0.82%)에 미치지 못한다. NPU 가 읽을 수 있는 9비트 `bfp16ebs8` 을 **최근접 짝수 반올림으로** Q4_K 가 아닌 텐서 전부에 쓰면(`mixed_full`) +0.75% 로 공식 판과 같은 급이 된다. 대가는 토큰당 NPU 읽기 2.27 → 3.66 GB(약 1.6배). 중간안(`mixed_lite` +3.81%·2.84 GB, `mixed_attn` +3.02%·3.01 GB)은 품질 이득이 읽기 증가에 비해 작다. 어느 방식을 엔진에 쓸지는 사용자 결정 사항이다(품질·속도 절충).
