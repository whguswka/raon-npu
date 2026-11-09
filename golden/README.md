<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon-1-mini CPU 골든 도구

이 디렉터리는 AMD XDNA 2 NPU 커널과 대조할 CPU 기준값을 만든다. `dump.py`는 로컬 체크포인트를 plain BF16 모델로 읽어 prefill과 greedy decode를 실행한다. MXFP8 체크포인트는 일반 FP8 quantizer로 해석하지 않고, shard를 하나씩 열어 routed expert의 E4M3 `weight`와 E8M0 `weight_scale_inv`를 원본 수식으로 BF16 변환한다. `rda_bf16_drift.py`는 RDA의 fp32 상태와 매 토큰 갱신 후 BF16으로 반올림한 상태의 발산을 측정한다.

실제 28.8B 체크포인트 실행은 테스트 범위가 아니다. pytest는 축소 무작위 모델과 합성 MXFP8 safetensors 체크포인트만 사용한다.

## 프롬프트와 단계 번호

`--prompt-file`은 UTF-8 텍스트, 채팅 메시지 JSON 배열, 또는 `{"input_ids": [...]}` 형식이다. 채팅 배열에는 `--chat-template`을 함께 사용한다.

`--decode-steps N`은 생성 토큰 수다. 생성 단계 번호는 CLI, manifest, 파일명 모두 1-based다. `step_000001.npz`가 prompt prefill로 첫 토큰을 고른 결과이고, 2단계부터 직전 생성 토큰 하나를 캐시에 이어 넣는다. 기본 전체 상태 snapshot은 생성 단계 1, 16, 64, 마지막이다. `--state-steps 1,8,32`처럼 바꿀 수 있다.

## 골든 덤프

저장소 최상위에서 실행한다.

```bash
python3 golden/dump.py \
  --model-dir ./hf/Raon-1-mini-preview \
  --code-dir ./code \
  --prompt-file prompt.txt \
  --decode-steps 64 \
  --out dumps/raon-fixed-64 \
  --dtype bf16 \
  --layers all \
  --max-gb 80
```

각 step NPZ에는 선택한 층의 decoder stream 입·출력, mixer 입·출력, MoE router 결과, logits top-20, 선택 시점의 RDA state 또는 MLA latent cache가 들어간다. RDA 스냅숏은 fp32 재귀 상태 `layer_XXX.rda_state`와 짧은 합성곱(커널 K)의 직전 K-1토큰 raw(pre-conv) q|k|v 행 `layer_XXX.rda_conv_state`(`cache.conv_states[idx]`, `[B, 3*proj, K-1]`)를 함께 담는다. 상태 스냅숏은 그 단계의 토큰을 처리한 **뒤**의 상태다 — modeling의 층 forward가 `(conv_states|recurrent_states)[idx]`를 그 단계 토큰까지 반영해 갱신하고 `dump.py`는 같은 단계 `model(...)` 반환 뒤 캐시를 읽으므로, 곧 다음 단계 시작 상태이며 이어 돌리기 시험은 다음 단계 입력 토큰부터 이 값으로 이어 돌리면 된다. MLA는 원본 `kv_a_proj_with_mqa` 출력을 나눈 뒤 실제 `kv_b_proj` 입력에 쓰이는 RMSNorm 후 `kv_lat`와 공유 RoPE key `k_pe`를 누적한다. 크기는 BF16 기준 `7층 × 576 × 2 B × 토큰 수`다. 선택 시점에는 전체 cache를 NPZ에 저장하고, 다른 단계에는 shape·norm·SHA-256을 manifest에 남긴다. 헤드별로 펼친 key/value도 필요하면 `--expanded-kv-cache`로 별도 기록할 수 있다. 전체 logits는 manifest에 norm과 원시 dtype 바이트 SHA-256만 남긴다. BF16 배열은 NPZ에서 `uint16` 비트열로 저장하고 `__dtypes_json__`에 원래 dtype을 기록한다.

`--max-gb`는 각 NPZ를 쓰기 전에 배열 `nbytes`와 컨테이너 여유를 예약한다. 임시 파일로 쓴 뒤 원자적으로 교체하며 실패한 임시 파일은 제거한다. `actual_dump_bytes`에는 마지막 `manifest.json` 크기도 포함된다.

기본 `--model-hash full`은 JSON, Jinja, safetensors/bin 파일 내용을 해시한다. 큰 체크포인트에서 파일명과 크기만 확인하려면 `--model-hash metadata`를 사용한다.

## RDA 입력과 BF16 상태 발산

```bash
python3 golden/dump.py \
  --model-dir ./hf/Raon-1-mini-preview \
  --code-dir ./code \
  --prompt-file prompt-32k.json \
  --decode-steps 0 \
  --capture-rda-inputs \
  --out dumps/rda-inputs-32k \
  --max-gb 64

python3 golden/rda_bf16_drift.py \
  --input dumps/rda-inputs-32k \
  --positions 128,512,2k,8k,32k \
  --out dumps/rda-inputs-32k/drift.json
```

합성 입력만으로 경로를 점검할 수도 있다.

```bash
python3 golden/rda_bf16_drift.py \
  --synthetic --tokens 512 --heads 4 --dim 32 \
  --positions 128,512 --out /tmp/rda-drift.json
```

## 시험

```bash
python3 -m pytest -q golden
```

`RAON_SOURCE_DIR`로 원본 코드 디렉터리를 바꿀 수 있으며 지정하지 않으면 원본이 필요한 시험은 건너뛴다. 테스트는 합성 MXFP8→BF16 비트 일치와 logits, padding 및 `K>Q` mask, 훅 전후 logits/cache 비트, 1-based snapshot, RDA 합성곱 상태(`rda_conv_state`) 스냅숏, 쓰기 전 예산 차단, 다중 토큰 RDA 상태 반영을 검사한다.
