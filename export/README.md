<!-- Copyright (C) 2026 raon-npu authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `export/` — Raon-1-mini D mode 내보내기

KRAFTON `Raon-1-mini-preview` 의 혼합 정밀도 가중치를 노트북(NPU) 실행 환경에 맞게 내보내는 도구.

- `export/export_d.py` : 전체 / 부분 내보내기
- `export/verify_d.py` : `manifest.json` + GGUF 원본 대조 검증
- `export/test_export_small.py` : 단위·통합 시험

## 출력 폴더 구조

```
<out_dir>/
├── manifest.json                       # 전체 묶음 메타데이터 + SHA256
├── model.safetensors.index.json        # host 텐서 이름 -> 파일 매핑 (host/weights.py 용)
├── host_global.safetensors             # 임베딩·output_norm 등 전역 host 가중치
├── lm_head.bin                         # 어휘 투영 (260096×2560 9-bit bfp16 GEMV)
└── L00/ .. L24/
    ├── host.safetensors                # 해당 층의 norm/router/mHC/conv1d 등 host 가중치
    ├── rda_in.bin | mla_in.bin         # 어텐션 입력 투영
    ├── f_b.bin | q_b.bin / k_b.bin / v_b.bin / g.bin
    ├── o_proj.bin
    ├── dense_gu.bin | moe_b1.bin       # Dense/MoE gate+up (q4nx GEMV)
    ├── dense_down.bin | moe_b2.bin     # Dense/MoE down (9-bit bfp16 GEMV)
    └── shared_gu.bin                   # MoE shared expert gate+up
```

`*.bin` 은 NPU 레이아웃 바이너리, `*.safetensors` 는 host CPU 가중치.

## `manifest.json` 형식

```json
{
  "version": "1.0",
  "model": "Raon-1-mini-preview",
  "mode": "mixed_full_d",
  "created_at": "...",
  "num_layers_total": 25,
  "exported_layers": [0, 1, ..., 24],
  "total_size_bytes": 26131993360,
  "total_size_mb": 24921.41,
  "total_size_gb": 24.337,
  "bundles": [
    {
      "bundle": "L00/rda_in",
      "layer": 0,
      "format": "bfp16_gemv",
      "file": "L00/rda_in.bin",
      "size_bytes": 59719680,
      "original_shape": [20688, 2560],
      "padded_shape": [20736, 2560],
      "components": [
        {
          "name": "model.layers.0.self_attn.q_proj.weight",
          "gguf_name": "blk.0.attn_q.weight",
          "short_name": "q",
          "row_start": 0,
          "row_end": 5120,
          "shape": [5120, 2560],
          "gguf_type": "Q6_K"
        }
      ],
      "gguf_types": ["Q6_K", "F16"],
      "sha256": "...",
      "description": "RDA input projection bundle [q; k; v; g; f_a; b; rda_w]"
    }
  ]
}
```

### 최상위 키

| 키 | 의미 |
|---|---|
| `version` | manifest 스키마 버전 |
| `model` | 모델 이름 |
| `mode` | `mixed_full_d` |
| `created_at` | ISO 8601 생성 시각 |
| `num_layers_total` | 전체 층 수 (25) |
| `exported_layers` | 현재 manifest 에 포함된 층 번호 목록 |
| `total_size_bytes` / `total_size_mb` / `total_size_gb` | 모든 bundle 파일 크기 합 |
| `bundles` | 묶음 항목 배열 |
| `host_tensors_status` | host 내보내기 실패 시 `status: "failed"`, `error`, `note` — 이미 내보낸 묶음은 `bundles` 에 보존되고 예외는 다시 올라옴 |

### 묶음 항목 키

| 키 | 의미 |
|---|---|
| `bundle` | 묶음 식별자, 예 `L00/rda_in`, `host_global` |
| `layer` | 층 인덱스, 전역 묶음은 `null` |
| `format` | `bfp16_gemv`, `q4nx_gemv`, `bfp16_gemv_bundle`, `q4nx_gemv_bundle`, `host_safetensors` |
| `file` | `manifest.json` 기준 상대 경로 |
| `size_bytes` | 파일 크기 |
| `original_shape` | 원본 텐서 모양 (패딩 전) |
| `padded_shape` | NPU 레이아웃용 0-패딩 후 모양 |
| `components` | 이 묶음에 들어간 원본 텐서 설명 |
| `gguf_types` | 사용된 GGUF 양자화 타입 |
| `sha256` | 파일 SHA256 |
| `description` | 사람이 읽는 설명 |
| `expert_count` / `expert_bytes` | MoE bundle 전용 |

### 구성요소(component) 키

일반 NPU 투영:

- `name`: HF 이름, 예 `model.layers.0.self_attn.q_proj.weight`
- `gguf_name`: GGUF 원본 이름, 예 `blk.0.attn_q.weight`
- `short_name`: 묶음 내 축약명 (`q`, `k`, `v`, `g`, ...)
- `row_start` / `row_end`: 묶음 내 행 범위
- `shape`: 원본 모양
- `padded_shape`: 패딩된 모양 (f_b, q_b 등)
- `gguf_type`: GGUF 양자화 타입

3차원 MoE 묶음 (`moe_b1` / `moe_b2`) 구성요소:

- `name` / `gguf_name`: 원본 텐서 설명 (라우트드 expert 묶음과 shared expert 는 항목이 나뉨)
- `shape_per_expert`: expert 하나의 원본 모양, moe_b1 은 `[2176, 2560]`, moe_b2 는 `[2560, 1088]`
- `padded_shape_per_expert`: expert 하나당 패딩 후 모양 (moe_b2 `[2560, 1280]`)
- `expert_range`: 이 항목이 덮는 expert 번호 범위, 예 `[0, 128]`
- `expert_id`: 단일 expert 항목의 번호, shared expert 는 `128`
- `gguf_type`: GGUF 양자화 타입 (`Q4_K`, `Q8_0` 등)

`k_b` / `v_b` 는 3차원 GGUF 형태를 2차원으로 펼친 정보를 추가 기록:

- `raw_gguf_shape`: GGUF 원본 모양, k_b 는 `[40, 512, 128]`, v_b 는 `[40, 128, 512]`
- `hf_interleave_rule`: HF `kv_b_proj.weight` 안에서 머리별 위치, 예
  - k_b: `"Head h=0..39 keys occupy rows [h*256 : h*256+128] in HF kv_b_proj.weight"`
  - v_b: `"Head h=0..39 values occupy rows [h*256+128 : (h+1)*256] in HF kv_b_proj.weight"`

padding 전용 항목:

- `name`: `"padding"`
- `row_start` / `row_end`
- `shape`
- `gguf_type`: `"ZERO_PAD"`

host `safetensors` 구성요소:

- `name`: 텐서 이름
- `shape`
- `dtype`: 원본 HF dtype (`bfloat16`, `float32`, ...)
- `stored_dtype`: 디스크 safetensors 헤더 dtype (`BF16`, `F32`, `F16`, `I64`, ...)

## 실행 명령

전체 내보내기:

```bash
python3 -m export.export_d \
  --gguf ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --hf   ./hf/Raon-1-mini-preview \
  --out  ./weights-d
```

부분 내보내기:

```bash
# 층 0,1,2 만 host 포함
python3 -m export.export_d --out /tmp/partial --layers 0,1,2

# 특정 묶음만
python3 -m export.export_d --out /tmp/partial --only lm_head,host

# host 를 제외하고 lm_head 만
python3 -m export.export_d --out /tmp/partial --only lm_head,host --skip-host
```

부분 실행은 기존 `manifest.json` 과 `model.safetensors.index.json` 와 병합하므로,
기존 출력 폴더에서 덧붙여 실행할 수 있다.

기존 출력의 host 헤더만 수리 (`*.bin` 은 건드리지 않음):

```bash
python3 -m export.export_d --fix-host-only ./weights-d
```

이 명령은

1. `manifest.json` 에서 `format == "host_safetensors"` 인 항목만 선택
2. 실제 파일 헤더를 읽어 `dtype == "bfloat16"` 인 텐서가 헤더에 `BF16` 로 저장돼 있는지 검사  
   (`manifest` 값이 아니라 실제 파일 헤더를 기준으로 판단한다)
3. 헤더가 예전 `U16` 인 텐서(manifest 와 모양이 같을 때만)는 동일한 비트 패턴을 유지한 채 `BF16` 로 재기록하고,  
   데이터 비트는 그대로 유지하며 텐서별 비교로 검증. `F32` 등 그 밖의 헤더는 변환하지 않고 파일을 건드리기 전에 실패
4. 모든 host 파일의 `manifest.json` `stored_dtype`, `size_bytes`, `sha256` 를 실제 파일 기준으로 갱신
5. 최상위 `total_size_bytes`·`total_size_mb`·`total_size_gb` 를 묶음 `size_bytes` 합으로 재계산
6. `model.safetensors.index.json` 재생성·병합

검증:

```bash
python3 -m export.verify_d \
  --manifest ./weights-d/manifest.json \
  --gguf   ./hf/gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --all
```

`verify_d` 는 마지막에 두 줄로 요약한다:

```
[Verifier] Integrity:  26 checked, 26 passed, 0 failed
[Verifier] Bit-exact: 171 checked, 171 passed, 0 failed
```

host 묶음은 무결성(host_integrity) 줄에, NPU `.bin` 묶음은 비트 대조(npu_bit_exact) 줄에 따로 집계된다.

## 크기 표

메인 실측 (`./weights-d/manifest.json`, 197개 묶음, 최상위 합계 = 묶음 `size_bytes` 합 `total_size_bytes` 26,131,993,360 B = `total_size_mb` 24,921.41 MB = `total_size_gb` 24.337 GB):

### 포맷별

| 포맷 | 묶음 수 | 크기 | 비고 |
|---|---:|---:|---|
| `bfp16_gemv` | 122 | 2.44 GB | 단일 행렬 9-bit bfp16 (rda_in, mla_in, o_proj, q_b/k_b/v_b/g, shared_gu, dense_down, lm_head 등) |
| `bfp16_gemv_bundle` | 24 | 10.63 GB | MoE down (129 experts × 2560×1280) |
| `q4nx_gemv` | 1 | 0.036 GB | L00 dense gate+up |
| `q4nx_gemv_bundle` | 24 | 9.96 GB | MoE gate+up (128 routed experts × 2176×2560) |
| `host_safetensors` | 26 | 1.27 GB | 25개 층 host 파일 + host_global |
| **합계** | **197** | **24.34 GB** | 최상위 `total_size_bytes` = 묶음 `size_bytes` 합 26,131,993,360 B |

### 층 역할별 (NPU 묶음만)

| 분류 | 묶음 예 | 전체 크기 |
|---|---|---:|
| L00 RDA + dense MLP (5 묶음) | rda_in, f_b, o_proj, dense_gu, dense_down | 0.14 GB |
| RDA + MoE 층 × 17 (층당 6 묶음, 총 102) | rda_in, f_b, o_proj, moe_b1, moe_b2, shared_gu | 15.89 GB |
| MLA + MoE 층 × 7 (층당 9 묶음, 총 63) | mla_in, q_b, k_b, v_b, g, o_proj, moe_b1, moe_b2, shared_gu | 6.34 GB |
| lm_head (1 묶음) | lm_head | 0.70 GB |
| host (26 묶음) | L00/host .. L24/host, host_global | 1.27 GB |

## 시험

```bash
python3 -m pytest -q export quant -p no:cacheprovider
```

`export/test_export_small.py` 에서는

- q4nx / bfp16 GEMV 왕복
- RDA/MLA 형태 변환 및 interleave 규칙
- host safetensors `BF16` 헤더 + 비트 불변
- `model.safetensors.index.json` 가 모든 host 텐서를 커버
- host 내보내기 실패 시에도 실패 전에 내보낸 묶음(예: `L00/dense_down`)과 `host_tensors_status` 가 `manifest.json` 에 보존되고 예외가 다시 올라옴
- `--fix-host-only` 가 `U16` 헤더를 `BF16` 로 바꾸고 데이터 비트 불변, 최상위 `total_size_*` 를 묶음 `size_bytes` 합으로 재계산

을 검증한다.
