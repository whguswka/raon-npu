<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon NPU 서버

Windows에서 NPU 생성 엔진(`serve.engine`)을 Ollama·OpenAI 호환 HTTP 서버로 엽니다. 기본 수신 주소는 `127.0.0.1`입니다. 인증·TLS가 없으므로 LAN 개방은 믿을 수 있는 망에서만 하세요.
## 실행

### PowerShell (`serve/run-serve.ps1`)

`run-serve.ps1` 는 노트북(Windows) 전용이다. 가중치 경로는 `-WeightsD` 또는 `RAON_WEIGHTS_D`로 지정합니다.

```powershell
# 일반 실행 (기본 포트 11434, keep-alive 300초)
serve\run-serve.ps1 -WeightsD .\weights-d

```

- 기본 포트: `11434`
- 가중치: `-WeightsD` 또는 `RAON_WEIGHTS_D`; 토크나이저는 같은 폴더 (`-ModelDir`로 변경 가능)
- 기본 요청 로그 경로: `$env:LOCALAPPDATA\raon-npu\logs\serve.log` (`-Log`)
- 기본 Keep-Alive: `300`초 (`-KeepAlive`)
- **콘솔 출력**: 서버 기동 메시지, 표준 출력, 오류(예: 포트 bind 실패 등)는 콘솔(터미널)에 직접 출력된다. 래퍼(`run-serve.ps1`)는 파일 리디렉션을 하지 않고 서버를 전경에서 실행하여 Python 자식의 종료 코드를 그대로 돌려준다. (백그라운드 상시 실행은 노트북 작업 스케줄러의 `serve-task.cmd`를 사용하며, stdout·stderr를 각각 `serve.out`·`serve.err`로 따로 보낸다.)
- **요청 로그**: HTTP 요청 및 응답 처리에 관한 로그는 서버 프로세스가 자체 `--log` 옵션으로 지정된 파일(`serve.log`, UTF-8)에 직접 기록한다. 래퍼가 파일에 출력을 덧쓰지 않으므로 PowerShell 5.1(기본 UTF-16)과의 인코딩 충돌 없이 UTF-8로 단일 유지된다.
- 포트 사용 여부는 서버의 배타 bind가 결정하며, 이미 점유된 포트인 경우 서버가 bind 실패로 종료되고 래퍼는 그 종료 코드를 그대로 돌려준다.

### 직접 (Python)

```powershell
$RepoRoot = (Resolve-Path ".").Path
$env:PYTHONPATH = "$RepoRoot;C:\dev\IRON"
$env:PYTHONUNBUFFERED = "1"
python -m serve.server --host 127.0.0.1 --port 11434 --weights-d C:\dev\raon\weights-d --model-dir C:\dev\raon\weights-d --keep-alive 300 --log C:\dev\raon\logs\serve.log
```

## 주요 기능 (1차 보완 반영)

1. **생각(`<think>`)과 답 분리**:
   - 생성 프롬프트가 `<think>` 로 끝났으면 모델 출력에서 `</think>`(토큰 `259214`) 앞은 생각, 뒤(앞쪽 공백 줄 제거)는 답으로 나눈다.
   - OpenAI `/v1/chat/completions`: 비스트리밍은 `message.reasoning_content` 와 `message.content`, 스트리밍(SSE)은 `delta.reasoning_content` 와 `delta.content` 로 전달한다.
   - Ollama `/api/chat`: `message.thinking` 과 `message.content` 로 전달한다.
   - 생각 끄기: OpenAI 요청의 `chat_template_kwargs.enable_thinking=false` 또는 `reasoning_effort="none"`, Ollama 요청의 `think=false` 지정 시 엔진에 `enable_thinking=False`(템플릿이 `<think></think>\n\n` 삽입)를 전달한다. 기본값은 생각 켬이다.
   - `max_tokens`(`num_predict`)가 생각 안에서 끝나면 `content` 는 빈 문자열(`""`)이고 `finish_reason`(또는 `done_reason`)은 `"length"` 가 된다.

2. **유휴 시 메모리 회수 (`--keep-alive` / `keep_alive`)**:
   - 마지막 요청 처리 완료 후 `--keep-alive`(기본 `300`초; `0`이면 즉시 해제, 음수면 계속 유지)가 지나면 잠금(`_busy_lock`) 안에서 엔진의 NPU 가중치 텐서와 호스트 가중치를 내려놓고 `gc.collect()` 를 수행해 메모리(약 25 GB)를 돌려준다.
   - 생성(요청 처리) 중에는 절대 내려놓지 않는다.
   - 내려놓은 뒤 다음 요청이 오면 잠금 안에서 다시 적재하고 처리한다.
   - 요청 본문의 `keep_alive` 칸(초 또는 `"5m"`, `"0"`, `"-1"` 등)도 지원한다.

3. **대화 이어 쓰기 지점 (프롬프트 체크포인트 재사용)**:
   - 대화 템플릿은 이전 턴의 assistant 메시지를 다시 그릴 때 생각을 지우고 `<think></think>\n\n`+답으로 그리므로, 턴 1의 전체 토큰열(프롬프트+생각+답)과 턴 2의 프롬프트가 달라진다.
   - 엔진은 각 요청에서 마지막 생성 프롬프트 토큰(`<|im_start|>assistant\n<think>\n` 부분)을 뺀 접두사 처리 직후의 상태(RDA 상태 18층·합성곱 상태 18층·MLA 캐시 길이 7층)를 백엔드 `get_*`/`set_*` 계약으로 한 벌 저장해 둔다.
   - 다음 요청이 이전 전체 토큰열로 시작하면 그대로 이어 쓰고, 그렇지 않더라도 저장해 둔 프롬프트 접두사로 시작하면 저장 지점을 복원해 이어 처리한다. 맞지 않으면 처음부터 다시 처리한다.

4. **실제 모델 메타데이터 노출 (`/api/tags`, `/api/show`, `/api/ps`)**:
   - `format`: `"raon-npu"`
   - `quantization_level`: `"D: q4nx + 9-bit bfp16"`
   - `parameter_size`: `"28.8B"`
   - `size`: `manifest.json` 의 `total_size_bytes` (`26,131,993,360` bytes)

5. **표본추출 옵션 경계값 (`top_p` 등)**:
   - `top_p`: `0.0 <= top_p < 1.0`에서 nucleus 필터링을 적용하며, `top_p=0.0`이면 최고 점수 1개 토큰만 남겨 greedy(argmax)로 동작하고, `top_p=1.0`이면 필터링을 건너뛴다.

## 엔드포인트

| 메서드 | 경로 | 설명 | 비고 |
|---|---|---|---|
| GET | `/healthz` | 서버 생존 및 엔진 적재 상태 | 항상 200, `{"status":"ok", "loaded": true/false}` |
| GET | `/api/version` | Ollama 버전 | `{"version":"0.5.13"}` |
| GET | `/api/tags` | Ollama 모델 목록 | `raon-1-mini:latest`, `manifest.json` 크기 및 D 양자화 정보 |
| POST | `/api/show` | 모델 상세 정보 | `raon-1-mini` / `raon-1-mini:latest` |
| GET | `/api/ps` | 현재 적재된 모델 및 만료 시각 | 적재 중이면 모델 정보와 `expires_at`, 내려놓았으면 `{"models":[]}` |
| POST | `/api/chat` | Ollama 채팅 | NDJSON 스트리밍/단일, `message.thinking`·`message.content`, `think`, `keep_alive` 지원 |
| POST | `/api/generate` | Ollama 텍스트 생성 | NDJSON 스트리밍/단일, `prompt`, `raw`, `stream`, `keep_alive` 지원 |
| GET | `/v1/models` | OpenAI 모델 목록 | `raon-1-mini` |
| POST | `/v1/chat/completions` | OpenAI 채팅 | SSE/단일, `reasoning_content`·`content`, `chat_template_kwargs.enable_thinking`, `reasoning_effort` 지원 |
| POST | `/v1/completions` | OpenAI 텍스트 생성 | SSE/단일 |

## 시험

```bash
python -m pytest -q serve -p no:cacheprovider
```

실제 토크나이저·골든 manifest 를 쓰는 시험은 환경 변수로 경로를 받으며(host 의 `check_golden.py` 와 같은 변수), 없으면 이유를 밝히고 skip 한다.

| 환경 변수 | 용도 |
|---|---|
| `RAON_MODEL_DIR` | HF 체크포인트 디렉터리(`tokenizer.json` 포함). 실제 토크나이저 시험 |
| `RAON_GOLDEN_DIR` | 골든 실행 디렉터리(`manifest.json` 포함). 골든 prompt 토큰 시험 |
| `RAON_WEIGHTS_D` | D 가중치 디렉터리. `/api/tags` 의 크기를 읽을 때 `manifest.json` 후보로 쓴다(선택) |

## 요청 매개변수 정책 (Request Parameter Policy)

각 엔드포인트는 공식 API 문서 기준 매개변수를 다음 세 범주로 엄격히 분류하여 처리한다:

1. **지원 (Supported)**:
   - 읽고 실행에 실제로 반영한다.
   - 예: `model`, `messages`, `prompt`, `stream`, `stream_options.include_usage`, `stop`, `tools`, `keep_alive`, `max_tokens`(`max_completion_tokens`), `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `enable_thinking`, `chat_template_kwargs`, `reasoning_effort` 등. (단, `/v1/completions`는 채팅 템플릿을 쓰지 않으므로 `enable_thinking`과 `chat_template_kwargs`는 미지원이며, Ollama `/api/generate` 또한 `think`를 지원하지 않는다.)
   - `reasoning_effort`(OpenAI 채팅)는 `"none"`(생각 끔)과 `"medium"`·`null`(모델 기본)만 지킨다. 이 모델에는 생각 양 조절이 없어 `"minimal"`·`"low"`·`"high"` 는 400(`unsupported_parameter`), 그 밖의 값은 400(`invalid_value`)이다.
   - `/v1/completions`의 `prompt`는 단일 문자열 또는 원소가 1개인 문자열 배열(`["..."]`)만 지원하며, 다중 배치 프롬프트(`[]` 또는 2개 이상)는 400 오류(`batch prompts are not supported`)로 거부된다.

2. **무시해도 뜻이 같음 (Ignored)**:
   - 과금·기록·캐시·라우팅이나 하드웨어·자원 제어에만 관계된 칸이라, 무시해도 답의 내용·형식·샘플링이 같다.
   - 대상:
     - 과금·기록·캐시·라우팅: `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier`, `stream_options.include_obfuscation`.
     - Ollama 하드웨어·자원 칸: `num_ctx`, `num_batch`, `num_gpu`, `main_gpu`, `use_mmap`, `num_thread`, `numa`, `low_vram`, `f16_kv`, `use_mlock`, `num_keep`, `vocab_only`.
     - 기본값 시 효과가 없는 칸: `repeat_last_n`, `penalize_newline`, `mirostat_tau`, `mirostat_eta`.
   - **공식 목록 밖 확장 칸**: 게이트웨이(예: API gateway)나 비표준 클라이언트가 추가하는 공식 목록 외의 임의 확장 칸은 호환성을 위해 무시된다.

3. **미지원 (Unsupported)**:
   - 서버가 지원하지 않으나 결과의 내용·형식·개수·샘플링을 바꾸는 매개변수는 "뜻 없는 값(No-op)"이 아닌 값이 전달될 경우 400 오류(`code="unsupported_parameter"`)로 즉시 거부한다.
   - "뜻 없는 값" 허용:
     - `n`, `best_of`: `1`, `null`
     - `echo`: `false`, `null`
     - `logprobs`: `false`, `0`, `null`
     - `top_logprobs`: `0`, `null`
     - `logit_bias`: `{}`, `null`
     - `frequency_penalty`: `0`, `null`
     - `response_format`: `{"type": "text"}`, `null`
     - `modalities`: `["text"]`, `null`
     - `verbosity`: `"medium"`, `null`
     - `audio`, `prediction`, `web_search_options`, `functions`, `function_call`: `null`
     - `suffix`, `format`, `template`, `system`: `""`, `null`
     - `images`, `context`: `[]`, `null`
     - `tool_choice`: `"auto"`, `null`
     - `parallel_tool_calls`: `true`, `null`
     - `/v1/completions`의 `enable_thinking`, `chat_template_kwargs`: `null`
     - Ollama `/api/generate`의 `think`: `null`
     - Ollama `repeat_penalty`, `typical_p`, `tfs_z`: `1.0`, `null`
     - Ollama `min_p`, `frequency_penalty`, `mirostat`: `0`, `null`
   - Ollama `/api/generate`의 `system`은 빈 문자열 또는 `null`이 아닌 값이 전달되면 400 오류로 거부된다.
   - Ollama 메시지 내 `images`는 지원되지 않으며 스키마 검증에서 거부된다.
