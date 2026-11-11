# raon-npu

[English](README.md)

AMD Ryzen AI XDNA 2 NPU에서 KRAFTON Raon-1-mini-preview(30B-A4B MoE)를 NPU로만 추론하는 엔진입니다. Ollama·OpenAI 호환 서버를 제공합니다. KRAFTON·AMD와 관계없는 비공식 개인 프로젝트이며 변환 가중치는 게시하지 않습니다.

## 상태와 측정값

측정 기계는 **AMD Ryzen AI 7 350(XDNA 2), 메모리 64 GB, Windows 11 Home build 26300**입니다. NPU 드라이버 32.0.203.314, XRT SDK 2.21.75, mlir-aie v1.4.4, llvm-aie 22.0.0.2026092101, IRON b3ba3cf3, Python 3.13.15를 사용했습니다. 아래는 2026-11-09 운영판 실측이며 다른 환경의 성능을 보장하지 않습니다.

| 항목 | 실측 |
|---|---|
| 445토큰 프롬프트 prefill | 약 14.5초(약 31 tokens/s) |
| decode | 약 7 tokens/s(130~145 ms/token) |
| 시험한 긴 프롬프트 | 약 6,700토큰 |
| 가중치 적재 시 메모리 | 약 25 GB |
| 첫 요청 가중치 적재 | 약 20~36초 |
| 유휴 후 다시 적재 | 약 16~20초 |
| 변환 묶음 | 26,131,993,360 bytes |
| 첫 커널 빌드 시간 | 약 4~5분(새 체크아웃의 첫 selfcheck 257초, 이후 실행 약 1분) |

## 요구 사항

Windows 11, AMD Ryzen AI 300 계열 XDNA 2 NPU, **메모리 64 GB**가 대상입니다. 위 조합만 시험했습니다. 다른 XDNA 2 칩, 32 GB 메모리, Linux는 확인하지 않았습니다. XDNA 1(Ryzen 7040·8040)은 지원하지 않습니다.

설치 전 준비:

- Git for Windows: `winget install Git.Git` 실행 후 PowerShell을 다시 엽니다.
- [python.org](https://www.python.org/downloads/windows/)의 **Python 3.13 64비트**를 설치하고 PATH 추가를 선택합니다. Microsoft Store 판은 사용하지 않습니다.
- 3단계(가중치 변환)에만 쓰는 **Python 3.12 64비트**를 python.org에서 하나 더 설치합니다. 설치 화면의 "Add python.exe to PATH"는 체크하지 않아 `python`이 3.13으로 남게 합니다(도구 설치가 3.13을 요구합니다). 3단계에서 `py -3.12`로 부릅니다. 변환을 다른 기계에서 하면 그 기계에 설치합니다.
- NPU 드라이버: 장치 관리자에 **NPU Compute Accelerator Device**가 표시되어야 합니다. 32.0.203.314에서 시험했습니다. 이 판 이상을 사용하되 더 새 판은 별도 확인이 필요합니다.
- 도구 설치에는 관리자 권한이 필요합니다. 서버는 일반 사용자로 실행합니다.

변환 기계에는 **원본 safetensors 32.7 GB + GGUF 19.7 GB**, **변환 출력 26.1 GB**, 도구와 임시 파일 공간이 필요합니다. 노트북에는 **가중치 26.1 GB + 빌드 폴더 약 0.5 GB**(파일 약 7,500개, xclbin 64개)와 도구 공간이 필요합니다. 다운로드·압축 해제를 위한 여유 공간도 확보하세요.

| 환경 | Python | 실행 및 의존성 |
|---|---|---|
| NPU 엔진 | 3.13, mlir-aie IRON 환경 | `scripts\run.cmd`, `requirements-npu.txt` |
| 변환 | 별도 Python 3.12 가상 환경 또는 다른 기계 | `python`, `requirements-convert.txt` |

**변환에는 IRON이 필요 없습니다.** 변환기의 import는 `quant/`의 CPU 구현을 사용합니다. 변환 의존성은 Python 3.12.2에서 쓴 정확한 판으로 고정했습니다. Python 3.13용 휠이 없으므로 NPU 환경에 설치하지 마세요. 엔진 실측 환경은 numpy 2.5.4, ml_dtypes 0.6.0, mlir-aie 1.4.4, llvm-aie 22.0.0.2026092101+0006955e, tokenizers 0.22.2, Jinja2 3.1.6입니다. 앞의 네 패키지는 `iron_setup.py`가 설치합니다.

| 가중치 폴더를 지정하는 곳 | 이름 |
|---|---|
| 서버·selfcheck CLI | `--weights-d` |
| PowerShell 스크립트 | `-WeightsD` |
| 환경 변수 | `RAON_WEIGHTS_D` |

## 설치 5단계

### 1. 저장소 받기

```powershell
git clone https://github.com/whguswka/raon-npu.git raon-npu
cd raon-npu
```

### 2. 도구 설치

저장소 안에서 **관리자 PowerShell**로 실행합니다.

```powershell
.\scripts\setup-toolchain.ps1
# 설치 위치를 바꾸려면:
# .\scripts\setup-toolchain.ps1 -Root D:\NpuTools
```

Windows는 기본적으로 PowerShell 스크립트 실행을 막을 수 있습니다. "running scripts is disabled" 오류가 나면 한 번 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 실행하거나, 각 스크립트를 `powershell -ExecutionPolicy Bypass -File <스크립트>`로 실행합니다.

Visual Studio Build Tools, XRT 2.21.75(SHA256 확인), mlir-aie v1.4.4, IRON 커밋 `b3ba3cf3a138c21e021d5ee2d6794573cd7a3287`을 설치·확인합니다. IRON 환경에 `requirements-npu.txt`를 설치하고 SAXPY를 실행합니다. 기본 위치는 `C:\dev\mlir-aie`, `C:\dev\IRON`, `C:\Xilinx\XRT`입니다. 로그는 `<Root>\raon-setup.log`에 남깁니다. 기존 설치 판이 다르면 체크아웃을 자동 변경하지 않고 멈춥니다. XRT 판 정보 파일이 없으면 판을 확인하지 못했다고 알리고 `pyxrt` 존재만 검사합니다.

설치 위치를 바꿨다면 실행 셸마다 `RAON_MLIR_AIE_DIR`·`RAON_IRON_DIR`를 설정하세요. 로그온 작업 등록 전에는 사용자 환경 변수에도 저장해야 합니다.

### 3. 가중치 받기·변환

**별도 Python 3.12 환경**을 사용합니다. Hugging Face 계정으로 [모델 페이지](https://huggingface.co/KRAFTON/Raon-1-mini-preview)에서 이용 조건에 동의하고 [GGUF 저장소](https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf)에도 접근할 수 있는지 확인한 뒤 인증합니다.

```powershell
py -3.12 -m venv .venv-convert
.\.venv-convert\Scripts\Activate.ps1
python -m pip install -r requirements-convert.txt
hf auth login
hf download KRAFTON/Raon-1-mini-preview --local-dir hf/Raon-1-mini-preview
hf download KRAFTON/Raon-1-mini-preview-gguf raon-1-mini-preview-Q4_K_M.gguf --local-dir hf/gguf
python -m export.export_d --gguf hf/gguf/raon-1-mini-preview-Q4_K_M.gguf --hf hf/Raon-1-mini-preview --out weights-d
python -m export.verify_d --manifest weights-d/manifest.json --gguf hf/gguf/raon-1-mini-preview-Q4_K_M.gguf --all
python -m export.verify_d --weights-dir weights-d --expected export/expected_sha256.txt
deactivate
```

GGUF는 **19,686,434,240 bytes**, SHA256 `ba56ad4bd6d15f8c4255876ff252890143074a5d3835fccfc656cee915999ee8`입니다. HF 원본은 safetensors만 받지 말고 전체를 받으세요. 변환기는 `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`과 모델 설정을 출력 폴더에 복사합니다. 토크나이저 파일이 없으면 오류로 끝납니다. 기준 해시 비교는 누락·추가·변경 파일을 알리고 실패 시 0이 아닌 종료 코드를 반환합니다. 생성 시각이 들어가는 `manifest.json`은 비교에서 제외합니다. 다른 기계에서 변환했다면 `weights-d` **폴더 전체**를 노트북으로 복사하세요.

### 4. 첫 빌드·자체 점검

Windows 노트북의 저장소 폴더에서 실행합니다.

```powershell
.\scripts\run.cmd scripts\selfcheck.py --weights-d .\weights-d
```

새 체크아웃에서 첫 selfcheck를 실행하면 xclbin 64개를 빌드하며 시험 노트북에서 **257초** 걸렸습니다. 다시 실행했을 때는 **62초** 걸렸고 새로 빌드한 파일은 없었습니다. 빌드 폴더는 파일 7,459개, 531 MB였습니다. 묶음 크기 0·16·64·256과 디코드 직접 경로의 바이트 대조까지 실행해도 빌드 파일은 늘지 않았습니다. 자체 점검은 적재 전에 커널 오브젝트 사본을 검사하고, `RAON_PREFILL_CHUNK`가 비어 있으면 256으로 설정하며 명시한 값은 유지합니다. 고정 영어 코드 리뷰 질문에 greedy·max_tokens 32로 생성합니다. `thinking + "|" + text`의 SHA256 앞 16자리 기준값은 **`3dc9191d4d3d6344`**입니다. 적재·prefill·decode 시간, 생성 토큰 수, `RESULT: PASS` 또는 `RESULT: FAIL`을 출력하고 0 또는 1로 종료합니다. `--model-dir` 기본값은 `--weights-d`와 같습니다.

### 5. 서버 실행

```powershell
.\serve\run-serve.ps1 -WeightsD .\weights-d
```

PowerShell 실행 정책 관련 안내는 [2단계](#2-도구-설치)를 보세요.

기본 주소는 `127.0.0.1:11434`, keep-alive는 **300초**입니다. 전경에서 실행하며 종료 코드를 PowerShell에 전달합니다.

선택: 로그온 때 자동 실행(현재 사용자, Interactive·Limited):

```powershell
.\scripts\install-task.ps1 -WeightsD .\weights-d -Start
# 작업·launcher를 지우고 그 작업이 띄운 서버도 중지:
.\scripts\install-task.ps1 -Uninstall
```

기본 작업 이름은 `Raon NPU server`이며 실행 파일과 로그는 `%LOCALAPPDATA%\raon-npu`에 둡니다. `-IronDir`, `-Port`, `-ListenAddress`, `-KeepAlive`, `-LogDir`, `-StateDir`, `-TaskName`으로 바꿀 수 있습니다. 배터리에서도 시작·유지하며 우선순위 7, 실행 시간 제한 없음, 실패 시 1분 간격 3회 재시작, 중복 실행 IgnoreNew입니다. 같은 이름의 작업을 덮어쓰므로 상태 폴더 하나에는 작업 하나를 사용하세요. 공백이 있는 경로를 지원합니다.

## API

자세한 설명은 [서버 문서](serve/README.md)를 보세요.

| 메서드 | 엔드포인트 | 용도 |
|---|---|---|
| GET / HEAD | `/` | Ollama 생존 확인 |
| GET | `/healthz` | 상태 확인 |
| GET | `/api/version`, `/api/tags`, `/api/ps` | Ollama 메타데이터·적재 모델 |
| POST | `/api/show` | 모델 정보 |
| POST | `/api/chat`, `/api/generate` | Ollama 대화·생성 |
| GET | `/v1/models` | OpenAI 모델 목록 |
| POST | `/v1/chat/completions`, `/v1/completions` | OpenAI 대화·완성 |

```powershell
'{"model":"raon-1-mini","prompt":"Hello","stream":false}' | curl.exe http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" --data-binary "@-"
```

## 업데이트

```powershell
git pull
.\scripts\run.cmd scripts\kernel_object_audit.py --fix
.\scripts\run.cmd scripts\selfcheck.py --weights-d .\weights-d
```

업데이트·오브젝트 수리 전에 서버를 멈추세요. Windows의 IRON은 커널 오브젝트를 설계 폴더에 복사한 뒤 기존 사본을 갱신하지 않습니다. `--fix`는 낡은 사본과 의존성 파일을 지우고 원본 오브젝트의 시간을 갱신해 재빌드를 유도합니다. 이 옵션이 없으면 읽기만 합니다. 다른 저장소는 `--repo`, JSON 보고 경로는 `--json-out`으로 지정합니다.

## 보안

서버에는 **인증·TLS가 없습니다.** 기본값 `127.0.0.1`은 이 PC에서만 연결할 수 있습니다. 믿을 수 있는 LAN에 열려면 `-ListenAddress 0.0.0.0`을 명시하세요. 두 실행 스크립트가 경고를 출력합니다. 인터넷에 직접 노출하지 마세요. 요청 로그(`serve.log`)에 질문 내용이 남을 수 있습니다.

## 문제 해결

| 증상 | 대응 |
|---|---|
| 첫 실행이 느림 | 커널 컴파일을 기다리고 콘솔·로그를 확인하세요. |
| 낡은 오브젝트 사본 | 서버를 멈추고 위의 `--fix` 검사 후 자체 점검을 실행하세요. |
| WinError 10048 | 다른 프로세스가 포트를 사용합니다. 멈추거나 `-Port`를 바꾸세요. |
| 메모리 부족 | 다른 큰 프로그램을 닫으세요. 시험 메모리는 64 GB, 적재 사용량은 약 25 GB입니다. |
| 쉬고 나면 느림 | 300초 유휴 시 가중치를 내립니다. 다음 요청은 적재부터 시작합니다. 필요하면 `-KeepAlive`를 바꾸세요. |
| 가중치·토크나이저 누락 | `-WeightsD`·`--weights-d`를 지정하고 변환 폴더 전체를 복사한 뒤 다시 검증하세요. |

## 환경 변수

| 변수 | 용도·기본값 |
|---|---|
| `RAON_WEIGHTS_D` | 서버 실행 스크립트·자체 점검의 가중치 폴더. CLI 경로가 없으면 필수 |
| `RAON_MODEL_DIR`, `RAON_GOLDEN_DIR` | 호스트 검사 입력. 없으면 CLI 인자 지정 |
| `RAON_IRON_DIR` | IRON 저장소. 실행기는 `C:\dev\IRON`으로 대체 |
| `RAON_MLIR_AIE_DIR` | mlir-aie 저장소. 실행기는 `C:\dev\mlir-aie`으로 대체. 선택 코덱 시험은 명시 필요 |
| `RAON_PREFILL_CHUNK` | 실행 스크립트·자체 점검에서 비어 있을 때만 256. 엔진 코드 기본값은 0 유지 |
| `RAON_GGUF` | 선택적인 실제 GGUF 변환 시험 자료 |
| `RAON_SOURCE_DIR`, `RAON_SRC_DIR` | golden·host 시험용 HF 모델 Python 원본 |
| `RAON_IRON_PATH` | 양자화 시험의 선택 IRON 참조 import 경로 |
| `RAON_MLIR_AIE_PY`, `RAON_IRON_PY` | API 호환성 시험용 Python 원본 트리 |
| `RAON_TXN_PATH` | 개발 시험용 선택 정적 GEMV transaction 바이너리 |

나머지 엔진 조정 변수는 기존 기본값을 유지합니다. 공개 패키징에서 수치 계산 동작은 바꾸지 않습니다.

## 저장소 구성

| 폴더 | 내용 |
|---|---|
| `quant/` | CPU 양자화·패킹·품질 평가 |
| `export/` | D 모드 변환·manifest·바이트 검증 |
| `golden/` | CPU 모델 스냅숏·RDA 참조 검사 |
| `kernels/` | AIE 커널 소스·장치 설계 |
| `bench/` | 개발 벤치마크. PowerShell은 개발용 — `C:\dev` 배치를 가정 |
| `host/` | NumPy 참조 모델·가중치 |
| `npu/` | NPU backend. PowerShell은 개발용 — `C:\dev` 배치를 가정 |
| `serve/` | Ollama·OpenAI 호환 HTTP 서버·수동 실행기 |
| `scripts/` | 설치·자체 점검·오브젝트 검사·선택 로그온 작업 |

기존 한국어 하위 문서는 한국어를 유지합니다.

일부 코드 주석은 공개하지 않은 내부 설계 노트(`DESIGN-*.md`의 절 번호)를 가리킵니다. 동작에는 영향이 없습니다.

## 라이선스·출처

[Apache-2.0 LICENSE](LICENSE), [NOTICE](NOTICE), [LLVM 예외](LICENSES/LLVM-exception.txt)를 확인하세요. KRAFTON Raon-1-mini-preview 가중치는 Apache-2.0으로 별도 배포되며 여기에는 포함되지 않습니다. KRAFTON·AMD 및 기타 상표는 각 소유자의 것입니다.
