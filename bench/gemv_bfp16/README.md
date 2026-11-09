<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# Raon-1-mini 일반 9비트 bfp16ebs8 GEMV 벤치마크 러너

상세 아키텍처, L1 예산 분석, 블록 내부 배치 다이어그램은 [kernels/gemv_bfp16/README.md](../../kernels/gemv_bfp16/README.md)를 참조하십시오.

## 노트북 NPU 벤치마크 실행 방법

### 1) PowerShell 스크립트 실행
```powershell
bench\gemv_bfp16\run-bfp16.ps1
```

### 2) 커맨드라인 직접 실행
```powershell
Set-Location C:\dev\raon\bfp16
$env:PYTHONPATH = 'C:\dev\raon\bfp16;C:\dev\IRON'
cmd /c "C:\dev\raon\run.cmd -u bench\gemv_bfp16\bench.py --preset raon --warmup 10 --iters 100 --timeout 300 --json-out C:\dev\raon\bfp16\last-run.json > C:\dev\raon\bfp16\last-run.log 2>&1"
```

## CPU 호스트 단위 테스트 실행
```bash
python3 -m pytest -q bench quant -p no:cacheprovider
```

실가중치에서 NPU 패킹 시 0으로 내려갈 비영(非零) `e < 7` 묶음은 다음처럼
확인합니다. GGUF는 메모리 매핑하고 한 번에 한 텐서의 일부 행만 역양자화합니다.

```bash
python3 bench/gemv_bfp16/check_tiny_blocks.py /path/to/model.gguf
# 대표 텐서만 빠르게 확인
python3 bench/gemv_bfp16/check_tiny_blocks.py /path/to/model.gguf --limit 3
```
