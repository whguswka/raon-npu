<!--
Copyright (C) 2026 raon-npu authors
SPDX-License-Identifier: Apache-2.0
-->

# MoE D 검증 실행기

`moe_d_bench.py`는 DispatchTime 경로를 사용하지 않고 정적 템플릿과
DDR 패치 경로를 5단계로 검증한다.

1. B1 6-slot 정적 템플릿
2. B1 6-slot 전문가 패치 5패턴
3. B2 9-bit 7-slot 정적 템플릿
4. B2 9-bit 전문가 패치 5패턴
5. MoE projection chain 1,000회

5단계의 시간은 세 GEMV, NPU↔호스트 복사, 호스트 SiTU-GLU와 bf16 변환을
포함한다. 라우터와 선택된 전문가의 가중합은 포함하지 않으므로 full layer
시간이 아니다.

노트북에서 전체 검증을 실행하는 명령은 다음과 같다.

```powershell
pwsh -File bench/moe_d/run-moe-d.ps1
```

NPU 없이 실행 흐름만 확인하려면 다음 명령을 사용한다.

```bash
python3 bench/moe_d/moe_d_bench.py --mock --stages 1 2 3 4 5 --stress-iters 1000
```
