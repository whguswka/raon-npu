# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"
Set-Location C:\dev\raon\moe-d
$env:PYTHONPATH = 'C:\dev\raon\moe-d;C:\dev\IRON'
$env:PYTHONUNBUFFERED = '1'
$started = Get-Date
cmd /c "C:\dev\raon\run.cmd -u bench\moe_d\moe_d_bench.py --timeout 240 --warmup 2 --iters 30 --stress-iters 1000 > C:\dev\raon\moe-d\last-run.log 2>&1"
$rc = $LASTEXITCODE
Get-Content C:\dev\raon\moe-d\last-run.log |
    Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*("|%|\^bb|\}|\))' } |
    Select-Object -Last 60 |
    ForEach-Object { $_.Substring(0, [Math]::Min(230, $_.Length)) }
"exit: $rc  elapsed_s: $([int]((Get-Date) - $started).TotalSeconds)"
exit $rc
