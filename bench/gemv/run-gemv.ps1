# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Run the raon-npu general q4nx GEMV benchmark on AMD XDNA 2 NPU. Full log: C:\dev\raon\gemv\last-run.log
Set-Location C:\dev\raon\gemv
$env:PYTHONPATH = 'C:\dev\raon\gemv;C:\dev\IRON'
$t = Get-Date
cmd /c "C:\dev\raon\run.cmd -u bench\gemv\gemv_bench.py --preset raon --warmup 10 --iters 100 --timeout 300 --json-out C:\dev\raon\gemv\last-run.json > C:\dev\raon\gemv\last-run.log 2>&1"
$rc = $LASTEXITCODE
$log = Get-Content C:\dev\raon\gemv\last-run.log
"== error lines"
$log | Select-String 'Error|error:|Traceback|DEADLOCK|CRITICAL|TIMEOUT|CRASH' | Select-Object -First 10 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(250, $_.Line.Length)) }
"== results summary"
$log | Select-String 'Result \[PASS\]|Result \[FAIL\]|Result \[TIMEOUT\]|Result \[CRASH\]|Shape / Layer|Bit \(ULP\)|Total benchmark finished' | ForEach-Object { $_.Line.Substring(0, [Math]::Min(250, $_.Line.Length)) }
"== tail"
$log | Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*("|%|\^bb|\}|\))' } | Select-Object -Last 30 | ForEach-Object { $_.Substring(0, [Math]::Min(220, $_.Length)) }
"exit: $rc  elapsed_s: $([int]((Get-Date) - $t).TotalSeconds)"
