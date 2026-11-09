# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

# Run the staged MoE benchmark (template + arg_plus patch) through run.cmd; log to C:\dev\raon\moe\last-run.log and print the summary.
Set-Location C:\dev\raon\moe
$env:PYTHONPATH = 'C:\dev\raon\moe;C:\dev\IRON'
$env:PYTHONUNBUFFERED = '1'
$t = Get-Date
cmd /c "C:\dev\raon\run.cmd -u bench\moe\moe_bench.py --timeout 180 --warmup 2 --iters 30 --stress-iters 1000 > C:\dev\raon\moe\last-run.log 2>&1"
$rc = $LASTEXITCODE
Get-Content C:\dev\raon\moe\last-run.log | Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*("|%|\^bb|\}|\))' } | Select-Object -Last 60 | ForEach-Object { $_.Substring(0, [Math]::Min(230, $_.Length)) }
"exit: $rc  elapsed_s: $([int]((Get-Date) - $t).TotalSeconds)"
if ($rc -ne 0) {
    exit $rc
}
