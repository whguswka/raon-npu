# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# NOTE: Laptop-only script (Windows + AMD XDNA 2 NPU). The C:\dev\... paths below are the laptop's
# layout and are intentionally left as-is; on other machines pass -RepoRoot / -RunnerCmd (and set PYTHONPATH).

param(
    [ValidateSet("smoke", "alt2", "all10", "stress", "exp_short_dma", "q4", "b1", "b2", "mixed", "stage2", "full")]
    [string]$Stage = "smoke",
    [string]$RepoRoot = "C:\dev\raon\gemv-unified",
    [string]$RunnerCmd = "C:\dev\raon\run.cmd",
    [int]$TimeoutSec = 900,
    [switch]$CompareStatic,
    [string]$JsonOut = ""
)

$ErrorActionPreference = "Stop"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "C:\dev\raon\gemv-unified;C:\dev\IRON"

function Invoke-StageCommand {
    param(
        [string]$Label,
        [string]$ScriptPath,
        [string]$Arguments
    )
    Write-Host "=== [$Label] cmd /c `"$RunnerCmd -u $ScriptPath $Arguments`" ===" -ForegroundColor Cyan
    cmd /c "`"$RunnerCmd`" -u $ScriptPath $Arguments"
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Label' failed with exit code $LASTEXITCODE"
    }
}

$extraFlags = "--timeout $TimeoutSec"
if ($CompareStatic) {
    $extraFlags += " --compare-static"
}
if ($JsonOut -ne "") {
    $extraFlags += " --json-out `"$JsonOut`""
}

if ($Stage -eq "exp_short_dma") {
    Invoke-StageCommand -Label "exp_short_dma" -ScriptPath "bench\gemv_unified\exp_short_dma.py" -Arguments "--first-timeout $TimeoutSec --short-timeout $TimeoutSec --soak-timeout $TimeoutSec"
} elseif ($Stage -eq "stage2") {
    Invoke-StageCommand -Label "stage2" -ScriptPath "bench\gemv_unified\stage2_bench.py" -Arguments "--stage all --timeout $TimeoutSec"
} elseif ($Stage -in @("q4", "b1", "b2", "mixed")) {
    Invoke-StageCommand -Label $Stage -ScriptPath "bench\gemv_unified\stage2_bench.py" -Arguments "--stage $Stage --timeout $TimeoutSec"
} elseif ($Stage -eq "full") {
    Invoke-StageCommand -Label "smoke"  -ScriptPath "bench\gemv_unified\unified_bench.py" -Arguments "--stage smoke $extraFlags"
    Invoke-StageCommand -Label "alt2"   -ScriptPath "bench\gemv_unified\unified_bench.py" -Arguments "--stage alt2 $extraFlags"
    Invoke-StageCommand -Label "all10"  -ScriptPath "bench\gemv_unified\unified_bench.py" -Arguments "--stage all10 $extraFlags"
    Invoke-StageCommand -Label "stress" -ScriptPath "bench\gemv_unified\unified_bench.py" -Arguments "--stage stress $extraFlags"
} else {
    Invoke-StageCommand -Label $Stage -ScriptPath "bench\gemv_unified\unified_bench.py" -Arguments "--stage $Stage $extraFlags"
}
