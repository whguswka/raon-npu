# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# NOTE: Laptop-only script (Windows + AMD XDNA 2 NPU). The C:\dev\... paths below are the laptop's
# layout and are intentionally left as-is; on other machines pass the parameters or use run_golden.py directly.
<#
.SYNOPSIS
    Run Raon-1-mini D mode Golden Verification on Laptop NPU.
.DESCRIPTION
    Sets PYTHONPATH to the repository root plus C:\dev\IRON and PYTHONUNBUFFERED=1.
    Invokes npu/run_golden.py via C:\dev\raon\run.cmd with staged execution guards.
.PARAMETER Stage
    Execution stage: 'stage1' (layer 0 single token), 'stage2' (layers 0..3 two tokens), 'full' (all 25 layers steps 2..16).
.PARAMETER Layers
    Custom limit for hidden layers (1..25).
.PARAMETER StartStep
    First decode step to evaluate; only 2 is supported (default 2). A run always resumes from snapshot step_000001.
.PARAMETER EndStep
    Last decode step to evaluate (default 16).
.PARAMETER TokenTimeout
    Wall-clock limit for each NPU model step, including the baseline step and first-step kernel loading.
    Expiry exits code 1 with thread stacks on stderr; a returned over-limit step is a failed verification.
.PARAMETER PythonCmd
    Execution launcher to use. It must resolve via Get-Command; use -PythonCmd python explicitly to select PATH's Python.
#>

[CmdletBinding()]
param(
    [ValidateSet("stage1", "stage2", "full", "custom")]
    [string]$Stage = "full",

    [int]$Layers = 25,
    [int]$StartStep = 2,
    [int]$EndStep = 16,
    [float]$TokenTimeout = 120.0,
    [string]$WeightsD = "C:\dev\raon\weights-d",
    [string]$GoldenDir = "C:\dev\raon\golden\ko16",
    [string]$ModelDir = "C:\dev\raon\weights-d\host",
    [string]$OutFile = "",
    [string]$PythonCmd = "C:\dev\raon\run.cmd"
)

$ErrorActionPreference = "Stop"

# Set Working Directory and Environment Variables
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $Root
$env:PYTHONPATH = "$Root;C:\dev\IRON"
$env:PYTHONUNBUFFERED = "1"
if (-not $OutFile) {
    $OutFile = Join-Path $Root "npu\out\golden_report.json"
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Raon-1-mini NPU Golden Verification Runner (PowerShell)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Working Directory:  $((Get-Location).Path)"
Write-Host "PYTHONPATH:         $env:PYTHONPATH"
Write-Host "Stage Profile:      $Stage"

# Configure Stage Presets
switch ($Stage) {
    "stage1" {
        Write-Host "-> Stage 1: Single layer (Layer 0), step 2 only (Fast Smoke Test)" -ForegroundColor Yellow
        $Layers = 1
        $StartStep = 2
        $EndStep = 2
        $TokenTimeout = 60.0
    }
    "stage2" {
        Write-Host "-> Stage 2: 4 Layers (RDA + MLA + MoE), steps 2..3 (Intermediate Stage)" -ForegroundColor Yellow
        $Layers = 4
        $StartStep = 2
        $EndStep = 3
        $TokenTimeout = 90.0
    }
    "full" {
        Write-Host "-> Full Run: All 25 Layers, steps 2..16 (Full Golden Run)" -ForegroundColor Green
        $Layers = 25
        $StartStep = 2
        $EndStep = 16
    }
}

# Determine Execution Wrapper
try {
    $pythonCmdResolved = (Get-Command -Name $PythonCmd -ErrorAction Stop).Source
    Write-Host "Using Execution Wrapper: $pythonCmdResolved"
} catch {
    Write-Host "Execution Wrapper '$PythonCmd' was not found: $_" -ForegroundColor Red
    exit 1
}

# Assemble Arguments
$pyArgs = @(
    "npu\run_golden.py",
    "--weights-d", $WeightsD,
    "--golden-dir", $GoldenDir,
    "--start-step", $StartStep,
    "--end-step", $EndStep,
    "--layers", $Layers,
    "--timeout-per-token", $TokenTimeout,
    "--out", $OutFile
)

if (Test-Path -Path $ModelDir) {
    $pyArgs += @("--model-dir", $ModelDir)
}

Write-Host "Executing command: $pythonCmdResolved $($pyArgs -join ' ')" -ForegroundColor DarkGray
Write-Host "-----------------------------------------------------------------"

# Run with timer
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pythonCmdResolved $pyArgs
    $processExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $stopwatch.Stop()
    Write-Host "-----------------------------------------------------------------"
    if ($null -eq $processExitCode) {
        Write-Host "Execution failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds without an exit code." -ForegroundColor Red
        exit 1
    }
    if ($processExitCode -ne 0) {
        Write-Host "Execution failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds with exit code $processExitCode." -ForegroundColor Red
        exit $processExitCode
    }
    Write-Host "Execution completed in $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds." -ForegroundColor Green
} catch {
    $stopwatch.Stop()
    Write-Host "-----------------------------------------------------------------"
    Write-Host "Execution failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds: $_" -ForegroundColor Red
    exit 1
}
exit 0
