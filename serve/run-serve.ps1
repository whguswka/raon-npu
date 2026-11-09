# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Start the Raon NPU server in the foreground (Windows)
param(
    [int]$Port = 11434,
    [string]$WeightsD = $env:RAON_WEIGHTS_D,
    [string]$ModelDir,
    [string]$Log = "$env:LOCALAPPDATA\raon-npu\logs\serve.log",
    [string]$KeepAlive = "300",
    # The launcher must resolve via Get-Command; use -PythonCmd python explicitly to select PATH's Python.
    [string]$IronDir = $(if ($env:RAON_IRON_DIR) { $env:RAON_IRON_DIR } else { "C:\dev\IRON" }),
    [string]$ListenAddress = "127.0.0.1",
    [string]$PythonCmd = (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\run.cmd")
)

$Root = Split-Path -Parent $PSScriptRoot
if (-not $WeightsD) { throw "Provide -WeightsD or set RAON_WEIGHTS_D." }
if (-not $ModelDir) { $ModelDir = $WeightsD }
if ($ListenAddress -ne "127.0.0.1") { Write-Warning "This server has no authentication or TLS. Listen on a non-loopback address only on a trusted LAN, and never expose it to the Internet." }
if (-not $env:RAON_PREFILL_CHUNK) { $env:RAON_PREFILL_CHUNK = "256" }
$env:RAON_IRON_DIR = $IronDir
Set-Location $Root

$env:PYTHONPATH = "$Root;$IronDir"
$env:PYTHONUNBUFFERED = "1"

$LogDir = Split-Path -Parent $Log
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}


# run.cmd 는 Python 런처.  PowerShell 내장 자동 변수 $Args 와 충돌하지 않도록 $ServerArgs 사용.
try {
    $Cmd = (Get-Command -Name $PythonCmd -ErrorAction Stop).Source
} catch {
    Write-Host "Execution Wrapper '$PythonCmd' was not found: $_" -ForegroundColor Red
    exit 1
}
$ServerArgs = @(
    "-u",
    "-m", "serve.server",
    "--host", $ListenAddress,
    "--port", "$Port",
    "--weights-d", $WeightsD,
    "--model-dir", $ModelDir,
    "--keep-alive", "$KeepAlive",
    "--log", $Log
)

Write-Host "Starting Raon NPU server on port $Port ..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"
Write-Host "WeightsD=$WeightsD, ModelDir=$ModelDir, KeepAlive=$KeepAlive"
Write-Host "Log: $Log"

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Cmd @ServerArgs
$processExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
$stopwatch.Stop()
if ($null -eq $processExitCode) {
    Write-Host "Server failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds without an exit code." -ForegroundColor Red
    exit 1
}
if ($processExitCode -ne 0) {
    Write-Host "Server failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds with exit code $processExitCode." -ForegroundColor Red
    exit $processExitCode
}
exit 0
