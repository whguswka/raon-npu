# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
param(
    [string]$WeightsD,
    [string]$IronDir = $(if ($env:RAON_IRON_DIR) { $env:RAON_IRON_DIR } else { 'C:\dev\IRON' }),
    [int]$Port = 11434,
    [string]$ListenAddress = '127.0.0.1',
    [int]$KeepAlive = 300,
    [string]$StateDir = "$env:LOCALAPPDATA\raon-npu",
    [string]$LogDir,
    [string]$TaskName = 'Raon NPU server',
    [switch]$Uninstall,
    [switch]$Start
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $StateDir 'serve-task.cmd'
if ($Uninstall) {
    $ServerLogPath = $null
    if (Test-Path -LiteralPath $Launcher) {
        $LauncherText = Get-Content -LiteralPath $Launcher -Raw
        $LogMatch = [regex]::Match($LauncherText, '--log\s+"([^"]+)"', [Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($LogMatch.Success) {
            $ServerLogPath = $LogMatch.Groups[1].Value
        } else {
            Write-Warning "Could not read the --log path from launcher; server processes will not be stopped: $Launcher"
        }
    } else {
        Write-Warning "Launcher not found; server processes will not be stopped: $Launcher"
    }
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    if ($ServerLogPath) {
        $Stopped = 0
        $ServerProcesses = Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Contains('serve.server') -and
            $_.CommandLine.Contains($ServerLogPath)
        }
        foreach ($Process in $ServerProcesses) {
            try {
                Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
                $Stopped++
            } catch {
                Write-Warning "Could not stop server process $($Process.ProcessId): $($_.Exception.Message)"
            }
        }
        Write-Host "Stopped $Stopped server process(es)."
    }
    if (Test-Path -LiteralPath $Launcher) { Remove-Item -LiteralPath $Launcher }
    exit 0
}
if (-not $WeightsD) { throw '-WeightsD is required.' }
$WeightsD = (Resolve-Path $WeightsD).Path
$IronDir = (Resolve-Path $IronDir).Path
if (-not $LogDir) { $LogDir = Join-Path $StateDir 'logs' }
if ($ListenAddress -ne '127.0.0.1') { Write-Warning 'This server has no authentication or TLS. Listen on a non-loopback address only on a trusted LAN, and never expose it to the Internet.' }
# These values enter cmd syntax; reject command delimiters rather than interpolating them.
foreach ($Value in @($Root, $WeightsD, $IronDir, $StateDir, $LogDir, $ListenAddress)) {
    if ($Value -match '["%\r\n&|<>^!]') { throw 'Paths/address contain unsupported cmd metacharacters.' }
}
New-Item -ItemType Directory -Force -Path $StateDir, $LogDir | Out-Null
$Launcher = [IO.Path]::GetFullPath($Launcher)
$LogDir = [IO.Path]::GetFullPath($LogDir)
$Lines = @(
    '@echo off',
    "cd /d `"$Root`"",
    "set `"PYTHONPATH=$Root;$IronDir`"",
    "set `"RAON_IRON_DIR=$IronDir`"",
    'set "PYTHONUNBUFFERED=1"',
    'set "PYTHONIOENCODING=utf-8"',
    'set "RAON_PREFILL_CHUNK=256"',
    "call `"$Root\scripts\run.cmd`" -u -X faulthandler -m serve.server --host `"$ListenAddress`" --port $Port --weights-d `"$WeightsD`" --model-dir `"$WeightsD`" --keep-alive $KeepAlive --log `"$LogDir\serve.log`" 1>>`"$LogDir\serve.out`" 2>>`"$LogDir\serve.err`"",
    'exit /b %ERRORLEVEL%'
)
Set-Content -LiteralPath $Launcher -Value $Lines -Encoding Default
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -WindowStyle Hidden -Command "cmd /c serve-task.cmd"' -WorkingDirectory (Split-Path -Parent $Launcher)
$User = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -Priority 7 -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3 -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
if ($Start) { Start-ScheduledTask -TaskName $TaskName }
Write-Host "Registered: $TaskName ($Launcher)"
