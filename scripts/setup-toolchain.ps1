# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Prerequisites: Python 3.13 64-bit from python.org (not Microsoft Store), Git for Windows,
# NPU driver, and an elevated Administrator PowerShell. Run after cloning this repository.
# Installs VS Build Tools, XRT SDK 2.21.75 and mlir-aie v1.4.4 / pinned IRON.
param([string]$Root = 'C:\dev')
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Run PowerShell as Administrator.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows: winget install Git.Git' }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Install Python 3.13 64-bit from https://www.python.org/downloads/windows/ (not Microsoft Store).' }
& python -c "import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 and 'WindowsApps' not in sys.executable"
if ($LASTEXITCODE) { throw 'Python must be 3.13 64-bit from python.org.' }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
Start-Transcript -Path (Join-Path $Root "raon-setup.log") -Append | Out-Null
function Step($m) { Write-Host "=== $(Get-Date -Format 'HH:mm:ss') $m" }

# 1. Visual Studio 2026 Build Tools: MSVC, Windows SDK and CMake (recommended set), Clang (doc section 1.1)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
function Find-VS { if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath } }
$vs = Find-VS
if (-not $vs) {
  Step 'Visual Studio Build Tools (the slow step, 10-30 min)'
  winget install -e --id Microsoft.VisualStudio.BuildTools --silent --accept-package-agreements --accept-source-agreements --override '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset'
  $vs = Find-VS
  if (-not $vs) { throw "Visual Studio Build Tools not found after install (winget exit $LASTEXITCODE)" }
}
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'

# Runs cmd lines inside the Visual Studio x64 environment; stdout and stderr go to the console and the log.
function Run-Cmd([string[]]$lines) {
  $f = Join-Path $Root "raon-step.cmd"
  Set-Content -Path $f -Encoding ascii -Value (@('@echo off', "call `"$vcvars`" >nul || exit /b 1") + $lines)
  $ErrorActionPreference = 'Continue'
  cmd /c "`"$f`" 2>&1" | Out-Host
  return $LASTEXITCODE
}

# 2. XRT SDK 2.21.75: headers, import libs, xclbinutil, pyxrt for CPython 3.13 (doc section 3; checksum from mlir-aie CI)
$xrt = 'C:\Xilinx\XRT'
if (-not (Test-Path "$xrt\python")) {
  if (Test-Path $xrt) { throw "$xrt exists but has no python folder - check it, remove it, and re-run" }
  Step 'XRT Windows SDK 2.21.75'
  $zip = "$env:TEMP\xrt_windows_sdk.zip"
  Invoke-WebRequest -UseBasicParsing -OutFile $zip https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip
  if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne 'CCC244C2C423588972ADE76142CDC01049477AAA39A35BE97E782B97EB7C5295') { throw 'XRT SDK checksum mismatch - not extracted' }
  $tmp = "$env:TEMP\xrt_sdk_x"
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  Expand-Archive -Path $zip -DestinationPath $tmp
  New-Item -ItemType Directory -Force -Path C:\Xilinx | Out-Null
  Move-Item "$tmp\xrt_sdk\xrt" $xrt
  Remove-Item -Recurse -Force $tmp, $zip
}

# 3. mlir-aie v1.4.4 checkout and IRON environment (doc section 4). iron_setup.py installs the wheels published for this tag.
$repo = Join-Path $Root "mlir-aie"
if ((Test-Path $repo) -and -not (Test-Path "$repo\.git")) { throw "$repo exists but is not a git checkout" }
if (-not (Test-Path $repo)) {
  Step 'mlir-aie v1.4.4 checkout'
  $rc = Run-Cmd @("git clone --depth 1 --branch v1.4.4 --recurse-submodules --shallow-submodules https://github.com/Xilinx/mlir-aie.git `"$repo`"")
  if ($rc) { throw "git clone failed (exit $rc)" }
}
$tag = & git -C $repo describe --tags
if ($LASTEXITCODE -ne 0 -or $tag -ne 'v1.4.4') { throw 'mlir-aie must be exactly v1.4.4; inspect your checkout and run git checkout v1.4.4.' }
if (-not (Test-Path "$repo\iron_env.cmd")) {
  Step 'IRON environment (iron_setup.py)'
  $rc = Run-Cmd @("cd /d `"$repo`" || exit /b 1", "python utils\iron_setup.py --xrt-root $xrt")
  if ($rc) { throw "iron_setup.py failed (exit $rc)" }
}

# Verify existing SDK metadata when available.
# This hash is the header value found after extracting XRT SDK 2.21.75 zip (sha256 CCC244C2…).
$versionFile = Join-Path $xrt 'include\xrt\detail\version.h'
$expectedHash = '4eb1f4392a012b4e6eca759762389c612537f7c7'
if (Test-Path $versionFile) {
  $versionText = Get-Content $versionFile -Raw
  if ($versionText -notmatch $expectedHash) { throw "XRT in $xrt is not the 2.21.75 SDK build (expected build hash $expectedHash)." }
} else { Write-Warning 'XRT version could not be verified; checking pyxrt only.' }
if (-not (Test-Path "$xrt\python\pyxrt.pyd")) { throw 'XRT pyxrt.pyd is missing.' }
$iron = Join-Path $Root 'IRON'
$commit = 'b3ba3cf3a138c21e021d5ee2d6794573cd7a3287'
if (-not (Test-Path $iron)) {
  & git clone https://github.com/amd/IRON.git $iron
  if ($LASTEXITCODE) { throw 'IRON clone failed.' }
  & git -C $iron checkout $commit
  if ($LASTEXITCODE) { throw 'IRON checkout failed.' }
}
$head = & git -C $iron rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head -ne $commit) { throw "IRON commit differs. Review local changes, then git -C `"$iron`" checkout $commit" }
$rc = Run-Cmd @("call `"$repo\iron_env.cmd`" || exit /b 1", "python -c `"import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8`" || exit /b 1", "python -m pip install -r `"$ProjectRoot\requirements-npu.txt`"")
if ($rc) { throw "Engine dependency install failed ($rc)." }
Write-Host "Set RAON_MLIR_AIE_DIR=$repo and RAON_IRON_DIR=$iron when using a custom -Root."

# 5. First check: compile and run the SAXPY example on the NPU (doc section 5). Success prints PASS!
Step 'SAXPY on the NPU'
$saxpy = Run-Cmd @("cd /d `"$repo`" || exit /b 1", 'call iron_env.cmd || exit /b 1', 'cd programming_examples\getting_started\01_SAXPY || exit /b 1', 'python saxpy.py')

$ErrorActionPreference = 'Continue'
Write-Host '--- RESULT'
Write-Host "vs: $vs"
Write-Host "xrt_sdk_pyxrt: $(Test-Path "$xrt\python\pyxrt.pyd")"
Write-Host "iron_env: $(Test-Path "$repo\iron_env.cmd")"
Write-Host "saxpy_exit: $saxpy"
Stop-Transcript | Out-Null
if ($saxpy) { exit $saxpy }
