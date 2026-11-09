@echo off
rem Copyright (C) 2026 raon-npu authors
rem SPDX-License-Identifier: Apache-2.0
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto no_vs
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL goto no_vs
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined RAON_MLIR_AIE_DIR set "RAON_MLIR_AIE_DIR=C:\dev\mlir-aie"
if not exist "%RAON_MLIR_AIE_DIR%\iron_env.cmd" (
    echo mlir-aie IRON environment not found at %RAON_MLIR_AIE_DIR% - run scripts\setup-toolchain.ps1 or set RAON_MLIR_AIE_DIR 1>&2
    exit /b 1
)
call "%RAON_MLIR_AIE_DIR%\iron_env.cmd" >nul || exit /b 1
if not defined RAON_IRON_DIR set "RAON_IRON_DIR=C:\dev\IRON"
if not exist "%RAON_IRON_DIR%" (
    echo IRON checkout not found at %RAON_IRON_DIR% - run scripts\setup-toolchain.ps1 or set RAON_IRON_DIR 1>&2
    exit /b 1
)
set "PYTHONPATH=%~dp0..;%RAON_IRON_DIR%;%PYTHONPATH%"
python %*
exit /b %ERRORLEVEL%
:no_vs
echo Visual Studio Build Tools missing - run scripts\setup-toolchain.ps1 first. 1>&2
exit /b 1
