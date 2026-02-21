@echo off
REM TTS Server Startup Script for Windows
REM 服务器启动脚本 (Windows)

setlocal enabledelayedexpansion

REM 配置变量
set MODE=%MODE:~0,5%
if "%MODE%"=="" set MODE=local
set PORT=%PORT:~0,4%
if "%PORT%"=="" set PORT=8080
set HOST=%HOST:~0,9%
if "%HOST%"=="" set HOST=0.0.0.0
set CONFIG=%CONFIG%
set PROFILE=%PROFILE:~0,8%
if "%PROFILE%"=="" set PROFILE=release

echo ╔═══════════════════════════════════════════════════════════╗
echo ║           SDKWork-TTS Server Startup                      ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM 检查配置文件
if not "%CONFIG%"=="" (
    if not exist "%CONFIG%" (
        echo 错误：配置文件不存在：%CONFIG%
        exit /b 1
    )
)

REM 构建项目
echo 构建项目...
if "%PROFILE%"=="release" (
    cargo build --release --no-default-features --features cpu
) else (
    cargo build --no-default-features --features cpu
)

if errorlevel 1 (
    echo 错误：构建失败
    exit /b 1
)

REM 设置二进制文件路径
if "%PROFILE%"=="release" (
    set BIN=.\target\release\sdkwork-tts.exe
) else (
    set BIN=.\target\debug\sdkwork-tts.exe
)

REM 检查二进制文件
if not exist "%BIN%" (
    echo 错误：二进制文件不存在：%BIN%
    exit /b 1
)

REM 启动服务器
echo 启动服务器...
echo 模式：%MODE%
echo 地址：http://%HOST%:%PORT%
echo.

if not "%CONFIG%"=="" (
    "%BIN%" server --config "%CONFIG%"
) else (
    "%BIN%" server --mode %MODE% --host %HOST% --port %PORT%
)

endlocal
