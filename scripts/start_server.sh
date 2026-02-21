#!/bin/bash
# TTS Server Startup Script
# 服务器启动脚本

set -e

# 配置变量
MODE="${MODE:-local}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
CONFIG="${CONFIG:-}"
PROFILE="${PROFILE:-release}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           SDKWork-TTS Server Startup                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查配置文件
if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
    echo -e "${RED}错误：配置文件不存在：$CONFIG${NC}"
    exit 1
fi

# 构建项目
echo -e "${YELLOW}构建项目...${NC}"
if [ "$PROFILE" = "release" ]; then
    cargo build --release --no-default-features --features cpu
else
    cargo build --no-default-features --features cpu
fi

# 设置二进制文件路径
if [ "$PROFILE" = "release" ]; then
    BIN="./target/release/sdkwork-tts"
else
    BIN="./target/debug/sdkwork-tts"
fi

# 检查二进制文件
if [ ! -f "$BIN" ]; then
    echo -e "${RED}错误：二进制文件不存在：$BIN${NC}"
    exit 1
fi

# 启动服务器
echo -e "${YELLOW}启动服务器...${NC}"
echo -e "${GREEN}模式：${MODE}${NC}"
echo -e "${GREEN}地址：http://${HOST}:${PORT}${NC}"
echo ""

if [ -n "$CONFIG" ]; then
    exec "$BIN" server --config "$CONFIG"
else
    exec "$BIN" server --mode "$MODE" --host "$HOST" --port "$PORT"
fi
