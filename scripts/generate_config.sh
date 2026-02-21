#!/bin/bash
# SDKWork-TTS Configuration Generator
# Generates server.yaml configuration file

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     SDKWork-TTS Configuration Generator                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Output file
OUTPUT_FILE="${1:-server.yaml}"

# Interactive prompts
echo -e "${YELLOW}Server Configuration${NC}"
echo ""

# Mode
echo "Select server mode:"
echo "  1) local   - Use local TTS engines"
echo "  2) cloud   - Use cloud TTS providers"
echo "  3) hybrid  - Local with cloud fallback"
read -p "Mode [1-3] (default: 1): " mode_choice

case $mode_choice in
    2) MODE="cloud" ;;
    3) MODE="hybrid" ;;
    *) MODE="local" ;;
esac

# Host and Port
read -p "Host [default: 0.0.0.0]: " HOST
HOST=${HOST:-0.0.0.0}

read -p "Port [default: 8080]: " PORT
PORT=${PORT:-8080}

echo ""
echo -e "${YELLOW}Local Engine Configuration${NC}"
echo ""

# GPU
read -p "Use GPU? [y/N] (requires CUDA): " USE_GPU
if [[ $USE_GPU =~ ^[Yy]$ ]]; then
    GPU_TRUE="true"
    read -p "Use FP16 precision? [y/N]: " USE_FP16
    if [[ $USE_FP16 =~ ^[Yy]$ ]]; then
        FP16_TRUE="true"
    else
        FP16_TRUE="false"
    fi
else
    GPU_TRUE="false"
    FP16_TRUE="false"
fi

# Batch size
read -p "Batch size [default: 4]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-4}

# Max concurrent
read -p "Max concurrent requests [default: 10]: " MAX_CONCURRENT
MAX_CONCURRENT=${MAX_CONCURRENT:-10}

echo ""
echo -e "${YELLOW}Cloud Configuration (if cloud/hybrid mode)${NC}"
echo ""

# OpenAI
read -p "Enable OpenAI channel? [y/N]: " ENABLE_OPENAI
if [[ $ENABLE_OPENAI =~ ^[Yy]$ ]]; then
    OPENAI_KEY="${OPENAI_API_KEY:-}"
    if [ -z "$OPENAI_KEY" ]; then
        read -p "OpenAI API Key: " OPENAI_KEY
    fi
fi

# Aliyun
read -p "Enable Aliyun channel? [y/N]: " ENABLE_ALIYUN
if [[ $ENABLE_ALIYUN =~ ^[Yy]$ ]]; then
    ALIYUN_KEY="${ALIYUN_API_KEY:-}"
    ALIYUN_SECRET="${ALIYUN_API_SECRET:-}"
    if [ -z "$ALIYUN_KEY" ]; then
        read -p "Aliyun API Key: " ALIYUN_KEY
    fi
    if [ -z "$ALIYUN_SECRET" ]; then
        read -p "Aliyun API Secret: " ALIYUN_SECRET
    fi
fi

echo ""
echo -e "${YELLOW}Speaker Library Configuration${NC}"
echo ""

read -p "Speaker library path [default: speaker_library]: " SPEAKER_LIB_PATH
SPEAKER_LIB_PATH=${SPEAKER_LIB_PATH:-speaker_library}

read -p "Max cached speakers [default: 1000]: " MAX_SPEAKERS
MAX_SPEAKERS=${MAX_SPEAKERS:-1000}

echo ""
echo -e "${YELLOW}Logging Configuration${NC}"
echo ""

echo "Log level:"
echo "  1) error"
echo "  2) warn"
echo "  3) info"
echo "  4) debug"
echo "  5) trace"
read -p "Level [1-5] (default: 3): " log_choice

case $log_choice in
    1) LOG_LEVEL="error" ;;
    2) LOG_LEVEL="warn" ;;
    4) LOG_LEVEL="debug" ;;
    5) LOG_LEVEL="trace" ;;
    *) LOG_LEVEL="info" ;;
esac

# Generate configuration
echo ""
echo -e "${YELLOW}Generating configuration...${NC}"

cat > "$OUTPUT_FILE" << EOF
# SDKWork-TTS Server Configuration
# Generated: $(date -Iseconds)

# Server mode: local, cloud, hybrid
mode: $MODE

# Server address and port
host: "$HOST"
port: $PORT

# Local mode configuration
local:
  enabled: true
  checkpoints_dir: "checkpoints"
  default_engine: "indextts2"
  use_gpu: $GPU_TRUE
  use_fp16: $FP16_TRUE
  batch_size: $BATCH_SIZE
  max_concurrent: $MAX_CONCURRENT

# Cloud mode configuration
cloud:
  enabled: $([ "$MODE" = "cloud" ] || [ "$MODE" = "hybrid" ] && echo "true" || echo "false")
  default_channel: null
  channels:
EOF

if [[ $ENABLE_OPENAI =~ ^[Yy]$ ]]; then
    cat >> "$OUTPUT_FILE" << EOF
    - name: openai
      type: openai
      api_key: "\${OPENAI_API_KEY:-$OPENAI_KEY}"
      models:
        - tts-1
        - tts-1-hd
      default_model: tts-1
      timeout: 30
      retries: 3
EOF
fi

if [[ $ENABLE_ALIYUN =~ ^[Yy]$ ]]; then
    cat >> "$OUTPUT_FILE" << EOF
    - name: aliyun
      type: aliyun
      api_key: "\${ALIYUN_API_KEY:-$ALIYUN_KEY}"
      api_secret: "\${ALIYUN_API_SECRET:-$ALIYUN_SECRET}"
      models:
        - tts-v1
      default_model: tts-v1
      timeout: 30
      retries: 3
EOF
fi

cat >> "$OUTPUT_FILE" << EOF

# Speaker library configuration
speaker_lib:
  enabled: true
  local_path: "$SPEAKER_LIB_PATH"
  cloud_enabled: true
  auto_sync: false
  max_cache_size: $MAX_SPEAKERS

# Authentication (optional)
auth:
  enabled: false
  # api_key: "your-api-key-here"
  # jwt_secret: "your-jwt-secret-here"
  # token_expiry: 24

# Logging configuration
logging:
  level: "$LOG_LEVEL"
  file: null
  rotation_size: 100
  keep_files: 5
  access_log: true
EOF

echo -e "${GREEN}✓ Configuration generated: $OUTPUT_FILE${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review and edit: $OUTPUT_FILE"
echo "2. Set environment variables (if using cloud):"
[[ $ENABLE_OPENAI =~ ^[Yy]$ ]] && echo "   export OPENAI_API_KEY=..."
[[ $ENABLE_ALIYUN =~ ^[Yy]$ ]] && echo "   export ALIYUN_API_KEY=..."
[[ $ENABLE_ALIYUN =~ ^[Yy]$ ]] && echo "   export ALIYUN_API_SECRET=..."
echo "3. Start server: sdkwork-tts server --config $OUTPUT_FILE"
echo ""
