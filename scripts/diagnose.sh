#!/bin/bash
# SDKWork-TTS Diagnostic Tool
# Collects system information and troubleshoots common issues

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SDKWork-TTS Diagnostic Tool                        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Output file
REPORT_FILE="sdkwork-tts-diagnostic-$(date +%Y%m%d-%H%M%S).txt"

# Collect information
{
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║        SDKWork-TTS Diagnostic Report                      ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Generated: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "SYSTEM INFORMATION"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # OS
    echo "Operating System:"
    if [ -f /etc/os-release ]; then
        cat /etc/os-release | grep -E "^(NAME|VERSION|ID)="
    else
        uname -a
    fi
    echo ""
    
    # Kernel
    echo "Kernel: $(uname -r)"
    echo ""
    
    # CPU
    echo "CPU:"
    if [ -f /proc/cpuinfo ]; then
        grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs
        grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs | xargs -I {} echo "  Cores: {}"
    else
        sysctl -n hw.ncpu 2>/dev/null | xargs -I {} echo "  Cores: {}"
    fi
    echo ""
    
    # Memory
    echo "Memory:"
    if command -v free &> /dev/null; then
        free -h | grep -E "^Mem:" | awk '{print "  Total: "$2", Used: "$3", Free: "$4}'
    else
        sysctl -n hw.memsize 2>/dev/null | awk '{printf "  Total: %.2f GB\n", $1/1024/1024/1024}'
    fi
    echo ""
    
    # Disk
    echo "Disk Space:"
    df -h . | tail -1 | awk '{print "  Available: "$4" on "$6}'
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "RUNTIME ENVIRONMENT"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Rust
    echo "Rust:"
    if command -v rustc &> /dev/null; then
        rustc --version
        cargo --version
    else
        echo "  Not installed"
    fi
    echo ""
    
    # Docker
    echo "Docker:"
    if command -v docker &> /dev/null; then
        docker --version
        if docker ps &> /dev/null; then
            echo "  Status: Running"
        else
            echo "  Status: Not running"
        fi
    else
        echo "  Not installed"
    fi
    echo ""
    
    # SDKWork-TTS
    echo "SDKWork-TTS:"
    if command -v sdkwork-tts &> /dev/null; then
        sdkwork-tts --version
        echo "  Location: $(which sdkwork-tts)"
    elif [ -f "$HOME/.sdkwork-tts/bin/sdkwork-tts" ]; then
        "$HOME/.sdkwork-tts/bin/sdkwork-tts" --version
        echo "  Location: $HOME/.sdkwork-tts/bin/sdkwork-tts"
    else
        echo "  Not installed"
    fi
    echo ""
    
    # Environment variables
    echo "Environment Variables:"
    env | grep -E "^(SDKWORK_TTS|RUST|CUDA|OPENAI|ALIYUN)" | sort || echo "  None set"
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "INSTALLATION CHECK"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Check installation
    INSTALL_DIR="${SDKWORK_TTS_INSTALL_DIR:-$HOME/.sdkwork-tts}"
    
    if [ -d "$INSTALL_DIR" ]; then
        echo "Installation Directory: $INSTALL_DIR"
        echo "  Contents:"
        ls -la "$INSTALL_DIR" | tail -n +2 | awk '{print "    "$9" ("$5" bytes)"}'
    else
        echo "Installation Directory: Not found"
    fi
    echo ""
    
    # Check data directory
    if [ -d "$INSTALL_DIR/data" ]; then
        echo "Data Directory: $INSTALL_DIR/data"
        SPEAKER_COUNT=$(find "$INSTALL_DIR/data/speaker_library" -name "*.json" 2>/dev/null | wc -l)
        echo "  Speakers: $SPEAKER_COUNT"
    else
        echo "Data Directory: Not found"
    fi
    echo ""
    
    # Check config
    if [ -f "$INSTALL_DIR/config/server.yaml" ]; then
        echo "Configuration: $INSTALL_DIR/config/server.yaml"
    else
        echo "Configuration: Not found"
    fi
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "GPU CHECK (if applicable)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
            echo "  $line"
        done
    else
        echo "NVIDIA GPU: Not detected"
    fi
    echo ""
    
    # CUDA
    if command -v nvcc &> /dev/null; then
        echo "CUDA: $(nvcc --version | grep release | cut -d, -f1)"
    else
        echo "CUDA: Not installed"
    fi
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "NETWORK CHECK"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # GitHub
    echo "GitHub:"
    if curl -s --connect-timeout 5 https://github.com > /dev/null 2>&1; then
        echo "  Reachable: Yes"
    else
        echo "  Reachable: No"
    fi
    
    # HuggingFace
    echo "HuggingFace:"
    if curl -s --connect-timeout 5 https://huggingface.co > /dev/null 2>&1; then
        echo "  Reachable: Yes"
    else
        echo "  Reachable: No"
    fi
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "COMMON ISSUES CHECK"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Port 8080
    echo "Port 8080:"
    if command -v ss &> /dev/null; then
        if ss -tlnp | grep -q ":8080"; then
            echo "  Status: In use"
            ss -tlnp | grep ":8080" | awk '{print "    "$7}'
        else
            echo "  Status: Free"
        fi
    else
        echo "  Status: Unknown (ss not available)"
    fi
    echo ""
    
    # Permissions
    echo "Permissions:"
    if [ -f "$INSTALL_DIR/bin/sdkwork-tts" ]; then
        if [ -x "$INSTALL_DIR/bin/sdkwork-tts" ]; then
            echo "  Binary: Executable"
        else
            echo "  Binary: Not executable (ISSUE)"
        fi
    fi
    echo ""
    
    echo "═══════════════════════════════════════════════════════════"
    echo "RECENT LOGS (if available)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Systemd logs
    if command -v journalctl &> /dev/null; then
        echo "Last 5 systemd log entries:"
        journalctl -u sdkwork-tts -n 5 --no-pager 2>/dev/null || echo "  No systemd logs found"
    fi
    echo ""
    
} | tee "$REPORT_FILE"

echo ""
echo -e "${GREEN}✓ Diagnostic report saved to: $REPORT_FILE${NC}"
echo ""
echo -e "${YELLOW}To share this report:${NC}"
echo "1. Review the report for sensitive information"
echo "2. Attach to GitHub issue or support request"
echo ""
