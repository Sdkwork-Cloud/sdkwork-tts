#!/bin/bash
# SDKWork-TTS Installation Verification Script
# Verifies that the installation is working correctly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     SDKWork-TTS Installation Verification                 ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

PASS=0
FAIL=0
WARN=0

# Test function
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ $2${NC}"
        ((FAIL++))
    fi
}

warn_result() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARN++))
}

# 1. Check binary
echo -e "${YELLOW}Checking binary...${NC}"
if command -v sdkwork-tts &> /dev/null; then
    test_result 0 "sdkwork-tts binary found"
    sdkwork-tts --version
else
    # Check local installation
    if [ -f "$HOME/.sdkwork-tts/bin/sdkwork-tts" ]; then
        test_result 0 "sdkwork-tts binary found (local)"
        "$HOME/.sdkwork-tts/bin/sdkwork-tts" --version
    else
        test_result 1 "sdkwork-tts binary not found"
        warn_result "Please install SDKWork-TTS first"
    fi
fi
echo ""

# 2. Check directories
echo -e "${YELLOW}Checking directories...${NC}"
if [ -d "$HOME/.sdkwork-tts" ]; then
    test_result 0 "Installation directory exists"
else
    test_result 1 "Installation directory not found"
fi

if [ -d "$HOME/.sdkwork-tts/data" ]; then
    test_result 0 "Data directory exists"
else
    test_result 1 "Data directory not found"
fi

if [ -d "$HOME/.sdkwork-tts/config" ]; then
    test_result 0 "Config directory exists"
else
    test_result 1 "Config directory not found"
fi
echo ""

# 3. Check environment variables
echo -e "${YELLOW}Checking environment variables...${NC}"
if [ ! -z "$SDKWORK_TTS_DATA" ]; then
    test_result 0 "SDKWORK_TTS_DATA is set"
else
    warn_result "SDKWORK_TTS_DATA not set"
fi

if [ ! -z "$SDKWORK_TTS_CONFIG" ]; then
    test_result 0 "SDKWORK_TTS_CONFIG is set"
else
    warn_result "SDKWORK_TTS_CONFIG not set"
fi
echo ""

# 4. Check Rust installation
echo -e "${YELLOW}Checking Rust installation...${NC}"
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    test_result 0 "Rust installed: $RUST_VERSION"
else
    test_result 1 "Rust not installed"
fi

if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version)
    test_result 0 "Cargo installed: $CARGO_VERSION"
else
    test_result 1 "Cargo not installed"
fi
echo ""

# 5. Check system requirements
echo -e "${YELLOW}Checking system requirements...${NC}"

# Check available memory
AVAILABLE_MEM=$(free -m 2>/dev/null | awk 'NR==2 {print $7}' || echo "0")
if [ "$AVAILABLE_MEM" -gt 4000 ]; then
    test_result 0 "Available memory: ${AVAILABLE_MEM}MB"
else
    warn_result "Low memory: ${AVAILABLE_MEM}MB (recommended: 4000+ MB)"
fi

# Check available disk space
AVAILABLE_DISK=$(df -h . 2>/dev/null | awk 'NR==2 {print $4}' || echo "Unknown")
test_result 0 "Available disk space: $AVAILABLE_DISK"

# Check OS
OS=$(uname -s)
test_result 0 "Operating system: $OS"
echo ""

# 6. Check configuration files
echo -e "${YELLOW}Checking configuration files...${NC}"
if [ -f "$HOME/.sdkwork-tts/config/server.yaml" ]; then
    test_result 0 "Configuration file exists"
else
    warn_result "Configuration file not found"
fi

if [ -f "server.example.yaml" ]; then
    test_result 0 "Example configuration exists"
else
    warn_result "Example configuration not found"
fi
echo ""

# 7. Network check (optional)
echo -e "${YELLOW}Checking network connectivity...${NC}"
if command -v curl &> /dev/null; then
    test_result 0 "curl is installed"
    
    # Try to reach GitHub (optional)
    if curl -s --connect-timeout 5 https://github.com > /dev/null 2>&1; then
        test_result 0 "GitHub is reachable"
    else
        warn_result "Cannot reach GitHub (may be behind firewall)"
    fi
else
    warn_result "curl not installed"
fi
echo ""

# 8. Docker check (optional)
echo -e "${YELLOW}Checking Docker (optional)...${NC}"
if command -v docker &> /dev/null; then
    test_result 0 "Docker is installed"
    
    if docker ps > /dev/null 2>&1; then
        test_result 0 "Docker daemon is running"
    else
        warn_result "Docker daemon is not running"
    fi
else
    warn_result "Docker not installed (optional)"
fi
echo ""

# Summary
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Verification Summary                         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Passed: $PASS${NC}"
if [ $WARN -gt 0 ]; then
    echo -e "${YELLOW}Warnings: $WARN${NC}"
fi
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL${NC}"
fi
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verification passed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Start server: sdkwork-tts server --mode local"
    echo "2. Test API: curl http://localhost:8080/health"
    echo "3. View docs: https://github.com/Sdkwork-Cloud/sdkwork-tts"
    exit 0
else
    echo -e "${RED}✗ Installation verification failed${NC}"
    echo ""
    echo -e "${YELLOW}Please fix the issues above and run again${NC}"
    exit 1
fi
