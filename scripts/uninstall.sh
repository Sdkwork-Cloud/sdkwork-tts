#!/bin/bash
# SDKWork-TTS Uninstall Script
# Completely removes SDKWork-TTS from the system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║        SDKWork-TTS Uninstall Script                       ║${NC}"
echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}WARNING: This will completely remove SDKWork-TTS from your system.${NC}"
echo ""
read -p "Are you sure you want to continue? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Uninstallation cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Starting uninstallation...${NC}"
echo ""

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.sdkwork-tts}"
BIN_DIR="$INSTALL_DIR/bin"

# Track removal
REMOVED=0
SKIPPED=0

# Remove binary
echo -e "${YELLOW}Removing binaries...${NC}"
if [ -f "$BIN_DIR/sdkwork-tts" ]; then
    rm -f "$BIN_DIR/sdkwork-tts"
    echo -e "${GREEN}✓ Removed: $BIN_DIR/sdkwork-tts${NC}"
    ((REMOVED++))
else
    echo -e "${YELLOW}⚠ Not found: $BIN_DIR/sdkwork-tts${NC}"
    ((SKIPPED++))
fi

if [ -f "$BIN_DIR/start_server.sh" ]; then
    rm -f "$BIN_DIR/start_server.sh"
    echo -e "${GREEN}✓ Removed: $BIN_DIR/start_server.sh${NC}"
    ((REMOVED++))
else
    echo -e "${YELLOW}⚠ Not found: $BIN_DIR/start_server.sh${NC}"
    ((SKIPPED++))
fi
echo ""

# Remove directories
echo -e "${YELLOW}Removing directories...${NC}"

# Keep data directory if it contains user data
if [ -d "$INSTALL_DIR/data" ]; then
    echo -e "${YELLOW}Data directory contains user data (models, speakers).${NC}"
    read -p "Do you want to remove it? (y/N): " remove_data
    
    if [[ $remove_data =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR/data"
        echo -e "${GREEN}✓ Removed: $INSTALL_DIR/data${NC}"
        ((REMOVED++))
    else
        echo -e "${YELLOW}⚠ Skipped: $INSTALL_DIR/data (preserved)${NC}"
        ((SKIPPED++))
    fi
fi

# Remove config directory
if [ -d "$INSTALL_DIR/config" ]; then
    echo -e "${YELLOW}Config directory may contain custom configurations.${NC}"
    read -p "Do you want to remove it? (y/N): " remove_config
    
    if [[ $remove_config =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR/config"
        echo -e "${GREEN}✓ Removed: $INSTALL_DIR/config${NC}"
        ((REMOVED++))
    else
        echo -e "${YELLOW}⚠ Skipped: $INSTALL_DIR/config (preserved)${NC}"
        ((SKIPPED++))
    fi
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo -e "${GREEN}✓ Removed: $INSTALL_DIR${NC}"
    ((REMOVED++))
fi
echo ""

# Remove environment variables
echo -e "${YELLOW}Removing environment variables...${NC}"

# Bash
if [ -f ~/.bashrc ]; then
    if grep -q "SDKWORK_TTS" ~/.bashrc; then
        sed -i '/# SDKWork-TTS/,/export SDKWORK_TTS_CONFIG/d' ~/.bashrc
        echo -e "${GREEN}✓ Removed from ~/.bashrc${NC}"
        ((REMOVED++))
    else
        echo -e "${YELLOW}⚠ Not found in ~/.bashrc${NC}"
        ((SKIPPED++))
    fi
fi

# Zsh
if [ -f ~/.zshrc ]; then
    if grep -q "SDKWORK_TTS" ~/.zshrc; then
        sed -i '/# SDKWork-TTS/,/export SDKWORK_TTS_CONFIG/d' ~/.zshrc
        echo -e "${GREEN}✓ Removed from ~/.zshrc${NC}"
        ((REMOVED++))
    else
        echo -e "${YELLOW}⚠ Not found in ~/.zshrc${NC}"
        ((SKIPPED++))
    fi
fi

# Fish
if [ -d ~/.config/fish ]; then
    if [ -f ~/.config/fish/config.fish ]; then
        if grep -q "SDKWORK_TTS" ~/.config/fish/config.fish; then
            sed -i '/SDKWORK_TTS/d' ~/.config/fish/config.fish
            echo -e "${GREEN}✓ Removed from ~/.config/fish/config.fish${NC}"
            ((REMOVED++))
        else
            echo -e "${YELLOW}⚠ Not found in ~/.config/fish/config.fish${NC}"
            ((SKIPPED++))
        fi
    fi
fi
echo ""

# Remove from PATH (manual instruction)
echo -e "${YELLOW}Note: You may need to manually remove $BIN_DIR from your PATH.${NC}"
echo -e "${YELLOW}Edit your shell configuration file and remove:$BIN_DIR${NC}"
echo ""

# Remove system-wide installation (if exists)
if [ -d "/opt/sdkwork-tts" ]; then
    echo -e "${YELLOW}System-wide installation found.${NC}"
    read -p "Do you want to remove it? (requires sudo) (y/N): " remove_system
    
    if [[ $remove_system =~ ^[Yy]$ ]]; then
        sudo rm -rf /opt/sdkwork-tts
        echo -e "${GREEN}✓ Removed: /opt/sdkwork-tts${NC}"
        ((REMOVED++))
        
        # Remove systemd service
        if [ -f /etc/systemd/system/sdkwork-tts.service ]; then
            sudo systemctl stop sdkwork-tts 2>/dev/null || true
            sudo systemctl disable sdkwork-tts 2>/dev/null || true
            sudo rm -f /etc/systemd/system/sdkwork-tts.service
            sudo systemctl daemon-reload
            echo -e "${GREEN}✓ Removed systemd service${NC}"
            ((REMOVED++))
        fi
    fi
fi
echo ""

# Remove Docker images (optional)
echo -e "${YELLOW}Docker cleanup (optional):${NC}"
if command -v docker &> /dev/null; then
    read -p "Remove Docker images? (y/N): " remove_docker
    
    if [[ $remove_docker =~ ^[Yy]$ ]]; then
        docker rmi sdkwork-tts:latest-cpu 2>/dev/null && \
            echo -e "${GREEN}✓ Removed Docker image: sdkwork-tts:latest-cpu${NC}" || \
            echo -e "${YELLOW}⚠ Image not found or in use${NC}"
        
        docker rmi sdkwork-tts:latest-gpu 2>/dev/null && \
            echo -e "${GREEN}✓ Removed Docker image: sdkwork-tts:latest-gpu${NC}" || \
            echo -e "${YELLOW}⚠ Image not found or in use${NC}"
        
        ((REMOVED++))
    fi
fi
echo ""

# Summary
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Uninstallation Summary                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Items removed: $REMOVED${NC}"
echo -e "${YELLOW}Items skipped: $SKIPPED${NC}"
echo ""

echo -e "${GREEN}✓ Uninstallation complete!${NC}"
echo ""
echo -e "${YELLOW}Note:${NC}"
echo "1. Restart your terminal to apply changes"
echo "2. Manually remove $BIN_DIR from PATH if needed"
echo "3. User data in $INSTALL_DIR/data may be preserved"
echo ""
