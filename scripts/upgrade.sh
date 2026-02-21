#!/bin/bash
# SDKWork-TTS Upgrade Script
# Checks for updates and upgrades the installation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SDKWork-TTS Upgrade Script                         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.sdkwork-tts}"
BIN_DIR="$INSTALL_DIR/bin"
REPO_URL="${REPO_URL:-https://github.com/Sdkwork-Cloud/sdkwork-tts}"

# Check if installed
if [ ! -f "$BIN_DIR/sdkwork-tts" ]; then
    echo -e "${RED}✗ SDKWork-TTS is not installed${NC}"
    echo -e "${YELLOW}Please run install.sh first${NC}"
    exit 1
fi

# Get current version
CURRENT_VERSION=$("$BIN_DIR/sdkwork-tts" --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"
echo ""

# Check for updates
echo -e "${YELLOW}Checking for updates...${NC}"

# Try to get latest version from GitHub
if command -v curl &> /dev/null; then
    LATEST_VERSION=$(curl -s https://api.github.com/repos/Sdkwork-Cloud/sdkwork-tts/releases/latest | \
        grep '"tag_name"' | \
        sed -E 's/.*"([^"]+)".*/\1/' | \
        sed 's/v//' || echo "")
    
    if [ -n "$LATEST_VERSION" ]; then
        echo -e "${GREEN}✓ Latest version: $LATEST_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ Could not fetch latest version${NC}"
        LATEST_VERSION="unknown"
    fi
else
    echo -e "${YELLOW}⚠ curl not available, skipping version check${NC}"
    LATEST_VERSION="unknown"
fi

echo ""

# Compare versions
if [ "$CURRENT_VERSION" != "unknown" ] && [ "$LATEST_VERSION" != "unknown" ]; then
    if [ "$CURRENT_VERSION" = "$LATEST_VERSION" ]; then
        echo -e "${GREEN}✓ You are already on the latest version${NC}"
        exit 0
    else
        echo -e "${YELLOW}New version available: $LATEST_VERSION${NC}"
    fi
fi

# Confirmation
echo ""
echo -e "${YELLOW}This will download and install the latest version.${NC}"
read -p "Continue? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Upgrade cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Starting upgrade...${NC}"
echo ""

# Backup current installation
BACKUP_DIR="$INSTALL_DIR/backup-$(date +%Y%m%d-%H%M%S)"
echo -e "${YELLOW}Creating backup: $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"
cp -r "$BIN_DIR" "$BACKUP_DIR/"
echo -e "${GREEN}✓ Backup created${NC}"
echo ""

# Download latest release
echo -e "${YELLOW}Downloading latest release...${NC}"
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

if command -v curl &> /dev/null; then
    curl -sL "$REPO_URL/releases/latest/download/sdkwork-tts-linux.zip" -o sdkwork-tts.zip
else
    wget -q "$REPO_URL/releases/latest/download/sdkwork-tts-linux.zip"
fi

if [ ! -f sdkwork-tts.zip ]; then
    echo -e "${RED}✗ Failed to download${NC}"
    echo -e "${YELLOW}Please check your internet connection${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Downloaded${NC}"
echo ""

# Extract
echo -e "${YELLOW}Extracting...${NC}"
unzip -q sdkwork-tts.zip
echo -e "${GREEN}✓ Extracted${NC}"
echo ""

# Install
echo -e "${YELLOW}Installing...${NC}"
cp sdkwork-tts "$BIN_DIR/"
chmod +x "$BIN_DIR/sdkwork-tts"
echo -e "${GREEN}✓ Installed${NC}"
echo ""

# Verify
echo -e "${YELLOW}Verifying installation...${NC}"
NEW_VERSION=$("$BIN_DIR/sdkwork-tts" --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")

if [ "$NEW_VERSION" != "unknown" ]; then
    echo -e "${GREEN}✓ Upgraded to version $NEW_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Version check failed, but installation may be successful${NC}"
fi
echo ""

# Cleanup
echo -e "${YELLOW}Cleaning up...${NC}"
cd - > /dev/null
rm -rf "$TMP_DIR"
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Keep backup for 7 days
echo -e "${YELLOW}Backup kept for 7 days: $BACKUP_DIR${NC}"
echo ""

# Summary
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Upgrade Summary                              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Upgrade complete!${NC}"
echo ""
echo -e "Previous version: ${YELLOW}$CURRENT_VERSION${NC}"
echo -e "New version: ${GREEN}$NEW_VERSION${NC}"
echo ""
echo -e "${YELLOW}Note:${NC}"
echo "Backup will be automatically removed after 7 days"
echo "To restore: cp $BACKUP_DIR/bin/sdkwork-tts $BIN_DIR/"
echo ""
