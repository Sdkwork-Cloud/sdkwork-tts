#!/bin/bash
# SDKWork-TTS systemd Service Installation Script

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Installing SDKWork-TTS systemd service...${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)$${NC}"
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/sdkwork-tts"
SERVICE_NAME="sdkwork-tts"
SERVICE_FILE="$(dirname "$0")/sdkwork-tts.service"

# Check if SDKWork-TTS is installed
if [ ! -f "$INSTALL_DIR/bin/sdkwork-tts" ]; then
    echo -e "${RED}✗ SDKWork-TTS not found at $INSTALL_DIR${NC}"
    echo -e "${YELLOW}Please install SDKWork-TTS first${NC}"
    exit 1
fi

# Create tts user if not exists
if ! id -u tts &>/dev/null; then
    echo -e "${YELLOW}Creating tts user...${NC}"
    useradd -r -s /bin/false -d "$INSTALL_DIR" tts
fi

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R tts:tts "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR/bin/sdkwork-tts"

# Create config directory
mkdir -p /etc/sdkwork-tts

# Create environment file
if [ ! -f /etc/sdkwork-tts/env ]; then
    echo -e "${YELLOW}Creating environment file...${NC}"
    cat > /etc/sdkwork-tts/env << EOF
# SDKWork-TTS Environment Variables
# Edit this file to configure API keys and other settings

# Server mode: local, cloud, hybrid
MODE=local

# API Keys (uncomment and set your keys)
# OPENAI_API_KEY=sk-your-key-here
# ALIYUN_API_KEY=your-key-here
# ALIYUN_API_SECRET=your-secret-here

# Logging
RUST_LOG=info
EOF
    chmod 600 /etc/sdkwork-tts/env
    chown root:tts /etc/sdkwork-tts/env
fi

# Install service file
echo -e "${YELLOW}Installing service...${NC}"
cp "$SERVICE_FILE" /etc/systemd/system/$SERVICE_NAME.service
systemctl daemon-reload

# Enable service
echo -e "${YELLOW}Enabling service...${NC}"
systemctl enable $SERVICE_NAME

echo ""
echo -e "${GREEN}✓ Service installed successfully${NC}"
echo ""
echo -e "${YELLOW}Service commands:${NC}"
echo "  Start:    sudo systemctl start $SERVICE_NAME"
echo "  Stop:     sudo systemctl stop $SERVICE_NAME"
echo "  Restart:  sudo systemctl restart $SERVICE_NAME"
echo "  Status:   sudo systemctl status $SERVICE_NAME"
echo "  Logs:     sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo -e "${YELLOW}To start the service now:${NC}"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
