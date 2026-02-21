#!/bin/bash
# SDKWork-TTS Installation Script
# Supports Linux and macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.sdkwork-tts}"
BIN_DIR="$INSTALL_DIR/bin"
DATA_DIR="$INSTALL_DIR/data"
CONFIG_DIR="$INSTALL_DIR/config"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SDKWork-TTS Installation Script                    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check system requirements
check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        echo -e "${RED}✗ Rust is not installed${NC}"
        echo "Please install Rust from https://rustup.rs/"
        exit 1
    else
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        echo -e "${GREEN}✓ Rust $RUST_VERSION is installed${NC}"
    fi
    
    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}✗ Cargo is not installed${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Cargo is installed${NC}"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    echo -e "${GREEN}✓ Available disk space: $AVAILABLE_SPACE${NC}"
    
    echo ""
}

# Create directories
create_directories() {
    echo -e "${YELLOW}Creating directories...${NC}"
    
    mkdir -p "$BIN_DIR"
    mkdir -p "$DATA_DIR/checkpoints"
    mkdir -p "$DATA_DIR/speaker_library"
    mkdir -p "$CONFIG_DIR"
    
    echo -e "${GREEN}✓ Directories created${NC}"
    echo ""
}

# Build from source
build_from_source() {
    echo -e "${YELLOW}Building SDKWork-TTS from source...${NC}"
    
    # Get project root (parent of script directory)
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    
    cd "$PROJECT_ROOT"
    
    # Build in release mode
    cargo build --release --no-default-features --features cpu
    
    # Copy binary
    cp target/release/sdkwork-tts "$BIN_DIR/"
    chmod +x "$BIN_DIR/sdkwork-tts"
    
    echo -e "${GREEN}✓ Build completed${NC}"
    echo ""
}

# Install configuration files
install_config() {
    echo -e "${YELLOW}Installing configuration files...${NC}"
    
    # Copy example config
    if [ -f "$PROJECT_ROOT/server.example.yaml" ]; then
        cp "$PROJECT_ROOT/server.example.yaml" "$CONFIG_DIR/server.yaml"
        echo -e "${GREEN}✓ Configuration file installed${NC}"
    fi
    
    # Copy startup scripts
    if [ -f "$PROJECT_ROOT/scripts/start_server.sh" ]; then
        cp "$PROJECT_ROOT/scripts/start_server.sh" "$BIN_DIR/"
        chmod +x "$BIN_DIR/start_server.sh"
    fi
    
    echo ""
}

# Setup environment
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # Add to PATH
    if ! grep -q "sdkwork-tts" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# SDKWork-TTS" >> ~/.bashrc
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> ~/.bashrc
        echo "export SDKWORK_TTS_DATA=\"$DATA_DIR\"" >> ~/.bashrc
        echo "export SDKWORK_TTS_CONFIG=\"$CONFIG_DIR\"" >> ~/.bashrc
        echo -e "${GREEN}✓ Environment variables added to ~/.bashrc${NC}"
    fi
    
    if ! grep -q "sdkwork-tts" ~/.zshrc 2>/dev/null; then
        echo "" >> ~/.zshrc
        echo "# SDKWork-TTS" >> ~/.zshrc
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> ~/.zshrc
        echo "export SDKWORK_TTS_DATA=\"$DATA_DIR\"" >> ~/.zshrc
        echo "export SDKWORK_TTS_CONFIG=\"$CONFIG_DIR\"" >> ~/.zshrc
        echo -e "${GREEN}✓ Environment variables added to ~/.zshrc${NC}"
    fi
    
    echo ""
}

# Print installation summary
print_summary() {
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        SDKWork-TTS Installation Complete!                 ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Installation Directory:${NC} $INSTALL_DIR"
    echo -e "${BLUE}Binary Location:${NC} $BIN_DIR/sdkwork-tts"
    echo -e "${BLUE}Data Directory:${NC} $DATA_DIR"
    echo -e "${BLUE}Config Directory:${NC} $CONFIG_DIR"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "2. Verify installation: sdkwork-tts --version"
    echo "3. Start server: sdkwork-tts server --mode local"
    echo "4. View documentation: https://github.com/Sdkwork-Cloud/sdkwork-tts"
    echo ""
}

# Main installation process
main() {
    check_requirements
    create_directories
    build_from_source
    install_config
    setup_environment
    print_summary
}

# Run installation
main
