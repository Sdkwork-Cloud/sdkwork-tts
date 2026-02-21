#!/bin/bash
# SDKWork-TTS Common Utilities
# Shared functions for all installation scripts

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

# Progress bar
progress_bar() {
    local duration=$1
    local width=$2
    local current=$3
    
    local percent=$((current * 100 / duration))
    local filled=$((current * width / duration))
    local empty=$((width - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%" $percent
}

show_progress() {
    local total=$1
    local message=$2
    
    for ((i=0; i<=total; i++)); do
        progress_bar $total 30 $i
        sleep 0.05
    done
    echo -e "\r\033[K${GREEN}✓ $message${NC}"
}

# Spinner animation
spinner() {
    local pid=$1
    local message=$2
    local delay=0.1
    local spinstr='|/-\\'
    
    printf "\r${CYAN}[%c]${NC} $message" "${spinstr:0:1}"
    while kill -0 $pid 2>/dev/null; do
        local temp=${spinstr#?}
        printf "\r${CYAN}[%c]${NC} $message" "${spinstr:0:1}"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\r\033[K"
    done
    printf "\r${GREEN}✓${NC} $message\n"
}

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root (use sudo)"
        return 1
    fi
    return 0
}

# Check disk space (in MB)
check_disk_space() {
    local required=$1
    local available=$(df -m . | awk 'NR==2 {print $4}')
    
    if [ "$available" -lt "$required" ]; then
        log_error "Insufficient disk space. Required: ${required}MB, Available: ${available}MB"
        return 1
    fi
    log_success "Disk space check passed (${available}MB available)"
    return 0
}

# Check memory (in MB)
check_memory() {
    local required=$1
    local available
    
    if command -v free &> /dev/null; then
        available=$(free -m | awk 'NR==2 {print $7}')
    else
        available=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1024/1024}')
    fi
    
    if [ "$available" -lt "$required" ]; then
        log_warning "Low memory. Required: ${required}MB, Available: ${available}MB"
        return 1
    fi
    log_success "Memory check passed (${available}MB available)"
    return 0
}

# Download with progress
download() {
    local url=$1
    local output=$2
    
    if command -v curl &> /dev/null; then
        curl -# -L "$url" -o "$output"
    elif command -v wget &> /dev/null; then
        wget --progress=bar:force "$url" -O "$output"
    else
        log_error "Neither curl nor wget is available"
        return 1
    fi
    return 0
}

# Retry command
retry() {
    local max_attempts=$1
    local delay=$2
    local command="${@:3}"
    
    for ((i=1; i<=max_attempts; i++)); do
        if eval "$command"; then
            return 0
        fi
        log_warning "Attempt $i/$max_attempts failed, retrying in ${delay}s..."
        sleep $delay
    done
    
    log_error "Command failed after $max_attempts attempts"
    return 1
}

# Confirm action
confirm() {
    local message=$1
    local default=${2:-N}
    
    if [ "$default" = "Y" ]; then
        read -p "$message [Y/n]: " confirm
        confirm=${confirm:-Y}
    else
        read -p "$message [y/N]: " confirm
        confirm=${confirm:-N}
    fi
    
    [[ $confirm =~ ^[Yy]$ ]]
}

# Print header
print_header() {
    local title=$1
    local width=60
    local padding=$(( (width - ${#title}) / 2 ))
    
    echo ""
    printf "${CYAN}╔"
    printf '═%.0s' $(seq 1 $width)
    printf "╗${NC}\n"
    
    printf "${CYAN}║${NC}"
    printf '%*.s' $padding ''
    echo -e "${WHITE}$title${NC}"
    
    printf "${CYAN}╚"
    printf '═%.0s' $(seq 1 $width)
    printf "╝${NC}\n"
    echo ""
}

# Print section
print_section() {
    local title=$1
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$title${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Validate semver
validate_semver() {
    local version=$1
    if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

# Compare versions (returns 0 if v1 >= v2)
version_gte() {
    local v1=$1
    local v2=$2
    
    if [[ "$v1" == "$v2" ]]; then
        return 0
    fi
    
    local IFS=.
    local i v1_arr=($v1) v2_arr=($v2)
    
    for ((i=0; i<${#v1_arr[@]} || i<${#v2_arr[@]}; i++)); do
        local n1=${v1_arr[i]:-0}
        local n2=${v2_arr[i]:-0}
        
        if ((n1 > n2)); then
            return 0
        elif ((n1 < n2)); then
            return 1
        fi
    done
    
    return 0
}

# Get script directory
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]; do
        local dir="$( cd -P "$( dirname "$source" )" && pwd )"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$( cd -P "$( dirname "$source" )" && pwd )"
}

# Cleanup on exit
cleanup_on_exit() {
    trap 'log_warning "Interrupted"; exit 1' INT TERM
}

# Mark todo as completed
todo_write() {
    local file=$1
    local id=$2
    local status=$3
    
    if [ -f "$file" ]; then
        sed -i "s/\[ \] \[id:$id\]/[$status] [id:$id]/g" "$file"
    fi
}
