# SDKWork-TTS Installation Verification Script (PowerShell)
# Verifies that the installation is working correctly

$ErrorActionPreference = "Continue"

# Colors
function Write-Pass { Write-Host "✓ $args" -ForegroundColor Green }
function Write-Fail { Write-Host "✗ $args" -ForegroundColor Red }
function Write-Warn { Write-Host "⚠ $args" -ForegroundColor Yellow }
function Write-Info { Write-Host $args"..." -ForegroundColor Cyan }

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     SDKWork-TTS Installation Verification                 ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$Pass = 0
$Fail = 0
$Warn = 0

# Test function
function Test-Result {
    param($Condition, $Message)
    if ($Condition) {
        Write-Pass $Message
        $script:Pass++
    } else {
        Write-Fail $Message
        $script:Fail++
    }
}

function Test-Warn {
    param($Message)
    Write-Warn $Message
    $script:Warn++
}

# 1. Check binary
Write-Info "Checking binary"
if (Get-Command sdkwork-tts -ErrorAction SilentlyContinue) {
    Test-Result $true "sdkwork-tts binary found"
    & sdkwork-tts --version
} elseif (Test-Path "$HOME\sdkwork-tts\bin\sdkwork-tts.exe") {
    Test-Result $true "sdkwork-tts binary found (local)"
    & "$HOME\sdkwork-tts\bin\sdkwork-tts.exe" --version
} else {
    Test-Result $false "sdkwork-tts binary not found"
    Test-Warn "Please install SDKWork-TTS first"
}
Write-Host ""

# 2. Check directories
Write-Info "Checking directories"
if (Test-Path "$HOME\sdkwork-tts") {
    Test-Result $true "Installation directory exists"
} else {
    Test-Result $false "Installation directory not found"
}

if (Test-Path "$HOME\sdkwork-tts\data") {
    Test-Result $true "Data directory exists"
} else {
    Test-Result $false "Data directory not found"
}

if (Test-Path "$HOME\sdkwork-tts\config") {
    Test-Result $true "Config directory exists"
} else {
    Test-Result $false "Config directory not found"
}
Write-Host ""

# 3. Check environment variables
Write-Info "Checking environment variables"
if ($env:SDKWORK_TTS_DATA) {
    Test-Result $true "SDKWORK_TTS_DATA is set"
} else {
    Test-Warn "SDKWORK_TTS_DATA not set"
}

if ($env:SDKWORK_TTS_CONFIG) {
    Test-Result $true "SDKWORK_TTS_CONFIG is set"
} else {
    Test-Warn "SDKWORK_TTS_CONFIG not set"
}
Write-Host ""

# 4. Check Rust installation
Write-Info "Checking Rust installation"
try {
    $rustVersion = & rustc --version
    Test-Result $true "Rust installed: $rustVersion"
} catch {
    Test-Result $false "Rust not installed"
}

try {
    $cargoVersion = & cargo --version
    Test-Result $true "Cargo installed: $cargoVersion"
} catch {
    Test-Result $false "Cargo not installed"
}
Write-Host ""

# 5. Check system requirements
Write-Info "Checking system requirements"

# Check available memory
$os = Get-CimInstance Win32_OperatingSystem
$freeMem = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
if ($freeMem -gt 4) {
    Test-Result $true "Available memory: ${freeMem}GB"
} else {
    Test-Warn "Low memory: ${freeMem}GB (recommended: 4+ GB)"
}

# Check available disk space
$drive = Get-PSDrive (Split-Path $HOME -Qualifier)
$freeSpace = [math]::Round($drive.Free / 1GB, 2)
Test-Result $true "Available disk space: ${freeSpace}GB"

# Check OS
Test-Result $true "Operating system: Windows $($os.Version)"
Write-Host ""

# 6. Check configuration files
Write-Info "Checking configuration files"
if (Test-Path "$HOME\sdkwork-tts\config\server.yaml") {
    Test-Result $true "Configuration file exists"
} else {
    Test-Warn "Configuration file not found"
}

if (Test-Path "server.example.yaml") {
    Test-Result $true "Example configuration exists"
} else {
    Test-Warn "Example configuration not found"
}
Write-Host ""

# 7. Network check (optional)
Write-Info "Checking network connectivity"
try {
    $response = Invoke-WebRequest -Uri "https://github.com" -TimeoutSec 5 -UseBasicParsing
    Test-Result $true "GitHub is reachable"
} catch {
    Test-Warn "Cannot reach GitHub (may be behind firewall)"
}
Write-Host ""

# 8. Docker check (optional)
Write-Info "Checking Docker (optional)"
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Test-Result $true "Docker is installed"
    
    try {
        & docker ps | Out-Null
        Test-Result $true "Docker daemon is running"
    } catch {
        Test-Warn "Docker daemon is not running"
    }
} else {
    Test-Warn "Docker not installed (optional)"
}
Write-Host ""

# Summary
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              Verification Summary                         ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Pass "Passed: $Pass"
if ($Warn -gt 0) {
    Write-Warn "Warnings: $Warn"
}
if ($Fail -gt 0) {
    Write-Fail "Failed: $Fail"
}
Write-Host ""

if ($Fail -eq 0) {
    Write-Pass "✓ Installation verification passed!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Start server: sdkwork-tts server --mode local"
    Write-Host "2. Test API: curl http://localhost:8080/health"
    Write-Host "3. View docs: https://github.com/Sdkwork-Cloud/sdkwork-tts"
    exit 0
} else {
    Write-Fail "✗ Installation verification failed"
    Write-Host ""
    Write-Host "Please fix the issues above and run again" -ForegroundColor Yellow
    exit 1
}
