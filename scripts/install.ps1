# SDKWork-TTS Installation Script for Windows
# PowerShell

param(
    [string]$InstallDir = "$HOME\sdkwork-tts",
    [switch]$SkipBuild,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "╔═══════════════════════════════════════════════════════════╗"
Write-Info "║        SDKWork-TTS Installation Script                    ║"
Write-Info "╚═══════════════════════════════════════════════════════════╝"
Write-Host ""

# Check requirements
function Check-Requirements {
    Write-Info "Checking system requirements..."
    
    # Check Rust
    try {
        $rustVersion = rustc --version
        Write-Success "✓ Rust is installed: $rustVersion"
    } catch {
        Write-Error "✗ Rust is not installed"
        Write-Info "Please install Rust from https://rustup.rs/"
        exit 1
    }
    
    # Check Cargo
    try {
        $cargoVersion = cargo --version
        Write-Success "✓ Cargo is installed: $cargoVersion"
    } catch {
        Write-Error "✗ Cargo is not installed"
        exit 1
    }
    
    # Check disk space
    $drive = Get-PSDrive (Split-Path $InstallDir -Qualifier)
    $freeSpace = [math]::Round($drive.Free / 1GB, 2)
    Write-Success "✓ Available disk space: ${freeSpace}GB"
    
    Write-Host ""
}

# Create directories
function Create-Directories {
    Write-Info "Creating directories..."
    
    $binDir = Join-Path $InstallDir "bin"
    $dataDir = Join-Path $InstallDir "data"
    $configDir = Join-Path $InstallDir "config"
    
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    New-Item -ItemType Directory -Force -Path $dataDir\checkpoints | Out-Null
    New-Item -ItemType Directory -Force -Path $dataDir\speaker_library | Out-Null
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
    
    Write-Success "✓ Directories created"
    Write-Host ""
    
    return @{
        Bin = $binDir
        Data = $dataDir
        Config = $configDir
    }
}

# Build from source
function Build-Source {
    param($Directories)
    
    if ($SkipBuild) {
        Write-Info "Skipping build (SkipBuild flag set)"
        return
    }
    
    Write-Info "Building SDKWork-TTS from source..."
    
    # Get project root
    $projectRoot = Split-Path $PSScriptRoot -Parent
    Set-Location $projectRoot
    
    # Build in release mode
    cargo build --release --no-default-features --features cpu
    
    # Copy binary
    Copy-Item "target\release\sdkwork-tts.exe" -Destination $Directories.Bin\ -Force
    
    Write-Success "✓ Build completed"
    Write-Host ""
}

# Install configuration
function Install-Config {
    param($Directories, $ProjectRoot)
    
    Write-Info "Installing configuration files..."
    
    # Copy example config
    $exampleConfig = Join-Path $ProjectRoot "server.example.yaml"
    if (Test-Path $exampleConfig) {
        Copy-Item $exampleConfig -Destination "$($Directories.Config)\server.yaml" -Force
        Write-Success "✓ Configuration file installed"
    }
    
    # Copy startup scripts
    $startScript = Join-Path $ProjectRoot "scripts\start_server.bat"
    if (Test-Path $startScript) {
        Copy-Item $startScript -Destination "$($Directories.Bin)\" -Force
    }
    
    Write-Host ""
}

# Setup environment
function Setup-Environment {
    param($Directories)
    
    Write-Info "Setting up environment..."
    
    # Add to PATH (user level)
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$($Directories.Bin)*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$currentPath;$($Directories.Bin)",
            "User"
        )
        Write-Success "✓ Added to PATH (user level)"
    }
    
    # Set environment variables
    [Environment]::SetEnvironmentVariable("SDKWORK_TTS_DATA", $Directories.Data, "User")
    [Environment]::SetEnvironmentVariable("SDKWORK_TTS_CONFIG", $Directories.Config, "User")
    
    Write-Success "✓ Environment variables configured"
    Write-Host ""
}

# Print summary
function Print-Summary {
    param($Directories)
    
    Write-Success "╔═══════════════════════════════════════════════════════════╗"
    Write-Success "║        SDKWork-TTS Installation Complete!                 ║"
    Write-Success "╚═══════════════════════════════════════════════════════════╝"
    Write-Host ""
    Write-Info "Installation Directory: $($InstallDir)"
    Write-Info "Binary Location: $($Directories.Bin)\sdkwork-tts.exe"
    Write-Info "Data Directory: $($Directories.Data)"
    Write-Info "Config Directory: $($Directories.Config)"
    Write-Host ""
    Write-Info "Next steps:"
    Write-Info "1. Restart your PowerShell session"
    Write-Info "2. Verify installation: sdkwork-tts --version"
    Write-Info "3. Start server: sdkwork-tts server --mode local"
    Write-Info "4. View documentation: https://github.com/Sdkwork-Cloud/sdkwork-tts"
    Write-Host ""
}

# Main
$projectRoot = Split-Path $PSScriptRoot -Parent
$directories = Create-Directories
Check-Requirements
Build-Source -Directories $directories
Install-Config -Directories $directories -ProjectRoot $projectRoot
Setup-Environment -Directories $directories
Print-Summary -Directories $directories
