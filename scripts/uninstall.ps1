# SDKWork-TTS Uninstall Script (PowerShell)
# Completely removes SDKWork-TTS from Windows system

param(
    [switch]$KeepData,
    [switch]$KeepConfig,
    [switch]$Force
)

$ErrorActionPreference = "Continue"

# Colors
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Red
Write-Host "║        SDKWork-TTS Uninstall Script                       ║" -ForegroundColor Red
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Red
Write-Host ""

# Confirmation
Write-Warning "WARNING: This will completely remove SDKWork-TTS from your system."
Write-Host ""

if (-not $Force) {
    $confirm = Read-Host "Are you sure you want to continue? (y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Info "Uninstallation cancelled."
        exit 0
    }
}

Write-Host ""
Write-Info "Starting uninstallation..."
Write-Host ""

# Configuration
$InstallDir = if ($env:SDKWORK_TTS_INSTALL_DIR) { $env:SDKWORK_TTS_INSTALL_DIR } else { "$HOME\sdkwork-tts" }
$BinDir = Join-Path $InstallDir "bin"
$DataDir = Join-Path $InstallDir "data"
$ConfigDir = Join-Path $InstallDir "config"

$Removed = 0
$Skipped = 0

# Remove binaries
Write-Info "Removing binaries..."
if (Test-Path "$BinDir\sdkwork-tts.exe") {
    Remove-Item "$BinDir\sdkwork-tts.exe" -Force
    Write-Success "✓ Removed: $BinDir\sdkwork-tts.exe"
    $Removed++
} else {
    Write-Warning "⚠ Not found: $BinDir\sdkwork-tts.exe"
    $Skipped++
}

if (Test-Path "$BinDir\start_server.bat") {
    Remove-Item "$BinDir\start_server.bat" -Force
    Write-Success "✓ Removed: $BinDir\start_server.bat"
    $Removed++
} else {
    Write-Warning "⚠ Not found: $BinDir\start_server.bat"
    $Skipped++
}

if (Test-Path "$BinDir\start_server.ps1") {
    Remove-Item "$BinDir\start_server.ps1" -Force
    Write-Success "✓ Removed: $BinDir\start_server.ps1"
    $Removed++
} else {
    Write-Warning "⚠ Not found: $BinDir\start_server.ps1"
    $Skipped++
}
Write-Host ""

# Remove directories
Write-Info "Removing directories..."

# Data directory
if (-not $KeepData -and (Test-Path $DataDir)) {
    Write-Warning "Data directory contains user data (models, speakers)."
    $remove = Read-Host "Do you want to remove it? (y/N)"
    
    if ($remove -eq 'y' -or $remove -eq 'Y') {
        Remove-Item $DataDir -Recurse -Force
        Write-Success "✓ Removed: $DataDir"
        $Removed++
    } else {
        Write-Warning "⚠ Skipped: $DataDir (preserved)"
        $Skipped++
    }
} elseif (Test-Path $DataDir) {
    Write-Warning "⚠ Skipped: $DataDir (KeepData flag)"
    $Skipped++
}

# Config directory
if (-not $KeepConfig -and (Test-Path $ConfigDir)) {
    Write-Warning "Config directory may contain custom configurations."
    $remove = Read-Host "Do you want to remove it? (y/N)"
    
    if ($remove -eq 'y' -or $remove -eq 'Y') {
        Remove-Item $ConfigDir -Recurse -Force
        Write-Success "✓ Removed: $ConfigDir"
        $Removed++
    } else {
        Write-Warning "⚠ Skipped: $ConfigDir (preserved)"
        $Skipped++
    }
} elseif (Test-Path $ConfigDir) {
    Write-Warning "⚠ Skipped: $ConfigDir (KeepConfig flag)"
    $Skipped++
}

# Remove installation directory
if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
    Write-Success "✓ Removed: $InstallDir"
    $Removed++
}
Write-Host ""

# Remove environment variables
Write-Info "Removing environment variables..."

# User level
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -like "*sdkwork-tts*") {
    $newPath = ($userPath -split ';' | Where-Object { $_ -notlike "*sdkwork-tts*" }) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Success "✓ Removed from PATH (User)"
    $Removed++
} else {
    Write-Warning "⚠ Not found in PATH (User)"
    $Skipped++
}

# Remove SDKWORK_TTS variables
foreach ($var in @("SDKWORK_TTS_DATA", "SDKWORK_TTS_CONFIG", "SDKWORK_TTS_INSTALL_DIR")) {
    if ([Environment]::GetEnvironmentVariable($var, "User")) {
        [Environment]::SetEnvironmentVariable($var, $null, "User")
        Write-Success "✓ Removed: $var"
        $Removed++
    }
}
Write-Host ""

# System-wide installation
$systemInstall = "C:\Program Files\sdkwork-tts"
if (Test-Path $systemInstall) {
    Write-Warning "System-wide installation found at $systemInstall"
    
    if (-not $Force) {
        $remove = Read-Host "Do you want to remove it? (requires admin) (y/N)"
        
        if ($remove -eq 'y' -or $remove -eq 'Y') {
            # Check for admin
            $isAdmin = ([Security.Principal.WindowsPrincipal] `
                [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
                [Security.Principal.WindowsBuiltInRole]::Administrator)
            
            if ($isAdmin) {
                Remove-Item $systemInstall -Recurse -Force
                Write-Success "✓ Removed: $systemInstall"
                $Removed++
                
                # Remove service
                $serviceName = "sdkwork-tts"
                if (Get-Service $serviceName -ErrorAction SilentlyContinue) {
                    Stop-Service $serviceName -Force -ErrorAction SilentlyContinue
                    sc.exe delete $serviceName
                    Write-Success "✓ Removed Windows service"
                    $Removed++
                }
            } else {
                Write-Warning "⚠ Run as Administrator to remove system installation"
                $Skipped++
            }
        }
    }
}
Write-Host ""

# Docker cleanup (optional)
Write-Info "Docker cleanup (optional)..."
if (Get-Command docker -ErrorAction SilentlyContinue) {
    if (-not $Force) {
        $removeDocker = Read-Host "Remove Docker images? (y/N)"
        
        if ($removeDocker -eq 'y' -or $removeDocker -eq 'Y') {
            docker rmi sdkwork-tts:latest-cpu 2>$null
            if ($?) {
                Write-Success "✓ Removed Docker image: sdkwork-tts:latest-cpu"
                $Removed++
            }
            
            docker rmi sdkwork-tts:latest-gpu 2>$null
            if ($?) {
                Write-Success "✓ Removed Docker image: sdkwork-tts:latest-gpu"
                $Removed++
            }
        }
    }
}
Write-Host ""

# Summary
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           Uninstallation Summary                          ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Success "Items removed: $Removed"
Write-Warning "Items skipped: $Skipped"
Write-Host ""

Write-Success "✓ Uninstallation complete!"
Write-Host ""
Write-Info "Note:"
Write-Host "1. Restart PowerShell to apply changes"
Write-Host "2. User data in $DataDir may be preserved"
Write-Host ""
