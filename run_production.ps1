# TikTok Automata Production Runner
# PowerShell script for running the production pipeline

param(
    [switch]$DryRun,
    [switch]$InitialSetup,
    [switch]$Help
)

# Set console to UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Show-Help {
    Write-Host @"
TikTok Automata Production Pipeline Runner

Usage:
    .\run_production.ps1                     # Normal daily run
    .\run_production.ps1 -DryRun             # Test run without generating content
    .\run_production.ps1 -InitialSetup       # First-time setup (process backlog)
    .\run_production.ps1 -InitialSetup -DryRun # Test initial setup

Options:
    -DryRun         Run in test mode without generating actual content
    -InitialSetup   Process backlog (1 week articles, 6 months videos)
    -Help           Show this help message

"@
}

if ($Help) {
    Show-Help
    exit 0
}

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "TikTok Automata Production Pipeline" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Using: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if production_pipeline.py exists
if (-not (Test-Path "production_pipeline.py")) {
    Write-Host "ERROR: production_pipeline.py not found in current directory" -ForegroundColor Red
    exit 1
}

# Build arguments
$args = @()
if ($DryRun) {
    $args += "--dry-run"
    Write-Host "Running in DRY RUN mode - no actual content will be generated" -ForegroundColor Yellow
}
if ($InitialSetup) {
    $args += "--initial-setup"
    Write-Host "Running INITIAL SETUP - processing backlog from last week/6 months" -ForegroundColor Yellow
} else {
    Write-Host "Running DAILY mode - processing content from last 24 hours" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting pipeline..." -ForegroundColor Cyan
Write-Host ""

# Run the pipeline
try {
    $process = Start-Process -FilePath "python" -ArgumentList ("production_pipeline.py", $args) -Wait -PassThru -NoNewWindow
    $exitCode = $process.ExitCode
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "Pipeline completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Pipeline failed with exit code $exitCode" -ForegroundColor Red
    }
} catch {
    Write-Host "ERROR running pipeline: $_" -ForegroundColor Red
    exit 1
}

# Show log location
if (Test-Path "production_pipeline.log") {
    Write-Host ""
    Write-Host "Log file: production_pipeline.log" -ForegroundColor Blue
}

# Show output directory
if (Test-Path "production_output") {
    $outputFiles = Get-ChildItem "production_output" -Filter "*.mp4" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($outputFiles) {
        Write-Host ""
        Write-Host "Recent output files:" -ForegroundColor Blue
        foreach ($file in $outputFiles) {
            Write-Host "  📹 $($file.Name)" -ForegroundColor Cyan
        }
    }
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

exit $exitCode
