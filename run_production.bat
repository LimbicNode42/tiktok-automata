@echo off
REM TikTok Automata Production Runner
REM Windows batch script for running the production pipeline

setlocal

REM Set UTF-8 encoding for console
chcp 65001 > nul

echo ====================================
echo TikTok Automata Production Pipeline
echo ====================================

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if production_pipeline.py exists
if not exist "production_pipeline.py" (
    echo ERROR: production_pipeline.py not found in current directory
    pause
    exit /b 1
)

REM Parse command line arguments
set DRY_RUN=
set INITIAL_SETUP=
set ARGS=

:parse_args
if "%~1"=="" goto :run_pipeline
if "%~1"=="--dry-run" (
    set DRY_RUN=--dry-run
    set ARGS=%ARGS% --dry-run
)
if "%~1"=="--initial-setup" (
    set INITIAL_SETUP=--initial-setup
    set ARGS=%ARGS% --initial-setup
)
shift
goto :parse_args

:run_pipeline
REM Show run mode
if defined DRY_RUN (
    echo Running in DRY RUN mode - no actual content will be generated
)
if defined INITIAL_SETUP (
    echo Running INITIAL SETUP - processing backlog from last week/6 months
) else (
    echo Running DAILY mode - processing content from last 24 hours
)

echo.
echo Starting pipeline...
echo.

REM Run the pipeline with proper error handling
python production_pipeline.py %ARGS%
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo Pipeline completed successfully!
) else (
    echo Pipeline failed with exit code %EXIT_CODE%
)

REM Show log location
if exist "production_pipeline.log" (
    echo.
    echo Log file: production_pipeline.log
)

echo.
pause
exit /b %EXIT_CODE%
