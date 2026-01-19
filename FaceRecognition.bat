@echo off
REM Face Recognition Application Launcher
REM This script runs the face recognition application with camera selector

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Suppress TensorFlow warnings
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

REM Find Python executable
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i

if "%PYTHON_EXE%"=="" (
    echo Error: Python not found
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

REM Run the main application with camera selector
echo.
echo Starting Face Recognition Application...
echo.
"%PYTHON_EXE%" main.py %*

exit /b %errorlevel%
