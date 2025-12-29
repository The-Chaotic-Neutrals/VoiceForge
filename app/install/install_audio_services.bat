@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: VoiceForge Audio Services Installer
:: (Preprocess + Postprocess + Background Audio)
:: ===============================

set "CONDA_ENV_NAME=audio_services"
set "REQ_FILE=%~dp0requirements_audio_services.txt"

echo.
echo =============================================
echo   VoiceForge Audio Services Installer
echo =============================================
echo   Env: %CONDA_ENV_NAME%
echo =============================================
echo.

call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda.
    pause
    exit /b 1
)

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    pause
    exit /b 1
)

:: Create environment if needed (Python 3.11)
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%CONDA_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%CONDA_ENV_NAME%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
)

call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 exit /b 1

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install PyTorch (for UVR5). Prefer CUDA 12.4 wheels.
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4 (for UVR5)...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo [INFO] Installing audio services requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

echo.
echo [INFO] Audio services requirements installed.
pause
endlocal
exit /b 0

:: ===============================
:: Helper: Find Conda
:: ===============================
:FIND_CONDA
set "CONDA_EXE="
set "CONDA_BASE="
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~D\Scripts\conda.exe" (
        set "CONDA_EXE=%%~D\Scripts\conda.exe"
        set "CONDA_BASE=%%~D"
        goto :EOF
    )
)
goto :EOF

:: ===============================
:: Helper: Activate Environment (PATH-based)
:: ===============================
:ACTIVATE_ENV
set "TARGET_ENV=%~1"
if not defined CONDA_BASE (
    echo [ERROR] Conda not found.
    exit /b 1
)

set "ENV_DIR=%CONDA_BASE%\envs\%TARGET_ENV%"
if not exist "%ENV_DIR%" (
    echo [ERROR] Environment "%TARGET_ENV%" does not exist.
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"
set "CONDA_DEFAULT_ENV=%TARGET_ENV%"
set "CONDA_PREFIX=%ENV_DIR%"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found after activation.
    exit /b 1
)
exit /b 0

