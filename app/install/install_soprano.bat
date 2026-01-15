@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Soprano TTS Environment Installer
:: https://huggingface.co/ekwek/Soprano-1.1-80M
:: ===============================

set "SOPRANO_ENV_NAME=soprano"
set "REQ_FILE=%~dp0requirements_soprano.txt"

echo.
echo =============================================
echo   Soprano TTS Environment Installer
echo =============================================
echo.

:: Find conda
call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Run installation
call :DO_INSTALL
set "INSTALL_RESULT=%ERRORLEVEL%"

if %INSTALL_RESULT% equ 0 (
    echo.
    echo [INFO] Soprano environment setup complete!
    echo [INFO] Environment name: %SOPRANO_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] Soprano environment setup failed!
    echo.
)

pause
endlocal
exit /b %INSTALL_RESULT%

:: ===============================
:: Installation Logic
:: ===============================
:DO_INSTALL
echo.
echo [INFO] Setting up Soprano TTS environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed (Python 3.11 recommended)
"%CONDA_EXE%" env list | findstr /C:"%SOPRANO_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%SOPRANO_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%SOPRANO_ENV_NAME%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%SOPRANO_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install requirements
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: NOTE: Windows CUDA users may need to reinstall PyTorch for CUDA support.
:: Soprano docs recommend torch 2.8.0 with CUDA 12.8:
:: pip uninstall -y torch
:: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

:: Quick verification
echo [INFO] Verifying Soprano installation...
python -c "from soprano import SopranoTTS; print('Soprano OK')"
if errorlevel 1 (
    echo [ERROR] Soprano import failed!
    exit /b 1
)

echo.
echo [INFO] Soprano TTS environment setup complete!
echo [INFO] Environment name: %SOPRANO_ENV_NAME%
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
:: Helper: Activate Environment
:: ===============================
:ACTIVATE_ENV
set "TARGET_ENV=%~1"
if not defined CONDA_BASE (
    echo [ERROR] Conda not found.
    exit /b 1
)

set "ENV_DIR=%CONDA_BASE%\envs\%TARGET_ENV%"
if not exist "%ENV_DIR%" (
    echo [INFO] Environment "%TARGET_ENV%" does not exist yet - will be created.
    exit /b 0
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
