@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: VoiceForge Main Environment Installer
:: ===============================

set "CONDA_ENV_NAME=voiceforge"
set "REQ_FILE=%~dp0requirements_main.txt"

echo.
echo =============================================
echo   VoiceForge Main Environment Installer
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
    echo [INFO] Main environment setup complete!
    echo.
) else (
    echo.
    echo [ERROR] Main environment setup failed!
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
echo [INFO] Setting up main conda environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed
:: Using Python 3.11 (stable, well-supported, no legacy constraints now that RVC is separate)
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%CONDA_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%CONDA_ENV_NAME%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Upgrade pip to latest
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install PyTorch CUDA explicitly (avoid pip pulling CPU wheels from PyPI)
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4...
python -m pip install --upgrade --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

:: Install requirements
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: Ensure fsspec stays compatible with datasets (prevents accidental upgrades)
echo [INFO] Ensuring fsspec is compatible with datasets...
python -m pip install --upgrade "fsspec[http]<=2025.10.0,>=2023.1.0" >nul 2>&1

:: Verify CUDA PyTorch is installed correctly
echo [INFO] Verifying CUDA PyTorch installation...
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'torch.version.cuda: {getattr(torch.version, \"cuda\", None)}'); print(f'cuda_available: {torch.cuda.is_available()}'); print(f'device_count: {torch.cuda.device_count()}'); print(f'device_0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count()>0 else \"n/a\"}')"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] CUDA available: False
    echo [INFO] If you have an NVIDIA GPU, install/update your NVIDIA driver and reboot.
    echo [INFO] If you do not have an NVIDIA GPU, this is expected and you can ignore this warning.
)

echo.
echo [INFO] Main environment setup complete!
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
