@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Chatterbox-Turbo TTS Environment Installer
:: https://huggingface.co/ResembleAI/chatterbox-turbo
:: ===============================

set "CHATTERBOX_ENV_NAME=chatterbox"
set "REQ_FILE=%~dp0requirements_chatterbox.txt"

echo.
echo =============================================
echo   Chatterbox-Turbo TTS Environment Installer
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
    echo [INFO] Chatterbox-Turbo environment setup complete!
    echo [INFO] Environment name: %CHATTERBOX_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] Chatterbox-Turbo environment setup failed!
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
echo [INFO] Setting up Chatterbox-Turbo TTS environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed (Python 3.11 as recommended)
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%CHATTERBOX_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%CHATTERBOX_ENV_NAME%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%CHATTERBOX_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Install build dependencies first
echo [INFO] Installing build dependencies...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Install PyTorch with CUDA FIRST (before chatterbox-tts can pull CPU version)
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

:: Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not verify CUDA availability
)

:: Install NumPy 1.x first (chatterbox-tts requires numpy<1.26.0)
echo [INFO] Installing NumPy 1.25.2 (required by chatterbox-tts)...
python -m pip install "numpy==1.25.2"

:: Install remaining requirements from file
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: Reinstall PyTorch CUDA in case chatterbox-tts pulled CPU version
echo [INFO] Ensuring PyTorch CUDA version is maintained...
python -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

:: Ensure NumPy 1.x is maintained (chatterbox-tts requirement)
echo [INFO] Ensuring NumPy 1.x is maintained...
python -m pip install "numpy==1.25.2" --force-reinstall --quiet

:: Final CUDA verification
echo [INFO] Verifying CUDA setup...
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] CUDA not available - Chatterbox will run on CPU (slower)
)

:: Verify Chatterbox installation
echo [INFO] Verifying Chatterbox-Turbo installation...
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('Chatterbox-Turbo OK')"
if errorlevel 1 (
    echo [ERROR] Chatterbox-Turbo import failed!
    exit /b 1
)

echo.
echo [INFO] Chatterbox-Turbo TTS environment setup complete!
echo [INFO] Environment name: %CHATTERBOX_ENV_NAME%
echo [INFO] Model will be downloaded on first use (~1.5GB)
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

