@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: GLM-ASR Environment Installer
:: ===============================

set "GLMASR_ENV_NAME=glm_asr"
set "REQ_FILE=%~dp0requirements_glmasr.txt"

echo.
echo =============================================
echo   GLM-ASR Environment Installer
echo =============================================
echo   Model: zai-org/GLM-ASR-Nano-2512
echo   - 1.5B parameters
echo   - Outperforms Whisper V3 on benchmarks
echo   - Excellent for low-volume/whisper speech
echo   - Great dialect support (Cantonese, etc.)
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
    echo [INFO] GLM-ASR environment setup complete!
    echo [INFO] Environment name: %GLMASR_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] GLM-ASR environment setup failed!
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
echo [INFO] Setting up GLM-ASR environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed
"%CONDA_EXE%" env list | findstr /C:"%GLMASR_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%GLMASR_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%GLMASR_ENV_NAME%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%GLMASR_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Install PyTorch with CUDA FIRST
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

:: Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not verify CUDA availability
)

:: Install remaining requirements from file
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: Install transformers from source (required for GLM-ASR-Nano-2512)
echo [INFO] Installing transformers from source (required for GLM-ASR)...
python -m pip install git+https://github.com/huggingface/transformers

:: Final CUDA verification
echo [INFO] Verifying CUDA setup...
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] CUDA not available - GLM-ASR will run on CPU (slower)
)

:: Verify transformers installation
echo [INFO] Verifying transformers installation...
python -c "from transformers import AutoProcessor, AutoModelForSeq2SeqLM; print('Transformers OK')" 2>nul
if errorlevel 1 (
    echo [WARN] Transformers may not be properly installed. GLM-ASR requires transformers from source.
)

echo.
echo [INFO] GLM-ASR environment setup complete!
echo [INFO] Environment name: %GLMASR_ENV_NAME%
echo.
echo [NOTE] The GLM-ASR model (~3GB) will be downloaded on first use.
echo        Model will be cached in: app\models\glm_asr
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

