@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Chatterbox Fine-Tuning Environment Installer
:: https://github.com/gokhaneraslan/chatterbox-finetuning
:: ===============================

set "CHATTERBOX_TRAIN_ENV=chatterbox_train"
set "CHATTERBOX_FINETUNE_DIR=%~dp0..\training\chatterbox-finetuning"

echo.
echo =============================================
echo   Chatterbox Fine-Tuning Environment Installer
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
    echo [INFO] Chatterbox Fine-Tuning environment setup complete!
    echo [INFO] Environment name: %CHATTERBOX_TRAIN_ENV%
    echo.
) else (
    echo.
    echo [ERROR] Chatterbox Fine-Tuning environment setup failed!
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
echo [INFO] Setting up Chatterbox Fine-Tuning environment...

:: Create training directory if needed
if not exist "%~dp0..\training" (
    mkdir "%~dp0..\training"
)

:: Create environment if needed (Python 3.11 recommended)
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_TRAIN_ENV%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%CHATTERBOX_TRAIN_ENV%" with Python 3.11...
    "%CONDA_EXE%" create -n "%CHATTERBOX_TRAIN_ENV%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%CHATTERBOX_TRAIN_ENV%"
if errorlevel 1 exit /b 1

:: Upgrade pip
echo [INFO] Upgrading pip and build tools...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Install PyTorch with CUDA first
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

:: Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not verify CUDA availability
)

:: Install NumPy 1.x (required by chatterbox)
echo [INFO] Installing NumPy 1.25.2 (required by chatterbox)...
python -m pip install "numpy==1.25.2"

:: Clone or update chatterbox-finetuning repository
if exist "%CHATTERBOX_FINETUNE_DIR%\.git" (
    echo [INFO] Updating chatterbox-finetuning repository...
    cd /d "%CHATTERBOX_FINETUNE_DIR%"
    git pull
) else (
    echo [INFO] Cloning chatterbox-finetuning repository...
    if exist "%CHATTERBOX_FINETUNE_DIR%" rd /s /q "%CHATTERBOX_FINETUNE_DIR%"
    git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git "%CHATTERBOX_FINETUNE_DIR%"
)

if not exist "%CHATTERBOX_FINETUNE_DIR%" (
    echo [ERROR] Failed to clone chatterbox-finetuning repository.
    exit /b 1
)

:: Install requirements from the repo
cd /d "%CHATTERBOX_FINETUNE_DIR%"
if exist "requirements.txt" (
    echo [INFO] Installing chatterbox-finetuning requirements...
    python -m pip install -r requirements.txt
)

:: Reinstall PyTorch CUDA in case requirements pulled CPU version
echo [INFO] Ensuring PyTorch CUDA version is maintained...
python -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

:: Ensure NumPy 1.x is maintained
echo [INFO] Ensuring NumPy 1.x is maintained...
python -m pip install "numpy==1.25.2" --force-reinstall --quiet

:: Install additional dependencies
echo [INFO] Installing additional training dependencies...
python -m pip install silero-vad accelerate safetensors tensorboard

:: Run setup.py to download base models
echo [INFO] Running setup.py to download base models...
echo [INFO] This may take a while on first run...
python setup.py
if errorlevel 1 (
    echo [WARN] setup.py returned an error, but continuing...
)

:: Create directories for VoiceForge integration
echo [INFO] Creating dataset directories...
if not exist "%~dp0..\datasets\chatterbox" mkdir "%~dp0..\datasets\chatterbox"
if not exist "%~dp0..\models\chatterbox_custom" mkdir "%~dp0..\models\chatterbox_custom"

:: Final CUDA verification
echo [INFO] Verifying CUDA setup...
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}')"

:: Verify chatterbox import
echo [INFO] Verifying Chatterbox Fine-Tuning installation...
python -c "import sys; sys.path.insert(0, '.'); from src.config import Config; print('Chatterbox Fine-Tuning OK')"
if errorlevel 1 (
    echo [WARN] Chatterbox Fine-Tuning config import check failed, but installation may still work.
)

echo.
echo [INFO] Chatterbox Fine-Tuning environment setup complete!
echo [INFO] Environment name: %CHATTERBOX_TRAIN_ENV%
echo [INFO] Fine-tuning toolkit: %CHATTERBOX_FINETUNE_DIR%
echo.
echo [INFO] Next steps:
echo   1. Prepare your dataset in LJSpeech format (metadata.csv + wavs/)
echo   2. Configure src/config.py in the toolkit directory
echo   3. Run train.py to start fine-tuning
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
