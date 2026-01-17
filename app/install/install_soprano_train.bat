@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Soprano-Factory Training Environment Installer
:: https://github.com/ekwek1/soprano (Soprano-Factory)
:: ===============================

set "SOPRANO_TRAIN_ENV=soprano_train"
set "SOPRANO_FACTORY_DIR=%~dp0..\training\soprano-factory"

echo.
echo =============================================
echo   Soprano-Factory Training Environment Installer
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
    echo [INFO] Soprano-Factory training environment setup complete!
    echo [INFO] Environment name: %SOPRANO_TRAIN_ENV%
    echo.
) else (
    echo.
    echo [ERROR] Soprano-Factory training environment setup failed!
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
echo [INFO] Setting up Soprano-Factory training environment...

:: Create training directory if needed
if not exist "%~dp0..\training" (
    mkdir "%~dp0..\training"
)

:: Create environment if needed (Python 3.11 recommended)
"%CONDA_EXE%" env list | findstr /C:"%SOPRANO_TRAIN_ENV%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%SOPRANO_TRAIN_ENV%" with Python 3.11...
    "%CONDA_EXE%" create -n "%SOPRANO_TRAIN_ENV%" python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%SOPRANO_TRAIN_ENV%"
if errorlevel 1 exit /b 1

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Install PyTorch with CUDA first
echo [INFO] Installing PyTorch 2.8.0 with CUDA 12.8...
python -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

:: Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not verify CUDA availability
)

:: Clone or update Soprano-Factory repository (TRAINING repo, not inference)
if exist "%SOPRANO_FACTORY_DIR%\.git" (
    echo [INFO] Updating Soprano-Factory repository...
    cd /d "%SOPRANO_FACTORY_DIR%"
    git pull
) else (
    echo [INFO] Cloning Soprano-Factory training repository...
    if exist "%SOPRANO_FACTORY_DIR%" rd /s /q "%SOPRANO_FACTORY_DIR%"
    git clone https://github.com/ekwek1/soprano-factory.git "%SOPRANO_FACTORY_DIR%"
)

if not exist "%SOPRANO_FACTORY_DIR%" (
    echo [ERROR] Failed to clone Soprano-Factory repository.
    exit /b 1
)

:: Install Soprano-Factory training requirements
cd /d "%SOPRANO_FACTORY_DIR%"
echo [INFO] Installing Soprano-Factory training dependencies...
python -m pip install -r requirements.txt

:: Also install soprano inference library for loading trained models
echo [INFO] Installing soprano-tts for inference...
python -m pip install soprano-tts

:: Install PEFT for LoRA fine-tuning (preserves base model hidden states)
echo [INFO] Installing PEFT for LoRA fine-tuning...
python -m pip install peft

:: Verify installation
echo [INFO] Verifying Soprano-Factory installation...
python -c "import torch; from transformers import AutoModelForCausalLM; print('Soprano-Factory dependencies OK')"
if errorlevel 1 (
    echo [ERROR] Soprano-Factory dependencies check failed!
    exit /b 1
)

:: Check train.py exists
if not exist "%SOPRANO_FACTORY_DIR%\train.py" (
    echo [ERROR] train.py not found - wrong repository was cloned!
    exit /b 1
)
echo [INFO] train.py found - correct repository!

echo.
echo [INFO] Soprano-Factory training environment setup complete!
echo [INFO] Environment name: %SOPRANO_TRAIN_ENV%
echo [INFO] Factory location: %SOPRANO_FACTORY_DIR%
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
