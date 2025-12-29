@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: RVC Environment Installer
:: ===============================

set "RVC_ENV_NAME=rvc"
set "REQ_FILE=%~dp0requirements_rvc.txt"
set "CUSTOM_DEPS=%CD%\app\assets\custom_dependencies"

echo.
echo =============================================
echo   RVC Environment Installer
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
    echo [INFO] RVC environment setup complete!
    echo [INFO] Environment name: %RVC_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] RVC environment setup failed!
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
echo [INFO] Setting up RVC environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed
"%CONDA_EXE%" env list | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%RVC_ENV_NAME%" with Python 3.10...
    "%CONDA_EXE%" create -n "%RVC_ENV_NAME%" python=3.10 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%RVC_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Install build dependencies first
echo [INFO] Installing build dependencies...
python -m pip install --upgrade pip wheel setuptools cython >nul 2>&1

:: ------------------------------------------------
:: pip compatibility workaround:
:: omegaconf==2.0.6 has invalid metadata that pip>=24.1 rejects.
:: Temporarily pin pip<24.1 for the requirements install, then upgrade after.
:: ------------------------------------------------
echo [INFO] Temporarily downgrading pip for legacy metadata compatibility...
python -m pip install "pip<24.1" >nul 2>&1
python -m pip --version

:: Install PyTorch 2.6.0 with CUDA 12.4 FIRST
echo [INFO] Installing PyTorch 2.6.0 with CUDA 12.4...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

:: Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed, CUDA available: {torch.cuda.is_available()}')" 2>nul

:: Install fairseq Windows wheel (pre-built, avoids compilation)
echo [INFO] Installing fairseq Windows wheel...
python -m pip install --no-deps https://github.com/daswer123/fairseq-windows-prebuild/releases/download/0.12.2/fairseq-0.12.2-cp310-cp310-win_amd64.whl >nul 2>&1

:: Install requirements from file
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: Upgrade pip back to latest after installing legacy packages
echo [INFO] Upgrading pip back to latest...
python -m pip install --upgrade pip >nul 2>&1
python -m pip --version

:: Install the local infer_rvc_python module
echo [INFO] Installing local infer_rvc_python module...
if exist "%CUSTOM_DEPS%\infer_rvc_python\setup.py" (
    python -m pip install -e "%CUSTOM_DEPS%\infer_rvc_python" --no-deps
) else (
    echo [WARN] infer_rvc_python module not found at %CUSTOM_DEPS%\infer_rvc_python
)

:: Reinstall PyTorch CUDA in case other packages overwrote it
echo [INFO] Ensuring PyTorch CUDA version is maintained...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

:: Ensure NumPy 1.x is maintained (faiss requirement)
python -m pip install "numpy==1.26.4" --force-reinstall >nul 2>&1

:: Final CUDA verification
echo [INFO] Final verification...
python -c "import torch; cuda_ok = torch.cuda.is_available(); dev = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'; print(f'PyTorch {torch.__version__} CUDA: {cuda_ok}, Device: {dev}')"
if errorlevel 1 (
    echo [ERROR] CUDA verification failed - PyTorch may not have GPU support.
    echo [INFO] Try: pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo [INFO] Verifying key packages...
python -c "import faiss; import torch; from infer_rvc_python import BaseLoader; print('RVC packages OK')"
if errorlevel 1 (
    echo [WARN] Some RVC packages may not be properly installed
)

echo.
echo [INFO] RVC environment setup complete!
echo [INFO] Environment: %RVC_ENV_NAME%
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
