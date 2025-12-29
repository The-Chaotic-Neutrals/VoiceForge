@echo off
cd /d "%~dp0..\.."

:: Find conda base directory
set "CONDA_BASE="
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~D\Scripts\conda.exe" (
        set "CONDA_BASE=%%~D"
        goto :found_conda
    )
)

:found_conda
if not defined CONDA_BASE (
    echo [ERROR] Conda not found. Please install Miniconda or ensure it's in the standard location.
    pause
    exit /b 1
)

call "%CONDA_BASE%\Scripts\activate.bat" "rvc"
if errorlevel 1 (
    echo [ERROR] Failed to activate RVC environment
    echo [INFO] Run Voice_Forge.bat ^> Utilities ^> Install All to setup the environment
    pause
    exit /b 1
)

:: Verify CUDA PyTorch
echo [INFO] Checking PyTorch CUDA...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK: {torch.cuda.get_device_name()}')"
if errorlevel 1 (
    echo [WARN] CUDA not available - RVC will run on CPU (slow)
)

echo [INFO] Starting RVC server...
echo.

python -X faulthandler -u app\servers\rvc_server.py --port 8891
if errorlevel 1 (
    echo [ERROR] RVC server crashed!
    pause
)
pause
