@echo off
cd /d "%~dp0..\.."

:: Unified ASR Server Launcher
:: Supports: Whisper + GLM-ASR

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

call "%CONDA_BASE%\Scripts\activate.bat" "asr"
if errorlevel 1 (
    echo [ERROR] Failed to activate ASR environment. Run install_asr.bat first.
    pause
    exit /b 1
)
echo Starting unified ASR server (Whisper + GLM-ASR)...
python -X faulthandler -u "app\servers\asr_server.py" --port 8889
if errorlevel 1 (
    echo [ERROR] ASR server crashed!
    pause
)
pause
