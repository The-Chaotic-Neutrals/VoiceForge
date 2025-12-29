@echo off
:: Chatterbox-Turbo TTS server launcher
cd /d "%~dp0.."

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

call "%CONDA_BASE%\Scripts\activate.bat" "chatterbox"
if errorlevel 1 (
    echo [ERROR] Failed to activate Chatterbox environment
    pause
    exit /b 1
)

echo [INFO] Starting Chatterbox-Turbo TTS server...
echo.

:: Run from the servers folder
cd /d "%~dp0..\servers"
python -X faulthandler -u chatterbox_server.py --port 8893
if errorlevel 1 (
    echo [ERROR] Chatterbox-Turbo server crashed!
    pause
)
pause

