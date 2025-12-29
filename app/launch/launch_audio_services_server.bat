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

call "%CONDA_BASE%\Scripts\activate.bat" "audio_services"
if errorlevel 1 (
    echo [ERROR] Failed to activate audio_services environment
    pause
    exit /b 1
)

echo [INFO] Starting Audio Services server...
echo.

python -X faulthandler -u "app\servers\audio_services_server.py" --port 8892
if errorlevel 1 (
    echo [ERROR] Audio Services server crashed!
    pause
)
pause

