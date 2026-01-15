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

call "%CONDA_BASE%\Scripts\activate.bat" "soprano"
if errorlevel 1 (
    echo [ERROR] Failed to activate Soprano environment
    echo [INFO] Run Voice_Forge.bat ^> Utilities ^> Install All to setup the environment
    pause
    exit /b 1
)

echo [INFO] Starting Soprano server...
echo.

python -X faulthandler -u app\servers\soprano_server.py --port 8894
if errorlevel 1 (
    echo [ERROR] Soprano server crashed!
    pause
)
pause
