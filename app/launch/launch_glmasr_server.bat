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

call "%CONDA_BASE%\Scripts\activate.bat" "glm_asr"
if errorlevel 1 (
    echo [ERROR] Failed to activate GLM-ASR environment
    echo [INFO] Run install_glmasr.bat first to create the environment
    pause
    exit /b 1
)
python -X faulthandler -u "app\servers\glmasr_server.py" --port 8890
if errorlevel 1 (
    echo [ERROR] GLM-ASR server crashed!
    pause
)
pause

