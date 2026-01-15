@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: VoiceForge Training Server Launcher
:: ===============================

set "ENV_NAME=voiceforge"
set "SERVER_SCRIPT=app\servers\training_server.py"
set "HOST=127.0.0.1"
set "PORT=8895"

echo.
echo =============================================
echo   VoiceForge Training Server
echo =============================================
echo.

:: Find conda
call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda.
    pause
    exit /b 1
)

:: Activate environment
call :ACTIVATE_ENV "%ENV_NAME%"
if errorlevel 1 (
    echo [ERROR] Failed to activate environment %ENV_NAME%
    pause
    exit /b 1
)

:: Check if server script exists
if not exist "%SERVER_SCRIPT%" (
    echo [ERROR] Server script not found: %SERVER_SCRIPT%
    pause
    exit /b 1
)

echo [INFO] Starting Training Server on %HOST%:%PORT%
echo.

:: Run server
python "%SERVER_SCRIPT%" --host %HOST% --port %PORT%

pause
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
    echo [ERROR] Environment "%TARGET_ENV%" does not exist.
    exit /b 1
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
