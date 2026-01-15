@echo off
cd /d "%~dp0"
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Configuration
:: ===============================
set "CONDA_ENV_NAME=voiceforge"
set "ASR_ENV_NAME=asr"
set "RVC_ENV_NAME=rvc"
set "AUDIO_SERVICES_ENV_NAME=audio_services"
set "CHATTERBOX_ENV_NAME=chatterbox"
set "SOPRANO_ENV_NAME=soprano"
set "SOPRANO_TRAIN_ENV_NAME=soprano_train"
set "CHATTERBOX_TRAIN_ENV_NAME=chatterbox_train"
set "REQ_FILE=%~dp0app\install\requirements_main.txt"
set "CUSTOM_DEPS=%~dp0app\assets\custom_dependencies"
set "PYTHONFAULTHANDLER=1"
set "ASR_SERVER_PORT=8889"
set "RVC_SERVER_PORT=8891"
set "AUDIO_SERVICES_SERVER_PORT=8892"
set "CHATTERBOX_SERVER_PORT=8893"
set "SOPRANO_SERVER_PORT=8894"
set "TRAINING_SERVER_PORT=8895"

:: Find conda
call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: ===============================
:: Main Menu
:: ===============================
:MENU
cls
echo.
echo =============================================
echo   VoiceForge Launcher
echo =============================================
echo   [1] Start       - Launch App + Services
echo   [2] Server      - Launch API + Services
echo   [3] Utilities   - Setup / Manage Environments
echo   [4] Exit
echo =============================================
echo.

choice /C 1234 /N /M "Choose [1-4]: "
if errorlevel 4 goto END
if errorlevel 3 goto UTILITIES_MENU
if errorlevel 2 goto RUN_SERVER
if errorlevel 1 goto RUN
goto MENU

:: ===============================
:: Utilities Menu
:: ===============================
:UTILITIES_MENU
cls
echo.
echo =============================================
echo   VoiceForge Utilities
echo =============================================
echo   [1] Update All     - Update All Environments
echo   [2] Install All    - Fresh Install All Envs
echo   [3] Delete All     - Remove All Environments
echo   [4] Training Setup - Install Training Envs
echo   [5] Back to Menu
echo =============================================
echo.

choice /C 12345 /N /M "Choose [1-5]: "
if errorlevel 5 goto MENU
if errorlevel 4 goto TRAINING_MENU
if errorlevel 3 goto DELETE_ALL_ENVS
if errorlevel 2 goto INSTALL_ALL_ENVS
if errorlevel 1 goto UPDATE_ALL
goto UTILITIES_MENU

:: ===============================
:: Training Menu
:: ===============================
:TRAINING_MENU
cls
echo.
echo =============================================
echo   TTS Training Setup
echo =============================================
echo   [1] Install Soprano Training (Soprano-Factory)
echo   [2] Install Chatterbox Training (Fine-tuning)
echo   [3] Install Both Training Envs
echo   [4] Launch Training Server
echo   [5] Back to Utilities
echo =============================================
echo.

choice /C 12345 /N /M "Choose [1-5]: "
if errorlevel 5 goto UTILITIES_MENU
if errorlevel 4 goto LAUNCH_TRAINING_SERVER
if errorlevel 3 goto INSTALL_BOTH_TRAINING
if errorlevel 2 goto INSTALL_CHATTERBOX_TRAINING
if errorlevel 1 goto INSTALL_SOPRANO_TRAINING
goto TRAINING_MENU

:INSTALL_SOPRANO_TRAINING
echo.
echo [INFO] Installing Soprano-Factory training environment...
call "%~dp0app\install\install_soprano_train.bat"
if errorlevel 1 (
    echo [ERROR] Soprano training environment setup failed.
) else (
    echo [INFO] Soprano training environment installed successfully!
)
pause
goto TRAINING_MENU

:INSTALL_CHATTERBOX_TRAINING
echo.
echo [INFO] Installing Chatterbox fine-tuning environment...
call "%~dp0app\install\install_chatterbox_train.bat"
if errorlevel 1 (
    echo [ERROR] Chatterbox training environment setup failed.
) else (
    echo [INFO] Chatterbox training environment installed successfully!
)
pause
goto TRAINING_MENU

:INSTALL_BOTH_TRAINING
echo.
echo [INFO] Installing both training environments...
echo.

call "%~dp0app\install\install_soprano_train.bat"
if errorlevel 1 (
    echo [WARN] Soprano training environment had issues.
)

call "%~dp0app\install\install_chatterbox_train.bat"
if errorlevel 1 (
    echo [WARN] Chatterbox training environment had issues.
)

echo.
echo [INFO] Training environments setup complete!
pause
goto TRAINING_MENU

:LAUNCH_TRAINING_SERVER
echo.
echo [INFO] Launching Training Server...
if not exist "%~dp0app\launch\launch_training_server.bat" (
    echo [ERROR] Training server launcher not found!
    pause
    goto TRAINING_MENU
)
start "VoiceForge Training Server" cmd /k "%~dp0app\launch\launch_training_server.bat"
echo [INFO] Training server starting on port %TRAINING_SERVER_PORT%...
timeout /t 2 /nobreak >nul
pause
goto TRAINING_MENU

:: ===============================
:: Helper: Launch Background Services
:: ===============================
:LAUNCH_SERVICES
echo.
echo [INFO] Starting background services...

:: Verify CONDA_BASE is set
if not defined CONDA_BASE (
    echo [ERROR] CONDA_BASE is not set - cannot launch services
    echo [ERROR] This should not happen - conda was found earlier
    echo [WARN] Continuing without background services...
    exit /b 0
)

echo [DEBUG] CONDA_BASE: %CONDA_BASE%
echo [DEBUG] Current directory: %CD%

:: Check and launch unified ASR server (Whisper + GLM-ASR, models loaded lazily)
echo [DEBUG] Checking for ASR environment: %ASR_ENV_NAME%
if not exist "%CONDA_BASE%\envs\%ASR_ENV_NAME%\python.exe" goto ASR_NOT_FOUND

echo [DEBUG] ASR environment found: %CONDA_BASE%\envs\%ASR_ENV_NAME%
echo [INFO] Starting unified ASR server in background...
if not exist "%~dp0app\launch\launch_asr_server.bat" (
    echo [ERROR] ASR launcher not found, skipping
    goto CHECK_AUDIO_SERVICES
)
start "VoiceForge ASR Server" cmd /k "%~dp0app\launch\launch_asr_server.bat"
echo [INFO] Unified ASR server starting...
timeout /t 2 /nobreak >nul
goto CHECK_AUDIO_SERVICES

:ASR_NOT_FOUND
echo [DEBUG] ASR environment not found - skipping
echo [INFO] Run Utilities ^> Install All to enable ASR

:CHECK_AUDIO_SERVICES
:: Check and launch Audio Services server (preprocess + postprocess + background audio)
echo [DEBUG] Checking for Audio Services environment: %AUDIO_SERVICES_ENV_NAME%
if not exist "%CONDA_BASE%\envs\%AUDIO_SERVICES_ENV_NAME%\python.exe" goto AUDIO_SERVICES_NOT_FOUND

echo [DEBUG] Audio Services environment found
if not exist "%~dp0app\launch\launch_audio_services_server.bat" goto CHECK_RVC
echo [INFO] Starting Audio Services server in background...
start "VoiceForge Audio Services Server" cmd /k "%~dp0app\launch\launch_audio_services_server.bat"
echo [INFO] Audio Services server starting...
timeout /t 2 /nobreak >nul
goto CHECK_RVC

:AUDIO_SERVICES_NOT_FOUND
echo [DEBUG] Audio Services environment not found - skipping

:CHECK_RVC
:: Check and launch RVC server (models loaded lazily on first request to save VRAM)
echo [DEBUG] Checking for RVC environment: %RVC_ENV_NAME%
if not exist "%CONDA_BASE%\envs\%RVC_ENV_NAME%\python.exe" goto RVC_NOT_FOUND

echo [DEBUG] RVC environment found
if not exist "%~dp0app\launch\launch_rvc_server.bat" goto CHECK_CHATTERBOX
echo [INFO] Starting RVC server in background...
start "VoiceForge RVC Server" cmd /k "%~dp0app\launch\launch_rvc_server.bat"
echo [INFO] RVC server starting...
timeout /t 2 /nobreak >nul
goto CHECK_CHATTERBOX

:RVC_NOT_FOUND
echo [DEBUG] RVC environment not found - skipping

:CHECK_CHATTERBOX
:: Check and launch Chatterbox-Turbo TTS server (models loaded lazily on first request to save VRAM)
echo [DEBUG] Checking for Chatterbox environment: %CHATTERBOX_ENV_NAME%
if not exist "%CONDA_BASE%\envs\%CHATTERBOX_ENV_NAME%\python.exe" goto CHATTERBOX_NOT_FOUND

echo [DEBUG] Chatterbox environment found
if not exist "%~dp0app\launch\launch_chatterbox_server.bat" goto CHECK_SOPRANO
echo [INFO] Starting Chatterbox-Turbo server in background...
start "VoiceForge Chatterbox-Turbo Server" cmd /k "%~dp0app\launch\launch_chatterbox_server.bat"
echo [INFO] Chatterbox-Turbo server starting...
timeout /t 2 /nobreak >nul
goto CHECK_SOPRANO

:CHATTERBOX_NOT_FOUND
echo [DEBUG] Chatterbox environment not found - skipping
goto CHECK_SOPRANO

::CHECK_SOPRANO
:: Check and launch Soprano TTS server
:CHECK_SOPRANO
echo [DEBUG] Checking for Soprano environment: %SOPRANO_ENV_NAME%
if not exist "%CONDA_BASE%\envs\%SOPRANO_ENV_NAME%\python.exe" goto SOPRANO_NOT_FOUND

echo [DEBUG] Soprano environment found
if not exist "%~dp0app\launch\launch_soprano_server.bat" goto SERVICES_DONE
echo [INFO] Starting Soprano server in background...
start "VoiceForge Soprano Server" cmd /k "%~dp0app\launch\launch_soprano_server.bat"
echo [INFO] Soprano server starting...
timeout /t 2 /nobreak >nul
goto SERVICES_DONE

::SOPRANO_NOT_FOUND
:SOPRANO_NOT_FOUND
echo [DEBUG] Soprano environment not found - skipping

:SERVICES_DONE
echo [INFO] Background services launch complete.
echo.
exit /b 0

:: ===============================
:: UPDATE ALL ENVIRONMENTS
:: ===============================
:UPDATE_ALL
echo.
echo [INFO] Updating all environments...
echo.

:: Update main env if exists
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %CONDA_ENV_NAME%...
    call "%~dp0app\install\install_main.bat"
    if errorlevel 1 echo [WARN] Main environment update had issues
) else (
    echo [SKIP] %CONDA_ENV_NAME% not installed
)

:: Update ASR env if exists (unified Whisper + GLM-ASR)
"%CONDA_EXE%" env list | findstr /C:"%ASR_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %ASR_ENV_NAME%...
    call "%~dp0app\install\install_asr.bat"
    if errorlevel 1 echo [WARN] ASR environment update had issues
) else (
    echo [SKIP] %ASR_ENV_NAME% not installed
)

:: Update Audio Services env if exists
"%CONDA_EXE%" env list | findstr /C:"%AUDIO_SERVICES_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %AUDIO_SERVICES_ENV_NAME%...
    call "%~dp0app\install\install_audio_services.bat"
    if errorlevel 1 echo [WARN] Audio Services environment update had issues
) else (
    echo [SKIP] %AUDIO_SERVICES_ENV_NAME% not installed
)

:: Update RVC env if exists
"%CONDA_EXE%" env list | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %RVC_ENV_NAME%...
    call "%~dp0app\install\install_rvc.bat"
    if errorlevel 1 echo [WARN] RVC environment update had issues
) else (
    echo [SKIP] %RVC_ENV_NAME% not installed
)

:: Update Chatterbox env if exists
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %CHATTERBOX_ENV_NAME%...
    call "%~dp0app\install\install_chatterbox.bat"
    if errorlevel 1 echo [WARN] Chatterbox environment update had issues
) else (
    echo [SKIP] %CHATTERBOX_ENV_NAME% not installed
)

:: Update Soprano env if exists
"%CONDA_EXE%" env list | findstr /C:"%SOPRANO_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %SOPRANO_ENV_NAME%...
    call "%~dp0app\install\install_soprano.bat"
    if errorlevel 1 echo [WARN] Soprano environment update had issues
) else (
    echo [SKIP] %SOPRANO_ENV_NAME% not installed
)

echo.
echo [INFO] Update complete!
pause
goto UTILITIES_MENU

:: ===============================
:: INSTALL ALL ENVIRONMENTS
:: ===============================
:INSTALL_ALL_ENVS
echo.
echo [INFO] Installing all environments...
echo [INFO] This will setup: %CONDA_ENV_NAME%, %ASR_ENV_NAME%, %RVC_ENV_NAME%, %AUDIO_SERVICES_ENV_NAME%, %CHATTERBOX_ENV_NAME%, %SOPRANO_ENV_NAME%
echo.
pause

:: Install main env
call "%~dp0app\install\install_main.bat"
if errorlevel 1 (
    echo [ERROR] Main environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Audio Services env (preprocess + postprocess + background audio)
call "%~dp0app\install\install_audio_services.bat"
if errorlevel 1 (
    echo [WARN] Audio Services dependency install had issues.
)

:: Install unified ASR env (Whisper + GLM-ASR in one environment)
call "%~dp0app\install\install_asr.bat"
if errorlevel 1 (
    echo [WARN] ASR environment setup had issues (optional component).
)

:: Install RVC env
call "%~dp0app\install\install_rvc.bat"
if errorlevel 1 (
    echo [ERROR] RVC environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Chatterbox-Turbo env
call "%~dp0app\install\install_chatterbox.bat"
if errorlevel 1 (
    echo [ERROR] Chatterbox-Turbo environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Soprano env
call "%~dp0app\install\install_soprano.bat"
if errorlevel 1 (
    echo [ERROR] Soprano environment setup failed.
    pause
    goto UTILITIES_MENU
)

echo.
echo [INFO] All environments installed successfully!
pause
goto UTILITIES_MENU

:: ===============================
:: DELETE ALL ENVIRONMENTS
:: ===============================
:DELETE_ALL_ENVS
echo.
echo =============================================
echo   WARNING: This will delete ALL environments!
echo =============================================
echo.
echo   Environments to be deleted:
echo     - %CONDA_ENV_NAME%
echo     - %ASR_ENV_NAME%
echo     - %AUDIO_SERVICES_ENV_NAME%
echo     - %RVC_ENV_NAME%
echo     - %CHATTERBOX_ENV_NAME%
echo     - %SOPRANO_ENV_NAME%
echo     - %SOPRANO_TRAIN_ENV_NAME% (if exists)
echo     - %CHATTERBOX_TRAIN_ENV_NAME% (if exists)
echo.
echo   Press Y to confirm, N to cancel.
echo.

choice /C YN /N /M "Delete all environments? [Y/N]: "
if errorlevel 2 goto UTILITIES_MENU

echo.
echo [INFO] Deleting environments...

:: Delete main env
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CONDA_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CONDA_ENV_NAME%" -y
)

:: Delete ASR env (unified Whisper + GLM-ASR)
"%CONDA_EXE%" env list | findstr /C:"%ASR_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %ASR_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%ASR_ENV_NAME%" -y
)

:: Delete Audio Services env
"%CONDA_EXE%" env list | findstr /C:"%AUDIO_SERVICES_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %AUDIO_SERVICES_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%AUDIO_SERVICES_ENV_NAME%" -y
)

:: Delete RVC env
"%CONDA_EXE%" env list | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %RVC_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%RVC_ENV_NAME%" -y
)

:: Delete Chatterbox env
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CHATTERBOX_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CHATTERBOX_ENV_NAME%" -y
)

:: Delete Soprano env
"%CONDA_EXE%" env list | findstr /C:"%SOPRANO_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %SOPRANO_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%SOPRANO_ENV_NAME%" -y
)

:: Delete Training envs (if they exist)
"%CONDA_EXE%" env list | findstr /C:"%SOPRANO_TRAIN_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %SOPRANO_TRAIN_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%SOPRANO_TRAIN_ENV_NAME%" -y
)

"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_TRAIN_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CHATTERBOX_TRAIN_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CHATTERBOX_TRAIN_ENV_NAME%" -y
)

echo.
echo [INFO] All environments deleted!
pause
goto UTILITIES_MENU

:: ===============================
:: RUN
:: ===============================
:RUN
echo [DEBUG] Starting RUN section...
call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 (
    echo [ERROR] Failed to activate environment "%CONDA_ENV_NAME%"
    echo [INFO] Please run Utilities ^> Install All to setup the environment
    pause
    goto MENU
)

echo [DEBUG] Environment activated successfully
echo [DEBUG] Verifying Python is accessible...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found or not working after activation
    pause
    goto MENU
)

set "PYTHONPATH=%CUSTOM_DEPS%;%PYTHONPATH%"

:: Set server URLs for main app
set "ASR_SERVER_URL=http://127.0.0.1:%ASR_SERVER_PORT%"
set "RVC_SERVER_URL=http://127.0.0.1:%RVC_SERVER_PORT%"
set "CHATTERBOX_SERVER_URL=http://127.0.0.1:%CHATTERBOX_SERVER_PORT%"
set "SOPRANO_SERVER_URL=http://127.0.0.1:%SOPRANO_SERVER_PORT%"
set "SOPRANO_SERVER_URL=http://127.0.0.1:%SOPRANO_SERVER_PORT%"

:: Launch ASR server in background (if env exists)
echo [DEBUG] About to launch background services...
echo [DEBUG] Press Ctrl+C now if you want to skip background services
timeout /t 1 /nobreak >nul 2>&1

echo [DEBUG] Calling LAUNCH_SERVICES function...
call :LAUNCH_SERVICES 2>&1
set "SERVICES_EXIT=%ERRORLEVEL%"
echo [DEBUG] LAUNCH_SERVICES returned with exit code: %SERVICES_EXIT%

if %SERVICES_EXIT% neq 0 (
    echo.
    echo =============================================
    echo [WARN] Background services launch had issues
    echo [WARN] Exit code: %SERVICES_EXIT%
    echo =============================================
    echo [WARN] This is usually OK - services are optional
    echo [INFO] Continuing with main application...
    echo [INFO] You can start services manually if needed
    echo.
    echo [DEBUG] Press any key to continue to main app...
    pause >nul
) else (
    echo [DEBUG] Background services launch completed successfully
    echo [DEBUG] Services should be running in background windows
)

echo [DEBUG] Checking if main.py exists...
cd /d "%~dp0"
if not exist "app\util\main.py" (
    echo [ERROR] main.py not found in directory: %CD%
    echo [ERROR] Expected location: %~dp0app\util\main.py
    pause
    goto MENU
)
echo [DEBUG] Found main.py in: %CD%\app\util

echo [INFO] Launching VoiceForge...
echo [INFO] ASR Server URL: %ASR_SERVER_URL% (Whisper + GLM-ASR)
echo [INFO] RVC Server URL: %RVC_SERVER_URL%
echo [INFO] Chatterbox Server URL: %CHATTERBOX_SERVER_URL%
echo [INFO] Soprano Server URL: %SOPRANO_SERVER_URL%
echo [INFO] Soprano Server URL: %SOPRANO_SERVER_URL%
echo [DEBUG] Python path: 
where python
echo [DEBUG] Current directory: %CD%
echo [DEBUG] Running: python -X faulthandler -u app\util\main.py
echo.

:: Set window title for easier identification
title VoiceForge - Running...

:: Create error log file path for reference
set "ERROR_LOG=%~dp0voiceforge_error.log"
echo [INFO] Starting VoiceForge...
echo [INFO] If errors occur, they will be displayed below.
echo [INFO] Error log backup: %ERROR_LOG%
echo.

:: Run Python with both stdout and stderr visible
:: Capture stderr explicitly to ensure errors are shown
python -X faulthandler -u app\util\main.py 2>&1
set "PYTHON_EXIT=%ERRORLEVEL%"

if %PYTHON_EXIT% neq 0 (
    echo.
    echo =============================================
    echo [ERROR] VoiceForge crashed with exit code %PYTHON_EXIT%!
    echo =============================================
    echo [INFO] Check the error messages above for details.
    echo [INFO] Common issues:
    echo   - Missing Python packages (run Utilities ^> Update All)
    echo   - Missing or corrupted environment
    echo   - Port already in use (check if another instance is running)
    echo.
    echo [INFO] Press any key to return to menu...
    pause >nul
) else (
    echo.
    echo [INFO] VoiceForge exited normally.
    pause
)

goto MENU

:: ===============================
:: RUN_SERVER
:: ===============================
:RUN_SERVER
call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 goto MENU

set "PYTHONPATH=%CUSTOM_DEPS%;%PYTHONPATH%"

:: Set server URLs
set "ASR_SERVER_URL=http://127.0.0.1:%ASR_SERVER_PORT%"
set "RVC_SERVER_URL=http://127.0.0.1:%RVC_SERVER_PORT%"
set "CHATTERBOX_SERVER_URL=http://127.0.0.1:%CHATTERBOX_SERVER_PORT%"

:: Launch ASR, RVC, and Chatterbox servers in background
call :LAUNCH_SERVICES

echo [INFO] Launching API server on port 8888...
echo [INFO] ASR Server URL: %ASR_SERVER_URL% (Whisper + GLM-ASR)
echo [INFO] RVC Server URL: %RVC_SERVER_URL%
echo [INFO] Chatterbox Server URL: %CHATTERBOX_SERVER_URL%
python -X faulthandler -u "app\servers\main_server.py" --port 8888

echo.
pause
goto MENU

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
    pause
    exit /b 1
)

set "ENV_DIR=%CONDA_BASE%\envs\%TARGET_ENV%"
if not exist "%ENV_DIR%" (
    echo [ERROR] Environment "%TARGET_ENV%" not found. Please run setup first.
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"
set "CONDA_DEFAULT_ENV=%TARGET_ENV%"
set "CONDA_PREFIX=%ENV_DIR%"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found after activation.
    pause
    exit /b 1
)
exit /b 0

:: ===============================
:: END
:: ===============================
:END
echo [INFO] Goodbye!
endlocal
exit /b 0
