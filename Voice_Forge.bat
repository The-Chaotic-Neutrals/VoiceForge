@echo off
cd /d "%~dp0"
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Configuration
:: ===============================
set "CONDA_ENV_NAME=voiceforge"
set "WHISPER_ENV_NAME=whisper_asr"
set "RVC_ENV_NAME=rvc"
set "AUDIO_SERVICES_ENV_NAME=audio_services"
set "CHATTERBOX_ENV_NAME=chatterbox"
set "REQ_FILE=%~dp0app\install\requirements_main.txt"
set "CUSTOM_DEPS=%~dp0app\assets\custom_dependencies"
set "PYTHONFAULTHANDLER=1"
set "ASR_SERVER_PORT=8889"
set "RVC_SERVER_PORT=8891"
set "AUDIO_SERVICES_SERVER_PORT=8892"
set "CHATTERBOX_SERVER_PORT=8893"

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
echo   [4] Back to Menu
echo =============================================
echo.

choice /C 1234 /N /M "Choose [1-4]: "
if errorlevel 4 goto MENU
if errorlevel 3 goto DELETE_ALL_ENVS
if errorlevel 2 goto INSTALL_ALL_ENVS
if errorlevel 1 goto UPDATE_ALL
goto UTILITIES_MENU

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

:: Check and launch Whisper ASR server (models loaded lazily on first request to save VRAM)
echo [DEBUG] Checking for Whisper ASR environment: %WHISPER_ENV_NAME%
echo [DEBUG] Running conda env list command...

:: Verify conda exe exists
if not exist "%CONDA_EXE%" (
    echo [ERROR] Conda executable not found: %CONDA_EXE%
    echo [ERROR] Skipping ASR server check
    goto CHECK_AUDIO_SERVICES
)

:: Try using conda.bat if available (more reliable on Windows)
set "CONDA_CMD=%CONDA_EXE%"
if exist "%CONDA_BASE%\Scripts\conda.bat" (
    set "CONDA_CMD=%CONDA_BASE%\Scripts\conda.bat"
    echo [DEBUG] Using conda.bat instead of conda.exe
)

:: Try to run conda env list, but fall back to directory check if it fails
echo [DEBUG] Checking for ASR environment using conda command...
"%CONDA_CMD%" env list 2>&1 | findstr /C:"%WHISPER_ENV_NAME%" >nul 2>&1
set "ASR_ENV_FOUND=%ERRORLEVEL%"

:: If conda command failed, try checking directory directly
if %ASR_ENV_FOUND% neq 0 (
    echo [DEBUG] Conda command failed or env not in list, checking directory...
    if exist "%CONDA_BASE%\envs\%WHISPER_ENV_NAME%" (
        echo [DEBUG] ASR environment directory found: %CONDA_BASE%\envs\%WHISPER_ENV_NAME%
        set "ASR_ENV_FOUND=0"
    ) else (
        echo [DEBUG] ASR environment directory not found
    )
)

if %ASR_ENV_FOUND% equ 0 (
    echo [DEBUG] Whisper ASR environment found in conda list
    echo [INFO] Whisper ASR environment found, starting ASR server in background...
    
    :: Check if launcher batch file exists
    if not exist "%~dp0app\launch\launch_asr_server.bat" (
        echo [ERROR] ASR launcher batch file not found: %~dp0app\launch\launch_asr_server.bat
        echo [ERROR] Skipping ASR server launch
        goto CHECK_AUDIO_SERVICES
    )
    
    echo [DEBUG] Launching Whisper ASR server using launch_asr_server.bat...
    start "VoiceForge Whisper ASR Server" cmd /k "%~dp0app\launch\launch_asr_server.bat"
    if errorlevel 1 (
        echo [ERROR] Failed to start Whisper ASR server process
    ) else (
        echo [INFO] Whisper ASR server starting in background window...
        timeout /t 2 /nobreak >nul
    )
) else (
    echo [WARN] ASR environment "%WHISPER_ENV_NAME%" not found in conda environments
    echo [DEBUG] This is normal if you haven't installed ASR support yet
    echo [INFO] Run Utilities ^> Install All to enable Whisper ASR
    echo [DEBUG] Continuing without Whisper ASR server...
)

:CHECK_AUDIO_SERVICES
:: Check and launch Audio Services server (preprocess + postprocess + background audio)
echo [DEBUG] Checking for Audio Services environment: %AUDIO_SERVICES_ENV_NAME%

:: Use same CONDA_CMD as set above (or set it again)
if not defined CONDA_CMD (
    set "CONDA_CMD=%CONDA_EXE%"
    if exist "%CONDA_BASE%\Scripts\conda.bat" (
        set "CONDA_CMD=%CONDA_BASE%\Scripts\conda.bat"
    )
)

echo [DEBUG] Checking for Audio Services environment using conda command...
"%CONDA_CMD%" env list 2>&1 | findstr /C:"%AUDIO_SERVICES_ENV_NAME%" >nul 2>&1
set "AUDIO_SERVICES_ENV_FOUND=%ERRORLEVEL%"

if %AUDIO_SERVICES_ENV_FOUND% neq 0 (
    echo [DEBUG] Conda command failed or env not in list, checking directory...
    if exist "%CONDA_BASE%\envs\%AUDIO_SERVICES_ENV_NAME%" (
        echo [DEBUG] Audio Services environment directory found: %CONDA_BASE%\envs\%AUDIO_SERVICES_ENV_NAME%
        set "AUDIO_SERVICES_ENV_FOUND=0"
    ) else (
        echo [DEBUG] Audio Services environment directory not found
    )
)

if %AUDIO_SERVICES_ENV_FOUND% equ 0 (
    if exist "%~dp0app\servers\audio_services_server.py" (
        if exist "%~dp0app\launch\launch_audio_services_server.bat" (
            echo [INFO] Audio Services environment found, starting Audio Services server in background...
            start "VoiceForge Audio Services Server" cmd /k "%~dp0app\launch\launch_audio_services_server.bat"
            if errorlevel 1 (
                echo [ERROR] Failed to start Audio Services server process
            ) else (
                echo [INFO] Audio Services server starting in background window...
                timeout /t 2 /nobreak >nul
            )
        ) else (
            echo [WARN] Audio Services launcher batch file not found: %~dp0app\launch\launch_audio_services_server.bat
            echo [WARN] Skipping Audio Services server launch
        )
    ) else (
        echo [WARN] Audio Services server script not found at: %~dp0app\servers\audio_services_server.py
        echo [WARN] Skipping Audio Services server
    )
) else (
    echo [WARN] Audio Services environment not found - skipping Audio Services server
    echo [INFO] Run Utilities ^> Install All to enable Audio Services
)

:CHECK_RVC
:: Check and launch RVC server (models loaded lazily on first request to save VRAM)
echo [DEBUG] Checking for RVC environment: %RVC_ENV_NAME%

:: Use same CONDA_CMD as set above (or set it again)
if not defined CONDA_CMD (
    set "CONDA_CMD=%CONDA_EXE%"
    if exist "%CONDA_BASE%\Scripts\conda.bat" (
        set "CONDA_CMD=%CONDA_BASE%\Scripts\conda.bat"
    )
)

:: Try to run conda env list, but fall back to directory check if it fails
echo [DEBUG] Checking for RVC environment using conda command...
"%CONDA_CMD%" env list 2>&1 | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
set "RVC_ENV_FOUND=%ERRORLEVEL%"

:: If conda command failed, try checking directory directly
if %RVC_ENV_FOUND% neq 0 (
    echo [DEBUG] Conda command failed or env not in list, checking directory...
    if exist "%CONDA_BASE%\envs\%RVC_ENV_NAME%" (
        echo [DEBUG] RVC environment directory found: %CONDA_BASE%\envs\%RVC_ENV_NAME%
        set "RVC_ENV_FOUND=0"
    ) else (
        echo [DEBUG] RVC environment directory not found
    )
)

if %RVC_ENV_FOUND% equ 0 (
    echo [DEBUG] RVC environment found in conda list
    if exist "%~dp0app\servers\rvc_server.py" (
        echo [INFO] RVC environment found, starting RVC server in background...
        
        :: Check if launcher batch file exists
        if not exist "%~dp0app\launch\launch_rvc_server.bat" (
            echo [ERROR] RVC launcher batch file not found: %~dp0app\launch\launch_rvc_server.bat
            echo [ERROR] Skipping RVC server launch
            goto CHECK_CHATTERBOX
        )
        
        echo [DEBUG] Launching RVC server using launch_rvc_server.bat...
        start "VoiceForge RVC Server" cmd /k "%~dp0app\launch\launch_rvc_server.bat"
        if errorlevel 1 (
            echo [ERROR] Failed to start RVC server process
        ) else (
            echo [INFO] RVC server starting in background window...
            timeout /t 2 /nobreak >nul
        )
    ) else (
        echo [WARN] RVC server script not found at: %~dp0app\servers\rvc_server.py
        echo [WARN] Skipping RVC server
    )
) else (
    echo [WARN] RVC environment not found - skipping RVC server
    echo [INFO] Run Utilities ^> Install All to enable RVC
)

:CHECK_CHATTERBOX
:: Check and launch Chatterbox-Turbo TTS server (models loaded lazily on first request to save VRAM)
echo [DEBUG] Checking for Chatterbox environment: %CHATTERBOX_ENV_NAME%

:: Use same CONDA_CMD as set above (or set it again)
if not defined CONDA_CMD (
    set "CONDA_CMD=%CONDA_EXE%"
    if exist "%CONDA_BASE%\Scripts\conda.bat" (
        set "CONDA_CMD=%CONDA_BASE%\Scripts\conda.bat"
    )
)

:: Try to run conda env list, but fall back to directory check if it fails
echo [DEBUG] Checking for Chatterbox environment using conda command...
"%CONDA_CMD%" env list 2>&1 | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
set "CHATTERBOX_ENV_FOUND=%ERRORLEVEL%"

:: If conda command failed, try checking directory directly
if %CHATTERBOX_ENV_FOUND% neq 0 (
    echo [DEBUG] Conda command failed or env not in list, checking directory...
    if exist "%CONDA_BASE%\envs\%CHATTERBOX_ENV_NAME%" (
        echo [DEBUG] Chatterbox environment directory found: %CONDA_BASE%\envs\%CHATTERBOX_ENV_NAME%
        set "CHATTERBOX_ENV_FOUND=0"
    ) else (
        echo [DEBUG] Chatterbox environment directory not found
    )
)

if %CHATTERBOX_ENV_FOUND% equ 0 (
    echo [DEBUG] Chatterbox environment found in conda list
    if exist "%~dp0app\servers\chatterbox_server.py" (
        echo [INFO] Chatterbox environment found, starting Chatterbox-Turbo server in background...
        
        :: Check if launcher batch file exists
        if not exist "%~dp0app\launch\launch_chatterbox_server.bat" (
            echo [ERROR] Chatterbox launcher batch file not found: %~dp0app\launch\launch_chatterbox_server.bat
            echo [ERROR] Skipping Chatterbox server launch
            goto SERVICES_DONE
        )
        
        echo [DEBUG] Launching Chatterbox-Turbo server using launch_chatterbox_server.bat...
        start "VoiceForge Chatterbox-Turbo Server" cmd /k "%~dp0app\launch\launch_chatterbox_server.bat"
        if errorlevel 1 (
            echo [ERROR] Failed to start Chatterbox-Turbo server process
        ) else (
            echo [INFO] Chatterbox-Turbo server starting in background window...
            timeout /t 2 /nobreak >nul
        )
    ) else (
        echo [WARN] Chatterbox server script not found at: %~dp0app\servers\chatterbox_server.py
        echo [WARN] Skipping Chatterbox-Turbo server
    )
) else (
    echo [WARN] Chatterbox environment not found - skipping Chatterbox-Turbo server
    echo [INFO] Run Utilities ^> Install All to enable Chatterbox-Turbo TTS
)

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

:: Update Whisper env if exists
"%CONDA_EXE%" env list | findstr /C:"%WHISPER_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %WHISPER_ENV_NAME%...
    call "%~dp0app\install\install_whisper.bat"
    if errorlevel 1 echo [WARN] Whisper environment update had issues
) else (
    echo [SKIP] %WHISPER_ENV_NAME% not installed
)

:: Update Postprocess env if exists
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
echo [INFO] This will setup: %CONDA_ENV_NAME%, %WHISPER_ENV_NAME%, %RVC_ENV_NAME%, %AUDIO_SERVICES_ENV_NAME%, %CHATTERBOX_ENV_NAME%
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

:: Install Whisper env
call "%~dp0app\install\install_whisper.bat"
if errorlevel 1 (
    echo [ERROR] Whisper environment setup failed.
    pause
    goto UTILITIES_MENU
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
echo     - %WHISPER_ENV_NAME%
echo     - %AUDIO_SERVICES_ENV_NAME%
echo     - %RVC_ENV_NAME%
echo     - %CHATTERBOX_ENV_NAME%
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

:: Delete Whisper env
"%CONDA_EXE%" env list | findstr /C:"%WHISPER_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %WHISPER_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%WHISPER_ENV_NAME%" -y
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
echo [INFO] ASR Server URL: %ASR_SERVER_URL%
echo [INFO] RVC Server URL: %RVC_SERVER_URL%
echo [INFO] Chatterbox Server URL: %CHATTERBOX_SERVER_URL%
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
echo [INFO] ASR Server URL: %ASR_SERVER_URL%
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
