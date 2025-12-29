@echo off
cd /d "%~dp0..\comfyui"

echo [INFO] Starting ComfyUI...
echo [INFO] ComfyUI will be available at http://127.0.0.1:8188
echo.

:: Use the embedded Python that comes with ComfyUI
if exist "python_embeded\python.exe" (
    .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen
) else (
    echo [ERROR] Embedded Python not found at: %CD%\python_embeded\python.exe
    echo [ERROR] Please ensure ComfyUI is properly installed in app\comfyui
    pause
    exit /b 1
)

echo.
echo If you see this and ComfyUI did not start, try updating your Nvidia Drivers to the latest.
pause
