# main.py - VoiceForge Web UI Entry Point
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    try:
        # Try to set UTF-8 encoding for stdout/stderr (Python 3.7+)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # If reconfigure fails, we'll handle encoding errors in print statements
        pass

# This file is in app/util/, so we navigate up to find directories
SCRIPT_DIR = Path(__file__).parent  # app/util
APP_DIR = SCRIPT_DIR.parent  # app
ROOT_DIR = APP_DIR.parent  # VoiceForge root
CONFIG_DIR = APP_DIR / "config"  # app/config
MODELS_DIR = APP_DIR / "models"  # app/models
SERVERS_DIR = APP_DIR / "servers"  # app/servers

# Add directories to path so imports work
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(SERVERS_DIR))

# Configure warnings and logging before importing libraries
from logging_utils import configure_warnings, configure_logging
configure_warnings()
configure_logging()

# Change to app directory for relative paths
os.chdir(APP_DIR)

# Import and run the web server
import main_server
import uvicorn

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VoiceForge Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    args = parser.parse_args()
    
    # Print startup message (handle encoding errors on Windows)
    try:
        print("🎙️ Starting VoiceForge Web UI...")
    except (UnicodeEncodeError, UnicodeError):
        # Fallback for Windows console that doesn't support emoji
        print("Starting VoiceForge Web UI...")
    print(f"Open your browser to http://localhost:{args.port}")
    uvicorn.run(
        main_server.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
