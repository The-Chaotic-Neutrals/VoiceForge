# main.py - VoiceForge Web UI Entry Point
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    try:
        # Try to set UTF-8 encoding for stdout/stderr (Python 3.7+)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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


def _safe_print(msg: str) -> None:
    """Print helper that won't crash on Windows consoles with limited encoding."""
    try:
        print(msg)
    except (UnicodeEncodeError, UnicodeError):
        # Strip/replace non-ASCII characters as a last resort
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii", errors="replace"))
        except Exception:
            # Give up quietly; printing must never kill startup
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VoiceForge Web UI")

    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")

    # IMPORTANT: These flags fix issues when running behind an HTTPS reverse proxy
    # (e.g., tailscale serve / nginx / caddy). They allow FastAPI/Starlette to
    # respect X-Forwarded-* headers so request.url.scheme becomes "https".
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        default=True,
        help="Trust X-Forwarded-* headers from a reverse proxy (default: enabled).",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="*",
        help="Which proxy IPs to trust for forwarded headers (default: '*').",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        help="Uvicorn log level (default: info).",
    )

    args = parser.parse_args()

    _safe_print("🎙️ Starting VoiceForge Web UI...")
    _safe_print(f"Local URL (direct): http://localhost:{args.port}")
    _safe_print("If you're using Tailscale Serve HTTPS, access it via your tailnet HTTPS URL.")

    uvicorn.run(
        main_server.app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # ✅ Critical for HTTPS behind reverse proxies like Tailscale Serve
        proxy_headers=bool(args.proxy_headers),
        forwarded_allow_ips=str(args.forwarded_allow_ips),
    )


if __name__ == "__main__":
    main()
