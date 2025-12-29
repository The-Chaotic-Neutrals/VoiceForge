# config.py - ALL configuration handling for VoiceForge
# This is the ONLY file that handles config.json

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

APP_DIR = Path(__file__).parent.parent.resolve()

# File paths
CONFIG_FILE = APP_DIR / "config" / "config.json"
VOICE_FILE = APP_DIR / "config" / "voices.json"
ASSETS_DIR = APP_DIR / "assets"
FX_DIR = APP_DIR / "assets" / "fx"
SOUNDS_DIR = APP_DIR / "assets" / "sounds"
MODEL_DIR = APP_DIR / "models" / "rvc_user"
OUTPUT_DIR = APP_DIR.parent / "output"
SCRIPT_DIR = APP_DIR / "assets" / "scripts"

# Audio
CHUNK_DURATION_SECONDS = 60

# TTS Engine (Chatterbox is the only supported engine)
TTS_ENGINE = os.getenv("TTS_ENGINE", "chatterbox")

# VRAM
VRAM_SAFETY_MARGIN_GB = os.getenv("VRAM_SAFETY_MARGIN")
VRAM_AUTO_UNLOAD = os.getenv("VRAM_AUTO_UNLOAD", "true").lower() == "true"


# =============================================================================
# CONFIG.JSON READ/WRITE
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Read and return config.json."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading config.json: {e}")
        return {}


def save_config(data: Dict[str, Any]) -> bool:
    """Save data to config.json (merges with existing)."""
    try:
        current = get_config()
        current.update(data)
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config.json: {e}")
        return False


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a single value from config."""
    return get_config().get(key, default)


# =============================================================================
# CONVENIENCE GETTERS
# =============================================================================

def is_rvc_enabled() -> bool:
    return get_config_value("enable_rvc", True)

def is_post_enabled() -> bool:
    return get_config_value("enable_post", True)

def is_background_enabled() -> bool:
    return get_config_value("enable_background", False)

def get_bg_tracks() -> list:
    return get_config_value("bg_tracks", [])

def get_tts_engine() -> str:
    return get_config_value("tts_engine", TTS_ENGINE)


# =============================================================================
# UTILITY
# =============================================================================

def ensure_dir(path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
