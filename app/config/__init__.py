# VoiceForge Configuration - Single source of truth
# All config handling in one place

from .config import (
    # Paths
    APP_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
    FX_DIR,
    SOUNDS_DIR,
    SCRIPT_DIR,
    VOICE_FILE,
    CONFIG_FILE,
    # Constants
    TTS_ENGINE,
    CHUNK_DURATION_SECONDS,
    VRAM_SAFETY_MARGIN_GB,
    VRAM_AUTO_UNLOAD,
    # Functions
    ensure_dir,
    get_config,
    save_config,
    get_config_value,
    is_rvc_enabled,
    is_post_enabled,
    is_background_enabled,
    get_bg_tracks,
    get_tts_engine,
)

__all__ = [
    # Paths
    "APP_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "ASSETS_DIR",
    "FX_DIR",
    "SOUNDS_DIR",
    "SCRIPT_DIR",
    "VOICE_FILE",
    "CONFIG_FILE",
    # Constants
    "TTS_ENGINE",
    "CHUNK_DURATION_SECONDS",
    "VRAM_SAFETY_MARGIN_GB",
    "VRAM_AUTO_UNLOAD",
    # Functions
    "ensure_dir",
    "get_config",
    "save_config",
    "get_config_value",
    "is_rvc_enabled",
    "is_post_enabled",
    "is_background_enabled",
    "get_bg_tracks",
    "get_tts_engine",
]
