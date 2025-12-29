# models/uvr5_loader.py
"""UVR5 audio separator model loader - supports multiple models."""

import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Track if UVR5 is available
_uvr5_available = None
_separator_instances: Dict[str, object] = {}


def get_models_dir() -> str:
    """Get the UVR5 models directory."""
    from config import APP_DIR
    return os.path.join(APP_DIR, "models", "uvr5")


def get_model_path(model_key: str = "hp5_vocals") -> str:
    """Get path to a specific model file."""
    from uvr5 import AVAILABLE_MODELS
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    return os.path.join(get_models_dir(), AVAILABLE_MODELS[model_key]["filename"])


def get_available_models() -> dict:
    """Get dictionary of all available UVR5 models."""
    from uvr5 import AVAILABLE_MODELS
    return AVAILABLE_MODELS


def is_available() -> bool:
    """Check if UVR5 dependencies are available."""
    global _uvr5_available
    
    if _uvr5_available is not None:
        return _uvr5_available
    
    try:
        import torch
        import librosa
        import soundfile
        _uvr5_available = True
    except ImportError as e:
        logger.info(f"UVR5 dependencies not available: {e}")
        _uvr5_available = False
    
    return _uvr5_available


def is_model_downloaded(model_key: str = "hp5_vocals") -> bool:
    """Check if a specific model file exists."""
    try:
        return os.path.exists(get_model_path(model_key))
    except ValueError:
        return False


def get_model_info(model_key: str = None) -> dict:
    """Get information about UVR5 model(s)."""
    from uvr5 import AVAILABLE_MODELS
    
    if model_key:
        # Info for specific model
        if model_key not in AVAILABLE_MODELS:
            return {"error": f"Unknown model: {model_key}"}
        
        model_path = get_model_path(model_key)
        exists = os.path.exists(model_path)
        info = {
            "key": model_key,
            "name": AVAILABLE_MODELS[model_key]["filename"],
            "description": AVAILABLE_MODELS[model_key]["description"],
            "exists": exists,
            "model_path": model_path if exists else None,
            "dependencies_available": is_available()
        }
        if exists:
            info["model_size_mb"] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
        return info
    
    # Info for all models
    models_info = {}
    for key, model in AVAILABLE_MODELS.items():
        model_path = os.path.join(get_models_dir(), model["filename"])
        exists = os.path.exists(model_path)
        models_info[key] = {
            "filename": model["filename"],
            "description": model["description"],
            "exists": exists,
            "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2) if exists else None
        }
    
    return {
        "dependencies_available": is_available(),
        "models_dir": get_models_dir(),
        "models": models_info
    }


def get_separator(model_key: str = "hp5_vocals", device: str = "cuda", aggression: int = 10):
    """
    Get or create a UVR5 separator instance.
    
    Args:
        model_key: Model key (hp5_vocals, deecho_aggressive, deecho_dereverb, deecho_normal)
        device: Device to use ('cuda' or 'cpu')
        aggression: Processing aggressiveness (0-20)
    
    Returns:
        UVR5Separator instance
    
    Raises:
        ImportError: If dependencies not available
    """
    global _separator_instances
    
    if not is_available():
        raise ImportError("UVR5 dependencies not available (torch, librosa, soundfile)")
    
    # Lazy import to avoid loading torch unless needed
    from uvr5 import UVR5Separator
    
    # Create instance if not cached (model will auto-download on first use)
    if model_key not in _separator_instances:
        _separator_instances[model_key] = UVR5Separator(
            model_key=model_key,
            model_dir=get_models_dir(),
            device=device,
            aggression=aggression
        )
    
    return _separator_instances[model_key]


def separate_vocals(
    audio_path: str,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    aggression: int = 10,
    model_key: str = "hp5_vocals"
) -> str:
    """
    Process audio with UVR5 (vocal separation or deecho/dereverb).
    
    Args:
        audio_path: Path to input audio
        output_dir: Directory for output (temp if None)
        device: Device to use
        aggression: Processing aggressiveness
        model_key: Which model to use
    
    Returns:
        Path to processed audio file
    """
    import tempfile
    
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    separator = get_separator(model_key=model_key, device=device, aggression=aggression)
    
    if not separator.load_model():
        raise RuntimeError(f"Failed to load UVR5 model: {model_key}")
    
    return separator.separate(audio_path, output_dir)


def unload(model_key: str = None):
    """
    Unload separator model(s) to free GPU memory.
    
    Args:
        model_key: Specific model to unload, or None to unload all
    """
    global _separator_instances
    
    if model_key:
        # Unload specific model
        if model_key in _separator_instances:
            try:
                _separator_instances[model_key].unload()
            except Exception as e:
                logger.warning(f"Error unloading UVR5 {model_key}: {e}")
            del _separator_instances[model_key]
    else:
        # Unload all models
        for key in list(_separator_instances.keys()):
            try:
                _separator_instances[key].unload()
            except Exception as e:
                logger.warning(f"Error unloading UVR5 {key}: {e}")
        _separator_instances.clear()
    
    # Clear CUDA cache
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
