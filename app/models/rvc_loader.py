# models/rvc_loader.py
"""RVC model loader and utilities."""

import os
import logging
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Track if RVC dependencies are available
_rvc_available = None


def get_models_dir() -> str:
    """Get the RVC user models directory."""
    from config import APP_DIR
    return os.path.join(APP_DIR, "models", "rvc_user")


def get_base_models_dir() -> str:
    """Get the RVC base models directory (hubert, rmvpe)."""
    from config import APP_DIR
    return os.path.join(APP_DIR, "models", "rvc_main")


def is_available() -> bool:
    """Check if local RVC dependencies are installed."""
    global _rvc_available
    
    if _rvc_available is not None:
        return _rvc_available
    
    try:
        from infer_rvc_python import BaseLoader
        _rvc_available = True
    except ImportError:
        logger.info("Local RVC dependencies not installed - will use RVC server only")
        _rvc_available = False
    
    return _rvc_available


def list_models() -> List[str]:
    """List available RVC voice models.
    
    Returns:
        List of model names (directory names that contain model.pth and model.index)
    """
    models = []
    model_dir = get_models_dir()
    
    if not os.path.exists(model_dir):
        return models
    
    for name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, name)
        if os.path.isdir(model_path):
            pth_file = os.path.join(model_path, "model.pth")
            index_file = os.path.join(model_path, "model.index")
            if os.path.exists(pth_file) and os.path.exists(index_file):
                models.append(name)
    
    return sorted(models)


def get_model_paths(model_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Get paths to model.pth and model.index for a model.
    
    Args:
        model_name: Name of the RVC model
        
    Returns:
        Tuple of (model_path, index_path) or (None, None) if not found
    """
    model_dir = get_models_dir()
    model_path = os.path.join(model_dir, model_name, "model.pth")
    index_path = os.path.join(model_dir, model_name, "model.index")
    
    if os.path.exists(model_path) and os.path.exists(index_path):
        return model_path, index_path
    
    return None, None


def get_model_info(model_name: str) -> dict:
    """Get information about an RVC model.
    
    Args:
        model_name: Name of the RVC model
        
    Returns:
        Dict with model info
    """
    model_path, index_path = get_model_paths(model_name)
    model_dir = os.path.join(get_models_dir(), model_name)
    metadata_path = os.path.join(model_dir, "metadata.json")
    
    info = {
        "name": model_name,
        "exists": model_path is not None,
        "model_path": model_path,
        "index_path": index_path,
        "metadata": None
    }
    
    # Try to load metadata if it exists
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, 'r') as f:
                info["metadata"] = json.load(f)
        except Exception:
            pass
    
    # Get file sizes
    if model_path and os.path.exists(model_path):
        info["model_size_mb"] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
    if index_path and os.path.exists(index_path):
        info["index_size_mb"] = round(os.path.getsize(index_path) / (1024 * 1024), 2)
    
    return info


def get_loader(only_cpu: bool = False):
    """
    Create a new RVC loader instance.
    
    Args:
        only_cpu: Whether to use CPU only
    
    Returns:
        BaseLoader instance
    
    Raises:
        ImportError: If RVC dependencies are not installed
    """
    if not is_available():
        raise ImportError(
            "Local RVC dependencies not installed. "
            "Either install the rvc conda environment or ensure the RVC server is running."
        )
    
    # Suppress warnings before importing
    try:
        from logging_utils import configure_warnings
        configure_warnings()
    except ImportError:
        pass
    
    from infer_rvc_python import BaseLoader
    
    # Path to rvc_main directory where hubert_base.pt and rmvpe.pt are stored
    rvc_main_dir = get_base_models_dir()
    hubert_path = os.path.join(rvc_main_dir, "hubert_base.pt")
    rmvpe_path = os.path.join(rvc_main_dir, "rmvpe.pt")
    
    # Only use these paths if the files exist
    if not os.path.exists(hubert_path):
        hubert_path = None
    if not os.path.exists(rmvpe_path):
        rmvpe_path = None
    
    return BaseLoader(only_cpu=only_cpu, hubert_path=hubert_path, rmvpe_path=rmvpe_path)
