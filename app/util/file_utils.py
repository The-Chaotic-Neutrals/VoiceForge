# file_utils.py
"""
Centralized file loading and path resolution for VoiceForge UI/API.

ALL file loading logic for the UI and API lives here:
- Path resolution and validation
- Model file discovery
- Audio file discovery
- Asset file loading
- Configuration file loading
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

# Import config paths
from config import (
    APP_DIR, 
    FX_DIR, 
    SOUNDS_DIR, 
    MODEL_DIR, 
    ASSETS_DIR,
    OUTPUT_DIR,
    SCRIPT_DIR,
    CONFIG_FILE,
)


# ============================================
# AUDIO FILE EXTENSIONS
# ============================================

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus')
WAV_EXTENSIONS = ('.wav',)
LOSSLESS_EXTENSIONS = ('.wav', '.flac')


# ============================================
# PATH RESOLUTION
# ============================================

def resolve_path(path: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Resolve a file path, checking multiple directories.
    
    Args:
        path: Relative or absolute path to file
        search_dirs: Optional list of directories to search
    
    Returns:
        Resolved absolute path if found, None otherwise
    """
    if not path or not path.strip():
        return None
    
    path = path.strip()
    
    # If absolute path exists, return it
    if os.path.isabs(path) and os.path.exists(path):
        return path
    
    # If relative path exists, return absolute version
    if os.path.exists(path):
        return os.path.abspath(path)
    
    # Search in provided directories
    if search_dirs:
        for dir_path in search_dirs:
            if not os.path.exists(dir_path):
                continue
            
            full = os.path.join(dir_path, path)
            if os.path.exists(full):
                return os.path.abspath(full)
    
    return None


def resolve_audio_path(path: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Resolve an audio file path, checking multiple directories.
    
    Args:
        path: Relative or absolute path to audio file
        search_dirs: Optional list of directories to search (defaults to FX_DIR and SOUNDS_DIR)
    
    Returns:
        Resolved absolute path if found, None otherwise
    """
    if not path or not path.strip():
        return None
    
    path = path.strip()
    
    # If absolute path exists, return it
    if os.path.isabs(path) and os.path.exists(path):
        return path
    
    # If relative path exists, return absolute version
    if os.path.exists(path):
        return os.path.abspath(path)
    
    # Search in default directories if not provided
    if search_dirs is None:
        search_dirs = [FX_DIR, SOUNDS_DIR]
    
    # Try each search directory
    for dir_path in search_dirs:
        if not os.path.exists(dir_path):
            continue
        
        # Try direct path
        full = os.path.join(dir_path, path)
        if os.path.exists(full):
            return os.path.abspath(full)
        
        # Try with assets/ prefix removed if path already has it
        if path.startswith("assets/"):
            rel_path = path[7:]  # Remove "assets/" prefix
            full = os.path.join(dir_path, rel_path)
            if os.path.exists(full):
                return os.path.abspath(full)
        
        # Try adding assets/ prefix
        full = os.path.join("assets", path)
        if os.path.exists(full):
            return os.path.abspath(full)
    
    return None


def resolve_script_path(script_name: str) -> Optional[str]:
    """
    Resolve a script file path.
    
    Args:
        script_name: Name of the script file
    
    Returns:
        Resolved absolute path if found, None otherwise
    """
    return resolve_path(script_name, [SCRIPT_DIR])


# ============================================
# RVC MODEL FILES
# ============================================

def resolve_model_path(model_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve paths for RVC model files.
    
    Args:
        model_name: Name of the RVC model (folder name)
    
    Returns:
        Tuple of (model_path, index_path) if both exist, (None, None) otherwise
    """
    if not model_name or model_name in ("(no models found)", "(no models folder)"):
        return None, None
    
    model_path = os.path.join(MODEL_DIR, model_name, "model.pth")
    index_path = os.path.join(MODEL_DIR, model_name, "model.index")
    
    if os.path.exists(model_path) and os.path.exists(index_path):
        return model_path, index_path
    
    return None, None


def validate_model_exists(model_name: str) -> bool:
    """Check if an RVC model exists."""
    model_path, index_path = resolve_model_path(model_name)
    return model_path is not None and index_path is not None


def list_rvc_models() -> List[str]:
    """
    List all available RVC models.
    
    Returns:
        List of model folder names
    """
    if not os.path.exists(MODEL_DIR):
        return []
    
    models = []
    for name in os.listdir(MODEL_DIR):
        model_dir = os.path.join(MODEL_DIR, name)
        if os.path.isdir(model_dir):
            model_path = os.path.join(model_dir, "model.pth")
            index_path = os.path.join(model_dir, "model.index")
            if os.path.exists(model_path) and os.path.exists(index_path):
                models.append(name)
    
    return sorted(models)


# ============================================
# AUDIO FILE DISCOVERY
# ============================================

def list_audio_files(directory: str, recursive: bool = False) -> List[str]:
    """
    List all audio files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
    
    Returns:
        List of absolute paths to audio files
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(AUDIO_EXTENSIONS):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(directory, filename))
    
    return sorted(files)


def list_background_audio() -> List[Dict[str, str]]:
    """
    List all available background audio files.
    
    Returns:
        List of dicts with 'name' and 'path' keys
    """
    files = []
    
    for dir_path in [FX_DIR, SOUNDS_DIR]:
        if not os.path.exists(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                files.append({
                    "name": os.path.splitext(filename)[0],
                    "path": os.path.join(dir_path, filename)
                })
    
    return sorted(files, key=lambda x: x["name"])


def list_fx_files() -> List[str]:
    """List all FX audio files."""
    return list_audio_files(FX_DIR)


def list_sound_files() -> List[str]:
    """List all sound audio files."""
    return list_audio_files(SOUNDS_DIR)


# ============================================
# OUTPUT FILES
# ============================================

def get_output_path(filename: str, subdir: Optional[str] = None) -> str:
    """
    Get path for an output file.
    
    Args:
        filename: Name of the output file
        subdir: Optional subdirectory within output
    
    Returns:
        Absolute path for the output file
    """
    if subdir:
        output_dir = os.path.join(OUTPUT_DIR, subdir)
    else:
        output_dir = OUTPUT_DIR
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, filename)


def list_output_files(subdir: Optional[str] = None, extension: Optional[str] = None) -> List[str]:
    """
    List files in output directory.
    
    Args:
        subdir: Optional subdirectory
        extension: Optional file extension filter (e.g., '.wav')
    
    Returns:
        List of absolute paths
    """
    if subdir:
        output_dir = os.path.join(OUTPUT_DIR, subdir)
    else:
        output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            if extension is None or filename.lower().endswith(extension):
                files.append(filepath)
    
    return sorted(files)


# ============================================
# CONFIGURATION FILES
# ============================================

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_json_file(path: str, data: Dict[str, Any], indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        path: Path to JSON file
        data: Data to save
        indent: JSON indentation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception:
        return False


def load_config() -> Optional[Dict[str, Any]]:
    """Load config - use config.get_config() instead."""
    from config import get_config
    return get_config()


def save_config_file(data: Dict[str, Any]) -> bool:
    """Save config - use config.save_config() instead."""
    from config import save_config
    return save_config(data)


# ============================================
# FILE VALIDATION
# ============================================

def validate_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return os.path.isfile(path)


def validate_dir_exists(path: str) -> bool:
    """Check if a directory exists."""
    return os.path.isdir(path)


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def get_file_extension(path: str) -> str:
    """Get file extension (lowercase)."""
    return os.path.splitext(path)[1].lower()


def get_filename(path: str, with_extension: bool = True) -> str:
    """Get filename from path."""
    basename = os.path.basename(path)
    if with_extension:
        return basename
    return os.path.splitext(basename)[0]


# ============================================
# DIRECTORY UTILITIES
# ============================================

def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Union[str, Path]) -> None:
    """Ensure the parent directory of a file path exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_directories(path: str) -> List[str]:
    """List subdirectories in a path."""
    if not os.path.exists(path):
        return []
    
    return sorted([
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ])


__all__ = [
    # Path resolution
    'resolve_path',
    'resolve_audio_path',
    'resolve_script_path',
    # RVC models
    'resolve_model_path',
    'validate_model_exists',
    'list_rvc_models',
    # Audio discovery
    'list_audio_files',
    'list_background_audio',
    'list_fx_files',
    'list_sound_files',
    # Output files
    'get_output_path',
    'list_output_files',
    # Config files
    'load_json_file',
    'save_json_file',
    'load_config',
    'save_config',
    'load_voices',
    'save_voices',
    # Validation
    'validate_file_exists',
    'validate_dir_exists',
    'get_file_size',
    'get_file_extension',
    'get_filename',
    # Directories
    'ensure_dir',
    'ensure_parent_dir',
    'list_directories',
    # Constants
    'AUDIO_EXTENSIONS',
    'WAV_EXTENSIONS',
    'LOSSLESS_EXTENSIONS',
]
