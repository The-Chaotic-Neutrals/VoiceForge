"""
Shared utilities for temporary file handling.

Eliminates duplicate temp file creation, cleanup, and audio conversion patterns
across routers and services.
"""

import os
import tempfile
import shutil
from typing import Optional, List, Callable, Any
from contextlib import contextmanager
from pathlib import Path

from util.audio_utils import convert_to_wav


class TempFileManager:
    """Context manager for managing temporary files with automatic cleanup."""
    
    def __init__(self):
        self.files: List[str] = []
    
    def create_temp_file(self, suffix: str = "", prefix: str = "voiceforge_") -> str:
        """Create a temporary file and track it for cleanup."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        self.files.append(path)
        return path
    
    def cleanup(self):
        """Remove all tracked temporary files."""
        for path in self.files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass  # Ignore cleanup errors
        self.files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# Note: temp_audio_file removed - use process_audio_upload or save_upload_to_temp instead


async def save_upload_to_temp(upload_file, suffix: str = "", prefix: str = "voiceforge_") -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        upload_file: FastAPI UploadFile object
        suffix: File extension suffix
        prefix: Prefix for temp file name
    
    Returns:
        Path to temporary file
    """
    file_ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=file_ext, prefix=prefix)
    os.close(fd)
    
    content = await upload_file.read()
    with open(path, "wb") as f:
        f.write(content)
    
    return path


def ensure_wav_format(input_path: str, sample_rate: int = 44100, channels: int = 2) -> str:
    """
    Ensure an audio file is in WAV format, converting if necessary.
    
    Args:
        input_path: Path to input audio file
        sample_rate: Target sample rate
        channels: Target channel count
    
    Returns:
        Path to WAV file (may be same as input if already WAV)
    """
    if input_path.lower().endswith('.wav'):
        return input_path
    
    # Create temp WAV file
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="voiceforge_")
    os.close(fd)
    
    convert_to_wav(input_path, wav_path, sample_rate=sample_rate, channels=channels)
    return wav_path

