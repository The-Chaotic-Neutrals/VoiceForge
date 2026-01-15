# audio_utils.py
"""
Centralized audio handling for VoiceForge.

ALL audio processing logic for the entire program lives here:
- Reading/writing audio files (wav, mp3, etc.)
- Format conversion (wav to mp3, opus, aac, flac, pcm)
- FFmpeg operations
- Audio segment manipulation
- Sample rate conversion
- Normalization
"""

import os
import sys
import io
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional, Literal, Union
from io import BytesIO

import numpy as np
import soundfile as sf


# ============================================
# ATOMIC FILE OPERATIONS
# ============================================

def _mkstemp_near(dst_dir: Path, suffix: str) -> str:
    """Create a temporary file in the same directory for safe atomic replace on Windows."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst_dir), suffix=suffix)
    os.close(fd)
    return tmp


def _atomic_write(src_tmp_path: str, dst_path: Path):
    """Move a temp file into place atomically. Falls back safely across drives on Windows."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src_tmp_path, str(dst_path))
        return
    except OSError:
        try:
            near_tmp = _mkstemp_near(dst_path.parent, suffix=dst_path.suffix or "")
            shutil.copyfile(src_tmp_path, near_tmp)
            os.replace(near_tmp, str(dst_path))
        finally:
            try:
                os.remove(src_tmp_path)
            except Exception:
                pass


# ============================================
# BASIC WAV READ/WRITE
# ============================================

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file and return audio data and sample rate.
    
    Args:
        path: Path to WAV file
    
    Returns:
        Tuple of (audio_data as float32 numpy array, sample_rate)
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def save_wav(path: str, data: np.ndarray, sr: int):
    """
    Save audio data to WAV file (mono PCM16) with atomic write for safety.
    
    Args:
        path: Output file path
        data: Audio data as numpy array
        sr: Sample rate
    """
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    dst_path = Path(path)
    suffix = dst_path.suffix or ".wav"
    tmp_path = _mkstemp_near(dst_path.parent, suffix=suffix)

    try:
        sf.write(tmp_path, data, sr, subtype="PCM_16")
        _atomic_write(tmp_path, dst_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def read_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Read any audio file format and return audio data and sample rate.
    Uses soundfile for common formats, falls back to ffmpeg for others.
    
    Args:
        path: Path to audio file
    
    Returns:
        Tuple of (audio_data as float32 numpy array, sample_rate)
    """
    try:
        return read_wav(path)
    except Exception:
        # Fall back to ffmpeg conversion
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            convert_to_wav(path, tmp_wav)
            return read_wav(tmp_wav)
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


def get_audio_info(path: str) -> dict:
    """
    Get audio file information.
    
    Args:
        path: Path to audio file
    
    Returns:
        Dict with samplerate, channels, frames, duration, format info
    """
    try:
        info = sf.info(path)
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        return {}


# ============================================
# FFMPEG OPERATIONS
# ============================================

def convert_to_wav(
    input_path: str, 
    output_path: str, 
    sample_rate: int = 44100, 
    channels: int = 1
) -> None:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to input audio file (any format ffmpeg supports)
        output_path: Path to output WAV file
        sample_rate: Output sample rate in Hz (default: 44100)
        channels: Number of output channels (default: 1 for mono)
    
    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails
    """
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        output_path
    ]
    subprocess.run(cmd, check=True)


def build_ffmpeg_base_cmd(
    input_path: str,
    output_path: str,
    filters: Optional[str] = None,
    filter_complex: Optional[str] = None,
    output_channels: Optional[int] = None,
    map_output: Optional[str] = None,
) -> List[str]:
    """
    Build base FFmpeg command with common options.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        filters: Audio filter string (for -af)
        filter_complex: Complex filter string (for -filter_complex)
        output_channels: Number of output channels (for -ac)
        map_output: Output stream to map (for -map)
    
    Returns:
        List of command arguments
    """
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
    ]
    
    if filter_complex:
        cmd.extend(["-filter_complex", filter_complex])
        if map_output:
            cmd.extend(["-map", map_output])
    elif filters:
        cmd.extend(["-af", filters])
    
    cmd.extend(["-c:a", "pcm_s16le"])
    
    if output_channels:
        cmd.extend(["-ac", str(output_channels)])
    
    cmd.append(output_path)
    
    return cmd


def run_ffmpeg(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run an FFmpeg command.
    
    Args:
        cmd: Command as list of arguments
        check: Whether to raise on error
    
    Returns:
        CompletedProcess result
    """
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


# ============================================
# FORMAT CONVERSION
# ============================================

def convert_to_format(
    wav_path: str,
    output_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"],
    speed: float = 1.0
) -> bytes:
    """
    Convert WAV file to requested format with high quality settings.
    
    Args:
        wav_path: Path to input WAV file
        output_format: Desired output format
        speed: Speed multiplier (0.25-4.0)
    
    Returns:
        Audio data as bytes
    """
    if output_format == "wav" and speed == 1.0:
        with open(wav_path, "rb") as f:
            return f.read()
    
    fd, tmp = tempfile.mkstemp(suffix=f".{output_format}")
    os.close(fd)
    
    try:
        # Read original sample rate to preserve it
        try:
            info = sf.info(wav_path)
            original_sr = info.samplerate
        except Exception:
            original_sr = None
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-hide_banner", "-loglevel", "error",
            "-i", wav_path,
        ]
        
        # Build filter chain (preserve sample rate, apply speed if needed)
        filters = []
        if original_sr:
            filters.append(f"aresample={original_sr}")
        
        if speed != 1.0:
            filters.append(f"atempo={speed}")
        
        if filters:
            cmd.extend(["-af", ",".join(filters)])
        
        # Set output format with high quality settings
        if output_format == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-q:a", "0", "-b:a", "320k"])
        elif output_format == "opus":
            cmd.extend(["-codec:a", "libopus", "-b:a", "256k", "-vbr", "on", "-compression_level", "10"])
        elif output_format == "aac":
            cmd.extend(["-codec:a", "aac", "-b:a", "320k", "-profile:a", "aac_low"])
        elif output_format == "flac":
            cmd.extend(["-codec:a", "flac", "-compression_level", "12"])
        elif output_format == "wav":
            cmd.extend(["-codec:a", "pcm_s16le"])
        elif output_format == "pcm":
            cmd.extend(["-codec:a", "pcm_s16le", "-f", "s16le"])
        else:
            # Default to high quality MP3
            cmd.extend(["-codec:a", "libmp3lame", "-q:a", "0", "-b:a", "320k"])
        
        cmd.append(tmp)
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if not os.path.exists(tmp):
            raise RuntimeError(f"FFmpeg conversion failed: output file not created")
        
        # Read the converted file
        with open(tmp, "rb") as f:
            data = f.read()
        
        if len(data) == 0:
            raise RuntimeError("FFmpeg conversion produced empty file")
        
        return data
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def get_mime_type(format: str) -> str:
    """
    Get MIME type for audio format.
    
    Args:
        format: Audio format string
    
    Returns:
        MIME type string
    """
    mime_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm"
    }
    return mime_types.get(format, "audio/mpeg")


# ============================================
# PYDUB/AUDIOSEGMENT OPERATIONS
# ============================================

def mp3_bytes_to_wav(audio_bytes: bytes, output_path: Optional[str] = None) -> Union[str, Tuple[np.ndarray, int]]:
    """
    Convert MP3 bytes to WAV.
    
    Args:
        audio_bytes: MP3 audio as bytes
        output_path: Optional path to save WAV file. If None, returns numpy array.
    
    Returns:
        If output_path provided: path to WAV file
        If output_path is None: tuple of (audio_data, sample_rate)
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required for MP3 conversion. Install with: pip install pydub")
    
    buf = BytesIO(audio_bytes)
    seg = AudioSegment.from_file(buf, format="mp3")
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    
    if seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = samples / 32768.0  # Normalize int16 to float32
    sr = seg.frame_rate
    
    if output_path:
        save_wav(output_path, samples, sr)
        return output_path
    
    return samples, sr


def load_audio_segment(path: str) -> "AudioSegment":
    """
    Load an audio file as a pydub AudioSegment.
    
    Args:
        path: Path to audio file
    
    Returns:
        AudioSegment object
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Install with: pip install pydub")
    
    return AudioSegment.from_file(path)


def combine_audio_segments(segments: List["AudioSegment"]) -> "AudioSegment":
    """
    Combine multiple AudioSegments into one.
    
    Args:
        segments: List of AudioSegment objects
    
    Returns:
        Combined AudioSegment
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Install with: pip install pydub")
    
    if not segments:
        return AudioSegment.empty()
    
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    
    return combined


def segment_from_wav(path: str) -> "AudioSegment":
    """
    Load a WAV file as AudioSegment.
    
    Args:
        path: Path to WAV file
    
    Returns:
        AudioSegment object
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Install with: pip install pydub")
    
    return AudioSegment.from_wav(path)


def segment_to_wav(segment: "AudioSegment", path: str):
    """
    Export AudioSegment to WAV file.
    
    Args:
        segment: AudioSegment object
        path: Output path
    """
    segment.export(path, format="wav")


def prepare_audio_for_processing(
    input_path: str,
    target_sr: int = 24000,
    channels: int = 1
) -> str:
    """
    Prepare an audio file for processing (resample, convert channels).
    
    Args:
        input_path: Path to input audio
        target_sr: Target sample rate
        channels: Target number of channels
    
    Returns:
        Path to prepared audio file (temp file)
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Install with: pip install pydub")
    
    fd, prepared_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    seg = AudioSegment.from_file(input_path)
    seg = seg.set_channels(channels).set_frame_rate(target_sr)
    seg.export(prepared_path, format="wav")
    
    return prepared_path


# ============================================
# AUDIO NORMALIZATION
# ============================================

def normalize_audio(data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        data: Audio data as numpy array
        target_peak: Target peak level (0.0 to 1.0)
    
    Returns:
        Normalized audio data
    """
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data * (target_peak / max_val)
    return data


def ensure_mono(data: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (single channel).
    
    Args:
        data: Audio data as numpy array
    
    Returns:
        Mono audio data
    """
    if data.ndim > 1 and data.shape[1] > 1:
        return data.mean(axis=1)
    elif data.ndim > 1:
        return data.flatten()
    return data


def ensure_float32(data: np.ndarray) -> np.ndarray:
    """
    Ensure audio data is float32.
    
    Args:
        data: Audio data
    
    Returns:
        Float32 audio data normalized to [-1, 1]
    """
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        return data.astype(np.float32)
    return data


# ============================================
# AUDIO FILE EXTENSIONS & VALIDATION
# ============================================

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus')
WAV_EXTENSIONS = ('.wav',)
LOSSLESS_EXTENSIONS = ('.wav', '.flac')


def is_audio_file(path: str) -> bool:
    """Check if a path is an audio file based on extension."""
    return path.lower().endswith(AUDIO_EXTENSIONS)


def is_wav_file(path: str) -> bool:
    """Check if a path is a WAV file."""
    return path.lower().endswith(WAV_EXTENSIONS)


def get_audio_extension(path: str) -> str:
    """Get the audio file extension (lowercase)."""
    return os.path.splitext(path)[1].lower()


# ============================================
# TEMP FILE HELPERS
# ============================================

def create_temp_wav() -> str:
    """Create a temporary WAV file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path


def create_temp_audio(suffix: str = ".wav") -> str:
    """Create a temporary audio file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def cleanup_temp_file(path: str):
    """Safely remove a temporary file."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def cleanup_temp_files(paths: List[str]):
    """Safely remove multiple temporary files."""
    for path in paths:
        cleanup_temp_file(path)


__all__ = [
    # Basic read/write
    'read_wav',
    'save_wav',
    'read_audio',
    'get_audio_info',
    # FFmpeg operations
    'convert_to_wav',
    'build_ffmpeg_base_cmd',
    'run_ffmpeg',
    # Format conversion
    'convert_to_format',
    'get_mime_type',
    # AudioSegment operations
    'mp3_bytes_to_wav',
    'load_audio_segment',
    'combine_audio_segments',
    'segment_from_wav',
    'segment_to_wav',
    'prepare_audio_for_processing',
    # Normalization
    'normalize_audio',
    'ensure_mono',
    'ensure_float32',
    # Validation
    'is_audio_file',
    'is_wav_file',
    'get_audio_extension',
    'AUDIO_EXTENSIONS',
    'WAV_EXTENSIONS',
    'LOSSLESS_EXTENSIONS',
    # Temp files
    'create_temp_wav',
    'create_temp_audio',
    'cleanup_temp_file',
    'cleanup_temp_files',
]
