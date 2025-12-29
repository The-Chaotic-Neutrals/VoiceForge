"""
TTS Service - Text-to-Speech generation logic.

Handles Chatterbox TTS engine for voice cloning.
Supports two generation modes:
- chunked: Splits text into chunks, generates all, returns combined audio
- streaming: Splits text into chunks, streams each chunk as it's generated
"""

import os
import sys
import tempfile
import threading
from typing import Callable, Optional, Literal, Dict

from pydub import AudioSegment

# Add app directory to path
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from util.clients import get_chatterbox_client
from util.text_utils import split_text

# Cache for prepared prompt audio (path -> (prepared_path, mtime))
_prompt_cache: Dict[str, tuple] = {}
_prompt_cache_lock = threading.Lock()
MAX_PROMPT_CACHE_SIZE = 10


def _get_prepared_prompt(prompt_audio_path: str) -> str:
    """
    Get or create a prepared prompt audio file (24kHz mono WAV).
    Uses cache to avoid re-preparing the same prompt repeatedly.
    """
    try:
        mtime = os.path.getmtime(prompt_audio_path)
        cache_key = prompt_audio_path
        
        with _prompt_cache_lock:
            if cache_key in _prompt_cache:
                cached_path, cached_mtime = _prompt_cache[cache_key]
                if cached_mtime == mtime and os.path.exists(cached_path):
                    return cached_path
        
        # Prepare prompt
        fd, prepared_path = tempfile.mkstemp(suffix="_prompt_cached.wav")
        os.close(fd)
        
        seg = AudioSegment.from_file(prompt_audio_path)
        seg = seg.set_channels(1).set_frame_rate(24000)
        seg.export(prepared_path, format="wav")
        
        with _prompt_cache_lock:
            _prompt_cache[cache_key] = (prepared_path, mtime)
            
            # Evict old entries
            if len(_prompt_cache) > MAX_PROMPT_CACHE_SIZE:
                oldest_key = next(iter(_prompt_cache))
                old_path, _ = _prompt_cache.pop(oldest_key)
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except:
                    pass
        
        return prepared_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to prepare prompt audio: {e}")


def generate_tts_wav(
    text: str,
    prompt_audio_path: str,
    mode: Literal["chunked", "streaming"] = "chunked",
    max_tokens_per_batch: int = 100,
    token_method: str = "tiktoken",
    seed: int = 0,
    status_update: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    on_chunk: Optional[Callable[[bytes, int, int], None]] = None
) -> str:
    """
    Generate TTS and return path to WAV file using Chatterbox.
    
    Args:
        text: Text to synthesize
        prompt_audio_path: Reference audio for voice cloning (required, 5+ sec)
        mode: Generation mode - 'chunked' (default) or 'streaming'
        max_tokens_per_batch: Max tokens per batch for text chunking
        token_method: Token counting method ('tiktoken' or 'words')
        seed: Random seed (-1 = new random each time, 0 = no seeding, >0 = specific seed)
        status_update: Callback for status messages
        progress_callback: Callback for progress (0.0-1.0)
        on_chunk: Callback(audio_bytes, chunk_index, total_chunks) for streaming mode
        
    Returns:
        Path to generated WAV file
    """
    if status_update is None:
        status_update = lambda x: None
    
    # Use streaming mode if requested
    if mode == "streaming":
        return _generate_tts_streaming(
            text, status_update, progress_callback,
            prompt_audio_path, seed,
            max_tokens_per_batch, on_chunk
        )
    
    # Chunked mode (default)
    batches = split_text(text, max_tokens=max_tokens_per_batch, token_method=token_method)
    
    if len(batches) <= 1:
        return _generate_tts_single(
            text, status_update, progress_callback,
            prompt_audio_path, seed
        )
    
    # Multiple batches
    status_update(f"Processing {len(batches)} batches...")
    audio_segments = []
    temp_files = []
    
    try:
        for i, batch_text in enumerate(batches):
            status_update(f"Batch {i+1}/{len(batches)}...")
            
            def batch_progress(p):
                if progress_callback:
                    progress_callback(i / len(batches) + p / len(batches))
            
            batch_file = _generate_tts_single(
                batch_text, lambda x: None, batch_progress,
                prompt_audio_path, seed
            )
            temp_files.append(batch_file)
            audio_segments.append(AudioSegment.from_wav(batch_file))
        
        # Combine segments
        combined = audio_segments[0]
        for seg in audio_segments[1:]:
            combined += seg
        
        fd, output_path = tempfile.mkstemp(suffix="_tts.wav")
        os.close(fd)
        combined.export(output_path, format="wav")
        
        if progress_callback:
            progress_callback(1.0)
        
        return output_path
    finally:
        for tf in temp_files:
            try:
                os.remove(tf)
            except:
                pass


def _generate_tts_streaming(
    text: str,
    status_update: Callable[[str], None],
    progress_callback: Optional[Callable[[float], None]],
    prompt_audio_path: str,
    seed: int,
    max_tokens: int,
    on_chunk: Optional[Callable[[bytes, int, int], None]]
) -> str:
    """Generate TTS using streaming mode - chunks are streamed as generated."""
    if not prompt_audio_path or not os.path.exists(prompt_audio_path):
        raise ValueError("Chatterbox requires a prompt audio file for voice cloning.")
    
    # Get prepared prompt from cache (or prepare if not cached)
    prepared_prompt = _get_prepared_prompt(prompt_audio_path)
    
    status_update("Streaming TTS generation...")
    
    client = get_chatterbox_client()
    
    return client.generate_stream(
        text=text,
        prompt_audio_path=prepared_prompt,
        seed=seed,
        max_tokens=max_tokens,
        on_chunk=on_chunk,
        on_progress=progress_callback
    )
    # Note: Don't delete prepared_prompt - it's cached for reuse


def _generate_tts_single(
    text: str,
    status_update: Callable[[str], None],
    progress_callback: Optional[Callable[[float], None]],
    prompt_audio_path: str,
    seed: int
) -> str:
    """Generate single TTS segment using Chatterbox."""
    if not prompt_audio_path or not os.path.exists(prompt_audio_path):
        raise ValueError("Chatterbox requires a prompt audio file for voice cloning.")
    
    if progress_callback:
        progress_callback(0.0)
    
    # Get prepared prompt from cache (or prepare if not cached)
    prepared_prompt = _get_prepared_prompt(prompt_audio_path)
    
    status_update("Generating with Chatterbox-Turbo...")
    
    client = get_chatterbox_client()
    result = client.generate(
        text=text,
        prompt_audio_path=prepared_prompt,
        seed=seed
    )
    
    if progress_callback:
        progress_callback(1.0)
    
    return result
    # Note: Don't delete prepared_prompt - it's cached for reuse
