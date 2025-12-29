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
from typing import Callable, Optional, Literal

from pydub import AudioSegment

# Add app directory to path
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from util.clients import get_chatterbox_client
from util.text_utils import split_text


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
    
    # Prepare prompt audio (mono WAV @ 24kHz for best results)
    fd, prepared_prompt = tempfile.mkstemp(suffix="_prompt.wav")
    os.close(fd)
    try:
        seg = AudioSegment.from_file(prompt_audio_path)
        seg = seg.set_channels(1).set_frame_rate(24000)
        seg.export(prepared_prompt, format="wav")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare prompt audio: {e}")
    
    status_update("Streaming TTS generation...")
    
    try:
        client = get_chatterbox_client()
        
        return client.generate_stream(
            text=text,
            prompt_audio_path=prepared_prompt,
            seed=seed,
            max_tokens=max_tokens,
            on_chunk=on_chunk,
            on_progress=progress_callback
        )
    finally:
        try:
            os.remove(prepared_prompt)
        except:
            pass


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
    
    # Prepare prompt audio (mono WAV @ 24kHz for best results)
    fd, prepared_prompt = tempfile.mkstemp(suffix="_prompt.wav")
    os.close(fd)
    try:
        seg = AudioSegment.from_file(prompt_audio_path)
        seg = seg.set_channels(1).set_frame_rate(24000)
        seg.export(prepared_prompt, format="wav")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare prompt audio: {e}")
    
    status_update("Generating with Chatterbox-Turbo...")
    
    try:
        client = get_chatterbox_client()
        result = client.generate(
            text=text,
            prompt_audio_path=prepared_prompt,
            seed=seed
        )
        
        if progress_callback:
            progress_callback(1.0)
        
        return result
    finally:
        try:
            os.remove(prepared_prompt)
        except:
            pass
