"""
TTS Service - Thin wrapper for Chatterbox TTS generation.

Just calls the Chatterbox client - all TTS logic is in chatterbox_server.py.
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
    
    client = get_chatterbox_client()
    
    # Use streaming mode if requested
    if mode == "streaming":
        status_update("Streaming TTS generation...")
        return client.generate_stream(
            text=text,
            prompt_audio_path=prompt_audio_path,
            seed=seed,
            max_tokens=max_tokens_per_batch,
            on_chunk=on_chunk,
            on_progress=progress_callback
        )
    
    # Chunked mode - split text and generate each batch
    batches = split_text(text, max_tokens=max_tokens_per_batch, token_method=token_method)
    
    # Single batch - direct generation
    if len(batches) <= 1:
        status_update("Generating with Chatterbox...")
        if progress_callback:
            progress_callback(0.0)
        result = client.generate(
            text=text,
            prompt_audio_path=prompt_audio_path,
            seed=seed,
            max_tokens=max_tokens_per_batch
        )
        if progress_callback:
            progress_callback(1.0)
        return result
    
    # Multiple batches - generate and combine
    status_update(f"Processing {len(batches)} batches...")
    audio_segments = []
    temp_files = []
    
    try:
        for i, batch_text in enumerate(batches):
            status_update(f"Batch {i+1}/{len(batches)}...")
            
            if progress_callback:
                progress_callback(i / len(batches))
            
            batch_file = client.generate(
                text=batch_text,
                prompt_audio_path=prompt_audio_path,
                seed=seed if seed <= 0 else seed + i,  # Increment seed for variety
                max_tokens=max_tokens_per_batch
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
