"""
Audio Pipeline Service - Unified TTS generation with RVC, post-processing, and blending.

This is the single source of truth for the audio generation pipeline.
Both /v1/audio/speech and /api/generate use this.
"""

import asyncio
import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

# Add app directory to path
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from util.clients import (
    run_rvc,
    run_rvc_stream,
    run_postprocess,
    run_blend,
    run_save,
    is_rvc_server_available,
    is_postprocess_server_available,
    get_rvc_client,
)
from util.audio_utils import convert_to_format, get_mime_type
from util.file_utils import resolve_audio_path
from config import (
    get_config,
    is_rvc_enabled,
    is_post_enabled,
    is_background_enabled,
    get_bg_tracks,
)

# Import from services (relative)
from .tts_service import generate_tts_wav

# Import models
from servers.models.params import (
    RVCParams,
    PostProcessParams,
    BackgroundParams,
    get_default_rvc_params,
    get_default_post_params,
)


from servers.models.requests import TTSRequest
import subprocess

# Standard sample rate for the pipeline (44.1kHz is CD quality standard)
PIPELINE_SAMPLE_RATE = 44100


def resample_to_pipeline_rate(audio_path: str, request_id: str = None) -> str:
    """
    Resample audio to the pipeline's standard sample rate (44.1kHz).
    Uses SoXr for high-quality resampling (same quality as ffmpeg's soxr).
    
    Args:
        audio_path: Path to input audio file
        request_id: Optional request ID for logging
    
    Returns:
        Path to resampled audio file (or original if already at target rate)
    """
    import soundfile as sf
    import soxr
    import numpy as np
    
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        # Read audio and check sample rate in one operation (no ffprobe needed)
        data, current_sr = sf.read(audio_path, dtype='float32')
        
        # If already at target rate, return original
        if current_sr == PIPELINE_SAMPLE_RATE:
            return audio_path
        
        print(f"{req_tag}Resampling {current_sr}Hz -> {PIPELINE_SAMPLE_RATE}Hz (SoXr VHQ)")
        
        # Use SoXr with Very High Quality setting (same as ffmpeg precision=28)
        resampled = soxr.resample(data, current_sr, PIPELINE_SAMPLE_RATE, quality='VHQ')
        
        # Create output file
        fd, output_path = tempfile.mkstemp(suffix="_44k.wav")
        os.close(fd)
        
        # Write resampled audio
        sf.write(output_path, resampled.astype(np.float32), PIPELINE_SAMPLE_RATE)
        return output_path
        
    except Exception as e:
        print(f"{req_tag}Resample failed: {e}, returning original")
        return audio_path  # Fall back to original


def apply_output_volume(audio_path: str, volume: float, request_id: str = None) -> str:
    """
    Apply output volume to audio file using numpy (fast in-memory).
    
    Args:
        audio_path: Path to input audio file
        volume: Volume multiplier (1.0 = 100%, 0.5 = 50%, 2.0 = 200%)
        request_id: Optional request ID for logging
    
    Returns:
        Path to volume-adjusted audio file
    """
    import soundfile as sf
    import numpy as np
    
    # If volume is 1.0, just return the original
    if abs(volume - 1.0) < 0.01:
        return audio_path
    
    req_tag = f"[{request_id}] " if request_id else ""
    print(f"{req_tag}Applying output volume: {volume:.2f}x ({int(volume * 100)}%) (in-memory)")
    
    try:
        # Read audio
        data, sr = sf.read(audio_path, dtype='float32')
        
        # Apply volume (simple linear scaling)
        data = data * volume
        
        # Clip to prevent clipping distortion
        data = np.clip(data, -1.0, 1.0)
        
        # Create output file
        fd, output_path = tempfile.mkstemp(suffix="_vol.wav")
        os.close(fd)
        
        # Write adjusted audio
        sf.write(output_path, data, sr)
        return output_path
        
    except Exception as e:
        print(f"{req_tag}Volume adjustment failed: {e}")
        # Return original on failure
        return audio_path


def resample_and_adjust_volume(
    audio_path: str, 
    target_sr: int = PIPELINE_SAMPLE_RATE,
    volume: float = 1.0,
    request_id: str = None
) -> str:
    """
    Combined resample + volume adjustment in single read/write cycle.
    Uses SoXr for high-quality resampling.
    
    Args:
        audio_path: Path to input audio file
        target_sr: Target sample rate (default: 44100)
        volume: Volume multiplier (1.0 = no change)
        request_id: Optional request ID for logging
    
    Returns:
        Path to processed audio file
    """
    import soundfile as sf
    import soxr
    import numpy as np
    
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        # Read audio
        data, current_sr = sf.read(audio_path, dtype='float32')
        
        needs_resample = current_sr != target_sr
        needs_volume = abs(volume - 1.0) >= 0.01
        
        # If nothing to do, return original
        if not needs_resample and not needs_volume:
            return audio_path
        
        ops = []
        if needs_resample:
            ops.append(f"resample {current_sr}→{target_sr}Hz (SoXr)")
        if needs_volume:
            ops.append(f"volume {int(volume*100)}%")
        print(f"{req_tag}Processing: {', '.join(ops)}")
        
        # Resample if needed using SoXr VHQ
        if needs_resample:
            data = soxr.resample(data, current_sr, target_sr, quality='VHQ')
            current_sr = target_sr
        
        # Apply volume if needed
        if needs_volume:
            data = data * volume
            data = np.clip(data, -1.0, 1.0)
        
        # Create output file
        fd, output_path = tempfile.mkstemp(suffix="_proc.wav")
        os.close(fd)
        
        # Write processed audio
        sf.write(output_path, data.astype(np.float32), current_sr)
        return output_path
        
    except Exception as e:
        print(f"{req_tag}Processing failed: {e}, returning original")
        return audio_path


# Shared thread pool for blocking operations (TTS, RVC, Post)
_executor: Optional[ThreadPoolExecutor] = None

# Cache for prepared prompt audio (path -> (prepared_path, mtime))
# Avoids re-preparing the same prompt audio repeatedly
_prompt_cache: Dict[str, tuple] = {}
_prompt_cache_lock = threading.Lock()
MAX_PROMPT_CACHE_SIZE = 10  # Keep last N prompts cached


def get_prepared_prompt(prompt_audio_path: str, request_id: str = None) -> str:
    """
    Get or create a prepared prompt audio file (24kHz mono WAV).
    Uses LRU cache to avoid re-preparing the same prompt repeatedly.
    
    Args:
        prompt_audio_path: Path to original prompt audio
        request_id: Optional request ID for logging
    
    Returns:
        Path to prepared prompt audio (cached or newly created)
    """
    from pydub import AudioSegment
    
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        # Get file modification time for cache invalidation
        mtime = os.path.getmtime(prompt_audio_path)
        cache_key = prompt_audio_path
        
        with _prompt_cache_lock:
            # Check cache
            if cache_key in _prompt_cache:
                cached_path, cached_mtime = _prompt_cache[cache_key]
                if cached_mtime == mtime and os.path.exists(cached_path):
                    print(f"{req_tag}Using cached prompt audio: {os.path.basename(prompt_audio_path)}")
                    return cached_path
        
        # Prepare prompt (24kHz mono for Chatterbox)
        print(f"{req_tag}Preparing prompt audio: {os.path.basename(prompt_audio_path)}")
        fd, prepared_path = tempfile.mkstemp(suffix="_prompt_cached.wav")
        os.close(fd)
        
        seg = AudioSegment.from_file(prompt_audio_path)
        seg = seg.set_channels(1).set_frame_rate(24000)
        seg.export(prepared_path, format="wav")
        
        with _prompt_cache_lock:
            # Add to cache
            _prompt_cache[cache_key] = (prepared_path, mtime)
            
            # Evict old entries if cache is too large
            if len(_prompt_cache) > MAX_PROMPT_CACHE_SIZE:
                # Remove oldest entry (simple FIFO, not true LRU)
                oldest_key = next(iter(_prompt_cache))
                old_path, _ = _prompt_cache.pop(oldest_key)
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except:
                    pass
        
        return prepared_path
        
    except Exception as e:
        print(f"{req_tag}Prompt preparation failed: {e}")
        raise


def get_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor for compute tasks."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", os.cpu_count() or 4)))
    return _executor


# =============================================================================
# Cancellation System - Global flags that persist until explicitly cleared
# =============================================================================
_cancelled_requests: Set[str] = set()  # Persists until workers see it
_active_requests: Dict[str, bool] = {}  # request_id -> is_active
_cancel_lock = threading.Lock()
_cancel_timestamps: Dict[str, float] = {}  # When cancellation was requested


def cancel_generation(request_id: str) -> bool:
    """Cancel a generation request by ID. Flag persists until workers acknowledge."""
    import time
    with _cancel_lock:
        # Allow cancellation even if request was just unregistered (race condition)
        _cancelled_requests.add(request_id)
        _cancel_timestamps[request_id] = time.time()
        print(f"[Cancel] ⚠️ Request {request_id} CANCELLED - flag will persist for workers")
        return True


def cancel_all_generations() -> int:
    """Cancel all active generation requests."""
    import time
    with _cancel_lock:
        count = 0
        now = time.time()
        for req_id in list(_active_requests.keys()):
            _cancelled_requests.add(req_id)
            _cancel_timestamps[req_id] = now
            count += 1
        if count > 0:
            print(f"[Cancel] ⚠️ Cancelled {count} active request(s)")
        return count


def is_cancelled(request_id: str) -> bool:
    """Check if a request has been cancelled. Workers should call this in loops."""
    with _cancel_lock:
        cancelled = request_id in _cancelled_requests
        if cancelled:
            print(f"[Cancel] ✓ Request {request_id} IS CANCELLED - worker stopping!")
        return cancelled


def _register_request(request_id: str):
    """Register a new active request."""
    with _cancel_lock:
        _active_requests[request_id] = True
        # Clear any stale cancellation from previous request with same ID
        _cancelled_requests.discard(request_id)
        _cancel_timestamps.pop(request_id, None)
        print(f"[Cancel] Registered request {request_id}")


def _unregister_request(request_id: str):
    """Unregister request from active tracking. Does NOT clear cancellation flag."""
    with _cancel_lock:
        _active_requests.pop(request_id, None)
        # DON'T clear _cancelled_requests here - workers may still need to see it!
        print(f"[Cancel] Unregistered request {request_id} (cancel flag preserved)")


def _cleanup_old_cancellations(max_age_seconds: float = 300):
    """Clean up old cancellation flags (call periodically)."""
    import time
    with _cancel_lock:
        now = time.time()
        to_remove = [
            req_id for req_id, ts in _cancel_timestamps.items()
            if now - ts > max_age_seconds
        ]
        for req_id in to_remove:
            _cancelled_requests.discard(req_id)
            _cancel_timestamps.pop(req_id, None)
        if to_remove:
            print(f"[Cancel] Cleaned up {len(to_remove)} old cancellation flags")


def get_active_requests() -> List[str]:
    """Get list of active request IDs."""
    with _cancel_lock:
        return list(_active_requests.keys())


class CancelledException(Exception):
    """Raised when a generation is cancelled."""
    pass


@dataclass
class AudioPipelineResult:
    """Result of audio pipeline execution."""
    success: bool
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    mime_type: str = "audio/mpeg"
    duration: Optional[float] = None
    error: Optional[str] = None
    temp_files: List[str] = field(default_factory=list)
    
    def cleanup(self):
        """Clean up temporary files."""
        for path in self.temp_files:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass


async def generate_audio(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    request_id: Optional[str] = None,
) -> AudioPipelineResult:
    """
    CHUNKED PIPELINE: Generate audio through the full pipeline.
    
    TTS -> RVC -> PostProcess -> Blend -> Master
    
    Uses CHUNKED endpoints (/v1/tts/chunked, /v1/rvc/chunked) that wait for
    complete results before returning. Good for batch processing.
    
    For real-time streaming, use generate_audio_streaming() instead.
    
    Args:
        request: Unified TTS request
        status_callback: Optional callback for status updates
        progress_callback: Optional callback for progress (0.0-1.0)
        request_id: Optional ID for cancellation tracking
        
    Returns:
        AudioPipelineResult with audio data or error
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    
    # Register this request for cancellation tracking
    _register_request(request_id)
    
    def check_cancelled():
        if is_cancelled(request_id):
            raise CancelledException(f"Generation {request_id} was cancelled")
    
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"[{request_id}] {msg}")
    
    def progress(p: float):
        if progress_callback:
            progress_callback(p)
    
    try:
        # Load config for defaults
        config = get_config()
        
        # Determine what's enabled
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_background = request.enable_background if request.enable_background is not None else is_background_enabled()
        
        # Resolve prompt audio for voice cloning
        prompt_audio = request.chatterbox_prompt_audio
        if prompt_audio:
            prompt_audio = resolve_audio_path(prompt_audio)
        if not prompt_audio or not os.path.exists(prompt_audio or ""):
            return AudioPipelineResult(
                success=False,
                error="Chatterbox requires a prompt audio file for voice cloning",
                temp_files=temp_files,
            )
        
        # Step 1: TTS Generation
        check_cancelled()
        status("Generating TTS...")
        progress(0.1)
        
        executor = get_executor()
        
        def do_tts():
            return generate_tts_wav(
                text=request.input,
                prompt_audio_path=prompt_audio,
                mode=request.tts_mode,
                max_tokens_per_batch=request.tts_batch_tokens,
                token_method=request.tts_token_method,
                seed=request.chatterbox_seed,
            )
        
        tts_path = await asyncio.get_event_loop().run_in_executor(executor, do_tts)
        temp_files.append(tts_path)
        current_path = tts_path
        status("TTS completed")
        progress(0.3)
        
        # Step 2: RVC Voice Conversion
        check_cancelled()
        if do_rvc:
            status("Running RVC...")
            rvc_model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
            # Don't send RVC params - let the server use config.json directly
            # This ensures preload and conversion always use the same params
            
            def do_rvc_convert():
                return run_rvc(current_path, rvc_model, {}, lambda s: None, None, request_id=request_id)
            
            rvc_path = await asyncio.get_event_loop().run_in_executor(executor, do_rvc_convert)
            temp_files.append(rvc_path)
            current_path = rvc_path
            status("RVC completed")
            
            # Resample to pipeline standard rate (44.1kHz) after RVC
            def do_resample(path=current_path):
                return resample_to_pipeline_rate(path, request_id=request_id)
            
            resampled_path = await asyncio.get_event_loop().run_in_executor(executor, do_resample)
            if resampled_path != current_path:
                temp_files.append(resampled_path)
                current_path = resampled_path
        progress(0.5)
        
        # Step 3: Post-Processing
        check_cancelled()
        if do_post:
            post_params = request.get_post_params()
            
            # Apply config overrides for any params not explicitly set
            defaults = get_default_post_params()
            for key in post_params.to_dict():
                config_val = config.get(key)
                if config_val is not None:
                    # Only use config if request didn't explicitly set it
                    req_val = getattr(request, key, None)
                    default_val = getattr(defaults, key, None)
                    if req_val == default_val:
                        setattr(post_params, key, config_val)
            
            # Skip HTTP call if no effects are actually enabled
            if post_params.needs_processing():
                status("Post-processing...")
                def do_post_process():
                    return run_postprocess(current_path, post_params.to_dict(), lambda s: None, request_id=request_id)
                
                post_path = await asyncio.get_event_loop().run_in_executor(executor, do_post_process)
                temp_files.append(post_path)
                current_path = post_path
                status("Post-processing completed")
            else:
                print(f"[{request_id}] Skipping post-processing (no effects enabled)")
        progress(0.7)
        
        # Step 4: Background Blending
        check_cancelled()
        if do_background:
            bg_params = request.get_background_params()
            
            # Get background tracks
            bg_files = []
            bg_vols = []
            bg_delays = []
            bg_fade_ins = []
            bg_fade_outs = []
            
            if bg_params.use_config_tracks:
                # Use tracks from config
                for track in get_bg_tracks():
                    if track and track.get("file"):
                        resolved = resolve_audio_path(str(track["file"]))
                        if resolved and os.path.exists(resolved):
                            vol = float(track.get("volume", 0.3))
                            if vol > 0:
                                bg_files.append(resolved)
                                bg_vols.append(vol)
                                bg_delays.append(float(track.get("delay", 0)))
                                bg_fade_ins.append(float(track.get("fade_in", 0)))
                                bg_fade_outs.append(float(track.get("fade_out", 0)))
            else:
                # Use tracks from request
                for i, f in enumerate(bg_params.files):
                    resolved = resolve_audio_path(f)
                    if resolved and os.path.exists(resolved):
                        vol = bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3
                        delay = bg_params.delays[i] if i < len(bg_params.delays) else 0.0
                        fade_in = bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0.0
                        fade_out = bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0.0
                        if vol > 0:
                            bg_files.append(resolved)
                            bg_vols.append(vol)
                            bg_delays.append(delay)
                            bg_fade_ins.append(fade_in)
                            bg_fade_outs.append(fade_out)
            
            if bg_files:
                status("Blending background...")
                # Use 1.0 for blending - output_volume controls final loudness
                main_volume = 1.0
                
                def do_blend():
                    return run_blend(current_path, bg_files, bg_vols, main_volume, lambda s: None, bg_delays, bg_fade_ins, bg_fade_outs, request_id=request_id)
                
                blend_path = await asyncio.get_event_loop().run_in_executor(executor, do_blend)
                temp_files.append(blend_path)
                current_path = blend_path
                status("Background blending completed")
        progress(0.85)
        
        # Step 5: Apply output volume (if not 1.0)
        output_volume = getattr(request, 'output_volume', 1.0)
        if output_volume != 1.0:
            check_cancelled()
            status(f"Applying output volume ({int(output_volume * 100)}%)...")
            
            def do_volume():
                return apply_output_volume(current_path, output_volume, request_id=request_id)
            
            volume_path = await asyncio.get_event_loop().run_in_executor(executor, do_volume)
            temp_files.append(volume_path)
            current_path = volume_path
        progress(0.90)
        
        # Step 6: Save to output folder
        try:
            status("Saving to output folder...")
            
            def do_save():
                return run_save(current_path, request.input, lambda s: None, request_id=request_id)
            
            saved_path = await asyncio.get_event_loop().run_in_executor(executor, do_save)
            status(f"Saved to: {saved_path}")
        except Exception as e:
            # Don't fail the pipeline if saving fails - just log and continue
            status(f"Warning: Could not save to output folder: {e}")
        progress(0.95)
        
        # Step 7: Convert to requested format
        status("Converting to output format...")
        
        def do_convert():
            return convert_to_format(current_path, request.response_format, request.speed)
        
        audio_data = await asyncio.to_thread(do_convert)
        mime_type = get_mime_type(request.response_format)
        
        progress(1.0)
        status("Complete!")
        
        return AudioPipelineResult(
            success=True,
            audio_data=audio_data,
            mime_type=mime_type,
            temp_files=temp_files,
        )
        
    except CancelledException as e:
        status("Generation cancelled")
        return AudioPipelineResult(
            success=False,
            error="Generation cancelled",
            temp_files=temp_files,
        )
    except Exception as e:
        return AudioPipelineResult(
            success=False,
            error=str(e),
            temp_files=temp_files,
        )
    finally:
        _unregister_request(request_id)


def generate_audio_sync(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> AudioPipelineResult:
    """
    Synchronous wrapper for generate_audio.
    
    Use this for non-async contexts.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            generate_audio(request, status_callback, progress_callback)
        )
    finally:
        loop.close()


async def generate_audio_streaming(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    request_id: Optional[str] = None,
):
    """
    STREAMING PIPELINE: Real-time audio generation with pipelined processing.
    
    TTS -> RVC -> PostProcess -> yield (async pipelined)
    
    Uses STREAMING endpoints (/v1/tts/stream, /v1/rvc/stream) that return 
    SSE events with audio chunks as they're generated.
    
    While RVC processes chunk 1, TTS can generate chunk 2, etc.
    Background blending is skipped in streaming mode.
    
    For batch processing, use generate_audio() instead.
    
    Yields:
        Dict with type ('start', 'chunk', 'complete', 'error') and data
    """
    import base64
    from pydub import AudioSegment
    
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    
    # Register this request for cancellation tracking
    _register_request(request_id)
    
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"[{request_id}] Stream: {msg}")
    
    try:
        config = get_config()
        
        # Determine what's enabled
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_background = request.enable_background if request.enable_background is not None else is_background_enabled()
        
        # Resolve prompt audio
        prompt_audio = request.chatterbox_prompt_audio
        if prompt_audio:
            prompt_audio = resolve_audio_path(prompt_audio)
        if not prompt_audio or not os.path.exists(prompt_audio or ""):
            yield {"type": "error", "message": "Chatterbox requires a prompt audio file"}
            return
        
        # Get prepared prompt audio from cache (or prepare if not cached)
        # This avoids re-preparing the same prompt for every generation
        try:
            prepared_prompt = get_prepared_prompt(prompt_audio, request_id)
            # Don't add to temp_files - cached prompts are managed separately
        except Exception as e:
            yield {"type": "error", "message": f"Failed to prepare prompt audio: {e}"}
            return
        
        # Split text into chunks for streaming
        # Short text (single sentences from SillyTavern) won't be split further
        # Long text (from main UI) will be chunked for progressive streaming
        from util.text_utils import split_text
        chunks = split_text(
            request.input, 
            max_tokens=request.tts_batch_tokens, 
            token_method=request.tts_token_method
        )
        
        if not chunks:
            yield {"type": "error", "message": "No text to process"}
            return
        
        # Get clients and params
        from util.clients import get_chatterbox_client
        chatterbox = get_chatterbox_client()
        
        rvc_model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
        # Don't pass RVC params - let server use config.json directly
        rvc_params = {} if do_rvc else None
        
        # Get post params and check if any effects are actually enabled
        post_params_obj = request.get_post_params() if do_post else None
        post_needs_processing = post_params_obj.needs_processing() if post_params_obj else False
        post_params = post_params_obj.to_dict() if post_needs_processing else None
        
        # Update do_post to reflect if processing is actually needed
        if do_post and not post_needs_processing:
            print(f"[{request_id}] Post-processing disabled (no effects enabled)")
            do_post = False
        
        executor = get_executor()
        total_chunks = len(chunks)
        
        # Gather background tracks for client-side mixing
        background_tracks = []
        if do_background:
            bg_params = request.get_background_params()
            
            if bg_params.use_config_tracks:
                # Use tracks from config
                for track in get_bg_tracks():
                    if track and track.get("file"):
                        resolved = resolve_audio_path(str(track["file"]))
                        if resolved and os.path.exists(resolved):
                            vol = float(track.get("volume", 0.3))
                            delay = float(track.get("delay", 0))
                            fade_in = float(track.get("fade_in", 0))
                            fade_out = float(track.get("fade_out", 0))
                            if vol > 0:
                                background_tracks.append({
                                    "file": resolved,
                                    "volume": vol,
                                    "delay": delay,
                                    "fade_in": fade_in,
                                    "fade_out": fade_out,
                                })
            else:
                # Use tracks from request
                for i, f in enumerate(bg_params.files):
                    resolved = resolve_audio_path(f)
                    if resolved and os.path.exists(resolved):
                        vol = bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3
                        delay = bg_params.delays[i] if i < len(bg_params.delays) else 0.0
                        fade_in = bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0.0
                        fade_out = bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0.0
                        if vol > 0:
                            background_tracks.append({
                                "file": resolved,
                                "volume": vol,
                                "delay": delay,
                                "fade_in": fade_in,
                                "fade_out": fade_out,
                            })
        
        # Background audio: client will fetch a separate stream from audio_services
        # and play it in sync with voice chunks (starts on first chunk, stops when done)
        has_background = len(background_tracks) > 0
        bg_session_id = request_id if has_background else None
        
        if has_background:
            print(f"[{request_id}] Background audio enabled: {len(background_tracks)} tracks (client will stream)")
        
        # Preload RVC model ASYNC - but skip if already preloaded this session
        # Uses module-level cache to avoid repeated preload calls
        rvc_preload_future = None
        if do_rvc and rvc_model:
            if not hasattr(generate_audio_streaming, '_preloaded_models'):
                generate_audio_streaming._preloaded_models = set()
            
            if rvc_model not in generate_audio_streaming._preloaded_models:
                print(f"[{request_id}] Preloading RVC model: {rvc_model}...")
                
                def preload_rvc():
                    try:
                        client = get_rvc_client()
                        result = client.load_model(rvc_model)
                        generate_audio_streaming._preloaded_models.add(rvc_model)
                        return result
                    except Exception as e:
                        print(f"[{request_id}] RVC preload failed: {e}")
                        return None
                
                rvc_preload_future = asyncio.get_event_loop().run_in_executor(executor, preload_rvc)
            else:
                print(f"[{request_id}] RVC model '{rvc_model}' already preloaded, skipping")
        
        # Send start event with background stream info
        # Client will call /v1/background/stream to get background audio separately
        yield {
            "type": "start",
            "request_id": request_id,  # For cancellation
            "chunks": total_chunks,
            "rvc_enabled": do_rvc,
            "post_enabled": do_post,
            "background_enabled": has_background,
            "background_session_id": bg_session_id,
            "background_tracks": background_tracks if has_background else [],  # Client needs track info for stream request
            "sample_rate": PIPELINE_SAMPLE_RATE,  # So client knows the rate
        }
        
        status(f"Starting pipelined generation: {total_chunks} chunks")
        
        # Create pipeline queues
        # Each queue holds: (index, chunk_text, audio_path, temp_files_list)
        tts_queue = asyncio.Queue()      # Input to TTS stage
        rvc_queue = asyncio.Queue()      # TTS output -> RVC input
        post_queue = asyncio.Queue()     # RVC output -> Post input
        output_queue = asyncio.Queue()   # Post output -> yield
        
        # Sentinel value to signal completion
        DONE = object()
        CANCELLED = object()  # Signal cancellation
        
        # Error tracking
        pipeline_error = None
        pipeline_cancelled = False
        
        def check_pipeline_cancelled():
            return pipeline_cancelled or is_cancelled(request_id)
        
        # TTS Stage - generates audio from text chunks
        # Note: text is already split, so each chunk is processed as-is by Chatterbox
        tts_max_tokens = request.tts_batch_tokens or 100
        
        async def tts_stage():
            nonlocal pipeline_error, pipeline_cancelled
            try:
                while True:
                    if check_pipeline_cancelled():
                        await rvc_queue.put(CANCELLED)
                        break
                    
                    item = await tts_queue.get()
                    if item is DONE or item is CANCELLED:
                        await rvc_queue.put(DONE if item is DONE else CANCELLED)
                        break
                    
                    idx, chunk_text = item
                    status(f"TTS chunk {idx+1}/{total_chunks}")
                    
                    def do_tts(text=chunk_text, i=idx, req_id=request_id):
                        print(f"[{req_id}] TTS chunk {i+1} starting via STREAM endpoint")
                        # Seed: -1 = random each time, 0 = no seeding, >0 = specific seed
                        if request.chatterbox_seed == -1:
                            chunk_seed = -1  # Let server generate random
                        elif request.chatterbox_seed > 0:
                            chunk_seed = request.chatterbox_seed + i
                        else:
                            chunk_seed = 0
                        # Use STREAMING endpoint - streams SSE chunks from Chatterbox
                        result = chatterbox.generate_stream(
                            text=text,
                            prompt_audio_path=prepared_prompt,
                            seed=chunk_seed,
                            max_tokens=tts_max_tokens,
                            request_id=req_id,
                        )
                        print(f"[{req_id}] TTS chunk {i+1} finished (streamed)")
                        return result
                    
                    tts_path = await asyncio.get_event_loop().run_in_executor(executor, do_tts)
                    
                    if check_pipeline_cancelled():
                        print(f"[{request_id}] Cancelled after TTS chunk {idx+1}!")
                        await rvc_queue.put(CANCELLED)
                        break
                    
                    await rvc_queue.put((idx, chunk_text, tts_path, [tts_path]))
            except Exception as e:
                pipeline_error = e
                await rvc_queue.put(DONE)
        
        # RVC Stage - voice conversion
        async def rvc_stage():
            nonlocal pipeline_error, pipeline_cancelled
            try:
                while True:
                    if check_pipeline_cancelled():
                        await post_queue.put(CANCELLED)
                        break
                    
                    item = await rvc_queue.get()
                    if item is DONE or item is CANCELLED:
                        await post_queue.put(DONE if item is DONE else CANCELLED)
                        break
                    
                    idx, chunk_text, audio_path, temps = item
                    
                    if do_rvc:
                        status(f"RVC chunk {idx+1}/{total_chunks} (streaming)")
                        current = audio_path
                        
                        def do_rvc_convert(path=current, req_id=request_id):
                            # Use STREAMING endpoint for RVC
                            return run_rvc_stream(path, rvc_model, rvc_params, lambda s: None, None, request_id=req_id)
                        
                        rvc_path = await asyncio.get_event_loop().run_in_executor(executor, do_rvc_convert)
                        
                        if check_pipeline_cancelled():
                            await post_queue.put(CANCELLED)
                            break
                        
                        temps.append(rvc_path)
                        # Resample + volume is done in post_stage (combined for efficiency)
                        await post_queue.put((idx, chunk_text, rvc_path, temps))
                    else:
                        await post_queue.put((idx, chunk_text, audio_path, temps))
            except Exception as e:
                pipeline_error = e
                await post_queue.put(DONE)
        
        # Post-processing Stage (includes resample + output volume combined)
        output_volume = getattr(request, 'output_volume', 1.0)
        
        # Track elapsed time for spatial audio continuity in streaming mode
        # Without this, each chunk's panning would reset to the start position
        streaming_elapsed_time = 0.0
        
        async def post_stage():
            nonlocal pipeline_error, pipeline_cancelled, streaming_elapsed_time
            try:
                while True:
                    if check_pipeline_cancelled():
                        await output_queue.put(CANCELLED)
                        break
                    
                    item = await post_queue.get()
                    if item is DONE or item is CANCELLED:
                        await output_queue.put(DONE if item is DONE else CANCELLED)
                        break
                    
                    idx, chunk_text, audio_path, temps = item
                    current = audio_path
                    
                    # Post-processing effects (EQ, reverb, etc.)
                    if do_post:
                        status(f"Post chunk {idx+1}/{total_chunks}")
                        
                        # Inject time offset for spatial audio continuity
                        # This ensures panning continues from where the last chunk ended
                        chunk_post_params = post_params.copy() if post_params else {}
                        chunk_post_params['spatial_time_offset'] = streaming_elapsed_time
                        
                        print(f"[{request_id}] Post chunk {idx+1}: spatial_time_offset={streaming_elapsed_time:.2f}s")
                        
                        def do_post_process(path=current, params=chunk_post_params, req_id=request_id):
                            return run_postprocess(path, params, lambda s: None, request_id=req_id)
                        
                        post_path = await asyncio.get_event_loop().run_in_executor(executor, do_post_process)
                        temps.append(post_path)
                        current = post_path
                    
                    # Combined resample (44.1kHz) + volume adjustment in one operation
                    # This is much faster than separate operations
                    def do_final_process(path=current, vol=output_volume, req_id=request_id):
                        return resample_and_adjust_volume(
                            path, 
                            target_sr=PIPELINE_SAMPLE_RATE, 
                            volume=vol, 
                            request_id=req_id
                        )
                    
                    final_path = await asyncio.get_event_loop().run_in_executor(executor, do_final_process)
                    if final_path != current:
                        temps.append(final_path)
                        current = final_path
                    
                    # Update elapsed time for spatial audio continuity
                    # Get chunk duration from the final processed file
                    try:
                        import soundfile as sf
                        info = sf.info(current)
                        chunk_duration = info.duration
                        streaming_elapsed_time += chunk_duration
                        print(f"[{request_id}] Chunk {idx+1} duration: {chunk_duration:.2f}s, total elapsed: {streaming_elapsed_time:.2f}s")
                    except Exception as e:
                        # Fallback: estimate from sample rate
                        streaming_elapsed_time += 5.0  # Assume ~5s chunks
                        print(f"[{request_id}] Chunk {idx+1} duration unknown (fallback 5s), total elapsed: {streaming_elapsed_time:.2f}s")
                    
                    await output_queue.put((idx, chunk_text, current, temps))
            except Exception as e:
                pipeline_error = e
                await output_queue.put(DONE)
        
        # Start pipeline stages as background tasks
        tts_task = asyncio.create_task(tts_stage())
        rvc_task = asyncio.create_task(rvc_stage())
        post_task = asyncio.create_task(post_stage())
        
        # Feed all chunks into the TTS queue upfront
        for i, chunk_text in enumerate(chunks):
            await tts_queue.put((i, chunk_text))
        await tts_queue.put(DONE)
        
        # Consume output queue and yield results
        chunks_sent = 0
        audio_segments = []  # Collect audio segments for final combine+save
        total_duration = 0.0
        
        while True:
            # Check for cancellation before waiting for next item
            if check_pipeline_cancelled():
                pipeline_cancelled = True
                status("Generation cancelled")
                yield {"type": "cancelled", "message": "Generation cancelled", "chunks_sent": chunks_sent}
                break
            
            item = await output_queue.get()
            if item is DONE:
                break
            if item is CANCELLED:
                status("Generation cancelled")
                yield {"type": "cancelled", "message": "Generation cancelled", "chunks_sent": chunks_sent}
                break
            
            if pipeline_error:
                raise pipeline_error
            
            idx, chunk_text, audio_path, temps = item
            
            try:
                # Read and encode audio for streaming to client
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Get duration and collect segment for final save
                try:
                    seg = AudioSegment.from_file(audio_path)
                    duration = len(seg) / 1000.0
                    audio_segments.append(seg)  # Collect for final combine
                    total_duration += duration
                except:
                    duration = 0
                
                status(f"Chunk {idx+1}/{total_chunks}: Complete ({duration:.1f}s)")
                chunks_sent += 1
                
                yield {
                    "type": "chunk",
                    "index": idx,
                    "total": total_chunks,
                    "audio": audio_b64,
                    "duration": round(duration, 2),
                    "text": chunk_text[:100],
                }
            finally:
                # Clean up temp files for this chunk
                for tf in temps:
                    try:
                        if tf and os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
        
        # Wait for all tasks to complete
        await asyncio.gather(tts_task, rvc_task, post_task, return_exceptions=True)
        
        if pipeline_error:
            raise pipeline_error
        
        # Only yield complete if not cancelled
        if not pipeline_cancelled and not check_pipeline_cancelled():
            # Combine all chunks and save once at the end
            saved_path = None
            if audio_segments:
                try:
                    status("Combining and saving final audio...")
                    
                    # Combine all segments
                    combined = audio_segments[0]
                    for seg in audio_segments[1:]:
                        combined += seg
                    
                    # Write combined to temp file
                    fd, combined_path = tempfile.mkstemp(suffix="_combined.wav")
                    os.close(fd)
                    combined.export(combined_path, format="wav")
                    
                    # Save to output folder (single file, not per-chunk)
                    def do_final_save():
                        return run_save(combined_path, request.input[:50], lambda s: None, request_id=request_id)
                    
                    saved_path = await asyncio.get_event_loop().run_in_executor(executor, do_final_save)
                    status(f"Saved: {os.path.basename(saved_path)}")
                    
                    # Cleanup temp combined file
                    try:
                        os.remove(combined_path)
                    except:
                        pass
                        
                except Exception as save_err:
                    status(f"Save failed: {save_err}")
            
            status("Streaming complete")
            yield {
                "type": "complete", 
                "chunks_sent": chunks_sent,
                "total_duration": round(total_duration, 2),
                "file": saved_path,  # Single combined file path
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
    
    finally:
        # Background stream cleanup is handled by client calling /v1/background/stop-stream
        # when voice playback completes
        
        # Unregister request
        _unregister_request(request_id)
        
        # Clean up shared temp files
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf):
                    os.remove(tf)
            except:
                pass
