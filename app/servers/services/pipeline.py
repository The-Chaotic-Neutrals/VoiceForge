"""
Audio Pipeline Service - Thin coordinator for TTS → RVC → PostProcess → Blend.

This module just coordinates calls to the microservices via their clients.
All actual processing happens in the servers:
- chatterbox_server: TTS generation
- rvc_server: Voice conversion  
- audio_services_server: Post-processing, blending, resampling, saving
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Set
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf  # For debug logging

from util.clients import (
    get_chatterbox_client,
    get_soprano_client,
    run_rvc,
    run_rvc_stream,
    run_postprocess,
    run_blend,
    run_save,
    run_resample,
)
from util.audio_utils import convert_to_format, get_mime_type, get_audio_info
from util.file_utils import resolve_audio_path
from config import get_config, is_rvc_enabled, is_post_enabled, is_background_enabled, get_bg_tracks
from servers.models.requests import TTSRequest

# Pipeline sample rate (44.1kHz CD quality)
PIPELINE_SAMPLE_RATE = 44100

# Shared executor for blocking operations
_executor: Optional[ThreadPoolExecutor] = None

def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", os.cpu_count() or 4)))
    return _executor


# Simple cancellation tracking
_cancelled: Set[str] = set()

def cancel_generation(request_id: str) -> bool:
    _cancelled.add(request_id)
    return True

def cancel_all_generations() -> int:
    count = len(_active)
    _cancelled.update(_active)
    return count

def is_cancelled(request_id: str) -> bool:
    return request_id in _cancelled

def get_active_requests() -> List[str]:
    return list(_active)

_active: Set[str] = set()


class CancelledException(Exception):
    pass


def _needs_resample(path: str, target_sr: int, volume: float) -> bool:
    info = get_audio_info(path)
    current_sr = info.get("samplerate")
    if abs(volume - 1.0) >= 0.01:
        return True
    if current_sr is None:
        return True
    return current_sr != target_sr


def _get_tts_backend(request: TTSRequest) -> str:
    backend = (getattr(request, "tts_backend", None) or "chatterbox").lower()
    if backend not in ("chatterbox", "soprano"):
        backend = "chatterbox"
    return backend


@dataclass
class AudioPipelineResult:
    """Result of audio pipeline execution."""
    success: bool
    audio_data: Optional[bytes] = None
    mime_type: str = "audio/mpeg"
    error: Optional[str] = None
    temp_files: List[str] = field(default_factory=list)
    
    def cleanup(self):
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
    Pipeline: TTS → RVC → PostProcess → Blend → Format
    
    Each step calls the appropriate server via clients.py.
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    _active.add(request_id)
    _cancelled.discard(request_id)
    
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"[{request_id}] {msg}")
    
    def progress(p: float):
        if progress_callback:
            progress_callback(p)
    
    def check():
        if is_cancelled(request_id):
            raise CancelledException()
    
    try:
        config = get_config()
        executor = get_executor()
        tts_backend = _get_tts_backend(request)
        
        # Flags
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
        
        # Resolve prompt audio (only for Chatterbox)
        prompt = None
        if tts_backend == "chatterbox":
            prompt = resolve_audio_path(request.chatterbox_prompt_audio)
            if not prompt or not os.path.exists(prompt or ""):
                return AudioPipelineResult(success=False, error="Prompt audio required", temp_files=temp_files)
        
        # === Step 1: TTS ===
        check()
        status("Generating TTS...")
        progress(0.1)
        
        if tts_backend == "soprano":
            soprano = get_soprano_client()
            # Pass through request values - None values use soprano module defaults
            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: soprano.generate(
                    text=request.input,
                    temperature=request.soprano_temperature,
                    top_p=request.soprano_top_p,
                    repetition_penalty=request.soprano_repetition_penalty,
                    request_id=request_id
                )
            )
        else:
            chatterbox = get_chatterbox_client()
            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: chatterbox.generate(
                    text=request.input,
                    prompt_audio_path=prompt,
                    seed=request.chatterbox_seed or 0,
                    max_tokens=request.tts_batch_tokens or 200,
                )
            )
        temp_files.append(tts_path)
        current = tts_path
        progress(0.3)
        
        # === Step 2: RVC ===
        check()
        if do_rvc:
            status("Running RVC...")
            model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
            rvc_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_rvc(current, model, {}, request_id=request_id)
            )
            temp_files.append(rvc_path)
            current = rvc_path
        progress(0.5)
        
        # === Step 2.5: Normalize to Pipeline Sample Rate ===
        # CRITICAL: Different TTS backends output different sample rates (Chatterbox=24kHz, Soprano=32kHz)
        # Post-processing effects (especially spatial audio) behave differently at different sample rates
        # Normalizing here ensures IDENTICAL behavior regardless of TTS backend
        check()
        if do_post and _needs_resample(current, PIPELINE_SAMPLE_RATE, 1.0):
            status("Normalizing sample rate...")
            norm_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_resample(current, PIPELINE_SAMPLE_RATE, 1.0, request_id=request_id)
            )
            if norm_path != current:
                temp_files.append(norm_path)
                current = norm_path
        
        # === Step 3: Post-Processing ===
        check()
        if do_post:
            post_params = request.get_post_params()
            if post_params.needs_processing():
                status("Post-processing...")
                post_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_postprocess(current, post_params.to_dict(), request_id=request_id)
                )
                temp_files.append(post_path)
                current = post_path
        progress(0.7)
        
        # === Step 4: Background Blend ===
        check()
        if do_bg:
            bg_params = request.get_background_params()
            bg_files, bg_vols, bg_delays, bg_fins, bg_fouts = [], [], [], [], []
            
            tracks = get_bg_tracks() if bg_params.use_config_tracks else [
                {"file": f, "volume": bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3,
                 "delay": bg_params.delays[i] if i < len(bg_params.delays) else 0,
                 "fade_in": bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0,
                 "fade_out": bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0}
                for i, f in enumerate(bg_params.files)
            ]
            
            for t in tracks:
                if not t or not t.get("file"):
                    continue
                resolved = resolve_audio_path(str(t["file"]))
                if resolved and os.path.exists(resolved) and float(t.get("volume", 0.3)) > 0:
                    bg_files.append(resolved)
                    bg_vols.append(float(t.get("volume", 0.3)))
                    bg_delays.append(float(t.get("delay", 0)))
                    bg_fins.append(float(t.get("fade_in", 0)))
                    bg_fouts.append(float(t.get("fade_out", 0)))
            
            if bg_files:
                status("Blending background...")
                blend_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_blend(current, bg_files, bg_vols, 1.0, bg_delays=bg_delays, 
                                     bg_fade_ins=bg_fins, bg_fade_outs=bg_fouts, request_id=request_id)
                )
                temp_files.append(blend_path)
                current = blend_path
        progress(0.85)
        
        # === Step 5: Output Volume ===
        vol = getattr(request, 'output_volume', 1.0)
        if _needs_resample(current, PIPELINE_SAMPLE_RATE, vol):
            vol_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_resample(current, PIPELINE_SAMPLE_RATE, vol, request_id=request_id)
            )
            if vol_path != current:
                temp_files.append(vol_path)
                current = vol_path
        progress(0.9)
        
        # === Step 6: Save (optional) ===
        if getattr(request, "save_output", False):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_save(current, request.input, request_id=request_id)
                )
            except Exception as e:
                status(f"Warning: Save failed: {e}")
        progress(0.95)
        
        # === Step 7: Convert to output format ===
        audio_data = await asyncio.to_thread(
            convert_to_format, current, request.response_format, request.speed
        )
        progress(1.0)
        status("Complete!")
        
        return AudioPipelineResult(
            success=True,
            audio_data=audio_data,
            mime_type=get_mime_type(request.response_format),
            temp_files=temp_files,
        )
        
    except CancelledException:
        return AudioPipelineResult(success=False, error="Cancelled", temp_files=temp_files)
    except Exception as e:
        return AudioPipelineResult(success=False, error=str(e), temp_files=temp_files)
    finally:
        _active.discard(request_id)
        _cancelled.discard(request_id)


async def generate_audio_streaming(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    request_id: Optional[str] = None,
):
    """
    Streaming pipeline: TTS → RVC → PostProcess per chunk, yielded as SSE events.
    
    Background blending info is included for client-side mixing.
    """
    import base64
    import tempfile
    import threading
    from pydub import AudioSegment
    
    request_id = request_id or str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    _active.add(request_id)
    _cancelled.discard(request_id)
    
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"[{request_id}] {msg}")
    
    try:
        config = get_config()
        executor = get_executor()
        
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
        tts_backend = _get_tts_backend(request)
        
        prompt = None
        if tts_backend == "chatterbox":
            prompt = resolve_audio_path(request.chatterbox_prompt_audio)
            if not prompt or not os.path.exists(prompt or ""):
                yield {"type": "error", "message": "Prompt audio required"}
                return
        
        # Gather background track info for client
        bg_tracks = []
        if do_bg:
            bg_params = request.get_background_params()
            tracks = get_bg_tracks() if bg_params.use_config_tracks else [
                {"file": f, "volume": bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3,
                 "delay": bg_params.delays[i] if i < len(bg_params.delays) else 0,
                 "fade_in": bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0,
                 "fade_out": bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0}
                for i, f in enumerate(bg_params.files)
            ]
            for t in tracks:
                if t and t.get("file"):
                    resolved = resolve_audio_path(str(t["file"]))
                    if resolved and os.path.exists(resolved):
                        bg_tracks.append({"file": resolved, "volume": float(t.get("volume", 0.3)),
                                         "delay": float(t.get("delay", 0)), "fade_in": float(t.get("fade_in", 0)),
                                         "fade_out": float(t.get("fade_out", 0))})
        
        chatterbox = get_chatterbox_client()
        soprano = get_soprano_client()
        rvc_model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
        post_params = request.get_post_params().to_dict() if do_post and request.get_post_params().needs_processing() else None
        output_vol = getattr(request, 'output_volume', 1.0)
        
        # Track time offset for spatial audio panning continuity
        spatial_time_offset = 0.0

        # Both Chatterbox and Soprano use the same streaming infrastructure
        event_queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_event_loop()
        
        def reader():
            try:
                if tts_backend == "soprano":
                    # Pass through request values - None values use soprano module defaults
                    print(f"[{request_id}] Starting Soprano stream_events...")
                    stream = soprano.stream_events(
                        text=request.input,
                        temperature=request.soprano_temperature,
                        top_p=request.soprano_top_p,
                        repetition_penalty=request.soprano_repetition_penalty,
                        chunk_size=request.tts_batch_tokens or 1,  # Soprano uses small chunk sizes
                        request_id=request_id,
                        stop_event=stop_event
                    )
                    print(f"[{request_id}] Got Soprano stream generator")
                else:
                    stream = chatterbox.stream_events(
                        text=request.input,
                        prompt_audio_path=prompt,
                        seed=request.chatterbox_seed or 0,
                        max_tokens=request.tts_batch_tokens or 200,
                        request_id=request_id,
                        stop_event=stop_event
                    )
                event_count = 0
                for event in stream:
                    # Check if cancelled BEFORE processing event
                    if stop_event.is_set():
                        print(f"[{request_id}] Reader thread: stop_event set, breaking after {event_count} events")
                        break
                    
                    event_count += 1
                    event_type = event.get("type", "unknown")
                    print(f"[{request_id}] Received event #{event_count}: type={event_type}")
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
                print(f"[{request_id}] Stream iteration complete, received {event_count} events")
            except Exception as e:
                print(f"[{request_id}] Stream reader exception: {e}")
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "error", "message": str(e)}
                )
            finally:
                print(f"[{request_id}] Reader thread finishing")
                loop.call_soon_threadsafe(event_queue.put_nowait, None)
        
        threading.Thread(target=reader, daemon=True).start()
        
        total_duration = 0.0
        total_chunks = None
        chunks_sent = 0
        
        while True:
            event = await event_queue.get()
            if event is None:
                break
            
            if is_cancelled(request_id):
                stop_event.set()
                yield {"type": "cancelled", "message": "Cancelled", "chunks_sent": chunks_sent}
                return
            
            event_type = event.get("type")
            
            if event_type == "start":
                total_chunks = event.get("chunks")
                yield {
                    "type": "start",
                    "request_id": request_id,
                    "chunks": total_chunks or 1,
                    "rvc_enabled": do_rvc,
                    "post_enabled": bool(post_params),
                    "background_enabled": bool(bg_tracks),
                    "background_tracks": bg_tracks,
                    "sample_rate": PIPELINE_SAMPLE_RATE,
                    "tts_backend": tts_backend,
                }
                status(f"Streaming {total_chunks or 1} chunks")
                continue
            
            if event_type == "error":
                yield {"type": "error", "message": event.get("message", "Unknown error")}
                return
            
            if event_type == "complete":
                yield {"type": "complete", "chunks_sent": chunks_sent, "total_duration": round(total_duration, 2)}
                return
            
            if event_type != "chunk":
                continue
            
            chunk_temps = []
            chunk_index = event.get("index", chunks_sent)
            chunk_text = event.get("text", "")
            
            # Write raw TTS chunk to temp file
            audio_bytes = event.get("audio_bytes")
            if not audio_bytes:
                continue
            
            fd, tts_path = tempfile.mkstemp(suffix="_tts_chunk.wav")
            os.close(fd)
            with open(tts_path, "wb") as f:
                f.write(audio_bytes)
            chunk_temps.append(tts_path)
            
            # Debug: log audio properties from TTS
            try:
                import soundfile as sf
                _info = sf.info(tts_path)
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: sr={_info.samplerate}, channels={_info.channels}, frames={_info.frames}, duration={_info.frames/_info.samplerate:.2f}s, bytes={len(audio_bytes)}")
            except Exception as e:
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: failed to get info: {e}")
            
            # Standard path - works for both Chatterbox (small chunks) and Soprano (single chunk)
            current = tts_path
            
            # RVC (blocking for small chunks is fine)
            if do_rvc:
                status(f"RVC chunk {chunk_index + 1}")
                rvc_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current: run_rvc_stream(p, rvc_model, {}, request_id=request_id)
                )
                chunk_temps.append(rvc_path)
                current = rvc_path
                # Debug: log after RVC
                try:
                    _info = sf.info(current)
                    print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after RVC: sr={_info.samplerate}, frames={_info.frames}")
                except:
                    pass
            
            # Normalize sample rate before post-processing
            # CRITICAL: Ensures spatial audio behaves identically for Chatterbox (24kHz) and Soprano (32kHz)
            if post_params and _needs_resample(current, PIPELINE_SAMPLE_RATE, 1.0):
                norm_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current: run_resample(p, PIPELINE_SAMPLE_RATE, 1.0, request_id=request_id)
                )
                if norm_path != current:
                    chunk_temps.append(norm_path)
                    current = norm_path
                    print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after resample: sr=44100")
            
            # Debug: log before post-process
            try:
                _info = sf.info(current)
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} BEFORE POST: sr={_info.samplerate}, frames={_info.frames}, duration={_info.frames/_info.samplerate:.2f}s")
            except:
                pass
            
            # Post-process
            if post_params:
                status(f"Post chunk {chunk_index + 1}")
                # Pass time offset for panning continuity across chunks
                chunk_post_params = post_params.copy()
                chunk_post_params["spatial_time_offset"] = spatial_time_offset
                post_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current, pp=chunk_post_params: run_postprocess(p, pp, request_id=request_id)
                )
                chunk_temps.append(post_path)
                current = post_path
            
            # Resample + volume (only if needed)
            try:
                _info = sf.info(current)
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} BEFORE FINAL RESAMPLE: sr={_info.samplerate}, target={PIPELINE_SAMPLE_RATE}")
            except:
                pass
            
            if _needs_resample(current, PIPELINE_SAMPLE_RATE, output_vol):
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: Resampling to {PIPELINE_SAMPLE_RATE}")
                final_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current: run_resample(p, PIPELINE_SAMPLE_RATE, output_vol, request_id=request_id)
                )
                if final_path != current:
                    chunk_temps.append(final_path)
                    current = final_path
                    try:
                        _info = sf.info(current)
                        print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} AFTER RESAMPLE: sr={_info.samplerate}")
                    except:
                        pass
            else:
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: No resample needed")
            
            # Read and encode
            with open(current, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            try:
                import soundfile as sf
                info = sf.info(current)
                duration = (info.frames / info.samplerate) if info.samplerate else 0
            except Exception:
                try:
                    seg = AudioSegment.from_file(current)
                    duration = len(seg) / 1000.0
                except Exception:
                    duration = 0
            
            total_duration += duration
            spatial_time_offset += duration  # Track time for panning continuity
            chunks_sent += 1
            
            yield {
                "type": "chunk",
                "index": chunk_index,
                "total": total_chunks or 1,
                "audio": audio_b64,
                "duration": round(duration, 2),
                "text": (chunk_text or "")[:100],
            }
            
            # Cleanup chunk temps
            for tf in chunk_temps:
                try:
                    if tf and os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
    finally:
        _active.discard(request_id)
        _cancelled.discard(request_id)
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf):
                    os.remove(tf)
            except:
                pass
