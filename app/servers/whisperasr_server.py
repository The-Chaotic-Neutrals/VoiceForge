# whisperasr_server.py
"""
Standalone Whisper ASR microservice using Faster Whisper (CTranslate2).

Runs in the whisper_asr conda environment.
The main VoiceForge server calls this via HTTP.
"""
import os
import sys
from pathlib import Path

# This file is in app/servers/, set up paths
SCRIPT_DIR = Path(__file__).parent  # app/servers
APP_DIR = SCRIPT_DIR.parent  # app
UTIL_DIR = APP_DIR / "util"  # app/util
CONFIG_DIR = APP_DIR / "config"  # app/config
MODELS_DIR = APP_DIR / "models"  # app/models

# Add paths for imports
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(MODELS_DIR))

# Set up logging BEFORE any other imports
from logging_utils import create_server_logger, suppress_all_logging

# Create server-specific logging functions
log_info, log_warn, log_error = create_server_logger("WHISPERASR")

log_info("Starting Whisper ASR server...")

import warnings
warnings.filterwarnings("ignore")

import logging
import tempfile
import traceback
import time
import platform
from contextlib import contextmanager

# Suppress all library logging
suppress_all_logging()

# Windows-specific fix for temp file cleanup issues
if platform.system() == "Windows":
    _original_temp_dir_cleanup = tempfile.TemporaryDirectory.cleanup
    
    def _safe_temp_cleanup(self):
        """Wrapper that ignores Windows file lock errors during cleanup."""
        try:
            _original_temp_dir_cleanup(self)
        except (PermissionError, NotADirectoryError, OSError):
            pass
    
    tempfile.TemporaryDirectory.cleanup = _safe_temp_cleanup
    log_info("Applied Windows temp file cleanup fix")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import soundfile as sf
import numpy as np
from typing import Optional, Literal, AsyncGenerator
import threading
import json
import asyncio
import queue
import io
import base64

# Check PyTorch/CUDA
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log_info(f"CUDA available: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
else:
    log_warn("=" * 60)
    log_warn("CUDA NOT AVAILABLE - ASR will run on CPU (slow!)")
    log_warn("To fix: Install CUDA-enabled PyTorch in whisper_asr env:")
    log_warn("  conda activate whisper_asr")
    log_warn("  pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    log_warn("=" * 60)

# UVR5 vocal cleaning (preprocess microservice)
try:
    from clients import clean_vocals_uvr5, is_preprocess_server_available
    PREPROCESS_CLIENT_AVAILABLE = True
except Exception as e:
    PREPROCESS_CLIENT_AVAILABLE = False

# Postprocess client for audio enhancement
try:
    from clients import run_postprocess, is_postprocess_server_available
    POSTPROCESS_CLIENT_AVAILABLE = True
except Exception as e:
    POSTPROCESS_CLIENT_AVAILABLE = False
    log_warn(f"Postprocess client not available: {e}")

# Faster Whisper
FASTER_WHISPER_AVAILABLE = False
FASTER_WHISPER_ERROR = None
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    log_info("Faster Whisper available")
except ImportError as e:
    FASTER_WHISPER_ERROR = str(e)
    log_error(f"Faster Whisper not installed: {e}")

# Whisper model management
_whisper_model = None
_whisper_model_name = None
_whisper_model_lock = threading.Lock()

# Available Faster Whisper models
FASTER_WHISPER_MODELS = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}
DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# Environment config
WHISPERASR_MODEL_NAME = os.getenv("WHISPERASR_MODEL_NAME", os.getenv("ASR_MODEL_NAME", f"whisper-{DEFAULT_WHISPER_MODEL}"))

# Thread pool for GPU operations
from concurrent.futures import ThreadPoolExecutor
_cuda_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda_worker")


def load_whisper_model(model_name: str = None):
    """Load Faster Whisper model (thread-safe, lazy). Can switch models."""
    global _whisper_model, _whisper_model_name
    
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(f"Faster Whisper not available: {FASTER_WHISPER_ERROR}")
    
    if model_name is None:
        model_name = DEFAULT_WHISPER_MODEL
    
    # Map friendly names to model identifiers
    model_id = FASTER_WHISPER_MODELS.get(model_name, model_name)
    
    with _whisper_model_lock:
        # Check if we need to switch models
        if _whisper_model is not None and _whisper_model_name != model_name:
            log_info(f"Switching Whisper model from {_whisper_model_name} to {model_name}...")
            del _whisper_model
            _whisper_model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if _whisper_model is None:
            log_info(f"Loading Faster Whisper model: {model_name} ({model_id})")
            t_start = time.perf_counter()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            _whisper_model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(APP_DIR / "models" / "whisper"),
            )
            
            _whisper_model_name = model_name
            t_load = time.perf_counter() - t_start
            log_info(f"Whisper model loaded in {t_load:.1f}s on {device.upper()} ({compute_type})")
        
        return _whisper_model


def unload_whisper_model():
    """Unload Whisper model and free GPU memory."""
    global _whisper_model, _whisper_model_name
    
    freed_gb = 0.0
    with _whisper_model_lock:
        if _whisper_model is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1e9
            
            del _whisper_model
            _whisper_model = None
            _whisper_model_name = None
            
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / 1e9
                freed_gb = before - after
            
            log_info(f"Whisper model unloaded, freed {freed_gb:.2f}GB")
    
    return freed_gb


def transcribe_with_whisper(audio_path: str, model_name: str = None, language: str = "en") -> dict:
    """Transcribe audio using Faster Whisper."""
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(f"Faster Whisper not available: {FASTER_WHISPER_ERROR}")
    
    model = load_whisper_model(model_name)
    
    audio_info = sf.info(audio_path)
    total_duration = audio_info.duration
    
    segments_iter, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    
    segments = []
    full_text_parts = []
    last_end_time = 0
    last_text = ""
    repeat_count = 0
    max_repeats = 3
    max_expected_segments = int(total_duration / 0.5) + 50
    
    for segment in segments_iter:
        # Safety checks for loops
        if segment.end <= last_end_time and len(segments) > 5:
            continue
        
        if segment.text.strip() == last_text and last_text:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue
        else:
            repeat_count = 0
        
        last_end_time = segment.end
        last_text = segment.text.strip()
        
        segments.append({
            "id": len(segments),
            "seek": int(segment.seek),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        })
        full_text_parts.append(segment.text.strip())
        
        if len(segments) > max_expected_segments:
            log_warn(f"[WHISPER] Breaking: too many segments ({len(segments)})")
            break
    
    full_text = " ".join(full_text_parts)
    
    return {
        "text": full_text,
        "segments": segments,
        "language": info.language,
        "duration": total_duration,
    }


def transcribe_with_whisper_streaming(
    audio_path: str,
    update_queue: queue.Queue,
    model_name: str = None,
    language: str = "en"
) -> str:
    """Transcribe audio using Whisper with chunked streaming updates."""
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(f"Faster Whisper not available: {FASTER_WHISPER_ERROR}")
    
    def send_update(type_: str, data: dict):
        update_queue.put({"type": type_, **data})
    
    model = load_whisper_model(model_name)
    
    # Load audio
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    total_duration = len(audio_data) / sr
    log_info(f"[WHISPER-STREAM] Transcribing: {audio_path} ({total_duration:.1f}s)")
    t_start = time.perf_counter()
    
    # Chunked processing for real-time updates
    chunk_duration = 30.0  # seconds per chunk
    overlap_duration = 2.0  # overlap between chunks
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    
    all_text_parts = []
    processed_duration = 0.0
    chunk_idx = 0
    
    send_update("status", {"message": f"Processing {total_duration:.1f}s audio...", "progress": 10})
    
    # Process in chunks
    pos = 0
    while pos < len(audio_data):
        chunk_end = min(pos + chunk_samples, len(audio_data))
        chunk = audio_data[pos:chunk_end]
        chunk_duration_actual = len(chunk) / sr
        
        # Save chunk to temp file
        chunk_path = audio_path + f"_chunk{chunk_idx}.wav"
        sf.write(chunk_path, chunk, sr)
        
        try:
            # Transcribe chunk
            segments_iter, info = model.transcribe(
                chunk_path,
                language=language if language != "auto" else None,
                beam_size=5,
                best_of=3,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
            )
            
            chunk_texts = []
            for segment in segments_iter:
                text = segment.text.strip()
                if text:
                    chunk_texts.append(text)
                    
                    # Send segment update with adjusted timestamps
                    seg_data = {
                        "id": len(all_text_parts) + len(chunk_texts) - 1,
                        "start": processed_duration + segment.start,
                        "end": processed_duration + segment.end,
                        "text": text,
                    }
                    send_update("segment", {
                        "segment": seg_data,
                        "duration_processed": processed_duration + segment.end,
                        "total_duration": total_duration,
                    })
            
            if chunk_texts:
                chunk_text = " ".join(chunk_texts)
                all_text_parts.append(chunk_text)
                
                # Send chunk text update
                current_full_text = " ".join(all_text_parts)
                send_update("text", {
                    "text": current_full_text,
                    "chunk_text": chunk_text,
                    "chunk_idx": chunk_idx,
                    "duration_processed": processed_duration + chunk_duration_actual,
                })
            
        finally:
            # Cleanup chunk file
            try:
                os.unlink(chunk_path)
            except:
                pass
        
        # Update progress
        processed_duration += chunk_duration_actual - (overlap_duration if chunk_end < len(audio_data) else 0)
        progress = min(90, int(10 + (processed_duration / total_duration) * 80))
        send_update("progress", {"progress": progress, "duration_processed": processed_duration})
        
        # Move to next chunk with overlap
        pos += chunk_samples - overlap_samples
        chunk_idx += 1
    
    # Final result
    full_text = " ".join(all_text_parts)
    
    t_elapsed = time.perf_counter() - t_start
    rtf = t_elapsed / total_duration if total_duration > 0 else 0
    log_info(f"[WHISPER-STREAM] Done: {len(full_text)} chars, {chunk_idx} chunks, RTF={rtf:.2f}")
    
    send_update("progress", {"progress": 100, "duration_processed": total_duration})
    send_update("complete", {"text": full_text})
    
    return full_text


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="VoiceForge Whisper ASR Server",
    description="Speech-to-text transcription using Faster Whisper",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    whisper_available: bool
    whisper_model_loaded: bool
    whisper_model_name: Optional[str]
    cuda_available: bool
    gpu_name: Optional[str]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
    
    return HealthResponse(
        status="healthy",
        whisper_available=FASTER_WHISPER_AVAILABLE,
        whisper_model_loaded=_whisper_model is not None,
        whisper_model_name=_whisper_model_name,
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
    )


@app.post("/warmup")
async def warmup():
    """Pre-load Whisper model for faster first inference."""
    try:
        load_whisper_model()
        return {"status": "ok", "message": "Whisper model loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


@app.post("/clear_gpu_cache")
async def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1e9
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        after = torch.cuda.memory_allocated() / 1e9
        return {"freed_gb": round(before - after, 2), "current_gb": round(after, 2)}
    return {"message": "CUDA not available"}


@app.post("/unload")
async def unload_model():
    """Unload Whisper model and free GPU memory."""
    freed = unload_whisper_model()
    return {"success": True, "message": "Model unloaded", "freed_gb": round(freed, 2)}


@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models."""
    return {
        "whisper": {
            "available": FASTER_WHISPER_AVAILABLE,
            "loaded": _whisper_model is not None,
            "model_name": _whisper_model_name,
            "available_models": list(FASTER_WHISPER_MODELS.keys()),
        }
    }


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _clean_vocals(audio_path: str, skip_if_exists: bool = True, original_filename: str = None) -> str:
    """Clean vocals using preprocess microservice (UVR5)."""
    if not PREPROCESS_CLIENT_AVAILABLE:
        log_warn("Preprocess client not available, skipping vocal cleaning")
        return audio_path

    if not is_preprocess_server_available():
        log_warn("Preprocess server not available, skipping vocal cleaning")
        return audio_path

    try:
        vocals_path = clean_vocals_uvr5(
            audio_path, 
            aggression=10, 
            device=None,
            skip_if_cached=skip_if_exists,
            original_filename=original_filename,
        )
        return vocals_path
    except Exception as e:
        log_warn(f"Vocal cleaning failed: {e}")
        return audio_path


def _postprocess_audio(audio_path: str, params: dict = None) -> str:
    """Apply post-processing to audio before transcription."""
    if not POSTPROCESS_CLIENT_AVAILABLE:
        log_warn("Postprocess client not available, skipping audio enhancement")
        return audio_path

    if not is_postprocess_server_available():
        log_warn("Postprocess server not available, skipping audio enhancement")
        return audio_path

    try:
        if params is None:
            params = {
                "highpass": 80.0,
                "lowpass": 12000.0,
                "bass_freq": 60.0,
                "bass_gain": 0.0,
                "treble_freq": 8000.0,
                "treble_gain": 0.0,
                "reverb_in_gain": 0.0,
                "reverb_out_gain": 0.0,
                "reverb_delay": 0.0,
                "reverb_decay": 0.0,
                "crystalizer": 0.0,
                "deesser": 0.3,
                "stereo_width": 1.0,
                "air_freq": 10000.0,
                "air_gain": 0.0,
                "air_width": 1.0,
            }
        
        log_info(f"[POSTPROCESS] Enhancing audio before transcription...")
        processed_path = run_postprocess(audio_path, params)
        log_info(f"[POSTPROCESS] Audio enhanced: {processed_path}")
        return processed_path
    except Exception as e:
        log_warn(f"Audio post-processing failed: {e}")
        return audio_path


# ==========================================
# STREAMING TRANSCRIPTION ENDPOINT
# ==========================================

_streaming_queues = {}


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    clean_vocals: bool = Form(default=False),
    skip_existing_vocals: bool = Form(default=True),
    postprocess_audio: bool = Form(default=False),
    device: str = Form(default="gpu"),
    model: Optional[str] = Form(default=None)
):
    """Streaming transcription endpoint using Server-Sent Events."""
    
    if not FASTER_WHISPER_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"Whisper not available: {FASTER_WHISPER_ERROR}")
    
    # Determine model
    model_name = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
    if model_name.startswith("whisper-"):
        model_name = model_name.replace("whisper-", "")
    
    async def generate() -> AsyncGenerator[str, None]:
        temp_path = None
        wav_path = None
        
        try:
            # Save uploaded file
            content = await file.read()
            fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1])
            os.close(fd)
            with open(temp_path, "wb") as f:
                f.write(content)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing audio...', 'progress': 2})}\n\n"
            
            # Optional vocal cleaning
            audio_to_process = temp_path
            original_filename = file.filename or "audio"
            if clean_vocals:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Cleaning vocals...', 'progress': 5})}\n\n"
                vocals_path = _clean_vocals(temp_path, skip_if_exists=skip_existing_vocals, original_filename=original_filename)
                if vocals_path != temp_path:
                    audio_to_process = vocals_path
            
            # Optional post-processing
            if postprocess_audio:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Enhancing audio...', 'progress': 8})}\n\n"
                processed_path = _postprocess_audio(audio_to_process)
                if processed_path != audio_to_process:
                    audio_to_process = processed_path
            
            # Convert to mono WAV
            wav_path = temp_path + "_mono.wav"
            data, sr = sf.read(audio_to_process)
            
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            
            if sr != 16000:
                import torchaudio.functional as F
                data_tensor = torch.from_numpy(data).float()
                data_tensor = F.resample(data_tensor, sr, 16000)
                data = data_tensor.numpy()
                sr = 16000
            
            sf.write(wav_path, data, sr)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Transcribing...', 'progress': 10})}\n\n"
            
            # Set up queue for updates
            update_queue = queue.Queue()
            session_id = id(update_queue)
            _streaming_queues[session_id] = update_queue
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            
            def run_transcription():
                try:
                    transcribe_with_whisper_streaming(wav_path, update_queue, model_name, language)
                except Exception as e:
                    log_error(f"Transcription error: {e}")
                    traceback.print_exc()
                    update_queue.put({"type": "error", "error": str(e)})
            
            future = loop.run_in_executor(_cuda_executor, run_transcription)
            
            # Stream updates
            while True:
                try:
                    update = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: update_queue.get(timeout=0.1)),
                        timeout=120.0
                    )
                    
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    if update.get("type") in ("complete", "error"):
                        break
                        
                except asyncio.TimeoutError:
                    if future.done():
                        break
                except queue.Empty:
                    continue
            
            await future
            
        except Exception as e:
            log_error(f"Stream error: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        finally:
            # Cleanup
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ==========================================
# MAIN TRANSCRIPTION ENDPOINT
# ==========================================

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    clean_vocals: bool = Form(default=False),
    skip_existing_vocals: bool = Form(default=True),
    postprocess_audio: bool = Form(default=False),
    device: str = Form(default="gpu"),
    model: Optional[str] = Form(default=None)
):
    """Transcribe audio using Whisper. Compatible with OpenAI API format."""
    
    if not FASTER_WHISPER_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"Whisper not available: {FASTER_WHISPER_ERROR}")
    
    # Determine model
    model_name = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
    if model_name.startswith("whisper-"):
        model_name = model_name.replace("whisper-", "")
    
    temp_path = None
    wav_path = None
    
    try:
        # Save uploaded file
        content = await file.read()
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1])
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        log_info(f"Transcribing: {file.filename}, model={model_name}, language={language}")
        
        # Optional vocal cleaning
        audio_to_process = temp_path
        original_filename = file.filename or "audio"
        if clean_vocals:
            log_info(f"Cleaning vocals for: {original_filename}")
            vocals_path = _clean_vocals(temp_path, skip_if_exists=skip_existing_vocals, original_filename=original_filename)
            if vocals_path != temp_path:
                audio_to_process = vocals_path
        
        # Optional post-processing
        if postprocess_audio:
            log_info(f"Enhancing audio before transcription...")
            processed_path = _postprocess_audio(audio_to_process)
            if processed_path != audio_to_process:
                audio_to_process = processed_path
        
        # Convert to mono WAV
        wav_path = temp_path + "_mono.wav"
        data, sr = sf.read(audio_to_process)
        
        log_info(f"Loaded audio: shape={data.shape}, dtype={data.dtype}, sr={sr}")
        
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        
        if sr != 16000:
            import torchaudio.functional as F
            data_tensor = torch.from_numpy(data).float()
            data_tensor = F.resample(data_tensor, sr, 16000)
            data = data_tensor.numpy()
            sr = 16000
        
        sf.write(wav_path, data, sr)
        
        # Transcribe
        t_start = time.perf_counter()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cuda_executor,
            lambda: transcribe_with_whisper(wav_path, model_name, language)
        )
        
        t_elapsed = time.perf_counter() - t_start
        log_info(f"Transcribed in {t_elapsed*1000:.0f}ms")
        
        text = result["text"]
        
        # Format response
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": result.get("language", language),
                "duration": result.get("duration"),
                "text": text,
                "segments": result.get("segments", []),
            }
        else:  # json
            return {"text": text}
    
    except Exception as e:
        log_error(f"Transcription error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


# ==========================================
# LIVE TRANSCRIPTION (WebSocket)
# ==========================================

class LiveTranscriptionSession:
    """Manages a live transcription session with audio buffering."""
    
    def __init__(self, model: str = None, language: str = "en"):
        self.model = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
        self.language = language
        self.audio_buffer = []
        self.sample_rate = 16000
        self.buffer_duration = 3.0
        self.overlap_duration = 0.5
        self.min_buffer_duration = 1.0
        self.full_transcript = []
        self.last_partial = ""
        self.is_processing = False
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        with self.lock:
            self.audio_buffer.extend(audio_data.tolist())
    
    def get_buffer_duration(self) -> float:
        return len(self.audio_buffer) / self.sample_rate
    
    def should_process(self) -> bool:
        return self.get_buffer_duration() >= self.buffer_duration and not self.is_processing
    
    def get_audio_chunk(self) -> np.ndarray:
        with self.lock:
            if len(self.audio_buffer) == 0:
                return None
            
            audio = np.array(self.audio_buffer, dtype=np.float32)
            overlap_samples = int(self.overlap_duration * self.sample_rate)
            
            if len(audio) > overlap_samples:
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
            else:
                self.audio_buffer = []
            
            return audio
    
    def get_final_audio(self) -> np.ndarray:
        with self.lock:
            if len(self.audio_buffer) == 0:
                return None
            audio = np.array(self.audio_buffer, dtype=np.float32)
            self.audio_buffer = []
            return audio


def _transcribe_audio_chunk_sync(audio: np.ndarray, sample_rate: int, model: str, language: str) -> str:
    """Transcribe an audio chunk using Whisper (sync version for thread pool)."""
    if not FASTER_WHISPER_AVAILABLE:
        return ""
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, sample_rate)
    
    try:
        model_name = model.replace("whisper-", "") if model.startswith("whisper-") else model
        whisper_model = load_whisper_model(model_name)
        
        segments_iter, info = whisper_model.transcribe(
            tmp_path,
            language=language if language != "auto" else None,
            beam_size=3,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        
        texts = []
        for segment in segments_iter:
            texts.append(segment.text.strip())
        return " ".join(texts)
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.websocket("/v1/audio/transcriptions/live")
async def live_transcription(
    websocket: WebSocket,
    model: str = "whisper-large-v3-turbo",
    language: str = "en"
):
    """WebSocket endpoint for live/real-time transcription."""
    await websocket.accept()
    
    if not FASTER_WHISPER_AVAILABLE:
        await websocket.send_json({"type": "error", "message": "Whisper not available"})
        await websocket.close()
        return
    
    session = LiveTranscriptionSession(model=model, language=language)
    
    log_info(f"[LIVE] New session: model={model}, language={language}")
    await websocket.send_json({"type": "ready"})
    
    processing_task = None
    stop_processing = asyncio.Event()
    
    async def process_audio_loop():
        """Background loop to process buffered audio."""
        while not stop_processing.is_set():
            try:
                if session.should_process():
                    session.is_processing = True
                    audio = session.get_audio_chunk()
                    
                    if audio is not None and len(audio) > session.sample_rate * session.min_buffer_duration:
                        log_info(f"[LIVE] Processing {len(audio)/session.sample_rate:.1f}s audio")
                        
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(
                            _cuda_executor,
                            lambda a=audio, s=session: _transcribe_audio_chunk_sync(
                                a, s.sample_rate, s.model, s.language
                            )
                        )
                        
                        if text:
                            session.full_transcript.append(text)
                            full_text = " ".join(session.full_transcript)
                            
                            await websocket.send_json({
                                "type": "final",
                                "text": text
                            })
                            await websocket.send_json({
                                "type": "transcript", 
                                "text": full_text
                            })
                            log_info(f"[LIVE] Transcribed: {text[:50]}...")
                    
                    session.is_processing = False
                
                await asyncio.sleep(0.1)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                log_error(f"[LIVE] Processing error: {e}")
                session.is_processing = False
                await asyncio.sleep(0.5)
    
    processing_task = asyncio.create_task(process_audio_loop())
    
    try:
        while True:
            try:
                message = await websocket.receive_json()
                msg_type = message.get("type", "")
                
                if msg_type == "audio":
                    audio_b64 = message.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                        session.add_audio(audio_data)
                
                elif msg_type == "config":
                    if "sample_rate" in message:
                        session.sample_rate = message["sample_rate"]
                    if "language" in message:
                        session.language = message["language"]
                    if "model" in message:
                        session.model = message["model"]
                    await websocket.send_json({"type": "config_updated"})
                
                elif msg_type == "end":
                    stop_processing.set()
                    if processing_task:
                        processing_task.cancel()
                    
                    audio = session.get_final_audio()
                    if audio is not None and len(audio) > session.sample_rate * 0.5:
                        log_info(f"[LIVE] Processing final {len(audio)/session.sample_rate:.1f}s audio")
                        
                        text = await asyncio.get_event_loop().run_in_executor(
                            _cuda_executor,
                            lambda a=audio, s=session: _transcribe_audio_chunk_sync(
                                a, s.sample_rate, s.model, s.language
                            )
                        )
                        if text:
                            session.full_transcript.append(text)
                    
                    full_text = " ".join(session.full_transcript)
                    await websocket.send_json({
                        "type": "complete",
                        "text": full_text
                    })
                    log_info(f"[LIVE] Session complete: {len(full_text)} chars")
                    break
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except WebSocketDisconnect:
                log_info("[LIVE] Client disconnected")
                break
                
    except Exception as e:
        log_error(f"[LIVE] WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    
    finally:
        stop_processing.set()
        if processing_task:
            processing_task.cancel()
        log_info("[LIVE] Session ended")


# ==========================================
# STARTUP
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup if configured."""
    if os.getenv("WHISPERASR_PRELOAD", os.getenv("ASR_PRELOAD", "false")).lower() == "true":
        try:
            load_whisper_model()
        except Exception as e:
            log_error(f"Failed to pre-load model: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Whisper ASR Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8889, help="Port to bind to")
    args = parser.parse_args()
    
    log_info(f"Starting Whisper ASR server on {args.host}:{args.port}")
    log_info(f"Whisper available: {FASTER_WHISPER_AVAILABLE}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
