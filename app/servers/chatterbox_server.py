# Chatterbox-Turbo TTS FastAPI Server
# https://huggingface.co/ResembleAI/chatterbox-turbo
# Copyright (c) 2025

import os
import sys
import asyncio

# Add app directory to path for imports
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Disable HF token requirement for non-gated models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Patch huggingface_hub early to not require tokens for non-gated models
def _patch_huggingface_hub():
    """Monkey-patch huggingface_hub to not require tokens."""
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download as original_download
        from huggingface_hub import snapshot_download as original_snapshot
        
        def patched_hf_hub_download(*args, **kwargs):
            # Remove token=True, allow None or False
            if kwargs.get('token') is True:
                kwargs['token'] = None
            return original_download(*args, **kwargs)
        
        def patched_snapshot_download(*args, **kwargs):
            # Remove token=True, allow None or False
            if kwargs.get('token') is True:
                kwargs['token'] = None
            return original_snapshot(*args, **kwargs)
        
        huggingface_hub.hf_hub_download = patched_hf_hub_download
        huggingface_hub.snapshot_download = patched_snapshot_download
        
        # Also patch the file_download module directly
        try:
            from huggingface_hub import file_download
            file_download.hf_hub_download = patched_hf_hub_download
        except:
            pass
            
    except ImportError:
        pass

_patch_huggingface_hub()

import logging
import tempfile
import uuid
from typing import Optional
from pathlib import Path
import threading
import io

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Chunking is done client-side in tts_service.py - servers receive pre-chunked text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Chatterbox-Turbo TTS Server", version="1.0.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache
MODEL_CACHE = {
    "model": None,
    "loaded": False,
    "sample_rate": None,
}
_model_lock = threading.Lock()


def get_model():
    """Get or load the Chatterbox-Turbo model (lazy loading)."""
    global MODEL_CACHE
    
    if MODEL_CACHE["loaded"] and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"]
    
    with _model_lock:
        # Double-check after acquiring lock
        if MODEL_CACHE["loaded"] and MODEL_CACHE["model"] is not None:
            return MODEL_CACHE["model"]
        
        logger.info("Loading Chatterbox-Turbo model...")
        
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        
        # Try with token=False first, fall back to default
        try:
            model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, token=False)
        except TypeError:
            # If token param not supported, try without it
            model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        
        MODEL_CACHE["model"] = model
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["sample_rate"] = model.sr
        
        logger.info(f"Chatterbox-Turbo loaded on {DEVICE}, sample_rate={model.sr}")
        
        return model


@app.get("/health")
async def health():
    """Health check endpoint."""
    vram_info = {}
    if torch.cuda.is_available():
        vram_info = {
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        }
    
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": MODEL_CACHE["loaded"],
        "sample_rate": MODEL_CACHE["sample_rate"],
        "vram": vram_info,
        "features": {
            "paralinguistic_tags": True,
            "voice_cloning": True,
            "supported_tags": ["[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]", "[groan]", "[yawn]", "[clear throat]"]
        }
    }


@app.post("/warmup")
async def warmup():
    """Pre-load the Chatterbox model."""
    try:
        model = get_model()
        return {
            "status": "ok",
            "message": "Chatterbox-Turbo model loaded",
            "sample_rate": model.sr,
            "device": DEVICE
        }
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Unload Chatterbox-Turbo model to free GPU memory."""
    global MODEL_CACHE
    
    # Get memory before unload
    before_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    
    if not MODEL_CACHE["loaded"]:
        return {
            "success": True,
            "message": "No model loaded",
            "freed_gb": 0
        }
    
    with _model_lock:
        if MODEL_CACHE["model"] is not None:
            del MODEL_CACHE["model"]
        MODEL_CACHE["model"] = None
        MODEL_CACHE["loaded"] = False
        MODEL_CACHE["sample_rate"] = None
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get memory after unload
    after_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    freed = before_reserved - after_reserved
    
    logger.info(f"Chatterbox-Turbo model unloaded, freed {freed:.2f}GB VRAM")
    
    return {
        "success": True,
        "message": "Model unloaded",
        "freed_gb": round(freed, 2)
    }


@app.get("/model_info")
async def get_model_info():
    """Get information about the currently loaded model."""
    # Estimated model size for Chatterbox-Turbo (350M params)
    ESTIMATED_SIZE_GB = 2.0
    
    actual_vram_gb = 0
    if torch.cuda.is_available():
        actual_vram_gb = torch.cuda.memory_allocated() / 1e9
    
    return {
        "model_id": "chatterbox-turbo",
        "model_name": "Chatterbox-Turbo",
        "model_size": "350M parameters",
        "loaded": MODEL_CACHE["loaded"],
        "sample_rate": MODEL_CACHE["sample_rate"],
        "estimated_size_gb": ESTIMATED_SIZE_GB,
        "actual_vram_gb": round(actual_vram_gb, 2),
        "device": DEVICE,
        "features": [
            "Zero-shot voice cloning",
            "Paralinguistic tags ([laugh], [chuckle], etc.)",
            "Low latency (optimized for voice agents)",
            "Single-step mel decoding"
        ]
    }


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = props.total_memory / 1e9
    free = total - reserved
    
    return {
        "available": True,
        "device": torch.cuda.get_device_name(),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "usage_percent": round((reserved / total) * 100, 1)
    }


# ============================================================================
# TEXT SPLITTING UTILITIES
# ============================================================================

import re

# Removed split_text_into_sentences() and split_text_by_tokens() - chunking is done client-side in tts_service.py


@app.post("/v1/tts/chunked")
async def generate_tts_chunked(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    seed: int = Form(0),
    max_tokens: int = Form(200),
    request_id: str = Form(None),  # Accept request_id from client for unified tracking
):
    """
    Chunked TTS - splits long text into sentences/chunks, generates each, and concatenates.
    
    Ideal for long-form content like scripts or articles.
    
    - text: Text to synthesize (will be split into chunks)
    - prompt_audio: Reference audio file for voice cloning (5+ seconds required, 10+ recommended)
    - seed: Random seed for reproducibility (0 = random)
    - max_tokens: Approximate max tokens per chunk (default 200)
    - request_id: Optional request ID for unified logging across services
    
    Note: Chatterbox-Turbo doesn't support exaggeration/cfg_weight controls.
    """
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    try:
        import time
        t_start = time.perf_counter()
        
        # Save uploaded audio to temp file
        temp_prompt = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_prompt.write(await prompt_audio.read())
        temp_prompt.close()
        prompt_path = temp_prompt.name
        
        # Check audio duration
        try:
            waveform, sr = ta.load(prompt_path)
            duration = waveform.shape[1] / sr
            if duration < 5.0:
                os.unlink(prompt_path)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Prompt audio is too short ({duration:.1f}s). Chatterbox requires at least 5 seconds."
                )
            logger.info(f"[{request_id}] Prompt audio: {duration:.1f}s @ {sr}Hz")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[{request_id}] Could not verify prompt duration: {e}")
        
        # Chunking is done client-side in tts_service.py - this endpoint receives pre-chunked text
        # Just treat the text as a single chunk (chunking already happened upstream)
        chunks = [text]
        logger.info(f"[{request_id}] Chunked TTS: {len(chunks)} chunks from {len(text)} chars (max_tokens={max_tokens})")
        
        if not chunks:
            os.unlink(prompt_path)
            raise HTTPException(status_code=400, detail="No valid text to synthesize")
        
        # Get model
        model = get_model()
        
        # Run generation in executor to not block the async event loop
        # This allows other requests to be received while GPU is busy
        def do_generate():
            """Blocking generation - runs in thread pool executor."""
            # Generate each chunk - pass audio_prompt_path directly to generate() (official API)
            all_audio = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{request_id}-{i+1}"
                logger.info(f"[{chunk_id}] Chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
                
                t_chunk_start = time.perf_counter()
                
                # Set seed (increment for each chunk for variety while maintaining reproducibility)
                # -1 = random each time, 0 = no seeding, >0 = specific seed
                if seed == -1:
                    chunk_seed = torch.randint(0, 2**31, (1,)).item() + i
                elif seed > 0:
                    chunk_seed = seed + i
                else:
                    chunk_seed = 0
                
                if chunk_seed > 0:
                    torch.manual_seed(chunk_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(chunk_seed)
                
                # Generate with audio_prompt_path directly (official Chatterbox API)
                with torch.inference_mode():
                    wav = model.generate(chunk, audio_prompt_path=prompt_path)
                
                all_audio.append(wav)
                
                t_chunk_end = time.perf_counter()
                chunk_duration = wav.shape[1] / model.sr
                chunk_time = t_chunk_end - t_chunk_start
                logger.info(f"[{chunk_id}] Done: {chunk_duration:.1f}s audio in {chunk_time:.1f}s")
            
            return all_audio
        
        # Run in executor so we don't block the event loop
        loop = asyncio.get_event_loop()
        all_audio = await loop.run_in_executor(None, do_generate)
        logger.info(f"[{request_id}] Generation complete")
        
        # Concatenate all audio (outside the lock - we have our data)
        combined = torch.cat(all_audio, dim=1)
        
        # Save output
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        ta.save(output_path, combined, model.sr)
        
        # Cleanup
        try:
            os.unlink(prompt_path)
        except:
            pass
        
        # Log timing
        total_duration = combined.shape[1] / model.sr
        total_time = time.perf_counter() - t_start
        rtf = total_time / total_duration if total_duration > 0 else 0
        
        logger.info(f"[{request_id}] Chunked TTS complete: {total_duration:.1f}s audio in {total_time:.1f}s (RTF: {rtf:.2f}x, chunks: {len(chunks)})")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Chunked TTS failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/stream")
async def generate_tts_stream(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    seed: int = Form(0),
    max_tokens: int = Form(200),
    request_id: str = Form(None),  # Accept request_id from client for unified tracking
):
    """
    Streaming TTS - streams audio chunks as they are generated via SSE.
    
    Each chunk is generated and sent immediately, allowing playback to start
    before the full audio is complete. Ideal for real-time/interactive use.
    
    - text: Text to synthesize (will be split into chunks and streamed)
    - prompt_audio: Reference audio file for voice cloning (5+ seconds required, 10+ recommended)
    - seed: Random seed for reproducibility (0 = random)
    - max_tokens: Approximate max tokens per chunk (default 200)
    - request_id: Optional request ID for unified logging across services
    
    Returns: SSE stream with base64-encoded WAV chunks
    """
    import base64
    import json
    import time
    
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Save uploaded audio to temp file first (before generator)
    temp_prompt = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_prompt.write(await prompt_audio.read())
    temp_prompt.close()
    prompt_path = temp_prompt.name
    
    # Check audio duration
    try:
        waveform, sr = ta.load(prompt_path)
        duration = waveform.shape[1] / sr
        if duration < 5.0:
            os.unlink(prompt_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Prompt audio is too short ({duration:.1f}s). Chatterbox requires at least 5 seconds."
            )
        logger.info(f"[{request_id}] Stream: Prompt audio: {duration:.1f}s @ {sr}Hz")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[{request_id}] Could not verify prompt duration: {e}")
    
    # Split text into chunks for streaming
    from util.text_utils import split_text
    chunks = split_text(text, max_tokens=max_tokens, token_method="tiktoken")
    
    if not chunks:
        os.unlink(prompt_path)
        raise HTTPException(status_code=400, detail="No valid text to synthesize")
    
    logger.info(f"[{request_id}] Streaming TTS: {len(chunks)} chunks from {len(text)} chars")
    
    async def generate_stream():
        """Generator that yields SSE events with audio chunks."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        t_start = time.perf_counter()
        model = get_model()
        sample_rate = model.sr
        
        # Use executor to avoid blocking the async event loop during GPU work
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_gen")
        
        try:
            # Send initial event with metadata
            yield f"data: {json.dumps({'type': 'start', 'chunks': len(chunks), 'sample_rate': sample_rate})}\n\n"
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{request_id}-{i+1}"
                logger.info(f"[{chunk_id}] Streaming chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
                
                t_chunk_start = time.perf_counter()
                
                # Set seed (-1 = random each time, 0 = no seeding, >0 = specific seed)
                if seed == -1:
                    chunk_seed = torch.randint(0, 2**31, (1,)).item() + i
                elif seed > 0:
                    chunk_seed = seed + i
                else:
                    chunk_seed = 0
                
                def do_generate(text, prompt, seed_val):
                    """Blocking TTS generation - runs in thread pool."""
                    if seed_val > 0:
                        torch.manual_seed(seed_val)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(seed_val)
                    with torch.inference_mode():
                        return model.generate(text, audio_prompt_path=prompt)
                
                # Run in executor to not block event loop
                # Parallel GPU access has contention but smaller gaps between chunks
                loop = asyncio.get_event_loop()
                wav = await loop.run_in_executor(executor, do_generate, chunk, prompt_path, chunk_seed)
                
                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
                ta.save(wav_buffer, wav, sample_rate, format="wav")
                wav_bytes = wav_buffer.getvalue()
                wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                
                chunk_duration = wav.shape[1] / sample_rate
                chunk_time = time.perf_counter() - t_chunk_start
                
                logger.info(f"[{chunk_id}] Done: {chunk_duration:.1f}s audio in {chunk_time:.1f}s")
                
                # Send chunk event
                yield f"data: {json.dumps({'type': 'chunk', 'index': i, 'total': len(chunks), 'audio': wav_b64, 'duration': round(chunk_duration, 2), 'text': chunk[:100]})}\n\n"
            
            # Send completion event
            total_time = time.perf_counter() - t_start
            logger.info(f"[{request_id}] Stream complete in {total_time:.1f}s")
            yield f"data: {json.dumps({'type': 'complete', 'total_time': round(total_time, 2)})}\n\n"
            
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Cleanup
            executor.shutdown(wait=False)
            try:
                os.unlink(prompt_path)
            except:
                pass
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8893, help="Port to bind to")
    parser.add_argument("--warmup", action="store_true", help="Load model on startup")
    args = parser.parse_args()
    
    logger.info(f"Starting Chatterbox-Turbo TTS server on {args.host}:{args.port}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total VRAM: {total_vram:.1f}GB")
    
    if args.warmup:
        logger.info("Pre-loading model...")
        get_model()
    
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=3600)

