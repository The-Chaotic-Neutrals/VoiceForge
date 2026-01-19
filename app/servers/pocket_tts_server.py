from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
from pocket_tts import TTSModel
import scipy.io.wavfile
import scipy.signal
import io
import numpy as np
import torch
import logging
from contextlib import asynccontextmanager
import json
import os
import time
import hashlib
import tempfile
import subprocess
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Built-in voices available without voice cloning
BUILTIN_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

# Voice paths for cloning model (only used if cloning is available)
VOICE_PATHS = {
    "alba": "alba-mackenna/casual.wav",
    "marius": "voice-donations/Selfie.wav",
    "javert": "voice-donations/Butter.wav",
    "jean": "ears/p010/freeform_speech_01.wav",
    "fantine": "vctk/p244_023.wav",
    "cosette": "expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "eponine": "vctk/p262_023.wav",
    "azelma": "vctk/p303_023.wav",
}

tts_model: Optional[TTSModel] = None
voice_states: Dict[str, Any] = {}
voice_cloning_enabled = False

# Cache for cloned voice states derived from prompts
cloned_voice_states: Dict[str, Dict[str, Any]] = {}
CLONE_CACHE_MAX = 128
CLONE_CACHE_TTL_SECONDS = 24 * 3600  # 24h


def _cleanup_clone_cache():
    now = time.time()

    # TTL cleanup
    expired_keys = [
        k for k, v in cloned_voice_states.items()
        if (now - v["created_at"]) > CLONE_CACHE_TTL_SECONDS
    ]
    for k in expired_keys:
        cloned_voice_states.pop(k, None)

    # Size bound cleanup (drop oldest)
    if len(cloned_voice_states) > CLONE_CACHE_MAX:
        items = sorted(cloned_voice_states.items(), key=lambda kv: kv[1]["created_at"])
        overflow = len(items) - CLONE_CACHE_MAX
        for k, _ in items[:overflow]:
            cloned_voice_states.pop(k, None)


def _normalize_prompt_location(loc: str) -> str:
    loc = (loc or "").strip()
    if loc.startswith("file://"):
        loc = loc[len("file://"):]
    return loc


def _looks_remote(loc: str) -> bool:
    return loc.startswith("hf://") or loc.startswith("http://") or loc.startswith("https://")


def _normalize_prompt_audio_for_cloning(prompt_loc: str, target_sr: int) -> str:
    """Normalize prompt audio for cloning (mono, target sample rate)."""
    prompt_loc = (prompt_loc or "").strip()
    if not prompt_loc or _looks_remote(prompt_loc):
        return prompt_loc

    if not os.path.exists(prompt_loc) or not os.path.isfile(prompt_loc):
        return prompt_loc

    try:
        st = os.stat(prompt_loc)
        key_src = f"{prompt_loc}|{st.st_mtime_ns}|{target_sr}".encode("utf-8")
    except Exception:
        key_src = f"{prompt_loc}|{target_sr}".encode("utf-8")

    key = hashlib.sha256(key_src).hexdigest()[:16]
    out_dir = os.path.join(tempfile.gettempdir(), "pocket_tts_prompt_cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"prompt_{key}_{target_sr}hz_mono.wav")

    if os.path.exists(out_path):
        return out_path

    # Prefer ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", prompt_loc, "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s16le", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return out_path
    except Exception:
        pass

    # Fallback: SciPy for WAV
    try:
        sr_in, x = scipy.io.wavfile.read(prompt_loc)
        if isinstance(x, np.ndarray) and x.ndim == 2:
            x = x.mean(axis=1)
        if x.dtype == np.int16:
            xf = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            xf = x.astype(np.float32) / 2147483648.0
        else:
            xf = x.astype(np.float32)
        if sr_in != target_sr:
            xf = scipy.signal.resample_poly(xf, target_sr, sr_in)
        y = (np.clip(xf, -1.0, 1.0) * 32767.0).astype(np.int16)
        scipy.io.wavfile.write(out_path, target_sr, y)
        return out_path
    except Exception:
        return prompt_loc


def _get_voice_state(voice_value: str):
    """
    Get voice state for generation.
    - If voice cloning enabled and voice is in voice_states, use pre-loaded state
    - If voice cloning enabled and voice is a path, clone from it
    - If voice cloning disabled, return None (will use built-in voice name directly)
    """
    global voice_cloning_enabled
    
    if not tts_model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    voice_value = (voice_value or "").strip()
    if not voice_value:
        voice_value = "alba"  # Default voice

    # If voice cloning is not enabled, return None - will use voice name directly
    if not voice_cloning_enabled:
        if voice_value not in BUILTIN_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Voice cloning is not available. Please use one of the built-in voices: {', '.join(BUILTIN_VOICES)}"
            )
        return None

    # Voice cloning is enabled - check pre-loaded voices first
    if voice_value in voice_states:
        return voice_states[voice_value]

    # Try to clone from path
    _cleanup_clone_cache()
    prompt_loc = _normalize_prompt_location(voice_value)

    if not _looks_remote(prompt_loc):
        if not os.path.exists(prompt_loc):
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_value}' not found as a built-in voice, and file does not exist: {prompt_loc}"
            )
        if not os.path.isfile(prompt_loc):
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_value}' is not a file: {prompt_loc}"
            )

    cache_key = f"prompt:{prompt_loc}"
    cached = cloned_voice_states.get(cache_key)
    if cached is not None:
        return cached["state"]

    target_sr = getattr(tts_model, "sample_rate", None) or 24000
    norm_loc = _normalize_prompt_audio_for_cloning(prompt_loc, target_sr)

    logger.info(f"Cloning voice from: {prompt_loc}")
    try:
        state = tts_model.get_state_for_audio_prompt(norm_loc)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone voice from '{norm_loc}': {str(e)}"
        )

    cloned_voice_states[cache_key] = {"state": state, "created_at": time.time()}
    _cleanup_clone_cache()
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, voice_states, voice_cloning_enabled

    logger.info("Loading TTS model...")
    try:
        tts_model = TTSModel.load_model()
        logger.info("Model loaded successfully!")

        # Try to load voice states (requires voice cloning model)
        try:
            logger.info("Checking voice cloning availability...")
            test_voice_url = f"hf://kyutai/tts-voices/{VOICE_PATHS['alba']}"
            test_state = tts_model.get_state_for_audio_prompt(test_voice_url)
            voice_states["alba"] = test_state
            voice_cloning_enabled = True
            logger.info("Voice cloning is ENABLED")
            
            # Load remaining voices
            for voice_name, voice_path in VOICE_PATHS.items():
                if voice_name == "alba":
                    continue  # Already loaded
                logger.info(f"Loading voice: {voice_name}...")
                voice_url = f"hf://kyutai/tts-voices/{voice_path}"
                voice_states[voice_name] = tts_model.get_state_for_audio_prompt(voice_url)
                logger.info(f"Voice '{voice_name}' ready")
            
            logger.info("All voices loaded with cloning support!")
        except Exception as e:
            logger.warning(f"Voice cloning not available: {e}")
            logger.info("Running in NON-CLONING mode - using built-in voices only")
            voice_cloning_enabled = False
            voice_states = {}

        logger.info(f"Server ready! Voice cloning: {'enabled' if voice_cloning_enabled else 'disabled'}")
        logger.info(f"Available voices: {', '.join(BUILTIN_VOICES)}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Pocket TTS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "model": "pocket-tts",
                "input": "Hello, world!",
                "voice": "alba",
                "response_format": "wav",
                "speed": 1.0,
            }
        }
    )
    
    model: str = Field(default="pocket-tts", description="Model to use")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="alba", description="Voice to use (or prompt location for cloning)")
    response_format: str = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of speech")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}")
    logger.error(f"Validation errors: {json.dumps(exc.errors(), indent=2)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            },
            "detail": exc.errors(),
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": tts_model is not None,
        "voice_cloning_enabled": voice_cloning_enabled,
        "available_voices": BUILTIN_VOICES,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "pocket-tts",
                "object": "model",
                "created": 0,
                "owned_by": "pocket-tts",
                "voice_cloning": voice_cloning_enabled,
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": BUILTIN_VOICES,
        "voice_cloning_enabled": voice_cloning_enabled,
    }


def _split_text_into_sentences(text: str) -> list:
    """Split text into sentences for progress tracking.
    
    Uses a simple approach that splits on sentence-ending punctuation while
    keeping reasonable chunk sizes.
    """
    import re
    # Split on sentence boundaries (., !, ?) followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and merge very short sentences
    result = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) < 100:  # Merge short sentences
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                result.append(current)
            current = s
    if current:
        result.append(current)
    return result if result else [text]


def _generate_audio_for_text(text: str, voice_state) -> np.ndarray:
    """Generate audio for a single text segment using pre-loaded voice state."""
    audio_chunks = []
    
    stream = tts_model.generate_audio_stream(voice_state, text, copy_state=True)
    
    for chunk in stream:
        if isinstance(chunk, torch.Tensor):
            chunk_np = chunk.detach().cpu().numpy()
        else:
            chunk_np = np.array(chunk)
        
        if chunk_np.dtype != np.int16:
            chunk_np = (np.clip(chunk_np, -1.0, 1.0) * 32767.0).astype(np.int16)
        
        audio_chunks.append(chunk_np)
    
    if audio_chunks:
        return np.concatenate(audio_chunks)
    return np.array([], dtype=np.int16)


@app.post("/v1/audio/speech/stream")
async def create_speech_streaming(request: AudioSpeechRequest):
    """Generate speech with SSE progress updates.
    
    Returns Server-Sent Events with:
    - type: "progress" - progress updates with sentence info
    - type: "audio" - base64-encoded WAV audio data
    - type: "complete" - generation finished
    - type: "error" - error occurred
    """
    import base64
    import copy
    
    logger.info(
        f"Processing streaming speech request - voice: {request.voice}, "
        f"input length: {len(request.input) if request.input else 0}, "
        f"cloning: {voice_cloning_enabled}"
    )

    if not tts_model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    async def event_generator():
        try:
            voice_state = _get_voice_state(request.voice)
            voice_name = request.voice if request.voice in BUILTIN_VOICES else "alba"
            sample_rate = tts_model.sample_rate
            
            # Split text into sentences for progress tracking
            sentences = _split_text_into_sentences(request.input)
            total_sentences = len(sentences)
            
            yield f"data: {json.dumps({'type': 'start', 'total_sentences': total_sentences, 'total_chars': len(request.input)})}\n\n"
            
            # Get voice state ONCE before the loop
            if voice_state is None:
                voice_state = tts_model.get_state_for_audio_prompt(voice_name)
            
            all_audio = []
            total_samples = 0
            start_time = time.time()
            
            for i, sentence in enumerate(sentences):
                sentence_start = time.time()
                
                # Progress update
                progress = (i / total_sentences) * 100
                yield f"data: {json.dumps({'type': 'progress', 'sentence': i + 1, 'total': total_sentences, 'progress': round(progress, 1), 'text_preview': sentence[:50] + '...' if len(sentence) > 50 else sentence})}\n\n"
                
                # Generate audio for this sentence
                try:
                    audio_np = _generate_audio_for_text(sentence, voice_state)
                    if len(audio_np) > 0:
                        all_audio.append(audio_np)
                        total_samples += len(audio_np)
                        
                        # Calculate timing
                        sentence_time = time.time() - sentence_start
                        audio_duration = len(audio_np) / sample_rate
                        rtf = audio_duration / sentence_time if sentence_time > 0 else 0
                        
                        yield f"data: {json.dumps({'type': 'sentence_complete', 'sentence': i + 1, 'audio_duration': round(audio_duration, 2), 'generation_time': round(sentence_time, 2), 'rtf': round(rtf, 2)})}\n\n"
                except Exception as e:
                    logger.error(f"Error generating sentence {i + 1}: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Sentence {i + 1} failed: {str(e)}'})}\n\n"
            
            if not all_audio:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio generated'})}\n\n"
                return
            
            # Combine all audio
            combined_audio = np.concatenate(all_audio)
            
            # Resample to 48kHz
            target_rate = 48000
            if sample_rate != target_rate:
                x = combined_audio.astype(np.float32) / 32768.0
                x = scipy.signal.resample_poly(x, target_rate, sample_rate)
                combined_audio = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
                sample_rate = target_rate
            
            # Create WAV
            audio_buffer = io.BytesIO()
            scipy.io.wavfile.write(audio_buffer, sample_rate, combined_audio)
            audio_bytes = audio_buffer.getvalue()
            
            # Encode as base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            total_time = time.time() - start_time
            audio_duration = len(combined_audio) / sample_rate
            
            yield f"data: {json.dumps({'type': 'audio', 'format': 'wav', 'sample_rate': sample_rate, 'duration': round(audio_duration, 2), 'size_bytes': len(audio_bytes), 'data': audio_b64})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'total_time': round(total_time, 2), 'audio_duration': round(audio_duration, 2), 'rtf': round(audio_duration / total_time, 2) if total_time > 0 else 0})}\n\n"
            
            logger.info(f"Streaming generation complete: {audio_duration:.1f}s audio in {total_time:.1f}s ({audio_duration/total_time:.1f}x RT)")
            
        except HTTPException as e:
            yield f"data: {json.dumps({'type': 'error', 'message': e.detail})}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    logger.info(
        f"Processing speech request - voice: {request.voice}, "
        f"input length: {len(request.input) if request.input else 0}, "
        f"cloning: {voice_cloning_enabled}"
    )

    if not tts_model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    # Force wav format
    request.response_format = "wav"

    try:
        voice_state = _get_voice_state(request.voice)
        voice_name = request.voice if request.voice in BUILTIN_VOICES else "alba"
        sample_rate = tts_model.sample_rate

        # Split into sentences for progress logging
        sentences = _split_text_into_sentences(request.input)
        total_sentences = len(sentences)
        
        logger.info(f"Generating {total_sentences} sentences...")
        
        # Get voice state ONCE before the loop (avoids re-downloading embeddings for each sentence)
        if voice_state is None:
            logger.info(f"Loading voice state for '{voice_name}' (one-time)...")
            voice_state = tts_model.get_state_for_audio_prompt(voice_name)
            logger.info(f"Voice state loaded!")
        
        all_audio = []
        start_time = time.time()
        
        for i, sentence in enumerate(sentences):
            sentence_start = time.time()
            logger.info(f"[{i+1}/{total_sentences}] Generating: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
            
            audio_np = _generate_audio_for_text(sentence, voice_state)
            if len(audio_np) > 0:
                all_audio.append(audio_np)
                sentence_time = time.time() - sentence_start
                audio_duration = len(audio_np) / sample_rate
                logger.info(f"[{i+1}/{total_sentences}] Generated {audio_duration:.1f}s audio in {sentence_time:.1f}s ({audio_duration/sentence_time:.1f}x RT)")

        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio chunks generated")

        audio_np = np.concatenate(all_audio)

        # Resample to 48kHz for compatibility
        target_rate = 48000
        if sample_rate != target_rate:
            x = audio_np.astype(np.float32) / 32768.0
            x = scipy.signal.resample_poly(x, target_rate, sample_rate)
            audio_np = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
            sample_rate = target_rate

        audio_buffer = io.BytesIO()
        scipy.io.wavfile.write(audio_buffer, sample_rate, audio_np)
        audio_bytes = audio_buffer.getvalue()

        total_time = time.time() - start_time
        audio_duration = len(audio_np) / sample_rate
        logger.info(f"Total: {audio_duration:.1f}s audio in {total_time:.1f}s ({audio_duration/total_time:.1f}x RT), {len(audio_bytes)} bytes")

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Pocket TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8894, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting Pocket TTS Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
