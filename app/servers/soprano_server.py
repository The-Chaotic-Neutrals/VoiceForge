# Soprano TTS FastAPI Server
# Uses soprano-tts module: https://github.com/ekwek1/soprano
# Model: https://huggingface.co/ekwek/Soprano-1.1-80M

import os
import sys
import io
import base64
import json
import tempfile
import threading
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import numpy as np

# Add app directory to path for imports
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Model cache directory
_MODEL_CACHE_DIR = os.path.join(_APP_DIR, "models", "soprano", "hub")
os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _MODEL_CACHE_DIR)

# Custom models directory
_CUSTOM_MODELS_DIR = os.path.join(_APP_DIR, "models", "soprano_custom")
os.makedirs(_CUSTOM_MODELS_DIR, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Soprano TTS Server", version="1.0.0")

# Enable CORS for UI model switching
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server configuration from environment
SOPRANO_DEVICE = os.getenv("SOPRANO_DEVICE", "auto")
SOPRANO_BACKEND = os.getenv("SOPRANO_BACKEND", "auto")
SOPRANO_CACHE_MB = int(os.getenv("SOPRANO_CACHE_SIZE_MB", "100"))
SOPRANO_DECODER_BATCH = int(os.getenv("SOPRANO_DECODER_BATCH_SIZE", "1"))

# Soprano outputs at 32kHz
SOPRANO_SAMPLE_RATE = 32000


class SopranoTTSRequest(BaseModel):
    """Request model matching soprano-tts module parameters."""
    input: str
    temperature: Optional[float] = None  # Module default: 0.3
    top_p: Optional[float] = None        # Module default: 0.95
    repetition_penalty: Optional[float] = None  # Module default: 1.2
    chunk_size: Optional[int] = None     # Tokens per chunk for streaming


class SopranoModel:
    """Wrapper around soprano-tts SopranoTTS class."""
    
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.loaded = False
        self.sample_rate = SOPRANO_SAMPLE_RATE
        self.device = None
        self.backend = None
        self.model_path = None  # Path to custom model (None = default model)
        self.model_name = "default"  # Name of loaded model

    def set_model(self, model_path: str = None, model_name: str = None):
        """Set the model to load. If model_path is None, use default model."""
        with self.lock:
            if model_path != self.model_path:
                # Unload current model before changing
                if self.loaded:
                    logger.info(f"Unloading current model '{self.model_name}' to switch to '{model_name or 'default'}'")
                    self.model = None
                    self.loaded = False
                
                self.model_path = model_path
                self.model_name = model_name or "default"

    def load_model(self):
        """Load the SopranoTTS model using the module's built-in initialization."""
        if self.loaded and self.model is not None:
            return self.model

        with self.lock:
            if self.loaded and self.model is not None:
                return self.model

            logger.info("Loading Soprano TTS model...")
            
            # Import the soprano module
            from soprano import SopranoTTS
            from soprano import tts as soprano_tts_module
            from config import get_config_value
            
            # Resolve device
            device = SOPRANO_DEVICE
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            # Get user's saved settings from config for warmup
            # The soprano module's __init__ runs warmup with invalid temperature=0.0
            warmup_temp = get_config_value("soprano_temperature", 0.001)
            warmup_top_p = get_config_value("soprano_top_p", 0.95)
            warmup_rep = get_config_value("soprano_repetition_penalty", 1.2)
            
            # soprano-tts requires temp > 0, use tiny epsilon for temp=0 (greedy-like)
            if warmup_temp is None or warmup_temp <= 0:
                warmup_temp = 0.001
            
            logger.info(f"Using warmup params from config: temp={warmup_temp}, top_p={warmup_top_p}, rep={warmup_rep}")
            
            # Patch infer to use config values during warmup
            original_infer = soprano_tts_module.SopranoTTS.infer
            def patched_infer(self_inner, text, out_path=None, top_p=0.95, temperature=0.0, repetition_penalty=1.2):
                # Use user's config values for warmup (fixes invalid temperature=0.0)
                # soprano-tts requires temp > 0, use tiny epsilon for temp=0
                if temperature <= 0:
                    temperature = warmup_temp if warmup_temp > 0 else 0.001
                if top_p is None:
                    top_p = warmup_top_p
                if repetition_penalty is None:
                    repetition_penalty = warmup_rep
                return original_infer(self_inner, text, out_path, top_p, temperature, repetition_penalty)
            
            soprano_tts_module.SopranoTTS.infer = patched_infer
            
            try:
                # Build kwargs - add model_path if custom model is set
                model_kwargs = {
                    "backend": SOPRANO_BACKEND,
                    "device": device,
                    "cache_size_mb": SOPRANO_CACHE_MB,
                    "decoder_batch_size": SOPRANO_DECODER_BATCH,
                }
                
                if self.model_path:
                    # Check if SopranoTTS supports model_path parameter
                    import inspect
                    sig = inspect.signature(SopranoTTS.__init__)
                    if 'model_path' in sig.parameters:
                        model_kwargs["model_path"] = self.model_path
                        logger.info(f"Loading custom model from: {self.model_path}")
                    else:
                        logger.warning(f"SopranoTTS does not support model_path parameter. Using default model.")
                
                self.model = SopranoTTS(**model_kwargs)
            finally:
                # Restore original method - user params will pass through normally
                soprano_tts_module.SopranoTTS.infer = original_infer
            
            self.device = device
            self.backend = SOPRANO_BACKEND
            self.loaded = True
            
            # Try to get actual sample rate from the model
            actual_sr = self._get_model_sample_rate()
            logger.info(f"Model sample rate detection: detected={actual_sr}, current={self.sample_rate}")
            if actual_sr and actual_sr != self.sample_rate:
                logger.info(f"Updating sample rate from {self.sample_rate} to {actual_sr} (from model)")
                self.sample_rate = actual_sr
            
            # Also log available attributes for debugging
            if self.model:
                model_attrs = [a for a in dir(self.model) if not a.startswith('_') and not callable(getattr(self.model, a, None))]
                logger.info(f"Model attributes: {model_attrs[:20]}")  # First 20 to avoid spam
            
            if self.model_path:
                logger.info(f"Soprano loaded CUSTOM model '{self.model_name}': device={device}, backend={SOPRANO_BACKEND}, "
                           f"cache={SOPRANO_CACHE_MB}MB, batch={SOPRANO_DECODER_BATCH}, sr={self.sample_rate}")
            else:
                logger.info(f"Soprano loaded DEFAULT model: device={device}, backend={SOPRANO_BACKEND}, "
                           f"cache={SOPRANO_CACHE_MB}MB, batch={SOPRANO_DECODER_BATCH}, sr={self.sample_rate}")
            
            return self.model
    
    def _get_model_sample_rate(self) -> int:
        """Try to get the actual sample rate from the loaded model."""
        if self.model is None:
            return SOPRANO_SAMPLE_RATE
        
        # Try common attribute names
        for attr in ("sample_rate", "sr", "sampling_rate", "audio_sr"):
            if hasattr(self.model, attr):
                try:
                    sr = getattr(self.model, attr)
                    if callable(sr):
                        sr = sr()
                    sr = int(sr)
                    if 8000 <= sr <= 96000:  # Sanity check
                        logger.info(f"Found model sample rate: {attr}={sr}")
                        return sr
                except Exception as e:
                    logger.debug(f"Failed to get {attr}: {e}")
        
        # Try pipeline/decoder attributes
        if hasattr(self.model, 'pipeline'):
            pipeline = self.model.pipeline
            for attr in ("sample_rate", "sr", "sampling_rate"):
                if hasattr(pipeline, attr):
                    try:
                        sr = int(getattr(pipeline, attr))
                        if 8000 <= sr <= 96000:
                            logger.info(f"Found pipeline sample rate: {attr}={sr}")
                            return sr
                    except:
                        pass
        
        # Try decoder attributes
        if hasattr(self.model, 'decoder'):
            decoder = self.model.decoder
            for attr in ("sample_rate", "sr", "sampling_rate"):
                if hasattr(decoder, attr):
                    try:
                        sr = int(getattr(decoder, attr))
                        if 8000 <= sr <= 96000:
                            logger.info(f"Found decoder sample rate: {attr}={sr}")
                            return sr
                    except:
                        pass
        
        logger.warning(f"Could not determine model sample rate, using default {SOPRANO_SAMPLE_RATE}")
        return SOPRANO_SAMPLE_RATE

    def generate(self, text: str, params: Dict[str, Any]) -> str:
        """Generate TTS audio using the module's infer() method."""
        with self.lock:
            model = self.load_model()
            
            # Create temp file for output
            fd, output_path = tempfile.mkstemp(suffix="_soprano.wav")
            os.close(fd)

            # Build kwargs - only include non-None values to use module defaults
            infer_kwargs = {}
            if params.get("temperature") is not None:
                temp = params["temperature"]
                # soprano-tts requires temp > 0, use tiny epsilon for temp=0 (greedy-like)
                infer_kwargs["temperature"] = max(temp, 0.001) if temp <= 0 else temp
            if params.get("top_p") is not None:
                infer_kwargs["top_p"] = params["top_p"]
            if params.get("repetition_penalty") is not None:
                infer_kwargs["repetition_penalty"] = params["repetition_penalty"]

            logger.info(f"Soprano generate: text={text[:50]}..., kwargs={infer_kwargs}")
            
            # Use module's infer method directly - it handles saving to file
            model.infer(text, output_path, **infer_kwargs)
            
            return output_path

    def supports_streaming(self) -> bool:
        """Check if the loaded model supports streaming."""
        if self.model is None:
            return False
        return hasattr(self.model, "infer_stream")

    def _reset_model(self):
        """Reset the model after a corrupted state (e.g., interrupted stream)."""
        logger.warning("Resetting corrupted model state...")
        with self.lock:
            if self.model is not None:
                try:
                    del self.model
                except:
                    pass
                self.model = None
                self.loaded = False
                
                # Clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
        
        # Reload the model
        return self.load_model()

    def stream(self, text: str, params: Dict[str, Any]):
        """
        Stream TTS audio using the module's infer_stream() method.
        
        Yields (audio_array, sample_rate) tuples for each chunk.
        """
        logger.info(f"stream() called with text: {text[:50]}...")
        model = self.load_model()
        logger.info(f"Model loaded, has infer_stream: {hasattr(model, 'infer_stream')}")

        # Chunk size for streaming (tokens per chunk)
        chunk_size = params.get("chunk_size", 1)
        
        # Build kwargs for streaming
        infer_kwargs = {}
        if params.get("temperature") is not None:
            temp = params["temperature"]
            # soprano-tts requires temp > 0, use tiny epsilon for temp=0 (greedy-like)
            infer_kwargs["temperature"] = max(temp, 0.001) if temp <= 0 else temp
        if params.get("top_p") is not None:
            infer_kwargs["top_p"] = params["top_p"]
        if params.get("repetition_penalty") is not None:
            infer_kwargs["repetition_penalty"] = params["repetition_penalty"]

        logger.info(f"Using infer_stream: chunk_size={chunk_size}, kwargs={infer_kwargs}")
        
        # Use module's infer_stream - returns generator of audio chunks
        chunk_count = 0
        try:
            for chunk in model.infer_stream(text, chunk_size=chunk_size, **infer_kwargs):
                chunk_count += 1
                logger.info(f"infer_stream yielded chunk {chunk_count}, type={type(chunk)}")
                yield chunk
            logger.info(f"infer_stream complete, total chunks: {chunk_count}")
        except TypeError as e:
            # Model state corrupted (hidden_states is None) - reset and retry streaming
            if "NoneType" in str(e) and "subscriptable" in str(e):
                logger.error(f"Model state corrupted after {chunk_count} chunks, resetting and retrying stream...")
                model = self._reset_model()
                
                # Retry streaming with fresh model
                for chunk in model.infer_stream(text, chunk_size=chunk_size, **infer_kwargs):
                    chunk_count += 1
                    logger.info(f"infer_stream (retry) yielded chunk {chunk_count}, type={type(chunk)}")
                    yield chunk
            else:
                raise


# Global model instance
SOPRANO = SopranoModel()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": SOPRANO.loaded,
        "device": SOPRANO.device or SOPRANO_DEVICE,
        "backend": SOPRANO.backend or SOPRANO_BACKEND,
        "sample_rate": SOPRANO.sample_rate,
        "cache_size_mb": SOPRANO_CACHE_MB,
        "decoder_batch_size": SOPRANO_DECODER_BATCH,
    }


@app.post("/v1/tts")
async def generate_tts(request: SopranoTTSRequest):
    """
    Generate TTS audio (non-streaming).
    
    Returns complete WAV file.
    """
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        output_path = SOPRANO.generate(request.input, request.model_dump())
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename="soprano_output.wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Soprano TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """Convert audio array (numpy or torch tensor) to WAV bytes."""
    import torch
    
    # Convert torch tensor to numpy if needed
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.detach().cpu().numpy()
    
    # Ensure float32 for proper soundfile handling
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Normalize if values are outside [-1, 1] range (e.g., raw model output)
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@app.post("/v1/tts/stream")
async def generate_tts_stream(request: SopranoTTSRequest):
    """
    Generate TTS audio with streaming (Server-Sent Events).
    
    Yields chunks as they're generated for low-latency playback.
    Each chunk is a complete WAV file (base64 encoded).
    
    Note: If true streaming isn't supported, falls back to generating
    complete audio and returning it as a single chunk.
    """
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Track active generation - used for cleanup on disconnect
    import threading
    cancelled = threading.Event()

    def sync_stream_generator():
        """
        Synchronous generator that yields SSE-formatted strings.
        This will be wrapped to run in a thread-safe manner.
        """
        sample_rate = SOPRANO.sample_rate
        logger.info(f"sync_stream_generator started, sample_rate={sample_rate}")
        
        # Send start event
        start_event = f"data: {json.dumps({'type': 'start', 'sample_rate': sample_rate})}\n\n"
        logger.info(f"Yielding start event")
        yield start_event
        
        chunk_index = 0
        try:
            logger.info(f"Calling SOPRANO.stream() for: {request.input[:50]}...")
            stream = SOPRANO.stream(request.input, request.model_dump())
            logger.info(f"Got stream generator: {stream}")
            
            for chunk in stream:
                # Check if generation was cancelled
                if cancelled.is_set():
                    logger.info(f"Generation cancelled after {chunk_index} chunks")
                    return
                logger.info(f"Got chunk from stream, type={type(chunk)}")
                # Handle different return formats from infer_stream
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    audio_array, chunk_sr = chunk
                    logger.info(f"Chunk is tuple: array shape={audio_array.shape}, sr={chunk_sr}")
                else:
                    audio_array = chunk
                    chunk_sr = sample_rate
                    logger.info(f"Chunk is array: shape={audio_array.shape}")
                
                # Debug audio properties
                import torch
                if isinstance(audio_array, torch.Tensor):
                    arr_for_debug = audio_array.detach().cpu().numpy()
                else:
                    arr_for_debug = audio_array
                logger.info(f"Audio chunk {chunk_index}: dtype={arr_for_debug.dtype}, min={arr_for_debug.min():.4f}, max={arr_for_debug.max():.4f}, len={len(arr_for_debug)}, sr={chunk_sr}")
                
                # Convert to WAV bytes and base64 encode
                wav_bytes = _audio_to_wav_bytes(audio_array, chunk_sr)
                wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                
                payload = {
                    "type": "chunk",
                    "index": chunk_index,
                    "audio": wav_b64,
                    "sample_rate": chunk_sr,
                }
                chunk_event = f"data: {json.dumps(payload)}\n\n"
                logger.info(f"Yielding chunk {chunk_index}, wav_bytes={len(wav_bytes)}, b64_len={len(wav_b64)}")
                yield chunk_event
                chunk_index += 1
            
            logger.info(f"Stream iteration complete, got {chunk_index} chunks")
                
        except GeneratorExit:
            # Client disconnected - stop gracefully
            logger.info(f"Client disconnected (GeneratorExit) after {chunk_index} chunks")
            cancelled.set()
            return
        except Exception as e:
            logger.error(f"Stream generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        finally:
            # Always mark as cancelled when exiting to clean up any background work
            cancelled.set()
        
        # Send complete event
        logger.info(f"Yielding complete event, chunks={chunk_index}")
        yield f"data: {json.dumps({'type': 'complete', 'chunks': chunk_index})}\n\n"

    # Use a simple streaming response with the sync generator
    # FastAPI/Starlette will handle running this appropriately
    return StreamingResponse(
        sync_stream_generator(), 
        media_type="text/event-stream"
    )


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "loaded": SOPRANO.loaded,
        "device": SOPRANO.device,
        "backend": SOPRANO.backend,
        "sample_rate": SOPRANO.sample_rate,
        "cache_size_mb": SOPRANO_CACHE_MB,
        "decoder_batch_size": SOPRANO_DECODER_BATCH,
    }


@app.post("/unload")
async def unload():
    """Unload the model to free memory."""
    with SOPRANO.lock:
        if SOPRANO.model is not None:
            try:
                del SOPRANO.model
                SOPRANO.model = None
                SOPRANO.loaded = False
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                logger.info("Soprano model unloaded")
                return {"success": True, "message": "Model unloaded"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": True, "message": "No model loaded"}


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "available": True,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
            }
        return {"available": False, "error": "CUDA not available"}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/v1/models")
async def list_models():
    """List available Soprano models (default + custom)."""
    models = [
        {
            "name": "default",
            "type": "default",
            "path": None,
            "description": "Default Soprano-1.1-80M model from HuggingFace",
            "valid": True,
            "error": None
        }
    ]
    
    # Scan custom models directory
    if os.path.exists(_CUSTOM_MODELS_DIR):
        for name in os.listdir(_CUSTOM_MODELS_DIR):
            model_path = os.path.join(_CUSTOM_MODELS_DIR, name)
            if os.path.isdir(model_path):
                # Check for model files
                has_safetensors = any(f.endswith('.safetensors') for f in os.listdir(model_path))
                has_config = os.path.exists(os.path.join(model_path, 'config.json'))
                
                # Full validation
                is_valid, error_msg = _validate_custom_model(model_path)
                
                models.append({
                    "name": name,
                    "type": "custom",
                    "path": model_path,
                    "has_safetensors": has_safetensors,
                    "has_config": has_config,
                    "valid": is_valid,
                    "error": error_msg if not is_valid else None
                })
    
    return {
        "models": models,
        "current": SOPRANO.model_name
    }


def _validate_custom_model(model_path: str) -> tuple[bool, str]:
    """
    Validate that a custom model directory contains required files.
    
    Returns (is_valid, error_message).
    """
    if not os.path.isdir(model_path):
        return False, "Path is not a directory"
    
    # Check for config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False, "Missing config.json - model has not been trained or saved correctly"
    
    # Check config.json has model_type
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if "model_type" not in config:
            return False, "config.json is missing 'model_type' key - model was not saved correctly"
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Cannot read config.json: {e}"
    
    # Check for model weights
    has_safetensors = any(f.endswith('.safetensors') for f in os.listdir(model_path))
    has_bin = any(f.endswith('.bin') and 'pytorch_model' in f for f in os.listdir(model_path))
    if not has_safetensors and not has_bin:
        return False, "Missing model weights (.safetensors or pytorch_model.bin) - model has not been trained yet"
    
    return True, ""


@app.post("/v1/models/switch")
async def switch_model(model_name: str = "default"):
    """Switch to a different Soprano model."""
    logger.info(f"=== MODEL SWITCH REQUEST: '{model_name}' ===")
    
    if model_name == "default":
        SOPRANO.set_model(model_path=None, model_name="default")
        logger.info("Switched to default Soprano model")
        return {"success": True, "model": "default", "message": "Switched to default model"}
    
    # Check custom models
    model_path = os.path.join(_CUSTOM_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Validate model has required files
    is_valid, error_msg = _validate_custom_model(model_path)
    if not is_valid:
        logger.error(f"Custom model '{model_name}' is invalid: {error_msg}")
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' is not a valid Soprano model: {error_msg}. "
                   f"Please train the model first or ensure the model files are correctly saved."
        )
    
    SOPRANO.set_model(model_path=model_path, model_name=model_name)
    logger.info(f"Switching to custom model: {model_name}")
    
    return {
        "success": True,
        "model": model_name,
        "path": model_path,
        "message": f"Switched to model '{model_name}'. Model will load on next generation."
    }


@app.get("/v1/models/current")
async def current_model():
    """Get the currently loaded model."""
    return {
        "name": SOPRANO.model_name,
        "path": SOPRANO.model_path,
        "loaded": SOPRANO.loaded,
        "sample_rate": SOPRANO.sample_rate
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Soprano TTS Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8894, help="Port to bind to")
    parser.add_argument("--cache-size", type=int, default=None, 
                       help="Cache size in MB (overrides SOPRANO_CACHE_SIZE_MB)")
    parser.add_argument("--decoder-batch-size", type=int, default=None,
                       help="Decoder batch size (overrides SOPRANO_DECODER_BATCH_SIZE)")
    args = parser.parse_args()
    
    # Override from CLI if provided
    if args.cache_size is not None:
        SOPRANO_CACHE_MB = args.cache_size
    if args.decoder_batch_size is not None:
        SOPRANO_DECODER_BATCH = args.decoder_batch_size

    logger.info(f"Starting Soprano TTS server on {args.host}:{args.port}")
    logger.info(f"Config: cache={SOPRANO_CACHE_MB}MB, batch={SOPRANO_DECODER_BATCH}")
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=3600)
