"""
Post-Processing Router - Audio effects endpoints.

Handles:
- /api/post-process/audio - Apply post-processing effects
- /api/preprocess/uvr5/* - UVR5 vocal separation

Note: Post-processing config is managed via /api/config endpoint.
"""

import json
import os
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File

# Import common (sets up sys.path)
from .common import verify_auth

from util.clients import run_postprocess, is_postprocess_server_available
from servers.utils.param_parsing import get_post_process_params_dict
from config import get_config
from util.audio_upload_utils import process_audio_upload


router = APIRouter(tags=["Post-Processing"])


@router.post("/api/post-process/audio")
async def post_process_audio_file(
    request: Request,
    audio_file: UploadFile = File(...),
    post_params: Optional[str] = Form(None),
    _: bool = Depends(verify_auth)
):
    """
    Post-process an audio file using effects.
    
    Proxies directly to audio services server to avoid double HTTP hop.
    
    Args:
        audio_file: Audio file to process
        post_params: JSON string of post-processing parameters (optional)
    """
    import time
    from fastapi.responses import Response
    
    print(f"[POST-ROUTER] Received request for file: {audio_file.filename}")
    t_start = time.perf_counter()
    
    # Parse post-processing parameters using unified parsing utility
    post_params_dict = get_post_process_params_dict(post_params)
    
    # Check if any effects are actually enabled - skip processing if not
    from servers.models.params import PostProcessParams
    params_obj = PostProcessParams.from_dict(post_params_dict)
    
    if not params_obj.needs_processing():
        # No effects enabled - return input file as-is
        print(f"[POST] Skipping post-processing (no effects enabled)")
        content = await audio_file.read()
        return Response(
            content=content,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="postprocessed_audio.wav"'}
        )
    
    # Stream directly to audio services server (avoid saving to disk + re-reading)
    from util.clients import AUDIO_SERVICES_SERVER_URL, is_postprocess_server_available, get_shared_session
    import asyncio
    
    if not is_postprocess_server_available():
        print(f"[POST-ROUTER] Audio services server not available!")
        raise HTTPException(status_code=503, detail="Audio services server not available")
    
    # Read file content once
    content = await audio_file.read()
    file_size_kb = len(content) / 1024
    file_size_mb = file_size_kb / 1024
    print(f"[POST-ROUTER] Read {file_size_mb:.1f}MB from upload")
    
    # Warn about large files
    if file_size_mb > 100:
        print(f"[POST-ROUTER] WARNING: Large file ({file_size_mb:.0f}MB) - this may take a while!")
    
    t_http_start = time.perf_counter()
    
    # For large files, save to temp and stream from file (more efficient than memory)
    import tempfile
    
    def do_http_call():
        tmp_path = None
        try:
            # Write to temp file for streaming upload
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, content)
            os.close(fd)
            
            print(f"[POST-ROUTER] Starting upload ({file_size_mb:.1f}MB)...")
            
            with open(tmp_path, 'rb') as f:
                import requests
                # Dynamic timeout: 10 min base + 1 min per 20MB, max 60 min
                # 321MB file = ~18 min timeout
                read_timeout = max(600, min(3600, 600 + int(file_size_mb / 20) * 60))
                if file_size_mb > 100:
                    print(f"[POST-ROUTER] Large file timeout: {read_timeout}s ({read_timeout//60} min)")
                response = requests.post(
                    f"{AUDIO_SERVICES_SERVER_URL}/v1/postprocess",
                    files={"audio": ("input.wav", f, "audio/wav")},
                    data=post_params_dict,
                    timeout=(30, read_timeout)  # (connect=30s, read=dynamic)
                )
            return response
        except Exception as e:
            print(f"[POST-ROUTER] HTTP call failed: {type(e).__name__}: {e}")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    print(f"[POST-ROUTER] Sending {file_size_mb:.1f}MB to audio services...")
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, do_http_call)
    except Exception as e:
        print(f"[POST-ROUTER] Failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send to audio services: {e}")
    
    t_http_end = time.perf_counter()
    print(f"[POST-ROUTER] Got response: {response.status_code}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    print(f"[POST-ROUTER] HTTP complete: {(t_http_end - t_http_start)*1000:.0f}ms, reading content...")
    
    # For large files, this loads the entire response into memory
    t_content_start = time.perf_counter()
    content = response.content
    content_size_mb = len(content) / (1024 * 1024)
    print(f"[POST-ROUTER] Content read: {content_size_mb:.1f}MB in {(time.perf_counter() - t_content_start)*1000:.0f}ms")
    
    t_end = time.perf_counter()
    print(f"[POST-ROUTER] Done: {(t_http_end - t_http_start)*1000:.0f}ms for {file_size_kb:.0f}KB, total: {(t_end - t_start)*1000:.0f}ms")
    print(f"[POST-ROUTER] Returning response to client...")
    
    return Response(
        content=content,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="postprocessed_audio.wav"'}
    )


@router.post("/api/preprocess/uvr5/clean-vocals")
async def preprocess_uvr5_clean_vocals(
    audio: UploadFile = File(...),
    aggression: int = Form(10),
    model_key: str = Form("hp5_vocals"),
    skip_if_cached: str = Form("true"),
    _: bool = Depends(verify_auth)
):
    """
    Clean vocals using UVR5.
    
    Models:
    - hp5_vocals: Standard vocal separation
    - demucs: Higher quality but slower
    - deecho_normal: Standard echo removal
    - deecho_aggressive: Aggressive echo/delay removal
    - deecho_dereverb: Remove echo and reverb together
    """
    try:
        from util.clients import process_uvr5, is_preprocess_server_available
        
        if not is_preprocess_server_available():
            raise HTTPException(
                status_code=503, 
                detail="Preprocessing server not available"
            )
        
        from util.temp_file_utils import save_upload_to_temp, TempFileManager
        
        manager = TempFileManager()
        try:
            # Save uploaded file
            tmp_path = await save_upload_to_temp(audio, suffix=".wav")
            manager.files.append(tmp_path)
            
            # Process
            result = process_uvr5(
                audio_path=tmp_path,
                aggression=aggression,
                model_key=model_key,
                skip_if_cached=(skip_if_cached.lower() == "true")
            )
            
            if result and os.path.exists(result):
                manager.files.append(result)
                with open(result, "rb") as f:
                    audio_data = f.read()
                
                from fastapi.responses import Response
                return Response(
                    content=audio_data,
                    media_type="audio/wav",
                    headers={"Content-Disposition": 'attachment; filename="cleaned_vocals.wav"'}
                )
            else:
                raise HTTPException(status_code=500, detail="UVR5 processing failed")
        finally:
            manager.cleanup()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/preprocess/uvr5/unload")
async def unload_uvr5_model(_: bool = Depends(verify_auth)):
    """Unload UVR5 model to free GPU memory."""
    try:
        from util.clients import is_postprocess_server_available, get_shared_session, AUDIO_SERVICES_SERVER_URL
        
        if not is_postprocess_server_available():
            return {"success": True, "message": "Server not running, nothing to unload"}
        
        response = get_shared_session().post(
            f"{AUDIO_SERVICES_SERVER_URL}/v1/preprocess/uvr5/unload",
            timeout=30
        )
        
        if response.ok:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
