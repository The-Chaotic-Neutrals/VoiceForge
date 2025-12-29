"""
RVC Router - Voice Conversion endpoints.

Handles:
- /api/rvc/audio - Process audio through RVC
- /api/models - List available RVC models
"""

import json
import subprocess
import os
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File

# Import common (sets up sys.path)
from .common import verify_auth, MODEL_DIR

from util.clients import run_rvc
from servers.utils.param_parsing import get_rvc_params_dict
from util.audio_upload_utils import process_audio_upload


router = APIRouter(tags=["RVC"])


@router.get("/api/models")
async def get_models(_: bool = Depends(verify_auth)):
    """List available RVC models."""
    models = []
    if os.path.exists(MODEL_DIR):
        for item in os.listdir(MODEL_DIR):
            model_path = os.path.join(MODEL_DIR, item)
            if os.path.isdir(model_path):
                model_pth = os.path.join(model_path, "model.pth")
                model_index = os.path.join(model_path, "model.index")
                if os.path.exists(model_pth) and os.path.exists(model_index):
                    models.append(item)
    return {"models": models}


@router.post("/api/rvc/audio")
async def rvc_process_audio_file(
    request: Request,
    audio_file: UploadFile = File(...),
    model_name: str = Form(...),
    rvc_params: Optional[str] = Form(None),
    _: bool = Depends(verify_auth)
):
    """
    Process an audio file through RVC voice conversion.
    
    Args:
        audio_file: Audio file to process
        model_name: RVC model name
        rvc_params: JSON string of RVC parameters (optional)
    """
    # Parse RVC parameters using unified parsing utility
    rvc_params_dict = get_rvc_params_dict(rvc_params)
    
    # Add model_name to params
    rvc_params_dict['model_name'] = model_name
    
    # Processor function
    def processor(audio_path: str, params: Dict[str, Any]) -> str:
        return run_rvc(audio_path, params['model_name'], params, lambda s: None, None)
    
    try:
        return await process_audio_upload(
            upload_file=audio_file,
            processor=processor,
            params=rvc_params_dict,
            convert_to_wav=True,
            sample_rate=44100,
            channels=1,  # RVC uses mono
            output_filename="rvc_processed_audio.wav"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
