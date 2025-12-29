"""
UVR5 VR Audio Separators - Multiple model support
Based on Mangio-RVC-Fork uvr5 implementation.

Supported models:
- HP5_only_main_vocal.pth - Isolate main vocals from music
- VR-DeEchoAggressive.pth - Remove echo/delay aggressively
- VR-DeEchoDeReverb.pth - Remove echo and reverb
- VR-DeEchoNormal.pth - Standard echo removal
"""
import os
import logging
import urllib.request
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from .networks import CascadedASPPNet, CascadedNet
from . import spec_utils

logger = logging.getLogger(__name__)


class ModelParameters:
    """Model parameters wrapper - matches Mangio RVC's ModelParameters class."""
    
    def __init__(self, param_dict):
        self.param = param_dict
        # Ensure required keys exist
        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            if k not in self.param:
                self.param[k] = False


# ============================================================================
# Model Configurations
# ============================================================================

# HP5 uses 4band_v2 parameters - for vocal isolation
MODEL_PARAMS_HP5 = {
    "bins": 672,
    "unstable_bins": 8,
    "reduction_bins": 637,
    "band": {
        1: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 640,
            "crop_start": 0,
            "crop_stop": 85,
            "lpf_start": 25,
            "lpf_stop": 53,
            "res_type": "polyphase"
        },
        2: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 320,
            "crop_start": 4,
            "crop_stop": 87,
            "hpf_start": 25,
            "hpf_stop": 12,
            "lpf_start": 31,
            "lpf_stop": 62,
            "res_type": "polyphase"
        },
        3: {
            "sr": 14700,
            "hl": 160,
            "n_fft": 512,
            "crop_start": 17,
            "crop_stop": 216,
            "hpf_start": 48,
            "hpf_stop": 24,
            "lpf_start": 139,
            "lpf_stop": 210,
            "res_type": "polyphase"
        },
        4: {
            "sr": 44100,
            "hl": 480,
            "n_fft": 960,
            "crop_start": 78,
            "crop_stop": 383,
            "hpf_start": 130,
            "hpf_stop": 86,
            "res_type": "kaiser_fast"
        }
    },
    "sr": 44100,
    "pre_filter_start": 668,
    "pre_filter_stop": 672,
    "mid_side": False,
    "mid_side_b2": False,
    "reverse": False
}

# DeEcho models use 4band_v3 parameters - for echo/reverb removal
# From Mangio-RVC-Fork/lib/uvr5_pack/lib_v5/modelparams/4band_v3.json
MODEL_PARAMS_DEECHO = {
    "bins": 672,
    "unstable_bins": 8,
    "reduction_bins": 530,  # v3 uses 530, not 637
    "band": {
        1: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 640,
            "crop_start": 0,
            "crop_stop": 85,
            "lpf_start": 25,
            "lpf_stop": 53,
            "res_type": "polyphase"
        },
        2: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 320,
            "crop_start": 4,
            "crop_stop": 87,
            "hpf_start": 25,
            "hpf_stop": 12,
            "lpf_start": 31,
            "lpf_stop": 62,
            "res_type": "polyphase"
        },
        3: {
            "sr": 14700,
            "hl": 160,
            "n_fft": 512,
            "crop_start": 17,
            "crop_stop": 216,
            "hpf_start": 48,
            "hpf_stop": 24,
            "lpf_start": 139,
            "lpf_stop": 210,
            "res_type": "polyphase"
        },
        4: {
            "sr": 44100,
            "hl": 480,
            "n_fft": 960,
            "crop_start": 78,
            "crop_stop": 383,
            "hpf_start": 130,
            "hpf_stop": 86,
            "res_type": "kaiser_fast"
        }
    },
    "sr": 44100,
    "pre_filter_start": 668,
    "pre_filter_stop": 672,
    "mid_side": False,
    "mid_side_b2": False,
    "reverse": False
}

# Available models registry
# Note: DeEcho models use a different architecture that may not be compatible
# with our current implementation. HP5 is the most reliable option.
AVAILABLE_MODELS = {
    "hp5_vocals": {
        "filename": "HP5_only_main_vocal.pth",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth",
        "params": MODEL_PARAMS_HP5,
        "description": "Isolate main vocals from music/background (RECOMMENDED)",
        "output_type": "vocals",  # Returns vocals (removes instrumental)
        "architecture": "cascaded_aspp",  # Uses CascadedASPPNet from nets.py
    },
    "deecho_aggressive": {
        "filename": "VR-DeEchoAggressive.pth",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth",
        "params": MODEL_PARAMS_DEECHO,
        "description": "Aggressively remove echo and delay",
        "output_type": "clean",  # Returns cleaned audio (removes echo)
        "architecture": "cascaded_new",  # Uses CascadedNet from nets_new.py
        "nout": 48,  # Standard echo removal
    },
    "deecho_dereverb": {
        "filename": "VR-DeEchoDeReverb.pth",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth",
        "params": MODEL_PARAMS_DEECHO,
        "description": "Remove echo and reverb together",
        "output_type": "clean",
        "architecture": "cascaded_new",  # Uses CascadedNet from nets_new.py
        "nout": 64,  # DeReverb uses more channels
    },
    "deecho_normal": {
        "filename": "VR-DeEchoNormal.pth",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth",
        "params": MODEL_PARAMS_DEECHO,
        "description": "Standard echo removal",
        "output_type": "clean",
        "architecture": "cascaded_new",  # Uses CascadedNet from nets_new.py
        "nout": 48,  # Standard echo removal
    },
}

# Legacy alias
MODEL_PARAMS = MODEL_PARAMS_HP5


def make_padding(width, cropsize, offset):
    """Calculate padding for inference."""
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left
    return left, right, roi_size


def inference(X_spec, device, model, aggressiveness, data):
    """Run model inference - from Mangio RVC utils.py"""
    
    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(n_window), desc="Processing"):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + data["window_size"]]
                X_mag_window = torch.from_numpy(X_mag_window)
                if is_half:
                    X_mag_window = X_mag_window.half()
                X_mag_window = X_mag_window.to(device)

                pred = model.predict(X_mag_window, aggressiveness)
                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)
        return pred

    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)
        return X_mag, X_phase

    X_mag, X_phase = preprocess(X_spec)
    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))
    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    # Check if model is half precision
    if list(model.state_dict().values())[0].dtype == torch.float16:
        is_half = True
    else:
        is_half = False

    pred = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half)
    pred = pred[:, :, :n_frame]

    return pred * coef, X_mag, np.exp(1.0j * X_phase)


class UVR5Separator:
    """
    Generic UVR5 VR model separator.
    Supports multiple models: HP5 vocals, DeEcho, DeReverb.
    """
    
    def __init__(
        self, 
        model_key: str = "hp5_vocals",
        model_dir: str = None, 
        device: str = None, 
        aggression: int = 10, 
        is_half: bool = True
    ):
        """
        Initialize the UVR5 separator.
        
        Args:
            model_key: Key from AVAILABLE_MODELS (e.g., 'hp5_vocals', 'deecho_aggressive')
            model_dir: Directory to store/load model from
            device: Torch device ('cuda' or 'cpu')
            aggression: Aggressiveness (0-100, default 10)
            is_half: Use half precision (FP16) for faster inference
        """
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key
        self.model_info = AVAILABLE_MODELS[model_key]
        
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "uvr5")
        
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, self.model_info["filename"])
        self.mp = ModelParameters(self.model_info["params"].copy())
        self.is_half = is_half
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Aggressiveness settings
        self.aggression = aggression
        self.aggressiveness = {
            "value": aggression / 100.0,
            "split_bin": self.mp.param["band"][1]["crop_stop"]
        }
        
        # Processing settings
        self.data = {
            "postprocess": False,
            "tta": False,
            "window_size": 512,
            "agg": aggression,
            "high_end_process": "mirroring",
        }
        
        self.model = None
        self._loaded = False
    
    def _download_model(self) -> bool:
        """Download the model if not present."""
        if os.path.exists(self.model_path):
            logger.info(f"{self.model_info['filename']} already exists: {self.model_path}")
            return True
        
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Downloading {self.model_info['filename']}...")
        
        try:
            urllib.request.urlretrieve(self.model_info["url"], self.model_path)
            logger.info(f"Model downloaded to: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the model into memory."""
        if self._loaded:
            return True
        
        if not self._download_model():
            logger.error(f"Failed to download {self.model_info['filename']}")
            return False
        
        try:
            logger.info(f"Loading {self.model_info['filename']} from: {self.model_path}")
            
            # Load weights first
            logger.info("Loading model weights...")
            cpk = torch.load(self.model_path, map_location="cpu", weights_only=False)
            
            # Debug: Log some checkpoint keys
            cpk_keys = list(cpk.keys())[:5]
            logger.info(f"Checkpoint has {len(cpk.keys())} keys, first few: {cpk_keys}")
            
            n_fft = self.mp.param["bins"] * 2
            architecture = self.model_info.get("architecture", "cascaded_aspp")
            logger.info(f"Using architecture: {architecture}, n_fft={n_fft}")
            
            # Use the architecture specified in model config
            if architecture == "cascaded_new":
                # DeEcho/DeReverb models use CascadedNet from nets_new.py
                nout = self.model_info.get("nout", 48)
                logger.info(f"Creating CascadedNet with n_fft={n_fft}, nout={nout}")
                self.model = CascadedNet(n_fft, nout=nout)
            else:
                # HP5 and standard models use CascadedASPPNet
                logger.info(f"Creating CascadedASPPNet with n_fft={n_fft}")
                self.model = CascadedASPPNet(n_fft)
            
            try:
                self.model.load_state_dict(cpk)
                logger.info(f"Successfully loaded model weights")
            except RuntimeError as e:
                logger.warning(f"Strict load failed: {str(e)[:200]}...")
                # Try with strict=False as fallback
                logger.warning("Trying with strict=False...")
                self.model.load_state_dict(cpk, strict=False)
                logger.warning("Loaded with strict=False - some weights may be missing")
            
            self.model.eval()
            
            # Convert to half precision if requested and using CUDA
            if self.is_half and self.device.type == "cuda":
                self.model = self.model.half()
                logger.info("Using half precision (FP16)")
            
            self.model.to(self.device)
            
            self._loaded = True
            logger.info(f"{self.model_info['filename']} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def separate(self, audio_path: str, output_dir: str = None, save_secondary: bool = False) -> str:
        """
        Process audio file with the loaded model.
        
        For HP5 (vocal isolation): Returns vocals, optionally saves instrumental
        For DeEcho models: Returns cleaned audio, optionally saves removed echo/reverb
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output (uses temp if not provided)
            save_secondary: If True, also save the secondary output (instrumental/echo)
        
        Returns:
            Path to processed audio file, or original path if processing fails
        """
        if not self.load_model():
            logger.warning("Model not loaded, returning original audio")
            return audio_path
        
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        try:
            name = os.path.basename(audio_path)
            output_type = self.model_info.get("output_type", "vocals")
            logger.info(f"[UVR5:{self.model_key}] Processing: {name}")
            
            # Load audio through all bands
            X_wave = {}
            X_spec_s = {}
            bands_n = len(self.mp.param["band"])
            
            for d in range(bands_n, 0, -1):
                bp = self.mp.param["band"][d]
                
                if d == bands_n:
                    # High-end band
                    X_wave[d], _ = librosa.load(
                        audio_path,
                        sr=bp["sr"],
                        mono=False,
                        dtype=np.float32,
                        res_type=bp["res_type"]
                    )
                    
                    if X_wave[d].ndim == 1:
                        X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
                    
                    # Log input audio level
                    input_max = np.max(np.abs(X_wave[d]))
                    print(f"[UVR5] Input audio max level: {input_max:.4f}, shape: {X_wave[d].shape}", flush=True)
                else:
                    # Lower bands - resample from higher band
                    X_wave[d] = librosa.resample(
                        X_wave[d + 1],
                        orig_sr=self.mp.param["band"][d + 1]["sr"],
                        target_sr=bp["sr"],
                        res_type=bp["res_type"]
                    )
                
                # Convert to spectrogram
                X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                    X_wave[d],
                    bp["hl"],
                    bp["n_fft"],
                    self.mp.param["mid_side"],
                    self.mp.param["mid_side_b2"],
                    self.mp.param["reverse"]
                )
                
                # Store high-end for mirroring
                if d == bands_n and self.data["high_end_process"] != "none":
                    input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                        self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                    )
                    input_high_end = X_spec_s[d][
                        :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                    ]
            
            # Combine spectrograms
            X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
            
            # Run inference
            with torch.no_grad():
                pred, X_mag, X_phase = inference(
                    X_spec_m, self.device, self.model, self.aggressiveness, self.data
                )
            
            # Create output spectrograms
            # The model predicts a mask that when applied gives one component
            y_spec_m = pred * X_phase  # What the model extracted
            v_spec_m = X_spec_m - y_spec_m  # The remainder
            
            # Determine which output is primary based on model type
            # IMPORTANT: DeEcho models have REVERSED outputs vs HP5!
            # - HP5: pred extracts instrumental, remainder is vocals
            # - DeEcho: pred extracts echo/reverb, remainder is clean (SWAPPED in original Mangio code)
            if output_type == "vocals":
                # HP5: model extracts instrumental, we want the remainder (vocals)
                primary_spec = v_spec_m
                secondary_spec = y_spec_m
                primary_label = "vocals"
                secondary_label = "instrumental"
            else:  # "clean" for deecho models
                # DeEcho: model extracts what it "removes", we want y_spec_m (the extracted part is actually clean)
                # Per Mangio code comment: "3个VR模型vocal和ins是反的" (3 VR models vocal and ins are reversed)
                primary_spec = y_spec_m  # SWAPPED - the extracted part is clean audio
                secondary_spec = v_spec_m  # The remainder is the echo/reverb
                primary_label = "cleaned"
                secondary_label = "removed"
            
            print(f"[UVR5] Creating {primary_label} audio...", flush=True)
            
            # Convert primary spectrogram to wave
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(
                    self.data["high_end_process"], primary_spec, input_high_end, self.mp
                )
                wav_primary = spec_utils.cmb_spectrogram_to_wave(
                    primary_spec, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_primary = spec_utils.cmb_spectrogram_to_wave(primary_spec, self.mp)
            
            # Get the VoiceForge output directory
            voiceforge_output = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")
            os.makedirs(voiceforge_output, exist_ok=True)
            
            # Base name without extension
            base_name = os.path.splitext(name)[0]
            
            # Save primary output
            output_path = os.path.join(
                output_dir, f"{primary_label}_{name}_{self.data['agg']}.wav"
            )
            
            # Check audio levels and duration
            wav_primary_array = np.array(wav_primary)
            primary_max = np.max(np.abs(wav_primary_array))
            primary_duration = len(wav_primary_array) / self.mp.param["sr"]
            print(f"[UVR5] {primary_label.title()} audio: shape={wav_primary_array.shape}, max={primary_max:.4f}, duration={primary_duration:.1f}s", flush=True)
            
            # Normalize output to good level
            if primary_max < 0.001:
                print(f"[UVR5] WARNING: {primary_label.title()} audio is very quiet! Processing may have failed.", flush=True)
            elif primary_max > 0:
                target_level = 0.8
                if primary_max < target_level:
                    boost_factor = target_level / primary_max
                    print(f"[UVR5] Boosting {primary_label} level by {boost_factor:.1f}x", flush=True)
                    wav_primary_array = wav_primary_array * boost_factor
                    wav_primary_array = np.clip(wav_primary_array, -1.0, 1.0)
            
            sf.write(
                output_path,
                (wav_primary_array * 32768).astype("int16"),
                self.mp.param["sr"]
            )
            
            # Also save to VoiceForge output directory
            primary_output_path = os.path.join(voiceforge_output, f"{base_name}_{primary_label}.wav")
            sf.write(
                primary_output_path,
                (wav_primary_array * 32768).astype("int16"),
                self.mp.param["sr"]
            )
            print(f"[UVR5] {primary_label.title()} saved to: {primary_output_path}", flush=True)
            
            # Save secondary output if requested
            if save_secondary:
                print(f"[UVR5] Creating {secondary_label} audio...", flush=True)
                if self.data["high_end_process"].startswith("mirroring"):
                    input_high_end_sec = spec_utils.mirroring(
                        self.data["high_end_process"], secondary_spec, input_high_end, self.mp
                    )
                    wav_secondary = spec_utils.cmb_spectrogram_to_wave(
                        secondary_spec, self.mp, input_high_end_h, input_high_end_sec
                    )
                else:
                    wav_secondary = spec_utils.cmb_spectrogram_to_wave(secondary_spec, self.mp)
                
                wav_secondary_array = np.array(wav_secondary)
                secondary_max = np.max(np.abs(wav_secondary_array))
                secondary_duration = len(wav_secondary_array) / self.mp.param["sr"]
                print(f"[UVR5] {secondary_label.title()} audio: shape={wav_secondary_array.shape}, max={secondary_max:.4f}, duration={secondary_duration:.1f}s", flush=True)
                
                secondary_output_path = os.path.join(voiceforge_output, f"{base_name}_{secondary_label}.wav")
                sf.write(
                    secondary_output_path,
                    (wav_secondary_array * 32768).astype("int16"),
                    self.mp.param["sr"]
                )
                print(f"[UVR5] {secondary_label.title()} saved to: {secondary_output_path}", flush=True)
            
            logger.info(f"[UVR5:{self.model_key}] Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"[UVR5:{self.model_key}] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return audio_path
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[UVR5:{self.model_key}] Model unloaded")


# Legacy alias for backwards compatibility
class HP5VocalSeparator(UVR5Separator):
    """
    Vocal separator using HP5_only_main_vocal.pth model.
    Isolates main vocals from audio, removing background music and noise.
    
    This is a convenience wrapper around UVR5Separator for backwards compatibility.
    """
    
    MODEL_NAME = "HP5_only_main_vocal.pth"
    MODEL_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth"
    
    def __init__(self, model_dir: str = None, device: str = None, aggression: int = 10, is_half: bool = True):
        super().__init__(
            model_key="hp5_vocals",
            model_dir=model_dir,
            device=device,
            aggression=aggression,
            is_half=is_half
        )
