"""
Fast Spatial Audio Processing for VoiceForge.

Provides immersive binaural audio with:
- ILD (Interaural Level Difference) - proper equal-power panning with headphone compensation
- Head shadow effect - high frequencies attenuated at far ear
- Smooth per-sample panning with no artifacts

Uses scipy for fast signal processing. Much more realistic than simple L-R panning.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from typing import Tuple, Optional
import tempfile
import os


def apply_dynamic_panning(
    audio: np.ndarray,
    sample_rate: int = 44100,
    speed_hz: float = 0.1,
    start_angle: float = -90.0,
    end_angle: float = 90.0,
    mode: str = "sweep",  # "sweep", "rotate", "extreme" (skip center)
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
    block_size: int = 1024,  # Not used anymore, kept for API compatibility
) -> np.ndarray:
    """
    Apply dynamic spatial panning to audio with smooth per-sample transitions.
    
    Uses vectorized operations for speed - processes entire audio at once.
    No block artifacts or crackling.
    
    Args:
        audio: Mono audio signal
        sample_rate: Audio sample rate
        speed_hz: Panning speed in Hz (cycles per second)
        start_angle: Starting angle for sweep mode (-90 = left, 90 = right)
        end_angle: Ending angle for sweep mode  
        mode: "sweep" (linear), "rotate" (sine), "extreme" (skip center, stay at edges)
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        
    Returns:
        Stereo audio with dynamic spatial positioning
    """
    # Ensure mono and float32
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    n_samples = len(audio)
    
    # Create time array for the entire audio
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    
    # Calculate angle over time based on mode
    angle_range = end_angle - start_angle
    mid_angle = (start_angle + end_angle) / 2
    
    if mode == "extreme":
        # EXTREME MODE: Dwell at arc extremes, quick transition through center
        wave = np.sin(2 * np.pi * speed_hz * t)  # -1 to 1
        
        # tanh(x*15) = ~95% time at peaks, ~5% in transition
        extreme_wave = np.tanh(wave * 15) * 1.01
        extreme_wave = np.clip(extreme_wave, -1.0, 1.0)
        
        angles = mid_angle + (angle_range / 2) * extreme_wave
    elif mode == "sweep":
        # Triangle wave (ping-pong): use asin(sin()) trick for smooth linear sweep
        # This creates a triangle wave that goes back and forth
        wave = np.sin(2 * np.pi * speed_hz * t)
        triangle = (2 / np.pi) * np.arcsin(wave)  # -1 to 1, triangle shape
        angles = mid_angle + (angle_range / 2) * triangle
    else:  # rotate
        # Smooth sine wave oscillation
        wave = np.sin(2 * np.pi * speed_hz * t)
        angles = mid_angle + (angle_range / 2) * wave
    
    # Convert angles to pan position using sine for natural 360° wrapping
    # 0° = front (center), 90° = right, 180° = back (center), -90°/270° = left
    # Using sin(angle) gives: 0° -> 0, 90° -> 1, 180° -> 0, 270° -> -1
    pan = np.sin(np.radians(angles))
    pan = np.clip(pan, -1.0, 1.0)
    
    # Track if we're in the back hemisphere (for potential future effects)
    # Back is when |angle| > 90°
    is_back = np.abs(angles) > 90
    
    # Equal-power panning law (vectorized)
    left_gain = np.sqrt(0.5 * (1 - pan))
    right_gain = np.sqrt(0.5 * (1 + pan))
    
    # Headphone compensation: boost when panned to sides (~4.5dB at extremes)
    compensation = 1.0 + 0.68 * (pan ** 2)
    left_gain *= compensation
    right_gain *= compensation
    
    # Apply gains to create stereo
    left_out = audio * left_gain
    right_out = audio * right_gain
    
    # Apply head shadow effect (high frequency attenuation at far ear)
    if head_shadow and head_shadow_intensity > 0:
        # Create a gentle low-pass filtered version for the "far" ear effect
        # Use a smooth crossfade based on pan position
        
        # Design low-pass filter for head shadow
        cutoff_hz = 4000  # Head shadow affects frequencies above ~4kHz
        nyquist = sample_rate / 2
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(2, normalized_cutoff, btype='low')
        
        # Apply filter to get "shadowed" version
        audio_shadowed = signal.lfilter(b, a, audio).astype(np.float32)
        
        # Blend between normal and shadowed based on pan position
        # When panned right (pan > 0), left ear gets more shadow
        # When panned left (pan < 0), right ear gets more shadow
        shadow_amount = np.abs(pan) * head_shadow_intensity
        
        # Left ear: more shadow when sound is to the right (pan > 0)
        left_shadow = np.where(pan > 0, shadow_amount, 0)
        left_out = left_out * (1 - left_shadow) + (audio_shadowed * left_gain) * left_shadow
        
        # Right ear: more shadow when sound is to the left (pan < 0)  
        right_shadow = np.where(pan < 0, shadow_amount, 0)
        right_out = right_out * (1 - right_shadow) + (audio_shadowed * right_gain) * right_shadow
    
    # Stack to stereo
    stereo = np.column_stack([left_out, right_out])
    
    # Soft clip to prevent harsh clipping (tanh-based)
    max_val = np.abs(stereo).max()
    if max_val > 0.95:
        # Gentle soft clipping
        stereo = np.tanh(stereo * (1.0 / max_val) * 1.5) * 0.95
    
    return stereo.astype(np.float32)


def apply_static_position(
    audio: np.ndarray,
    angle: float,
    sample_rate: int = 44100,
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
) -> np.ndarray:
    """
    Apply static spatial positioning to audio.
    
    Args:
        audio: Mono audio signal
        angle: Position angle (-90 = left, 0 = center, 90 = right)
        sample_rate: Audio sample rate
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        
    Returns:
        Stereo audio
    """
    # Ensure mono and float32
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    # Convert angle to pan position
    pan = np.clip(angle / 90.0, -1.0, 1.0)
    
    # Equal-power panning with headphone compensation
    left_gain = np.sqrt(0.5 * (1 - pan))
    right_gain = np.sqrt(0.5 * (1 + pan))
    compensation = 1.0 + 0.68 * (pan ** 2)
    
    left_out = audio * left_gain * compensation
    right_out = audio * right_gain * compensation
    
    # Apply head shadow if panned to side
    if head_shadow and head_shadow_intensity > 0 and abs(pan) > 0.1:
        cutoff_hz = 4000
        nyquist = sample_rate / 2
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(2, normalized_cutoff, btype='low')
        audio_shadowed = signal.lfilter(b, a, audio).astype(np.float32)
        
        shadow_amount = abs(pan) * head_shadow_intensity
        if pan > 0:  # Sound to right, shadow left ear
            left_out = left_out * (1 - shadow_amount) + audio_shadowed * left_gain * compensation * shadow_amount
        else:  # Sound to left, shadow right ear
            right_out = right_out * (1 - shadow_amount) + audio_shadowed * right_gain * compensation * shadow_amount
    
    return np.column_stack([left_out, right_out]).astype(np.float32)


def process_spatial_audio_file(
    input_path: str,
    output_path: str,
    mode: str = "sweep",
    speed_hz: float = 0.1,
    start_angle: float = -90.0,
    end_angle: float = 90.0,
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
) -> str:
    """
    Process an audio file with spatial panning effects.
    """
    sample_rate, audio = wavfile.read(input_path)
    
    # Convert to float32 normalized
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    if mode == "static":
        stereo = apply_static_position(audio, start_angle, sample_rate, head_shadow, head_shadow_intensity)
    else:
        stereo = apply_dynamic_panning(audio, sample_rate, speed_hz, start_angle, end_angle, mode, head_shadow, head_shadow_intensity)
    
    stereo_int16 = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, stereo_int16)
    
    return output_path


def process_spatial_audio_buffer(
    audio: np.ndarray,
    sample_rate: int,
    mode: str = "sweep",
    speed_hz: float = 0.1,
    start_angle: float = -90.0,
    end_angle: float = 90.0,
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
) -> np.ndarray:
    """
    Process audio buffer with spatial panning effects.
    
    Args:
        audio: Input audio (mono or stereo, any dtype)
        sample_rate: Sample rate
        mode: "sweep", "rotate", or "static"
        speed_hz: Panning speed (for sweep/rotate modes)
        start_angle: Start angle (-90 = left, 0 = center, 90 = right)
        end_angle: End angle (for sweep mode)
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        
    Returns:
        Processed stereo audio (float32, normalized)
    """
    # Convert to float32 normalized if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    if mode == "static":
        return apply_static_position(audio, start_angle, sample_rate, head_shadow, head_shadow_intensity)
    else:
        return apply_dynamic_panning(audio, sample_rate, speed_hz, start_angle, end_angle, mode, head_shadow, head_shadow_intensity)

