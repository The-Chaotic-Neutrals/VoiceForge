"""
Ultra-Immersive Spatial Audio Processing for VoiceForge.

Provides cinema-quality binaural audio with:
- ILD (Interaural Level Difference) - proper equal-power panning with headphone compensation
- ITD (Interaural Time Difference) - critical for realistic 3D positioning
- Head shadow effect - frequency-dependent attenuation at far ear (HRTF-like)
- Near-field proximity effect - bass boost and enhanced presence for close sounds
- Natural crossfeed - subtle bleed between channels for realism
- Micro-movements - organic subtle variations to prevent static/artificial feel
- Air absorption - distance-based high frequency rolloff
- Multiple quality presets for different use cases

Uses scipy for fast signal processing. Designed for ASMR and immersive audio.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional, Dict, Any
import tempfile
import os


# =============================================================================
# Quality Presets
# =============================================================================

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        # Lightweight - good for real-time/streaming
        "itd_enabled": True,
        "head_shadow_bands": 1,  # Single band
        "proximity_enabled": False,
        "crossfeed_enabled": False,
        "micro_movements": False,
        "air_absorption": False,
    },
    "balanced": {
        # Good quality with reasonable performance
        "itd_enabled": True,
        "head_shadow_bands": 2,  # Two bands (mid + high)
        "proximity_enabled": True,
        "crossfeed_enabled": True,
        "micro_movements": True,
        "air_absorption": False,
    },
    "ultra": {
        # Maximum immersion - for final renders
        "itd_enabled": True,
        "head_shadow_bands": 3,  # Three bands (HRTF-like)
        "proximity_enabled": True,
        "crossfeed_enabled": True,
        "micro_movements": True,
        "air_absorption": True,
    },
}


# =============================================================================
# Speech-Aware Panning - Natural Transition Timing
# =============================================================================

def detect_speech_breaks(
    audio: np.ndarray,
    sample_rate: int,
    min_break_ms: float = 120.0,  # Longer minimum = more definitive pauses only
    energy_threshold_ratio: float = 0.08,  # Stricter = truly quiet regions only
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect natural break points in speech (pauses, breaths, sentence gaps).
    
    IMPROVED VERSION with stricter detection to avoid mid-word transitions:
    - Longer minimum break duration (120ms vs 40ms)
    - Stricter energy threshold (8% vs 15%)  
    - Smoothed energy curve to avoid transient false positives
    - Returns both break points AND energy curve for gating
    
    Args:
        audio: Mono audio signal
        sample_rate: Audio sample rate
        min_break_ms: Minimum duration to count as a break (ms)
        energy_threshold_ratio: Threshold as ratio of speech energy (lower = stricter)
    
    Returns:
        Tuple of (break_points array, energy_per_sample array)
    """
    # Larger window for more stable energy measurement (25ms)
    window_ms = 25
    window_size = int(sample_rate * window_ms / 1000)
    hop_size = window_size // 4  # 75% overlap for smoother curve
    
    n_samples = len(audio)
    n_windows = (n_samples - window_size) // hop_size + 1
    
    if n_windows <= 0:
        return np.array([n_samples // 2], dtype=np.int64), np.ones(n_samples, dtype=np.float32)
    
    # Calculate RMS energy for each window
    indices = np.arange(n_windows) * hop_size
    windows = np.array([audio[i:i + window_size] for i in indices])
    energies = np.sqrt(np.mean(windows ** 2, axis=1))
    
    # Smooth the energy curve to avoid transient dips being detected as breaks
    # Use a moving average (~50ms smoothing)
    smooth_windows = max(1, int(50 / (window_ms * 0.25)))  # 0.25 because of 75% overlap
    if len(energies) > smooth_windows:
        kernel = np.ones(smooth_windows) / smooth_windows
        energies_smooth = np.convolve(energies, kernel, mode='same')
    else:
        energies_smooth = energies
    
    # Dynamic threshold - based on actual speech levels
    # Use tighter percentiles for cleaner detection
    energy_floor = np.percentile(energies_smooth, 5)   # True silence level
    energy_speech = np.percentile(energies_smooth, 85)  # Typical speech level
    
    # Break threshold: only truly quiet regions
    threshold = energy_floor + (energy_speech - energy_floor) * energy_threshold_ratio
    
    # Interpolate energy to per-sample for later gating
    energy_indices = indices + window_size // 2  # Center of each window
    energy_per_sample = np.interp(
        np.arange(n_samples),
        energy_indices,
        energies_smooth
    ).astype(np.float32)
    
    # Find quiet regions
    is_quiet = energies_smooth < threshold
    
    # Find contiguous quiet regions that are long enough
    min_break_windows = int(min_break_ms / (window_ms * 0.25))  # Adjusted for overlap
    break_points = []
    
    in_break = False
    break_start = 0
    
    for i, quiet in enumerate(is_quiet):
        if quiet and not in_break:
            in_break = True
            break_start = i
        elif not quiet and in_break:
            in_break = False
            break_length = i - break_start
            
            # Only count as break if:
            # 1. Long enough
            # 2. Has speech before (not start of audio)
            # 3. Has speech after (not end of audio)
            if break_length >= min_break_windows:
                # Check for speech context before and after
                context_windows = max(1, int(100 / (window_ms * 0.25)))  # 100ms context
                
                has_speech_before = break_start > context_windows and \
                    np.any(energies_smooth[max(0, break_start - context_windows):break_start] > threshold * 1.5)
                has_speech_after = i + context_windows < len(energies_smooth) and \
                    np.any(energies_smooth[i:min(len(energies_smooth), i + context_windows)] > threshold * 1.5)
                
                if has_speech_before and has_speech_after:
                    # Center of the break
                    break_center_window = break_start + break_length // 2
                    break_center_sample = int(break_center_window * hop_size + window_size // 2)
                    break_points.append(min(break_center_sample, n_samples - 1))
    
    # Handle break at end of audio (less strict - allow if speech before)
    if in_break:
        break_length = len(is_quiet) - break_start
        if break_length >= min_break_windows:
            context_windows = max(1, int(100 / (window_ms * 0.25)))
            has_speech_before = break_start > context_windows and \
                np.any(energies_smooth[max(0, break_start - context_windows):break_start] > threshold * 1.5)
            if has_speech_before:
                break_center_window = break_start + break_length // 2
                break_center_sample = int(break_center_window * hop_size + window_size // 2)
                break_points.append(min(break_center_sample, n_samples - 1))
    
    return np.array(break_points, dtype=np.int64), energy_per_sample


def create_speech_aware_pan(
    audio: np.ndarray,
    sample_rate: int,
    speed_hz: float,
    start_angle: float,
    end_angle: float,
    mode: str,
    break_snap_window_ms: float = 500.0,  # Wider window to find good breaks
) -> Optional[np.ndarray]:
    """
    Create panning curve that snaps transitions to speech breaks.
    
    IMPROVED VERSION with energy gating:
    - Transitions only happen during confirmed speech pauses
    - Energy gate prevents transitions during speech
    - Wider snap window (500ms) to find better break points
    - Falls back gracefully if breaks aren't available
    
    Args:
        audio: Mono audio signal (used for break detection)
        sample_rate: Audio sample rate
        speed_hz: Base panning speed in Hz
        start_angle: Start angle for sweep
        end_angle: End angle for sweep
        mode: Panning mode ("sweep", "rotate", "extreme")
        break_snap_window_ms: How far to look for breaks (ms)
    
    Returns:
        Pan position array (-1 to +1) synchronized to speech breaks, or None to use default
    """
    n_samples = len(audio)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    
    # Detect speech breaks and get energy curve
    breaks, energy = detect_speech_breaks(audio, sample_rate)
    
    # Log diagnostic info
    duration_sec = n_samples / sample_rate
    print(f"[SPATIAL-SPEECH] Audio: {duration_sec:.2f}s @ {sample_rate}Hz, {len(breaks)} breaks detected")
    
    # Calculate energy threshold for "speaking" vs "pause"
    # Use 85th percentile - more permissive to avoid blocking too many transitions
    energy_threshold = np.percentile(energy, 85)  # Above this = definitely speaking
    energy_median = np.percentile(energy, 50)
    print(f"[SPATIAL-SPEECH] Energy: median={energy_median:.4f}, threshold={energy_threshold:.4f}")
    
    # If very few breaks detected, fall back to time-based
    expected_transitions = duration_sec * speed_hz * 2  # 2 transitions per cycle
    if len(breaks) < max(2, int(expected_transitions * 0.3)):  # Need at least 30% coverage
        print(f"[SPATIAL-SPEECH] Only {len(breaks)} breaks found (need {max(2, int(expected_transitions * 0.3))}), using time-based panning")
        return None  # Signal to use default
    
    # Calculate the "ideal" time-based pan curve first
    angle_range = end_angle - start_angle
    mid_angle = (start_angle + end_angle) / 2
    
    if mode == "extreme":
        wave = np.sin(2 * np.pi * speed_hz * t)
        extreme_wave = np.tanh(wave * 15) * 1.01
        extreme_wave = np.clip(extreme_wave, -1.0, 1.0)
        ideal_angles = mid_angle + (angle_range / 2) * extreme_wave
    elif mode == "sweep":
        wave = np.sin(2 * np.pi * speed_hz * t)
        triangle = (2 / np.pi) * np.arcsin(wave)
        ideal_angles = mid_angle + (angle_range / 2) * triangle
    else:  # rotate
        wave = np.sin(2 * np.pi * speed_hz * t)
        ideal_angles = mid_angle + (angle_range / 2) * wave
    
    ideal_pan = np.sin(np.radians(ideal_angles))
    
    # Find where the ideal pan crosses through center (transition zones)
    pan_sign = np.sign(ideal_pan)
    pan_sign[pan_sign == 0] = 1
    sign_changes = np.where(np.diff(pan_sign) != 0)[0]
    
    if len(sign_changes) == 0:
        return ideal_pan  # No transitions to snap
    
    break_snap_samples = int(break_snap_window_ms * sample_rate / 1000)
    adjusted_pan = ideal_pan.copy()
    transitions_snapped = 0
    transitions_blocked = 0
    
    for transition_idx in sign_changes:
        # ENERGY GATE: Check if we're currently in speech
        # Look at energy around the proposed transition point
        gate_window = int(0.05 * sample_rate)  # 50ms window
        gate_start = max(0, transition_idx - gate_window)
        gate_end = min(n_samples, transition_idx + gate_window)
        local_energy = np.mean(energy[gate_start:gate_end])
        
        # If energy is high (speaking), we MUST find a break nearby
        is_speaking = local_energy > energy_threshold
        
        # Find nearest break within snap window
        distances = np.abs(breaks - transition_idx)
        nearby_mask = distances < break_snap_samples
        
        if not np.any(nearby_mask):
            if is_speaking:
                # No break found during speech - delay this transition
                # Find the next break point that's after this transition
                future_breaks = breaks[breaks > transition_idx]
                if len(future_breaks) > 0:
                    closest_break = future_breaks[0]
                    if closest_break - transition_idx < break_snap_samples * 2:
                        # Use this future break even though it's far
                        shift = closest_break - transition_idx
                    else:
                        transitions_blocked += 1
                        continue  # Too far, skip adjustment
                else:
                    transitions_blocked += 1
                    continue
            else:
                continue  # Not speaking, original timing is fine
        else:
            # Found nearby breaks - pick the best one
            nearby_breaks = breaks[nearby_mask]
            nearby_distances = distances[nearby_mask]
            
            # Prefer breaks that are in quiet regions
            break_energies = np.array([energy[min(b, n_samples-1)] for b in nearby_breaks])
            
            # Score: closer is better, quieter is better
            scores = nearby_distances / break_snap_samples + break_energies / (energy_threshold + 0.001)
            best_idx = np.argmin(scores)
            closest_break = nearby_breaks[best_idx]
            shift = closest_break - transition_idx
        
        if abs(shift) < int(0.02 * sample_rate):  # Within 20ms is close enough
            transitions_snapped += 1
            continue
        
        # Create smooth adjustment window
        window_size = min(break_snap_samples, abs(shift) * 3)
        start = max(0, transition_idx - window_size)
        end = min(n_samples, transition_idx + window_size + abs(shift))
        
        if end - start < 100:
            continue
        
        original_length = end - start
        
        # Time warp to shift transition
        if shift > 0:
            mid_point = transition_idx - start
            new_mid = mid_point + shift
            
            if new_mid >= original_length or new_mid <= 0:
                continue
            
            first_half = np.linspace(0, mid_point, int(new_mid), dtype=np.float32)
            second_half = np.linspace(mid_point, original_length - 1, 
                                     original_length - int(new_mid), dtype=np.float32)
            remap = np.concatenate([first_half, second_half])
        else:
            mid_point = transition_idx - start
            new_mid = mid_point + shift
            
            if new_mid >= original_length or new_mid <= 0:
                continue
            
            first_half = np.linspace(0, mid_point, max(1, int(new_mid)), dtype=np.float32)
            second_half = np.linspace(mid_point, original_length - 1,
                                     original_length - max(1, int(new_mid)), dtype=np.float32)
            remap = np.concatenate([first_half, second_half])
        
        if len(remap) == original_length:
            original_segment = ideal_pan[start:end]
            adjusted_pan[start:end] = np.interp(
                np.arange(original_length),
                remap,
                original_segment
            )
            transitions_snapped += 1
    
    # Log statistics
    total_transitions = len(sign_changes)
    print(f"[SPATIAL-SPEECH] {transitions_snapped}/{total_transitions} transitions snapped to breaks, {transitions_blocked} blocked during speech")
    
    # IMPORTANT: If too many transitions were blocked, the panning will be stuck/broken
    # Fall back to time-based panning which always works smoothly
    if total_transitions > 0 and transitions_blocked > total_transitions * 0.5:
        print(f"[SPATIAL-SPEECH] WARNING: {transitions_blocked}/{total_transitions} transitions blocked - falling back to time-based panning")
        return None  # Signal to use default time-based panning
    
    # Sanity check: make sure pan actually varies AND isn't heavily biased
    adjusted_pan = np.clip(adjusted_pan, -1.0, 1.0)
    pan_range = adjusted_pan.max() - adjusted_pan.min()
    pan_mean = adjusted_pan.mean()
    print(f"[SPATIAL-SPEECH] Pan output: range={pan_range:.3f}, mean={pan_mean:.3f}, min={adjusted_pan.min():.3f}, max={adjusted_pan.max():.3f}")
    
    # If pan is stuck (very small range), fall back to time-based
    if pan_range < 0.3:
        print(f"[SPATIAL-SPEECH] WARNING: Pan range too small ({pan_range:.3f}) - falling back to time-based panning")
        return None
    
    # If pan is heavily biased to one side (mean far from 0), fall back to time-based
    # This catches cases where panning varies but stays mostly on one side
    if abs(pan_mean) > 0.4:
        print(f"[SPATIAL-SPEECH] WARNING: Pan heavily biased (mean={pan_mean:.3f}) - falling back to time-based panning")
        return None
    
    return adjusted_pan


# =============================================================================
# ITD (Interaural Time Difference) - THE KEY TO REALISM
# =============================================================================

def apply_itd(
    left: np.ndarray,
    right: np.ndarray,
    pan: np.ndarray,
    sample_rate: int,
    max_itd_ms: float = 0.7,  # Human head ~0.6-0.7ms max
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Interaural Time Difference - the time delay between ears.
    
    This is THE most important cue for horizontal localization.
    When sound is to the right, it reaches the right ear ~0.7ms before the left.
    
    Uses linear interpolation for sub-sample accuracy (critical for realism).
    """
    n_samples = len(left)
    max_delay_samples = int(max_itd_ms * sample_rate / 1000)
    
    # Calculate delay in samples for each position
    # pan: -1 (left) to +1 (right)
    # When pan > 0 (right), delay left channel; when pan < 0 (left), delay right
    delay_samples = pan * max_delay_samples  # Fractional samples
    
    # Create output arrays with padding for delays
    left_out = np.zeros(n_samples, dtype=np.float32)
    right_out = np.zeros(n_samples, dtype=np.float32)
    
    # Process in larger chunks for efficiency (panning changes slowly)
    # 65536 samples = ~1.5 seconds at 44.1kHz - plenty of resolution for panning
    chunk_size = 65536
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    # Log progress for large files
    import time as _time
    _t_start = _time.perf_counter()
    _last_log = _t_start
    
    for i, start in enumerate(range(0, n_samples, chunk_size)):
        # Log progress every 25% for large files
        if n_chunks > 100 and i > 0 and i % (n_chunks // 4) == 0:
            pct = (i / n_chunks) * 100
            elapsed = _time.perf_counter() - _t_start
            print(f"[SPATIAL-ITD] {pct:.0f}% ({i}/{n_chunks} chunks, {elapsed:.1f}s elapsed)")
        
        end = min(start + chunk_size, n_samples)
        chunk_len = end - start
        
        # Average delay for this chunk
        avg_delay = np.mean(delay_samples[start:end])
        
        if abs(avg_delay) < 0.5:
            # Nearly centered - no delay needed
            left_out[start:end] = left[start:end]
            right_out[start:end] = right[start:end]
        elif avg_delay > 0:
            # Sound to right - delay left ear
            delay_int = int(avg_delay)
            delay_frac = avg_delay - delay_int
            
            # Source indices for left channel (delayed)
            src_start = max(0, start - delay_int - 1)
            src_end = min(n_samples, end - delay_int)
            
            if src_end > src_start:
                copy_len = min(src_end - src_start, end - start)
                # Linear interpolation between samples
                if delay_frac > 0 and src_start > 0:
                    src_a = left[src_start:src_start + copy_len]
                    src_b = left[max(0, src_start - 1):max(0, src_start - 1) + copy_len]
                    # Ensure same length
                    min_len = min(len(src_a), len(src_b), copy_len)
                    left_out[start:start + min_len] = (
                        src_a[:min_len] * (1 - delay_frac) +
                        src_b[:min_len] * delay_frac
                    )
                else:
                    left_out[start:start + copy_len] = left[src_start:src_start + copy_len]
            
            right_out[start:end] = right[start:end]
        else:
            # Sound to left - delay right ear
            delay_int = int(-avg_delay)
            delay_frac = -avg_delay - delay_int
            
            src_start = max(0, start - delay_int - 1)
            src_end = min(n_samples, end - delay_int)
            
            if src_end > src_start:
                copy_len = min(src_end - src_start, end - start)
                if delay_frac > 0 and src_start > 0:
                    src_a = right[src_start:src_start + copy_len]
                    src_b = right[max(0, src_start - 1):max(0, src_start - 1) + copy_len]
                    # Ensure same length
                    min_len = min(len(src_a), len(src_b), copy_len)
                    right_out[start:start + min_len] = (
                        src_a[:min_len] * (1 - delay_frac) +
                        src_b[:min_len] * delay_frac
                    )
                else:
                    right_out[start:start + copy_len] = right[src_start:src_start + copy_len]
            
            left_out[start:end] = left[start:end]
    
    return left_out, right_out


# =============================================================================
# Enhanced Head Shadow (HRTF-like frequency-dependent filtering)
# =============================================================================

def apply_enhanced_head_shadow(
    audio: np.ndarray,
    pan: np.ndarray,
    sample_rate: int,
    intensity: float = 0.5,
    num_bands: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply frequency-dependent head shadow effect (simplified HRTF).
    
    Real HRTF has complex frequency response. We approximate with multiple bands:
    - Low frequencies (< 500Hz): Minimal shadow (wavelength > head size)
    - Mid frequencies (500-4000Hz): Moderate shadow
    - High frequencies (> 4000Hz): Strong shadow (most affected by head)
    
    This is much more realistic than a single low-pass filter.
    """
    nyquist = sample_rate / 2
    
    if num_bands == 1:
        # Simple single-band (original behavior)
        cutoff = min(4000 / nyquist, 0.99)
        b, a = signal.butter(2, cutoff, btype='low')
        shadowed = signal.lfilter(b, a, audio).astype(np.float32)
        
        shadow_amount = np.abs(pan) * intensity
        left_shadow = np.where(pan > 0, shadow_amount, 0)
        right_shadow = np.where(pan < 0, shadow_amount, 0)
        
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        
        left_out = audio * left_gain * (1 - left_shadow) + shadowed * left_gain * left_shadow
        right_out = audio * right_gain * (1 - right_shadow) + shadowed * right_gain * right_shadow
        
        return left_out, right_out
    
    # Multi-band HRTF-like processing
    import time as _time
    _log = lambda msg: print(f"[SPATIAL-SHADOW] {msg}")
    bands = []
    shadow_amounts = []
    
    if num_bands >= 2:
        # Low band: < 500Hz - minimal shadow
        _t = _time.perf_counter()
        low_cut = min(500 / nyquist, 0.99)
        b_low, a_low = signal.butter(2, low_cut, btype='low')
        bands.append(signal.lfilter(b_low, a_low, audio).astype(np.float32))
        shadow_amounts.append(0.1 * intensity)  # Only 10% shadow
        _log(f"Low band filter: {(_time.perf_counter()-_t)*1000:.0f}ms")
        
        # High band: > 4000Hz - strong shadow
        _t = _time.perf_counter()
        high_cut = min(4000 / nyquist, 0.99)
        b_high, a_high = signal.butter(2, high_cut, btype='high')
        bands.append(signal.lfilter(b_high, a_high, audio).astype(np.float32))
        shadow_amounts.append(1.0 * intensity)  # Full shadow
        _log(f"High band filter: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    if num_bands >= 3:
        # Mid band: 500-4000Hz - moderate shadow
        _t = _time.perf_counter()
        mid_low = min(500 / nyquist, 0.99)
        mid_high = min(4000 / nyquist, 0.99)
        b_mid, a_mid = signal.butter(2, [mid_low, mid_high], btype='band')
        bands.append(signal.lfilter(b_mid, a_mid, audio).astype(np.float32))
        shadow_amounts.append(0.5 * intensity)  # 50% shadow
        _log(f"Mid band filter: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Calculate gains
    left_gain = np.sqrt(0.5 * (1 - pan))
    right_gain = np.sqrt(0.5 * (1 + pan))
    
    # Process each band with its shadow amount
    left_out = np.zeros_like(audio)
    right_out = np.zeros_like(audio)
    
    for band, shadow_mult in zip(bands, shadow_amounts):
        shadow_amount = np.abs(pan) * shadow_mult
        left_shadow = np.where(pan > 0, shadow_amount, 0)
        right_shadow = np.where(pan < 0, shadow_amount, 0)
        
        # For shadowed ear, reduce this band; for near ear, keep full
        left_out += band * left_gain * (1 - left_shadow * 0.7)
        right_out += band * right_gain * (1 - right_shadow * 0.7)
    
    return left_out.astype(np.float32), right_out.astype(np.float32)


# =============================================================================
# Near-Field Proximity Effect (ASMR "In Your Ear")
# =============================================================================

def apply_proximity_effect(
    audio: np.ndarray,
    distance: float,  # 0.0 = touching ear, 1.0 = normal distance
    sample_rate: int,
    intensity: float = 0.7,
) -> np.ndarray:
    """
    Apply proximity effect for close sounds (< 1 meter).
    
    Real close-mic/whisper characteristics:
    - Bass boost (proximity effect from directional mics, but also psychoacoustic)
    - Increased presence/detail
    - Slight compression (close sounds feel more "solid")
    
    Critical for ASMR "in your ear" sensation.
    """
    if distance >= 1.0 or intensity <= 0:
        return audio
    
    # Proximity factor: 1.0 when touching, 0.0 at normal distance
    proximity = (1.0 - distance) * intensity
    
    # Bass boost: +3-6dB at 100Hz when very close
    nyquist = sample_rate / 2
    bass_cut = min(200 / nyquist, 0.99)
    b_bass, a_bass = signal.butter(2, bass_cut, btype='low')
    bass = signal.lfilter(b_bass, a_bass, audio).astype(np.float32)
    
    # Add bass boost
    bass_boost = 1.0 + proximity * 0.8  # Up to 80% more bass
    output = audio + bass * (bass_boost - 1.0) * proximity
    
    # Presence boost: slight emphasis 2-5kHz
    pres_low = min(2000 / nyquist, 0.99)
    pres_high = min(5000 / nyquist, 0.99)
    b_pres, a_pres = signal.butter(2, [pres_low, pres_high], btype='band')
    presence = signal.lfilter(b_pres, a_pres, audio).astype(np.float32)
    
    presence_boost = proximity * 0.3  # Subtle presence
    output = output + presence * presence_boost
    
    return output.astype(np.float32)


# =============================================================================
# Natural Crossfeed (Inter-channel Bleed)
# =============================================================================

def apply_crossfeed(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    amount: float = 0.15,  # 15% bleed
    delay_ms: float = 0.3,  # Slight delay for bleed
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply natural crossfeed between channels.
    
    In real life, sound from your right still reaches your left ear (and vice versa).
    Headphones isolate completely, which sounds unnatural. Adding subtle crossfeed
    with delay and high-frequency rolloff mimics natural acoustics.
    """
    if amount <= 0:
        return left, right
    
    # Delay for crossfeed (simulates sound traveling around head)
    delay_samples = int(delay_ms * sample_rate / 1000)
    
    # High-frequency rolloff for crossfeed (head blocks highs)
    nyquist = sample_rate / 2
    cutoff = min(3000 / nyquist, 0.99)
    b, a = signal.butter(2, cutoff, btype='low')
    
    # Create filtered, delayed crossfeed signals
    left_to_right = signal.lfilter(b, a, left).astype(np.float32)
    right_to_left = signal.lfilter(b, a, right).astype(np.float32)
    
    # Apply delay
    if delay_samples > 0:
        left_to_right = np.concatenate([np.zeros(delay_samples), left_to_right[:-delay_samples]])
        right_to_left = np.concatenate([np.zeros(delay_samples), right_to_left[:-delay_samples]])
    
    # Mix
    left_out = left * (1 - amount * 0.5) + right_to_left * amount
    right_out = right * (1 - amount * 0.5) + left_to_right * amount
    
    return left_out.astype(np.float32), right_out.astype(np.float32)


# =============================================================================
# Micro-Movements (Organic Natural Variation)
# =============================================================================

def add_micro_movements(
    pan: np.ndarray,
    sample_rate: int,
    intensity: float = 0.02,  # ±2% variation
    speed_hz: float = 2.0,  # 2Hz micro-oscillation
) -> np.ndarray:
    """
    Add subtle organic micro-movements to panning.
    
    Real sounds (and real heads) have tiny natural movements. Perfectly smooth
    panning sounds artificial. Adding subtle random variation creates a more
    organic, living feel.
    """
    if intensity <= 0:
        return pan
    
    n_samples = len(pan)
    
    # For very large files, use a simpler/faster approach
    if n_samples > 10_000_000:  # > ~4 minutes at 44.1kHz
        # Simple sine-based micro-movements (no expensive gaussian filter)
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        micro = (
            np.sin(2 * np.pi * speed_hz * t) * 0.4 +
            np.sin(2 * np.pi * speed_hz * 1.7 * t + 0.5) * 0.35 +
            np.sin(2 * np.pi * speed_hz * 0.3 * t + 1.2) * 0.25
        )
        micro = micro / np.abs(micro).max()
        return np.clip(pan + micro * intensity, -1.0, 1.0).astype(np.float32)
    
    # Original approach for smaller files (with gaussian smoothing)
    t = np.arange(n_samples) / sample_rate
    
    # Multiple layered micro-movements at different frequencies
    micro = (
        np.sin(2 * np.pi * speed_hz * t) * 0.4 +
        np.sin(2 * np.pi * speed_hz * 1.7 * t + 0.5) * 0.3 +
        np.sin(2 * np.pi * speed_hz * 0.3 * t + 1.2) * 0.3
    )
    
    # Smooth random variation (only for smaller files)
    noise = np.random.randn(n_samples // 100 + 1) * 0.5
    noise_upsampled = np.interp(
        np.arange(n_samples),
        np.linspace(0, n_samples, len(noise)),
        noise
    )
    noise_smooth = gaussian_filter1d(noise_upsampled, sigma=sample_rate // 50)
    
    micro = micro + noise_smooth
    micro = micro / np.abs(micro).max()  # Normalize
    
    return np.clip(pan + micro * intensity, -1.0, 1.0).astype(np.float32)


# =============================================================================
# Air Absorption (Distance-based HF rolloff)
# =============================================================================

def apply_air_absorption(
    audio: np.ndarray,
    distance: float,  # 0.0 = close, 1.0 = far
    sample_rate: int,
) -> np.ndarray:
    """
    Apply air absorption effect - high frequencies attenuate with distance.
    
    Sound traveling through air loses high frequencies. Close sounds are bright
    and detailed; distant sounds are darker and less defined.
    """
    if distance <= 0.1:
        return audio
    
    # Cutoff frequency decreases with distance
    # Close: 16kHz, Far: 4kHz
    base_cutoff = 16000 - distance * 12000
    cutoff = max(2000, min(16000, base_cutoff))
    
    nyquist = sample_rate / 2
    normalized = min(cutoff / nyquist, 0.99)
    
    b, a = signal.butter(2, normalized, btype='low')
    return signal.lfilter(b, a, audio).astype(np.float32)


# =============================================================================
# Main Dynamic Panning Function (Enhanced)
# =============================================================================

def apply_dynamic_panning(
    audio: np.ndarray,
    sample_rate: int = 44100,
    speed_hz: float = 0.1,
    start_angle: float = -90.0,
    end_angle: float = 90.0,
    mode: str = "sweep",  # "sweep", "rotate", "extreme"
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
    block_size: int = 1024,  # Kept for API compatibility
    # New enhanced parameters
    quality: str = "balanced",  # "fast", "balanced", "ultra"
    distance: float = 0.5,  # 0.0 = touching ear, 1.0 = normal distance
    itd_enabled: Optional[bool] = None,
    proximity_enabled: Optional[bool] = None,
    crossfeed_enabled: Optional[bool] = None,
    micro_movements: Optional[bool] = None,
    crossfeed_amount: float = 0.05,  # Reduced from 0.12 - preserves ear-to-ear separation
    speech_aware: bool = True,  # Snap transitions to speech breaks
    time_offset: float = 0.0,  # For streaming: time offset from previous chunks (seconds)
) -> np.ndarray:
    """
    Apply dynamic spatial panning to audio with smooth per-sample transitions.
    
    ENHANCED VERSION with:
    - ITD (Interaural Time Difference) for realistic localization
    - Multi-band head shadow (HRTF-like)
    - Near-field proximity effect for ASMR
    - Natural crossfeed
    - Organic micro-movements
    - Speech-aware transitions (snaps ear-to-ear changes to natural pauses)
    
    Args:
        audio: Mono audio signal
        sample_rate: Audio sample rate
        speed_hz: Panning speed in Hz (cycles per second)
        start_angle: Starting angle for sweep mode (-90 = left, 90 = right)
        end_angle: Ending angle for sweep mode  
        mode: "sweep" (linear), "rotate" (sine), "extreme" (skip center)
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        quality: Preset - "fast", "balanced", or "ultra"
        distance: Source distance (0 = touching ear, 1 = far)
        itd_enabled: Override preset ITD setting
        proximity_enabled: Override preset proximity setting
        crossfeed_enabled: Override preset crossfeed setting
        micro_movements: Override preset micro-movement setting
        crossfeed_amount: Crossfeed intensity (0-0.3 recommended)
        speech_aware: Snap pan transitions to speech breaks for natural feel
        
    Returns:
        Stereo audio with immersive spatial positioning
    """
    # Get preset settings
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    
    # Allow parameter overrides
    use_itd = itd_enabled if itd_enabled is not None else preset["itd_enabled"]
    use_proximity = proximity_enabled if proximity_enabled is not None else preset["proximity_enabled"]
    use_crossfeed = crossfeed_enabled if crossfeed_enabled is not None else preset["crossfeed_enabled"]
    use_micro = micro_movements if micro_movements is not None else preset["micro_movements"]
    use_air = preset.get("air_absorption", False)
    shadow_bands = preset.get("head_shadow_bands", 2)
    
    # Ensure mono and float32
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    n_samples = len(audio)
    import time as _time
    _log = lambda msg: print(f"[SPATIAL] {msg}")
    
    # Apply proximity effect (before spatialization)
    if use_proximity and distance < 1.0:
        _t = _time.perf_counter()
        audio = apply_proximity_effect(audio, distance, sample_rate, intensity=0.7)
        _log(f"Proximity effect: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Apply air absorption if far
    if use_air and distance > 0.3:
        _t = _time.perf_counter()
        audio = apply_air_absorption(audio, distance, sample_rate)
        _log(f"Air absorption: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Create panning curve - either speech-aware or time-based
    _t = _time.perf_counter()
    pan = None
    
    # For streaming: time_offset maintains phase continuity across chunks
    # Without this, each chunk would reset panning to the start position
    if time_offset > 0:
        _log(f"Using time_offset={time_offset:.2f}s for streaming continuity")
    
    # Try speech-aware panning first (snaps transitions to natural breaks)
    # Note: Speech-aware is disabled for streaming (time_offset > 0) since it
    # analyzes the whole audio and doesn't work well with chunked processing
    if speech_aware and mode in ("extreme", "sweep", "rotate") and time_offset == 0:
        pan = create_speech_aware_pan(
            audio, sample_rate, speed_hz,
            start_angle, end_angle, mode,
            break_snap_window_ms=500.0
        )
        if pan is not None:
            _log(f"Speech-aware pan: {(_time.perf_counter()-_t)*1000:.0f}ms (transitions synced to breaks)")
    
    # Fall back to time-based panning if speech-aware didn't work or streaming mode
    if pan is None:
        # Add time_offset for streaming continuity - this is crucial!
        # Without it, each streaming chunk restarts the panning from the beginning
        t = np.arange(n_samples, dtype=np.float32) / sample_rate + time_offset
        
        # Calculate angle over time based on mode
        angle_range = end_angle - start_angle
        mid_angle = (start_angle + end_angle) / 2
        
        if mode == "extreme":
            # EXTREME MODE: Dwell at arc extremes, quick transition through center
            wave = np.sin(2 * np.pi * speed_hz * t)
            extreme_wave = np.tanh(wave * 15) * 1.01
            extreme_wave = np.clip(extreme_wave, -1.0, 1.0)
            angles = mid_angle + (angle_range / 2) * extreme_wave
        elif mode == "sweep":
            # Triangle wave (ping-pong)
            wave = np.sin(2 * np.pi * speed_hz * t)
            triangle = (2 / np.pi) * np.arcsin(wave)
            angles = mid_angle + (angle_range / 2) * triangle
        else:  # rotate
            # Smooth sine wave oscillation
            wave = np.sin(2 * np.pi * speed_hz * t)
            angles = mid_angle + (angle_range / 2) * wave
        
        # Convert angles to pan position
        pan = np.sin(np.radians(angles))
        pan = np.clip(pan, -1.0, 1.0)
        _log(f"Time-based pan: {(_time.perf_counter()-_t)*1000:.0f}ms" + (f" (offset={time_offset:.2f}s)" if time_offset > 0 else ""))
    
    # Add micro-movements for organic feel
    if use_micro:
        _t = _time.perf_counter()
        pan = add_micro_movements(pan, sample_rate, intensity=0.015, speed_hz=1.5)
        _log(f"Micro-movements: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Apply enhanced head shadow (HRTF-like)
    _t = _time.perf_counter()
    if head_shadow and head_shadow_intensity > 0:
        _log(f"Starting head shadow ({shadow_bands} bands)...")
        left_out, right_out = apply_enhanced_head_shadow(
            audio, pan, sample_rate,
            intensity=head_shadow_intensity,
            num_bands=shadow_bands
        )
    else:
        # Basic equal-power panning
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        
        # Headphone compensation
        compensation = 1.0 + 0.68 * (pan ** 2)
        left_gain *= compensation
        right_gain *= compensation
        
        left_out = audio * left_gain
        right_out = audio * right_gain
    _log(f"Head shadow/panning: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Apply ITD (THE KEY TO REALISM)
    if use_itd:
        _t = _time.perf_counter()
        _log("Starting ITD...")
        left_out, right_out = apply_itd(left_out, right_out, pan, sample_rate)
        _log(f"ITD: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Apply natural crossfeed
    if use_crossfeed and crossfeed_amount > 0:
        _t = _time.perf_counter()
        _log("Starting crossfeed...")
        left_out, right_out = apply_crossfeed(
            left_out, right_out, sample_rate,
            amount=crossfeed_amount
        )
        _log(f"Crossfeed: {(_time.perf_counter()-_t)*1000:.0f}ms")
    
    # Stack to stereo
    stereo = np.column_stack([left_out, right_out])
    
    # Soft clip to prevent harsh clipping
    max_val = np.abs(stereo).max()
    if max_val > 0.95:
        stereo = np.tanh(stereo * (1.0 / max_val) * 1.5) * 0.95
    
    return stereo.astype(np.float32)


def apply_static_position(
    audio: np.ndarray,
    angle: float,
    sample_rate: int = 44100,
    head_shadow: bool = True,
    head_shadow_intensity: float = 0.4,
    # Enhanced parameters
    distance: float = 0.5,
    quality: str = "balanced",
    itd_enabled: Optional[bool] = None,
    crossfeed_enabled: Optional[bool] = None,
) -> np.ndarray:
    """
    Apply static spatial positioning to audio (enhanced version).
    
    Args:
        audio: Mono audio signal
        angle: Position angle (-90 = left, 0 = center, 90 = right)
        sample_rate: Audio sample rate
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        distance: Source distance (0 = touching ear, 1 = far)
        quality: Preset - "fast", "balanced", or "ultra"
        
    Returns:
        Stereo audio with immersive positioning
    """
    # Get preset settings
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    use_itd = itd_enabled if itd_enabled is not None else preset["itd_enabled"]
    use_crossfeed = crossfeed_enabled if crossfeed_enabled is not None else preset["crossfeed_enabled"]
    use_proximity = preset.get("proximity_enabled", False)
    shadow_bands = preset.get("head_shadow_bands", 2)
    
    # Ensure mono and float32
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    # Apply proximity effect if close
    if use_proximity and distance < 1.0:
        audio = apply_proximity_effect(audio, distance, sample_rate, intensity=0.7)
    
    # Convert angle to pan position
    pan_value = np.clip(angle / 90.0, -1.0, 1.0)
    pan = np.full(len(audio), pan_value, dtype=np.float32)
    
    # Apply enhanced head shadow
    if head_shadow and head_shadow_intensity > 0 and abs(pan_value) > 0.1:
        left_out, right_out = apply_enhanced_head_shadow(
            audio, pan, sample_rate,
            intensity=head_shadow_intensity,
            num_bands=shadow_bands
        )
    else:
        # Basic equal-power panning with headphone compensation
        left_gain = np.sqrt(0.5 * (1 - pan_value))
        right_gain = np.sqrt(0.5 * (1 + pan_value))
        compensation = 1.0 + 0.68 * (pan_value ** 2)
        
        left_out = audio * left_gain * compensation
        right_out = audio * right_gain * compensation
    
    # Apply ITD for static position
    if use_itd and abs(pan_value) > 0.1:
        left_out, right_out = apply_itd(left_out, right_out, pan, sample_rate)
    
    # Apply crossfeed (subtle - preserves ear-to-ear separation)
    if use_crossfeed:
        left_out, right_out = apply_crossfeed(left_out, right_out, sample_rate, amount=0.05)
    
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
    # Enhanced parameters
    quality: str = "balanced",
    distance: float = 0.5,
) -> str:
    """
    Process an audio file with immersive spatial panning effects.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        mode: "sweep", "rotate", "extreme", or "static"
        speed_hz: Panning speed (cycles per second)
        start_angle: Start/static angle (-90 = left, 90 = right)
        end_angle: End angle for sweep mode
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        quality: "fast", "balanced", or "ultra"
        distance: Source distance (0 = touching ear, 1 = far)
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
        stereo = apply_static_position(
            audio, start_angle, sample_rate, head_shadow, head_shadow_intensity,
            distance=distance, quality=quality
        )
    else:
        stereo = apply_dynamic_panning(
            audio, sample_rate, speed_hz, start_angle, end_angle, mode,
            head_shadow, head_shadow_intensity,
            quality=quality, distance=distance
        )
    
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
    # Enhanced parameters
    quality: str = "balanced",
    distance: float = 0.5,
    itd_enabled: Optional[bool] = None,
    proximity_enabled: Optional[bool] = None,
    crossfeed_enabled: Optional[bool] = None,
    micro_movements: Optional[bool] = None,
    speech_aware: bool = True,  # Snap transitions to speech breaks
    time_offset: float = 0.0,  # For streaming: accumulated time from previous chunks
) -> np.ndarray:
    """
    Process audio buffer with immersive spatial panning effects.
    
    ENHANCED VERSION for ultra-realistic 8D/16D ASMR audio.
    
    Args:
        audio: Input audio (mono or stereo, any dtype)
        sample_rate: Sample rate
        mode: "sweep", "rotate", "extreme", or "static"
        speed_hz: Panning speed (for dynamic modes)
        start_angle: Start angle (-90 = left, 0 = center, 90 = right)
        end_angle: End angle (for sweep mode)
        head_shadow: Enable head shadow effect
        head_shadow_intensity: Head shadow strength (0-1)
        quality: "fast" (streaming), "balanced" (default), "ultra" (maximum immersion)
        distance: Source distance (0 = touching ear, 1 = far) - affects proximity effect
        itd_enabled: Override ITD setting from preset
        proximity_enabled: Override proximity effect setting
        crossfeed_enabled: Override crossfeed setting
        micro_movements: Override micro-movement setting
        speech_aware: Snap pan transitions to natural speech breaks (default: True)
        time_offset: For streaming - accumulated time from previous chunks (seconds)
        
    Returns:
        Processed stereo audio (float32, normalized) with immersive spatial positioning
        
    Quality Presets:
        - "fast": ITD only - good for real-time streaming
        - "balanced": ITD + proximity + crossfeed + micro-movements
        - "ultra": All effects + multi-band HRTF + air absorption
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
        return apply_static_position(
            audio, start_angle, sample_rate, head_shadow, head_shadow_intensity,
            distance=distance, quality=quality,
            itd_enabled=itd_enabled, crossfeed_enabled=crossfeed_enabled
        )
    else:
        return apply_dynamic_panning(
            audio, sample_rate, speed_hz, start_angle, end_angle, mode,
            head_shadow, head_shadow_intensity,
            quality=quality, distance=distance,
            itd_enabled=itd_enabled, proximity_enabled=proximity_enabled,
            crossfeed_enabled=crossfeed_enabled, micro_movements=micro_movements,
            speech_aware=speech_aware,
            time_offset=time_offset
        )

