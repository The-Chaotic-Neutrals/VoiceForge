"""
Parameter dataclasses for VoiceForge audio processing.

These are the canonical definitions used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RVCParams:
    """RVC voice conversion parameters."""
    pitch_algo: str = "rmvpe+"
    pitch_lvl: int = 0
    index_influence: float = 0.75
    respiration_median_filtering: int = 3
    envelope_ratio: float = 0.25
    consonant_breath_protection: float = 0.33
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RVC server."""
        return {
            "pitch_algo": self.pitch_algo,
            "pitch_lvl": self.pitch_lvl,
            "index_influence": self.index_influence,
            "respiration_median_filtering": self.respiration_median_filtering,
            "envelope_ratio": self.envelope_ratio,
            "consonant_breath_protection": self.consonant_breath_protection,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RVCParams":
        """Create from dictionary, ignoring unknown keys."""
        return cls(
            pitch_algo=d.get("pitch_algo", "rmvpe+"),
            pitch_lvl=d.get("pitch_lvl", d.get("pitch_level", 0)),
            index_influence=d.get("index_influence", 0.75),
            respiration_median_filtering=d.get("respiration_median_filtering", 3),
            envelope_ratio=d.get("envelope_ratio", 0.25),
            consonant_breath_protection=d.get("consonant_breath_protection", 0.33),
        )


@dataclass
class PostProcessParams:
    """Post-processing audio effects parameters. All effects OFF by default.
    
    NOTE: This must match exactly what the VoiceForge UI exposes.
    """
    # EQ (0 = disabled/passthrough)
    highpass: float = 0.0  # 0 = no highpass filter
    lowpass: float = 0.0   # 0 = no lowpass filter  
    bass_freq: float = 100.0  # frequency target (only applies if bass_gain != 0)
    bass_gain: float = 0.0   # 0 = no bass boost/cut
    treble_freq: float = 8000.0  # frequency target (only applies if treble_gain != 0)
    treble_gain: float = 0.0  # 0 = no treble boost/cut
    # Reverb (0 = disabled)
    reverb_delay: float = 0.0
    reverb_decay: float = 0.0
    # Effects (0 = disabled)
    crystalizer: float = 0.0
    deesser: float = 0.0
    # Spatial Audio (8D)
    audio_8d_enabled: bool = False
    audio_8d_mode: str = "rotate"  # "center", "extreme", "sweep", "rotate", "static", "static_right"
    audio_8d_speed: float = 0.1  # Hz
    audio_8d_depth: float = 180.0  # Arc in degrees (180 = L↔R, 360 = full circle)
    audio_8d_distance: float = 0.3  # 0 = in ear, 1 = far (controls proximity effect)
    audio_8d_quality: str = "balanced"  # "fast", "balanced", "ultra"
    audio_8d_itd: bool = True  # Interaural Time Difference
    audio_8d_proximity: bool = True  # Bass boost when close
    audio_8d_crossfeed: bool = True  # Natural bleed between ears
    audio_8d_micro_movements: bool = True  # Subtle organic variation
    audio_8d_speech_aware: bool = True  # Transitions at speech pauses
    # Pitch Shift
    pitch_shift_enabled: bool = False
    pitch_shift_semitones: int = 0
    # ASMR Enhancement
    asmr_enabled: bool = False
    asmr_tingles: int = 60  # 0-100: 2-8kHz "tingle zone" enhancement
    asmr_breathiness: int = 65  # 0-100: high freq air/breath sounds
    asmr_crispness: int = 55  # 0-100: mouth sounds, consonants, crisp detail

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for post-process server."""
        return {
            "highpass": self.highpass,
            "lowpass": self.lowpass,
            "bass_freq": self.bass_freq,
            "bass_gain": self.bass_gain,
            "treble_freq": self.treble_freq,
            "treble_gain": self.treble_gain,
            "reverb_delay": self.reverb_delay,
            "reverb_decay": self.reverb_decay,
            "crystalizer": self.crystalizer,
            "deesser": self.deesser,
            "audio_8d_enabled": self.audio_8d_enabled,
            "audio_8d_mode": self.audio_8d_mode,
            "audio_8d_speed": self.audio_8d_speed,
            "audio_8d_depth": self.audio_8d_depth,
            "audio_8d_distance": self.audio_8d_distance,
            "audio_8d_quality": self.audio_8d_quality,
            "audio_8d_itd": self.audio_8d_itd,
            "audio_8d_proximity": self.audio_8d_proximity,
            "audio_8d_crossfeed": self.audio_8d_crossfeed,
            "audio_8d_micro_movements": self.audio_8d_micro_movements,
            "audio_8d_speech_aware": self.audio_8d_speech_aware,
            "pitch_shift_enabled": self.pitch_shift_enabled,
            "pitch_shift_semitones": self.pitch_shift_semitones,
            "asmr_enabled": self.asmr_enabled,
            "asmr_tingles": self.asmr_tingles,
            "asmr_breathiness": self.asmr_breathiness,
            "asmr_crispness": self.asmr_crispness,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PostProcessParams":
        """Create from dictionary, using defaults for missing keys."""
        defaults = cls()
        return cls(
            highpass=d.get("highpass", defaults.highpass),
            lowpass=d.get("lowpass", defaults.lowpass),
            bass_freq=d.get("bass_freq", defaults.bass_freq),
            bass_gain=d.get("bass_gain", defaults.bass_gain),
            treble_freq=d.get("treble_freq", defaults.treble_freq),
            treble_gain=d.get("treble_gain", defaults.treble_gain),
            reverb_delay=d.get("reverb_delay", defaults.reverb_delay),
            reverb_decay=d.get("reverb_decay", defaults.reverb_decay),
            crystalizer=d.get("crystalizer", defaults.crystalizer),
            deesser=d.get("deesser", defaults.deesser),
            audio_8d_enabled=d.get("audio_8d_enabled", defaults.audio_8d_enabled),
            audio_8d_mode=d.get("audio_8d_mode", defaults.audio_8d_mode),
            audio_8d_speed=d.get("audio_8d_speed", defaults.audio_8d_speed),
            audio_8d_depth=d.get("audio_8d_depth", defaults.audio_8d_depth),
            audio_8d_distance=d.get("audio_8d_distance", defaults.audio_8d_distance),
            audio_8d_quality=d.get("audio_8d_quality", defaults.audio_8d_quality),
            audio_8d_itd=d.get("audio_8d_itd", defaults.audio_8d_itd),
            audio_8d_proximity=d.get("audio_8d_proximity", defaults.audio_8d_proximity),
            audio_8d_crossfeed=d.get("audio_8d_crossfeed", defaults.audio_8d_crossfeed),
            audio_8d_micro_movements=d.get("audio_8d_micro_movements", defaults.audio_8d_micro_movements),
            audio_8d_speech_aware=d.get("audio_8d_speech_aware", defaults.audio_8d_speech_aware),
            pitch_shift_enabled=d.get("pitch_shift_enabled", defaults.pitch_shift_enabled),
            pitch_shift_semitones=d.get("pitch_shift_semitones", defaults.pitch_shift_semitones),
            asmr_enabled=d.get("asmr_enabled", defaults.asmr_enabled),
            asmr_tingles=d.get("asmr_tingles", defaults.asmr_tingles),
            asmr_breathiness=d.get("asmr_breathiness", defaults.asmr_breathiness),
            asmr_crispness=d.get("asmr_crispness", defaults.asmr_crispness),
        )
    
    def needs_processing(self) -> bool:
        """
        Check if any post-processing effects are actually enabled.
        Returns False if all settings are at their "passthrough" values.
        """
        # Check major effect toggles first (fast path)
        if self.audio_8d_enabled:
            return True
        if self.pitch_shift_enabled and self.pitch_shift_semitones != 0:
            return True
        if self.asmr_enabled:
            return True
        
        # Check EQ effects
        if self.highpass > 0:
            return True
        if self.lowpass > 0:
            return True
        if self.bass_gain != 0:
            return True
        if self.treble_gain != 0:
            return True
        
        # Check reverb (both delay and decay must be > 0)
        if self.reverb_delay > 0 and self.reverb_decay > 0:
            return True
        
        # Check enhancement effects
        if self.crystalizer > 0:
            return True
        if self.deesser > 0:
            return True
        
        # Nothing enabled - can skip post-processing entirely
        return False


@dataclass
class BackgroundParams:
    """Background audio blending parameters."""
    enabled: bool = False
    files: List[str] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    delays: List[float] = field(default_factory=list)
    fade_ins: List[float] = field(default_factory=list)
    fade_outs: List[float] = field(default_factory=list)
    main_volume: float = 1.0
    use_config_tracks: bool = False  # Use bg_tracks from config.json
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "files": self.files,
            "volumes": self.volumes,
            "delays": self.delays,
            "fade_ins": self.fade_ins,
            "fade_outs": self.fade_outs,
            "main_volume": self.main_volume,
            "use_config_tracks": self.use_config_tracks,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackgroundParams":
        """Create from dictionary."""
        return cls(
            enabled=d.get("enabled", d.get("enable_background", False)),
            files=d.get("files", d.get("bg_files", [])),
            volumes=d.get("volumes", d.get("bg_volumes", [])),
            delays=d.get("delays", d.get("bg_delays", [])),
            fade_ins=d.get("fade_ins", d.get("bg_fade_ins", [])),
            fade_outs=d.get("fade_outs", d.get("bg_fade_outs", [])),
            main_volume=d.get("main_volume", d.get("main_audio_volume", 1.0)),
            use_config_tracks=d.get("use_config_tracks", d.get("use_config_bg_tracks", False)),
        )


# Convenience functions
def get_default_rvc_params() -> RVCParams:
    """Get default RVC parameters."""
    return RVCParams()


def get_default_rvc_params_dict() -> Dict[str, Any]:
    """Get default RVC parameters as dictionary."""
    return RVCParams().to_dict()


def get_default_post_params() -> PostProcessParams:
    """Get default post-processing parameters."""
    return PostProcessParams()


def get_default_post_params_dict() -> Dict[str, Any]:
    """Get default post-processing parameters as dictionary."""
    return PostProcessParams().to_dict()
