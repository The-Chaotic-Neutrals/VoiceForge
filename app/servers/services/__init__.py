"""
VoiceForge Service Layer

Business logic separated from API routing.
"""

# Note: Imports are done lazily in functions to avoid circular imports
# and to allow routers to be imported first (they set up sys.path)

__all__ = [
    # TTS service
    "generate_tts_wav",
    "generate_chatterbox_wav",
    # Pipeline
    "generate_audio",
    "AudioPipelineResult",
]


def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name in ("generate_tts_wav", "generate_chatterbox_wav"):
        from .tts_service import generate_tts_wav, generate_chatterbox_wav
        return locals()[name]
    elif name in ("generate_audio", "AudioPipelineResult"):
        from .pipeline import generate_audio, AudioPipelineResult
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
