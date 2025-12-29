"""
UVR5 VR Audio Separators - Multiple model support
Based on Mangio-RVC-Fork uvr5 implementation.

Supported models:
- hp5_vocals: Isolate main vocals from music (HP5_only_main_vocal.pth)
- deecho_aggressive: Aggressively remove echo/delay (VR-DeEchoAggressive.pth)
- deecho_dereverb: Remove echo and reverb (VR-DeEchoDeReverb.pth)
- deecho_normal: Standard echo removal (VR-DeEchoNormal.pth)
"""
from .separator import UVR5Separator, HP5VocalSeparator, AVAILABLE_MODELS

__all__ = ['UVR5Separator', 'HP5VocalSeparator', 'AVAILABLE_MODELS']
