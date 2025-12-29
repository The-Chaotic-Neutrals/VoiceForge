# models/__init__.py
"""
Model management for VoiceForge.

This package provides model loaders for UVR5 (vocal separation).
Other loaders exist but are accessed via microservice clients.
"""

# Only uvr5 is actually used via "from models import uvr5"
from . import uvr5_loader as uvr5

__all__ = [
    'uvr5',
]
