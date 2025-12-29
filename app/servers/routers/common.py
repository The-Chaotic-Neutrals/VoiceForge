"""
Common utilities shared across routers.

Authentication, constants, and shared dependencies.
"""

import os
import secrets
import sys
from typing import Optional

from fastapi import HTTPException, Header, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPBearer, HTTPAuthorizationCredentials

# Add app directory to path for all routers
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from config import MODEL_DIR, OUTPUT_DIR, ASSETS_DIR, FX_DIR, SOUNDS_DIR, APP_DIR, SCRIPT_DIR


# Authentication configuration
API_KEY = os.getenv("API_KEY")
BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER")
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS")

# Default values
DEFAULT_MODEL = os.getenv("DEFAULT_RVC_MODEL", "Goddess_Nicole")

# Security
security_basic = HTTPBasic(auto_error=False)
security_bearer = HTTPBearer(auto_error=False)


async def verify_auth(
    authorization: Optional[str] = Header(default=None),
    basic_credentials: Optional[HTTPBasicCredentials] = Depends(security_basic),
    bearer_credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
):
    """
    Verify authentication if configured.
    
    Supports:
    - Bearer token (API_KEY env var)
    - Basic auth (BASIC_AUTH_USER/BASIC_AUTH_PASS env vars)
    - No auth (if neither is configured)
    """
    # If no auth is configured, allow all requests
    if not API_KEY and not BASIC_AUTH_USER:
        return True
    
    # Check Bearer token
    if API_KEY:
        if bearer_credentials and secrets.compare_digest(bearer_credentials.credentials, API_KEY):
            return True
        if authorization:
            if authorization.startswith("Bearer "):
                token = authorization[7:]
                if secrets.compare_digest(token, API_KEY):
                    return True
            elif secrets.compare_digest(authorization, API_KEY):
                return True
    
    # Check Basic auth
    if BASIC_AUTH_USER and BASIC_AUTH_PASS and basic_credentials:
        if (secrets.compare_digest(basic_credentials.username, BASIC_AUTH_USER) and
            secrets.compare_digest(basic_credentials.password, BASIC_AUTH_PASS)):
            return True
    
    # If auth is required but not provided, raise 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer" if API_KEY else "Basic"},
    )


# Re-export common paths for router use
__all__ = [
    "verify_auth",
    "DEFAULT_MODEL",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "ASSETS_DIR",
    "FX_DIR",
    "SOUNDS_DIR",
    "APP_DIR",
    "SCRIPT_DIR",
]
