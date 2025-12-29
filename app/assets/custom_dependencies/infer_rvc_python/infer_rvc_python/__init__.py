try:
    from .main import BaseLoader
except ImportError as e:
    # Re-raise the original import error - let Python handle it naturally
    # The error will show the actual missing dependency (e.g., faiss, torch, etc.)
    raise
