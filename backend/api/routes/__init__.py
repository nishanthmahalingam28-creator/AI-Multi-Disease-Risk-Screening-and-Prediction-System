"""API routes package initialization."""

from .health import health_bp
from .predictions import predictions_bp

__all__ = ["health_bp", "predictions_bp"]
