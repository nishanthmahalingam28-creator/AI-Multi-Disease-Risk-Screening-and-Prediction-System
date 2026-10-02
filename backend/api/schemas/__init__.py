"""API schemas package initialization."""

from .prediction import (
    AsthmaInputSchema,
    LungCancerInputSchema,
    ParkinsonsInputSchema,
    validate_payload_for_disease,
)

__all__ = [
    "LungCancerInputSchema",
    "AsthmaInputSchema",
    "ParkinsonsInputSchema",
    "validate_payload_for_disease",
]
