"""FastAPI backend for the PayLater credit scoring engine."""

from .service import get_service

__all__ = ["get_service"]
