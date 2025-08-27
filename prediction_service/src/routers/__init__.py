# routers/__init__.py

from .prediction import router as prediction_router
from .training import router as training_router

__all__ = [
    "training_router",
    "prediction_router",
]
