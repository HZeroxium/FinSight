# routers/__init__.py

from .training import router as training_router
from .prediction import router as prediction_router

__all__ = [
    "training_router",
    "prediction_router",
    "models_router",
]
