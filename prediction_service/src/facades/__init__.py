# facades/__init__.py

"""
Model Facades Package

This package provides specialized facades for model operations:
- ModelTrainingFacade: Specialized for training operations
- ModelServingFacade: Specialized for serving operations
- UnifiedModelFacade: Combined interface for both operations
- FacadeFactory: Factory for creating facade instances
"""

from .facade_factory import (FacadeFactory, FacadeType, get_default_facade,
                             get_enhanced_model_facade, get_model_facade,
                             get_serving_facade, get_training_facade,
                             get_unified_facade)
from .model_serving_facade import ModelServingFacade
from .model_training_facade import ModelTrainingFacade
from .unified_model_facade import UnifiedModelFacade

# Backward compatibility aliases
ModelFacade = UnifiedModelFacade
EnhancedModelFacade = UnifiedModelFacade

__all__ = [
    # Main facade classes
    "ModelTrainingFacade",
    "ModelServingFacade",
    "UnifiedModelFacade",
    # Factory
    "FacadeFactory",
    "FacadeType",
    # Convenience functions
    "get_training_facade",
    "get_serving_facade",
    "get_unified_facade",
    "get_default_facade",
    "get_model_facade",
    "get_enhanced_model_facade",
    # Backward compatibility
    "ModelFacade",
    "EnhancedModelFacade",
]
