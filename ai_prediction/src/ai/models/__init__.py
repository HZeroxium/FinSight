from .interface import ModelInterface
from .components import (
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    FinancialEmbedding,
)
from .transformer import FinancialTransformer, LightweightTransformer, HybridTransformer


# Model factory function
def create_model(model_type: str, config) -> ModelInterface:
    """
    Factory function to create models based on type

    Args:
        model_type: Type of model to create
        config: Model configuration

    Returns:
        ModelInterface: Created model instance
    """
    models = {
        "transformer": FinancialTransformer,
        "lightweight_transformer": LightweightTransformer,
        "hybrid_transformer": HybridTransformer,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](config)


__all__ = [
    # Interfaces and base classes
    "ModelInterface",
    # Components
    "PositionalEncoding",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "FinancialEmbedding",
    # Model implementations
    "FinancialTransformer",
    "LightweightTransformer",
    "HybridTransformer",
    # Factory function
    "create_model",
]
