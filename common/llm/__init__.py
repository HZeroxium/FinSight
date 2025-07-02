"""LLM module for FinSight - Simplified LLM integration with multiple providers"""

from .llm_interfaces import (
    LLMInterface,
    LLMProvider,
    GenerationConfig,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from .llm_factory import LLMFactory, StrategyType
from .llm_facade import LLMFacade, create_llm_and_execute
from .di_container import configure_container, get_llm, get_facade


# Convenience functions for quick usage
def create_openai_llm(api_key: str = None, model: str = "gpt-4o-mini", **kwargs):
    """Create OpenAI LLM quickly"""
    return LLMFactory.create_openai_llm(api_key=api_key, model=model, **kwargs)


def generate_text(
    prompt: str, provider: str = "openai", model: str = "gpt-4o-mini", **kwargs
) -> str:
    """Quick text generation"""
    provider_enum = LLMProvider(provider.lower())
    return create_llm_and_execute(prompt, provider=provider_enum, model=model, **kwargs)


def generate_structured(
    prompt: str, schema, provider: str = "openai", model: str = "gpt-4o-mini", **kwargs
):
    """Quick structured generation"""
    provider_enum = LLMProvider(provider.lower())
    return create_llm_and_execute(
        prompt, output_schema=schema, provider=provider_enum, model=model, **kwargs
    )


__all__ = [
    "LLMInterface",
    "LLMProvider",
    "GenerationConfig",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMFactory",
    "StrategyType",
    "LLMFacade",
    "create_llm_and_execute",
    "configure_container",
    "get_llm",
    "get_facade",
    "create_openai_llm",
    "generate_text",
    "generate_structured",
]
