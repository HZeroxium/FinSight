from typing import Optional, Dict, Any, Type
from enum import Enum
from pydantic import BaseModel, Field

from .llm_interfaces import (
    LLMInterface,
    LLMProvider,
    LLMAdapterInterface,
    LLMStrategyInterface,
)
from .adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
from .llm_strategies import (
    SimpleStrategy,
    RetryStrategy,
    FallbackStrategy,
    ValidationStrategy,
    ChainOfThoughtStrategy,
    CostOptimizedStrategy,
)
from ..logger import LoggerFactory, LoggerType, LogLevel

# Create logger for factory
logger = LoggerFactory.get_logger(
    name="llm-factory", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class StrategyType(Enum):
    """Available strategy types"""

    SIMPLE = "simple"
    RETRY = "retry"
    FALLBACK = "fallback"
    VALIDATION = "validation"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    COST_OPTIMIZED = "cost_optimized"


class LLMFactoryConfig(BaseModel):
    """Configuration for LLM factory"""

    default_provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    default_strategy: StrategyType = Field(default=StrategyType.SIMPLE)
    enable_caching: bool = Field(default=True)


class LLMFactory:
    """Factory for creating LLM instances"""

    _instances: Dict[str, LLMInterface] = {}
    _config: LLMFactoryConfig = LLMFactoryConfig()

    @classmethod
    def configure(cls, config: LLMFactoryConfig) -> None:
        """Configure factory settings"""
        cls._config = config
        logger.info(f"LLM factory configured with: {config}")

    @classmethod
    def get_llm(
        cls,
        provider: LLMProvider = None,
        strategy: StrategyType = None,
        name: str = "default",
        **kwargs: Any,
    ) -> LLMInterface:
        """
        Get or create LLM instance (cached if enabled)

        Args:
            provider: LLM provider to use
            strategy: Generation strategy to use
            name: Instance name for caching
            **kwargs: Additional configuration parameters

        Returns:
            LLM interface instance
        """
        provider = provider or cls._config.default_provider
        strategy = strategy or cls._config.default_strategy

        cache_key = f"{name}_{provider.value}_{strategy.value}"
        logger.debug(f"Requesting LLM instance: {cache_key}")

        if cls._config.enable_caching and cache_key in cls._instances:
            logger.debug(f"Returning cached LLM instance: {cache_key}")
            return cls._instances[cache_key]

        try:
            logger.debug(f"Creating new LLM instance: {cache_key}")
            adapter = cls._create_adapter(provider, **kwargs)
            strategy_instance = cls._create_strategy(strategy, **kwargs)

            # Create wrapped LLM instance
            llm_instance = StrategyWrappedLLM(adapter, strategy_instance)

            if cls._config.enable_caching:
                cls._instances[cache_key] = llm_instance
                logger.info(f"Created and cached LLM instance: {cache_key}")
            else:
                logger.info(f"Created LLM instance (no caching): {cache_key}")

            return llm_instance

        except Exception as e:
            logger.error(f"Failed to create LLM instance {cache_key}: {e}")
            raise

    @classmethod
    def create_llm(
        cls,
        provider: LLMProvider,
        strategy: StrategyType = StrategyType.SIMPLE,
        **kwargs: Any,
    ) -> LLMInterface:
        """
        Create new LLM instance (not cached)

        Args:
            provider: LLM provider to use
            strategy: Generation strategy to use
            **kwargs: Additional configuration parameters

        Returns:
            New LLM interface instance
        """
        logger.debug(f"Creating {provider.value} LLM with {strategy.value} strategy")

        try:
            adapter = cls._create_adapter(provider, **kwargs)
            strategy_instance = cls._create_strategy(strategy, **kwargs)

            llm_instance = StrategyWrappedLLM(adapter, strategy_instance)
            logger.info(
                f"Successfully created {provider.value} LLM with {strategy.value} strategy"
            )

            return llm_instance

        except Exception as e:
            logger.error(f"Failed to create {provider.value} LLM: {e}")
            raise

    @classmethod
    def _create_adapter(cls, provider: LLMProvider, **kwargs) -> LLMAdapterInterface:
        """Create adapter instance"""
        if provider == LLMProvider.OPENAI:
            return cls._create_openai_adapter(**kwargs)
        elif provider == LLMProvider.LANGCHAIN:
            raise NotImplementedError("Langchain adapter not implemented yet")
        elif provider == LLMProvider.GOOGLE_ADK:
            raise NotImplementedError("Google ADK adapter not implemented yet")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def _create_openai_adapter(cls, **kwargs) -> OpenAIAdapter:
        """Create OpenAI adapter with configuration"""
        # Filter OpenAI-specific parameters
        openai_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "api_key",
                "organization",
                "base_url",
                "timeout",
                "max_retries",
                "default_model",
            }
        }

        config = OpenAIConfig(**openai_params)
        return OpenAIAdapter(config)

    @classmethod
    def _create_strategy(
        cls, strategy_type: StrategyType, **kwargs
    ) -> LLMStrategyInterface:
        """Create strategy instance"""
        if strategy_type == StrategyType.SIMPLE:
            return SimpleStrategy()
        elif strategy_type == StrategyType.RETRY:
            return RetryStrategy(
                max_retries=kwargs.get("max_retries", 3),
                delay=kwargs.get("delay", 1.0),
                backoff_multiplier=kwargs.get("backoff_multiplier", 2.0),
            )
        elif strategy_type == StrategyType.FALLBACK:
            return FallbackStrategy(
                fallback_models=kwargs.get("fallback_models", ["gpt-3.5-turbo"])
            )
        elif strategy_type == StrategyType.VALIDATION:
            return ValidationStrategy(
                validation_prompt=kwargs.get("validation_prompt"),
                max_validation_retries=kwargs.get("max_validation_retries", 2),
            )
        elif strategy_type == StrategyType.CHAIN_OF_THOUGHT:
            return ChainOfThoughtStrategy(
                reasoning_prompt=kwargs.get(
                    "reasoning_prompt", "Let's think step by step."
                )
            )
        elif strategy_type == StrategyType.COST_OPTIMIZED:
            return CostOptimizedStrategy(
                cheap_model=kwargs.get("cheap_model", "gpt-3.5-turbo"),
                expensive_model=kwargs.get("expensive_model", "gpt-4"),
                complexity_threshold=kwargs.get("complexity_threshold", 500),
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @classmethod
    def create_openai_llm(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        strategy: StrategyType = StrategyType.SIMPLE,
        **kwargs,
    ) -> LLMInterface:
        """
        Convenience method to create OpenAI LLM

        Args:
            api_key: OpenAI API key
            model: Model to use
            strategy: Generation strategy
            **kwargs: Additional configuration

        Returns:
            OpenAI LLM instance
        """
        return cls.create_llm(
            provider=LLMProvider.OPENAI,
            strategy=strategy,
            api_key=api_key,
            default_model=model,
            **kwargs,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached instances"""
        logger.info(f"Clearing {len(cls._instances)} cached LLM instances")
        cls._instances.clear()

    @classmethod
    def get_cached_instances(cls) -> Dict[str, str]:
        """Get information about cached instances"""
        return {
            key: type(instance).__name__ for key, instance in cls._instances.items()
        }


class StrategyWrappedLLM(LLMInterface):
    """LLM wrapper that applies generation strategies"""

    def __init__(self, adapter: LLMAdapterInterface, strategy: LLMStrategyInterface):
        self.adapter = adapter
        self.strategy = strategy
        logger.debug(
            f"Created strategy-wrapped LLM: {adapter.get_provider().value} + {strategy.get_strategy_name()}"
        )

    def generate(self, request):
        """Generate using the wrapped strategy"""
        return self.strategy.execute(self.adapter, request)

    def generate_with_schema(self, prompt, output_schema, model=None, config=None):
        """Generate with schema using adapter"""
        return self.adapter.generate_with_schema(prompt, output_schema, model, config)

    def simple_generate(self, prompt, model=None, config=None):
        """Simple generate using adapter"""
        return self.adapter.simple_generate(prompt, model, config)

    def get_stats(self):
        """Get stats from adapter"""
        return self.adapter.get_stats()

    def reset_stats(self):
        """Reset stats on adapter"""
        return self.adapter.reset_stats()

    def is_available(self):
        """Check availability using adapter"""
        return self.adapter.is_available()

    def get_supported_models(self):
        """Get supported models from adapter"""
        return self.adapter.get_supported_models()
