from typing import Optional, Type, Any, Dict, List
from pydantic import BaseModel

from .llm_interfaces import LLMInterface, GenerationConfig, LLMMessage, LLMRequest
from .llm_factory import LLMFactory, LLMProvider, StrategyType
from ..logger import LoggerFactory, LoggerType, LogLevel

# Create logger for facade
logger = LoggerFactory.get_logger(
    name="llm-facade", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class LLMFacade:
    """Simplified facade for LLM operations"""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        strategy: StrategyType = StrategyType.SIMPLE,
        default_model: str = "gpt-4o-mini",
        **config_kwargs,
    ):
        """
        Initialize LLM facade

        Args:
            provider: LLM provider to use
            strategy: Default generation strategy
            default_model: Default model to use
            **config_kwargs: Additional configuration parameters
        """
        self.provider = provider
        self.strategy = strategy
        self.default_model = default_model
        self.config_kwargs = config_kwargs

        # Create LLM instance
        self.llm = LLMFactory.get_llm(
            provider=provider,
            strategy=strategy,
            default_model=default_model,
            **config_kwargs,
        )

        logger.info(
            f"LLM facade initialized with {provider.value} provider and {strategy.value} strategy"
        )

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate simple text response

        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance default)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        logger.debug(f"Generating text with prompt length: {len(prompt)}")

        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k in GenerationConfig.model_fields},
        )

        result = self.llm.simple_generate(
            prompt=prompt, model=model or self.default_model, config=config
        )

        logger.info(f"Generated text response with {len(result)} characters")
        return result

    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.3,  # Lower temperature for structured output
        **kwargs,
    ) -> BaseModel:
        """
        Generate structured response using Pydantic schema

        Args:
            prompt: Input prompt
            output_schema: Pydantic model for output structure
            model: Model to use (defaults to instance default)
            temperature: Generation temperature
            **kwargs: Additional generation parameters

        Returns:
            Parsed Pydantic model instance
        """
        logger.debug(
            f"Generating structured output for schema: {output_schema.__name__}"
        )

        config = GenerationConfig(
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k in GenerationConfig.model_fields},
        )

        result = self.llm.generate_with_schema(
            prompt=prompt,
            output_schema=output_schema,
            model=model or self.default_model,
            config=config,
        )

        logger.info(f"Generated structured output: {output_schema.__name__}")
        return result

    def generate_with_context(
        self,
        prompt: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate response with additional context

        Args:
            prompt: Main prompt
            context: Additional context information
            model: Model to use
            temperature: Generation temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        logger.debug("Generating with context")

        messages = [
            LLMMessage(role="system", content=f"Context: {context}"),
            LLMMessage(role="user", content=prompt),
        ]

        config = GenerationConfig(
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k in GenerationConfig.model_fields},
        )

        request = LLMRequest(
            messages=messages, model=model or self.default_model, config=config
        )

        response = self.llm.generate(request)
        logger.info("Generated response with context")
        return response.content

    def generate_with_examples(
        self,
        prompt: str,
        examples: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.5,
        **kwargs,
    ) -> str:
        """
        Generate response with few-shot examples

        Args:
            prompt: Input prompt
            examples: List of example input/output pairs
            model: Model to use
            temperature: Generation temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        logger.debug(f"Generating with {len(examples)} examples")

        messages = []

        # Add examples as conversation history
        for example in examples:
            messages.append(LLMMessage(role="user", content=example.get("input", "")))
            messages.append(
                LLMMessage(role="assistant", content=example.get("output", ""))
            )

        # Add actual prompt
        messages.append(LLMMessage(role="user", content=prompt))

        config = GenerationConfig(
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k in GenerationConfig.model_fields},
        )

        request = LLMRequest(
            messages=messages, model=model or self.default_model, config=config
        )

        response = self.llm.generate(request)
        logger.info("Generated response with examples")
        return response.content

    def analyze_with_schema(
        self,
        text: str,
        analysis_schema: Type[BaseModel],
        analysis_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Analyze text and return structured analysis

        Args:
            text: Text to analyze
            analysis_schema: Schema for analysis output
            analysis_prompt: Custom analysis prompt
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Structured analysis result
        """
        default_prompt = f"Please analyze the following text and provide a structured analysis:\n\n{text}"
        prompt = analysis_prompt or default_prompt

        logger.debug(f"Analyzing text with schema: {analysis_schema.__name__}")

        return self.generate_structured(
            prompt=prompt, output_schema=analysis_schema, model=model, **kwargs
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self.llm.get_stats()
        return {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": stats.get_success_rate(),
            "total_tokens_used": stats.total_tokens_used,
            "total_cost": stats.total_cost,
            "average_response_time": stats.average_response_time,
            "uptime": stats.get_uptime(),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.llm.reset_stats()
        logger.info("Usage statistics reset")

    def is_healthy(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            return self.llm.is_available()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Convenience function for quick usage
def create_llm_and_execute(
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    provider: LLMProvider = LLMProvider.OPENAI,
    strategy: StrategyType = StrategyType.SIMPLE,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> Any:
    """
    Convenience function to create LLM and execute generation in one call

    Args:
        prompt: Input prompt
        output_schema: Optional Pydantic schema for structured output
        provider: LLM provider
        strategy: Generation strategy
        model: Model to use
        **kwargs: Additional configuration

    Returns:
        Generated response (str or Pydantic model based on schema)
    """
    logger.debug("Creating LLM and executing generation")

    facade = LLMFacade(
        provider=provider, strategy=strategy, default_model=model, **kwargs
    )

    if output_schema:
        result = facade.generate_structured(prompt, output_schema, model=model)
        logger.info(f"Generated structured output: {output_schema.__name__}")
        return result
    else:
        result = facade.generate_text(prompt, model=model)
        logger.info("Generated text output")
        return result
