# common/llm/llm_strategies.py

import time
from typing import List, Optional

from .llm_interfaces import (
    GenerationConfig,
    LLMStrategyInterface,
    LLMInterface,
    LLMRequest,
    LLMResponse,
    LLMMessage,
)
from logger import LoggerFactory, LoggerType, LogLevel

# Create logger for strategies
logger = LoggerFactory.get_logger(
    name="llm-strategies", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class SimpleStrategy(LLMStrategyInterface):
    """Simple direct generation strategy"""

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute simple generation"""
        logger.debug("Executing simple strategy")
        return llm.generate(request)

    def get_strategy_name(self) -> str:
        return "simple"


class RetryStrategy(LLMStrategyInterface):
    """Strategy with retry logic"""

    def __init__(
        self, max_retries: int = 3, delay: float = 1.0, backoff_multiplier: float = 2.0
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_multiplier = backoff_multiplier

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute with retry logic"""
        logger.debug(f"Executing retry strategy with max_retries={self.max_retries}")

        last_exception = None
        current_delay = self.delay

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries}")
                    time.sleep(current_delay)
                    current_delay *= self.backoff_multiplier

                return llm.generate(request)

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries:
                    logger.error(f"All retry attempts failed. Last error: {e}")
                    break

        raise last_exception

    def get_strategy_name(self) -> str:
        return "retry"


class FallbackStrategy(LLMStrategyInterface):
    """Strategy with fallback models"""

    def __init__(self, fallback_models: List[str]):
        self.fallback_models = fallback_models

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute with fallback models"""
        logger.debug(
            f"Executing fallback strategy with models: {[request.model] + self.fallback_models}"
        )

        models_to_try = [request.model] + self.fallback_models
        last_exception = None

        for i, model in enumerate(models_to_try):
            try:
                if i > 0:
                    logger.info(f"Trying fallback model: {model}")

                # Create new request with fallback model
                fallback_request = LLMRequest(
                    messages=request.messages,
                    model=model,
                    config=request.config,
                    output_schema=request.output_schema,
                    system_prompt=request.system_prompt,
                )

                return llm.generate(fallback_request)

            except Exception as e:
                last_exception = e
                logger.warning(f"Model {model} failed: {e}")

        logger.error(f"All fallback models failed. Last error: {last_exception}")
        raise last_exception

    def get_strategy_name(self) -> str:
        return "fallback"


class ValidationStrategy(LLMStrategyInterface):
    """Strategy with output validation"""

    def __init__(
        self, validation_prompt: Optional[str] = None, max_validation_retries: int = 2
    ):
        self.validation_prompt = (
            validation_prompt
            or "Is the following response appropriate and accurate? Answer with just 'yes' or 'no':"
        )
        self.max_validation_retries = max_validation_retries

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute with output validation"""
        logger.debug("Executing validation strategy")

        for attempt in range(self.max_validation_retries + 1):
            response = llm.generate(request)

            if attempt == self.max_validation_retries:
                logger.warning(
                    "Max validation retries reached, returning last response"
                )
                return response

            # Validate response
            if self._validate_response(llm, response.content):
                logger.debug("Response validation passed")
                return response
            else:
                logger.info(
                    f"Response validation failed, retrying (attempt {attempt + 1})"
                )

        return response

    def _validate_response(self, llm: LLMInterface, content: str) -> bool:
        """Validate response content"""
        try:
            validation_request = LLMRequest(
                messages=[
                    LLMMessage(
                        role="user",
                        content=f"{self.validation_prompt}\n\nResponse: {content}",
                    )
                ],
                model="gpt-3.5-turbo",  # Use faster model for validation
                config=GenerationConfig(temperature=0.0, max_tokens=10),
            )

            validation_response = llm.generate(validation_request)
            return validation_response.content.strip().lower().startswith("yes")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return True  # Default to accepting if validation fails


class ChainOfThoughtStrategy(LLMStrategyInterface):
    """Strategy that encourages step-by-step reasoning"""

    def __init__(self, reasoning_prompt: str = "Let's think step by step."):
        self.reasoning_prompt = reasoning_prompt

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute with chain of thought prompting"""
        logger.debug("Executing chain of thought strategy")

        # Modify the last user message to include reasoning prompt
        modified_messages = request.messages.copy()
        if modified_messages and modified_messages[-1].role == "user":
            original_content = modified_messages[-1].content
            modified_messages[-1] = LLMMessage(
                role="user", content=f"{original_content}\n\n{self.reasoning_prompt}"
            )

        modified_request = LLMRequest(
            messages=modified_messages,
            model=request.model,
            config=request.config,
            output_schema=request.output_schema,
            system_prompt=request.system_prompt,
        )

        return llm.generate(modified_request)

    def get_strategy_name(self) -> str:
        return "chain_of_thought"


class CostOptimizedStrategy(LLMStrategyInterface):
    """Strategy that optimizes for cost by using cheaper models when possible"""

    def __init__(
        self,
        cheap_model: str = "gpt-3.5-turbo",
        expensive_model: str = "gpt-4",
        complexity_threshold: int = 500,
    ):
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.complexity_threshold = complexity_threshold

    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """Execute with cost optimization"""
        logger.debug("Executing cost optimized strategy")

        # Calculate prompt complexity (simple heuristic)
        total_content_length = sum(len(msg.content) for msg in request.messages)

        # Choose model based on complexity
        chosen_model = (
            self.expensive_model
            if total_content_length > self.complexity_threshold
            else self.cheap_model
        )

        if chosen_model != request.model:
            logger.info(
                f"Cost optimization: switching from {request.model} to {chosen_model}"
            )

        optimized_request = LLMRequest(
            messages=request.messages,
            model=chosen_model,
            config=request.config,
            output_schema=request.output_schema,
            system_prompt=request.system_prompt,
        )

        return llm.generate(optimized_request)

    def get_strategy_name(self) -> str:
        return "cost_optimized"
