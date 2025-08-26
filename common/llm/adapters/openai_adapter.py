# common/llm/adapters/openai_adapter.py

import json
import time
from typing import List, Optional, Type

from pydantic import BaseModel, Field

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


from logger import LoggerFactory, LoggerType, LogLevel

from ..llm_interfaces import (
    GenerationConfig,
    LLMAdapterInterface,
    LLMMessage,
    LLMModel,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStats,
)

# Create logger for OpenAI adapter
logger = LoggerFactory.get_logger(
    name="openai-adapter", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI adapter"""

    api_key: Optional[str] = Field(None, description="OpenAI API key")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    default_model: str = Field(
        default=LLMModel.GPT_4O_MINI.value, description="Default model to use"
    )


class OpenAIAdapter(LLMAdapterInterface):
    """OpenAI adapter implementation"""

    def __init__(self, config: OpenAIConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        self.config = config
        self.client = self._create_client()
        self.stats = LLMStats()

        logger.info(f"Initialized OpenAI adapter with model: {config.default_model}")

    def _create_client(self) -> "OpenAI":
        """Create OpenAI client"""
        kwargs = {}

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.organization:
            kwargs["organization"] = self.config.organization
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.timeout:
            kwargs["timeout"] = self.config.timeout
        if self.config.max_retries:
            kwargs["max_retries"] = self.config.max_retries

        return OpenAI(**kwargs)

    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        return LLMProvider.OPENAI

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported"""
        supported_models = [m.value for m in LLMModel]
        return model in supported_models

    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return [m.value for m in LLMModel]

    def is_available(self) -> bool:
        """Check if OpenAI service is available"""
        try:
            # Simple test request
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI service not available: {e}")
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI"""
        start_time = time.time()

        try:
            # Prepare messages
            messages = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]

            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})

            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "temperature": request.config.temperature,
                "top_p": request.config.top_p,
                "frequency_penalty": request.config.frequency_penalty,
                "presence_penalty": request.config.presence_penalty,
            }

            if request.config.max_tokens:
                params["max_tokens"] = request.config.max_tokens
            if request.config.stop:
                params["stop"] = request.config.stop
            if request.config.seed:
                params["seed"] = request.config.seed

            # Add structured output if schema provided
            if request.output_schema:
                params["response_format"] = {"type": "json_object"}
                # Add schema instruction to the last message
                schema_instruction = f"\n\nPlease respond with valid JSON that matches this schema: {request.output_schema.model_json_schema()}"
                if messages:
                    messages[-1]["content"] += schema_instruction

            logger.debug(f"Making OpenAI request with params: {params}")

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Process response
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Parse structured output if schema provided
            parsed_output = None
            if request.output_schema:
                try:
                    parsed_data = json.loads(content)
                    parsed_output = request.output_schema(**parsed_data)
                except Exception as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    raise ValueError(
                        f"Failed to parse response according to schema: {e}"
                    )

            response_time = time.time() - start_time

            # Record statistics
            self.stats.record_request(
                success=True,
                tokens_used=usage["total_tokens"],
                response_time=response_time,
            )

            logger.info(
                f"OpenAI generation successful. Tokens used: {usage['total_tokens']}"
            )

            return LLMResponse(
                content=content,
                model=request.model,
                usage=usage,
                metadata={"response_time": response_time},
                parsed_output=parsed_output,
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.stats.record_request(success=False, response_time=response_time)
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def generate_with_schema(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> BaseModel:
        """Generate structured output"""
        model = model or self.config.default_model
        config = config or GenerationConfig()

        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            model=model,
            config=config,
            output_schema=output_schema,
        )

        response = self.generate(request)

        if response.parsed_output is None:
            raise ValueError("Failed to generate structured output")

        return response.parsed_output

    def simple_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Simple text generation"""
        model = model or self.config.default_model
        config = config or GenerationConfig()

        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            model=model,
            config=config,
        )

        response = self.generate(request)
        return response.content

    def get_stats(self) -> LLMStats:
        """Get usage statistics"""
        return self.stats

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.stats = LLMStats()
        logger.info("OpenAI adapter statistics reset")
