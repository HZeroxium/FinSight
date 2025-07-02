import time
import json
from typing import List, Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from ..llm_interfaces import (
    LLMAdapterInterface,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStats,
    GenerationConfig,
    LLMMessage,
)
from ...logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="google-adk-adapter", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class GoogleConfig(BaseSettings):
    """Configuration for Google AI SDK adapter"""

    api_key: Optional[str] = Field(None, description="Google API key")
    default_model: str = Field(default="gemini-pro", description="Default model")
    timeout: int = Field(default=30, description="Request timeout")

    class Config:
        env_prefix = "GOOGLE_"


class GoogleADKAdapter(LLMAdapterInterface):
    """Google AI SDK adapter implementation"""

    def __init__(self, config: GoogleConfig):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google AI SDK not available. Install with: pip install google-generativeai"
            )

        self.config = config
        self.stats = LLMStats()

        # Configure API key
        if config.api_key:
            genai.configure(api_key=config.api_key)

        logger.info(
            f"Initialized Google AI SDK adapter with model: {config.default_model}"
        )

    def get_provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE_ADK

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported"""
        supported = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        return model in supported

    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"]

    def is_available(self) -> bool:
        """Check if Google AI service is available"""
        try:
            model = genai.GenerativeModel(self.config.default_model)
            model.generate_content(
                "test",
                generation_config=genai.types.GenerationConfig(max_output_tokens=1),
            )
            return True
        except Exception as e:
            logger.warning(f"Google AI service not available: {e}")
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google AI SDK"""
        start_time = time.time()

        try:
            model = genai.GenerativeModel(request.model)

            # Prepare content
            content_parts = []

            # Add system prompt if provided
            if request.system_prompt:
                content_parts.append(f"System: {request.system_prompt}")

            # Add conversation messages
            for msg in request.messages:
                content_parts.append(f"{msg.role.capitalize()}: {msg.content}")

            content = "\n".join(content_parts)

            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                max_output_tokens=request.config.max_tokens,
                stop_sequences=request.config.stop,
            )

            # Handle structured output
            if request.output_schema:
                content += f"\n\nPlease respond with valid JSON that matches this schema: {request.output_schema.model_json_schema()}"

            # Generate response
            response = model.generate_content(
                content, generation_config=generation_config
            )

            response_text = response.text
            response_time = time.time() - start_time

            # Parse structured output if needed
            parsed_output = None
            if request.output_schema:
                try:
                    parsed_data = json.loads(response_text)
                    parsed_output = request.output_schema(**parsed_data)
                except Exception as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    raise ValueError(
                        f"Failed to parse response according to schema: {e}"
                    )

            # Record stats
            self.stats.record_request(success=True, response_time=response_time)

            logger.info("Google AI generation successful")

            return LLMResponse(
                content=response_text,
                model=request.model,
                usage={"tokens": len(response_text.split())},  # Approximate
                metadata={"response_time": response_time},
                parsed_output=parsed_output,
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.stats.record_request(success=False, response_time=response_time)
            logger.error(f"Google AI generation failed: {e}")
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
        return self.stats

    def reset_stats(self) -> None:
        self.stats = LLMStats()
        logger.info("Google AI adapter statistics reset")
