import time
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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
    name="langchain-adapter", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class LangChainConfig(BaseSettings):
    """Configuration for LangChain adapter"""

    api_key: Optional[str] = Field(None, description="API key")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.7, description="Default temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    timeout: int = Field(default=30, description="Request timeout")

    class Config:
        env_prefix = "LANGCHAIN_"


class LangChainAdapter(LLMAdapterInterface):
    """LangChain adapter implementation"""

    def __init__(
        self, config: LangChainConfig, provider: LLMProvider = LLMProvider.OPENAI
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not available. Install with: pip install langchain langchain-openai"
            )

        self.config = config
        self.provider_type = provider
        self.client = self._create_client()
        self.stats = LLMStats()

        logger.info(f"Initialized LangChain adapter with {provider.value}")

    def _create_client(self):
        """Create LangChain client based on provider"""
        if self.provider_type == LLMProvider.OPENAI:
            return ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                openai_api_key=self.config.api_key,
                max_tokens=self.config.max_tokens,
                request_timeout=self.config.timeout,
            )
        elif self.provider_type == LLMProvider.GOOGLE_ADK:
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                google_api_key=self.config.api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider_type}")

    def get_provider(self) -> LLMProvider:
        return self.provider_type

    def validate_model(self, model: str) -> bool:
        """Basic model validation"""
        return len(model) > 0

    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        if self.provider_type == LLMProvider.OPENAI:
            return [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview",
                "gpt-4o",
                "gpt-4o-mini",
            ]
        elif self.provider_type == LLMProvider.GOOGLE_ADK:
            return ["gemini-pro", "gemini-pro-vision"]
        return []

    def is_available(self) -> bool:
        """Check if service is available"""
        try:
            messages = [HumanMessage(content="test")]
            self.client.invoke(messages)
            return True
        except Exception as e:
            logger.warning(f"LangChain service not available: {e}")
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LangChain"""
        start_time = time.time()

        try:
            # Convert messages to LangChain format
            messages = []

            if request.system_prompt:
                messages.append(SystemMessage(content=request.system_prompt))

            for msg in request.messages:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    messages.append(SystemMessage(content=msg.content))

            # Configure client for this request
            self.client.temperature = request.config.temperature
            if request.config.max_tokens:
                self.client.max_tokens = request.config.max_tokens

            # Generate response
            response = self.client.invoke(messages)
            content = response.content

            response_time = time.time() - start_time

            # Record stats
            self.stats.record_request(success=True, response_time=response_time)

            logger.info("LangChain generation successful")

            return LLMResponse(
                content=content,
                model=request.model,
                usage={"tokens": len(content.split())},  # Approximate
                metadata={"response_time": response_time},
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.stats.record_request(success=False, response_time=response_time)
            logger.error(f"LangChain generation failed: {e}")
            raise

    def generate_with_schema(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> BaseModel:
        """Generate structured output - simplified for LangChain"""
        # Add schema instruction to prompt
        schema_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema: {output_schema.model_json_schema()}"

        result = self.simple_generate(schema_prompt, model, config)

        try:
            import json

            parsed_data = json.loads(result)
            return output_schema(**parsed_data)
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}")
            raise ValueError(f"Failed to parse response according to schema: {e}")

    def simple_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Simple text generation"""
        config = config or GenerationConfig()

        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            model=model or self.config.model_name,
            config=config,
        )

        response = self.generate(request)
        return response.content

    def get_stats(self) -> LLMStats:
        return self.stats

    def reset_stats(self) -> None:
        self.stats = LLMStats()
        logger.info("LangChain adapter statistics reset")
