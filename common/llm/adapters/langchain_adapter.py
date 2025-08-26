# common/llm/adapters/langchain_adapter.py

import time
from typing import List, Optional, Type

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from logger import LoggerFactory, LoggerType, LogLevel

from ..llm_interfaces import (
    GenerationConfig,
    LLMAdapterInterface,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStats,
)

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
        elif self.provider_type == LLMProvider.GEMINI:
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
        elif self.provider_type == LLMProvider.GEMINI:
            return ["gemini-pro", "gemini-pro-vision"]
        return []

    def is_available(self) -> bool:
        """Check if service is available"""
        try:
            # Simple availability check
            test_messages = [HumanMessage(content="ping")]
            response = self.client.invoke(test_messages)
            return bool(response and response.content)
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
        """
        Generate structured output using LangChain's with_structured_output.

        Args:
            prompt: The user prompt.
            output_schema: A Pydantic model defining the desired output structure.
            model: Optional override of the model name.
            config: Optional GenerationConfig for temperature, tokens, etc.

        Returns:
            An instance of output_schema populated with the LLM’s response.
        """
        # Choose model name and config
        model_name = model or self.config.model_name
        gen_config = config or GenerationConfig()

        # Create a fresh ChatOpenAI (or ChatGoogleGenerativeAI) client for this request
        if self.provider_type == LLMProvider.OPENAI:
            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=gen_config.temperature,
                openai_api_key=self.config.api_key,
                max_tokens=gen_config.max_tokens,
                request_timeout=self.config.timeout,
            )
        else:
            llm_client = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=gen_config.temperature,
                google_api_key=self.config.api_key,
            )

        # Bind the Pydantic schema to the model, creating a structured-output Runnable
        structured_llm = llm_client.with_structured_output(output_schema)

        start_time = time.time()
        # Invoke the model; returns a Pydantic instance directly
        structured_response: BaseModel = structured_llm.invoke(prompt)
        elapsed = time.time() - start_time

        # Record statistics (we approximate tokens by word count here)
        approx_tokens = len(str(structured_response).split())
        self.stats.record_request(
            success=True,
            tokens_used=approx_tokens,
            response_time=elapsed,
        )
        logger.info(
            f"LangChain structured generation successful ({approx_tokens} tokens)"
        )

        return structured_response

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
