from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, Type
from enum import Enum
from pydantic import BaseModel, Field
import time


class LLMProvider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    LANGCHAIN = "langchain"
    GOOGLE_ADK = "google_adk"


class LLMModel(Enum):
    """Supported LLM models"""

    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class GenerationConfig(BaseModel):
    """Configuration for LLM generation"""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class LLMMessage(BaseModel):
    """Represents a message in LLM conversation"""

    role: str = Field(
        ..., description="Role of the message sender (system, user, assistant)"
    )
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Optional name of the sender")


class LLMRequest(BaseModel):
    """Request for LLM generation"""

    messages: List[LLMMessage] = Field(..., description="List of messages")
    model: str = Field(..., description="Model to use for generation")
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    output_schema: Optional[Type[BaseModel]] = Field(
        None, description="Expected output schema"
    )
    system_prompt: Optional[str] = Field(None, description="System prompt override")


class LLMResponse(BaseModel):
    """Response from LLM generation"""

    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(
        default_factory=dict, description="Token usage information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: float = Field(default_factory=time.time)
    parsed_output: Optional[BaseModel] = Field(
        None, description="Parsed output if schema provided"
    )


class LLMStats(BaseModel):
    """Statistics for LLM usage"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    start_time: float = Field(default_factory=time.time)

    def record_request(
        self,
        success: bool,
        tokens_used: int = 0,
        cost: float = 0.0,
        response_time: float = 0.0,
    ):
        """Record a request in statistics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens_used += tokens_used
        self.total_cost += cost

        # Update average response time
        if self.successful_requests > 0:
            current_total_time = self.average_response_time * (
                self.successful_requests - 1
            )
            self.average_response_time = (
                current_total_time + response_time
            ) / self.successful_requests

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        return (
            self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 0.0
        )


class LLMInterface(ABC):
    """Abstract interface for LLM implementations"""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from LLM

        Args:
            request: LLM request with messages and configuration

        Returns:
            LLM response with generated content
        """
        pass

    @abstractmethod
    def generate_with_schema(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> BaseModel:
        """
        Generate structured output using schema

        Args:
            prompt: Input prompt
            output_schema: Pydantic model for output structure
            model: Model to use (optional)
            config: Generation configuration (optional)

        Returns:
            Parsed output matching the schema
        """
        pass

    @abstractmethod
    def simple_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Simple text generation

        Args:
            prompt: Input prompt
            model: Model to use (optional)
            config: Generation configuration (optional)

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def get_stats(self) -> LLMStats:
        """Get usage statistics"""
        pass

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        pass


class LLMAdapterInterface(LLMInterface):
    """Interface for LLM adapters"""

    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the provider this adapter supports"""
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validate if model is supported by this adapter"""
        pass


class LLMStrategyInterface(ABC):
    """Interface for LLM generation strategies"""

    @abstractmethod
    def execute(self, llm: LLMInterface, request: LLMRequest) -> LLMResponse:
        """
        Execute generation strategy

        Args:
            llm: LLM interface to use
            request: Generation request

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass
