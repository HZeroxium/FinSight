# common/llm/adapters/google_adk_adapter.py

import json
import time
import uuid
from typing import List, Optional, Type

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ..llm_interfaces import (
    LLMAdapterInterface,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStats,
    GenerationConfig,
    LLMMessage,
)
from logger import LoggerFactory, LoggerType, LogLevel

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

logger = LoggerFactory.get_logger(
    name="google-adk-adapter", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class GoogleConfig(BaseSettings):
    """Configuration for Google ADK adapter"""

    # Default to a Gemini model, or any provider/model string LiteLlm supports
    default_model: str = Field(default="gemini-pro", description="Default model")
    timeout: int = Field(default=30, description="LLM call timeout (s)")

    class Config:
        env_prefix = "GOOGLE_ADK_"


class GoogleADKAdapter(LLMAdapterInterface):
    """Google ADK adapter implementation using LiteLlm"""

    def __init__(self, config: GoogleConfig):
        # initialize config and stats
        self.config = config
        self.stats = LLMStats()

        # create a LiteLlm client for direct LLM calls
        self.llm = LiteLlm(model=self.config.default_model)
        logger.info(
            f"Initialized Google ADK adapter with model: {self.config.default_model}"
        )

    def get_provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE_AGENT_DEVELOPMENT_KIT

    def validate_model(self, model: str) -> bool:
        # LiteLlm will error if unsupported; we do a basic length check here
        return bool(model)

    def get_supported_models(self) -> List[str]:
        # We don't maintain the full list here; adapters typically return their default
        return [self.config.default_model]

    def is_available(self) -> bool:
        try:
            # Quick sanity check
            _ = self.llm.generate_content(
                types.LlmRequest(
                    model=self.config.default_model,
                    messages=[
                        types.Content(role="user", parts=[types.Part(text="ping")])
                    ],
                )
            )
            return True
        except Exception as e:
            logger.warning(f"Google ADK service not available: {e}")
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()

        # prepare a fresh ADK session
        session_svc = InMemorySessionService()
        artifact_svc = InMemoryArtifactService()
        memory_svc = InMemoryMemoryService()
        session_id = str(uuid.uuid4())
        user_id = "adapter"

        session_svc.create_session(
            app_name="google-adapter",
            user_id=user_id,
            session_id=session_id,
            state={},
        )

        # wrap messages into a single instruction string
        # we append system prompt first if present
        instruction = ""
        if request.system_prompt:
            instruction += request.system_prompt + "\n\n"
        instruction += "\n".join(
            f"{msg.role}: {msg.content}" for msg in request.messages
        )

        # create the ADK agent
        agent = LlmAgent(
            name="google-adapter",
            model=LiteLlm(model=request.model),
            instruction=instruction,
            input_schema=None,
            output_schema=request.output_schema,
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
        )

        # runner to execute
        runner = Runner(
            app_name="google-adapter",
            agent=agent,
            session_service=session_svc,
            artifact_service=artifact_svc,
            memory_service=memory_svc,
        )

        # build the Content object
        user_content = types.Content(role="user", parts=[types.Part(text=instruction)])

        try:
            # collect streamed or sync parts
            response_parts: List[str] = []
            for event in runner.run(
                session_id=session_id,
                user_id=user_id,
                new_message=user_content,
            ):
                if getattr(event, "content", None):
                    for part in event.content.parts:
                        if part.text:
                            response_parts.append(part.text)

            full_text = "".join(response_parts)
            usage = {"tokens": len(full_text.split())}  # rough token count

            parsed_output = None
            if request.output_schema:
                data = json.loads(full_text)
                parsed_output = request.output_schema(**data)

            elapsed = time.time() - start_time
            self.stats.record_request(
                success=True, tokens_used=usage["tokens"], response_time=elapsed
            )
            logger.info(f"Google ADK generation successful ({usage['tokens']} tokens)")

            return LLMResponse(
                content=full_text,
                model=request.model,
                usage=usage,
                metadata={"response_time": elapsed},
                parsed_output=parsed_output,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.stats.record_request(success=False, response_time=elapsed)
            logger.error(f"Google ADK generation failed: {e}")
            raise

    def generate_with_schema(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> BaseModel:
        response = self.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model or self.config.default_model,
                config=config or GenerationConfig(),
                output_schema=output_schema,
            )
        )
        if response.parsed_output is None:
            raise ValueError("Failed to generate structured output")
        return response.parsed_output

    def simple_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        response = self.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model or self.config.default_model,
                config=config or GenerationConfig(),
            )
        )
        return response.content

    def get_stats(self) -> LLMStats:
        return self.stats

    def reset_stats(self) -> None:
        self.stats = LLMStats()
        logger.info("Google ADK adapter statistics reset")
