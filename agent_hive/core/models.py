"""Model abstraction layer for AgentHive."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelResponse:
    """Normalized model response structure."""

    content: str
    provider: str
    model_name: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model instances."""

    provider: ModelProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 30
    max_retries: int = 3
    additional_params: Optional[Dict[str, Any]] = field(default_factory=dict)


class ModelInterface(ABC):
    """Abstract interface for all model providers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.provider = config.provider.value
        self.model_name = config.model_name

    @abstractmethod
    async def generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def stream_generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generate responses from the model."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model is currently available."""
        pass


class OpenAIModel(ModelInterface):
    """OpenAI model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = ChatOpenAI(
            api_key=SecretStr(config.api_key),
            model=config.model_name,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            **(config.additional_params or {}),
        )

    async def generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ModelResponse:
        """Generate response using OpenAI."""
        try:
            response = await self.client.ainvoke(messages, **kwargs)

            # Handle content that might be string or list
            content = response.content
            if isinstance(content, list):
                # Convert list content to string
                content = "".join(str(item) for item in content)

            return ModelResponse(
                content=str(content),
                provider=self.provider,
                model_name=self.model_name,
                usage=getattr(response, "usage_metadata", None),
                metadata=getattr(response, "response_metadata", None),
                finish_reason=getattr(response, "finish_reason", None),
            )
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def stream_generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generate using OpenAI."""
        return self._stream_impl(messages, **kwargs)

    async def _stream_impl(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Internal streaming implementation."""
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                if chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        # Convert list content to string
                        content = "".join(str(item) for item in content)
                    yield str(content)
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check OpenAI availability."""
        try:
            test_messages = [HumanMessage(content="test")]
            await asyncio.wait_for(self.client.ainvoke(test_messages), timeout=5.0)
            return True
        except Exception:
            return False


class AnthropicModel(ModelInterface):
    """Anthropic model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = ChatAnthropic(
            api_key=SecretStr(config.api_key),
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens_to_sample=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            **(config.additional_params or {}),
        )

    async def generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ModelResponse:
        """Generate response using Anthropic."""
        try:
            response = await self.client.ainvoke(messages, **kwargs)

            # Handle content that might be string or list
            content = response.content
            if isinstance(content, list):
                # Convert list content to string
                content = "".join(str(item) for item in content)

            return ModelResponse(
                content=str(content),
                provider=self.provider,
                model_name=self.model_name,
                usage=getattr(response, "usage_metadata", None),
                metadata=getattr(response, "response_metadata", None),
                finish_reason=getattr(response, "finish_reason", None),
            )
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    def stream_generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generate using Anthropic."""
        return self._stream_impl(messages, **kwargs)

    async def _stream_impl(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Internal streaming implementation."""
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                if chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        # Convert list content to string
                        content = "".join(str(item) for item in content)
                    yield str(content)
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check Anthropic availability."""
        try:
            test_messages = [HumanMessage(content="test")]
            await asyncio.wait_for(self.client.ainvoke(test_messages), timeout=5.0)
            return True
        except Exception:
            return False


class GoogleModel(ModelInterface):
    """Google model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = ChatGoogleGenerativeAI(
            google_api_key=config.api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            **(config.additional_params or {}),
        )

    async def generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ModelResponse:
        """Generate response using Google."""
        try:
            response = await self.client.ainvoke(messages, **kwargs)

            # Handle content that might be string or list
            content = response.content
            if isinstance(content, list):
                # Convert list content to string
                content = "".join(str(item) for item in content)

            return ModelResponse(
                content=str(content),
                provider=self.provider,
                model_name=self.model_name,
                usage=getattr(response, "usage_metadata", None),
                metadata=getattr(response, "response_metadata", None),
                finish_reason=getattr(response, "finish_reason", None),
            )
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            raise

    def stream_generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generate using Google."""
        return self._stream_impl(messages, **kwargs)

    async def _stream_impl(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Internal streaming implementation."""
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                if chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        # Convert list content to string
                        content = "".join(str(item) for item in content)
                    yield str(content)
        except Exception as e:
            logger.error(f"Google streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check Google availability."""
        try:
            test_messages = [HumanMessage(content="test")]
            await asyncio.wait_for(self.client.ainvoke(test_messages), timeout=5.0)
            return True
        except Exception:
            return False


class ModelFactory:
    """Factory for creating model instances with fallback support."""

    def __init__(self) -> None:
        self._model_configs: Dict[str, ModelConfig] = {}
        self._model_instances: Dict[str, ModelInterface] = {}
        self._fallback_chains: Dict[str, List[str]] = {}

    def register_model(
        self, name: str, config: ModelConfig, fallbacks: Optional[List[str]] = None
    ) -> None:
        """Register a model configuration with optional fallbacks."""
        self._model_configs[name] = config
        self._fallback_chains[name] = fallbacks or []

        # Create model instance
        if config.provider == ModelProvider.OPENAI:
            self._model_instances[name] = OpenAIModel(config)
        elif config.provider == ModelProvider.ANTHROPIC:
            self._model_instances[name] = AnthropicModel(config)
        elif config.provider == ModelProvider.GOOGLE:
            self._model_instances[name] = GoogleModel(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    async def get_model(self, name: str) -> ModelInterface:
        """Get a model instance, with fallback if primary is unavailable."""
        if name not in self._model_instances:
            raise ValueError(f"Model '{name}' not registered")

        primary_model = self._model_instances[name]

        # Check if primary model is available
        if await primary_model.is_available():
            return primary_model

        # Try fallbacks
        for fallback_name in self._fallback_chains.get(name, []):
            if fallback_name in self._model_instances:
                fallback_model = self._model_instances[fallback_name]
                if await fallback_model.is_available():
                    logger.warning(
                        f"Primary model '{name}' unavailable, using fallback '{fallback_name}'"
                    )
                    return fallback_model

        # If no fallbacks work, return primary (will fail on use)
        logger.error(f"Model '{name}' and all fallbacks unavailable")
        return primary_model

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._model_instances.keys())

    async def check_all_models(self) -> Dict[str, bool]:
        """Check availability of all registered models."""
        results = {}
        for name, model in self._model_instances.items():
            results[name] = await model.is_available()
        return results


# Global model factory instance
model_factory = ModelFactory()


def get_model_factory() -> ModelFactory:
    """Get the global model factory instance."""
    return model_factory
