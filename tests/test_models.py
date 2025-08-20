"""Unit tests for model abstraction layer with proper typing."""

import pytest
from typing import List, Tuple, Any, AsyncIterator, Sequence
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import HumanMessage, BaseMessage

from agent_hive.core.models import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelInterface,
    OpenAIModel,
    AnthropicModel,
    GoogleModel,
    ModelFactory,
    get_model_factory
)


class MockModel(ModelInterface):
    """Mock model for testing."""
    
    def __init__(self, config: ModelConfig, available: bool = True) -> None:
        super().__init__(config)
        self.available = available
        self.generate_calls: List[Tuple[Sequence[BaseMessage], Any]] = []
        self.stream_calls: List[Tuple[Sequence[BaseMessage], Any]] = []
    
    async def generate(self, messages: Sequence[BaseMessage], **kwargs: Any) -> ModelResponse:
        self.generate_calls.append((messages, kwargs))
        if not self.available:
            raise Exception("Model unavailable")
        
        return ModelResponse(
            content="Mock response",
            provider=self.provider,
            model_name=self.model_name,
            usage={"tokens": 10},
            metadata={"test": True}
        )
    
    def stream_generate(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        return self._stream_impl(messages, **kwargs)
    
    async def _stream_impl(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        self.stream_calls.append((messages, kwargs))
        if not self.available:
            raise Exception("Model unavailable")
        
        for chunk in ["Mock ", "stream ", "response"]:
            yield chunk
    
    async def is_available(self) -> bool:
        return self.available


@pytest.fixture
def openai_config() -> ModelConfig:
    """OpenAI model configuration for testing."""
    return ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def anthropic_config() -> ModelConfig:
    """Anthropic model configuration for testing."""
    return ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        api_key="test-key",
        temperature=0.5,
        max_tokens=2000
    )


@pytest.fixture
def google_config() -> ModelConfig:
    """Google model configuration for testing."""
    return ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-pro",
        api_key="test-key",
        temperature=0.8,
        max_tokens=1500
    )


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self, openai_config: ModelConfig) -> None:
        """Test creating a model configuration."""
        assert openai_config.provider == ModelProvider.OPENAI
        assert openai_config.model_name == "gpt-4"
        assert openai_config.api_key == "test-key"
        assert openai_config.temperature == 0.7
        assert openai_config.max_tokens == 1000
    
    def test_model_config_defaults(self) -> None:
        """Test model configuration defaults."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.timeout == 30
        assert config.max_retries == 3


class TestModelResponse:
    """Test ModelResponse dataclass."""
    
    def test_model_response_creation(self) -> None:
        """Test creating a model response."""
        response = ModelResponse(
            content="Test response",
            provider="openai",
            model_name="gpt-4",
            usage={"tokens": 15},
            metadata={"test": True},
            finish_reason="stop"
        )
        
        assert response.content == "Test response"
        assert response.provider == "openai"
        assert response.model_name == "gpt-4"
        assert response.usage == {"tokens": 15}
        assert response.metadata == {"test": True}
        assert response.finish_reason == "stop"


class TestMockModel:
    """Test the mock model implementation."""
    
    @pytest.mark.asyncio
    async def test_mock_model_generate(self, openai_config: ModelConfig) -> None:
        """Test mock model generation."""
        model = MockModel(openai_config)
        messages = [HumanMessage(content="Test message")]
        
        response = await model.generate(messages)
        
        assert response.content == "Mock response"
        assert response.provider == "openai"
        assert response.model_name == "gpt-4"
        assert len(model.generate_calls) == 1
    
    @pytest.mark.asyncio
    async def test_mock_model_stream(self, openai_config: ModelConfig) -> None:
        """Test mock model streaming."""
        model = MockModel(openai_config)
        messages = [HumanMessage(content="Test message")]
        
        chunks = []
        async for chunk in model.stream_generate(messages):
            chunks.append(chunk)
        
        assert chunks == ["Mock ", "stream ", "response"]
        assert len(model.stream_calls) == 1
    
    @pytest.mark.asyncio
    async def test_mock_model_availability(self, openai_config: ModelConfig) -> None:
        """Test mock model availability."""
        available_model = MockModel(openai_config, available=True)
        unavailable_model = MockModel(openai_config, available=False)
        
        assert await available_model.is_available() is True
        assert await unavailable_model.is_available() is False
    
    @pytest.mark.asyncio
    async def test_mock_model_unavailable_generate(self, openai_config: ModelConfig) -> None:
        """Test mock model generation when unavailable."""
        model = MockModel(openai_config, available=False)
        messages = [HumanMessage(content="Test message")]
        
        with pytest.raises(Exception, match="Model unavailable"):
            await model.generate(messages)
    
    @pytest.mark.asyncio
    async def test_mock_model_unavailable_stream(self, openai_config: ModelConfig) -> None:
        """Test mock model streaming when unavailable."""
        model = MockModel(openai_config, available=False)
        messages = [HumanMessage(content="Test message")]
        
        with pytest.raises(Exception, match="Model unavailable"):
            async for _ in model.stream_generate(messages):
                pass


class TestOpenAIModel:
    """Test OpenAI model implementation."""
    
    def test_openai_model_creation(self, openai_config: ModelConfig) -> None:
        """Test creating OpenAI model."""
        with patch('agent_hive.core.models.ChatOpenAI') as mock_chat:
            model = OpenAIModel(openai_config)
            
            assert model.provider == "openai"
            assert model.model_name == "gpt-4"
            mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_model_generate(self, openai_config: ModelConfig) -> None:
        """Test OpenAI model generation."""
        with patch('agent_hive.core.models.ChatOpenAI') as mock_chat:
            mock_response = Mock()
            mock_response.content = "OpenAI response"
            mock_response.usage_metadata = {"tokens": 20}
            mock_response.response_metadata = {"model": "gpt-4"}
            mock_response.finish_reason = "stop"
            
            mock_chat.return_value.ainvoke = AsyncMock(return_value=mock_response)
            
            model = OpenAIModel(openai_config)
            messages = [HumanMessage(content="Test")]
            
            response = await model.generate(messages)
            
            assert response.content == "OpenAI response"
            assert response.provider == "openai"
            assert response.usage == {"tokens": 20}
    
    @pytest.mark.asyncio
    async def test_openai_model_stream(self, openai_config: ModelConfig) -> None:
        """Test OpenAI model streaming."""
        with patch('agent_hive.core.models.ChatOpenAI') as mock_chat:
            mock_chunks = [
                Mock(content="Hello "),
                Mock(content="world"),
                Mock(content="!")
            ]
            
            async def mock_astream(*args: Any, **kwargs: Any) -> AsyncIterator[Mock]:
                for chunk in mock_chunks:
                    yield chunk
            
            mock_chat.return_value.astream = mock_astream
            
            model = OpenAIModel(openai_config)
            messages = [HumanMessage(content="Test")]
            
            chunks = []
            async for chunk in model.stream_generate(messages):
                chunks.append(chunk)
            
            assert chunks == ["Hello ", "world", "!"]


class TestAnthropicModel:
    """Test Anthropic model implementation."""
    
    def test_anthropic_model_creation(self, anthropic_config: ModelConfig) -> None:
        """Test creating Anthropic model."""
        with patch('agent_hive.core.models.ChatAnthropic') as mock_chat:
            model = AnthropicModel(anthropic_config)
            
            assert model.provider == "anthropic"
            assert model.model_name == "claude-3-sonnet-20240229"
            mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_anthropic_model_generate(self, anthropic_config: ModelConfig) -> None:
        """Test Anthropic model generation."""
        with patch('agent_hive.core.models.ChatAnthropic') as mock_chat:
            mock_response = Mock()
            mock_response.content = "Anthropic response"
            mock_response.usage_metadata = {"tokens": 25}
            
            mock_chat.return_value.ainvoke = AsyncMock(return_value=mock_response)
            
            model = AnthropicModel(anthropic_config)
            messages = [HumanMessage(content="Test")]
            
            response = await model.generate(messages)
            
            assert response.content == "Anthropic response"
            assert response.provider == "anthropic"
            assert response.usage == {"tokens": 25}


class TestGoogleModel:
    """Test Google model implementation."""
    
    def test_google_model_creation(self, google_config: ModelConfig) -> None:
        """Test creating Google model."""
        with patch('agent_hive.core.models.ChatGoogleGenerativeAI') as mock_chat:
            model = GoogleModel(google_config)
            
            assert model.provider == "google"
            assert model.model_name == "gemini-pro"
            mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_google_model_generate(self, google_config: ModelConfig) -> None:
        """Test Google model generation."""
        with patch('agent_hive.core.models.ChatGoogleGenerativeAI') as mock_chat:
            mock_response = Mock()
            mock_response.content = "Google response"
            mock_response.usage_metadata = {"tokens": 30}
            
            mock_chat.return_value.ainvoke = AsyncMock(return_value=mock_response)
            
            model = GoogleModel(google_config)
            messages = [HumanMessage(content="Test")]
            
            response = await model.generate(messages)
            
            assert response.content == "Google response"
            assert response.provider == "google"
            assert response.usage == {"tokens": 30}


class TestModelFactory:
    """Test ModelFactory implementation."""
    
    def test_model_factory_creation(self) -> None:
        """Test creating a model factory."""
        factory = ModelFactory()
        assert len(factory.list_models()) == 0
    
    def test_register_openai_model(self, openai_config: ModelConfig) -> None:
        """Test registering an OpenAI model."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatOpenAI'):
            factory.register_model("test-openai", openai_config)
            
            assert "test-openai" in factory.list_models()
            assert len(factory.list_models()) == 1
    
    def test_register_anthropic_model(self, anthropic_config: ModelConfig) -> None:
        """Test registering an Anthropic model."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatAnthropic'):
            factory.register_model("test-anthropic", anthropic_config)
            
            assert "test-anthropic" in factory.list_models()
    
    def test_register_google_model(self, google_config: ModelConfig) -> None:
        """Test registering a Google model."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatGoogleGenerativeAI'):
            factory.register_model("test-google", google_config)
            
            assert "test-google" in factory.list_models()
    
    def test_register_unsupported_provider(self) -> None:
        """Test registering unsupported provider raises error."""
        factory = ModelFactory()
        
        # Create config with invalid provider (bypassing enum validation for test)
        invalid_config = ModelConfig(
            provider=ModelProvider.OPENAI,  # Valid enum
            model_name="test",
            api_key="test"
        )
        # Manually set invalid provider to test factory validation
        invalid_config.provider = "invalid"  # type: ignore
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            factory.register_model("invalid-model", invalid_config)
    
    @pytest.mark.asyncio
    async def test_get_available_model(self, openai_config: ModelConfig) -> None:
        """Test getting an available model."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatOpenAI'):
            factory.register_model("test-model", openai_config)
            
            # Mock the model to be available
            with patch.object(factory._model_instances["test-model"], 'is_available', new_callable=AsyncMock) as mock_available:
                mock_available.return_value = True
                
                model = await factory.get_model("test-model")
                assert model is not None
                assert model.model_name == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_get_model_with_fallback(self, openai_config: ModelConfig, anthropic_config: ModelConfig) -> None:
        """Test getting model with fallback when primary unavailable."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatOpenAI'), \
             patch('agent_hive.core.models.ChatAnthropic'):
            
            factory.register_model("primary", openai_config)
            factory.register_model("fallback", anthropic_config, fallbacks=[])
            factory.register_model("with-fallback", openai_config, fallbacks=["fallback"])
            
            # Mock primary as unavailable, fallback as available
            with patch.object(factory._model_instances["with-fallback"], 'is_available', new_callable=AsyncMock) as mock_primary, \
                 patch.object(factory._model_instances["fallback"], 'is_available', new_callable=AsyncMock) as mock_fallback:
                mock_primary.return_value = False
                mock_fallback.return_value = True
                
                model = await factory.get_model("with-fallback")
                assert model.model_name == "claude-3-sonnet-20240229"  # Should get fallback
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self) -> None:
        """Test getting non-existent model raises error."""
        factory = ModelFactory()
        
        with pytest.raises(ValueError, match="Model 'nonexistent' not registered"):
            await factory.get_model("nonexistent")
    
    @pytest.mark.asyncio
    async def test_check_all_models(self, openai_config: ModelConfig, anthropic_config: ModelConfig) -> None:
        """Test checking availability of all models."""
        factory = ModelFactory()
        
        with patch('agent_hive.core.models.ChatOpenAI'), \
             patch('agent_hive.core.models.ChatAnthropic'):
            
            factory.register_model("model1", openai_config)
            factory.register_model("model2", anthropic_config)
            
            # Mock availability
            with patch.object(factory._model_instances["model1"], 'is_available', new_callable=AsyncMock) as mock1, \
                 patch.object(factory._model_instances["model2"], 'is_available', new_callable=AsyncMock) as mock2:
                mock1.return_value = True
                mock2.return_value = False
                
                results = await factory.check_all_models()
                
                assert results["model1"] is True
                assert results["model2"] is False


class TestGlobalFactory:
    """Test global model factory functions."""
    
    def test_get_model_factory(self) -> None:
        """Test getting global model factory."""
        factory = get_model_factory()
        assert isinstance(factory, ModelFactory)
        
        # Should return same instance
        factory2 = get_model_factory()
        assert factory is factory2