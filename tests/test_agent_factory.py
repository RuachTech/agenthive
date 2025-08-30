"""Unit tests for the agent factory system."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent_hive.agents.factory import (
    AgentFactory,
    AgentNodeWrapper,
    AgentCapabilities,
    AgentError,
    AgentValidationError,
    AgentExecutionError,
    ErrorRecoveryStrategy,
    get_agent_factory,
)
from agent_hive.core.config import AgentConfig
from agent_hive.core.state import AgentState
from agent_hive.core.models import ModelInterface, ModelResponse


class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"

    async def _arun(self, query: str) -> str:
        return f"Mock async result for: {query}"


class MockModel(ModelInterface):
    """Mock model for testing."""

    def __init__(self, available: bool = True, should_timeout: bool = False):
        self.available = available
        self.should_timeout = should_timeout
        self.provider = "mock"
        self.model_name = "mock-model"

    async def generate(self, messages, **kwargs):
        if self.should_timeout:
            await asyncio.sleep(10)  # Simulate timeout

        return ModelResponse(
            content="Mock response", provider=self.provider, model_name=self.model_name
        )

    def stream_generate(self, messages, **kwargs):
        async def _stream():
            yield "Mock"
            yield " stream"
            yield " response"

        return _stream()

    async def is_available(self):
        return self.available


@pytest.fixture
def sample_agent_config():
    """Create a sample agent configuration for testing."""
    return AgentConfig(
        name="test_agent",
        display_name="Test Agent",
        description="A test agent",
        system_prompt="You are a test agent.",
        model_provider="openai",
        model_name="gpt-4",
        capabilities=["test_capability"],
        specialized_for=["testing"],
    )


@pytest.fixture
def sample_capabilities():
    """Create sample agent capabilities for testing."""
    return AgentCapabilities(
        required_capabilities=["test_capability"],
        optional_capabilities=["optional_test"],
        model_requirements=["function_calling"],
        required_tools=["mock_tool"],
        mcp_requirements=["test_mcp"],
    )


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return [MockTool()]


@pytest.fixture
def sample_state():
    """Create a sample agent state for testing."""
    return AgentState(
        task="Test task",
        messages=[HumanMessage(content="Hello")],
        next="",
        scratchpad={},
        mode="direct",
        active_agents=[],
        multimodal_content={},
        session_id="test_session",
        user_id="test_user",
        last_updated=datetime.now(),
        errors=[],
        task_status={},
    )


class TestAgentCapabilities:
    """Test AgentCapabilities dataclass."""

    def test_agent_capabilities_creation(self):
        """Test creating AgentCapabilities."""
        capabilities = AgentCapabilities(
            required_capabilities=["req1", "req2"],
            optional_capabilities=["opt1"],
            model_requirements=["vision"],
            required_tools=["tool1"],
            mcp_requirements=["mcp1"],
        )

        assert capabilities.required_capabilities == ["req1", "req2"]
        assert capabilities.optional_capabilities == ["opt1"]
        assert capabilities.model_requirements == ["vision"]
        assert capabilities.required_tools == ["tool1"]
        assert capabilities.mcp_requirements == ["mcp1"]


class TestAgentError:
    """Test agent error classes."""

    def test_agent_error_creation(self):
        """Test creating AgentError."""
        error = AgentError("test_agent", "test_error", "Test message")

        assert error.agent_name == "test_agent"
        assert error.error_type == "test_error"
        assert error.message == "Test message"
        assert str(error) == "test_agent: test_error - Test message"

    def test_agent_validation_error(self):
        """Test AgentValidationError inheritance."""
        error = AgentValidationError("test_agent", "validation", "Validation failed")

        assert isinstance(error, AgentError)
        assert error.error_type == "validation"

    def test_agent_execution_error(self):
        """Test AgentExecutionError inheritance."""
        error = AgentExecutionError("test_agent", "execution", "Execution failed")

        assert isinstance(error, AgentError)
        assert error.error_type == "execution"


class TestErrorRecoveryStrategy:
    """Test error recovery strategies."""

    @pytest.mark.asyncio
    async def test_handle_model_timeout(self, sample_state):
        """Test handling model timeout."""
        strategy = ErrorRecoveryStrategy()

        result_state = await strategy.handle_model_timeout(sample_state, "test_agent")

        assert len(result_state["errors"]) == 1
        assert result_state["errors"][0]["type"] == "model_timeout"
        assert result_state["errors"][0]["agent"] == "test_agent"
        assert "test_agent_recovery" in result_state["scratchpad"]

    @pytest.mark.asyncio
    async def test_handle_tool_failure(self, sample_state):
        """Test handling tool failure."""
        strategy = ErrorRecoveryStrategy()
        test_error = Exception("Tool failed")

        result_state = await strategy.handle_tool_failure(
            sample_state, "test_agent", "test_tool", test_error
        )

        assert len(result_state["errors"]) == 1
        assert result_state["errors"][0]["type"] == "tool_failure"
        assert result_state["errors"][0]["tool"] == "test_tool"
        assert "test_agent_tool_failure" in result_state["scratchpad"]

    @pytest.mark.asyncio
    async def test_handle_validation_failure(self, sample_state):
        """Test handling validation failure."""
        strategy = ErrorRecoveryStrategy()
        validation_errors = ["Error 1", "Error 2"]

        result_state = await strategy.handle_validation_failure(
            sample_state, "test_agent", validation_errors
        )

        assert len(result_state["errors"]) == 1
        assert result_state["errors"][0]["type"] == "validation_failure"
        assert result_state["errors"][0]["validation_errors"] == validation_errors
        assert "test_agent_validation_failed" in result_state["scratchpad"]


class TestAgentNodeWrapper:
    """Test AgentNodeWrapper functionality."""

    def test_wrapper_creation(
        self, sample_agent_config, sample_capabilities, mock_model, mock_tools
    ):
        """Test creating AgentNodeWrapper."""
        wrapper = AgentNodeWrapper(
            agent_name="test_agent",
            agent_config=sample_agent_config,
            model=mock_model,
            tools=mock_tools,
            capabilities=sample_capabilities,
        )

        assert wrapper.agent_name == "test_agent"
        assert wrapper.agent_config == sample_agent_config
        assert wrapper.model == mock_model
        assert wrapper.tools == mock_tools
        assert wrapper.capabilities == sample_capabilities

    @pytest.mark.asyncio
    async def test_successful_execution(
        self,
        sample_agent_config,
        sample_capabilities,
        mock_model,
        mock_tools,
        sample_state,
    ):
        """Test successful agent execution."""
        wrapper = AgentNodeWrapper(
            agent_name="test_agent",
            agent_config=sample_agent_config,
            model=mock_model,
            tools=mock_tools,
            capabilities=sample_capabilities,
        )

        initial_message_count = len(sample_state["messages"])
        result_state = await wrapper(sample_state)

        assert "test_agent" in result_state["active_agents"]
        assert result_state["last_updated"] is not None
        assert "test_agent_last_execution" in result_state["scratchpad"]
        assert len(result_state["messages"]) > initial_message_count

    @pytest.mark.asyncio
    async def test_validation_failure(
        self, sample_agent_config, sample_capabilities, mock_tools, sample_state
    ):
        """Test execution with validation failure."""
        # Create unavailable model
        unavailable_model = MockModel(available=False)

        wrapper = AgentNodeWrapper(
            agent_name="test_agent",
            agent_config=sample_agent_config,
            model=unavailable_model,
            tools=mock_tools,
            capabilities=sample_capabilities,
        )

        result_state = await wrapper(sample_state)

        assert len(result_state["errors"]) > 0
        assert result_state["errors"][0]["type"] == "validation_failure"

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self, sample_agent_config, sample_capabilities, mock_tools, sample_state
    ):
        """Test handling of model timeout."""
        # Create model that times out
        timeout_model = MockModel(should_timeout=True)

        wrapper = AgentNodeWrapper(
            agent_name="test_agent",
            agent_config=sample_agent_config,
            model=timeout_model,
            tools=mock_tools,
            capabilities=sample_capabilities,
        )

        # Mock asyncio.wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result_state = await wrapper(sample_state)

        assert len(result_state["errors"]) > 0
        assert result_state["errors"][0]["type"] == "model_timeout"

    def test_prepare_messages(
        self,
        sample_agent_config,
        sample_capabilities,
        mock_model,
        mock_tools,
        sample_state,
    ):
        """Test message preparation for model input."""
        wrapper = AgentNodeWrapper(
            agent_name="test_agent",
            agent_config=sample_agent_config,
            model=mock_model,
            tools=mock_tools,
            capabilities=sample_capabilities,
        )

        # Add some scratchpad data
        sample_state["scratchpad"]["other_agent_data"] = {"key": "value"}

        messages = wrapper._prepare_messages(sample_state)

        assert len(messages) >= 2  # System message + existing messages
        assert isinstance(messages[0], SystemMessage)
        assert sample_agent_config.system_prompt in messages[0].content


class TestAgentFactory:
    """Test AgentFactory functionality."""

    def test_factory_creation(self):
        """Test creating AgentFactory."""
        factory = AgentFactory()

        assert factory.model_factory is not None
        assert factory.agent_configs == {}
        assert factory.agent_capabilities == {}
        assert factory.agent_tools == {}

    def test_register_agent_config(
        self, sample_agent_config, sample_capabilities, mock_tools
    ):
        """Test registering agent configuration."""
        factory = AgentFactory()

        factory.register_agent_config(
            sample_agent_config, sample_capabilities, mock_tools
        )

        assert "test_agent" in factory.agent_configs
        assert factory.agent_configs["test_agent"] == sample_agent_config
        assert factory.agent_capabilities["test_agent"] == sample_capabilities
        assert factory.agent_tools["test_agent"] == mock_tools

    @pytest.mark.asyncio
    async def test_create_agent_node_unregistered(self):
        """Test creating agent node for unregistered agent."""
        factory = AgentFactory()

        with pytest.raises(ValueError, match="Agent 'nonexistent' not registered"):
            await factory.create_agent_node("nonexistent")

    @pytest.mark.asyncio
    async def test_create_agent_node_registered(
        self, sample_agent_config, sample_capabilities, mock_tools
    ):
        """Test creating agent node for registered agent."""
        factory = AgentFactory()
        factory.register_agent_config(
            sample_agent_config, sample_capabilities, mock_tools
        )

        # Mock the model factory
        mock_model = MockModel()
        with patch.object(factory.model_factory, "list_models", return_value=[]):
            with patch.object(factory.model_factory, "register_model"):
                with patch.object(
                    factory.model_factory, "get_model", return_value=mock_model
                ):
                    agent_node = await factory.create_agent_node("test_agent")

        assert callable(agent_node)
        assert isinstance(agent_node, AgentNodeWrapper)

    def test_load_agent_configurations(self):
        """Test loading agent configurations from files."""
        factory = AgentFactory()

        # Create temporary config files
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create a valid config file
            config_data = {
                "config": {
                    "name": "test_agent",
                    "display_name": "Test Agent",
                    "description": "Test description",
                    "system_prompt": "Test prompt",
                    "model_provider": "openai",
                    "model_name": "gpt-4",
                },
                "capabilities": {
                    "required_capabilities": ["test"],
                    "optional_capabilities": [],
                    "model_requirements": [],
                    "required_tools": [],
                    "mcp_requirements": [],
                },
            }

            config_file = config_dir / "test_agent.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            factory.load_agent_configurations(config_dir)

            assert "test_agent" in factory.agent_configs
            assert factory.agent_configs["test_agent"].name == "test_agent"

    def test_load_agent_configurations_invalid_file(self):
        """Test loading configurations with invalid JSON file."""
        factory = AgentFactory()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create invalid JSON file
            invalid_file = config_dir / "invalid.json"
            with open(invalid_file, "w", encoding="utf-8") as f:
                f.write("invalid json content")

            # Should not raise exception, just log error
            factory.load_agent_configurations(config_dir)

            assert len(factory.agent_configs) == 0

    def test_list_agents(self, sample_agent_config, sample_capabilities):
        """Test listing registered agents."""
        factory = AgentFactory()
        factory.register_agent_config(sample_agent_config, sample_capabilities)

        agents = factory.list_agents()

        assert "test_agent" in agents
        assert len(agents) == 1

    def test_get_agent_config(self, sample_agent_config, sample_capabilities):
        """Test getting agent configuration."""
        factory = AgentFactory()
        factory.register_agent_config(sample_agent_config, sample_capabilities)

        config = factory.get_agent_config("test_agent")
        assert config == sample_agent_config

        config = factory.get_agent_config("nonexistent")
        assert config is None

    def test_get_agent_capabilities(self, sample_agent_config, sample_capabilities):
        """Test getting agent capabilities."""
        factory = AgentFactory()
        factory.register_agent_config(sample_agent_config, sample_capabilities)

        capabilities = factory.get_agent_capabilities("test_agent")
        assert capabilities == sample_capabilities

        capabilities = factory.get_agent_capabilities("nonexistent")
        assert capabilities is None

    @pytest.mark.asyncio
    async def test_validate_agent_success(
        self, sample_agent_config, sample_capabilities, mock_tools
    ):
        """Test successful agent validation."""
        factory = AgentFactory()
        factory.register_agent_config(
            sample_agent_config, sample_capabilities, mock_tools
        )

        # Mock model factory
        with patch.object(
            factory.model_factory, "list_models", return_value=["test_agent_model"]
        ):
            with patch.object(
                factory.model_factory, "get_model", return_value=MockModel()
            ):
                result = await factory.validate_agent("test_agent")

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_agent_unregistered(self):
        """Test validation of unregistered agent."""
        factory = AgentFactory()

        result = await factory.validate_agent("nonexistent")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not registered" in result["errors"][0]


class TestGlobalFactory:
    """Test global factory instance."""

    def test_get_agent_factory(self):
        """Test getting global agent factory instance."""
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()

        assert factory1 is factory2  # Should be same instance
        assert isinstance(factory1, AgentFactory)


if __name__ == "__main__":
    pytest.main([__file__])
