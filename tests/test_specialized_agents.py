"""Unit tests for specialized agent configurations."""

from pathlib import Path
from typing import Any, AsyncIterator, List, Sequence, Set
from unittest.mock import patch

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from agent_hive.agents.factory import AgentFactory
from agent_hive.core.models import ModelInterface, ModelResponse
from agent_hive.core.state import AgentState


class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str
    description: str

    def __init__(self, tool_name: str) -> None:
        super().__init__()
        self.name = tool_name
        self.description = f"Mock {tool_name} tool"

    def _run(self, query: str) -> str:
        return f"Mock result from {self.name}"

    async def _arun(self, query: str) -> str:
        return f"Mock async result from {self.name}"


class MockModel(ModelInterface):
    """Mock model for testing specialized agents."""

    def __init__(self, model_name: str = "mock-model", provider: str = "mock") -> None:
        from agent_hive.core.models import ModelConfig, ModelProvider

        config = ModelConfig(
            provider=ModelProvider.OPENAI, model_name=model_name, api_key="mock-key"
        )
        super().__init__(config)
        self.model_name = model_name
        self.provider = provider
        self._available = True

    async def generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> ModelResponse:
        """Generate mock response based on agent type."""
        return ModelResponse(
            content=f"Mock response from {self.model_name}",
            provider=self.provider,
            model_name=self.model_name,
        )

    def stream_generate(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Mock streaming response."""
        return self._stream_impl()

    async def _stream_impl(self) -> AsyncIterator[str]:
        """Implementation of streaming."""
        yield f"Mock stream from {self.model_name}"

    async def is_available(self) -> bool:
        """Check if model is available."""
        return self._available


class TestSpecializedAgentConfigurations:
    """Test specialized agent configurations and their capabilities."""

    @pytest.fixture
    def agent_configs_dir(self) -> Path:
        """Get the agent configurations directory."""
        return Path("agent_configs")

    @pytest.fixture
    def agent_factory(self) -> AgentFactory:
        """Create agent factory for testing."""
        return AgentFactory()

    def test_agent_config_files_exist(self, agent_configs_dir: Path) -> None:
        """Test that all required agent configuration files exist."""
        required_agents = [
            "full_stack_engineer.json",
            "qa_engineer.json",
            "product_designer.json",
            "devops_engineer.json",
        ]

        for agent_file in required_agents:
            config_path = agent_configs_dir / agent_file
            assert (
                config_path.exists()
            ), f"Agent config file {agent_file} does not exist"

    def test_load_all_agent_configurations(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test loading all agent configurations from files."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        expected_agents = [
            "full_stack_engineer",
            "qa_engineer",
            "product_designer",
            "devops_engineer",
        ]
        loaded_agents = agent_factory.list_agents()

        for agent_name in expected_agents:
            assert agent_name in loaded_agents, f"Agent {agent_name} was not loaded"

    def test_full_stack_engineer_configuration(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test Full Stack Engineer agent configuration."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        config = agent_factory.get_agent_config("full_stack_engineer")
        capabilities = agent_factory.get_agent_capabilities("full_stack_engineer")

        assert config is not None
        assert capabilities is not None

        # Test configuration values
        assert config.name == "full_stack_engineer"
        assert config.display_name == "Full Stack Engineer"
        assert "full-stack development" in config.description.lower()
        assert "writing clean code" in config.system_prompt.lower()
        assert config.model_provider in ["openai", "anthropic", "google"]

        # Test capabilities
        assert "code_execution" in capabilities.required_capabilities
        assert "github" in capabilities.mcp_requirements

        # Test specialized areas
        expected_specializations = [
            "backend_development",
            "frontend_development",
            "api_design",
        ]
        for spec in expected_specializations:
            assert spec in config.specialized_for

        # Test Composio tools
        expected_composio_tools = ["github", "linear", "slack", "notion"]
        for tool in expected_composio_tools:
            assert tool in config.composio_tools

    def test_system_prompts_quality(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test that agent system prompts are comprehensive and role-specific."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        # Test Full Stack Engineer prompt
        fs_config = agent_factory.get_agent_config("full_stack_engineer")
        assert fs_config is not None
        fs_prompt = fs_config.system_prompt.lower()
        assert "full stack engineer" in fs_prompt
        assert "code" in fs_prompt
        assert "architecture" in fs_prompt
        assert "debugging" in fs_prompt

        # Test QA Engineer prompt
        qa_config = agent_factory.get_agent_config("qa_engineer")
        assert qa_config is not None
        qa_prompt = qa_config.system_prompt.lower()
        assert "qa engineer" in qa_prompt
        assert "test" in qa_prompt
        assert "quality" in qa_prompt
        assert "bug" in qa_prompt

        # Test Product Designer prompt
        pd_config = agent_factory.get_agent_config("product_designer")
        assert pd_config is not None
        pd_prompt = pd_config.system_prompt.lower()
        assert "product designer" in pd_prompt
        assert "design" in pd_prompt
        assert "user" in pd_prompt
        assert "accessibility" in pd_prompt

        # Test DevOps Engineer prompt
        do_config = agent_factory.get_agent_config("devops_engineer")
        assert do_config is not None
        do_prompt = do_config.system_prompt.lower()
        assert "devops engineer" in do_prompt
        assert "infrastructure" in do_prompt
        assert "deployment" in do_prompt
        assert "security" in do_prompt

    def test_agent_tool_mappings(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test that agents have appropriate tool mappings for their roles."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        # Full Stack Engineer should have development tools
        fs_config = agent_factory.get_agent_config("full_stack_engineer")
        assert fs_config is not None
        assert "github" in fs_config.composio_tools
        assert "linear" in fs_config.composio_tools
        assert "notion" in fs_config.composio_tools

        # QA Engineer should have testing tools
        qa_config = agent_factory.get_agent_config("qa_engineer")
        assert qa_config is not None
        assert "github" in qa_config.composio_tools
        assert "linear" in qa_config.composio_tools
        assert "browserbase" in qa_config.composio_tools
        assert "serpapi" in qa_config.composio_tools

        # Product Designer should have design tools
        pd_config = agent_factory.get_agent_config("product_designer")
        assert pd_config is not None
        assert "figma" in pd_config.composio_tools
        assert "miro" in pd_config.composio_tools
        assert "notion" in pd_config.composio_tools

        # DevOps Engineer should have infrastructure tools
        do_config = agent_factory.get_agent_config("devops_engineer")
        assert do_config is not None
        assert "aws" in do_config.composio_tools
        assert "docker" in do_config.composio_tools
        assert "kubernetes" in do_config.composio_tools
        assert "datadog" in do_config.composio_tools

    def test_agent_model_requirements(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test that agents have appropriate model requirements for their capabilities."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        # Product Designer should require vision capabilities
        pd_capabilities = agent_factory.get_agent_capabilities("product_designer")
        assert pd_capabilities is not None
        assert (
            "vision" in pd_capabilities.model_requirements
            or "multimodal" in pd_capabilities.model_requirements
        )

        # Check that agents have appropriate model requirements
        for agent_name in ["full_stack_engineer", "qa_engineer", "devops_engineer"]:
            capabilities = agent_factory.get_agent_capabilities(agent_name)
            assert capabilities is not None
            assert (
                "function_calling" in capabilities.model_requirements
                or "reasoning" in capabilities.model_requirements
            )

        # Product designer has vision requirements
        pd_capabilities = agent_factory.get_agent_capabilities("product_designer")
        assert pd_capabilities is not None
        assert (
            "vision" in pd_capabilities.model_requirements
            or "multimodal" in pd_capabilities.model_requirements
        )

    def test_agent_configuration_completeness(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test that all agent configurations are complete and valid."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        required_fields = [
            "name",
            "display_name",
            "description",
            "system_prompt",
            "model_provider",
            "model_name",
        ]

        for agent_name in [
            "full_stack_engineer",
            "qa_engineer",
            "product_designer",
            "devops_engineer",
        ]:
            config = agent_factory.get_agent_config(agent_name)
            capabilities = agent_factory.get_agent_capabilities(agent_name)

            assert config is not None
            assert capabilities is not None

            # Check required config fields
            for field in required_fields:
                assert hasattr(
                    config, field
                ), f"Agent {agent_name} missing field {field}"
                assert getattr(
                    config, field
                ), f"Agent {agent_name} has empty field {field}"

            # Check capabilities structure
            assert (
                capabilities.required_capabilities
            ), f"Agent {agent_name} has no required capabilities"
            assert (
                capabilities.mcp_requirements
            ), f"Agent {agent_name} has no MCP requirements"

            # Check that system prompt is substantial
            assert (
                len(config.system_prompt) > 100
            ), f"Agent {agent_name} system prompt too short"

            # Check that description is informative
            assert (
                len(config.description) > 50
            ), f"Agent {agent_name} description too short"

    @pytest.mark.asyncio
    async def test_agent_execution_with_state(
        self, agent_factory: AgentFactory, agent_configs_dir: Path
    ) -> None:
        """Test that agents can execute with proper state management."""
        agent_factory.load_agent_configurations(agent_configs_dir)

        # Add mock tools for full stack engineer
        mock_tools: List[BaseTool] = [
            MockTool("code_executor"),
            MockTool("file_manager"),
            MockTool("git_operations"),
        ]
        agent_factory.agent_tools["full_stack_engineer"] = mock_tools

        # Create test state
        test_state = AgentState(
            task="Test task for agent execution",
            messages=[],
            next="",
            scratchpad={},
            mode="direct",
            active_agents=[],
            multimodal_content={},
            session_id="test_session",
            user_id="test_user",
            last_updated=None,
            errors=[],
            task_status={},
        )

        # Test Full Stack Engineer execution
        mock_model = MockModel("gpt-4", "openai")
        with patch.object(agent_factory.model_factory, "list_models", return_value=[]):
            with patch.object(agent_factory.model_factory, "register_model"):
                with patch.object(
                    agent_factory.model_factory, "get_model", return_value=mock_model
                ):
                    fs_agent = await agent_factory.create_agent_node(
                        "full_stack_engineer"
                    )

                    # Execute agent
                    result_state = await fs_agent(test_state)

                    # Verify state updates
                    assert "full_stack_engineer" in result_state["active_agents"]
                    assert result_state["last_updated"] is not None
                    assert (
                        "full_stack_engineer_last_execution"
                        in result_state["scratchpad"]
                    )


class TestAgentConfigurationIntegration:
    """Test integration between agent configurations and other system components."""

    @pytest.fixture
    def sample_state(self) -> AgentState:
        """Create sample state for testing."""
        return AgentState(
            task="Integration test task",
            messages=[],
            next="",
            scratchpad={},
            mode="orchestration",
            active_agents=[],
            multimodal_content={},
            session_id="integration_test",
            user_id="test_user",
            last_updated=None,
            errors=[],
            task_status={},
        )

    @pytest.mark.asyncio
    async def test_multi_agent_compatibility(self, sample_state: AgentState) -> None:
        """Test that all agents can work together in orchestration mode."""
        factory = AgentFactory()
        factory.load_agent_configurations(Path("agent_configs"))

        agents = [
            "full_stack_engineer",
            "qa_engineer",
            "product_designer",
            "devops_engineer",
        ]

        # Create mock tools for each agent
        tool_mapping = {
            "full_stack_engineer": ["code_executor", "file_manager", "git_operations"],
            "qa_engineer": ["test_runner", "browser_automation"],
            "product_designer": [
                "image_analyzer",
                "color_analyzer",
                "design_validator",
            ],
            "devops_engineer": [
                "infrastructure_manager",
                "security_scanner",
                "deployment_manager",
            ],
        }

        # Mock all agents
        agent_nodes = {}
        for agent_name in agents:
            # Add mock tools for this agent
            mock_tools: List[BaseTool] = [
                MockTool(tool_name) for tool_name in tool_mapping[agent_name]
            ]
            factory.agent_tools[agent_name] = mock_tools

            mock_model = MockModel(f"model-{agent_name}", "mock")
            with patch.object(factory.model_factory, "list_models", return_value=[]):
                with patch.object(factory.model_factory, "register_model"):
                    with patch.object(
                        factory.model_factory, "get_model", return_value=mock_model
                    ):
                        agent_nodes[agent_name] = await factory.create_agent_node(
                            agent_name
                        )

        # Test that all agents can process the same state
        for agent_name, agent_node in agent_nodes.items():
            test_state = sample_state.copy()

            # Add multimodal content for product designer
            if agent_name == "product_designer":
                test_state["multimodal_content"] = {
                    "images": [
                        {
                            "id": "test_image",
                            "type": "image",
                            "content": "mock_image_data",
                        }
                    ]
                }

            result_state = await agent_node(test_state)
            assert agent_name in result_state["active_agents"]

    def test_agent_specialization_mapping(self) -> None:
        """Test that agent specializations are properly mapped to use cases."""
        factory = AgentFactory()
        factory.load_agent_configurations(Path("agent_configs"))

        # Define expected specialization mappings
        specialization_map = {
            "full_stack_engineer": [
                "backend_development",
                "frontend_development",
                "api_design",
            ],
            "qa_engineer": ["test_planning", "bug_analysis", "quality_assurance"],
            "product_designer": ["ui_design", "ux_design", "design_systems"],
            "devops_engineer": ["infrastructure", "deployment", "monitoring"],
        }

        for agent_name, expected_specs in specialization_map.items():
            config = factory.get_agent_config(agent_name)
            assert config is not None
            for spec in expected_specs:
                assert (
                    spec in config.specialized_for
                ), f"Agent {agent_name} missing specialization {spec}"

    def test_composio_tool_coverage(self) -> None:
        """Test that Composio tools provide comprehensive coverage for agent needs."""
        factory = AgentFactory()
        factory.load_agent_configurations(Path("agent_configs"))

        # Collect all Composio tools used by agents
        all_tools: Set[str] = set()
        for agent_name in factory.list_agents():
            config = factory.get_agent_config(agent_name)
            assert config is not None
            all_tools.update(config.composio_tools)

        # Verify essential tools are covered
        essential_tools = ["github", "linear", "slack", "figma", "aws", "notion"]
        for tool in essential_tools:
            assert tool in all_tools, f"Essential tool {tool} not covered by any agent"

    def test_agent_priority_and_routing(self) -> None:
        """Test agent priority settings for routing decisions."""
        factory = AgentFactory()
        factory.load_agent_configurations(Path("agent_configs"))

        # Load configurations and check priorities
        configs = {}
        for agent_name in factory.list_agents():
            config = factory.get_agent_config(agent_name)
            assert config is not None
            configs[agent_name] = config

        # Verify that agents have priority settings
        for agent_name, config in configs.items():
            assert hasattr(
                config, "priority"
            ), f"Agent {agent_name} missing priority setting"
            assert isinstance(
                config.priority, int
            ), f"Agent {agent_name} priority must be integer"
            assert (
                1 <= config.priority <= 5
            ), f"Agent {agent_name} priority must be between 1-5"


if __name__ == "__main__":
    pytest.main([__file__])
