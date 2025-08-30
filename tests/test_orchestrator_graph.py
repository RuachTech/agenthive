"""Integration tests for multi-agent orchestrator graph."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from agent_hive.core.graphs import OrchestratorGraphFactory, SessionManager
from agent_hive.core.state import AgentState
from agent_hive.agents.factory import AgentFactory, AgentConfig, AgentCapabilities
from agent_hive.core.models import ModelFactory


@pytest_asyncio.fixture
async def mock_model_factory():
    """Create a mock model factory for testing."""
    factory = Mock(spec=ModelFactory)

    # Mock model
    mock_model = AsyncMock()
    mock_model.is_available.return_value = True
    mock_model.model_name = "test-model"
    mock_model.generate.return_value = Mock(
        content="Test response from agent", model_name="test-model", provider="test"
    )

    factory.get_model.return_value = mock_model
    factory.list_models.return_value = ["test_model"]

    return factory


@pytest_asyncio.fixture
async def mock_agent_factory(mock_model_factory):
    """Create a mock agent factory with test agents."""
    factory = AgentFactory(mock_model_factory)

    # Register test agents
    test_agents = [
        {
            "name": "full_stack_engineer",
            "config": AgentConfig(
                name="full_stack_engineer",
                display_name="Full Stack Engineer",
                description="Test full stack engineer",
                system_prompt="You are a full stack engineer",
                model_provider="openai",
                model_name="gpt-4",
                max_tokens=1000,
                temperature=0.3,
                capabilities=["code_execution"],
                specialized_for=["backend_development"],
                core_tools=[],
                composio_tools=[],
                enabled=True,
                priority=1,
            ),
            "capabilities": AgentCapabilities(
                required_capabilities=["code_execution"],
                optional_capabilities=[],
                model_requirements=[],
                required_tools=[],
                mcp_requirements=[],
            ),
        },
        {
            "name": "qa_engineer",
            "config": AgentConfig(
                name="qa_engineer",
                display_name="QA Engineer",
                description="Test QA engineer",
                system_prompt="You are a QA engineer",
                model_provider="openai",
                model_name="gpt-4",
                max_tokens=1000,
                temperature=0.3,
                capabilities=["testing"],
                specialized_for=["quality_assurance"],
                core_tools=[],
                composio_tools=[],
                enabled=True,
                priority=2,
            ),
            "capabilities": AgentCapabilities(
                required_capabilities=["testing"],
                optional_capabilities=[],
                model_requirements=[],
                required_tools=[],
                mcp_requirements=[],
            ),
        },
    ]

    for agent_data in test_agents:
        factory.register_agent_config(agent_data["config"], agent_data["capabilities"])

    return factory


@pytest_asyncio.fixture
async def session_manager():
    """Create a session manager for testing."""
    manager = SessionManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest_asyncio.fixture
async def orchestrator_factory(session_manager, mock_agent_factory):
    """Create an orchestrator factory for testing."""
    factory = OrchestratorGraphFactory(session_manager)
    factory.agent_factory = mock_agent_factory
    return factory


class TestTaskAnalyzer:
    """Test the task analyzer node functionality."""

    @pytest.mark.asyncio
    async def test_analyze_development_task(self, orchestrator_factory):
        """Test task analysis for development-related tasks."""
        state: AgentState = {
            "task": "Build a REST API for user management",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.analyze_task_requirements(state)

        # Check that full stack engineer was identified
        assert "full_stack_engineer" in result["active_agents"]
        assert result["next"] == "full_stack_engineer"

        # Check task analysis in scratchpad
        task_analysis = result["scratchpad"]["task_analysis"]
        assert task_analysis["original_task"] == "Build a REST API for user management"
        assert "full_stack_engineer" in task_analysis["required_agents"]
        assert task_analysis["agent_count"] >= 1

    @pytest.mark.asyncio
    async def test_analyze_testing_task(self, orchestrator_factory):
        """Test task analysis for testing-related tasks."""
        state: AgentState = {
            "task": "Create comprehensive test suite for the application",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.analyze_task_requirements(state)

        # Check that QA engineer was identified
        assert "qa_engineer" in result["active_agents"]
        assert result["next"] == "qa_engineer"

        # Check task analysis
        task_analysis = result["scratchpad"]["task_analysis"]
        assert "qa_engineer" in task_analysis["required_agents"]

    @pytest.mark.asyncio
    async def test_analyze_complex_task(self, orchestrator_factory):
        """Test task analysis for complex tasks requiring multiple agents."""
        state: AgentState = {
            "task": "Build and test a new web application with proper deployment",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.analyze_task_requirements(state)

        # Should identify multiple agents
        assert len(result["active_agents"]) >= 2
        assert "full_stack_engineer" in result["active_agents"]

        # Check task analysis
        task_analysis = result["scratchpad"]["task_analysis"]
        assert task_analysis["agent_count"] >= 2


class TestRouter:
    """Test the router node functionality."""

    @pytest.mark.asyncio
    async def test_route_to_first_agent(self, orchestrator_factory):
        """Test routing to the first agent in the workflow."""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "next": "full_stack_engineer",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer", "qa_engineer"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.route_to_appropriate_agents(state)

        # Should route to next agent
        assert result["next"] == "qa_engineer"

        # Check routing info
        routing_info = result["scratchpad"]["routing_info"]
        assert routing_info["current_step"] == 1
        assert len(routing_info["routing_history"]) == 1
        assert routing_info["routing_history"][0]["agent"] == "full_stack_engineer"

    @pytest.mark.asyncio
    async def test_route_to_coordinator(self, orchestrator_factory):
        """Test routing to coordinator when all agents are processed."""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "next": "qa_engineer",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer", "qa_engineer"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.route_to_appropriate_agents(state)

        # Should route to coordinator (last agent in list)
        assert result["next"] == "coordinator"

        # Check routing completion
        routing_info = result["scratchpad"]["routing_info"]
        assert routing_info["routing_complete"] is True


class TestCoordinator:
    """Test the coordinator node functionality."""

    @pytest.mark.asyncio
    async def test_coordinate_single_agent_response(self, orchestrator_factory):
        """Test coordination with a single agent response."""
        state: AgentState = {
            "task": "Build a simple API",
            "messages": [],
            "next": "coordinator",
            "scratchpad": {
                "full_stack_engineer_analysis": {
                    "response_length": 500,
                    "model_used": "gpt-4",
                    "provider": "openai",
                    "timestamp": datetime.now().isoformat(),
                }
            },
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.coordinate_agent_responses(state)

        # Check that coordination message was added
        assert len(result["messages"]) == 1
        message_content = result["messages"][0].content
        assert "Multi-Agent Task Completion Report" in message_content
        assert "Full Stack Engineer Analysis" in message_content

        # Check task status
        assert result["task_status"]["status"] == "completed"
        assert result["task_status"]["coordination_complete"] is True

        # Check coordination info
        coordination_info = result["scratchpad"]["coordination_info"]
        assert coordination_info["synthesis_complete"] is True
        assert coordination_info["participating_agents"] == ["full_stack_engineer"]

    @pytest.mark.asyncio
    async def test_coordinate_multiple_agent_responses(self, orchestrator_factory):
        """Test coordination with multiple agent responses."""
        state: AgentState = {
            "task": "Build and test an application",
            "messages": [],
            "next": "coordinator",
            "scratchpad": {
                "full_stack_engineer_analysis": {
                    "response_length": 800,
                    "model_used": "gpt-4",
                    "provider": "openai",
                    "timestamp": datetime.now().isoformat(),
                },
                "qa_engineer_analysis": {
                    "response_length": 600,
                    "model_used": "gpt-4",
                    "provider": "openai",
                    "timestamp": datetime.now().isoformat(),
                },
            },
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer", "qa_engineer"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        result = await orchestrator_factory.coordinate_agent_responses(state)

        # Check that both agents are mentioned in the response
        message_content = result["messages"][0].content
        assert "Full Stack Engineer Analysis" in message_content
        assert "Qa Engineer Analysis" in message_content

        # Check coordination info
        coordination_info = result["scratchpad"]["coordination_info"]
        assert len(coordination_info["participating_agents"]) == 2


class TestOrchestratorGraph:
    """Test the complete orchestrator graph functionality."""

    @pytest.mark.asyncio
    async def test_create_orchestrator_graph(self, orchestrator_factory):
        """Test creation of the orchestrator graph."""
        graph = await orchestrator_factory.create_orchestrator_graph()

        # Check that graph was created
        assert graph is not None

        # Check that graph has expected structure
        graph_dict = graph.get_graph().to_json()
        nodes = [node["id"] for node in graph_dict["nodes"]]

        # Should have orchestration nodes
        assert "task_analyzer" in nodes
        assert "router" in nodes
        assert "coordinator" in nodes

        # Should have agent nodes
        assert "full_stack_engineer" in nodes
        assert "qa_engineer" in nodes

    @pytest.mark.asyncio
    async def test_determine_next_agent(self, orchestrator_factory):
        """Test the next agent determination logic."""
        # Test routing to valid agent
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "next": "full_stack_engineer",
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        next_agent = orchestrator_factory.determine_next_agent(state)
        assert next_agent == "full_stack_engineer"

        # Test routing to coordinator
        state["next"] = "coordinator"
        next_agent = orchestrator_factory.determine_next_agent(state)
        assert next_agent == "coordinator"

        # Test routing to invalid agent (should default to coordinator)
        state["next"] = "invalid_agent"
        next_agent = orchestrator_factory.determine_next_agent(state)
        assert next_agent == "coordinator"

    @pytest.mark.asyncio
    async def test_execute_orchestration(self, orchestrator_factory):
        """Test complete orchestration execution."""
        session_id = "test-orchestration-session"
        task = "Build a simple web application"

        # Mock the graph execution
        with patch.object(
            orchestrator_factory, "create_orchestrator_graph"
        ) as mock_create_graph:
            mock_graph = AsyncMock()
            mock_result = {
                "task": task,
                "messages": [Mock(content="Orchestration complete")],
                "active_agents": ["full_stack_engineer"],
                "task_status": {"status": "completed"},
                "scratchpad": {"coordination_info": {"synthesis_complete": True}},
                "last_updated": datetime.now(),
                "errors": [],
            }
            mock_graph.ainvoke.return_value = mock_result
            mock_create_graph.return_value = mock_graph

            result = await orchestrator_factory.execute_orchestration(
                session_id=session_id, task=task, user_id="test-user"
            )

            # Check execution result
            assert result["status"] == "success"
            assert result["mode"] == "orchestration"
            assert result["session_id"] == session_id
            assert "response" in result

            # Check orchestration info
            orchestration_info = result["orchestration_info"]
            assert orchestration_info["participating_agents"] == ["full_stack_engineer"]
            assert orchestration_info["coordination_complete"] is True

    @pytest.mark.asyncio
    async def test_stream_orchestration(self, orchestrator_factory):
        """Test streaming orchestration execution."""
        session_id = "test-stream-session"
        task = "Create a test application"

        # Mock the graph streaming
        with patch.object(
            orchestrator_factory, "create_orchestrator_graph"
        ) as mock_create_graph:
            mock_graph = AsyncMock()

            # Mock streaming chunks
            async def mock_astream(state, config):
                yield {
                    "task_analyzer": {
                        "task": task,
                        "active_agents": ["full_stack_engineer"],
                        "scratchpad": {
                            "task_analysis": {
                                "required_agents": ["full_stack_engineer"]
                            }
                        },
                    }
                }
                yield {
                    "full_stack_engineer": {
                        "messages": [Mock(content="Development analysis complete")],
                        "task_status": {"status": "in_progress"},
                    }
                }
                yield {
                    "coordinator": {
                        "messages": [Mock(content="Orchestration complete")],
                        "task_status": {"status": "completed"},
                    }
                }

            mock_graph.astream = mock_astream
            mock_create_graph.return_value = mock_graph

            # Collect streaming results
            results = []
            async for chunk in orchestrator_factory.stream_orchestration(
                session_id=session_id, task=task, user_id="test-user"
            ):
                results.append(chunk)

            # Check streaming results
            assert len(results) > 0

            # Should have progress updates
            progress_chunks = [r for r in results if r.get("type") == "progress"]
            assert len(progress_chunks) >= 3  # task_analyzer, agent, coordinator

            # Should have content updates
            content_chunks = [r for r in results if r.get("type") == "content"]
            assert len(content_chunks) >= 1

            # Should have completion signal
            complete_chunks = [r for r in results if r.get("type") == "complete"]
            assert len(complete_chunks) == 1
            assert complete_chunks[0]["mode"] == "orchestration"


class TestErrorHandling:
    """Test error handling in orchestrator graph."""

    @pytest.mark.asyncio
    async def test_orchestration_execution_error(self, orchestrator_factory):
        """Test error handling during orchestration execution."""
        session_id = "test-error-session"
        task = "Test task that will fail"

        # Mock graph to raise an exception
        with patch.object(
            orchestrator_factory, "create_orchestrator_graph"
        ) as mock_create_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.side_effect = Exception("Test orchestration error")
            mock_create_graph.return_value = mock_graph

            result = await orchestrator_factory.execute_orchestration(
                session_id=session_id, task=task, user_id="test-user"
            )

            # Check error handling
            assert result["status"] == "error"
            assert "Test orchestration error" in result["error"]
            assert "Orchestration failed" in result["response"]

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, orchestrator_factory):
        """Test error handling during streaming orchestration."""
        session_id = "test-stream-error-session"
        task = "Test streaming error"

        # Mock graph to raise an exception during streaming
        with patch.object(
            orchestrator_factory, "create_orchestrator_graph"
        ) as mock_create_graph:
            mock_graph = AsyncMock()

            async def mock_astream_error(state, config):
                yield {"task_analyzer": {"task": task}}
                raise Exception("Test streaming error")

            mock_graph.astream = mock_astream_error
            mock_create_graph.return_value = mock_graph

            # Collect streaming results
            results = []
            async for chunk in orchestrator_factory.stream_orchestration(
                session_id=session_id, task=task, user_id="test-user"
            ):
                results.append(chunk)

            # Should have error chunk
            error_chunks = [r for r in results if r.get("type") == "error"]
            assert len(error_chunks) == 1
            assert "Test streaming error" in error_chunks[0]["error"]


if __name__ == "__main__":
    pytest.main([__file__])
