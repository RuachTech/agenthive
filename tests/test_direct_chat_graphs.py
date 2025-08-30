"""Tests for single-agent direct chat graph factory."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_hive.core.graphs import (
    DirectChatGraphFactory,
    SessionManager,
    Session,
    GraphValidationError,
    get_session_manager,
    get_graph_factory,
    cleanup_global_instances,
)
from agent_hive.core.state import AgentState
from agent_hive.core.config import SystemConfig, AgentConfig
from agent_hive.agents.factory import AgentCapabilities


class TestSession:
    """Test Session class functionality."""

    def test_session_creation(self):
        """Test session creation with all fields."""
        now = datetime.now()
        state: AgentState = {
            "task": "test task",
            "messages": [],
            "next": "test_agent",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": "test_123",
            "user_id": "user_123",
            "last_updated": now,
            "errors": [],
            "task_status": {"status": "initialized"},
        }

        session = Session(
            session_id="test_123",
            user_id="user_123",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        assert session.session_id == "test_123"
        assert session.user_id == "user_123"
        assert session.mode == "direct"
        assert session.active_agent == "test_agent"
        assert session.state["task"] == "test task"
        assert len(session.multimodal_files) == 0

    def test_session_expiry_check(self):
        """Test session expiry logic."""
        now = datetime.now()
        old_time = now - timedelta(hours=2)

        state: AgentState = {
            "task": "",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": "test_123",
            "user_id": None,
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id="test_123",
            user_id="user_123",
            mode="direct",
            active_agent=None,
            created_at=now,
            last_activity=old_time,
            state=state,
            multimodal_files=[],
        )

        # Should be expired with 1 hour timeout
        assert session.is_expired(3600) is True

        # Should not be expired with 3 hour timeout
        assert session.is_expired(10800) is False

    def test_update_activity(self):
        """Test activity timestamp update."""
        now = datetime.now()
        old_time = now - timedelta(minutes=30)

        state: AgentState = {
            "task": "",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": "test_123",
            "user_id": None,
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id="test_123",
            user_id="user_123",
            mode="direct",
            active_agent=None,
            created_at=now,
            last_activity=old_time,
            state=state,
            multimodal_files=[],
        )

        original_activity = session.last_activity
        session.update_activity()

        assert session.last_activity > original_activity


class TestSessionManager:
    """Test SessionManager functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create a session manager for testing."""
        config = SystemConfig(session_timeout=3600)
        return SessionManager(config)

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation."""
        session = await session_manager.create_session(
            session_id="test_123",
            user_id="user_123",
            mode="direct",
            active_agent="test_agent",
            initial_task="Hello world",
        )

        assert session.session_id == "test_123"
        assert session.user_id == "user_123"
        assert session.mode == "direct"
        assert session.active_agent == "test_agent"
        assert session.state["task"] == "Hello world"
        assert session.state["mode"] == "direct"
        assert "test_agent" in session.state["active_agents"]

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test session retrieval."""
        # Create session
        await session_manager.create_session(session_id="test_123", user_id="user_123")

        # Retrieve session
        session = await session_manager.get_session("test_123")
        assert session is not None
        assert session.session_id == "test_123"

        # Try to get non-existent session
        missing_session = await session_manager.get_session("missing_123")
        assert missing_session is None

    @pytest.mark.asyncio
    async def test_update_session_state(self, session_manager):
        """Test session state updates."""
        # Create session
        await session_manager.create_session(session_id="test_123", user_id="user_123")

        # Update state
        new_state: AgentState = {
            "task": "updated task",
            "messages": [HumanMessage(content="test")],
            "next": "next_agent",
            "scratchpad": {"key": "value"},
            "mode": "direct",
            "active_agents": ["agent1", "agent2"],
            "multimodal_content": {},
            "session_id": "test_123",
            "user_id": "user_123",
            "last_updated": datetime.now(),
            "errors": [],
            "task_status": {"status": "updated"},
        }

        success = await session_manager.update_session_state("test_123", new_state)
        assert success is True

        # Verify update
        session = await session_manager.get_session("test_123")
        assert session.state["task"] == "updated task"
        assert len(session.state["messages"]) == 1
        assert session.state["scratchpad"]["key"] == "value"

        # Try to update non-existent session
        success = await session_manager.update_session_state("missing_123", new_state)
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager):
        """Test session deletion."""
        # Create session
        await session_manager.create_session(session_id="test_123", user_id="user_123")

        # Verify it exists
        session = await session_manager.get_session("test_123")
        assert session is not None

        # Delete session
        success = await session_manager.delete_session("test_123")
        assert success is True

        # Verify it's gone
        session = await session_manager.get_session("test_123")
        assert session is None

        # Try to delete non-existent session
        success = await session_manager.delete_session("missing_123")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_sessions(self, session_manager):
        """Test session listing."""
        # Create multiple sessions
        await session_manager.create_session("session_1", "user_1")
        await session_manager.create_session("session_2", "user_1")
        await session_manager.create_session("session_3", "user_2")

        # List all sessions
        all_sessions = await session_manager.list_sessions()
        assert len(all_sessions) == 3

        # List sessions for specific user
        user1_sessions = await session_manager.list_sessions("user_1")
        assert len(user1_sessions) == 2
        assert all(s.user_id == "user_1" for s in user1_sessions)

        user2_sessions = await session_manager.list_sessions("user_2")
        assert len(user2_sessions) == 1
        assert user2_sessions[0].user_id == "user_2"

    @pytest.mark.asyncio
    async def test_expired_session_cleanup(self, session_manager):
        """Test that expired sessions are not returned."""
        # Create session with short timeout
        session_manager.config.session_timeout = 1  # 1 second

        await session_manager.create_session("test_123", "user_123")

        # Session should exist immediately
        session = await session_manager.get_session("test_123")
        assert session is not None

        # Wait for expiry
        await asyncio.sleep(2)

        # Session should be cleaned up
        session = await session_manager.get_session("test_123")
        assert session is None


class TestDirectChatGraphFactory:
    """Test DirectChatGraphFactory functionality."""

    @pytest.fixture
    def mock_agent_factory(self):
        """Create a mock agent factory."""
        factory = Mock()
        factory.validate_agent = AsyncMock(
            return_value={
                "valid": True,
                "errors": [],
                "config": Mock(),
                "capabilities": Mock(),
            }
        )
        factory.create_agent_node = AsyncMock(return_value=Mock())
        factory.list_agents = Mock(return_value=["test_agent", "another_agent"])
        factory.get_agent_config = Mock(
            return_value=AgentConfig(
                name="test_agent",
                display_name="Test Agent",
                description="A test agent",
                system_prompt="You are a test agent",
                model_provider="openai",
                model_name="gpt-4",
            )
        )
        factory.get_agent_capabilities = Mock(
            return_value=AgentCapabilities(
                required_capabilities=["chat"],
                optional_capabilities=[],
                model_requirements=[],
                required_tools=[],
                mcp_requirements=[],
            )
        )
        return factory

    @pytest.fixture
    def session_manager(self):
        """Create a session manager for testing."""
        return SessionManager(SystemConfig(session_timeout=3600))

    @pytest.fixture
    def graph_factory(self, session_manager, mock_agent_factory):
        """Create a graph factory for testing."""
        factory = DirectChatGraphFactory(session_manager)
        factory.agent_factory = mock_agent_factory
        return factory

    @pytest.mark.asyncio
    async def test_create_direct_chat_graph_success(
        self, graph_factory, mock_agent_factory
    ):
        """Test successful graph creation."""

        # Mock agent node that returns updated state
        async def mock_agent_node(state):
            state["messages"].append(AIMessage(content="Test response"))
            return state

        mock_agent_factory.create_agent_node.return_value = mock_agent_node

        # Create graph
        graph = await graph_factory.create_direct_chat_graph("test_agent")

        assert graph is not None
        mock_agent_factory.validate_agent.assert_called_once_with("test_agent")
        mock_agent_factory.create_agent_node.assert_called_once_with("test_agent")

    @pytest.mark.asyncio
    async def test_create_direct_chat_graph_validation_failure(
        self, graph_factory, mock_agent_factory
    ):
        """Test graph creation with agent validation failure."""
        mock_agent_factory.validate_agent.return_value = {
            "valid": False,
            "errors": ["Agent not found", "Model unavailable"],
        }

        with pytest.raises(GraphValidationError) as exc_info:
            await graph_factory.create_direct_chat_graph("invalid_agent")

        assert "validation failed" in str(exc_info.value)
        assert "Agent not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graph_caching(self, graph_factory, mock_agent_factory):
        """Test that graphs are cached after creation."""
        # Mock agent node
        mock_agent_factory.create_agent_node.return_value = Mock()

        # Create graph twice
        graph1 = await graph_factory.create_direct_chat_graph("test_agent")
        graph2 = await graph_factory.create_direct_chat_graph("test_agent")

        # Should be the same instance (cached)
        assert graph1 is graph2

        # Agent factory should only be called once
        assert mock_agent_factory.create_agent_node.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_direct_chat_new_session(
        self, graph_factory, mock_agent_factory
    ):
        """Test direct chat execution with new session."""

        # Mock agent node that adds response
        async def mock_agent_node(state):
            state["messages"].append(AIMessage(content="Hello! How can I help you?"))
            return state

        mock_agent_factory.create_agent_node.return_value = mock_agent_node

        # Mock graph execution
        with patch.object(
            graph_factory, "create_direct_chat_graph"
        ) as mock_create_graph:
            mock_graph = Mock()
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "task": "Hello",
                    "messages": [
                        HumanMessage(content="Hello"),
                        AIMessage(content="Hello! How can I help you?"),
                    ],
                    "next": "",
                    "scratchpad": {},
                    "mode": "direct",
                    "active_agents": ["test_agent"],
                    "multimodal_content": {},
                    "session_id": "test_123",
                    "user_id": "user_123",
                    "last_updated": datetime.now(),
                    "errors": [],
                    "task_status": {},
                }
            )
            mock_create_graph.return_value = mock_graph

            result = await graph_factory.execute_direct_chat(
                session_id="test_123",
                agent_name="test_agent",
                message="Hello",
                user_id="user_123",
            )

            assert result["status"] == "success"
            assert result["response"] == "Hello! How can I help you?"
            assert result["session_id"] == "test_123"
            assert result["agent_name"] == "test_agent"

            # Verify session was created
            session = await graph_factory.session_manager.get_session("test_123")
            assert session is not None
            assert session.user_id == "user_123"
            assert session.mode == "direct"

    @pytest.mark.asyncio
    async def test_execute_direct_chat_existing_session(
        self, graph_factory, mock_agent_factory
    ):
        """Test direct chat execution with existing session."""
        # Create existing session
        await graph_factory.session_manager.create_session(
            session_id="existing_123",
            user_id="user_123",
            mode="direct",
            active_agent="test_agent",
        )

        # Mock agent node
        async def mock_agent_node(state):
            state["messages"].append(AIMessage(content="Follow-up response"))
            return state

        mock_agent_factory.create_agent_node.return_value = mock_agent_node

        # Mock graph execution
        with patch.object(
            graph_factory, "create_direct_chat_graph"
        ) as mock_create_graph:
            mock_graph = Mock()
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "task": "Follow-up question",
                    "messages": [
                        HumanMessage(content="Follow-up question"),
                        AIMessage(content="Follow-up response"),
                    ],
                    "next": "",
                    "scratchpad": {},
                    "mode": "direct",
                    "active_agents": ["test_agent"],
                    "multimodal_content": {},
                    "session_id": "existing_123",
                    "user_id": "user_123",
                    "last_updated": datetime.now(),
                    "errors": [],
                    "task_status": {},
                }
            )
            mock_create_graph.return_value = mock_graph

            result = await graph_factory.execute_direct_chat(
                session_id="existing_123",
                agent_name="test_agent",
                message="Follow-up question",
                user_id="user_123",
            )

            assert result["status"] == "success"
            assert result["response"] == "Follow-up response"

    @pytest.mark.asyncio
    async def test_execute_direct_chat_error_handling(
        self, graph_factory, mock_agent_factory
    ):
        """Test error handling in direct chat execution."""
        # Mock agent node
        mock_agent_factory.create_agent_node.return_value = Mock()

        # Mock graph that raises exception
        with patch.object(
            graph_factory, "create_direct_chat_graph"
        ) as mock_create_graph:
            mock_graph = Mock()
            mock_graph.ainvoke = AsyncMock(
                side_effect=Exception("Graph execution failed")
            )
            mock_create_graph.return_value = mock_graph

            result = await graph_factory.execute_direct_chat(
                session_id="error_123",
                agent_name="test_agent",
                message="This will fail",
                user_id="user_123",
            )

            assert result["status"] == "error"
            assert "Graph execution failed" in result["error"]
            assert "I encountered an error" in result["response"]

            # Verify error was recorded in session
            session = await graph_factory.session_manager.get_session("error_123")
            assert session is not None
            assert len(session.state["errors"]) > 0
            assert session.state["errors"][0]["type"] == "execution_error"

    @pytest.mark.asyncio
    async def test_stream_direct_chat(self, graph_factory, mock_agent_factory):
        """Test streaming direct chat execution."""
        # Mock agent node
        mock_agent_factory.create_agent_node.return_value = Mock()

        # Mock streaming graph
        with patch.object(
            graph_factory, "create_direct_chat_graph"
        ) as mock_create_graph:
            mock_graph = Mock()

            # Mock streaming response
            async def mock_astream(state, config):
                yield {
                    "test_agent": {
                        "messages": [AIMessage(content="Streaming response part 1")],
                        "task": state["task"],
                        "next": "",
                        "scratchpad": {},
                        "mode": "direct",
                        "active_agents": ["test_agent"],
                        "multimodal_content": {},
                        "session_id": state["session_id"],
                        "user_id": state.get("user_id"),
                        "last_updated": datetime.now(),
                        "errors": [],
                        "task_status": {},
                    }
                }
                yield {
                    "test_agent": {
                        "messages": [AIMessage(content="Streaming response part 2")],
                        "task": state["task"],
                        "next": "",
                        "scratchpad": {},
                        "mode": "direct",
                        "active_agents": ["test_agent"],
                        "multimodal_content": {},
                        "session_id": state["session_id"],
                        "user_id": state.get("user_id"),
                        "last_updated": datetime.now(),
                        "errors": [],
                        "task_status": {},
                    }
                }

            mock_graph.astream = mock_astream
            mock_create_graph.return_value = mock_graph

            chunks = []
            async for chunk in graph_factory.stream_direct_chat(
                session_id="stream_123",
                agent_name="test_agent",
                message="Stream this",
                user_id="user_123",
            ):
                chunks.append(chunk)

            # Should have content chunks and completion
            content_chunks = [c for c in chunks if c["type"] == "content"]
            complete_chunks = [c for c in chunks if c["type"] == "complete"]

            assert len(content_chunks) >= 1
            assert len(complete_chunks) == 1
            assert complete_chunks[0]["session_id"] == "stream_123"

    def test_validate_graph_structure(self, graph_factory):
        """Test graph structure validation."""
        # Mock a compiled graph
        mock_graph = Mock()
        mock_graph_dict = {
            "nodes": [{"id": "test_agent"}, {"id": "__start__"}, {"id": "__end__"}],
            "edges": [
                {"source": "__start__", "target": "test_agent"},
                {"source": "test_agent", "target": "__end__"},
            ],
        }

        mock_graph.get_graph.return_value.to_json.return_value = mock_graph_dict
        mock_graph.checkpointer = MemorySaver()

        result = graph_factory.validate_graph_structure(mock_graph, "test_agent")

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["info"]["node_count"] == 3
        assert result["info"]["edge_count"] == 2
        assert "test_agent" in result["info"]["nodes"]
        assert result["info"]["has_checkpointer"] is True

    def test_validate_graph_structure_missing_agent(self, graph_factory):
        """Test graph validation with missing agent node."""
        mock_graph = Mock()
        mock_graph_dict = {
            "nodes": [{"id": "wrong_agent"}, {"id": "__start__"}, {"id": "__end__"}],
            "edges": [
                {"source": "__start__", "target": "wrong_agent"},
                {"source": "wrong_agent", "target": "__end__"},
            ],
        }

        mock_graph.get_graph.return_value.to_json.return_value = mock_graph_dict

        result = graph_factory.validate_graph_structure(mock_graph, "test_agent")

        assert result["valid"] is False
        assert any(
            "Expected agent node 'test_agent' not found" in error
            for error in result["errors"]
        )

    @pytest.mark.asyncio
    async def test_get_available_agents(self, graph_factory, mock_agent_factory):
        """Test getting available agents list."""
        agents = await graph_factory.get_available_agents()

        assert len(agents) == 2
        assert agents[0]["name"] == "test_agent"
        assert agents[0]["display_name"] == "Test Agent"
        assert agents[0]["description"] == "A test agent"
        assert agents[0]["available"] is True

        mock_agent_factory.list_agents.assert_called_once()
        mock_agent_factory.get_agent_config.assert_called()
        mock_agent_factory.validate_agent.assert_called()

    def test_clear_graph_cache(self, graph_factory):
        """Test clearing the graph cache."""
        # Add something to cache
        graph_factory._compiled_graphs["test"] = Mock()
        assert len(graph_factory._compiled_graphs) == 1

        # Clear cache
        graph_factory.clear_graph_cache()
        assert len(graph_factory._compiled_graphs) == 0


class TestGlobalInstances:
    """Test global instance management."""

    @pytest.mark.asyncio
    async def test_get_session_manager(self):
        """Test getting global session manager."""
        # Clean up any existing instances
        await cleanup_global_instances()

        manager1 = await get_session_manager()
        manager2 = await get_session_manager()

        # Should be the same instance
        assert manager1 is manager2

        # Clean up
        await cleanup_global_instances()

    @pytest.mark.asyncio
    async def test_get_graph_factory(self):
        """Test getting global graph factory."""
        # Clean up any existing instances
        await cleanup_global_instances()

        factory1 = await get_graph_factory()
        factory2 = await get_graph_factory()

        # Should be the same instance
        assert factory1 is factory2

        # Clean up
        await cleanup_global_instances()

    @pytest.mark.asyncio
    async def test_cleanup_global_instances(self):
        """Test cleanup of global instances."""
        # Create instances
        await get_session_manager()
        await get_graph_factory()

        # Clean up
        await cleanup_global_instances()

        # New instances should be different
        manager1 = await get_session_manager()
        factory1 = await get_graph_factory()

        await cleanup_global_instances()

        manager2 = await get_session_manager()
        factory2 = await get_graph_factory()

        assert manager1 is not manager2
        assert factory1 is not factory2

        # Final cleanup
        await cleanup_global_instances()


if __name__ == "__main__":
    pytest.main([__file__])
