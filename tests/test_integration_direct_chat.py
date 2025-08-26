"""Integration tests for direct chat graph functionality."""

import pytest
import asyncio

from agent_hive.core.graphs import DirectChatGraphFactory, SessionManager
from agent_hive.core.config import AgentConfig, SystemConfig
from agent_hive.agents.factory import AgentFactory, AgentCapabilities
from agent_hive.core.models import ModelFactory


class TestDirectChatIntegration:
    """Integration tests for the complete direct chat system."""
    
    async def setup_system(self):
        """Set up a complete system for integration testing."""
        # Create model factory
        model_factory = ModelFactory()
        
        # Create agent factory
        agent_factory = AgentFactory(model_factory)
        
        # Create a test agent configuration
        test_config = AgentConfig(
            name="integration_test_agent",
            display_name="Integration Test Agent",
            description="Agent for integration testing",
            system_prompt="You are a test agent for integration testing.",
            model_provider="openai",
            model_name="gpt-4",
            capabilities=["chat", "test"],
            specialized_for=["testing"]
        )
        
        test_capabilities = AgentCapabilities(
            required_capabilities=["chat"],
            optional_capabilities=["test"],
            model_requirements=[],
            required_tools=[],
            mcp_requirements=[]
        )
        
        # Register the test agent
        agent_factory.register_agent_config(test_config, test_capabilities)
        
        # Create session manager
        session_manager = SessionManager(SystemConfig(session_timeout=3600))
        await session_manager.start()
        
        # Create graph factory
        graph_factory = DirectChatGraphFactory(session_manager)
        graph_factory.agent_factory = agent_factory
        
        return {
            "agent_factory": agent_factory,
            "session_manager": session_manager,
            "graph_factory": graph_factory,
            "model_factory": model_factory
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_without_models(self):
        """Test the complete workflow without requiring actual model calls."""
        system = await self.setup_system()
        graph_factory = system["graph_factory"]
        session_manager = system["session_manager"]
        agent_factory = system["agent_factory"]
        
        # Test 1: Agent validation
        validation_result = await agent_factory.validate_agent("integration_test_agent")
        assert "valid" in validation_result
        assert "errors" in validation_result
        
        # Test 2: Session creation
        session = await session_manager.create_session(
            session_id="integration_test_001",
            user_id="test_user",
            mode="direct",
            active_agent="integration_test_agent",
            initial_task="Test message"
        )
        
        assert session.session_id == "integration_test_001"
        assert session.mode == "direct"
        assert session.active_agent == "integration_test_agent"
        assert session.state["task"] == "Test message"
        
        # Test 3: Session retrieval and update
        retrieved_session = await session_manager.get_session("integration_test_001")
        assert retrieved_session is not None
        assert retrieved_session.session_id == session.session_id
        
        # Update session state
        retrieved_session.state["scratchpad"]["test_key"] = "test_value"
        success = await session_manager.update_session_state(
            "integration_test_001",
            retrieved_session.state
        )
        assert success is True
        
        # Test 4: Available agents
        available_agents = await graph_factory.get_available_agents()
        agent_names = [agent["name"] for agent in available_agents]
        assert "integration_test_agent" in agent_names
        
        # Test 5: Graph structure validation (mock)
        from unittest.mock import Mock
        from langgraph.checkpoint.memory import MemorySaver
        
        mock_graph = Mock()
        mock_graph.get_graph.return_value.to_json.return_value = {
            "nodes": [
                {"id": "integration_test_agent"},
                {"id": "__start__"},
                {"id": "__end__"}
            ],
            "edges": [
                {"source": "__start__", "target": "integration_test_agent"},
                {"source": "integration_test_agent", "target": "__end__"}
            ]
        }
        mock_graph.checkpointer = MemorySaver()
        
        validation_result = graph_factory.validate_graph_structure(
            mock_graph, 
            "integration_test_agent"
        )
        assert validation_result["valid"] is True
        assert validation_result["info"]["node_count"] == 3
        assert validation_result["info"]["has_checkpointer"] is True
        
        # Test 6: Session cleanup
        success = await session_manager.delete_session("integration_test_001")
        assert success is True
        
        deleted_session = await session_manager.get_session("integration_test_001")
        assert deleted_session is None
        
        # Cleanup system
        await system["session_manager"].stop()
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_and_agents(self):
        """Test handling multiple sessions and agents."""
        system = await self.setup_system()
        session_manager = system["session_manager"]
        agent_factory = system["agent_factory"]
        
        # Create a second agent
        second_config = AgentConfig(
            name="second_test_agent",
            display_name="Second Test Agent",
            description="Second agent for testing",
            system_prompt="You are the second test agent.",
            model_provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            capabilities=["chat", "analysis"],
            specialized_for=["analysis"]
        )
        
        second_capabilities = AgentCapabilities(
            required_capabilities=["chat"],
            optional_capabilities=["analysis"],
            model_requirements=[],
            required_tools=[],
            mcp_requirements=[]
        )
        
        agent_factory.register_agent_config(second_config, second_capabilities)
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            agent_name = "integration_test_agent" if i % 2 == 0 else "second_test_agent"
            session = await session_manager.create_session(
                session_id=f"multi_test_{i:03d}",
                user_id=f"user_{i}",
                mode="direct",
                active_agent=agent_name,
                initial_task=f"Task {i}"
            )
            sessions.append(session)
        
        # Verify all sessions exist
        assert len(sessions) == 3
        
        # Test session listing
        all_sessions = await session_manager.list_sessions()
        assert len(all_sessions) == 3
        
        user_0_sessions = await session_manager.list_sessions("user_0")
        assert len(user_0_sessions) == 1
        assert user_0_sessions[0].user_id == "user_0"
        
        # Test agent listing
        agents = agent_factory.list_agents()
        assert len(agents) == 2
        assert "integration_test_agent" in agents
        assert "second_test_agent" in agents
        
        # Cleanup all sessions
        for session in sessions:
            success = await session_manager.delete_session(session.session_id)
            assert success is True
        
        # Cleanup system
        await system["session_manager"].stop()
    
    @pytest.mark.asyncio
    async def test_error_scenarios(self):
        """Test various error scenarios."""
        system = await self.setup_system()
        graph_factory = system["graph_factory"]
        session_manager = system["session_manager"]
        
        # Test 1: Invalid agent name
        with pytest.raises(Exception):  # Should raise GraphValidationError or similar
            await graph_factory.create_direct_chat_graph("nonexistent_agent")
        
        # Test 2: Session operations on non-existent session
        missing_session = await session_manager.get_session("missing_session")
        assert missing_session is None
        
        update_success = await session_manager.update_session_state(
            "missing_session", 
            {"task": "test"}
        )
        assert update_success is False
        
        delete_success = await session_manager.delete_session("missing_session")
        assert delete_success is False
        
        # Test 3: Invalid graph structure validation
        from unittest.mock import Mock
        
        invalid_graph = Mock()
        invalid_graph.get_graph.return_value.to_json.return_value = {
            "nodes": [{"id": "wrong_node"}],
            "edges": []
        }
        
        validation_result = graph_factory.validate_graph_structure(
            invalid_graph, 
            "expected_agent"
        )
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0
        
        # Cleanup system
        await system["session_manager"].stop()
    
    @pytest.mark.asyncio
    async def test_session_expiry_handling(self):
        """Test session expiry and cleanup."""
        system = await self.setup_system()
        
        # Create session manager with very short timeout
        short_timeout_manager = SessionManager(SystemConfig(session_timeout=1))
        await short_timeout_manager.start()
        
        try:
            # Create a session
            await short_timeout_manager.create_session(
                session_id="expiry_test",
                user_id="test_user",
                mode="direct"
            )
            
            # Session should exist immediately
            retrieved = await short_timeout_manager.get_session("expiry_test")
            assert retrieved is not None
            
            # Wait for expiry
            await asyncio.sleep(2)
            
            # Session should be expired and cleaned up
            expired_session = await short_timeout_manager.get_session("expiry_test")
            assert expired_session is None
            
        finally:
            await short_timeout_manager.stop()
        
        # Cleanup system
        await system["session_manager"].stop()
    
    @pytest.mark.asyncio
    async def test_state_persistence_structure(self):
        """Test that state structure is maintained correctly."""
        system = await self.setup_system()
        session_manager = system["session_manager"]
        
        # Create session with initial state
        session = await session_manager.create_session(
            session_id="state_test",
            user_id="test_user",
            mode="direct",
            active_agent="integration_test_agent",
            initial_task="Test state persistence"
        )
        
        # Verify initial state structure
        state = session.state
        required_fields = [
            "task", "messages", "next", "scratchpad", "mode",
            "active_agents", "multimodal_content", "session_id",
            "user_id", "last_updated", "errors", "task_status"
        ]
        
        for field in required_fields:
            assert field in state, f"Required field '{field}' missing from state"
        
        # Verify field types and values
        assert isinstance(state["messages"], list)
        assert isinstance(state["scratchpad"], dict)
        assert isinstance(state["active_agents"], list)
        assert isinstance(state["multimodal_content"], dict)
        assert isinstance(state["errors"], list)
        assert isinstance(state["task_status"], dict)
        
        assert state["task"] == "Test state persistence"
        assert state["mode"] == "direct"
        assert state["session_id"] == "state_test"
        assert state["user_id"] == "test_user"
        assert "integration_test_agent" in state["active_agents"]
        
        # Test state updates
        from langchain_core.messages import HumanMessage, AIMessage
        
        updated_state = state.copy()
        updated_state["messages"].append(HumanMessage(content="Test message"))
        updated_state["messages"].append(AIMessage(content="Test response"))
        updated_state["scratchpad"]["test_data"] = {"key": "value"}
        updated_state["task_status"]["status"] = "in_progress"
        
        success = await session_manager.update_session_state("state_test", updated_state)
        assert success is True
        
        # Verify updates persisted
        updated_session = await session_manager.get_session("state_test")
        assert len(updated_session.state["messages"]) == 2
        assert updated_session.state["scratchpad"]["test_data"]["key"] == "value"
        assert updated_session.state["task_status"]["status"] == "in_progress"
        
        # Cleanup
        await session_manager.delete_session("state_test")
        await system["session_manager"].stop()


if __name__ == "__main__":
    pytest.main([__file__])