"""Integration tests for state management with agent system."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from agent_hive.core.state import StateManager
from langchain_core.messages import HumanMessage


class TestStateIntegration:
    """Test integration between state management and agent system."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return Mock()

    @pytest.fixture
    def state_manager(self, mock_redis):
        """Create StateManager with mock Redis."""
        with patch("agent_hive.core.state.redis.from_url", return_value=mock_redis):
            return StateManager(redis_url="redis://localhost:6379")

    def test_agent_state_compatibility(self, state_manager):
        """Test that AgentState works correctly with session management."""
        # Create a session with agent state
        session = state_manager.create_session(
            user_id="test_user", mode="direct", active_agent="full_stack_dev"
        )

        # Verify initial state structure
        assert session.state["mode"] == "direct"
        assert session.state["active_agents"] == ["full_stack_dev"]
        assert session.state["task"] == ""
        assert session.state["messages"] == []
        assert session.state["scratchpad"] == {}
        assert session.state["multimodal_content"] == {}

        # Simulate agent interaction
        session.state["task"] = "Create a React component"
        session.state["messages"].append(HumanMessage(content="Create a login form"))
        session.state["scratchpad"][
            "analysis"
        ] = "Need to create React component with form validation"

        # Update session
        success = state_manager.update_session(session)
        assert success is True

        # Verify state persistence structure
        assert session.state["task"] == "Create a React component"
        assert len(session.state["messages"]) == 1
        assert (
            session.state["scratchpad"]["analysis"]
            == "Need to create React component with form validation"
        )

    def test_multimodal_content_integration(self, state_manager):
        """Test multimodal content handling in sessions."""
        from agent_hive.core.state import ProcessedFile

        # Create session
        session = state_manager.create_session(
            user_id="designer_user", mode="direct", active_agent="product_designer"
        )

        # Add multimodal content
        processed_file = ProcessedFile(
            file_id="design-123",
            original_name="mockup.png",
            file_type="image",
            processed_content={
                "description": "Login form mockup with modern UI",
                "elements": ["email input", "password input", "submit button"],
                "colors": ["#007bff", "#ffffff", "#f8f9fa"],
            },
            metadata={"width": 1920, "height": 1080, "size": 245760},
            processing_timestamp=datetime.utcnow(),
        )

        session.multimodal_files.append(processed_file)
        session.state["multimodal_content"]["current_design"] = {
            "file_id": "design-123",
            "analysis": "Modern login form with clean design",
        }

        # Update and verify
        success = state_manager.update_session(session)
        assert success is True

        assert len(session.multimodal_files) == 1
        assert session.multimodal_files[0].file_id == "design-123"
        assert (
            session.state["multimodal_content"]["current_design"]["file_id"]
            == "design-123"
        )

    def test_orchestration_mode_state(self, state_manager):
        """Test state management for orchestration mode."""
        # Create orchestration session
        session = state_manager.create_session(
            user_id="project_manager", mode="orchestration"
        )

        # Simulate orchestration workflow
        session.state["task"] = "Build a complete e-commerce website"
        session.state["active_agents"] = [
            "full_stack_dev",
            "product_designer",
            "qa_engineer",
        ]
        session.state["next"] = "task_analyzer"

        # Add orchestration-specific data
        session.state["scratchpad"]["task_breakdown"] = {
            "design_phase": "product_designer",
            "development_phase": "full_stack_dev",
            "testing_phase": "qa_engineer",
        }

        session.state["task_status"] = {
            "design_phase": "in_progress",
            "development_phase": "pending",
            "testing_phase": "pending",
        }

        # Update session
        success = state_manager.update_session(session)
        assert success is True

        # Verify orchestration state
        assert session.state["mode"] == "orchestration"
        assert len(session.state["active_agents"]) == 3
        assert session.state["next"] == "task_analyzer"
        assert "task_breakdown" in session.state["scratchpad"]
        assert session.state["task_status"]["design_phase"] == "in_progress"

    def test_session_mode_switching(self, state_manager):
        """Test switching between direct and orchestration modes."""
        # Start with direct mode
        session = state_manager.create_session(
            user_id="flexible_user", mode="direct", active_agent="full_stack_dev"
        )

        # Add some direct mode data
        session.state["task"] = "Fix a bug in React component"
        session.state["messages"].append(
            HumanMessage(content="The login form is not validating")
        )

        # Switch to orchestration mode
        session.mode = "orchestration"
        session.state["mode"] = "orchestration"
        session.active_agent = None
        session.state["active_agents"] = ["full_stack_dev", "qa_engineer"]
        session.state["task"] = "Comprehensive bug fix and testing"

        # Update session
        success = state_manager.update_session(session)
        assert success is True

        # Verify mode switch
        assert session.mode == "orchestration"
        assert session.state["mode"] == "orchestration"
        assert session.active_agent is None
        assert len(session.state["active_agents"]) == 2
        assert "Comprehensive" in session.state["task"]

    def test_error_handling_in_agent_context(self, state_manager):
        """Test error handling and recovery in agent context."""
        session = state_manager.create_session(
            user_id="error_user", mode="direct", active_agent="full_stack_dev"
        )

        # Simulate agent error
        session.state["errors"].append(
            {
                "agent": "full_stack_dev",
                "error_type": "model_timeout",
                "message": "OpenAI API timeout after 30 seconds",
                "timestamp": datetime.utcnow().isoformat(),
                "recovery_action": "switched_to_anthropic",
            }
        )

        # Add recovery information
        session.state["scratchpad"]["error_recovery"] = {
            "original_model": "gpt-4",
            "fallback_model": "claude-3-sonnet",
            "retry_count": 1,
        }

        # Update session
        success = state_manager.update_session(session)
        assert success is True

        # Verify error tracking
        assert len(session.state["errors"]) == 1
        assert session.state["errors"][0]["error_type"] == "model_timeout"
        assert session.state["scratchpad"]["error_recovery"]["retry_count"] == 1

    def test_concurrent_user_sessions(self, state_manager, mock_redis):
        """Test handling multiple concurrent user sessions."""
        # Create sessions for different users
        user1_session = state_manager.create_session(
            user_id="user1", mode="direct", active_agent="full_stack_dev"
        )

        user2_session = state_manager.create_session(
            user_id="user2", mode="orchestration"
        )

        user1_session2 = state_manager.create_session(
            user_id="user1", mode="direct", active_agent="product_designer"
        )

        # Verify session isolation
        assert user1_session.user_id == "user1"
        assert user2_session.user_id == "user2"
        assert user1_session2.user_id == "user1"

        assert user1_session.session_id != user2_session.session_id
        assert user1_session.session_id != user1_session2.session_id

        # Verify different modes and agents
        assert user1_session.mode == "direct"
        assert user2_session.mode == "orchestration"
        assert user1_session.active_agent == "full_stack_dev"
        assert user1_session2.active_agent == "product_designer"


if __name__ == "__main__":
    pytest.main([__file__])
