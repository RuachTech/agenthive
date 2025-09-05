"""Tests for the project foundation setup."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from agent_hive.api import app
from agent_hive.core.state import AgentState
from agent_hive.core.config import AgentConfig, SystemConfig


class TestProjectFoundation:
    """Test the basic project foundation setup."""

    def test_fastapi_app_creation(self):
        """Test that the FastAPI app can be created successfully."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AgentHive API"
        assert data["version"] == "0.1.0"

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data

    def test_system_status_endpoint(self):
        """Test the system status endpoint."""
        client = TestClient(app)
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "configuration" in data
        assert "agents" in data

    def test_agent_state_structure(self):
        """Test that AgentState has all required fields."""
        # This tests the TypedDict structure by creating a valid state
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "next": "test_agent",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": "test_session_123",
            "user_id": "test_user",
            "last_updated": datetime.utcnow(),
            "errors": [],
            "task_status": {},
        }

        # Verify all required fields are present
        assert state["task"] == "Test task"
        assert state["mode"] in ["direct", "orchestration"]
        assert isinstance(state["messages"], list)
        assert isinstance(state["scratchpad"], dict)
        assert isinstance(state["active_agents"], list)
        assert isinstance(state["multimodal_content"], dict)
        assert isinstance(state["session_id"], str)

    def test_agent_config_creation(self):
        """Test AgentConfig dataclass creation and validation."""
        config = AgentConfig(
            name="test_agent",
            display_name="Test Agent",
            description="A test agent for validation",
            system_prompt="You are a test agent.",
            model_provider="openai",
            model_name="gpt-4",
            capabilities=["vision", "code_execution"],
            specialized_for=["testing", "validation"],
        )

        assert config.name == "test_agent"
        assert config.model_provider == "openai"
        assert config.temperature == 0.7  # default value
        assert config.max_tokens == 4000  # default value
        assert "vision" in config.capabilities
        assert config.enabled is True  # default value

    def test_agent_config_validation(self):
        """Test AgentConfig validation logic."""
        # Test invalid model provider
        with pytest.raises(ValueError, match="Unsupported model provider"):
            AgentConfig(
                name="test",
                display_name="Test",
                description="Test",
                system_prompt="Test",
                model_provider="invalid_provider",
                model_name="test-model",
            )

        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            AgentConfig(
                name="test",
                display_name="Test",
                description="Test",
                system_prompt="Test",
                model_provider="openai",
                model_name="gpt-4",
                temperature=3.0,
            )

        # Test empty name
        with pytest.raises(ValueError, match="Agent name is required"):
            AgentConfig(
                name="",
                display_name="Test",
                description="Test",
                system_prompt="Test",
                model_provider="openai",
                model_name="gpt-4",
            )

    def test_system_config_creation(self):
        """Test SystemConfig dataclass creation and validation."""
        config = SystemConfig(
            api_host="127.0.0.1",
            api_port=8080,
            redis_url="redis://localhost:6379",
            log_level="DEBUG",
        )

        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8080
        assert config.log_level == "DEBUG"
        assert config.session_timeout == 3600  # default value
        assert config.enable_metrics is True  # default value

    def test_system_config_validation(self):
        """Test SystemConfig validation logic."""
        # Test invalid port
        with pytest.raises(ValueError, match="API port must be between 1 and 65535"):
            SystemConfig(api_port=70000)

        # Test invalid session timeout
        with pytest.raises(ValueError, match="Session timeout must be positive"):
            SystemConfig(session_timeout=-1)
