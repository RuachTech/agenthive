"""Integration tests for AgentHive API gateway and routing system."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import UploadFile

from agent_hive.api import create_app, AgentHiveAPI
from agent_hive.core.config import SystemConfig
from agent_hive.core.state import StateManager, Session
from agent_hive.core.multimodal import (
    MultimodalProcessor,
    ProcessedFile,
    FileType,
    ProcessingStatus,
)
from agent_hive.agents.factory import AgentFactory, AgentConfig


class TestAgentHiveAPI:
    """Test suite for AgentHive API class."""

    @pytest.fixture
    def mock_config(self):
        """Mock system configuration."""
        config = Mock(spec=SystemConfig)
        config.redis_url = "redis://localhost:6379"
        config.cors_origins = ["*"]
        config.api_host = "localhost"
        config.api_port = 8000
        config.log_level = "INFO"
        config.enable_metrics = True
        config.session_timeout = 3600
        config.environment = "test"
        return config

    @pytest.fixture
    def mock_state_manager(self):
        """Mock state manager."""
        manager = Mock(spec=StateManager)
        manager.health_check = Mock(return_value={"status": "healthy"})
        manager.create_session = Mock()
        manager.get_session = Mock()
        manager.update_session_state = Mock(return_value=True)
        manager.delete_session = Mock(return_value=True)
        manager.get_user_sessions = Mock(return_value=[])
        manager.cleanup_expired_sessions = Mock(return_value=0)
        return manager

    @pytest.fixture
    def mock_multimodal_processor(self):
        """Mock multimodal processor."""
        processor = Mock(spec=MultimodalProcessor)
        processor.get_supported_formats = Mock(
            return_value={
                "image": ["image/jpeg", "image/png"],
                "pdf": ["application/pdf"],
                "document": ["text/plain"],
            }
        )
        return processor

    @pytest.fixture
    def mock_graph_factory(self):
        """Mock graph factory."""
        factory = Mock()
        factory.execute_direct_chat = AsyncMock()
        factory.stream_direct_chat = AsyncMock()
        factory.get_available_agents = AsyncMock(
            return_value=[
                {
                    "name": "full_stack_engineer",
                    "display_name": "Full Stack Engineer",
                    "description": "Expert in full-stack development",
                    "capabilities": ["code_generation", "debugging"],
                    "available": True,
                    "validation_errors": [],
                }
            ]
        )
        return factory

    @pytest.fixture
    def mock_agent_factory(self):
        """Mock agent factory."""
        factory = Mock(spec=AgentFactory)
        factory.list_agents = Mock(return_value=["full_stack_engineer", "qa_engineer"])

        mock_config = Mock(spec=AgentConfig)
        mock_config.core_tools = ["code_execution", "file_system"]
        mock_config.composio_tools = ["github", "linear"]
        mock_config.capabilities = ["code_generation", "debugging"]

        factory.get_agent_config = Mock(return_value=mock_config)
        return factory

    @pytest.fixture
    def mock_model_factory(self):
        """Mock model factory."""
        factory = Mock()
        factory.check_all_models = AsyncMock(
            return_value={"gpt-4": True, "claude-3": True, "gemini-pro": False}
        )
        return factory

    @pytest.fixture
    async def api_instance(
        self,
        mock_config,
        mock_state_manager,
        mock_multimodal_processor,
        mock_graph_factory,
        mock_agent_factory,
        mock_model_factory,
    ):
        """Create API instance with mocked dependencies."""
        api = AgentHiveAPI(mock_config)

        # Inject mocked dependencies
        api.state_manager = mock_state_manager
        api.multimodal_processor = mock_multimodal_processor
        api.graph_factory = mock_graph_factory
        api.agent_factory = mock_agent_factory
        api.model_factory = mock_model_factory
        api.session_manager = Mock()
        api.orchestrator_factory = Mock()
        api._startup_time = datetime.utcnow()

        return api

    @pytest.mark.asyncio
    async def test_direct_chat_success(self, api_instance):
        """Test successful direct chat interaction."""
        # Setup mocks
        session_id = str(uuid.uuid4())
        api_instance.graph_factory.execute_direct_chat.return_value = {
            "response": "Hello! I'm ready to help with your development tasks.",
            "session_id": session_id,
            "agent_name": "full_stack_engineer",
            "status": "success",
            "state": {
                "message_count": 2,
                "active_agents": ["full_stack_engineer"],
                "last_updated": datetime.utcnow().isoformat(),
                "errors": [],
            },
        }

        # Test direct chat
        result = await api_instance.direct_chat(
            agent_name="full_stack_engineer",
            message="Hello, I need help with a Python project",
            session_id=session_id,
            user_id="test_user",
        )

        # Verify result
        assert result["status"] == "success"
        assert result["agent_name"] == "full_stack_engineer"
        assert result["session_id"] == session_id
        assert "response" in result

        # Verify mock calls
        api_instance.graph_factory.execute_direct_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_chat_invalid_agent(self, api_instance):
        """Test direct chat with invalid agent name."""
        with pytest.raises(Exception) as exc_info:
            await api_instance.direct_chat(
                agent_name="nonexistent_agent", message="Hello", user_id="test_user"
            )

        # Should raise HTTPException or similar error
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_direct_chat_with_files(self, api_instance):
        """Test direct chat with file uploads."""
        # Mock file processing
        mock_processed_file = Mock(spec=ProcessedFile)
        mock_processed_file.file_id = "file_123"
        mock_processed_file.original_name = "test.png"
        mock_processed_file.file_type = FileType.IMAGE
        mock_processed_file.status = ProcessingStatus.COMPLETED

        api_instance.multimodal_processor.process_file = AsyncMock(
            return_value=mock_processed_file
        )

        # Mock file upload
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.png"
        mock_file.content_type = "image/png"
        mock_file.read = AsyncMock(return_value=b"fake_image_data")

        # Setup graph factory response
        api_instance.graph_factory.execute_direct_chat.return_value = {
            "response": "I can see the image you uploaded.",
            "session_id": "test_session",
            "agent_name": "full_stack_engineer",
            "status": "success",
        }

        # Test direct chat with files
        result = await api_instance.direct_chat(
            agent_name="full_stack_engineer",
            message="Analyze this image",
            files=[mock_file],
            user_id="test_user",
        )

        # Verify file processing was called
        api_instance.multimodal_processor.process_file.assert_called_once()

        # Verify result includes file information
        assert "processed_files" in result
        assert len(result["processed_files"]) == 1
        assert result["processed_files"][0]["file_id"] == "file_123"

    @pytest.mark.asyncio
    async def test_orchestrate_task_success(self, api_instance):
        """Test successful task orchestration."""
        # Mock orchestration execution
        api_instance._execute_orchestration = AsyncMock(
            return_value={
                "response": "Task completed successfully by multiple agents.",
                "session_id": "test_session",
                "mode": "orchestration",
                "status": "success",
                "participating_agents": ["full_stack_engineer", "qa_engineer"],
                "task_status": {"status": "completed"},
                "processed_files": [],
            }
        )

        # Test orchestration
        result = await api_instance.orchestrate_task(
            task="Build a web application with tests", user_id="test_user"
        )

        # Verify result
        assert result["status"] == "success"
        assert result["mode"] == "orchestration"
        assert len(result["participating_agents"]) == 2
        assert "full_stack_engineer" in result["participating_agents"]
        assert "qa_engineer" in result["participating_agents"]

    @pytest.mark.asyncio
    async def test_upload_multimodal_content_success(self, api_instance):
        """Test successful multimodal content upload."""
        # Mock processed files
        mock_processed_file = Mock(spec=ProcessedFile)
        mock_processed_file.file_id = "file_456"
        mock_processed_file.original_name = "document.pdf"
        mock_processed_file.file_type = FileType.PDF
        mock_processed_file.status = ProcessingStatus.COMPLETED
        mock_processed_file.file_size = 1024
        mock_processed_file.processing_timestamp = datetime.utcnow()
        mock_processed_file.metadata = {"pages": 5}

        api_instance.multimodal_processor.process_file = AsyncMock(
            return_value=mock_processed_file
        )

        # Mock file upload
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=b"fake_pdf_data")

        # Test file upload
        result = await api_instance.upload_multimodal_content(
            files=[mock_file], user_id="test_user"
        )

        # Verify result
        assert result["total_files"] == 1
        assert result["successful"] == 1
        assert result["failed"] == 0
        assert len(result["processed_files"]) == 1

        processed_file = result["processed_files"][0]
        assert processed_file["file_id"] == "file_456"
        assert processed_file["original_name"] == "document.pdf"
        assert processed_file["file_type"] == "pdf"

    @pytest.mark.asyncio
    async def test_get_agent_status(self, api_instance):
        """Test getting agent status information."""
        # Test without session ID
        result = await api_instance.get_agent_status()

        # Verify result structure
        assert "agents" in result
        assert "models" in result
        assert "system" in result

        assert result["agents"]["total_count"] == 1
        assert result["agents"]["healthy_count"] == 1

        assert result["models"]["total_count"] == 3
        assert result["models"]["healthy_count"] == 2

    @pytest.mark.asyncio
    async def test_get_agent_status_with_session(self, api_instance):
        """Test getting agent status with specific session."""
        # Mock session
        mock_session = Mock(spec=Session)
        mock_session.session_id = "test_session"
        mock_session.user_id = "test_user"
        mock_session.mode = "direct"
        mock_session.active_agent = "full_stack_engineer"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.state = {"messages": [], "active_agents": ["full_stack_engineer"]}
        mock_session.multimodal_files = []

        api_instance.session_manager.get_session = AsyncMock(return_value=mock_session)

        # Test with session ID
        result = await api_instance.get_agent_status(session_id="test_session")

        # Verify session information is included
        assert "session" in result
        assert result["session"]["session_id"] == "test_session"
        assert result["session"]["mode"] == "direct"
        assert result["session"]["active_agent"] == "full_stack_engineer"

    @pytest.mark.asyncio
    async def test_get_available_tools_specific_agent(self, api_instance):
        """Test getting tools for a specific agent."""
        result = await api_instance.get_available_tools(
            agent_name="full_stack_engineer"
        )

        # Verify result
        assert result["agent"] == "full_stack_engineer"
        assert "core_tools" in result
        assert "composio_tools" in result
        assert "capabilities" in result

        assert "code_execution" in result["core_tools"]
        assert "github" in result["composio_tools"]
        assert "code_generation" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_get_available_tools_all_agents(self, api_instance):
        """Test getting tools for all agents."""
        result = await api_instance.get_available_tools()

        # Verify result structure
        assert "agents" in result
        assert "supported_formats" in result

        assert "full_stack_engineer" in result["agents"]
        assert "qa_engineer" in result["agents"]

    @pytest.mark.asyncio
    async def test_create_session_success(self, api_instance):
        """Test successful session creation."""
        # Mock session creation
        mock_session = Mock(spec=Session)
        mock_session.session_id = "new_session_123"
        mock_session.user_id = "test_user"
        mock_session.mode = "direct"
        mock_session.active_agent = "full_stack_engineer"
        mock_session.created_at = datetime.utcnow()

        api_instance.state_manager.create_session.return_value = mock_session

        # Test session creation
        result = await api_instance.create_session(
            user_id="test_user", mode="direct", active_agent="full_stack_engineer"
        )

        # Verify result
        assert result["session_id"] == "new_session_123"
        assert result["user_id"] == "test_user"
        assert result["mode"] == "direct"
        assert result["active_agent"] == "full_stack_engineer"
        assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_create_session_invalid_mode(self, api_instance):
        """Test session creation with invalid mode."""
        with pytest.raises(Exception) as exc_info:
            await api_instance.create_session(user_id="test_user", mode="invalid_mode")

        assert "mode must be" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_delete_session_success(self, api_instance):
        """Test successful session deletion."""
        result = await api_instance.delete_session("test_session")

        # Verify result
        assert result["session_id"] == "test_session"
        assert result["status"] == "deleted"

        # Verify mock was called
        api_instance.state_manager.delete_session.assert_called_once_with(
            "test_session"
        )

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, api_instance):
        """Test deleting non-existent session."""
        api_instance.state_manager.delete_session.return_value = False

        with pytest.raises(Exception) as exc_info:
            await api_instance.delete_session("nonexistent_session")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_user_sessions(self, api_instance):
        """Test listing user sessions."""
        # Mock session data
        api_instance.state_manager.get_user_sessions.return_value = [
            "session1",
            "session2",
        ]

        mock_session = Mock(spec=Session)
        mock_session.session_id = "session1"
        mock_session.mode = "direct"
        mock_session.active_agent = "full_stack_engineer"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.state = {"messages": []}
        mock_session.multimodal_files = []

        api_instance.state_manager.get_session.return_value = mock_session

        # Test listing sessions
        result = await api_instance.list_user_sessions("test_user")

        # Verify result
        assert result["user_id"] == "test_user"
        assert result["total_count"] == 2
        assert len(result["sessions"]) == 2


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application."""
        config = SystemConfig()
        config.redis_url = "redis://localhost:6379"
        return create_app(config)

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    @pytest.fixture
    def mock_api_instance(self):
        """Mock API instance for dependency injection."""
        api = Mock(spec=AgentHiveAPI)

        # Mock all async methods
        api.direct_chat = AsyncMock()
        api.orchestrate_task = AsyncMock()
        api.upload_multimodal_content = AsyncMock()
        api.get_agent_status = AsyncMock()
        api.get_available_tools = AsyncMock()
        api.create_session = AsyncMock()
        api.delete_session = AsyncMock()
        api.list_user_sessions = AsyncMock()

        # Mock attributes
        api.state_manager = Mock()
        api.state_manager.health_check.return_value = {"status": "healthy"}
        api.model_factory = Mock()
        api.model_factory.check_all_models = AsyncMock(return_value={"gpt-4": True})
        api.graph_factory = Mock()
        api.graph_factory.get_available_agents = AsyncMock(return_value=[])
        api._startup_time = datetime.utcnow()
        api.config = Mock()
        api.config.api_host = "localhost"
        api.config.api_port = 8000
        api.config.log_level = "INFO"
        api.config.enable_metrics = True
        api.config.session_timeout = 3600
        api.config.environment = "test"
        api.agent_factory = Mock()
        api.agent_factory.list_agents.return_value = []

        return api

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "AgentHive API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data

    @patch("agent_hive.api.main.get_api")
    def test_health_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test health check endpoint."""
        mock_get_api.return_value = mock_api_instance

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
        assert "timestamp" in data

    @patch("agent_hive.api.main.get_api")
    def test_status_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test system status endpoint."""
        mock_get_api.return_value = mock_api_instance

        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        assert "system" in data
        assert "configuration" in data
        assert "agents" in data

    @patch("agent_hive.api.main.get_api")
    def test_direct_chat_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test direct chat endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.direct_chat.return_value = {
            "response": "Hello!",
            "session_id": "test_session",
            "status": "success",
        }

        response = client.post(
            "/api/v1/chat/direct",
            data={
                "agent_name": "full_stack_engineer",
                "message": "Hello",
                "user_id": "test_user",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "response" in data

    @patch("agent_hive.api.main.get_api")
    def test_orchestrate_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test orchestration endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.orchestrate_task.return_value = {
            "response": "Task completed",
            "session_id": "test_session",
            "status": "success",
            "participating_agents": ["full_stack_engineer"],
        }

        response = client.post(
            "/api/v1/chat/orchestrate",
            data={"task": "Build an application", "user_id": "test_user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "participating_agents" in data

    @patch("agent_hive.api.main.get_api")
    def test_upload_files_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test file upload endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.upload_multimodal_content.return_value = {
            "processed_files": [],
            "errors": [],
            "total_files": 1,
            "successful": 1,
            "failed": 0,
        }

        # Create a test file
        test_file_content = b"test file content"

        response = client.post(
            "/api/v1/files/upload",
            files={"files": ("test.txt", test_file_content, "text/plain")},
            data={"user_id": "test_user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1
        assert data["successful"] == 1

    @patch("agent_hive.api.main.get_api")
    def test_get_agents_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test get agents endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.get_agent_status.return_value = {
            "agents": {"available": [], "total_count": 0, "healthy_count": 0},
            "models": {"status": {}, "total_count": 0, "healthy_count": 0},
            "system": {
                "uptime_seconds": 100,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        response = client.get("/api/v1/agents")
        assert response.status_code == 200

        data = response.json()
        assert "agents" in data
        assert "models" in data
        assert "system" in data

    @patch("agent_hive.api.main.get_api")
    def test_create_session_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test create session endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.create_session.return_value = {
            "session_id": "new_session",
            "user_id": "test_user",
            "mode": "direct",
            "status": "created",
        }

        response = client.post(
            "/api/v1/sessions", data={"user_id": "test_user", "mode": "direct"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["user_id"] == "test_user"

    @patch("agent_hive.api.main.get_api")
    def test_delete_session_endpoint(self, mock_get_api, client, mock_api_instance):
        """Test delete session endpoint."""
        mock_get_api.return_value = mock_api_instance
        mock_api_instance.delete_session.return_value = {
            "session_id": "test_session",
            "status": "deleted",
        }

        response = client.delete("/api/v1/sessions/test_session")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "deleted"
        assert data["session_id"] == "test_session"


class TestErrorHandling:
    """Test suite for API error handling."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance for error testing."""
        config = Mock(spec=SystemConfig)
        api = AgentHiveAPI(config)
        api.agent_factory = Mock()
        api.agent_factory.list_agents.return_value = ["full_stack_engineer"]
        return api

    @pytest.mark.asyncio
    async def test_direct_chat_agent_not_found(self, api_instance):
        """Test direct chat with non-existent agent."""
        with pytest.raises(Exception) as exc_info:
            await api_instance.direct_chat(
                agent_name="nonexistent_agent", message="Hello", user_id="test_user"
            )

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_file_upload_validation_error(self, api_instance):
        """Test file upload with validation errors."""
        # Mock multimodal processor to raise validation error
        api_instance.multimodal_processor = Mock()
        api_instance.multimodal_processor.process_file = AsyncMock(
            side_effect=Exception("File validation failed")
        )

        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "invalid.exe"
        mock_file.content_type = "application/x-executable"
        mock_file.read = AsyncMock(return_value=b"invalid_content")

        # Should handle the error gracefully
        result = await api_instance.upload_multimodal_content(
            files=[mock_file], user_id="test_user"
        )

        # Should return error information
        assert result["failed"] > 0 or result["successful"] == 0

    @pytest.mark.asyncio
    async def test_session_creation_invalid_mode(self, api_instance):
        """Test session creation with invalid mode."""
        api_instance.state_manager = Mock()

        with pytest.raises(Exception) as exc_info:
            await api_instance.create_session(user_id="test_user", mode="invalid_mode")

        assert "mode must be" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
