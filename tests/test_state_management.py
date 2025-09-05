"""Tests for session and state management functionality."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agent_hive.core.state import (
    AgentState,
    Session,
    ProcessedFile,
    StateManager,
)
from langchain_core.messages import HumanMessage


class TestProcessedFile:
    """Test ProcessedFile dataclass functionality."""

    def test_processed_file_creation(self):
        """Test creating a ProcessedFile instance."""
        now = datetime.utcnow()
        file = ProcessedFile(
            file_id="test-123",
            original_name="test.pdf",
            file_type="pdf",
            processed_content={"text": "Sample content"},
            metadata={"size": 1024},
            processing_timestamp=now,
        )

        assert file.file_id == "test-123"
        assert file.original_name == "test.pdf"
        assert file.file_type == "pdf"
        assert file.processed_content == {"text": "Sample content"}
        assert file.metadata == {"size": 1024}
        assert file.processing_timestamp == now

    def test_processed_file_to_dict(self):
        """Test converting ProcessedFile to dictionary."""
        now = datetime.utcnow()
        file = ProcessedFile(
            file_id="test-123",
            original_name="test.pdf",
            file_type="pdf",
            processed_content={"text": "Sample content"},
            metadata={"size": 1024},
            processing_timestamp=now,
        )

        result = file.to_dict()

        assert result["file_id"] == "test-123"
        assert result["original_name"] == "test.pdf"
        assert result["file_type"] == "pdf"
        assert result["processed_content"] == {"text": "Sample content"}
        assert result["metadata"] == {"size": 1024}
        assert result["processing_timestamp"] == now.isoformat()

    def test_processed_file_from_dict(self):
        """Test creating ProcessedFile from dictionary."""
        now = datetime.utcnow()
        data = {
            "file_id": "test-123",
            "original_name": "test.pdf",
            "file_type": "pdf",
            "processed_content": {"text": "Sample content"},
            "metadata": {"size": 1024},
            "processing_timestamp": now.isoformat(),
        }

        file = ProcessedFile.from_dict(data)

        assert file.file_id == "test-123"
        assert file.original_name == "test.pdf"
        assert file.file_type == "pdf"
        assert file.processed_content == {"text": "Sample content"}
        assert file.metadata == {"size": 1024}
        assert file.processing_timestamp == now


class TestSession:
    """Test Session dataclass functionality."""

    def create_test_state(self, session_id: str = "test-session") -> AgentState:
        """Create a test AgentState."""
        return {
            "task": "Test task",
            "messages": [HumanMessage(content="Hello")],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": session_id,
            "user_id": "test_user",
            "last_updated": datetime.utcnow(),
            "errors": [],
            "task_status": {},
        }

    def test_session_creation(self):
        """Test creating a Session instance."""
        now = datetime.utcnow()
        state = self.create_test_state()

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        assert session.session_id == "test-session"
        assert session.user_id == "test_user"
        assert session.mode == "direct"
        assert session.active_agent == "test_agent"
        assert session.created_at == now
        assert session.last_activity == now
        assert session.state == state
        assert session.multimodal_files == []

    def test_session_post_init(self):
        """Test that session_id is set in state during initialization."""
        state = self.create_test_state("different-id")

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            state=state,
            multimodal_files=[],
        )

        # Should update state with correct session_id
        assert session.state["session_id"] == "test-session"

    def test_update_activity(self):
        """Test updating session activity timestamp."""
        now = datetime.utcnow()
        state = self.create_test_state()

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        original_time = session.last_activity

        # Wait a bit and update
        import time

        time.sleep(0.01)
        session.update_activity()

        assert session.last_activity > original_time
        assert session.state["last_updated"] == session.last_activity

    def test_is_expired(self):
        """Test session expiration checking."""
        now = datetime.utcnow()
        state = self.create_test_state()

        # Create session with old last_activity
        old_time = now - timedelta(hours=25)
        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=old_time,
            last_activity=old_time,
            state=state,
            multimodal_files=[],
        )

        # Should be expired with default 24 hour timeout
        assert session.is_expired() is True

        # Should not be expired with longer timeout
        assert session.is_expired(timeout_hours=48) is False

    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        now = datetime.utcnow()
        state = self.create_test_state()

        processed_file = ProcessedFile(
            file_id="test-file",
            original_name="test.pdf",
            file_type="pdf",
            processed_content={"text": "content"},
            metadata={},
            processing_timestamp=now,
        )

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[processed_file],
        )

        result = session.to_dict()

        assert result["session_id"] == "test-session"
        assert result["user_id"] == "test_user"
        assert result["mode"] == "direct"
        assert result["active_agent"] == "test_agent"
        assert result["created_at"] == now.isoformat()
        assert result["last_activity"] == now.isoformat()
        assert isinstance(result["state"], dict)
        assert len(result["multimodal_files"]) == 1

    def test_session_from_dict(self):
        """Test creating session from dictionary."""
        now = datetime.utcnow()

        data = {
            "session_id": "test-session",
            "user_id": "test_user",
            "mode": "direct",
            "active_agent": "test_agent",
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "state": {
                "task": "Test task",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "direct",
                "active_agents": ["test_agent"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test_user",
                "last_updated": now.isoformat(),
                "errors": [],
                "task_status": {},
            },
            "multimodal_files": [],
        }

        session = Session.from_dict(data)

        assert session.session_id == "test-session"
        assert session.user_id == "test_user"
        assert session.mode == "direct"
        assert session.active_agent == "test_agent"
        assert session.created_at == now
        assert session.last_activity == now
        assert session.state["last_updated"] == now


class TestStateManager:
    """Test StateManager functionality with mock Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return Mock()

    @pytest.fixture
    def state_manager(self, mock_redis):
        """Create StateManager with mock Redis."""
        with patch("agent_hive.core.state.redis.from_url", return_value=mock_redis):
            return StateManager(redis_url="redis://localhost:6379")

    def test_state_manager_init(self, mock_redis):
        """Test StateManager initialization."""
        with patch(
            "agent_hive.core.state.redis.from_url", return_value=mock_redis
        ) as mock_from_url:
            manager = StateManager(
                redis_url="redis://test:6379", session_timeout_hours=12
            )

            mock_from_url.assert_called_once_with(
                "redis://test:6379", decode_responses=True
            )
            assert manager.session_timeout_hours == 12
            assert manager.session_prefix == "agent_hive:session:"
            assert manager.user_sessions_prefix == "agent_hive:user_sessions:"

    def test_create_session(self, state_manager, mock_redis):
        """Test creating a new session."""
        # Mock UUID generation
        mock_uuid = Mock()
        mock_uuid.__str__ = Mock(return_value="test-uuid")
        with patch("agent_hive.core.state.uuid.uuid4", return_value=mock_uuid):
            session = state_manager.create_session(
                user_id="test_user", mode="direct", active_agent="test_agent"
            )

        assert session.session_id == "test-uuid"
        assert session.user_id == "test_user"
        assert session.mode == "direct"
        assert session.active_agent == "test_agent"
        assert session.state["mode"] == "direct"
        assert session.state["active_agents"] == ["test_agent"]
        assert session.state["session_id"] == "test-uuid"

        # Verify Redis calls
        mock_redis.setex.assert_called_once()
        mock_redis.sadd.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_get_session_success(self, state_manager, mock_redis):
        """Test successfully retrieving a session."""
        now = datetime.utcnow()
        session_data = {
            "session_id": "test-session",
            "user_id": "test_user",
            "mode": "direct",
            "active_agent": "test_agent",
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "state": {
                "task": "Test task",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "direct",
                "active_agents": ["test_agent"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test_user",
                "last_updated": now.isoformat(),
                "errors": [],
                "task_status": {},
            },
            "multimodal_files": [],
        }

        mock_redis.get.return_value = json.dumps(session_data)

        session = state_manager.get_session("test-session")

        assert session is not None
        assert session.session_id == "test-session"
        assert session.user_id == "test_user"
        mock_redis.get.assert_called_once_with("agent_hive:session:test-session")

    def test_get_session_not_found(self, state_manager, mock_redis):
        """Test retrieving non-existent session."""
        mock_redis.get.return_value = None

        session = state_manager.get_session("nonexistent")

        assert session is None
        mock_redis.get.assert_called_once_with("agent_hive:session:nonexistent")

    def test_get_session_expired(self, state_manager, mock_redis):
        """Test retrieving expired session."""
        # Create session data with old timestamp
        old_time = datetime.utcnow() - timedelta(hours=25)
        session_data = {
            "session_id": "test-session",
            "user_id": "test_user",
            "mode": "direct",
            "active_agent": "test_agent",
            "created_at": old_time.isoformat(),
            "last_activity": old_time.isoformat(),
            "state": {
                "task": "Test task",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "direct",
                "active_agents": ["test_agent"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test_user",
                "last_updated": old_time.isoformat(),
                "errors": [],
                "task_status": {},
            },
            "multimodal_files": [],
        }

        mock_redis.get.return_value = json.dumps(session_data)

        session = state_manager.get_session("test-session")

        # Should return None and clean up expired session
        assert session is None
        mock_redis.delete.assert_called_once()

    def test_get_session_corrupted_data(self, state_manager, mock_redis):
        """Test handling corrupted session data."""
        mock_redis.get.return_value = "invalid json"

        session = state_manager.get_session("test-session")

        assert session is None
        # Should attempt cleanup - delete is called in _handle_corrupted_session
        mock_redis.delete.assert_called_once_with("agent_hive:session:test-session")

    def test_update_session(self, state_manager, mock_redis):
        """Test updating an existing session."""
        now = datetime.utcnow()
        state = {
            "task": "Test task",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test_user",
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        result = state_manager.update_session(session)

        assert result is True
        mock_redis.setex.assert_called_once()

    def test_delete_session(self, state_manager, mock_redis):
        """Test deleting a session."""
        session_data = {"session_id": "test-session", "user_id": "test_user"}

        mock_redis.get.return_value = json.dumps(session_data)
        mock_redis.delete.return_value = 1  # Successful deletion

        result = state_manager.delete_session("test-session")

        assert result is True
        mock_redis.get.assert_called_once()
        mock_redis.delete.assert_called_once()
        mock_redis.srem.assert_called_once()

    def test_get_user_sessions(self, state_manager, mock_redis):
        """Test getting all sessions for a user."""
        mock_redis.smembers.return_value = {"session1", "session2", "session3"}

        # Mock get_session to return valid sessions for session1 and session2
        def mock_get_session(session_id):
            if session_id in ["session1", "session2"]:
                return Mock(session_id=session_id)
            return None  # session3 is expired/invalid

        with patch.object(state_manager, "get_session", side_effect=mock_get_session):
            sessions = state_manager.get_user_sessions("test_user")

        # Order doesn't matter for this test, just check that the right sessions are returned
        assert set(sessions) == {"session1", "session2"}
        mock_redis.smembers.assert_called_once_with(
            "agent_hive:user_sessions:test_user"
        )

    def test_cleanup_expired_sessions(self, state_manager, mock_redis):
        """Test cleaning up expired sessions."""
        mock_redis.keys.return_value = [
            "agent_hive:session:session1",
            "agent_hive:session:session2",
            "agent_hive:session:session3",
        ]

        # Mock get_session to return None for expired sessions
        def mock_get_session(session_id):
            if session_id in ["session1", "session3"]:
                return None  # Expired
            return Mock(session_id=session_id)  # Valid

        with patch.object(state_manager, "get_session", side_effect=mock_get_session):
            cleaned_count = state_manager.cleanup_expired_sessions()

        assert cleaned_count == 2  # session1 and session3 were expired
        mock_redis.keys.assert_called_once_with("agent_hive:session:*")

    def test_validate_session_state_valid(self, state_manager):
        """Test validating a valid session state."""
        now = datetime.utcnow()
        state = {
            "task": "Test task",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test_user",
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        # Should not raise exception
        state_manager._validate_session_state(session)

    def test_validate_session_state_missing_fields(self, state_manager):
        """Test validating session state with missing fields."""
        now = datetime.utcnow()
        state = {
            "task": "Test task",
            # Missing required fields
        }

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        # Should recover missing fields
        state_manager._validate_session_state(session)

        # Check that required fields were added
        assert "messages" in session.state
        assert "scratchpad" in session.state
        assert "active_agents" in session.state
        assert session.state["session_id"] == "test-session"

    def test_validate_session_state_invalid_types(self, state_manager):
        """Test validating session state with invalid field types."""
        now = datetime.utcnow()
        state = {
            "task": "Test task",
            "messages": "invalid",  # Should be list
            "next": "",
            "scratchpad": "invalid",  # Should be dict
            "mode": "invalid",  # Should be 'direct' or 'orchestration'
            "active_agents": "invalid",  # Should be list
            "multimodal_content": {},
            "session_id": "test-session",
            "user_id": "test_user",
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id="test-session",
            user_id="test_user",
            mode="direct",
            active_agent="test_agent",
            created_at=now,
            last_activity=now,
            state=state,
            multimodal_files=[],
        )

        # Should fix invalid types
        state_manager._validate_session_state(session)

        assert isinstance(session.state["messages"], list)
        assert isinstance(session.state["scratchpad"], dict)
        assert isinstance(session.state["active_agents"], list)
        assert session.state["mode"] == "direct"  # Fixed invalid mode

    def test_health_check_healthy(self, state_manager, mock_redis):
        """Test health check when system is healthy."""
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = ["session1", "session2", "session3"]

        health = state_manager.health_check()

        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["total_sessions"] == 3
        assert health["session_timeout_hours"] == 24

    def test_health_check_unhealthy(self, state_manager, mock_redis):
        """Test health check when Redis is unavailable."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        health = state_manager.health_check()

        assert health["status"] == "unhealthy"
        assert health["redis_connected"] is False
        assert "error" in health


if __name__ == "__main__":
    pytest.main([__file__])
