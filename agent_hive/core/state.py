"""Core state management for AgentHive."""

import json
import logging
import operator
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import TypedDict, Annotated, Any, Optional, Dict, List

import redis
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state object that maintains context across all agent interactions."""

    # Primary task description
    task: str

    # Conversation history with automatic message aggregation
    messages: Annotated[list[BaseMessage], operator.add]

    # Next agent to route to in orchestration mode
    next: str

    # Agent working memory for intermediate results and context
    scratchpad: dict[str, Any]

    # Operating mode: "direct" or "orchestration"
    mode: str

    # list of currently involved agents
    active_agents: list[str]

    # Processed files and media content
    multimodal_content: dict[str, Any]

    # Unique session identifier
    session_id: str

    # Optional user identifier for multi-user scenarios
    user_id: Optional[str]

    # Timestamp of last state update
    last_updated: Optional[datetime]

    # Error tracking and recovery information
    errors: list[dict[str, Any]]

    # Task completion status and progress tracking
    task_status: dict[str, Any]


@dataclass
class ProcessedFile:
    """Represents a processed multimodal file with metadata."""

    file_id: str
    original_name: str
    file_type: str  # "image", "pdf", "document"
    processed_content: Dict[str, Any]  # Extracted text, image analysis, etc.
    metadata: Dict[str, Any]
    processing_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["processing_timestamp"] = self.processing_timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedFile":
        """Create from dictionary for JSON deserialization."""
        data["processing_timestamp"] = datetime.fromisoformat(
            data["processing_timestamp"]
        )
        return cls(**data)


@dataclass
class Session:
    """Session management for user interactions with AgentHive."""

    session_id: str
    user_id: str
    mode: str  # "direct" or "orchestration"
    active_agent: Optional[str]
    created_at: datetime
    last_activity: datetime
    state: AgentState
    multimodal_files: List[ProcessedFile]

    def __post_init__(self) -> None:
        """Ensure session_id is set in the state."""
        if self.state.get("session_id") != self.session_id:
            self.state["session_id"] = self.session_id

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
        self.state["last_updated"] = self.last_activity

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session has expired based on last activity."""
        expiry_time = self.last_activity + timedelta(hours=timeout_hours)
        return datetime.utcnow() > expiry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "active_agent": self.active_agent,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "state": dict(self.state),  # Convert TypedDict to regular dict
            "multimodal_files": [f.to_dict() for f in self.multimodal_files],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from stored dictionary."""
        # Convert datetime strings back to datetime objects
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_activity"] = datetime.fromisoformat(data["last_activity"])

        # Convert multimodal files
        data["multimodal_files"] = [
            ProcessedFile.from_dict(f) for f in data.get("multimodal_files", [])
        ]

        # Ensure state has proper structure
        state = data["state"]
        if "last_updated" in state and isinstance(state["last_updated"], str):
            state["last_updated"] = datetime.fromisoformat(state["last_updated"])

        return cls(**data)


class StateValidationError(Exception):
    """Raised when state validation fails."""

    pass


class StateManager:
    """Manages session state with Redis-based persistence."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", session_timeout_hours: int = 24
    ):
        """
        Initialize StateManager with Redis connection.

        Args:
            redis_url: Redis connection URL
            session_timeout_hours: Hours before session expires
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.session_timeout_hours = session_timeout_hours
        self.session_prefix = "agent_hive:session:"
        self.user_sessions_prefix = "agent_hive:user_sessions:"

    def create_session(
        self, user_id: str, mode: str = "direct", active_agent: Optional[str] = None
    ) -> Session:
        """
        Create a new session for a user.

        Args:
            user_id: Unique user identifier
            mode: Operating mode ("direct" or "orchestration")
            active_agent: Initially active agent (for direct mode)

        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Initialize empty state
        initial_state: AgentState = {
            "task": "",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": mode,
            "active_agents": [active_agent] if active_agent else [],
            "multimodal_content": {},
            "session_id": session_id,
            "user_id": user_id,
            "last_updated": now,
            "errors": [],
            "task_status": {},
        }

        session = Session(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            active_agent=active_agent,
            created_at=now,
            last_activity=now,
            state=initial_state,
            multimodal_files=[],
        )

        # Store session in Redis
        self._store_session(session)
        self._add_to_user_sessions(user_id, session_id)

        logger.info("Created new session %s for user %s", session_id, user_id)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object if found, None otherwise
        """
        try:
            session_key = f"{self.session_prefix}{session_id}"
            session_data = self.redis_client.get(session_key)

            if not session_data:
                logger.warning("Session %s not found", session_id)
                return None

            session_dict = json.loads(session_data)
            session = Session.from_dict(session_dict)

            # Check if session has expired
            if session.is_expired(self.session_timeout_hours):
                logger.info("Session %s has expired, cleaning up", session_id)
                self.delete_session(session_id)
                return None

            # Validate session state
            self._validate_session_state(session)

            return session

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Error retrieving session %s: %s", session_id, e)
            # Attempt to recover or clean up corrupted session
            self._handle_corrupted_session(session_id)
            return None

    def update_session(self, session: Session) -> bool:
        """
        Update an existing session.

        Args:
            session: Session object to update

        Returns:
            True if successful, False otherwise
        """
        try:
            session.update_activity()
            self._validate_session_state(session)
            self._store_session(session)
            logger.debug("Updated session %s", session.session_id)
            return True

        except Exception as e:
            logger.error("Error updating session %s: %s", session.session_id, e)
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up associated data.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            session_key = f"{self.session_prefix}{session_id}"

            # Get session to find user_id for cleanup
            session_data = self.redis_client.get(session_key)
            if session_data:
                session_dict = json.loads(session_data)
                user_id = session_dict.get("user_id")
                if user_id:
                    self._remove_from_user_sessions(user_id, session_id)

            # Delete session data
            deleted = self.redis_client.delete(session_key)

            if deleted:
                logger.info("Deleted session %s", session_id)
                return True

            logger.warning("Session %s not found for deletion", session_id)
            return False

        except Exception as e:
            logger.error("Error deleting session %s: %s", session_id, e)
            return False

    def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get all active session IDs for a user.

        Args:
            user_id: User identifier

        Returns:
            List of session IDs
        """
        try:
            user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
            session_ids = self.redis_client.smembers(user_sessions_key)

            # Filter out expired sessions
            active_sessions = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session:  # get_session handles expiry checking
                    active_sessions.append(session_id)
                else:
                    # Clean up reference to expired/invalid session
                    self._remove_from_user_sessions(user_id, session_id)

            return active_sessions

        except Exception as e:
            logger.error("Error getting sessions for user %s: %s", user_id, e)
            return []

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        try:
            # Get all session keys
            session_pattern = f"{self.session_prefix}*"
            session_keys = self.redis_client.keys(session_pattern)

            cleaned_count = 0
            for session_key in session_keys:
                session_id = session_key.replace(self.session_prefix, "")
                session = self.get_session(session_id)  # This handles expiry checking
                if not session:
                    cleaned_count += 1

            logger.info("Cleaned up %d expired sessions", cleaned_count)
            return cleaned_count

        except Exception as e:
            logger.error("Error during session cleanup: %s", e)
            return 0

    def _store_session(self, session: Session) -> None:
        """Store session data in Redis."""
        session_key = f"{self.session_prefix}{session.session_id}"
        session_data = json.dumps(session.to_dict(), default=str)

        # Set with expiration (add buffer to timeout)
        expiry_seconds = (self.session_timeout_hours + 1) * 3600
        self.redis_client.setex(session_key, expiry_seconds, session_data)

    def _add_to_user_sessions(self, user_id: str, session_id: str) -> None:
        """Add session ID to user's session set."""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.sadd(user_sessions_key, session_id)

        # Set expiration on user sessions set
        expiry_seconds = (self.session_timeout_hours + 1) * 3600
        self.redis_client.expire(user_sessions_key, expiry_seconds)

    def _remove_from_user_sessions(self, user_id: str, session_id: str) -> None:
        """Remove session ID from user's session set."""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.srem(user_sessions_key, session_id)

    def _validate_session_state(self, session: Session) -> None:
        """
        Validate session state structure and recover if possible.

        Args:
            session: Session to validate

        Raises:
            StateValidationError: If state is invalid and cannot be recovered
        """
        state = session.state

        # Check required fields
        required_fields = [
            "task",
            "messages",
            "next",
            "scratchpad",
            "mode",
            "active_agents",
            "multimodal_content",
            "session_id",
        ]

        for field in required_fields:
            if field not in state:
                logger.warning(
                    "Missing required field '%s' in session %s",
                    field,
                    session.session_id,
                )
                # Attempt to recover with default values
                self._recover_missing_field(dict(state), field)

        # Validate field types
        if not isinstance(state.get("messages", []), list):
            logger.warning("Invalid messages type in session %s", session.session_id)
            state["messages"] = []

        if not isinstance(state.get("active_agents", []), list):
            logger.warning(
                "Invalid active_agents type in session %s", session.session_id
            )
            state["active_agents"] = []

        if not isinstance(state.get("scratchpad", {}), dict):
            logger.warning("Invalid scratchpad type in session %s", session.session_id)
            state["scratchpad"] = {}

        # Validate mode
        if state.get("mode") not in ["direct", "orchestration"]:
            logger.warning("Invalid mode in session %s", session.session_id)
            state["mode"] = "direct"

        # Ensure session_id matches
        if state.get("session_id") != session.session_id:
            state["session_id"] = session.session_id

    def _recover_missing_field(self, state: Dict[str, Any], field: str) -> None:
        """Recover missing state field with appropriate default value."""
        defaults: Dict[str, Any] = {
            "task": "",
            "messages": [],
            "next": "",
            "scratchpad": {},
            "mode": "direct",
            "active_agents": [],
            "multimodal_content": {},
            "session_id": str(uuid.uuid4()),
            "user_id": None,
            "last_updated": datetime.utcnow(),
            "errors": [],
            "task_status": {},
        }

        state[field] = defaults.get(field, None)
        logger.info("Recovered missing field '%s' with default value", field)

    def _handle_corrupted_session(self, session_id: str) -> None:
        """Handle corrupted session data by attempting cleanup."""
        try:
            logger.warning("Handling corrupted session %s", session_id)
            # Direct cleanup without trying to parse corrupted data
            session_key = f"{self.session_prefix}{session_id}"
            self.redis_client.delete(session_key)
        except Exception as e:
            logger.error("Error handling corrupted session %s: %s", session_id, e)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on state management system.

        Returns:
            Health status information
        """
        try:
            # Test Redis connection
            self.redis_client.ping()

            # Get basic stats
            session_pattern = f"{self.session_prefix}*"
            total_sessions = len(self.redis_client.keys(session_pattern))

            return {
                "status": "healthy",
                "redis_connected": True,
                "total_sessions": total_sessions,
                "session_timeout_hours": self.session_timeout_hours,
            }

        except Exception as e:
            return {"status": "unhealthy", "redis_connected": False, "error": str(e)}
