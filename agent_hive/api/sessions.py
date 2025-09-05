"""Session management API endpoints and functionality."""

from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
import structlog

from agent_hive.api.validation import (
    RequestValidator,
    ResponseFormatter,
    APIError,
    ValidationError,
    ServiceUnavailableError,
    create_http_exception,
)

if TYPE_CHECKING:
    from agent_hive.api.core import AgentHiveAPI

logger = structlog.get_logger()


class SessionService:
    """Service class for session management operations."""

    def __init__(self, api_instance: "AgentHiveAPI") -> None:
        self.api = api_instance

    async def create_session(
        self, user_id: str, mode: str = "direct", active_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new session.

        Args:
            user_id: User identifier
            mode: Session mode ("direct" or "orchestration")
            active_agent: Initially active agent (for direct mode)

        Returns:
            Created session information
        """
        try:
            # Validate inputs
            user_id = RequestValidator.validate_user_id(user_id)
            mode = RequestValidator.validate_operation_mode(mode)

            if mode == "direct" and active_agent:
                available_agents = self.api.agent_factory.list_agents()
                active_agent = RequestValidator.validate_agent_name(
                    active_agent, available_agents
                )
            elif mode == "direct" and not active_agent:
                raise ValidationError(
                    "Active agent must be specified for direct mode",
                    field="active_agent",
                )

            session = self.api.state_manager.create_session(
                user_id=user_id, mode=mode, active_agent=active_agent
            )

            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "mode": session.mode,
                "active_agent": session.active_agent,
                "created_at": session.created_at.isoformat(),
                "status": "created",
            }

            return ResponseFormatter.success_response(
                data=session_data, message="Session created successfully"
            )

        except APIError as e:
            logger.error("Session creation validation failed", error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error("Failed to create session", error=str(e))
            error = ServiceUnavailableError("session_management", str(e))
            raise create_http_exception(error)

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            Deletion result
        """
        try:
            # Validate session ID
            validated_session_id = RequestValidator.validate_session_id(session_id)
            if not validated_session_id:
                raise ValidationError("Session ID is required", field="session_id")
            session_id = validated_session_id

            success = self.api.state_manager.delete_session(session_id)

            if success:
                deletion_data = {
                    "session_id": session_id,
                    "status": "deleted",
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return ResponseFormatter.success_response(
                    data=deletion_data, message="Session deleted successfully"
                )
            else:
                raise APIError(
                    "Session not found",
                    status_code=404,
                    details={"session_id": session_id},
                )

        except APIError as e:
            logger.error("Session deletion failed", session_id=session_id, error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error(
                "Failed to delete session", session_id=session_id, error=str(e)
            )
            error = ServiceUnavailableError("session_management", str(e))
            raise create_http_exception(error)

    async def list_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """
        List all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of user sessions
        """
        try:
            # Validate user ID
            user_id = RequestValidator.validate_user_id(user_id)

            session_ids = self.api.state_manager.get_user_sessions(user_id)
            sessions = []

            for session_id in session_ids:
                session = self.api.state_manager.get_session(session_id)
                if session:
                    sessions.append(
                        {
                            "session_id": session.session_id,
                            "mode": session.mode,
                            "active_agent": session.active_agent,
                            "created_at": session.created_at.isoformat(),
                            "last_activity": session.last_activity.isoformat(),
                            "message_count": len(session.state.get("messages", [])),
                            "multimodal_files": len(session.multimodal_files),
                        }
                    )

            sessions_data = {
                "user_id": user_id,
                "sessions": sessions,
                "total_count": len(sessions),
            }

            return ResponseFormatter.success_response(
                data=sessions_data,
                message=f"Retrieved {len(sessions)} sessions for user",
            )

        except APIError as e:
            logger.error("Failed to list user sessions", user_id=user_id, error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error("Failed to list user sessions", user_id=user_id, error=str(e))
            error = ServiceUnavailableError("session_management", str(e))
            raise create_http_exception(error)

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Session information
        """
        try:
            # Validate session ID
            validated_session_id = RequestValidator.validate_session_id(session_id)
            if not validated_session_id:
                raise ValidationError("Session ID is required", field="session_id")
            session_id = validated_session_id

            session = self.api.state_manager.get_session(session_id)

            if not session:
                raise APIError(
                    "Session not found",
                    status_code=404,
                    details={"session_id": session_id},
                )

            session_info = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "mode": session.mode,
                "active_agent": session.active_agent,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "state": {
                    "message_count": len(session.state.get("messages", [])),
                    "active_agents": session.state.get("active_agents", []),
                    "task_status": session.state.get("task_status", {}),
                    "errors": session.state.get("errors", []),
                },
                "multimodal_files": [
                    {
                        "file_id": f.file_id,
                        "original_name": f.original_name,
                        "file_type": f.file_type.value,
                        "processing_timestamp": f.processing_timestamp.isoformat(),
                    }
                    for f in session.multimodal_files
                ],
            }

            return ResponseFormatter.success_response(
                data=session_info, message="Session information retrieved successfully"
            )

        except APIError as e:
            logger.error(
                "Failed to get session info", session_id=session_id, error=str(e)
            )
            raise create_http_exception(e)
        except Exception as e:
            logger.error(
                "Failed to get session info", session_id=session_id, error=str(e)
            )
            error = ServiceUnavailableError("session_management", str(e))
            raise create_http_exception(error)

    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """
        Trigger cleanup of expired sessions.

        Returns:
            Cleanup result
        """
        try:
            cleaned_count = self.api.state_manager.cleanup_expired_sessions()

            cleanup_data = {
                "cleaned_sessions": cleaned_count,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
            }

            return ResponseFormatter.success_response(
                data=cleanup_data,
                message=f"Cleaned up {cleaned_count} expired sessions",
            )

        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            error = ServiceUnavailableError("session_cleanup", str(e))
            raise create_http_exception(error)
