"""Request validation and error handling for AgentHive API."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import re
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status


class OperationMode(str, Enum):
    """Valid operation modes for AgentHive."""

    DIRECT = "direct"
    ORCHESTRATION = "orchestration"


class ChatRequest(BaseModel):
    """Request model for direct chat interactions."""

    agent_name: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, pattern=r"^[a-f0-9-]{36}$")
    user_id: str = Field("default_user", min_length=1, max_length=100)
    stream: bool = Field(False)

    @validator("agent_name")
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent name format."""
        if not re.match(r"^[a-z_][a-z0-9_]*$", v):
            raise ValueError(
                "Agent name must contain only lowercase letters, numbers, and underscores"
            )
        return v

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "User ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class OrchestrationRequest(BaseModel):
    """Request model for task orchestration."""

    task: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, pattern=r"^[a-f0-9-]{36}$")
    user_id: str = Field("default_user", min_length=1, max_length=100)
    stream: bool = Field(False)

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "User ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class SessionCreateRequest(BaseModel):
    """Request model for session creation."""

    user_id: str = Field(..., min_length=1, max_length=100)
    mode: OperationMode = Field(OperationMode.DIRECT)
    active_agent: Optional[str] = Field(None, min_length=1, max_length=100)

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "User ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @validator("active_agent")
    def validate_active_agent(
        cls, v: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        """Validate active agent is provided for direct mode."""
        if values.get("mode") == OperationMode.DIRECT and not v:
            raise ValueError("Active agent must be specified for direct mode")
        if v and not re.match(r"^[a-z_][a-z0-9_]*$", v):
            raise ValueError(
                "Agent name must contain only lowercase letters, numbers, and underscores"
            )
        return v


class FileUploadRequest(BaseModel):
    """Request model for file uploads."""

    user_id: str = Field("default_user", min_length=1, max_length=100)

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "User ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Exception for validation errors."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(
            message=message, status_code=status.HTTP_400_BAD_REQUEST, details=details
        )


class AgentNotFoundError(APIError):
    """Exception for agent not found errors."""

    def __init__(self, agent_name: str, available_agents: List[str]):
        super().__init__(
            message=f"Agent '{agent_name}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"agent_name": agent_name, "available_agents": available_agents},
        )


class SessionNotFoundError(APIError):
    """Exception for session not found errors."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id},
        )


class FileProcessingError(APIError):
    """Exception for file processing errors."""

    def __init__(self, filename: str, error_message: str):
        super().__init__(
            message=f"Failed to process file '{filename}': {error_message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"filename": filename, "error_message": error_message},
        )


class ServiceUnavailableError(APIError):
    """Exception for service unavailability errors."""

    def __init__(self, service: str, details: Optional[str] = None):
        message = f"Service '{service}' is currently unavailable"
        if details:
            message += f": {details}"

        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service, "details": details},
        )


class RateLimitError(APIError):
    """Exception for rate limiting errors."""

    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after},
        )


class RequestValidator:
    """Utility class for request validation."""

    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> Optional[str]:
        """Validate session ID format."""
        if session_id is None:
            return None

        if not re.match(r"^[a-f0-9-]{36}$", session_id):
            raise ValidationError(
                "Invalid session ID format. Must be a valid UUID",
                field="session_id",
                value=session_id,
            )

        return session_id

    @staticmethod
    def validate_agent_name(agent_name: str, available_agents: List[str]) -> str:
        """Validate agent name exists and is available."""
        if not agent_name:
            raise ValidationError("Agent name is required", field="agent_name")

        if not re.match(r"^[a-z_][a-z0-9_]*$", agent_name):
            raise ValidationError(
                "Agent name must contain only lowercase letters, numbers, and underscores",
                field="agent_name",
                value=agent_name,
            )

        if agent_name not in available_agents:
            raise AgentNotFoundError(agent_name, available_agents)

        return agent_name

    @staticmethod
    def validate_message_content(message: str) -> str:
        """Validate message content."""
        if not message or not message.strip():
            raise ValidationError("Message content cannot be empty", field="message")

        if len(message) > 10000:
            raise ValidationError(
                "Message content exceeds maximum length of 10,000 characters",
                field="message",
                value=f"{len(message)} characters",
            )

        return message.strip()

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate user ID format."""
        if not user_id:
            raise ValidationError("User ID is required", field="user_id")

        if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
            raise ValidationError(
                "User ID must contain only alphanumeric characters, hyphens, and underscores",
                field="user_id",
                value=user_id,
            )

        if len(user_id) > 100:
            raise ValidationError(
                "User ID exceeds maximum length of 100 characters",
                field="user_id",
                value=user_id,
            )

        return user_id

    @staticmethod
    def validate_operation_mode(mode: str) -> str:
        """Validate operation mode."""
        valid_modes = [m.value for m in OperationMode]

        if mode not in valid_modes:
            raise ValidationError(
                f"Invalid operation mode. Must be one of: {', '.join(valid_modes)}",
                field="mode",
                value=mode,
            )

        return mode

    @staticmethod
    def validate_file_upload(
        filename: str, content_type: Optional[str], file_size: int
    ) -> None:
        """Validate file upload parameters."""
        # Maximum file size: 10MB
        MAX_FILE_SIZE = 10 * 1024 * 1024

        if not filename:
            raise ValidationError("Filename is required", field="filename")

        if file_size > MAX_FILE_SIZE:
            raise ValidationError(
                f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)",
                field="file_size",
                value=file_size,
            )

        # Validate filename format
        if not re.match(r"^[a-zA-Z0-9._-]+$", filename):
            raise ValidationError(
                "Filename contains invalid characters. Only alphanumeric characters, dots, hyphens, and underscores are allowed",
                field="filename",
                value=filename,
            )

        # Check for potentially dangerous file extensions
        dangerous_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif", ".com", ".jar"]
        file_extension = (
            "." + filename.split(".")[-1].lower() if "." in filename else ""
        )

        if file_extension in dangerous_extensions:
            raise ValidationError(
                f"File type '{file_extension}' is not allowed for security reasons",
                field="filename",
                value=filename,
            )


class ResponseFormatter:
    """Utility class for formatting API responses."""

    @staticmethod
    def success_response(
        data: Any,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format successful response."""
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        if message:
            response["message"] = message

        if metadata:
            response["metadata"] = metadata

        return response

    @staticmethod
    def error_response(
        error: Union[APIError, Exception], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format error response."""
        if isinstance(error, APIError):
            response = {
                "success": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": {
                    "message": error.message,
                    "type": type(error).__name__,
                    "details": error.details,
                },
            }
        else:
            response = {
                "success": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                    "details": {},
                },
            }

        if request_id:
            response["request_id"] = request_id

        return response

    @staticmethod
    def paginated_response(
        items: List[Any],
        total_count: int,
        page: int = 1,
        page_size: int = 20,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format paginated response."""
        total_pages = (total_count + page_size - 1) // page_size

        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1,
            },
        }

        if metadata:
            response["metadata"] = metadata

        return response


def create_http_exception(error: APIError) -> HTTPException:
    """Convert APIError to HTTPException."""
    return HTTPException(
        status_code=error.status_code, detail=ResponseFormatter.error_response(error)
    )


def validate_request_size(
    content_length: Optional[int], max_size: int = 50 * 1024 * 1024
) -> None:
    """Validate request content length."""
    if content_length and content_length > max_size:
        raise ValidationError(
            f"Request size ({content_length} bytes) exceeds maximum allowed size ({max_size} bytes)",
            field="content_length",
            value=content_length,
        )


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove null bytes
    text = text.replace("\x00", "")

    # Limit length
    if len(text) > 10000:
        text = text[:10000]

    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()
