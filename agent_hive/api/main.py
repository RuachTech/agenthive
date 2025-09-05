"""Main FastAPI application for AgentHive."""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Dict, Any, Union
import structlog
from datetime import datetime

from agent_hive.core.config import SystemConfig
from agent_hive.api.core import AgentHiveAPI, get_api, set_api_instance
from agent_hive.api.chat import ChatService
from agent_hive.api.files import FileService
from agent_hive.api.agents import AgentService
from agent_hive.api.sessions import SessionService
from agent_hive.api.health import HealthService
from agent_hive.api.validation import (
    APIError,
    ResponseFormatter,
    ValidationError,
    validate_request_size,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Service instances
_chat_service: Optional[ChatService] = None
_file_service: Optional[FileService] = None
_agent_service: Optional[AgentService] = None
_session_service: Optional[SessionService] = None
_health_service: Optional[HealthService] = None


def get_chat_service() -> ChatService:
    """Get chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(get_api())
    return _chat_service


def get_file_service() -> FileService:
    """Get file service instance."""
    global _file_service
    if _file_service is None:
        _file_service = FileService(get_api())
    return _file_service


def get_agent_service() -> AgentService:
    """Get agent service instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService(get_api())
    return _agent_service


def get_session_service() -> SessionService:
    """Get session service instance."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService(get_api())
    return _session_service


def get_health_service() -> HealthService:
    """Get health service instance."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService(get_api())
    return _health_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    # Startup
    logger.info("Starting AgentHive API server")

    try:
        # Initialize API instance
        config = getattr(app.state, "config", SystemConfig())
        api_instance = AgentHiveAPI(config)
        await api_instance.initialize()

        # Set global instance
        set_api_instance(api_instance)

        # Store in app state
        app.state.api = api_instance

        logger.info("AgentHive API server started successfully")

    except Exception as e:
        logger.error("Failed to start AgentHive API server", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down AgentHive API server")

    try:
        api_instance = get_api()
        await api_instance.cleanup()
        logger.info("AgentHive API server shutdown completed")

    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def create_app(config: SystemConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    if config is None:
        config = SystemConfig()

    app = FastAPI(
        title="AgentHive API",
        description="Unified multi-agent system built on LangGraph",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state
    app.state.config = config

    # Add custom exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handle custom API errors."""
        logger.error(
            "API error occurred",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )

        return JSONResponse(
            status_code=exc.status_code, content=ResponseFormatter.error_response(exc)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        logger.warning(
            "Request validation failed",
            path=request.url.path,
            method=request.method,
            errors=exc.errors(),
        )

        validation_error = ValidationError(
            message="Request validation failed", field="request_body"
        )
        validation_error.details["validation_errors"] = exc.errors()

        return JSONResponse(
            status_code=400, content=ResponseFormatter.error_response(validation_error)
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected errors."""
        logger.error(
            "Unexpected error occurred",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )

        from agent_hive.api.validation import ServiceUnavailableError

        error = ServiceUnavailableError(
            service="api",
            details="An unexpected error occurred. Please try again later.",
        )

        return JSONResponse(
            status_code=500, content=ResponseFormatter.error_response(error)
        )

    # Add request size validation middleware
    @app.middleware("http")
    async def validate_request_size_middleware(
        request: Request, call_next: Any
    ) -> Response:
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                validate_request_size(int(content_length))
            except ValidationError as e:
                return JSONResponse(
                    status_code=413, content=ResponseFormatter.error_response(e)
                )

        response = await call_next(request)
        return response  # type: ignore

    return app


# Create the main application instance
app = create_app()


# Health and Status Endpoints
@app.get("/health")
async def health_check(
    health_service: HealthService = Depends(get_health_service),
) -> Dict[str, Any]:
    """Health check endpoint to verify system status."""
    return await health_service.health_check()


@app.get("/")
async def root(
    health_service: HealthService = Depends(get_health_service),
) -> Dict[str, Any]:
    """Root endpoint with basic API information."""
    return health_service.get_api_info()


@app.get("/status")
async def system_status(
    health_service: HealthService = Depends(get_health_service),
) -> Dict[str, Any]:
    """Detailed system status endpoint."""
    return health_service.get_system_status()


# Core API Endpoints
@app.post("/api/v1/chat/direct", response_model=None)
async def direct_chat_endpoint(
    agent_name: str = Form(...),
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
    stream: bool = Form(False),
    files: Optional[List[UploadFile]] = File(None),
    chat_service: ChatService = Depends(get_chat_service),
) -> Union[Dict[str, Any], StreamingResponse]:
    """Direct chat with a specific agent."""
    return await chat_service.direct_chat(
        agent_name=agent_name,
        message=message,
        session_id=session_id,
        user_id=user_id,
        files=files or [],
        stream=stream,
    )


@app.post("/api/v1/chat/orchestrate", response_model=None)
async def orchestrate_task_endpoint(
    task: str = Form(...),
    session_id: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
    stream: bool = Form(False),
    files: Optional[List[UploadFile]] = File(None),
    chat_service: ChatService = Depends(get_chat_service),
) -> Union[Dict[str, Any], StreamingResponse]:
    """Orchestrate a complex task across multiple agents."""
    return await chat_service.orchestrate_task(
        task=task,
        session_id=session_id,
        user_id=user_id,
        files=files or [],
        stream=stream,
    )


@app.post("/api/v1/files/upload")
async def upload_files_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form("default_user"),
    file_service: FileService = Depends(get_file_service),
) -> Dict[str, Any]:
    """Upload and process multimodal files."""
    return await file_service.upload_multimodal_content(files=files, user_id=user_id)


@app.get("/api/v1/files/formats")
async def get_supported_formats_endpoint(
    file_service: FileService = Depends(get_file_service),
) -> Dict[str, Any]:
    """Get supported file formats."""
    return file_service.get_supported_formats()


# Agent Endpoints
@app.get("/api/v1/agents")
async def get_agents_endpoint(
    agent_service: AgentService = Depends(get_agent_service),
) -> Dict[str, Any]:
    """Get available agents and their status."""
    return await agent_service.get_agent_status()


@app.get("/api/v1/agents/list")
async def list_agents_endpoint(
    agent_service: AgentService = Depends(get_agent_service),
) -> Dict[str, Any]:
    """Get list of all available agents."""
    return await agent_service.get_available_agents()


@app.get("/api/v1/agents/{agent_name}/tools")
async def get_agent_tools_endpoint(
    agent_name: str, agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get available tools for a specific agent."""
    return await agent_service.get_available_tools(agent_name=agent_name)


@app.get("/api/v1/tools")
async def get_all_tools_endpoint(
    agent_service: AgentService = Depends(get_agent_service),
) -> Dict[str, Any]:
    """Get all available tools across agents."""
    return await agent_service.get_available_tools()


# Session Management Endpoints
@app.post("/api/v1/sessions")
async def create_session_endpoint(
    user_id: str = Form(...),
    mode: str = Form("direct"),
    active_agent: Optional[str] = Form(None),
    session_service: SessionService = Depends(get_session_service),
) -> Dict[str, Any]:
    """Create a new session."""
    return await session_service.create_session(
        user_id=user_id, mode=mode, active_agent=active_agent
    )


@app.get("/api/v1/sessions/{session_id}")
async def get_session_endpoint(
    session_id: str, session_service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """Get detailed information about a specific session."""
    return await session_service.get_session_info(session_id=session_id)


@app.get("/api/v1/sessions/{session_id}/status")
async def get_session_status_endpoint(
    session_id: str, agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get status for a specific session."""
    return await agent_service.get_agent_status(session_id=session_id)


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: str, session_service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """Delete a session."""
    return await session_service.delete_session(session_id=session_id)


@app.get("/api/v1/users/{user_id}/sessions")
async def list_user_sessions_endpoint(
    user_id: str, session_service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """List all sessions for a user."""
    return await session_service.list_user_sessions(user_id=user_id)


# Administrative Endpoints
@app.post("/api/v1/admin/cleanup")
async def cleanup_expired_sessions_endpoint(
    background_tasks: BackgroundTasks,
    session_service: SessionService = Depends(get_session_service),
) -> Dict[str, Any]:
    """Trigger cleanup of expired sessions."""

    def cleanup_task() -> None:
        result = session_service.cleanup_expired_sessions()
        logger.info("Session cleanup completed", result=result)

    background_tasks.add_task(cleanup_task)

    return ResponseFormatter.success_response(
        data={
            "status": "cleanup_scheduled",
            "timestamp": datetime.utcnow().isoformat(),
        },
        message="Session cleanup scheduled",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
