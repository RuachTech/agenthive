"""Main FastAPI application for AgentHive."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
import structlog
from datetime import datetime

from agent_hive.core.config import SystemConfig

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    # Startup
    logger.info("Starting AgentHive API server")

    # Initialize system components here
    # TODO: Initialize Redis connection, load agent configurations, etc.

    yield

    # Shutdown
    logger.info("Shutting down AgentHive API server")

    # Cleanup resources here
    # TODO: Close Redis connection, cleanup temporary files, etc.


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

    return app


# Create the main application instance
app = create_app()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify system status."""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "components": {
                "api": "healthy",
                # TODO: Add checks for Redis, model providers, etc.
                # "redis": await check_redis_health(),
                # "models": await check_model_providers(),
                # "agents": await check_agent_availability(),
            },
        }

        logger.info("Health check completed", status="healthy")
        return health_status

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            },
        )


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    return {
        "name": "AgentHive API",
        "version": "0.1.0",
        "description": "Unified multi-agent system built on LangGraph",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/status")
async def system_status() -> Dict[str, Any]:
    """Detailed system status endpoint."""
    config = app.state.config

    return {
        "system": {
            "name": "AgentHive",
            "version": "0.1.0",
            "uptime": "TODO: Calculate uptime",
            "mode": "development",  # TODO: Determine from environment
        },
        "configuration": {
            "api_host": config.api_host,
            "api_port": config.api_port,
            "log_level": config.log_level,
            "metrics_enabled": config.enable_metrics,
            "session_timeout": config.session_timeout,
        },
        "agents": {
            "total_configured": 0,  # TODO: Count loaded agents
            "active": 0,  # TODO: Count active agents
            "available_modes": ["direct", "orchestration"],
        },
    }
