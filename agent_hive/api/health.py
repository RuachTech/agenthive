"""Health check and system status endpoints."""

from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING
import structlog

from agent_hive.api.validation import (
    ResponseFormatter,
    ServiceUnavailableError,
    create_http_exception,
)

if TYPE_CHECKING:
    from agent_hive.api.core import AgentHiveAPI

logger = structlog.get_logger()


class HealthService:
    """Service class for health and status operations."""

    def __init__(self, api_instance: "AgentHiveAPI") -> None:
        self.api = api_instance

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check endpoint.

        Returns:
            System health status
        """
        try:
            # Check state manager health
            state_health = (
                self.api.state_manager.health_check()
                if self.api.state_manager
                else {"status": "not_initialized"}
            )

            # Check model availability
            model_status = await self.api.model_factory.check_all_models()

            # Check agent availability
            available_agents = await self.api.graph_factory.get_available_agents()
            healthy_agents = sum(1 for agent in available_agents if agent["available"])

            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "0.1.0",
                "components": {
                    "api": "healthy",
                    "state_manager": state_health["status"],
                    "models": {
                        "total": len(model_status),
                        "healthy": sum(1 for status in model_status.values() if status),
                        "details": model_status,
                    },
                    "agents": {
                        "total": len(available_agents),
                        "healthy": healthy_agents,
                    },
                },
            }

            # Determine overall health
            if (
                state_health["status"] != "healthy"
                or sum(model_status.values()) == 0
                or healthy_agents == 0
            ):
                health_status["status"] = "degraded"

            logger.info("Health check completed", status=health_status["status"])

            return ResponseFormatter.success_response(
                data=health_status, message="Health check completed"
            )

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            error = ServiceUnavailableError("health_check", str(e))
            raise create_http_exception(error)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get detailed system status information.

        Returns:
            System status details
        """
        try:
            system_info = self.api.get_system_info()

            return ResponseFormatter.success_response(
                data=system_info, message="System status retrieved successfully"
            )

        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            error = ServiceUnavailableError("system_status", str(e))
            raise create_http_exception(error)

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get basic API information.

        Returns:
            API information and available endpoints
        """
        try:
            api_info = {
                "name": "AgentHive API",
                "version": "0.1.0",
                "description": "Unified multi-agent system built on LangGraph",
                "docs": "/docs",
                "health": "/health",
                "endpoints": {
                    "direct_chat": "/api/v1/chat/direct",
                    "orchestrate": "/api/v1/chat/orchestrate",
                    "upload": "/api/v1/files/upload",
                    "agents": "/api/v1/agents",
                    "sessions": "/api/v1/sessions",
                },
            }

            return ResponseFormatter.success_response(
                data=api_info, message="API information retrieved successfully"
            )

        except Exception as e:
            logger.error("Failed to get API info", error=str(e))
            error = ServiceUnavailableError("api_info", str(e))
            raise create_http_exception(error)
