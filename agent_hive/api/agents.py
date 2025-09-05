"""Agent-related API endpoints and functionality."""

from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
import structlog

from agent_hive.api.validation import (
    ResponseFormatter,
    APIError,
    ServiceUnavailableError,
    create_http_exception,
)

if TYPE_CHECKING:
    from agent_hive.api.core import AgentHiveAPI

logger = structlog.get_logger()


class AgentService:
    """Service class for agent-related operations."""

    def __init__(self, api_instance: "AgentHiveAPI") -> None:
        self.api = api_instance

    async def get_agent_status(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get agent status and session information.

        Args:
            session_id: Optional session ID to get specific session status

        Returns:
            Agent and session status information
        """
        try:
            # Get available agents
            available_agents = await self.api.graph_factory.get_available_agents()

            # Get model status
            model_status = await self.api.model_factory.check_all_models()

            status_info = {
                "agents": {
                    "available": available_agents,
                    "total_count": len(available_agents),
                    "healthy_count": sum(
                        1 for agent in available_agents if agent["available"]
                    ),
                },
                "models": {
                    "status": model_status,
                    "total_count": len(model_status),
                    "healthy_count": sum(
                        1 for status in model_status.values() if status
                    ),
                },
                "system": {
                    "uptime_seconds": (
                        datetime.utcnow() - self.api._startup_time
                    ).total_seconds(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            # Add session-specific information if session_id provided
            if session_id:
                session = await self.api.session_manager.get_session(session_id)
                if session:
                    status_info["session"] = {
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "mode": session.mode,
                        "active_agent": session.active_agent,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "message_count": len(session.state.get("messages", [])),
                        "active_agents": session.state.get("active_agents", []),
                        "multimodal_files": len(session.multimodal_files),
                    }
                else:
                    status_info["session"] = {"error": "Session not found"}

            return ResponseFormatter.success_response(
                data=status_info, message="Agent status retrieved successfully"
            )

        except Exception as e:
            logger.error("Failed to get agent status", error=str(e))
            error = ServiceUnavailableError("agent_status", str(e))
            raise create_http_exception(error)

    async def get_available_tools(
        self, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available tools for agents.

        Args:
            agent_name: Optional specific agent name

        Returns:
            Available tools information
        """
        try:
            if agent_name:
                # Get tools for specific agent
                available_agents = self.api.agent_factory.list_agents()
                if agent_name not in available_agents:
                    raise APIError(
                        f"Agent '{agent_name}' not found",
                        status_code=404,
                        details={
                            "agent_name": agent_name,
                            "available_agents": available_agents,
                        },
                    )

                config = self.api.agent_factory.get_agent_config(agent_name)
                tools_info = {
                    "agent": agent_name,
                    "core_tools": config.core_tools if config else [],
                    "composio_tools": config.composio_tools if config else [],
                    "capabilities": config.capabilities if config else [],
                }
            else:
                # Get tools for all agents
                tools_info: Dict[str, Any] = {"agents": {}}
                for agent in self.api.agent_factory.list_agents():
                    config = self.api.agent_factory.get_agent_config(agent)
                    tools_info["agents"][agent] = {
                        "core_tools": config.core_tools if config else [],
                        "composio_tools": config.composio_tools if config else [],
                        "capabilities": config.capabilities if config else [],
                    }

                # Add supported file formats
                tools_info["supported_formats"] = (
                    self.api.multimodal_processor.get_supported_formats()
                )

            return ResponseFormatter.success_response(
                data=tools_info, message="Available tools retrieved successfully"
            )

        except APIError as e:
            logger.error("Failed to get available tools", error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error("Failed to get available tools", error=str(e))
            error = ServiceUnavailableError("tools_discovery", str(e))
            raise create_http_exception(error)

    async def get_available_agents(self) -> Dict[str, Any]:
        """
        Get list of all available agents with their details.

        Returns:
            List of available agents with metadata
        """
        try:
            agents = await self.api.graph_factory.get_available_agents()

            return ResponseFormatter.success_response(
                data={"agents": agents},
                message="Available agents retrieved successfully",
            )

        except Exception as e:
            logger.error("Failed to get available agents", error=str(e))
            error = ServiceUnavailableError("agent_discovery", str(e))
            raise create_http_exception(error)
