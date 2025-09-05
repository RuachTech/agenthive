"""Core API functionality and AgentHiveAPI class."""

from datetime import datetime
from typing import Dict, Any, Optional
import structlog


from agent_hive.core.config import SystemConfig
from agent_hive.core.state import StateManager
from agent_hive.core.multimodal import (
    MultimodalProcessor,
)
from agent_hive.core.graphs import (
    get_graph_factory,
    get_session_manager,
    OrchestratorGraphFactory,
)
from agent_hive.agents.factory import get_agent_factory
from agent_hive.core.models import get_model_factory
from agent_hive.api.validation import (
    ServiceUnavailableError,
)

logger = structlog.get_logger()


class AgentHiveAPI:
    """Main API class for AgentHive with all endpoint implementations."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.state_manager: Optional[StateManager] = None
        self.multimodal_processor: Optional[MultimodalProcessor] = None
        self.graph_factory: Optional[Any] = None
        self.session_manager: Optional[Any] = None
        self.orchestrator_factory: Optional[OrchestratorGraphFactory] = None
        self.agent_factory = get_agent_factory()
        self.model_factory = get_model_factory()
        self._startup_time = datetime.utcnow()

    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            # Initialize state management
            redis_url = getattr(self.config, "redis_url", "redis://localhost:6379")
            self.state_manager = StateManager(redis_url=redis_url)

            # Initialize multimodal processor
            self.multimodal_processor = MultimodalProcessor()

            # Initialize graph factories
            self.session_manager = await get_session_manager()
            self.graph_factory = await get_graph_factory()
            self.orchestrator_factory = OrchestratorGraphFactory(self.session_manager)

            logger.info("AgentHive API initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize AgentHive API", error=str(e))
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.session_manager:
                await self.session_manager.stop()
            logger.info("AgentHive API cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))

    def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status information."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "uptime_seconds": (datetime.utcnow() - self._startup_time).total_seconds(),
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system configuration information."""
        uptime = datetime.utcnow() - self._startup_time

        return {
            "system": {
                "name": "AgentHive",
                "version": "0.1.0",
                "uptime_seconds": uptime.total_seconds(),
                "uptime_human": str(uptime),
                "mode": "development",  # Default mode since environment not in SystemConfig
            },
            "configuration": {
                "api_host": self.config.api_host,
                "api_port": self.config.api_port,
                "log_level": self.config.log_level,
                "metrics_enabled": self.config.enable_metrics,
                "session_timeout": self.config.session_timeout,
            },
            "agents": {
                "total_configured": len(self.agent_factory.list_agents()),
                "available_modes": ["direct", "orchestration"],
            },
        }


# Global API instance
_api_instance: Optional[AgentHiveAPI] = None


def get_api() -> AgentHiveAPI:
    """Dependency to get API instance."""
    if _api_instance is None:
        raise ServiceUnavailableError("api", "API not initialized")
    return _api_instance


def set_api_instance(api: AgentHiveAPI) -> None:
    """Set the global API instance."""
    global _api_instance
    _api_instance = api
