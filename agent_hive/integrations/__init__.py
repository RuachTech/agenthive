"""External service integrations."""

from .mcp import (
    MCPTool,
    MCPServerConfig,
    MCPExecutionResult,
    MCPError,
    MCPServerUnavailableError,
    MCPToolNotFoundError,
    MCPExecutionTimeoutError,
    MCPToolCategory,
    MCPServerInterface,
    ComposioMCPServer,
)
from .composio_client import (
    ComposioMCPClient,
    AgentToolMapping,
    get_mcp_client,
    shutdown_mcp_client,
)
from .composio_sdk import (
    ComposioSDKClient,
    ComposioAgentIntegration,
    ComposioTool,
    ComposioConnection,
    ComposioExecutionResult,
    ComposioIntegrationError,
    ComposioNotAvailableError,
    ComposioAuthenticationError,
    ComposioToolExecutionError,
    ComposioToolkit,
    get_composio_client,
    get_agent_integration,
    COMPOSIO_AVAILABLE,
)
from .unified_client import (
    UnifiedIntegrationClient,
    UnifiedTool,
    UnifiedExecutionResult,
    IntegrationMode,
    get_unified_client,
    shutdown_unified_client,
)

__all__ = [
    # MCP base classes
    "MCPTool",
    "MCPServerConfig",
    "MCPExecutionResult",
    "MCPError",
    "MCPServerUnavailableError",
    "MCPToolNotFoundError",
    "MCPExecutionTimeoutError",
    "MCPToolCategory",
    "MCPServerInterface",
    "ComposioMCPServer",
    # MCP Client classes
    "ComposioMCPClient",
    "AgentToolMapping",
    "get_mcp_client",
    "shutdown_mcp_client",
    # Composio SDK classes
    "ComposioSDKClient",
    "ComposioAgentIntegration",
    "ComposioTool",
    "ComposioConnection",
    "ComposioExecutionResult",
    "ComposioIntegrationError",
    "ComposioNotAvailableError",
    "ComposioAuthenticationError",
    "ComposioToolExecutionError",
    "ComposioToolkit",
    "get_composio_client",
    "get_agent_integration",
    "COMPOSIO_AVAILABLE",
    # Unified client classes
    "UnifiedIntegrationClient",
    "UnifiedTool",
    "UnifiedExecutionResult",
    "IntegrationMode",
    "get_unified_client",
    "shutdown_unified_client",
]
