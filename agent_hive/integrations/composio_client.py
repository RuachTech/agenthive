"""Composio MCP client for managing multiple MCP servers and tool execution."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .mcp import (
    MCPTool,
    MCPServerConfig,
    MCPExecutionResult,
    MCPServerUnavailableError,
    MCPToolNotFoundError,
    MCPToolCategory,
    ComposioMCPServer,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentToolMapping:
    """Mapping of agent types to their relevant tool categories."""

    agent_type: str
    required_categories: List[MCPToolCategory]
    optional_categories: List[MCPToolCategory] = field(default_factory=list)
    priority_tools: List[str] = field(default_factory=list)


class ComposioMCPClient:
    """Main client for managing Composio MCP servers and tool execution."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.servers: Dict[str, ComposioMCPServer] = {}
        self.tool_registry: Dict[str, List[MCPTool]] = {}
        self.agent_mappings: Dict[str, AgentToolMapping] = {}
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

        # Initialize default agent mappings
        self._setup_default_agent_mappings()

    async def initialize(self) -> None:
        """Initialize the MCP client and all configured servers."""
        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                # Load MCP server configurations
                server_configs = await self._load_server_configs()

                # Initialize MCP servers
                for config in server_configs:
                    if not config.disabled:
                        server = ComposioMCPServer(config)
                        self.servers[config.name] = server

                        # Connect to server and discover tools
                        try:
                            await server.connect()
                            tools = await server.list_tools()
                            self._register_tools(config.name, tools)
                            logger.info(
                                "Initialized MCP server: %s with %d tools",
                                config.name,
                                len(tools),
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to initialize MCP server %s: %s", config.name, e
                            )

                self._initialized = True
                logger.info(
                    "ComposioMCPClient initialized with %d servers", len(self.servers)
                )

            except Exception as e:
                logger.error("Failed to initialize ComposioMCPClient: %s", e)
                raise

    async def shutdown(self) -> None:
        """Shutdown all MCP servers and cleanup resources."""
        logger.info("Shutting down ComposioMCPClient")

        # Disconnect all servers
        disconnect_tasks = []
        for server in self.servers.values():
            disconnect_tasks.append(server.disconnect())

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.servers.clear()
        self.tool_registry.clear()
        self._initialized = False

    async def get_tools_for_agent(
        self, agent_type: str, capabilities: Optional[List[str]] = None
    ) -> List[MCPTool]:
        """Get relevant tools for a specific agent type."""
        if not self._initialized:
            await self.initialize()

        if agent_type not in self.agent_mappings:
            logger.warning("Unknown agent type: %s", agent_type)
            return []

        mapping = self.agent_mappings[agent_type]
        relevant_tools = []

        # Get tools from required categories
        for category in mapping.required_categories:
            tools = self._get_tools_by_category(category)
            relevant_tools.extend(tools)

        # Get tools from optional categories if capabilities match
        if capabilities:
            for category in mapping.optional_categories:
                if category.value in capabilities:
                    tools = self._get_tools_by_category(category)
                    relevant_tools.extend(tools)

        # Prioritize specific tools
        if mapping.priority_tools:
            prioritized = []
            others = []

            for tool in relevant_tools:
                if tool.name in mapping.priority_tools:
                    prioritized.append(tool)
                else:
                    others.append(tool)

            relevant_tools = prioritized + others

        logger.debug(
            "Found %d tools for agent type %s", len(relevant_tools), agent_type
        )
        return relevant_tools

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        server_name: Optional[str] = None,
    ) -> MCPExecutionResult:
        """Execute a tool with given parameters."""
        if not self._initialized:
            await self.initialize()

        # Find the tool and its server
        target_server = None

        if server_name:
            # Use specific server
            if server_name not in self.servers:
                raise MCPServerUnavailableError(
                    f"Server '{server_name}' not available", server_name=server_name
                )
            target_server = self.servers[server_name]
        else:
            # Find server that has the tool
            for server_name, tools in self.tool_registry.items():
                for tool in tools:
                    if tool.name == tool_name:
                        target_server = self.servers[server_name]
                        break
                if target_server:
                    break

        if not target_server:
            raise MCPToolNotFoundError(
                f"Tool '{tool_name}' not found in any available server",
                tool_name=tool_name,
            )

        # Check server health before execution
        if not await target_server.is_healthy():
            raise MCPServerUnavailableError(
                f"Server for tool '{tool_name}' is not healthy",
                server_name=target_server.config.name,
                tool_name=tool_name,
            )

        # Execute the tool
        try:
            result = await target_server.execute_tool(tool_name, parameters)
            logger.info(
                "Successfully executed tool %s on server %s",
                tool_name,
                target_server.config.name,
            )
            return result

        except Exception as e:
            logger.error("Failed to execute tool %s: %s", tool_name, e)
            raise

    async def check_tool_availability(self, tool_name: str) -> bool:
        """Check if a specific tool is available."""
        if not self._initialized:
            await self.initialize()

        for tools in self.tool_registry.values():
            if any(tool.name == tool_name for tool in tools):
                return True
        return False

    async def get_available_tools(
        self, category: Optional[MCPToolCategory] = None
    ) -> Dict[str, List[MCPTool]]:
        """Get all available tools, optionally filtered by category."""
        if not self._initialized:
            await self.initialize()

        if category:
            filtered_registry = {}
            for server_name, tools in self.tool_registry.items():
                filtered_tools = [tool for tool in tools if tool.category == category]
                if filtered_tools:
                    filtered_registry[server_name] = filtered_tools
            return filtered_registry

        return self.tool_registry.copy()

    async def reload_server_config(self, server_name: str) -> bool:
        """Reload configuration for a specific server."""
        if server_name not in self.servers:
            logger.warning("Server %s not found for reload", server_name)
            return False

        try:
            # Disconnect existing server
            await self.servers[server_name].disconnect()

            # Reload configuration
            server_configs = await self._load_server_configs()
            new_config = next(
                (config for config in server_configs if config.name == server_name),
                None,
            )

            if not new_config or new_config.disabled:
                # Remove server if disabled or not found
                del self.servers[server_name]
                if server_name in self.tool_registry:
                    del self.tool_registry[server_name]
                logger.info("Removed server %s", server_name)
                return True

            # Reinitialize server
            server = ComposioMCPServer(new_config)
            await server.connect()
            tools = await server.list_tools()

            self.servers[server_name] = server
            self._register_tools(server_name, tools)

            logger.info("Reloaded server %s with %d tools", server_name, len(tools))
            return True

        except Exception as e:
            logger.error("Failed to reload server %s: %s", server_name, e)
            return False

    async def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all servers."""
        status = {}

        for server_name, server in self.servers.items():
            try:
                is_healthy = await server.is_healthy()
                tool_count = len(self.tool_registry.get(server_name, []))

                status[server_name] = {
                    "connected": server.connected,
                    "healthy": is_healthy,
                    "tool_count": tool_count,
                    "last_health_check": server.last_health_check.isoformat(),
                    "config": {
                        "disabled": server.config.disabled,
                        "timeout": server.config.timeout,
                        "max_retries": server.config.max_retries,
                    },
                }
            except Exception as e:
                status[server_name] = {
                    "connected": False,
                    "healthy": False,
                    "error": str(e),
                }

        return status

    def _setup_default_agent_mappings(self) -> None:
        """Setup default agent type to tool category mappings."""
        self.agent_mappings = {
            "full_stack_engineer": AgentToolMapping(
                agent_type="full_stack_engineer",
                required_categories=[MCPToolCategory.GITHUB],
                optional_categories=[
                    MCPToolCategory.LINEAR,
                    MCPToolCategory.SLACK,
                    MCPToolCategory.NOTION,
                ],
                priority_tools=[
                    "create_repository",
                    "create_pull_request",
                    "list_issues",
                    "create_issue",
                ],
            ),
            "qa_engineer": AgentToolMapping(
                agent_type="qa_engineer",
                required_categories=[MCPToolCategory.GITHUB, MCPToolCategory.LINEAR],
                optional_categories=[
                    MCPToolCategory.SLACK,
                    MCPToolCategory.BROWSERBASE,
                    MCPToolCategory.SERPAPI,
                ],
                priority_tools=[
                    "list_issues",
                    "create_issue",
                    "update_issue",
                    "create_pull_request",
                ],
            ),
            "product_designer": AgentToolMapping(
                agent_type="product_designer",
                required_categories=[MCPToolCategory.FIGMA],
                optional_categories=[
                    MCPToolCategory.NOTION,
                    MCPToolCategory.SLACK,
                    MCPToolCategory.MIRO,
                    MCPToolCategory.AIRTABLE,
                ],
                priority_tools=[
                    "get_file",
                    "export_image",
                    "create_page",
                    "send_message",
                ],
            ),
            "devops_engineer": AgentToolMapping(
                agent_type="devops_engineer",
                required_categories=[MCPToolCategory.GITHUB, MCPToolCategory.AWS],
                optional_categories=[
                    MCPToolCategory.DOCKER,
                    MCPToolCategory.KUBERNETES,
                    MCPToolCategory.DATADOG,
                ],
                priority_tools=["create_repository", "create_pull_request"],
            ),
        }

    def _register_tools(self, server_name: str, tools: List[MCPTool]) -> None:
        """Register tools from a server in the tool registry."""
        self.tool_registry[server_name] = tools
        logger.debug("Registered %d tools for server %s", len(tools), server_name)

    def _get_tools_by_category(self, category: MCPToolCategory) -> List[MCPTool]:
        """Get all tools belonging to a specific category."""
        tools = []
        for server_tools in self.tool_registry.values():
            tools.extend([tool for tool in server_tools if tool.category == category])
        return tools

    async def _load_server_configs(self) -> List[MCPServerConfig]:
        """Load MCP server configurations from file or environment."""
        configs = []

        # Try to load from config file first
        if self.config_path:
            try:
                config_file = Path(self.config_path)
                if config_file.exists():
                    with open(config_file, "r") as f:
                        config_data = json.load(f)
                    configs = self._parse_config_data(config_data)
                    logger.info("Loaded MCP config from file: %s", self.config_path)
                    return configs
            except Exception as e:
                logger.error(
                    "Failed to load config from file %s: %s", self.config_path, e
                )

        # Fallback to default configurations
        configs = self._get_default_server_configs()
        logger.info("Using default MCP server configurations")
        return configs

    def _parse_config_data(self, config_data: Dict[str, Any]) -> List[MCPServerConfig]:
        """Parse configuration data into MCPServerConfig objects."""
        configs = []

        mcp_servers = config_data.get("mcpServers", {})
        for server_name, server_config in mcp_servers.items():
            try:
                config = MCPServerConfig(
                    name=server_name,
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    timeout=server_config.get("timeout", 30),
                    max_retries=server_config.get("max_retries", 3),
                    disabled=server_config.get("disabled", False),
                    auto_approve=server_config.get("autoApprove", []),
                    health_check_interval=server_config.get(
                        "health_check_interval", 60
                    ),
                )
                configs.append(config)
            except Exception as e:
                logger.error("Failed to parse config for server %s: %s", server_name, e)

        return configs

    def _get_default_server_configs(self) -> List[MCPServerConfig]:
        """Get default MCP server configurations."""
        return [
            MCPServerConfig(
                name="github-mcp",
                command="uvx",
                args=["composio-github-mcp@latest"],
                env={"GITHUB_TOKEN": ""},
                timeout=30,
                max_retries=3,
                disabled=False,
            ),
            MCPServerConfig(
                name="linear-mcp",
                command="uvx",
                args=["composio-linear-mcp@latest"],
                env={"LINEAR_API_KEY": ""},
                timeout=30,
                max_retries=3,
                disabled=False,
            ),
            MCPServerConfig(
                name="slack-mcp",
                command="uvx",
                args=["composio-slack-mcp@latest"],
                env={"SLACK_BOT_TOKEN": ""},
                timeout=30,
                max_retries=3,
                disabled=False,
            ),
            MCPServerConfig(
                name="notion-mcp",
                command="uvx",
                args=["composio-notion-mcp@latest"],
                env={"NOTION_TOKEN": ""},
                timeout=30,
                max_retries=3,
                disabled=False,
            ),
            MCPServerConfig(
                name="figma-mcp",
                command="uvx",
                args=["composio-figma-mcp@latest"],
                env={"FIGMA_TOKEN": ""},
                timeout=30,
                max_retries=3,
                disabled=False,
            ),
        ]


# Global MCP client instance
_mcp_client: Optional[ComposioMCPClient] = None


async def get_mcp_client(config_path: Optional[str] = None) -> ComposioMCPClient:
    """Get the global MCP client instance."""
    global _mcp_client

    if _mcp_client is None:
        _mcp_client = ComposioMCPClient(config_path)
        await _mcp_client.initialize()

    return _mcp_client


async def shutdown_mcp_client() -> None:
    """Shutdown the global MCP client instance."""
    global _mcp_client

    if _mcp_client:
        await _mcp_client.shutdown()
        _mcp_client = None
