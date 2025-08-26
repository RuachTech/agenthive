"""Model Context Protocol (MCP) integration for Composio tools."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPToolCategory(Enum):
    """Categories of MCP tools available through Composio."""

    GITHUB = "github"
    LINEAR = "linear"
    SLACK = "slack"
    NOTION = "notion"
    FIGMA = "figma"
    AWS = "aws"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DATADOG = "datadog"
    BROWSERBASE = "browserbase"
    SERPAPI = "serpapi"
    MIRO = "miro"
    AIRTABLE = "airtable"


@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata and execution details."""

    name: str
    category: MCPToolCategory
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    server_name: str
    timeout: int = 30
    retry_count: int = 3
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    disabled: bool = False
    auto_approve: List[str] = field(default_factory=list)
    health_check_interval: int = 60  # seconds


@dataclass
class MCPExecutionResult:
    """Result of MCP tool execution."""

    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    server_name: str = ""
    tool_name: str = ""
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self, message: str, server_name: str = "", tool_name: str = ""):
        self.server_name = server_name
        self.tool_name = tool_name
        super().__init__(message)


class MCPServerUnavailableError(MCPError):
    """Raised when MCP server is unavailable."""

    pass


class MCPToolNotFoundError(MCPError):
    """Raised when requested MCP tool is not found."""

    pass


class MCPExecutionTimeoutError(MCPError):
    """Raised when MCP tool execution times out."""

    pass


class MCPServerInterface(ABC):
    """Abstract interface for MCP server communication."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the server is healthy and responsive."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List all available tools from this server."""
        pass

    @abstractmethod
    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> MCPExecutionResult:
        """Execute a tool with given parameters."""
        pass


class ComposioMCPServer(MCPServerInterface):
    """Composio MCP server implementation."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.connected = False
        self.last_health_check = datetime.min
        self._available_tools: Dict[str, MCPTool] = {}
        self._connection_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to the Composio MCP server."""
        async with self._connection_lock:
            if self.connected:
                return True

            try:
                # Simulate MCP server connection
                # In real implementation, this would establish actual MCP connection
                logger.info("Connecting to MCP server: %s", self.config.name)

                # Mock connection delay
                await asyncio.sleep(0.1)

                self.connected = True
                await self._discover_tools()

                logger.info(
                    "Successfully connected to MCP server: %s", self.config.name
                )
                return True

            except Exception as e:
                logger.error(
                    "Failed to connect to MCP server %s: %s", self.config.name, e
                )
                raise MCPServerUnavailableError(
                    f"Failed to connect to server {self.config.name}: {e}",
                    server_name=self.config.name,
                )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        async with self._connection_lock:
            if not self.connected:
                return

            try:
                logger.info("Disconnecting from MCP server: %s", self.config.name)
                # Simulate disconnection
                await asyncio.sleep(0.05)
                self.connected = False
                self._available_tools.clear()

            except Exception as e:
                logger.error(
                    "Error during MCP server disconnection %s: %s", self.config.name, e
                )

    async def is_healthy(self) -> bool:
        """Check if the server is healthy."""
        if not self.connected:
            return False

        # Check if we need to perform health check
        now = datetime.now()
        if (now - self.last_health_check).seconds < self.config.health_check_interval:
            return True

        try:
            # Simulate health check
            await asyncio.sleep(0.01)
            self.last_health_check = now
            return True

        except Exception as e:
            logger.warning(
                "Health check failed for MCP server %s: %s", self.config.name, e
            )
            return False

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools from this server."""
        if not self.connected:
            await self.connect()

        return list(self._available_tools.values())

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> MCPExecutionResult:
        """Execute a tool with given parameters."""
        if not self.connected:
            await self.connect()

        if tool_name not in self._available_tools:
            raise MCPToolNotFoundError(
                f"Tool '{tool_name}' not found in server {self.config.name}",
                server_name=self.config.name,
                tool_name=tool_name,
            )

        tool = self._available_tools[tool_name]
        start_time = asyncio.get_event_loop().time()

        # Validate required parameters first (before try block)
        missing_params = [
            param for param in tool.required_params if param not in parameters
        ]
        if missing_params:
            raise MCPError(
                f"Missing required parameters: {missing_params}",
                server_name=self.config.name,
                tool_name=tool_name,
            )

        try:
            # Execute tool with timeout
            result = await asyncio.wait_for(
                self._execute_tool_impl(tool, parameters), timeout=tool.timeout
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return MCPExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                server_name=self.config.name,
                tool_name=tool_name,
            )

        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            raise MCPExecutionTimeoutError(
                f"Tool execution timed out after {tool.timeout}s",
                server_name=self.config.name,
                tool_name=tool_name,
            )
        except MCPError:
            # Re-raise MCP errors
            raise
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return MCPExecutionResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                server_name=self.config.name,
                tool_name=tool_name,
            )

    async def _execute_tool_impl(
        self, tool: MCPTool, parameters: Dict[str, Any]
    ) -> Any:
        """Internal tool execution implementation."""
        # Simulate tool execution based on category
        await asyncio.sleep(0.1)  # Simulate processing time

        if tool.category == MCPToolCategory.GITHUB:
            return self._mock_github_response(tool.name, parameters)
        elif tool.category == MCPToolCategory.LINEAR:
            return self._mock_linear_response(tool.name, parameters)
        elif tool.category == MCPToolCategory.SLACK:
            return self._mock_slack_response(tool.name, parameters)
        elif tool.category == MCPToolCategory.NOTION:
            return self._mock_notion_response(tool.name, parameters)
        elif tool.category == MCPToolCategory.FIGMA:
            return self._mock_figma_response(tool.name, parameters)
        else:
            return {
                "status": "success",
                "message": f"Executed {tool.name} with {parameters}",
            }

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        # Mock tool discovery based on server name
        if "github" in self.config.name.lower():
            self._available_tools.update(self._get_github_tools())
        elif "linear" in self.config.name.lower():
            self._available_tools.update(self._get_linear_tools())
        elif "slack" in self.config.name.lower():
            self._available_tools.update(self._get_slack_tools())
        elif "notion" in self.config.name.lower():
            self._available_tools.update(self._get_notion_tools())
        elif "figma" in self.config.name.lower():
            self._available_tools.update(self._get_figma_tools())

    def _get_github_tools(self) -> Dict[str, MCPTool]:
        """Get GitHub-specific tools."""
        return {
            "create_repository": MCPTool(
                name="create_repository",
                category=MCPToolCategory.GITHUB,
                description="Create a new GitHub repository",
                parameters={"name": "str", "description": "str", "private": "bool"},
                required_params=["name"],
                server_name=self.config.name,
            ),
            "create_pull_request": MCPTool(
                name="create_pull_request",
                category=MCPToolCategory.GITHUB,
                description="Create a pull request",
                parameters={
                    "title": "str",
                    "body": "str",
                    "head": "str",
                    "base": "str",
                },
                required_params=["title", "head", "base"],
                server_name=self.config.name,
            ),
            "list_issues": MCPTool(
                name="list_issues",
                category=MCPToolCategory.GITHUB,
                description="List repository issues",
                parameters={"state": "str", "labels": "list"},
                required_params=[],
                server_name=self.config.name,
            ),
        }

    def _get_linear_tools(self) -> Dict[str, MCPTool]:
        """Get Linear-specific tools."""
        return {
            "create_issue": MCPTool(
                name="create_issue",
                category=MCPToolCategory.LINEAR,
                description="Create a new Linear issue",
                parameters={"title": "str", "description": "str", "priority": "int"},
                required_params=["title"],
                server_name=self.config.name,
            ),
            "update_issue": MCPTool(
                name="update_issue",
                category=MCPToolCategory.LINEAR,
                description="Update an existing Linear issue",
                parameters={
                    "issue_id": "str",
                    "title": "str",
                    "description": "str",
                    "state": "str",
                },
                required_params=["issue_id"],
                server_name=self.config.name,
            ),
        }

    def _get_slack_tools(self) -> Dict[str, MCPTool]:
        """Get Slack-specific tools."""
        return {
            "send_message": MCPTool(
                name="send_message",
                category=MCPToolCategory.SLACK,
                description="Send a message to a Slack channel",
                parameters={"channel": "str", "text": "str", "thread_ts": "str"},
                required_params=["channel", "text"],
                server_name=self.config.name,
            ),
            "create_channel": MCPTool(
                name="create_channel",
                category=MCPToolCategory.SLACK,
                description="Create a new Slack channel",
                parameters={"name": "str", "is_private": "bool"},
                required_params=["name"],
                server_name=self.config.name,
            ),
        }

    def _get_notion_tools(self) -> Dict[str, MCPTool]:
        """Get Notion-specific tools."""
        return {
            "create_page": MCPTool(
                name="create_page",
                category=MCPToolCategory.NOTION,
                description="Create a new Notion page",
                parameters={"title": "str", "content": "str", "parent_id": "str"},
                required_params=["title"],
                server_name=self.config.name,
            ),
            "search_pages": MCPTool(
                name="search_pages",
                category=MCPToolCategory.NOTION,
                description="Search Notion pages",
                parameters={"query": "str", "filter": "dict"},
                required_params=["query"],
                server_name=self.config.name,
            ),
        }

    def _get_figma_tools(self) -> Dict[str, MCPTool]:
        """Get Figma-specific tools."""
        return {
            "get_file": MCPTool(
                name="get_file",
                category=MCPToolCategory.FIGMA,
                description="Get Figma file information",
                parameters={"file_key": "str"},
                required_params=["file_key"],
                server_name=self.config.name,
            ),
            "export_image": MCPTool(
                name="export_image",
                category=MCPToolCategory.FIGMA,
                description="Export Figma node as image",
                parameters={"file_key": "str", "node_id": "str", "format": "str"},
                required_params=["file_key", "node_id"],
                server_name=self.config.name,
            ),
        }

    def _mock_github_response(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock GitHub API response."""
        if tool_name == "create_repository":
            return {
                "id": 12345,
                "name": parameters.get("name"),
                "full_name": f"user/{parameters.get('name')}",
                "html_url": f"https://github.com/user/{parameters.get('name')}",
                "created_at": datetime.now().isoformat(),
            }
        elif tool_name == "create_pull_request":
            return {
                "id": 67890,
                "number": 1,
                "title": parameters.get("title"),
                "html_url": "https://github.com/user/repo/pull/1",
                "state": "open",
            }
        elif tool_name == "list_issues":
            return [
                {
                    "id": 1,
                    "number": 1,
                    "title": "Sample Issue",
                    "state": "open",
                    "html_url": "https://github.com/user/repo/issues/1",
                }
            ]
        return {"status": "success"}

    def _mock_linear_response(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock Linear API response."""
        if tool_name == "create_issue":
            return {
                "id": "issue-123",
                "title": parameters.get("title"),
                "identifier": "ENG-123",
                "url": "https://linear.app/team/issue/ENG-123",
            }
        elif tool_name == "update_issue":
            return {
                "id": parameters.get("issue_id"),
                "title": parameters.get("title", "Updated Issue"),
                "state": parameters.get("state", "in_progress"),
            }
        return {"status": "success"}

    def _mock_slack_response(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock Slack API response."""
        if tool_name == "send_message":
            return {
                "ok": True,
                "channel": parameters.get("channel"),
                "ts": "1234567890.123456",
                "message": {"text": parameters.get("text"), "user": "U1234567890"},
            }
        elif tool_name == "create_channel":
            return {
                "ok": True,
                "channel": {
                    "id": "C1234567890",
                    "name": parameters.get("name"),
                    "is_private": parameters.get("is_private", False),
                },
            }
        return {"ok": True}

    def _mock_notion_response(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock Notion API response."""
        if tool_name == "create_page":
            return {
                "id": "page-123",
                "title": parameters.get("title"),
                "url": "https://notion.so/page-123",
                "created_time": datetime.now().isoformat(),
            }
        elif tool_name == "search_pages":
            return {
                "results": [
                    {
                        "id": "page-456",
                        "title": "Search Result",
                        "url": "https://notion.so/page-456",
                    }
                ]
            }
        return {"status": "success"}

    def _mock_figma_response(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock Figma API response."""
        if tool_name == "get_file":
            return {
                "name": "Design File",
                "lastModified": datetime.now().isoformat(),
                "thumbnailUrl": "https://figma.com/thumbnail.png",
                "version": "1.0",
            }
        elif tool_name == "export_image":
            return {
                "images": {
                    parameters.get("node_id"): "https://figma.com/exported-image.png"
                }
            }
        return {"status": "success"}
