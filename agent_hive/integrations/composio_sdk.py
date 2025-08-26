"""Direct Composio SDK integration for AgentHive."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    from composio import Composio
    from composio.types import ExecuteRequestFn
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False
    Composio = None
    ExecuteRequestFn = None

logger = logging.getLogger(__name__)


class ComposioToolkit(Enum):
    """Supported Composio toolkits."""
    GITHUB = "github"
    LINEAR = "linear"
    SLACK = "slack"
    NOTION = "notion"
    GMAIL = "gmail"
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
class ComposioConnection:
    """Represents a Composio connection for a user."""
    user_id: str
    toolkit: str
    connection_id: str
    status: str
    redirect_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ComposioTool:
    """Represents a Composio tool."""
    name: str
    slug: str
    toolkit: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ComposioExecutionResult:
    """Result of Composio tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_slug: str = ""
    user_id: str = ""


class ComposioIntegrationError(Exception):
    """Base exception for Composio integration errors."""
    pass


class ComposioNotAvailableError(ComposioIntegrationError):
    """Raised when Composio SDK is not available."""
    pass


class ComposioAuthenticationError(ComposioIntegrationError):
    """Raised when authentication fails."""
    pass


class ComposioToolExecutionError(ComposioIntegrationError):
    """Raised when tool execution fails."""
    pass


class ComposioSDKClient:
    """Client for direct Composio SDK integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not COMPOSIO_AVAILABLE:
            raise ComposioNotAvailableError(
                "Composio SDK not available. Install with: pip install composio"
            )
        
        self.api_key = api_key
        self.composio = Composio(api_key=api_key) if api_key else Composio()
        self.connections: Dict[str, List[ComposioConnection]] = {}
        self._tools_cache: Dict[str, List[ComposioTool]] = {}
        
        logger.info("ComposioSDKClient initialized")
    
    async def authorize_toolkit(
        self, 
        user_id: str, 
        toolkit: Union[str, ComposioToolkit],
        redirect_url: Optional[str] = None
    ) -> ComposioConnection:
        """Authorize a toolkit for a user."""
        if isinstance(toolkit, ComposioToolkit):
            toolkit = toolkit.value
        
        try:
            # Initialize connection request
            connection_request = self.composio.toolkits.authorize(
                user_id=user_id,
                toolkit=toolkit
            )
            
            connection = ComposioConnection(
                user_id=user_id,
                toolkit=toolkit,
                connection_id=connection_request.connection_id,
                status="pending",
                redirect_url=connection_request.redirect_url
            )
            
            # Store connection
            if user_id not in self.connections:
                self.connections[user_id] = []
            self.connections[user_id].append(connection)
            
            logger.info("Authorization initiated for user %s, toolkit %s", user_id, toolkit)
            return connection
            
        except Exception as e:
            logger.error("Failed to authorize toolkit %s for user %s: %s", toolkit, user_id, e)
            raise ComposioAuthenticationError(f"Authorization failed: {e}")
    
    async def wait_for_connection(
        self, 
        connection: ComposioConnection,
        timeout: int = 300
    ) -> bool:
        """Wait for a connection to be established."""
        try:
            # In real implementation, this would use the connection_request.wait_for_connection()
            # For now, we'll simulate the wait
            await asyncio.sleep(1)  # Simulate connection time
            
            connection.status = "active"
            logger.info("Connection established for user %s, toolkit %s", 
                       connection.user_id, connection.toolkit)
            return True
            
        except Exception as e:
            logger.error("Failed to establish connection: %s", e)
            connection.status = "failed"
            return False
    
    async def get_tools(
        self,
        user_id: str,
        toolkits: Optional[List[str]] = None,
        tools: Optional[List[str]] = None
    ) -> List[ComposioTool]:
        """Get available tools for a user."""
        cache_key = f"{user_id}:{','.join(toolkits or [])}:{','.join(tools or [])}"
        
        if cache_key in self._tools_cache:
            return self._tools_cache[cache_key]
        
        try:
            # Get tools from Composio
            composio_tools = self.composio.tools.get(
                user_id=user_id,
                toolkits=toolkits,
                tools=tools
            )
            
            # Convert to our format
            tools_list = []
            for tool in composio_tools:
                composio_tool = ComposioTool(
                    name=tool.get('name', ''),
                    slug=tool.get('slug', ''),
                    toolkit=tool.get('toolkit', ''),
                    description=tool.get('description', ''),
                    parameters=tool.get('parameters', {}),
                    required_params=tool.get('required_params', []),
                    metadata=tool.get('metadata', {})
                )
                tools_list.append(composio_tool)
            
            # Cache the results
            self._tools_cache[cache_key] = tools_list
            
            logger.info("Retrieved %d tools for user %s", len(tools_list), user_id)
            return tools_list
            
        except Exception as e:
            logger.error("Failed to get tools for user %s: %s", user_id, e)
            raise ComposioToolExecutionError(f"Failed to get tools: {e}")
    
    async def execute_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any],
        connected_account_id: Optional[str] = None
    ) -> ComposioExecutionResult:
        """Execute a Composio tool."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute tool using Composio SDK
            result = self.composio.tools.execute(
                slug=tool_slug,
                user_id=user_id,
                arguments=arguments,
                connected_account_id=connected_account_id
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ComposioExecutionResult(
                success=True,
                data=result,
                execution_time=execution_time,
                tool_slug=tool_slug,
                user_id=user_id
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error("Tool execution failed for %s: %s", tool_slug, e)
            
            return ComposioExecutionResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
                tool_slug=tool_slug,
                user_id=user_id
            )
    
    async def get_user_connections(self, user_id: str) -> List[ComposioConnection]:
        """Get all connections for a user."""
        return self.connections.get(user_id, [])
    
    async def get_active_connections(self, user_id: str) -> List[ComposioConnection]:
        """Get active connections for a user."""
        user_connections = self.connections.get(user_id, [])
        return [conn for conn in user_connections if conn.status == "active"]
    
    def create_custom_tool(
        self,
        slug: str,
        name: str,
        description: str,
        input_params: Dict[str, Any],
        execute_fn: callable,
        toolkit: Optional[str] = None
    ) -> str:
        """Create a custom Composio tool."""
        try:
            if toolkit:
                # Toolkit-based custom tool
                @self.composio.tools.custom_tool(toolkit=toolkit)
                def custom_tool_impl(request, execute_request: ExecuteRequestFn, auth_credentials: dict):
                    return execute_fn(request, execute_request, auth_credentials)
            else:
                # Standalone custom tool
                @self.composio.tools.custom_tool
                def custom_tool_impl(request):
                    return execute_fn(request)
            
            # Set tool metadata
            custom_tool_impl.slug = slug
            custom_tool_impl.name = name
            custom_tool_impl.description = description
            
            logger.info("Created custom tool: %s", slug)
            return slug
            
        except Exception as e:
            logger.error("Failed to create custom tool %s: %s", slug, e)
            raise ComposioToolExecutionError(f"Failed to create custom tool: {e}")
    
    async def get_toolkit_info(self, toolkit: str) -> Dict[str, Any]:
        """Get information about a specific toolkit."""
        try:
            # This would use the actual Composio API to get toolkit info
            # For now, return mock data
            return {
                "name": toolkit,
                "description": f"Integration with {toolkit}",
                "tools_count": 10,
                "auth_type": "oauth2",
                "supported": True
            }
        except Exception as e:
            logger.error("Failed to get toolkit info for %s: %s", toolkit, e)
            return {"error": str(e)}


class ComposioAgentIntegration:
    """Integration layer between AgentHive agents and Composio."""
    
    def __init__(self, composio_client: ComposioSDKClient):
        self.composio = composio_client
        self.agent_toolkit_mappings = self._setup_agent_mappings()
    
    def _setup_agent_mappings(self) -> Dict[str, List[str]]:
        """Setup default agent to toolkit mappings."""
        return {
            "full_stack_engineer": ["github", "linear", "slack", "notion"],
            "qa_engineer": ["github", "linear", "slack", "browserbase"],
            "product_designer": ["figma", "notion", "slack", "miro"],
            "devops_engineer": ["github", "aws", "docker", "kubernetes", "datadog"],
            "data_scientist": ["github", "notion", "airtable", "serpapi"],
            "marketing_specialist": ["slack", "notion", "airtable", "serpapi"],
            "project_manager": ["linear", "slack", "notion", "airtable"]
        }
    
    async def setup_agent_tools(
        self,
        user_id: str,
        agent_type: str,
        auto_authorize: bool = False
    ) -> Dict[str, Any]:
        """Setup tools for a specific agent type."""
        if agent_type not in self.agent_toolkit_mappings:
            raise ComposioIntegrationError(f"Unknown agent type: {agent_type}")
        
        toolkits = self.agent_toolkit_mappings[agent_type]
        setup_results = {
            "agent_type": agent_type,
            "toolkits": toolkits,
            "connections": [],
            "tools": [],
            "authorization_urls": []
        }
        
        # Authorize toolkits
        for toolkit in toolkits:
            try:
                connection = await self.composio.authorize_toolkit(user_id, toolkit)
                setup_results["connections"].append({
                    "toolkit": toolkit,
                    "status": connection.status,
                    "connection_id": connection.connection_id
                })
                
                if connection.redirect_url:
                    setup_results["authorization_urls"].append({
                        "toolkit": toolkit,
                        "url": connection.redirect_url
                    })
                
                # If auto_authorize is True, wait for connection
                if auto_authorize:
                    await self.composio.wait_for_connection(connection)
                
            except Exception as e:
                logger.error("Failed to setup toolkit %s for agent %s: %s", 
                           toolkit, agent_type, e)
                setup_results["connections"].append({
                    "toolkit": toolkit,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Get available tools
        try:
            tools = await self.composio.get_tools(user_id, toolkits=toolkits)
            setup_results["tools"] = [
                {
                    "name": tool.name,
                    "slug": tool.slug,
                    "toolkit": tool.toolkit,
                    "description": tool.description
                }
                for tool in tools
            ]
        except Exception as e:
            logger.error("Failed to get tools for agent %s: %s", agent_type, e)
            setup_results["tools_error"] = str(e)
        
        return setup_results
    
    async def execute_agent_tool(
        self,
        user_id: str,
        agent_type: str,
        tool_slug: str,
        arguments: Dict[str, Any]
    ) -> ComposioExecutionResult:
        """Execute a tool on behalf of an agent."""
        logger.info("Executing tool %s for agent %s", tool_slug, agent_type)
        
        return await self.composio.execute_tool(
            user_id=user_id,
            tool_slug=tool_slug,
            arguments=arguments
        )
    
    async def get_agent_status(self, user_id: str, agent_type: str) -> Dict[str, Any]:
        """Get the status of an agent's tool integrations."""
        if agent_type not in self.agent_toolkit_mappings:
            return {"error": f"Unknown agent type: {agent_type}"}
        
        toolkits = self.agent_toolkit_mappings[agent_type]
        connections = await self.composio.get_user_connections(user_id)
        
        status = {
            "agent_type": agent_type,
            "required_toolkits": toolkits,
            "toolkit_status": {},
            "ready": True
        }
        
        for toolkit in toolkits:
            toolkit_connections = [
                conn for conn in connections 
                if conn.toolkit == toolkit
            ]
            
            if toolkit_connections:
                latest_connection = max(toolkit_connections, key=lambda x: x.connection_id)
                status["toolkit_status"][toolkit] = {
                    "connected": latest_connection.status == "active",
                    "status": latest_connection.status,
                    "connection_id": latest_connection.connection_id
                }
                
                if latest_connection.status != "active":
                    status["ready"] = False
            else:
                status["toolkit_status"][toolkit] = {
                    "connected": False,
                    "status": "not_authorized"
                }
                status["ready"] = False
        
        return status


# Global Composio client instance
_composio_client: Optional[ComposioSDKClient] = None


def get_composio_client(api_key: Optional[str] = None) -> ComposioSDKClient:
    """Get the global Composio client instance."""
    global _composio_client
    
    if _composio_client is None:
        _composio_client = ComposioSDKClient(api_key)
    
    return _composio_client


def get_agent_integration(api_key: Optional[str] = None) -> ComposioAgentIntegration:
    """Get the Composio agent integration instance."""
    client = get_composio_client(api_key)
    return ComposioAgentIntegration(client)