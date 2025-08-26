"""Unified integration client that combines MCP and direct Composio SDK integration."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .composio_client import ComposioMCPClient
from .mcp import MCPTool, MCPExecutionResult, MCPError
from .composio_sdk import (
    ComposioSDKClient, ComposioAgentIntegration, ComposioTool, 
    ComposioExecutionResult, ComposioIntegrationError, COMPOSIO_AVAILABLE
)

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for tool execution."""
    MCP_ONLY = "mcp_only"
    COMPOSIO_ONLY = "composio_only"
    HYBRID = "hybrid"  # Prefer Composio, fallback to MCP
    AUTO = "auto"  # Automatically choose based on availability


@dataclass
class UnifiedTool:
    """Unified tool representation."""
    name: str
    slug: str
    category: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    source: str  # "mcp" or "composio"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedExecutionResult:
    """Unified execution result."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_slug: str = ""
    user_id: str = ""
    source: str = ""  # "mcp" or "composio"
    metadata: Optional[Dict[str, Any]] = None


class UnifiedIntegrationClient:
    """Unified client that combines MCP and Composio SDK integration."""
    
    def __init__(
        self,
        mode: IntegrationMode = IntegrationMode.AUTO,
        composio_api_key: Optional[str] = None,
        mcp_config_path: Optional[str] = None
    ):
        self.mode = mode
        self.mcp_client: Optional[ComposioMCPClient] = None
        self.composio_client: Optional[ComposioSDKClient] = None
        self.composio_integration: Optional[ComposioAgentIntegration] = None
        
        # Initialize based on mode and availability
        self._initialize_clients(composio_api_key, mcp_config_path)
        
        logger.info("UnifiedIntegrationClient initialized in %s mode", mode.value)
    
    def _initialize_clients(
        self, 
        composio_api_key: Optional[str], 
        mcp_config_path: Optional[str]
    ) -> None:
        """Initialize the appropriate clients based on mode and availability."""
        
        # Initialize Composio SDK client if available and needed
        if (self.mode in [IntegrationMode.COMPOSIO_ONLY, IntegrationMode.HYBRID, IntegrationMode.AUTO] 
            and COMPOSIO_AVAILABLE):
            try:
                self.composio_client = ComposioSDKClient(composio_api_key)
                self.composio_integration = ComposioAgentIntegration(self.composio_client)
                logger.info("Composio SDK client initialized")
            except Exception as e:
                logger.warning("Failed to initialize Composio SDK client: %s", e)
                if self.mode == IntegrationMode.COMPOSIO_ONLY:
                    raise
        
        # Initialize MCP client if needed
        if self.mode in [IntegrationMode.MCP_ONLY, IntegrationMode.HYBRID, IntegrationMode.AUTO]:
            try:
                self.mcp_client = ComposioMCPClient(mcp_config_path)
                logger.info("MCP client initialized")
            except Exception as e:
                logger.warning("Failed to initialize MCP client: %s", e)
                if self.mode == IntegrationMode.MCP_ONLY:
                    raise
        
        # Validate that at least one client is available
        if not self.mcp_client and not self.composio_client:
            raise RuntimeError("No integration clients available")
    
    async def initialize(self) -> None:
        """Initialize all clients."""
        tasks = []
        
        if self.mcp_client:
            tasks.append(self.mcp_client.initialize())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("UnifiedIntegrationClient initialization complete")
    
    async def shutdown(self) -> None:
        """Shutdown all clients."""
        tasks = []
        
        if self.mcp_client:
            tasks.append(self.mcp_client.shutdown())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("UnifiedIntegrationClient shutdown complete")
    
    async def authorize_toolkit(
        self,
        user_id: str,
        toolkit: str,
        redirect_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Authorize a toolkit for a user (Composio SDK only)."""
        if not self.composio_client:
            raise ComposioIntegrationError("Composio SDK not available for authorization")
        
        connection = await self.composio_client.authorize_toolkit(
            user_id, toolkit, redirect_url
        )
        
        return {
            "user_id": connection.user_id,
            "toolkit": connection.toolkit,
            "connection_id": connection.connection_id,
            "status": connection.status,
            "redirect_url": connection.redirect_url
        }
    
    async def get_tools_for_agent(
        self,
        user_id: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None
    ) -> List[UnifiedTool]:
        """Get tools for a specific agent type from all available sources."""
        all_tools = []
        
        # Get tools from MCP if available
        if self.mcp_client:
            try:
                mcp_tools = await self.mcp_client.get_tools_for_agent(agent_type, capabilities)
                for tool in mcp_tools:
                    unified_tool = UnifiedTool(
                        name=tool.name,
                        slug=tool.name,
                        category=tool.category.value,
                        description=tool.description,
                        parameters=tool.parameters,
                        required_params=tool.required_params,
                        source="mcp",
                        metadata=tool.metadata
                    )
                    all_tools.append(unified_tool)
            except Exception as e:
                logger.warning("Failed to get MCP tools: %s", e)
        
        # Get tools from Composio SDK if available
        if self.composio_integration:
            try:
                setup_result = await self.composio_integration.setup_agent_tools(
                    user_id, agent_type, auto_authorize=False
                )
                
                for tool_info in setup_result.get("tools", []):
                    unified_tool = UnifiedTool(
                        name=tool_info["name"],
                        slug=tool_info["slug"],
                        category=tool_info["toolkit"],
                        description=tool_info["description"],
                        parameters={},  # Would need to fetch detailed tool info
                        required_params=[],
                        source="composio",
                        metadata={"toolkit": tool_info["toolkit"]}
                    )
                    all_tools.append(unified_tool)
            except Exception as e:
                logger.warning("Failed to get Composio tools: %s", e)
        
        logger.info("Retrieved %d unified tools for agent %s", len(all_tools), agent_type)
        return all_tools
    
    async def execute_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any],
        preferred_source: Optional[str] = None
    ) -> UnifiedExecutionResult:
        """Execute a tool using the best available source."""
        
        # Determine execution source
        source = self._determine_execution_source(tool_slug, preferred_source)
        
        if source == "composio" and self.composio_client:
            return await self._execute_composio_tool(user_id, tool_slug, arguments)
        elif source == "mcp" and self.mcp_client:
            return await self._execute_mcp_tool(user_id, tool_slug, arguments)
        else:
            # Fallback logic
            if self.composio_client:
                return await self._execute_composio_tool(user_id, tool_slug, arguments)
            elif self.mcp_client:
                return await self._execute_mcp_tool(user_id, tool_slug, arguments)
            else:
                raise RuntimeError("No execution clients available")
    
    async def _execute_composio_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any]
    ) -> UnifiedExecutionResult:
        """Execute tool using Composio SDK."""
        try:
            result = await self.composio_client.execute_tool(user_id, tool_slug, arguments)
            
            return UnifiedExecutionResult(
                success=result.success,
                data=result.data,
                error=result.error,
                execution_time=result.execution_time,
                tool_slug=result.tool_slug,
                user_id=result.user_id,
                source="composio"
            )
        except Exception as e:
            logger.error("Composio tool execution failed: %s", e)
            return UnifiedExecutionResult(
                success=False,
                data=None,
                error=str(e),
                tool_slug=tool_slug,
                user_id=user_id,
                source="composio"
            )
    
    async def _execute_mcp_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any]
    ) -> UnifiedExecutionResult:
        """Execute tool using MCP."""
        try:
            result = await self.mcp_client.execute_tool(tool_slug, arguments)
            
            return UnifiedExecutionResult(
                success=result.success,
                data=result.result,
                error=result.error,
                execution_time=result.execution_time,
                tool_slug=result.tool_name,
                user_id=user_id,
                source="mcp",
                metadata=result.metadata
            )
        except Exception as e:
            logger.error("MCP tool execution failed: %s", e)
            return UnifiedExecutionResult(
                success=False,
                data=None,
                error=str(e),
                tool_slug=tool_slug,
                user_id=user_id,
                source="mcp"
            )
    
    def _determine_execution_source(
        self,
        tool_slug: str,
        preferred_source: Optional[str]
    ) -> str:
        """Determine which source to use for tool execution."""
        
        if preferred_source in ["composio", "mcp"]:
            return preferred_source
        
        # Auto-determination logic based on mode
        if self.mode == IntegrationMode.COMPOSIO_ONLY:
            return "composio"
        elif self.mode == IntegrationMode.MCP_ONLY:
            return "mcp"
        elif self.mode == IntegrationMode.HYBRID:
            # Prefer Composio for real integrations, MCP for development/testing
            return "composio" if self.composio_client else "mcp"
        else:  # AUTO mode
            # Choose based on availability and tool characteristics
            if self.composio_client and COMPOSIO_AVAILABLE:
                return "composio"
            elif self.mcp_client:
                return "mcp"
            else:
                raise RuntimeError("No execution sources available")
    
    async def get_integration_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of all integrations."""
        status = {
            "mode": self.mode.value,
            "mcp_available": self.mcp_client is not None,
            "composio_available": self.composio_client is not None,
            "mcp_status": {},
            "composio_status": {}
        }
        
        # Get MCP status
        if self.mcp_client:
            try:
                mcp_status = await self.mcp_client.get_server_status()
                status["mcp_status"] = mcp_status
            except Exception as e:
                status["mcp_status"] = {"error": str(e)}
        
        # Get Composio status
        if self.composio_client:
            try:
                connections = await self.composio_client.get_user_connections(user_id)
                status["composio_status"] = {
                    "connections": len(connections),
                    "active_connections": len([c for c in connections if c.status == "active"])
                }
            except Exception as e:
                status["composio_status"] = {"error": str(e)}
        
        return status
    
    async def setup_agent_integrations(
        self,
        user_id: str,
        agent_type: str,
        auto_authorize: bool = False
    ) -> Dict[str, Any]:
        """Setup all integrations for an agent."""
        results = {
            "agent_type": agent_type,
            "user_id": user_id,
            "mcp_setup": {},
            "composio_setup": {}
        }
        
        # Setup MCP (no authorization needed)
        if self.mcp_client:
            try:
                tools = await self.mcp_client.get_tools_for_agent(agent_type)
                results["mcp_setup"] = {
                    "success": True,
                    "tools_count": len(tools),
                    "categories": list(set(tool.category.value for tool in tools))
                }
            except Exception as e:
                results["mcp_setup"] = {"success": False, "error": str(e)}
        
        # Setup Composio (requires authorization)
        if self.composio_integration:
            try:
                setup_result = await self.composio_integration.setup_agent_tools(
                    user_id, agent_type, auto_authorize
                )
                results["composio_setup"] = setup_result
            except Exception as e:
                results["composio_setup"] = {"success": False, "error": str(e)}
        
        return results


# Global unified client instance
_unified_client: Optional[UnifiedIntegrationClient] = None


async def get_unified_client(
    mode: IntegrationMode = IntegrationMode.AUTO,
    composio_api_key: Optional[str] = None,
    mcp_config_path: Optional[str] = None
) -> UnifiedIntegrationClient:
    """Get the global unified integration client."""
    global _unified_client
    
    if _unified_client is None:
        _unified_client = UnifiedIntegrationClient(mode, composio_api_key, mcp_config_path)
        await _unified_client.initialize()
    
    return _unified_client


async def shutdown_unified_client() -> None:
    """Shutdown the global unified client."""
    global _unified_client
    
    if _unified_client:
        await _unified_client.shutdown()
        _unified_client = None