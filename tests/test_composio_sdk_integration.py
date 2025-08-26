"""Tests for Composio SDK integration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from agent_hive.integrations import (
    ComposioSDKClient, ComposioAgentIntegration, ComposioTool, ComposioConnection,
    ComposioExecutionResult, ComposioIntegrationError, ComposioNotAvailableError,
    ComposioAuthenticationError, COMPOSIO_AVAILABLE
)


class TestComposioSDKClient:
    """Test ComposioSDKClient functionality."""
    
    def test_composio_not_available(self):
        """Test behavior when Composio SDK is not available."""
        with patch('agent_hive.integrations.composio_sdk.COMPOSIO_AVAILABLE', False):
            with pytest.raises(ComposioNotAvailableError):
                ComposioSDKClient()
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    def test_client_initialization(self):
        """Test client initialization."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            client = ComposioSDKClient("test-api-key")
            
            assert client.api_key == "test-api-key"
            mock_composio.assert_called_once_with(api_key="test-api-key")
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    @pytest.mark.asyncio
    async def test_authorize_toolkit(self):
        """Test toolkit authorization."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            # Setup mock
            mock_connection_request = Mock()
            mock_connection_request.connection_id = "conn_123"
            mock_connection_request.redirect_url = "https://auth.example.com"
            
            mock_composio_instance = Mock()
            mock_composio_instance.toolkits.authorize.return_value = mock_connection_request
            mock_composio.return_value = mock_composio_instance
            
            client = ComposioSDKClient("test-api-key")
            
            # Test authorization
            connection = await client.authorize_toolkit("user123", "github")
            
            assert connection.user_id == "user123"
            assert connection.toolkit == "github"
            assert connection.connection_id == "conn_123"
            assert connection.status == "pending"
            assert connection.redirect_url == "https://auth.example.com"
            
            mock_composio_instance.toolkits.authorize.assert_called_once_with(
                user_id="user123", toolkit="github"
            )
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    @pytest.mark.asyncio
    async def test_get_tools(self):
        """Test getting tools."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            # Setup mock
            mock_tools = [
                {
                    "name": "Create Repository",
                    "slug": "GITHUB_CREATE_REPOSITORY",
                    "toolkit": "github",
                    "description": "Create a new repository",
                    "parameters": {"name": "string"},
                    "required_params": ["name"],
                    "metadata": {}
                }
            ]
            
            mock_composio_instance = Mock()
            mock_composio_instance.tools.get.return_value = mock_tools
            mock_composio.return_value = mock_composio_instance
            
            client = ComposioSDKClient("test-api-key")
            
            # Test getting tools
            tools = await client.get_tools("user123", toolkits=["github"])
            
            assert len(tools) == 1
            assert isinstance(tools[0], ComposioTool)
            assert tools[0].name == "Create Repository"
            assert tools[0].slug == "GITHUB_CREATE_REPOSITORY"
            assert tools[0].toolkit == "github"
            
            mock_composio_instance.tools.get.assert_called_once_with(
                user_id="user123", toolkits=["github"], tools=None
            )
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            # Setup mock
            mock_result = {"id": 123, "name": "test-repo", "url": "https://github.com/user/test-repo"}
            
            mock_composio_instance = Mock()
            mock_composio_instance.tools.execute.return_value = mock_result
            mock_composio.return_value = mock_composio_instance
            
            client = ComposioSDKClient("test-api-key")
            
            # Test tool execution
            result = await client.execute_tool(
                "user123", 
                "GITHUB_CREATE_REPOSITORY",
                {"name": "test-repo"}
            )
            
            assert isinstance(result, ComposioExecutionResult)
            assert result.success is True
            assert result.data == mock_result
            assert result.tool_slug == "GITHUB_CREATE_REPOSITORY"
            assert result.user_id == "user123"
            
            mock_composio_instance.tools.execute.assert_called_once_with(
                slug="GITHUB_CREATE_REPOSITORY",
                user_id="user123",
                arguments={"name": "test-repo"},
                connected_account_id=None
            )
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        """Test tool execution failure."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            # Setup mock to raise exception
            mock_composio_instance = Mock()
            mock_composio_instance.tools.execute.side_effect = Exception("API Error")
            mock_composio.return_value = mock_composio_instance
            
            client = ComposioSDKClient("test-api-key")
            
            # Test tool execution failure
            result = await client.execute_tool(
                "user123",
                "GITHUB_CREATE_REPOSITORY", 
                {"name": "test-repo"}
            )
            
            assert isinstance(result, ComposioExecutionResult)
            assert result.success is False
            assert result.error == "API Error"
            assert result.data is None


class TestComposioAgentIntegration:
    """Test ComposioAgentIntegration functionality."""
    
    @pytest.fixture
    def mock_composio_client(self):
        """Create a mock Composio client."""
        client = Mock(spec=ComposioSDKClient)
        client.authorize_toolkit = AsyncMock()
        client.wait_for_connection = AsyncMock()
        client.get_tools = AsyncMock()
        client.execute_tool = AsyncMock()
        client.get_user_connections = AsyncMock()
        return client
    
    def test_agent_mappings_setup(self, mock_composio_client):
        """Test agent mappings are properly set up."""
        integration = ComposioAgentIntegration(mock_composio_client)
        
        assert "full_stack_engineer" in integration.agent_toolkit_mappings
        assert "qa_engineer" in integration.agent_toolkit_mappings
        assert "product_designer" in integration.agent_toolkit_mappings
        
        # Check specific mappings
        fs_toolkits = integration.agent_toolkit_mappings["full_stack_engineer"]
        assert "github" in fs_toolkits
        assert "linear" in fs_toolkits
    
    @pytest.mark.asyncio
    async def test_setup_agent_tools(self, mock_composio_client):
        """Test setting up tools for an agent."""
        # Setup mocks
        mock_connection = ComposioConnection(
            user_id="user123",
            toolkit="github",
            connection_id="conn_123",
            status="pending",
            redirect_url="https://auth.example.com"
        )
        mock_composio_client.authorize_toolkit.return_value = mock_connection
        mock_composio_client.get_tools.return_value = [
            ComposioTool(
                name="Create Repository",
                slug="GITHUB_CREATE_REPOSITORY",
                toolkit="github",
                description="Create a repository",
                parameters={},
                required_params=[]
            )
        ]
        
        integration = ComposioAgentIntegration(mock_composio_client)
        
        # Test setup
        result = await integration.setup_agent_tools("user123", "full_stack_engineer")
        
        assert result["agent_type"] == "full_stack_engineer"
        assert len(result["toolkits"]) > 0
        assert "github" in result["toolkits"]
        assert len(result["connections"]) > 0
        assert len(result["tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_agent_tool(self, mock_composio_client):
        """Test executing a tool for an agent."""
        # Setup mock
        mock_result = ComposioExecutionResult(
            success=True,
            data={"id": 123, "name": "test-repo"},
            tool_slug="GITHUB_CREATE_REPOSITORY",
            user_id="user123"
        )
        mock_composio_client.execute_tool.return_value = mock_result
        
        integration = ComposioAgentIntegration(mock_composio_client)
        
        # Test execution
        result = await integration.execute_agent_tool(
            "user123",
            "full_stack_engineer",
            "GITHUB_CREATE_REPOSITORY",
            {"name": "test-repo"}
        )
        
        assert result.success is True
        assert result.data["name"] == "test-repo"
        
        mock_composio_client.execute_tool.assert_called_once_with(
            user_id="user123",
            tool_slug="GITHUB_CREATE_REPOSITORY",
            arguments={"name": "test-repo"}
        )
    
    @pytest.mark.asyncio
    async def test_get_agent_status(self, mock_composio_client):
        """Test getting agent status."""
        # Setup mock connections
        mock_connections = [
            ComposioConnection(
                user_id="user123",
                toolkit="github",
                connection_id="conn_123",
                status="active"
            ),
            ComposioConnection(
                user_id="user123", 
                toolkit="linear",
                connection_id="conn_456",
                status="pending"
            )
        ]
        mock_composio_client.get_user_connections.return_value = mock_connections
        
        integration = ComposioAgentIntegration(mock_composio_client)
        
        # Test status
        status = await integration.get_agent_status("user123", "full_stack_engineer")
        
        assert status["agent_type"] == "full_stack_engineer"
        assert "github" in status["toolkit_status"]
        assert "linear" in status["toolkit_status"]
        assert status["toolkit_status"]["github"]["connected"] is True
        assert status["toolkit_status"]["linear"]["connected"] is False
        assert status["ready"] is False  # Not all toolkits are active
    
    @pytest.mark.asyncio
    async def test_unknown_agent_type(self, mock_composio_client):
        """Test handling unknown agent type."""
        integration = ComposioAgentIntegration(mock_composio_client)
        
        with pytest.raises(ComposioIntegrationError):
            await integration.setup_agent_tools("user123", "unknown_agent")
        
        status = await integration.get_agent_status("user123", "unknown_agent")
        assert "error" in status


class TestComposioIntegrationScenarios:
    """Test realistic Composio integration scenarios."""
    
    @pytest.mark.skipif(not COMPOSIO_AVAILABLE, reason="Composio SDK not available")
    @pytest.mark.asyncio
    async def test_full_agent_workflow(self):
        """Test a complete agent workflow."""
        with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
            # Setup comprehensive mocks
            mock_connection_request = Mock()
            mock_connection_request.connection_id = "conn_123"
            mock_connection_request.redirect_url = "https://auth.example.com"
            
            mock_tools = [
                {
                    "name": "Create Repository",
                    "slug": "GITHUB_CREATE_REPOSITORY",
                    "toolkit": "github",
                    "description": "Create a repository",
                    "parameters": {},
                    "required_params": []
                }
            ]
            
            mock_execution_result = {"id": 123, "name": "test-repo"}
            
            mock_composio_instance = Mock()
            mock_composio_instance.toolkits.authorize.return_value = mock_connection_request
            mock_composio_instance.tools.get.return_value = mock_tools
            mock_composio_instance.tools.execute.return_value = mock_execution_result
            mock_composio.return_value = mock_composio_instance
            
            # Initialize client and integration
            client = ComposioSDKClient("test-api-key")
            integration = ComposioAgentIntegration(client)
            
            user_id = "workflow-user"
            agent_type = "full_stack_engineer"
            
            # Setup agent tools
            setup_result = await integration.setup_agent_tools(user_id, agent_type)
            assert setup_result["agent_type"] == agent_type
            assert len(setup_result["tools"]) > 0
            
            # Execute a tool
            execution_result = await integration.execute_agent_tool(
                user_id, agent_type, "GITHUB_CREATE_REPOSITORY", {"name": "workflow-repo"}
            )
            assert execution_result.success is True
            assert execution_result.data["name"] == "test-repo"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test authentication error (skip Composio not available test since it's complex to mock)
        if COMPOSIO_AVAILABLE:
            with patch('agent_hive.integrations.composio_sdk.Composio') as mock_composio:
                mock_composio_instance = Mock()
                mock_composio_instance.toolkits.authorize.side_effect = Exception("Auth failed")
                mock_composio.return_value = mock_composio_instance
                
                client = ComposioSDKClient("test-api-key")
                
                with pytest.raises(ComposioAuthenticationError):
                    await client.authorize_toolkit("user123", "github")


@pytest.mark.asyncio
async def test_global_client_functions():
    """Test global client functions."""
    from agent_hive.integrations import get_composio_client, get_agent_integration
    
    if not COMPOSIO_AVAILABLE:
        pytest.skip("Composio SDK not available")
    
    with patch('agent_hive.integrations.composio_sdk.Composio'):
        # Test getting global client
        client1 = get_composio_client("test-key")
        client2 = get_composio_client("test-key")
        
        # Should return same instance
        assert client1 is client2
        
        # Test getting agent integration
        integration = get_agent_integration("test-key")
        assert isinstance(integration, ComposioAgentIntegration)