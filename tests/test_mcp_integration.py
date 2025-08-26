"""Integration tests for MCP (Model Context Protocol) functionality."""

import asyncio
import json
import pytest
import pytest_asyncio

from agent_hive.integrations import (
    ComposioMCPClient,
    MCPTool,
    MCPServerConfig,
    MCPExecutionResult,
    MCPError,
    MCPServerUnavailableError,
    MCPToolNotFoundError,
    MCPExecutionTimeoutError,
    MCPToolCategory,
    ComposioMCPServer,
    AgentToolMapping,
)


class TestMCPTool:
    """Test MCPTool data structure."""

    def test_mcp_tool_creation(self):
        """Test creating an MCP tool."""
        tool = MCPTool(
            name="create_repository",
            category=MCPToolCategory.GITHUB,
            description="Create a new GitHub repository",
            parameters={"name": "str", "description": "str"},
            required_params=["name"],
            server_name="github-mcp",
        )

        assert tool.name == "create_repository"
        assert tool.category == MCPToolCategory.GITHUB
        assert tool.description == "Create a new GitHub repository"
        assert tool.parameters == {"name": "str", "description": "str"}
        assert tool.required_params == ["name"]
        assert tool.server_name == "github-mcp"
        assert tool.timeout == 30  # default
        assert tool.retry_count == 3  # default


class TestMCPServerConfig:
    """Test MCP server configuration."""

    def test_server_config_creation(self):
        """Test creating server configuration."""
        config = MCPServerConfig(
            name="github-mcp",
            command="uvx",
            args=["composio-github-mcp@latest"],
            env={"GITHUB_TOKEN": "test-token"},
            timeout=45,
            max_retries=5,
            disabled=False,
            auto_approve=["create_repository"],
            health_check_interval=120,
        )

        assert config.name == "github-mcp"
        assert config.command == "uvx"
        assert config.args == ["composio-github-mcp@latest"]
        assert config.env == {"GITHUB_TOKEN": "test-token"}
        assert config.timeout == 45
        assert config.max_retries == 5
        assert config.disabled is False
        assert config.auto_approve == ["create_repository"]
        assert config.health_check_interval == 120


class TestComposioMCPServer:
    """Test ComposioMCPServer implementation."""

    @pytest.fixture
    def server_config(self):
        """Create a test server configuration."""
        return MCPServerConfig(
            name="github-test",
            command="uvx",
            args=["composio-github-mcp@latest"],
            env={"GITHUB_TOKEN": "test-token"},
        )

    @pytest.fixture
    def mcp_server(self, server_config):
        """Create a test MCP server."""
        return ComposioMCPServer(server_config)

    @pytest.mark.asyncio
    async def test_server_connection(self, mcp_server):
        """Test server connection."""
        assert not mcp_server.connected

        # Connect to server
        result = await mcp_server.connect()
        assert result is True
        assert mcp_server.connected is True

        # Test tools were discovered
        tools = await mcp_server.list_tools()
        assert len(tools) > 0
        assert all(isinstance(tool, MCPTool) for tool in tools)

        # Disconnect
        await mcp_server.disconnect()
        assert mcp_server.connected is False

    @pytest.mark.asyncio
    async def test_server_health_check(self, mcp_server):
        """Test server health checking."""
        # Health check should fail when not connected
        assert await mcp_server.is_healthy() is False

        # Connect and check health
        await mcp_server.connect()
        assert await mcp_server.is_healthy() is True

        await mcp_server.disconnect()

    @pytest.mark.asyncio
    async def test_tool_discovery(self, mcp_server):
        """Test tool discovery for GitHub server."""
        await mcp_server.connect()
        tools = await mcp_server.list_tools()

        # Check that GitHub tools are discovered
        tool_names = [tool.name for tool in tools]
        assert "create_repository" in tool_names
        assert "create_pull_request" in tool_names
        assert "list_issues" in tool_names

        # Check tool properties
        create_repo_tool = next(
            tool for tool in tools if tool.name == "create_repository"
        )
        assert create_repo_tool.category == MCPToolCategory.GITHUB
        assert "name" in create_repo_tool.required_params

        await mcp_server.disconnect()

    @pytest.mark.asyncio
    async def test_tool_execution_success(self, mcp_server):
        """Test successful tool execution."""
        await mcp_server.connect()

        # Execute create_repository tool
        result = await mcp_server.execute_tool(
            "create_repository", {"name": "test-repo", "description": "Test repository"}
        )

        assert isinstance(result, MCPExecutionResult)
        assert result.success is True
        assert result.error is None
        assert result.server_name == "github-test"
        assert result.tool_name == "create_repository"
        assert result.execution_time > 0

        # Check result content
        assert isinstance(result.result, dict)
        assert result.result["name"] == "test-repo"
        assert "id" in result.result

        await mcp_server.disconnect()

    @pytest.mark.asyncio
    async def test_tool_execution_missing_params(self, mcp_server):
        """Test tool execution with missing required parameters."""
        await mcp_server.connect()

        # Try to execute without required parameter
        with pytest.raises(MCPError) as exc_info:
            await mcp_server.execute_tool("create_repository", {})

        assert "Missing required parameters" in str(exc_info.value)
        assert exc_info.value.server_name == "github-test"
        assert exc_info.value.tool_name == "create_repository"

        await mcp_server.disconnect()

    @pytest.mark.asyncio
    async def test_tool_execution_not_found(self, mcp_server):
        """Test execution of non-existent tool."""
        await mcp_server.connect()

        with pytest.raises(MCPToolNotFoundError) as exc_info:
            await mcp_server.execute_tool("nonexistent_tool", {})

        assert "not found" in str(exc_info.value)
        assert exc_info.value.server_name == "github-test"
        assert exc_info.value.tool_name == "nonexistent_tool"

        await mcp_server.disconnect()

    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self, mcp_server):
        """Test tool execution timeout."""
        await mcp_server.connect()

        # Mock a slow execution
        original_execute = mcp_server._execute_tool_impl

        async def slow_execute(tool, parameters):
            await asyncio.sleep(2)  # Longer than default timeout
            return await original_execute(tool, parameters)

        mcp_server._execute_tool_impl = slow_execute

        # Set short timeout for testing
        tools = await mcp_server.list_tools()
        create_repo_tool = next(
            tool for tool in tools if tool.name == "create_repository"
        )
        create_repo_tool.timeout = 0.1  # Very short timeout

        with pytest.raises(MCPExecutionTimeoutError):
            await mcp_server.execute_tool("create_repository", {"name": "test-repo"})

        await mcp_server.disconnect()


class TestComposioMCPClient:
    """Test ComposioMCPClient functionality."""

    @pytest.fixture
    def mock_config_data(self):
        """Mock MCP configuration data."""
        return {
            "mcpServers": {
                "github-mcp": {
                    "command": "uvx",
                    "args": ["composio-github-mcp@latest"],
                    "env": {"GITHUB_TOKEN": "test-token"},
                    "disabled": False,
                    "autoApprove": ["create_repository"],
                },
                "linear-mcp": {
                    "command": "uvx",
                    "args": ["composio-linear-mcp@latest"],
                    "env": {"LINEAR_API_KEY": "test-key"},
                    "disabled": False,
                },
                "disabled-server": {
                    "command": "uvx",
                    "args": ["disabled-mcp@latest"],
                    "disabled": True,
                },
            }
        }

    @pytest_asyncio.fixture
    async def mcp_client(self, mock_config_data, tmp_path):
        """Create a test MCP client."""
        # Create temporary config file
        config_file = tmp_path / "mcp_config.json"
        with open(config_file, "w") as f:
            json.dump(mock_config_data, f)

        client = ComposioMCPClient(str(config_file))
        await client.initialize()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_client_initialization(self, mcp_client):
        """Test MCP client initialization."""
        assert mcp_client._initialized is True
        assert len(mcp_client.servers) >= 2  # github and linear, not disabled
        assert "github-mcp" in mcp_client.servers
        assert "linear-mcp" in mcp_client.servers
        assert "disabled-server" not in mcp_client.servers

    @pytest.mark.asyncio
    async def test_get_tools_for_agent(self, mcp_client):
        """Test getting tools for specific agent types."""
        # Test full stack engineer
        tools = await mcp_client.get_tools_for_agent("full_stack_engineer")
        assert len(tools) > 0

        # Should have GitHub tools (required)
        github_tools = [
            tool for tool in tools if tool.category == MCPToolCategory.GITHUB
        ]
        assert len(github_tools) > 0

        # Check priority tools are first
        priority_tools = ["create_repository", "create_pull_request", "list_issues"]
        first_tools = [tool.name for tool in tools[:3]]
        assert any(tool in priority_tools for tool in first_tools)

        # Test QA engineer
        qa_tools = await mcp_client.get_tools_for_agent("qa_engineer")
        assert len(qa_tools) > 0

        # Should have both GitHub and Linear tools (both required)
        categories = {tool.category for tool in qa_tools}
        assert MCPToolCategory.GITHUB in categories
        assert MCPToolCategory.LINEAR in categories

        # Test unknown agent type
        unknown_tools = await mcp_client.get_tools_for_agent("unknown_agent")
        assert len(unknown_tools) == 0

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_client):
        """Test successful tool execution."""
        result = await mcp_client.execute_tool(
            "create_repository", {"name": "test-repo", "description": "Test repository"}
        )

        assert isinstance(result, MCPExecutionResult)
        assert result.success is True
        assert result.error is None
        assert result.tool_name == "create_repository"
        assert isinstance(result.result, dict)

    @pytest.mark.asyncio
    async def test_execute_tool_with_server_name(self, mcp_client):
        """Test tool execution with specific server name."""
        result = await mcp_client.execute_tool(
            "create_repository", {"name": "test-repo"}, server_name="github-mcp"
        )

        assert result.success is True
        assert result.server_name == "github-mcp"

    @pytest.mark.asyncio
    async def test_execute_tool_server_not_found(self, mcp_client):
        """Test tool execution with non-existent server."""
        with pytest.raises(MCPServerUnavailableError):
            await mcp_client.execute_tool(
                "create_repository",
                {"name": "test-repo"},
                server_name="nonexistent-server",
            )

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mcp_client):
        """Test execution of non-existent tool."""
        with pytest.raises(MCPToolNotFoundError):
            await mcp_client.execute_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_check_tool_availability(self, mcp_client):
        """Test checking tool availability."""
        # Existing tool
        assert await mcp_client.check_tool_availability("create_repository") is True

        # Non-existent tool
        assert await mcp_client.check_tool_availability("nonexistent_tool") is False

    @pytest.mark.asyncio
    async def test_get_available_tools(self, mcp_client):
        """Test getting all available tools."""
        # Get all tools
        all_tools = await mcp_client.get_available_tools()
        assert isinstance(all_tools, dict)
        assert len(all_tools) > 0

        # Get tools by category
        github_tools = await mcp_client.get_available_tools(MCPToolCategory.GITHUB)
        assert "github-mcp" in github_tools
        assert all(
            tool.category == MCPToolCategory.GITHUB
            for tools in github_tools.values()
            for tool in tools
        )

    @pytest.mark.asyncio
    async def test_server_status(self, mcp_client):
        """Test getting server status."""
        status = await mcp_client.get_server_status()

        assert isinstance(status, dict)
        assert "github-mcp" in status
        assert "linear-mcp" in status

        # Check status structure
        github_status = status["github-mcp"]
        assert "connected" in github_status
        assert "healthy" in github_status
        assert "tool_count" in github_status
        assert "config" in github_status

        assert github_status["connected"] is True
        assert github_status["healthy"] is True
        assert github_status["tool_count"] > 0

    @pytest.mark.asyncio
    async def test_reload_server_config(self, mcp_client):
        """Test reloading server configuration."""
        # Test reloading existing server
        result = await mcp_client.reload_server_config("github-mcp")
        assert result is True

        # Test reloading non-existent server
        result = await mcp_client.reload_server_config("nonexistent-server")
        assert result is False

    @pytest.mark.asyncio
    async def test_agent_tool_mappings(self, mcp_client):
        """Test agent tool mappings configuration."""
        # Check that default mappings are set up
        assert "full_stack_engineer" in mcp_client.agent_mappings
        assert "qa_engineer" in mcp_client.agent_mappings
        assert "product_designer" in mcp_client.agent_mappings
        assert "devops_engineer" in mcp_client.agent_mappings

        # Check full stack engineer mapping
        fs_mapping = mcp_client.agent_mappings["full_stack_engineer"]
        assert isinstance(fs_mapping, AgentToolMapping)
        assert MCPToolCategory.GITHUB in fs_mapping.required_categories
        assert "create_repository" in fs_mapping.priority_tools

        # Check QA engineer mapping
        qa_mapping = mcp_client.agent_mappings["qa_engineer"]
        assert MCPToolCategory.GITHUB in qa_mapping.required_categories
        assert MCPToolCategory.LINEAR in qa_mapping.required_categories

    @pytest.mark.asyncio
    async def test_client_shutdown(self, mock_config_data, tmp_path):
        """Test client shutdown."""
        # Create and initialize client
        config_file = tmp_path / "mcp_config.json"
        with open(config_file, "w") as f:
            json.dump(mock_config_data, f)

        client = ComposioMCPClient(str(config_file))
        await client.initialize()

        assert client._initialized is True
        assert len(client.servers) > 0

        # Shutdown client
        await client.shutdown()

        assert client._initialized is False
        assert len(client.servers) == 0
        assert len(client.tool_registry) == 0


class TestMCPIntegrationScenarios:
    """Test realistic MCP integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_stack_workflow(self):
        """Test a complete full-stack development workflow."""
        client = ComposioMCPClient()
        await client.initialize()

        try:
            # Get tools for full stack engineer
            tools = await client.get_tools_for_agent("full_stack_engineer")
            assert len(tools) > 0

            # Create a repository
            repo_result = await client.execute_tool(
                "create_repository",
                {"name": "new-project", "description": "A new project repository"},
            )
            assert repo_result.success is True

            # Create a pull request
            pr_result = await client.execute_tool(
                "create_pull_request",
                {
                    "title": "Initial project setup",
                    "body": "Setting up the initial project structure",
                    "head": "feature/setup",
                    "base": "main",
                },
            )
            assert pr_result.success is True

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_qa_workflow(self):
        """Test a QA engineer workflow."""
        client = ComposioMCPClient()
        await client.initialize()

        try:
            # Get tools for QA engineer
            tools = await client.get_tools_for_agent("qa_engineer")
            qa_categories = {tool.category for tool in tools}

            # Should have both GitHub and Linear tools
            assert MCPToolCategory.GITHUB in qa_categories
            assert MCPToolCategory.LINEAR in qa_categories

            # List issues from GitHub
            issues_result = await client.execute_tool("list_issues", {})
            assert issues_result.success is True

            # Create a Linear issue for bug tracking
            linear_result = await client.execute_tool(
                "create_issue",
                {
                    "title": "Bug: Login form validation",
                    "description": "The login form doesn't validate email format",
                    "priority": 2,
                },
            )
            assert linear_result.success is True

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery scenarios."""
        client = ComposioMCPClient()
        await client.initialize()

        try:
            # Test tool not found error
            with pytest.raises(MCPToolNotFoundError):
                await client.execute_tool("invalid_tool", {})

            # Test missing parameters error
            with pytest.raises(MCPError):
                await client.execute_tool("create_repository", {})

            # Test server unavailable error
            with pytest.raises(MCPServerUnavailableError):
                await client.execute_tool(
                    "create_repository", {"name": "test"}, server_name="invalid_server"
                )

        finally:
            await client.shutdown()


@pytest.mark.asyncio
async def test_global_mcp_client():
    """Test global MCP client functions."""
    from agent_hive.integrations import get_mcp_client, shutdown_mcp_client

    # Get global client
    client1 = await get_mcp_client()
    assert isinstance(client1, ComposioMCPClient)
    assert client1._initialized is True

    # Get same instance
    client2 = await get_mcp_client()
    assert client1 is client2

    # Shutdown global client
    await shutdown_mcp_client()

    # Should create new instance after shutdown
    client3 = await get_mcp_client()
    assert client3 is not client1

    await shutdown_mcp_client()
