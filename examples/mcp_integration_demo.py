#!/usr/bin/env python3
"""
Demo script showing how to use the Composio MCP integration.

This script demonstrates:
1. Initializing the MCP client
2. Getting tools for different agent types
3. Executing tools with proper error handling
4. Checking tool availability and server status
"""

import asyncio
import logging

from agent_hive.integrations import (
    ComposioMCPClient,
    get_mcp_client,
    shutdown_mcp_client,
    MCPToolCategory,
    MCPError,
    MCPServerUnavailableError,
    MCPToolNotFoundError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_mcp_usage():
    """Demonstrate basic MCP client usage."""
    print("=== Basic MCP Client Usage Demo ===\n")

    # Initialize MCP client
    client = ComposioMCPClient()
    await client.initialize()

    try:
        # Get server status
        print("1. Server Status:")
        status = await client.get_server_status()
        for server_name, server_status in status.items():
            print(
                f"   {server_name}: Connected={server_status['connected']}, "
                f"Healthy={server_status['healthy']}, Tools={server_status['tool_count']}"
            )
        print()

        # Get all available tools
        print("2. Available Tools by Category:")
        github_tools = await client.get_available_tools(MCPToolCategory.GITHUB)
        linear_tools = await client.get_available_tools(MCPToolCategory.LINEAR)

        if github_tools:
            print("   GitHub Tools:")
            for server_name, tools in github_tools.items():
                for tool in tools:
                    print(f"     - {tool.name}: {tool.description}")

        if linear_tools:
            print("   Linear Tools:")
            for server_name, tools in linear_tools.items():
                for tool in tools:
                    print(f"     - {tool.name}: {tool.description}")
        print()

        # Check tool availability
        print("3. Tool Availability Check:")
        tools_to_check = ["create_repository", "create_issue", "nonexistent_tool"]
        for tool_name in tools_to_check:
            available = await client.check_tool_availability(tool_name)
            print(f"   {tool_name}: {'Available' if available else 'Not Available'}")
        print()

    finally:
        await client.shutdown()


async def demo_agent_specific_tools():
    """Demonstrate getting tools for specific agent types."""
    print("=== Agent-Specific Tools Demo ===\n")

    client = ComposioMCPClient()
    await client.initialize()

    try:
        agent_types = [
            "full_stack_engineer",
            "qa_engineer",
            "product_designer",
            "devops_engineer",
        ]

        for agent_type in agent_types:
            print(f"Tools for {agent_type}:")
            tools = await client.get_tools_for_agent(agent_type)

            if tools:
                # Group tools by category
                categories = {}
                for tool in tools:
                    category = tool.category.value
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(tool.name)

                for category, tool_names in categories.items():
                    print(f"   {category}: {', '.join(tool_names)}")
            else:
                print("   No tools available")
            print()

    finally:
        await client.shutdown()


async def demo_tool_execution():
    """Demonstrate tool execution with error handling."""
    print("=== Tool Execution Demo ===\n")

    client = ComposioMCPClient()
    await client.initialize()

    try:
        # Successful tool execution
        print("1. Successful Tool Execution:")
        try:
            result = await client.execute_tool(
                "create_repository",
                {
                    "name": "demo-project",
                    "description": "A demo project created via MCP",
                },
            )
            print(f"   Success: {result.success}")
            print(f"   Result: {result.result}")
            print(f"   Execution time: {result.execution_time:.3f}s")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # Tool execution with missing parameters
        print("2. Tool Execution with Missing Parameters:")
        try:
            result = await client.execute_tool("create_repository", {})
            print(f"   Unexpected success: {result}")
        except MCPError as e:
            print(f"   Expected error: {e}")
        print()

        # Tool not found
        print("3. Non-existent Tool Execution:")
        try:
            result = await client.execute_tool("nonexistent_tool", {})
            print(f"   Unexpected success: {result}")
        except MCPToolNotFoundError as e:
            print(f"   Expected error: {e}")
        print()

        # Server not found
        print("4. Tool Execution with Non-existent Server:")
        try:
            result = await client.execute_tool(
                "create_repository", {"name": "test"}, server_name="nonexistent-server"
            )
            print(f"   Unexpected success: {result}")
        except MCPServerUnavailableError as e:
            print(f"   Expected error: {e}")
        print()

    finally:
        await client.shutdown()


async def demo_workflow_scenarios():
    """Demonstrate realistic workflow scenarios."""
    print("=== Workflow Scenarios Demo ===\n")

    client = ComposioMCPClient()
    await client.initialize()

    try:
        # Full Stack Engineer Workflow
        print("1. Full Stack Engineer Workflow:")
        fs_tools = await client.get_tools_for_agent("full_stack_engineer")
        print(f"   Available tools: {len(fs_tools)}")

        # Create a repository
        repo_result = await client.execute_tool(
            "create_repository",
            {
                "name": "new-microservice",
                "description": "A new microservice for the platform",
            },
        )
        print(f"   Created repository: {repo_result.result.get('name', 'N/A')}")

        # Create a pull request
        pr_result = await client.execute_tool(
            "create_pull_request",
            {
                "title": "Initial service setup",
                "body": "Setting up the basic microservice structure",
                "head": "feature/initial-setup",
                "base": "main",
            },
        )
        print(f"   Created PR: #{pr_result.result.get('number', 'N/A')}")
        print()

        # QA Engineer Workflow
        print("2. QA Engineer Workflow:")
        qa_tools = await client.get_tools_for_agent("qa_engineer")
        print(f"   Available tools: {len(qa_tools)}")

        # List existing issues
        issues_result = await client.execute_tool("list_issues", {})
        print(f"   Found {len(issues_result.result)} existing issues")

        # Create a bug report
        bug_result = await client.execute_tool(
            "create_issue",
            {
                "title": "Bug: API endpoint returns 500 error",
                "description": "The /api/users endpoint is returning 500 errors intermittently",
                "priority": 1,
            },
        )
        print(f"   Created bug report: {bug_result.result.get('identifier', 'N/A')}")
        print()

        # Product Designer Workflow
        print("3. Product Designer Workflow:")
        designer_tools = await client.get_tools_for_agent("product_designer")
        print(f"   Available tools: {len(designer_tools)}")

        # Get Figma file info
        figma_result = await client.execute_tool(
            "get_file", {"file_key": "demo-design-file-key"}
        )
        print(f"   Figma file: {figma_result.result.get('name', 'N/A')}")

        # Create documentation
        doc_result = await client.execute_tool(
            "create_page",
            {
                "title": "Design System Guidelines",
                "content": "Guidelines for using our design system components",
            },
        )
        print(f"   Created documentation: {doc_result.result.get('title', 'N/A')}")
        print()

    finally:
        await client.shutdown()


async def demo_global_client():
    """Demonstrate using the global MCP client."""
    print("=== Global MCP Client Demo ===\n")

    # Get global client instance
    client1 = await get_mcp_client()
    print(f"Client 1 initialized: {client1._initialized}")

    # Get same instance
    client2 = await get_mcp_client()
    print(f"Client 2 is same instance: {client1 is client2}")

    # Use the client
    tools = await client1.get_tools_for_agent("full_stack_engineer")
    print(f"Available tools for full stack engineer: {len(tools)}")

    # Shutdown global client
    await shutdown_mcp_client()
    print("Global client shutdown")

    # Get new instance after shutdown
    client3 = await get_mcp_client()
    print(f"Client 3 is new instance: {client1 is not client3}")

    await shutdown_mcp_client()


async def main():
    """Run all demo scenarios."""
    print("Composio MCP Integration Demo")
    print("=" * 50)
    print()

    try:
        await demo_basic_mcp_usage()
        await demo_agent_specific_tools()
        await demo_tool_execution()
        await demo_workflow_scenarios()
        await demo_global_client()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")

    except Exception as e:
        logger.error("Demo failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
