#!/usr/bin/env python3
"""
Comprehensive demo showing full Composio integration with AgentHive.

This demo shows:
1. MCP integration (mock/development mode)
2. Direct Composio SDK integration (production mode)
3. Unified client that combines both approaches
4. Real OAuth flows and tool execution
5. Agent-specific tool setup and execution
"""

import asyncio
import logging
import os

# from typing import Dict, Any

from agent_hive.integrations import (
    # MCP Integration
    get_mcp_client,
    # Composio SDK Integration
    ComposioAgentIntegration,
    get_composio_client,
    COMPOSIO_AVAILABLE,
    # Unified Integration
    UnifiedIntegrationClient,
    IntegrationMode,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_mcp_integration():
    """Demonstrate MCP integration (mock/development mode)."""
    print("=== MCP Integration Demo ===\n")

    client = await get_mcp_client()

    try:
        # Get tools for different agent types
        print("1. Agent-specific tools via MCP:")
        agent_types = ["full_stack_engineer", "qa_engineer", "product_designer"]

        for agent_type in agent_types:
            tools = await client.get_tools_for_agent(agent_type)
            print(f"   {agent_type}: {len(tools)} tools")
            if tools:
                categories = set(tool.category.value for tool in tools)
                print(f"     Categories: {', '.join(categories)}")
        print()

        # Execute a tool
        print("2. Tool execution via MCP:")
        result = await client.execute_tool(
            "create_repository",
            {"name": "mcp-demo-repo", "description": "Demo repository via MCP"},
        )
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")
        print()

    finally:
        await client.shutdown()


async def demo_composio_sdk_integration():
    """Demonstrate direct Composio SDK integration (production mode)."""
    print("=== Composio SDK Integration Demo ===\n")

    if not COMPOSIO_AVAILABLE:
        print("Composio SDK not available. Install with: pip install composio")
        return

    # Initialize with API key from environment
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        print("COMPOSIO_API_KEY not set. Skipping Composio SDK demo.")
        return

    try:
        client = get_composio_client(api_key)
        integration = ComposioAgentIntegration(client)
        user_id = "demo-user@example.com"

        # Setup agent integrations
        print("1. Setting up agent integrations:")
        setup_result = await integration.setup_agent_tools(
            user_id, "full_stack_engineer", auto_authorize=False
        )

        print(f"   Agent type: {setup_result['agent_type']}")
        print(f"   Required toolkits: {setup_result['toolkits']}")
        print(f"   Tools available: {len(setup_result.get('tools', []))}")

        # Show authorization URLs
        if setup_result.get("authorization_urls"):
            print("\n   Authorization required for:")
            for auth in setup_result["authorization_urls"]:
                print(f"     {auth['toolkit']}: {auth['url']}")
        print()

        # Get agent status
        print("2. Agent integration status:")
        status = await integration.get_agent_status(user_id, "full_stack_engineer")
        print(f"   Ready: {status['ready']}")
        for toolkit, toolkit_status in status["toolkit_status"].items():
            print(f"   {toolkit}: {toolkit_status['status']}")
        print()

        # Note: Actual tool execution would require completed OAuth flows
        print("3. Tool execution (requires OAuth completion):")
        print("   Would execute tools after user completes OAuth flows")
        print()

    except Exception as e:
        print(f"Composio SDK demo failed: {e}")


async def demo_unified_integration():
    """Demonstrate unified integration client."""
    print("=== Unified Integration Demo ===\n")

    # Test different modes
    modes = [IntegrationMode.MCP_ONLY, IntegrationMode.AUTO, IntegrationMode.HYBRID]

    for mode in modes:
        print(f"Testing {mode.value} mode:")

        try:
            client = UnifiedIntegrationClient(mode=mode)
            await client.initialize()

            user_id = "unified-demo-user@example.com"

            # Get integration status
            status = await client.get_integration_status(user_id)
            print(f"   MCP available: {status['mcp_available']}")
            print(f"   Composio available: {status['composio_available']}")

            # Get tools for agent
            tools = await client.get_tools_for_agent(user_id, "full_stack_engineer")
            print(f"   Tools found: {len(tools)}")

            # Group by source
            sources = {}
            for tool in tools:
                if tool.source not in sources:
                    sources[tool.source] = 0
                sources[tool.source] += 1

            for source, count in sources.items():
                print(f"     {source}: {count} tools")

            # Execute a tool (will use best available source)
            if tools:
                sample_tool = tools[0]
                print(
                    f"   Executing sample tool: {sample_tool.name} (from {sample_tool.source})"
                )

                result = await client.execute_tool(
                    user_id,
                    sample_tool.slug,
                    {"name": "unified-demo", "description": "Demo via unified client"},
                )
                print(f"     Success: {result.success}")
                print(f"     Source used: {result.source}")

            await client.shutdown()
            print()

        except Exception as e:
            print(f"   Error in {mode.value} mode: {e}")
            print()


async def demo_agent_workflow():
    """Demonstrate a complete agent workflow."""
    print("=== Complete Agent Workflow Demo ===\n")

    client = UnifiedIntegrationClient(mode=IntegrationMode.AUTO)
    await client.initialize()

    try:
        user_id = "workflow-demo-user@example.com"
        agent_type = "full_stack_engineer"

        # Setup agent integrations
        print("1. Setting up agent integrations:")
        setup_result = await client.setup_agent_integrations(
            user_id, agent_type, auto_authorize=False
        )

        print(f"   Agent: {setup_result['agent_type']}")
        print(f"   User: {setup_result['user_id']}")

        if setup_result.get("mcp_setup", {}).get("success"):
            mcp_setup = setup_result["mcp_setup"]
            print(
                f"   MCP: {mcp_setup['tools_count']} tools, categories: {mcp_setup['categories']}"
            )

        if setup_result.get("composio_setup", {}).get("success"):
            composio_setup = setup_result["composio_setup"]
            print(f"   Composio: {len(composio_setup.get('tools', []))} tools")
        print()

        # Simulate agent tasks
        print("2. Simulating agent tasks:")

        tasks = [
            {
                "description": "Create a new repository",
                "tool": "create_repository",
                "args": {
                    "name": "agent-created-repo",
                    "description": "Repository created by AI agent",
                },
            },
            {
                "description": "Create a pull request",
                "tool": "create_pull_request",
                "args": {
                    "title": "AI-generated feature",
                    "body": "This PR was created by an AI agent",
                    "head": "feature/ai-generated",
                    "base": "main",
                },
            },
            {
                "description": "Create a Linear issue",
                "tool": "create_issue",
                "args": {
                    "title": "Bug reported by AI agent",
                    "description": "The AI agent detected a potential issue",
                    "priority": 2,
                },
            },
        ]

        for i, task in enumerate(tasks, 1):
            print(f"   Task {i}: {task['description']}")

            try:
                result = await client.execute_tool(user_id, task["tool"], task["args"])

                if result.success:
                    print(f"     ✅ Success via {result.source}")
                    if isinstance(result.data, dict) and "name" in result.data:
                        print(f"     Created: {result.data['name']}")
                else:
                    print(f"     ❌ Failed: {result.error}")

            except Exception as e:
                print(f"     ❌ Error: {e}")

        print()

    finally:
        await client.shutdown()


async def demo_custom_tools():
    """Demonstrate custom tool creation."""
    print("=== Custom Tools Demo ===\n")

    if not COMPOSIO_AVAILABLE:
        print("Composio SDK not available for custom tools demo")
        return

    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        print("COMPOSIO_API_KEY not set. Skipping custom tools demo.")
        return

    try:
        client = get_composio_client(api_key)

        # Create a standalone custom tool
        print("1. Creating standalone custom tool:")

        def calculate_fibonacci(request):
            """Calculate fibonacci number."""
            n = request.n
            if n <= 1:
                return n

            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

        tool_slug = client.create_custom_tool(
            slug="CALCULATE_FIBONACCI",
            name="Calculate Fibonacci",
            description="Calculate the nth Fibonacci number",
            input_params={
                "n": {
                    "type": "integer",
                    "description": "The position in Fibonacci sequence",
                }
            },
            execute_fn=calculate_fibonacci,
        )

        print(f"   Created tool: {tool_slug}")

        # Execute the custom tool
        print("2. Executing custom tool:")
        result = await client.execute_tool(
            user_id="demo-user", tool_slug=tool_slug, arguments={"n": 10}
        )

        print(f"   Fibonacci(10) = {result.data}")
        print()

    except Exception as e:
        print(f"Custom tools demo failed: {e}")


async def main():
    """Run all integration demos."""
    print("Composio Full Integration Demo")
    print("=" * 50)
    print()

    try:
        # Run all demos
        await demo_mcp_integration()
        await demo_composio_sdk_integration()
        await demo_unified_integration()
        await demo_agent_workflow()
        await demo_custom_tools()

        print("=" * 50)
        print("All demos completed!")

        # Show integration summary
        print("\nIntegration Summary:")
        print("- MCP Integration: ✅ Mock tools for development/testing")
        print("- Composio SDK: ✅ Real API integrations (requires API key)")
        print("- Unified Client: ✅ Combines both approaches seamlessly")
        print("- Agent Workflows: ✅ Complete agent tool setup and execution")
        print("- Custom Tools: ✅ Create custom tools for specific needs")

        print("\nNext Steps:")
        print("1. Set COMPOSIO_API_KEY environment variable for real integrations")
        print("2. Complete OAuth flows for production tool access")
        print("3. Integrate with your agent framework of choice")
        print("4. Create custom tools for domain-specific needs")

    except Exception as e:
        logger.error("Demo failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
