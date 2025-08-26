#!/usr/bin/env python3
"""
Demonstration of the AgentHive agent factory system.

This script shows how to:
1. Create and register agent configurations
2. Load agent configurations from files
3. Create agent nodes for LangGraph workflows
4. Validate agent capabilities and requirements
"""

import asyncio
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

# Import AgentHive components
from agent_hive.agents.factory import AgentCapabilities, get_agent_factory
from agent_hive.core.config import AgentConfig
from agent_hive.core.state import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoTool(BaseTool):
    """A simple demo tool for testing."""

    name: str = "demo_tool"
    description: str = "A demonstration tool that echoes input"

    def _run(self, query: str) -> str:
        return f"Demo tool processed: {query}"

    async def _arun(self, query: str) -> str:
        return f"Demo tool async processed: {query}"


async def demo_agent_factory() -> None:
    """Demonstrate the agent factory system."""

    print("🤖 AgentHive Agent Factory Demo")
    print("=" * 50)

    # 1. Create agent factory
    print("\n1. Creating Agent Factory...")
    factory = get_agent_factory()

    # 2. Create a sample agent configuration
    print("\n2. Creating Agent Configuration...")
    agent_config = AgentConfig(
        name="demo_agent",
        display_name="Demo Agent",
        description="A demonstration agent for testing the factory system",
        system_prompt="You are a helpful demo agent. Respond concisely and helpfully.",
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        capabilities=["demo_capability"],
        specialized_for=["demonstration", "testing"],
    )

    # 3. Create agent capabilities
    print("\n3. Defining Agent Capabilities...")
    capabilities = AgentCapabilities(
        required_capabilities=["demo_capability"],
        optional_capabilities=["advanced_demo"],
        model_requirements=["function_calling"],
        required_tools=["demo_tool"],
        mcp_requirements=[],
    )

    # 4. Create demo tools
    print("\n4. Creating Demo Tools...")
    demo_tools: list[BaseTool] = [DemoTool()]

    # 5. Register agent configuration
    print("\n5. Registering Agent Configuration...")
    factory.register_agent_config(agent_config, capabilities, demo_tools)

    # 6. List registered agents
    print("\n6. Listing Registered Agents...")
    agents = factory.list_agents()
    print(f"   Registered agents: {agents}")

    # 7. Validate agent
    print("\n7. Validating Agent...")
    try:
        validation_result = await factory.validate_agent("demo_agent")
        print(f"   Validation result: {validation_result['valid']}")
        if validation_result["errors"]:
            print(f"   Validation errors: {validation_result['errors']}")
    except Exception as e:
        print(f"   Validation failed: {e}")

    # 8. Load configurations from files (if available)
    print("\n8. Loading Agent Configurations from Files...")
    config_dir = Path("agent_configs")
    if config_dir.exists():
        factory.load_agent_configurations(config_dir)
        updated_agents = factory.list_agents()
        print(f"   Agents after loading configs: {updated_agents}")
    else:
        print("   No agent_configs directory found, skipping file loading")

    # 9. Create agent node (mock model for demo)
    print("\n9. Creating Agent Node...")
    try:
        # Note: In a real scenario, you'd have proper model configurations
        # For demo purposes, we'll show the structure without actual model calls
        print("   Agent node creation would require proper model setup")
        print("   Structure: factory.create_agent_node('demo_agent')")

        # Show agent configuration details
        config = factory.get_agent_config("demo_agent")
        if config:
            print(f"   Agent config loaded: {config.name} - {config.display_name}")
            print(f"   Model: {config.model_provider}/{config.model_name}")
            print(f"   Capabilities: {config.capabilities}")

    except Exception as e:
        print(
            f"   Agent node creation failed (expected without proper model setup): {e}"
        )

    # 10. Demonstrate state structure
    print("\n10. Sample Agent State Structure...")
    sample_state = AgentState(
        task="Demonstrate the agent factory system",
        messages=[HumanMessage(content="Hello, demo agent!")],
        next="",
        scratchpad={},
        mode="direct",
        active_agents=[],
        multimodal_content={},
        session_id="demo_session_123",
        user_id="demo_user",
        last_updated=None,
        errors=[],
        task_status={},
    )

    print(f"   Sample state keys: {list(sample_state.keys())}")
    print(f"   Task: {sample_state['task']}")
    print(f"   Mode: {sample_state['mode']}")
    print(f"   Session ID: {sample_state['session_id']}")

    print("\n✅ Agent Factory Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("- ✓ Agent configuration creation and registration")
    print("- ✓ Agent capability definition and validation")
    print("- ✓ Tool integration and management")
    print("- ✓ Configuration file loading")
    print("- ✓ Agent state structure")
    print("- ✓ Error handling and recovery strategies")


async def demo_error_recovery() -> None:
    """Demonstrate error recovery strategies."""

    print("\n🔧 Error Recovery Demo")
    print("=" * 30)

    from agent_hive.agents.factory import ErrorRecoveryStrategy
    from datetime import datetime

    # Create sample state for error scenarios
    error_state = AgentState(
        task="Test error handling",
        messages=[],
        next="",
        scratchpad={},
        mode="direct",
        active_agents=["test_agent"],
        multimodal_content={},
        session_id="error_test_session",
        user_id="test_user",
        last_updated=datetime.now(),
        errors=[],
        task_status={},
    )

    recovery = ErrorRecoveryStrategy()

    # 1. Model timeout recovery
    print("\n1. Model Timeout Recovery...")
    timeout_state = await recovery.handle_model_timeout(error_state, "test_agent")
    print(f"   Errors added: {len(timeout_state['errors'])}")
    print(
        f"   Recovery info in scratchpad: {'test_agent_recovery' in timeout_state['scratchpad']}"
    )

    # 2. Tool failure recovery
    print("\n2. Tool Failure Recovery...")
    tool_error = Exception("Tool connection failed")
    tool_state = await recovery.handle_tool_failure(
        error_state, "test_agent", "demo_tool", tool_error
    )
    print(
        f"   Tool failure recorded: {tool_state['errors'][-1]['type'] == 'tool_failure'}"
    )

    # 3. Validation failure recovery
    print("\n3. Validation Failure Recovery...")
    validation_errors = ["Missing required capability", "Model unavailable"]
    validation_state = await recovery.handle_validation_failure(
        error_state, "test_agent", validation_errors
    )
    print(
        f"   Validation errors recorded: {len(validation_state['errors'][-1]['validation_errors'])}"
    )

    print("\n✅ Error Recovery Demo Complete!")


if __name__ == "__main__":
    print("Starting AgentHive Agent Factory Demonstration...")

    # Run the main demo
    asyncio.run(demo_agent_factory())

    # Run error recovery demo
    asyncio.run(demo_error_recovery())

    print("\n🎉 All demonstrations completed successfully!")
