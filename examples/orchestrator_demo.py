"""Demo script showing the multi-agent orchestrator in action."""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from agent_hive.core.graphs import OrchestratorGraphFactory, SessionManager
from agent_hive.core.state import AgentState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_task_analysis():
    """Demonstrate task analysis functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Task Analysis")
    print("=" * 60)

    session_manager = SessionManager()
    await session_manager.start()

    try:
        factory = OrchestratorGraphFactory(session_manager)

        # Load agent configurations
        config_dir = Path("agent_configs")
        factory.agent_factory.load_agent_configurations(config_dir)

        # Test different types of tasks
        test_tasks = [
            "Build a REST API for user management with authentication",
            "Create comprehensive test suite for the web application",
            "Design a modern UI/UX for the mobile app",
            "Deploy the application to AWS with proper monitoring",
            "Build, test, design, and deploy a complete e-commerce platform",
        ]

        for i, task in enumerate(test_tasks, 1):
            print(f"\n{i}. Task: {task}")

            state: AgentState = {
                "task": task,
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": [],
                "multimodal_content": {},
                "session_id": f"demo-session-{i}",
                "user_id": "demo-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.analyze_task_requirements(state)

            print(f"   Required Agents: {', '.join(result['active_agents'])}")
            print(f"   Next Agent: {result['next']}")
            print(
                f"   Agent Count: {result['scratchpad']['task_analysis']['agent_count']}"
            )

    finally:
        await session_manager.stop()


async def demo_routing_workflow():
    """Demonstrate routing workflow functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Routing Workflow")
    print("=" * 60)

    session_manager = SessionManager()
    await session_manager.start()

    try:
        factory = OrchestratorGraphFactory(session_manager)

        # Simulate a multi-agent workflow
        agents = ["full_stack_engineer", "qa_engineer", "devops_engineer"]

        state: AgentState = {
            "task": "Build, test, and deploy a web application",
            "messages": [],
            "next": agents[0],
            "scratchpad": {},
            "mode": "orchestration",
            "active_agents": agents,
            "multimodal_content": {},
            "session_id": "demo-routing-session",
            "user_id": "demo-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        print(f"Initial state: {agents}")
        print(f"Starting with: {state['next']}")

        # Simulate routing through all agents
        for step in range(len(agents) + 1):  # +1 to reach coordinator
            print(f"\nStep {step + 1}:")
            print(f"  Current agent: {state['next']}")

            if state["next"] == "coordinator":
                print("  → Reached coordinator, workflow complete!")
                break

            # Route to next agent
            result = await factory.route_to_appropriate_agents(state)

            routing_info = result["scratchpad"]["routing_info"]
            print(f"  → Next agent: {result['next']}")
            print(f"  → Routing step: {routing_info['current_step']}")
            print(
                f"  → Routing complete: {routing_info.get('routing_complete', False)}"
            )

            state = result

    finally:
        await session_manager.stop()


async def demo_coordination():
    """Demonstrate coordination functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Agent Coordination")
    print("=" * 60)

    session_manager = SessionManager()
    await session_manager.start()

    try:
        factory = OrchestratorGraphFactory(session_manager)

        # Simulate state with multiple agent contributions
        state: AgentState = {
            "task": "Build a comprehensive web application",
            "messages": [],
            "next": "coordinator",
            "scratchpad": {
                "full_stack_engineer_analysis": {
                    "response_length": 1200,
                    "model_used": "gpt-4",
                    "provider": "openai",
                    "timestamp": datetime.now().isoformat(),
                },
                "qa_engineer_analysis": {
                    "response_length": 800,
                    "model_used": "claude-3",
                    "provider": "anthropic",
                    "timestamp": datetime.now().isoformat(),
                },
                "devops_engineer_analysis": {
                    "response_length": 600,
                    "model_used": "gemini-pro",
                    "provider": "google",
                    "timestamp": datetime.now().isoformat(),
                },
            },
            "mode": "orchestration",
            "active_agents": ["full_stack_engineer", "qa_engineer", "devops_engineer"],
            "multimodal_content": {},
            "session_id": "demo-coordination-session",
            "user_id": "demo-user",
            "last_updated": None,
            "errors": [],
            "task_status": {},
        }

        print(f"Task: {state['task']}")
        print(f"Participating agents: {', '.join(state['active_agents'])}")
        print("\nAgent contributions:")
        for agent in state["active_agents"]:
            analysis_key = f"{agent}_analysis"
            if analysis_key in state["scratchpad"]:
                analysis = state["scratchpad"][analysis_key]
                print(
                    f"  • {agent}: {analysis['response_length']} chars using {analysis['model_used']}"
                )

        # Coordinate responses
        result = await factory.coordinate_agent_responses(state)

        print("\nCoordination complete!")
        print(f"Final message added: {len(result['messages'])} messages total")
        print(f"Task status: {result['task_status']['status']}")
        print(
            f"Coordination complete: {result['task_status']['coordination_complete']}"
        )

        # Show part of the synthesized response
        if result["messages"]:
            response = result["messages"][0].content
            print("\nSynthesized response preview:")
            print(f"  {response[:200]}...")

    finally:
        await session_manager.stop()


async def demo_graph_creation():
    """Demonstrate orchestrator graph creation."""
    print("\n" + "=" * 60)
    print("DEMO: Orchestrator Graph Creation")
    print("=" * 60)

    session_manager = SessionManager()
    await session_manager.start()

    try:
        factory = OrchestratorGraphFactory(session_manager)

        # Load agent configurations
        config_dir = Path("agent_configs")
        factory.agent_factory.load_agent_configurations(config_dir)

        print("Available agents:")
        for agent in factory.agent_factory.list_agents():
            config = factory.agent_factory.get_agent_config(agent)
            if config:
                print(f"  • {config.display_name} ({agent})")
                print(f"    - {config.description}")
                print(f"    - Capabilities: {', '.join(config.capabilities)}")

        print("\nCreating orchestrator graph...")

        # Mock agent node creation to avoid model dependencies
        original_create_agent_node = factory.agent_factory.create_agent_node

        async def mock_create_agent_node(agent_name):
            async def mock_agent_node(state: AgentState) -> AgentState:
                print(f"    [Mock] {agent_name} processing task...")
                return state

            return mock_agent_node

        factory.agent_factory.create_agent_node = mock_create_agent_node

        try:
            graph = await factory.create_orchestrator_graph()

            # Analyze graph structure
            graph_dict = graph.get_graph().to_json()
            nodes = [node["id"] for node in graph_dict["nodes"]]
            edges = graph_dict["edges"]

            print("Graph created successfully!")
            print(f"  • Total nodes: {len(nodes)}")
            print(f"  • Total edges: {len(edges)}")

            print("\nOrchestration nodes:")
            orchestration_nodes = ["task_analyzer", "router", "coordinator"]
            for node in orchestration_nodes:
                if node in nodes:
                    print(f"  ✓ {node}")
                else:
                    print(f"  ✗ {node} (missing)")

            print("\nAgent nodes:")
            agent_nodes = [
                n
                for n in nodes
                if n not in orchestration_nodes and n not in ["__start__", "__end__"]
            ]
            for node in agent_nodes:
                print(f"  ✓ {node}")

        finally:
            # Restore original method
            factory.agent_factory.create_agent_node = original_create_agent_node

    finally:
        await session_manager.stop()


async def main():
    """Run all demos."""
    print("🤖 AgentHive Multi-Agent Orchestrator Demo")
    print("=" * 60)

    try:
        await demo_task_analysis()
        await demo_routing_workflow()
        await demo_coordination()
        await demo_graph_creation()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    asyncio.run(main())
