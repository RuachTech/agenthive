"""
Demo script showing single-agent direct chat graph functionality.

This example demonstrates:
1. Creating and configuring agents
2. Building direct chat graphs
3. Session management
4. Direct chat execution
5. Graph validation
"""

import asyncio
import logging
import os
from datetime import datetime

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

from agent_hive.core.graphs import DirectChatGraphFactory, SessionManager
from agent_hive.core.config import AgentConfig, SystemConfig
from agent_hive.agents.factory import AgentFactory, AgentCapabilities
from agent_hive.core.models import ModelFactory, ModelConfig, ModelProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_demo_environment():
    """Set up the demo environment with agents and models."""
    logger.info("Setting up demo environment...")
    
    # Check for required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        logger.warning("No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
        logger.info("For demo purposes, we'll continue with mock setup.")
        return await setup_mock_environment()
    
    # Create model factory and register real models
    model_factory = ModelFactory()
    
    # Register OpenAI model if key is available
    if openai_key:
        openai_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-5-nano-2025-08-07",  # Using mini for cost efficiency
            api_key=openai_key,
            temperature=0.7,
            max_tokens=2000
        )
        model_factory.register_model("openai_model", openai_config)
        logger.info("Registered OpenAI model: gpt-4o-mini")
    
    # Register Anthropic model if key is available
    if anthropic_key:
        anthropic_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",  # Using haiku for cost efficiency
            api_key=anthropic_key,
            temperature=0.7,
            max_tokens=2000
        )
        model_factory.register_model("anthropic_model", anthropic_config)
        logger.info("Registered Anthropic model: claude-3-haiku-20240307")
    
    # Create agent factory
    agent_factory = AgentFactory(model_factory)
    
    # Create multiple demo agents with different models
    agents_created = []
    
    if openai_key:
        # OpenAI-powered assistant
        openai_agent_config = AgentConfig(
            name="openai_assistant",
            display_name="OpenAI Assistant",
            description="A helpful assistant powered by OpenAI's GPT-4o-mini",
            system_prompt="You are a helpful and concise assistant. Respond clearly and helpfully to user questions. Keep responses focused and practical.",
            model_provider="openai",
            model_name="gpt-5-nano-2025-08-07",
            capabilities=["chat", "reasoning", "analysis"],
            specialized_for=["general_assistance", "problem_solving"]
        )
        
        openai_capabilities = AgentCapabilities(
            required_capabilities=["chat"],
            optional_capabilities=["reasoning", "analysis"],
            model_requirements=[],
            required_tools=[],
            mcp_requirements=[]
        )
        
        agent_factory.register_agent_config(openai_agent_config, openai_capabilities)
        agents_created.append("openai_assistant")
    
    if anthropic_key:
        # Anthropic-powered assistant
        anthropic_agent_config = AgentConfig(
            name="claude_assistant",
            display_name="Claude Assistant",
            description="A thoughtful assistant powered by Anthropic's Claude-3-Haiku",
            system_prompt="You are Claude, a thoughtful and analytical assistant. Provide detailed, well-reasoned responses while being concise. Focus on being helpful and accurate.",
            model_provider="anthropic",
            model_name="claude-3-haiku-20240307",
            capabilities=["chat", "reasoning", "analysis", "writing"],
            specialized_for=["analysis", "writing", "detailed_explanations"]
        )
        
        claude_capabilities = AgentCapabilities(
            required_capabilities=["chat"],
            optional_capabilities=["reasoning", "analysis", "writing"],
            model_requirements=[],
            required_tools=[],
            mcp_requirements=[]
        )
        
        agent_factory.register_agent_config(anthropic_agent_config, claude_capabilities)
        agents_created.append("claude_assistant")
    
    logger.info(f"Demo environment setup complete with {len(agents_created)} agents: {agents_created}")
    return agent_factory, agents_created


async def setup_mock_environment():
    """Set up a mock environment when no API keys are available."""
    logger.info("Setting up mock environment (no API keys provided)...")
    
    # Create model factory
    model_factory = ModelFactory()
    
    # Create agent factory
    agent_factory = AgentFactory(model_factory)
    
    # Define a mock agent configuration
    mock_agent_config = AgentConfig(
        name="mock_assistant",
        display_name="Mock Assistant",
        description="A mock assistant for testing when no API keys are available",
        system_prompt="You are a helpful assistant (mock mode).",
        model_provider="openai",
        model_name="gpt-4",
        capabilities=["chat", "reasoning"],
        specialized_for=["general_assistance", "demo"]
    )
    
    # Define agent capabilities
    mock_capabilities = AgentCapabilities(
        required_capabilities=["chat"],
        optional_capabilities=["reasoning"],
        model_requirements=[],
        required_tools=[],
        mcp_requirements=[]
    )
    
    # Register the agent
    agent_factory.register_agent_config(mock_agent_config, mock_capabilities)
    
    logger.info("Mock environment setup complete")
    return agent_factory, ["mock_assistant"]


async def demo_session_management():
    """Demonstrate session management functionality."""
    logger.info("\n=== Session Management Demo ===")
    
    # Create session manager
    config = SystemConfig(session_timeout=3600)  # 1 hour timeout
    session_manager = SessionManager(config)
    await session_manager.start()
    
    try:
        # Create a new session
        session = await session_manager.create_session(
            session_id="demo_session_001",
            user_id="demo_user",
            mode="direct",
            active_agent="demo_assistant",
            initial_task="Hello, I'd like to test the chat functionality"
        )
        
        logger.info(f"Created session: {session.session_id}")
        logger.info(f"Session mode: {session.mode}")
        logger.info(f"Active agent: {session.active_agent}")
        logger.info(f"Initial task: {session.state['task']}")
        
        # Retrieve the session
        retrieved_session = await session_manager.get_session("demo_session_001")
        assert retrieved_session is not None
        logger.info("Successfully retrieved session")
        
        # Update session state
        retrieved_session.state["scratchpad"]["demo_key"] = "demo_value"
        success = await session_manager.update_session_state(
            "demo_session_001", 
            retrieved_session.state
        )
        assert success
        logger.info("Successfully updated session state")
        
        # List sessions
        sessions = await session_manager.list_sessions("demo_user")
        logger.info(f"Found {len(sessions)} sessions for demo_user")
        
        return session_manager
        
    except Exception as e:
        logger.error(f"Session management demo failed: {e}")
        await session_manager.stop()
        raise


async def demo_graph_creation(agent_factory, available_agents):
    """Demonstrate graph creation and validation."""
    logger.info("\n=== Graph Creation Demo ===")
    
    # Create session manager
    session_manager = SessionManager()
    await session_manager.start()
    
    try:
        # Create graph factory
        graph_factory = DirectChatGraphFactory(session_manager)
        graph_factory.agent_factory = agent_factory
        
        logger.info("Creating and validating graphs for available agents...")
        
        for agent_name in available_agents:
            logger.info(f"\nProcessing agent: {agent_name}")
            
            # Show agent validation
            logger.info("Step 1: Agent validation")
            validation_result = await agent_factory.validate_agent(agent_name)
            logger.info(f"  Agent validation result: {validation_result['valid']}")
            
            if not validation_result['valid']:
                logger.info(f"  Validation errors: {validation_result['errors']}")
                continue
            
            # Try to create the graph (this will work with real API keys)
            logger.info("Step 2: Graph creation")
            try:
                graph = await graph_factory.create_direct_chat_graph(agent_name)
                logger.info(f"  ✓ Successfully created graph for {agent_name}")
                
                # Validate graph structure
                validation_result = graph_factory.validate_graph_structure(graph, agent_name)
                logger.info(f"  Graph structure validation: {validation_result['valid']}")
                
                if validation_result["valid"]:
                    info = validation_result["info"]
                    logger.info(f"    - Nodes: {info['node_count']}")
                    logger.info(f"    - Edges: {info['edge_count']}")
                    logger.info(f"    - Has checkpointer: {info['has_checkpointer']}")
                
            except Exception as e:
                logger.warning(f"  ⚠ Graph creation failed for {agent_name}: {str(e)}")
                logger.info("    This is expected if no API keys are provided")
        
        # Show available agents summary
        logger.info("\nStep 3: Available agents summary")
        agents = agent_factory.list_agents()
        logger.info(f"  Total registered agents: {len(agents)}")
        
        for agent_name in agents:
            config = agent_factory.get_agent_config(agent_name)
            if config:
                logger.info(f"    - {config.display_name}: {config.description}")
        
        return graph_factory
        
    except Exception as e:
        logger.error(f"Graph creation demo failed: {e}")
        await session_manager.stop()
        raise


async def demo_direct_chat_execution(graph_factory, available_agents):
    """Demonstrate direct chat execution."""
    logger.info("\n=== Direct Chat Execution Demo ===")
    
    # Check if we have API keys for real execution
    has_api_keys = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if not has_api_keys:
        logger.info("No API keys found - showing mock execution structure...")
        await demo_mock_execution()
        return
    
    # Try real execution with available agents
    logger.info("Attempting real chat execution with available agents...")
    
    for agent_name in available_agents[:1]:  # Test with first available agent
        logger.info(f"\nTesting direct chat with {agent_name}...")
        
        try:
            # Test regular execution
            logger.info("Executing direct chat...")
            result = await graph_factory.execute_direct_chat(
                session_id=f"demo_chat_{agent_name}",
                agent_name=agent_name,
                message="Hello! Can you briefly explain what nutrients are responsible for a toddlers bone strength and muscular health?",
                user_id="demo_user"
            )
            
            logger.info("✓ Direct chat execution successful!")
            logger.info(f"  Status: {result['status']}")
            logger.info(f"  Session ID: {result['session_id']}")
            logger.info(f"  Agent: {result['agent_name']}")
            logger.info(f"  Response: {result['response'][:200]}...")
            
            if result.get('state'):
                state = result['state']
                logger.info(f"  Message count: {state.get('message_count', 'N/A')}")
                logger.info(f"  Active agents: {state.get('active_agents', [])}")
                if state.get('errors'):
                    logger.warning(f"  Errors: {len(state['errors'])}")
            
            # Test follow-up message
            logger.info("\nTesting follow-up message...")
            followup_result = await graph_factory.execute_direct_chat(
                session_id=f"demo_chat_{agent_name}",
                agent_name=agent_name,
                message="What's 2+2-2?",
                user_id="demo_user"
            )
            
            logger.info("✓ Follow-up message successful!")
            logger.info(f"  Response: {followup_result['response']}")
            
            # Test streaming execution
            logger.info("\nTesting streaming chat...")
            stream_chunks = []
            
            async for chunk in graph_factory.stream_direct_chat(
                session_id=f"demo_stream_{agent_name}",
                agent_name=agent_name,
                message="Count from 1 to 5, one number per line.",
                user_id="demo_user"
            ):
                stream_chunks.append(chunk)
                if chunk["type"] == "content":
                    print(f"Stream: {chunk['content']}", end="", flush=True)
                elif chunk["type"] == "complete":
                    print(f"\n✓ Streaming complete for session: {chunk['session_id']}")
            
            logger.info(f"Received {len(stream_chunks)} stream chunks")
            break  # Success with one agent is enough for demo
            
        except Exception as e:
            logger.error(f"Chat execution failed for {agent_name}: {str(e)}")
            logger.info("This might be due to API rate limits or network issues")
            continue
    
    else:
        logger.warning("All agents failed - falling back to mock execution")
        await demo_mock_execution()


async def demo_mock_execution():
    """Show mock execution when real API calls aren't available."""
    logger.info("Showing mock execution structure...")
    
    mock_result = {
        "response": "Hello! I'm an AI assistant. This system allows you to chat directly with specialized AI agents. Each agent has specific capabilities and can maintain conversation context through sessions.",
        "session_id": "demo_chat_001",
        "agent_name": "demo_assistant",
        "status": "success",
        "state": {
            "message_count": 2,
            "active_agents": ["demo_assistant"],
            "last_updated": datetime.now(),
            "errors": []
        }
    }
    
    logger.info("Mock chat execution result:")
    logger.info(f"  Response: {mock_result['response'][:100]}...")
    logger.info(f"  Status: {mock_result['status']}")
    logger.info(f"  Session ID: {mock_result['session_id']}")
    logger.info(f"  Message count: {mock_result['state']['message_count']}")
    
    # Demonstrate streaming structure
    logger.info("\nMock streaming structure:")
    mock_chunks = [
        {"type": "content", "content": "Hello! ", "session_id": "demo_stream_001"},
        {"type": "content", "content": "I can help you ", "session_id": "demo_stream_001"},
        {"type": "content", "content": "with various tasks.", "session_id": "demo_stream_001"},
        {"type": "complete", "session_id": "demo_stream_001", "agent_name": "demo_assistant"}
    ]
    
    for chunk in mock_chunks:
        logger.info(f"  {chunk}")


async def demo_error_handling(graph_factory):
    """Demonstrate error handling capabilities."""
    logger.info("\n=== Error Handling Demo ===")
    
    try:
        # Test validation error
        logger.info("Testing validation error handling...")
        try:
            await graph_factory.create_direct_chat_graph("nonexistent_agent")
        except Exception as e:
            logger.info(f"Caught expected validation error: {type(e).__name__}")
        
        # Test graph validation with invalid structure
        logger.info("Testing graph structure validation...")
        from unittest.mock import Mock
        
        mock_graph = Mock()
        mock_graph.get_graph.return_value.to_json.return_value = {
            "nodes": [{"id": "wrong_agent"}],
            "edges": []
        }
        
        validation_result = graph_factory.validate_graph_structure(mock_graph, "expected_agent")
        logger.info(f"Invalid graph validation result: {validation_result['valid']}")
        logger.info(f"Validation errors: {validation_result['errors']}")
        
        logger.info("Error handling demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error handling demo failed: {e}")
        raise


async def main():
    """Run the complete demo."""
    logger.info("Starting AgentHive Direct Chat Graph Demo")
    logger.info("=" * 50)
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        logger.info("✓ OpenAI API key found")
    if anthropic_key:
        logger.info("✓ Anthropic API key found")
    
    if not openai_key and not anthropic_key:
        logger.warning("⚠ No API keys found - demo will run in mock mode")
        logger.info("To test with real models, set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
    
    try:
        # Setup
        setup_result = await setup_demo_environment()
        if isinstance(setup_result, tuple):
            agent_factory, available_agents = setup_result
        else:
            agent_factory, available_agents = setup_result, ["mock_assistant"]
        
        # Demo session management
        session_manager = await demo_session_management()
        
        # Demo graph creation
        graph_factory = await demo_graph_creation(agent_factory, available_agents)
        
        # Demo direct chat execution
        await demo_direct_chat_execution(graph_factory, available_agents)
        
        # Demo error handling
        await demo_error_handling(graph_factory)
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        logger.info("\nKey features demonstrated:")
        logger.info("✓ Session management with automatic cleanup")
        logger.info("✓ Direct chat graph creation and validation")
        logger.info("✓ Graph caching for performance")
        logger.info("✓ Agent availability checking")
        logger.info("✓ Error handling and recovery")
        logger.info("✓ State persistence across interactions")
        
        if openai_key or anthropic_key:
            logger.info("✓ Real AI model integration")
            logger.info("✓ Actual conversation execution")
            logger.info("✓ Streaming response handling")
        
        # Cleanup
        await session_manager.stop()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())