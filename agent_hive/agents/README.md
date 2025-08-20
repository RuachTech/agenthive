# AgentHive Agent Factory System

The Agent Factory System is the core component responsible for creating, managing, and executing specialized AI agents within the AgentHive platform. It provides a standardized way to define agent capabilities, handle errors, and create LangGraph-compatible agent nodes.

## Overview

The system consists of several key components:

- **AgentFactory**: Main factory class for creating and managing agents
- **AgentNodeWrapper**: Wrapper that handles state updates and error recovery
- **AgentCapabilities**: Defines what capabilities an agent has and requires
- **ErrorRecoveryStrategy**: Handles various error scenarios during agent execution
- **Configuration System**: Loads agent definitions from JSON files

## Key Features

### 1. Standardized Agent Creation

```python
from agent_hive.agents import get_agent_factory, AgentCapabilities
from agent_hive.core.config import AgentConfig

# Create agent configuration
config = AgentConfig(
    name="my_agent",
    display_name="My Specialized Agent",
    description="An agent specialized for specific tasks",
    system_prompt="You are a helpful specialized agent...",
    model_provider="openai",
    model_name="gpt-4",
    capabilities=["code_execution", "web_search"]
)

# Define capabilities
capabilities = AgentCapabilities(
    required_capabilities=["code_execution"],
    optional_capabilities=["web_search"],
    model_requirements=["function_calling"],
    required_tools=["code_executor"],
    mcp_requirements=["github"]
)

# Register and create agent
factory = get_agent_factory()
factory.register_agent_config(config, capabilities, tools)
agent_node = await factory.create_agent_node("my_agent")
```

### 2. Error Handling and Recovery

The system provides comprehensive error handling for common scenarios:

- **Model Timeouts**: Automatic fallback to alternative models
- **Tool Failures**: Graceful degradation with alternative approaches
- **Validation Errors**: Clear error reporting and recovery suggestions

```python
from agent_hive.agents import ErrorRecoveryStrategy

strategy = ErrorRecoveryStrategy()

# Handle model timeout
recovered_state = await strategy.handle_model_timeout(state, "agent_name")

# Handle tool failure
recovered_state = await strategy.handle_tool_failure(
    state, "agent_name", "tool_name", error
)
```

### 3. Configuration File Loading

Agents can be defined in JSON configuration files for easy management:

```json
{
  "config": {
    "name": "full_stack_engineer",
    "display_name": "Full Stack Engineer",
    "description": "Specialized agent for full-stack development",
    "system_prompt": "You are a Full Stack Engineer AI agent...",
    "model_provider": "openai",
    "model_name": "gpt-4",
    "capabilities": ["code_execution", "file_system"],
    "specialized_for": ["backend_development", "frontend_development"]
  },
  "capabilities": {
    "required_capabilities": ["code_execution"],
    "optional_capabilities": ["file_system", "web_search"],
    "model_requirements": ["function_calling"],
    "required_tools": ["code_executor"],
    "mcp_requirements": ["github"]
  }
}
```

Load configurations from a directory:

```python
from pathlib import Path

factory = get_agent_factory()
factory.load_agent_configurations(Path("agent_configs"))
```

### 4. Agent Validation

Validate that agents can be created and executed:

```python
validation_result = await factory.validate_agent("agent_name")

if validation_result["valid"]:
    print("Agent is ready to use")
else:
    print(f"Validation errors: {validation_result['errors']}")
```

## Agent State Management

All agents work with a shared `AgentState` structure that maintains context across interactions:

```python
from agent_hive.core.state import AgentState
from langchain_core.messages import HumanMessage

state = AgentState(
    task="Primary task description",
    messages=[HumanMessage(content="User input")],
    next="",  # Next agent to route to
    scratchpad={},  # Agent working memory
    mode="direct",  # "direct" or "orchestration"
    active_agents=[],  # Currently involved agents
    multimodal_content={},  # Processed files and media
    session_id="unique_session_id",
    user_id="user_identifier",
    last_updated=datetime.now(),
    errors=[],  # Error tracking
    task_status={}  # Task completion tracking
)
```

## Specialized Agent Types

The system includes pre-configured specialized agents:

### Full Stack Engineer
- **Capabilities**: Code generation, architecture design, debugging
- **Tools**: Code execution, file system access, git operations
- **Integrations**: GitHub, Linear, Slack, Notion

### QA Engineer
- **Capabilities**: Test planning, bug analysis, quality assessment
- **Tools**: Test frameworks, browser automation, performance testing
- **Integrations**: GitHub, Linear, Browserbase, SerpAPI

### Product Designer
- **Capabilities**: UI/UX design, wireframing, accessibility review
- **Tools**: Image analysis, design validation, color analysis
- **Integrations**: Figma, Notion, Miro, Airtable

### DevOps Engineer
- **Capabilities**: Infrastructure design, deployment automation, security
- **Tools**: Infrastructure management, security scanning, monitoring
- **Integrations**: AWS, GitHub, Docker, Kubernetes, Datadog

## Usage Examples

### Creating a Simple Agent

```python
import asyncio
from agent_hive.agents import get_agent_factory, AgentCapabilities
from agent_hive.core.config import AgentConfig
from agent_hive.core.state import AgentState
from langchain_core.messages import HumanMessage

async def create_and_use_agent():
    # Setup
    factory = get_agent_factory()
    
    config = AgentConfig(
        name="helper_agent",
        display_name="Helper Agent",
        description="A general purpose helper agent",
        system_prompt="You are a helpful assistant.",
        model_provider="openai",
        model_name="gpt-3.5-turbo"
    )
    
    capabilities = AgentCapabilities(
        required_capabilities=[],
        optional_capabilities=[],
        model_requirements=[],
        required_tools=[],
        mcp_requirements=[]
    )
    
    # Register and create
    factory.register_agent_config(config, capabilities)
    agent_node = await factory.create_agent_node("helper_agent")
    
    # Use the agent
    state = AgentState(
        task="Help me understand Python decorators",
        messages=[HumanMessage(content="Explain Python decorators")],
        next="",
        scratchpad={},
        mode="direct",
        active_agents=[],
        multimodal_content={},
        session_id="demo_session"
    )
    
    result_state = await agent_node(state)
    print(f"Agent response: {result_state['messages'][-1].content}")

# Run the example
asyncio.run(create_and_use_agent())
```

### Loading Multiple Agents from Configuration

```python
from pathlib import Path
from agent_hive.agents import get_agent_factory

def load_all_agents():
    factory = get_agent_factory()
    
    # Load all agent configurations
    config_dir = Path("agent_configs")
    factory.load_agent_configurations(config_dir)
    
    # List available agents
    agents = factory.list_agents()
    print(f"Available agents: {agents}")
    
    # Validate each agent
    for agent_name in agents:
        result = await factory.validate_agent(agent_name)
        status = "✓" if result["valid"] else "✗"
        print(f"{status} {agent_name}: {result['valid']}")

asyncio.run(load_all_agents())
```

## Testing

The agent factory system includes comprehensive unit tests covering:

- Agent configuration and registration
- Agent node creation and execution
- Error handling and recovery
- Configuration file loading
- Validation and capability checking

Run tests with:

```bash
python -m pytest tests/test_agent_factory.py -v
```

## Integration with LangGraph

Agent nodes created by the factory are fully compatible with LangGraph workflows:

```python
from langgraph.graph import StateGraph, END

# Create workflow
workflow = StateGraph(AgentState)

# Add agent nodes
agent_node = await factory.create_agent_node("full_stack_engineer")
workflow.add_node("engineer", agent_node)

# Define workflow
workflow.set_entry_point("engineer")
workflow.add_edge("engineer", END)

# Compile and use
graph = workflow.compile()
result = await graph.ainvoke(initial_state)
```

## Best Practices

1. **Agent Specialization**: Create focused agents with clear specializations
2. **Error Handling**: Always implement proper error recovery strategies
3. **Validation**: Validate agent configurations before deployment
4. **Testing**: Write comprehensive tests for custom agents
5. **Configuration Management**: Use JSON files for agent definitions
6. **Capability Mapping**: Clearly define required vs optional capabilities
7. **Tool Integration**: Ensure all required tools are available and tested

## Troubleshooting

### Common Issues

1. **Agent Not Found**: Ensure agent is registered before creating nodes
2. **Model Unavailable**: Check model configuration and API keys
3. **Tool Missing**: Verify all required tools are installed and configured
4. **Validation Errors**: Review capability requirements and availability

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about agent creation, validation, and execution processes.