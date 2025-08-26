# AgentHive Integrations

This module provides comprehensive integration capabilities for AgentHive, supporting both **Model Context Protocol (MCP)** for development/testing and **direct Composio SDK integration** for production use.

## Overview

AgentHive integrations enable AI agents to interact with external tools and services through multiple approaches:

1. **MCP Integration** - Mock implementations for development and testing
2. **Composio SDK Integration** - Real API integrations with OAuth authentication
3. **Unified Client** - Seamlessly combines both approaches

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentHive Agents                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Unified Client                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │    MCP Integration      │  │  Composio SDK Integration│   │
│  │   (Development/Test)    │  │     (Production)        │   │
│  └─────────────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              External Services                              │
│  GitHub • Linear • Slack • Notion • Figma • AWS • etc.    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic MCP Integration (Development)

```python
from agent_hive.integrations import get_mcp_client

# Initialize MCP client
client = await get_mcp_client()

# Get tools for an agent type
tools = await client.get_tools_for_agent("full_stack_engineer")

# Execute a tool
result = await client.execute_tool(
    "create_repository",
    {"name": "my-repo", "description": "A new repository"}
)

print(f"Success: {result.success}")
print(f"Result: {result.result}")
```

### 2. Composio SDK Integration (Production)

```python
import os
from agent_hive.integrations import get_composio_client, get_agent_integration

# Set your Composio API key
os.environ["COMPOSIO_API_KEY"] = "your-api-key"

# Initialize Composio client
client = get_composio_client()
integration = get_agent_integration()

# Setup agent tools (requires OAuth)
user_id = "user@example.com"
setup_result = await integration.setup_agent_tools(
    user_id, "full_stack_engineer"
)

# Complete OAuth flows using the provided URLs
for auth in setup_result["authorization_urls"]:
    print(f"Authorize {auth['toolkit']}: {auth['url']}")

# Execute tools after authorization
result = await integration.execute_agent_tool(
    user_id, "full_stack_engineer", "GITHUB_CREATE_REPOSITORY",
    {"name": "production-repo", "description": "Production repository"}
)
```

### 3. Unified Client (Recommended)

```python
from agent_hive.integrations import get_unified_client, IntegrationMode

# Initialize unified client (auto-detects available integrations)
client = await get_unified_client(mode=IntegrationMode.AUTO)

# Get tools from all available sources
tools = await client.get_tools_for_agent("user@example.com", "full_stack_engineer")

# Execute tool (automatically chooses best source)
result = await client.execute_tool(
    "user@example.com", "create_repository",
    {"name": "unified-repo", "description": "Created via unified client"}
)

print(f"Success: {result.success}, Source: {result.source}")
```

## Integration Modes

The unified client supports multiple integration modes:

- **`AUTO`** - Automatically choose the best available integration
- **`MCP_ONLY`** - Use only MCP integration (development/testing)
- **`COMPOSIO_ONLY`** - Use only Composio SDK (production)
- **`HYBRID`** - Prefer Composio SDK, fallback to MCP

## Agent Tool Mappings

AgentHive automatically maps agent types to relevant tool categories:

| Agent Type | Required Tools | Optional Tools |
|------------|----------------|----------------|
| `full_stack_engineer` | GitHub | Linear, Slack, Notion |
| `qa_engineer` | GitHub, Linear | Slack, Browserbase |
| `product_designer` | Figma | Notion, Slack, Miro |
| `devops_engineer` | GitHub, AWS | Docker, Kubernetes, Datadog |
| `data_scientist` | GitHub, Notion | Airtable, SerpAPI |

## Supported Tools & Services

### Development (MCP)
- GitHub (create repos, PRs, issues)
- Linear (create/update issues)
- Slack (send messages, create channels)
- Notion (create pages, search)
- Figma (get files, export images)

### Production (Composio SDK)
- **100+ integrations** including:
  - **Development**: GitHub, GitLab, Bitbucket
  - **Project Management**: Linear, Jira, Asana, Trello
  - **Communication**: Slack, Discord, Microsoft Teams
  - **Documentation**: Notion, Confluence, Google Docs
  - **Design**: Figma, Adobe Creative Suite
  - **Cloud**: AWS, Google Cloud, Azure
  - **Monitoring**: Datadog, New Relic, Sentry
  - **And many more...**

## Configuration

### MCP Configuration

Create `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "uvx",
      "args": ["composio-github-mcp@latest"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      },
      "disabled": false,
      "autoApprove": ["list_issues"]
    }
  }
}
```

### Composio Configuration

Set environment variables:

```bash
export COMPOSIO_API_KEY="your-composio-api-key"
export GITHUB_TOKEN="your-github-token"
export LINEAR_API_KEY="your-linear-api-key"
# ... other service tokens
```

## Authentication

### MCP Integration
- No authentication required (uses mock implementations)
- Perfect for development and testing

### Composio SDK Integration
- Requires Composio API key
- OAuth flows for each service
- Secure token management
- Fine-grained permissions

## Error Handling

The integration layer provides comprehensive error handling:

```python
from agent_hive.integrations import (
    MCPError, ComposioIntegrationError, 
    ComposioAuthenticationError, ComposioToolExecutionError
)

try:
    result = await client.execute_tool("tool_name", {"param": "value"})
except ComposioAuthenticationError:
    print("Authentication failed - check OAuth tokens")
except ComposioToolExecutionError as e:
    print(f"Tool execution failed: {e}")
except MCPError as e:
    print(f"MCP error: {e}")
```

## Custom Tools

Create custom tools for domain-specific needs:

```python
from agent_hive.integrations import get_composio_client

client = get_composio_client()

# Create a standalone custom tool
def calculate_fibonacci(request):
    n = request.n
    # ... fibonacci calculation
    return result

tool_slug = client.create_custom_tool(
    slug="CALCULATE_FIBONACCI",
    name="Calculate Fibonacci",
    description="Calculate nth Fibonacci number",
    input_params={"n": {"type": "integer"}},
    execute_fn=calculate_fibonacci
)

# Execute custom tool
result = await client.execute_tool(
    "user@example.com", tool_slug, {"n": 10}
)
```

## Testing

Run the comprehensive test suite:

```bash
# Test MCP integration
python -m pytest tests/test_mcp_integration.py -v

# Test Composio SDK integration
python -m pytest tests/test_composio_sdk_integration.py -v

# Run all integration tests
python -m pytest tests/test_*integration*.py -v
```

## Examples

### Complete Workflow Example

```python
import asyncio
from agent_hive.integrations import get_unified_client

async def agent_workflow():
    client = await get_unified_client()
    user_id = "developer@company.com"
    
    # Setup agent integrations
    setup = await client.setup_agent_integrations(
        user_id, "full_stack_engineer"
    )
    
    # Execute development workflow
    tasks = [
        ("create_repository", {"name": "new-feature", "description": "New feature branch"}),
        ("create_pull_request", {"title": "Add new feature", "head": "feature", "base": "main"}),
        ("create_issue", {"title": "Test new feature", "priority": 2})
    ]
    
    for tool, args in tasks:
        result = await client.execute_tool(user_id, tool, args)
        print(f"{tool}: {'✅' if result.success else '❌'}")

asyncio.run(agent_workflow())
```

### Demo Scripts

Run the included demo scripts:

```bash
# MCP integration demo
python examples/mcp_integration_demo.py

# Full integration demo (MCP + Composio)
python examples/composio_full_integration_demo.py
```

## Production Deployment

### 1. Environment Setup

```bash
# Install Composio SDK for production
pip install composio

# Set required environment variables
export COMPOSIO_API_KEY="your-api-key"
export GITHUB_TOKEN="your-github-token"
# ... other service tokens
```

### 2. OAuth Flow Setup

```python
from agent_hive.integrations import get_composio_client

client = get_composio_client()

# Initiate OAuth for a user
connection = await client.authorize_toolkit("user@example.com", "github")
print(f"Visit: {connection.redirect_url}")

# Wait for user to complete OAuth
await client.wait_for_connection(connection)
```

### 3. Production Configuration

```python
from agent_hive.integrations import get_unified_client, IntegrationMode

# Use Composio SDK for production
client = await get_unified_client(
    mode=IntegrationMode.COMPOSIO_ONLY,
    composio_api_key="your-api-key"
)
```

## Monitoring & Observability

The integration layer provides comprehensive logging and monitoring:

```python
import logging

# Enable debug logging
logging.getLogger("agent_hive.integrations").setLevel(logging.DEBUG)

# Monitor integration status
status = await client.get_integration_status("user@example.com")
print(f"MCP Available: {status['mcp_available']}")
print(f"Composio Available: {status['composio_available']}")
```

## Best Practices

1. **Development**: Use MCP integration for rapid development and testing
2. **Production**: Use Composio SDK for real integrations with proper authentication
3. **Hybrid**: Use unified client with AUTO mode for seamless transitions
4. **Error Handling**: Always handle authentication and execution errors
5. **Monitoring**: Monitor integration status and tool execution metrics
6. **Security**: Never commit API keys, use environment variables
7. **Testing**: Write comprehensive tests for custom tools and workflows

## Troubleshooting

### Common Issues

1. **Composio SDK not available**
   ```bash
   pip install composio
   ```

2. **Authentication failures**
   - Check API keys and tokens
   - Verify OAuth completion
   - Check token permissions

3. **Tool execution timeouts**
   - Increase timeout settings
   - Check network connectivity
   - Verify service availability

4. **MCP server connection issues**
   - Check MCP server configuration
   - Verify uvx installation
   - Check environment variables

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Add new tool integrations to the appropriate client
2. Update agent tool mappings for new agent types
3. Add comprehensive tests for new functionality
4. Update documentation and examples

## License

This integration layer is part of AgentHive and follows the same license terms.