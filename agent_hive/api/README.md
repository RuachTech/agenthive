# AgentHive API Architecture

This directory contains the refactored AgentHive API implementation, organized following SOLID principles for better maintainability and readability.

## Structure Overview

```
agent_hive/api/
├── __init__.py          # Module exports
├── main.py              # FastAPI app and route definitions (~300 lines)
├── core.py              # Core AgentHiveAPI class and initialization
├── chat.py              # Chat and orchestration services
├── files.py             # File upload and multimodal processing
├── agents.py            # Agent discovery and tools management
├── sessions.py          # Session management operations
├── health.py            # Health checks and system status
└── validation.py        # Request validation and error handling
```

## Design Principles

### Single Responsibility Principle (SRP)
Each service class has a single, well-defined responsibility:
- `ChatService`: Handles direct chat and task orchestration
- `FileService`: Manages file uploads and multimodal content
- `AgentService`: Provides agent discovery and tools information
- `SessionService`: Manages session lifecycle operations
- `HealthService`: Handles health checks and system status

### Dependency Injection
Services receive the `AgentHiveAPI` instance as a dependency, allowing for:
- Easy testing with mock dependencies
- Clear separation of concerns
- Flexible configuration and initialization

### Open/Closed Principle
The architecture allows for extension without modification:
- New services can be added without changing existing code
- New endpoints can be added by creating new service methods
- Validation and error handling are centralized and reusable

## Service Classes

### ChatService (`chat.py`)
Handles conversational interactions:
- `direct_chat()` - Direct agent communication
- `orchestrate_task()` - Multi-agent task coordination
- File processing integration
- Streaming response support

### FileService (`files.py`)
Manages multimodal content:
- `upload_multimodal_content()` - File upload and processing
- `get_supported_formats()` - Available file format information
- Validation and error handling for uploads

### AgentService (`agents.py`)
Provides agent information:
- `get_agent_status()` - Agent availability and health
- `get_available_tools()` - Tool discovery per agent
- `get_available_agents()` - Complete agent listing

### SessionService (`sessions.py`)
Manages user sessions:
- `create_session()` - New session creation
- `delete_session()` - Session cleanup
- `list_user_sessions()` - User session management
- `get_session_info()` - Detailed session information

### HealthService (`health.py`)
System monitoring:
- `health_check()` - Comprehensive health status
- `get_system_status()` - Detailed system information
- `get_api_info()` - API metadata and endpoints

## Benefits of Refactoring

### Maintainability
- **Reduced file size**: Main file reduced from 1000+ to ~300 lines
- **Logical organization**: Related functionality grouped together
- **Clear boundaries**: Each service has well-defined responsibilities

### Testability
- **Isolated testing**: Services can be tested independently
- **Mock-friendly**: Dependency injection enables easy mocking
- **Focused tests**: Each service can have targeted test suites

### Readability
- **Clear structure**: Easy to find specific functionality
- **Consistent patterns**: All services follow the same design
- **Self-documenting**: Service names clearly indicate purpose

### Extensibility
- **Easy additions**: New services can be added without modification
- **Flexible routing**: Endpoints can be easily reorganized
- **Reusable components**: Services can be composed in different ways

## Usage Examples

### Creating Services
```python
from agent_hive.api import AgentHiveAPI
from agent_hive.api.chat import ChatService

api = AgentHiveAPI()
await api.initialize()

chat_service = ChatService(api)
result = await chat_service.direct_chat(
    agent_name="full_stack_engineer",
    message="Hello, world!"
)
```

### FastAPI Integration
```python
from agent_hive.api import create_app

app = create_app()
# Services are automatically injected via Depends()
```

### Testing Services
```python
from unittest.mock import Mock
from agent_hive.api.agents import AgentService

# Mock the API instance
mock_api = Mock()
mock_api.agent_factory.list_agents.return_value = ["test_agent"]

# Test the service
agent_service = AgentService(mock_api)
result = await agent_service.get_available_agents()
```

## Error Handling

All services use the centralized error handling from `validation.py`:
- Consistent error responses across all endpoints
- Proper HTTP status codes
- Structured error information
- Request validation and sanitization

## Future Enhancements

The modular structure supports easy addition of:
- New service classes for additional functionality
- Enhanced validation rules per service
- Service-specific middleware
- Independent service scaling
- Microservice decomposition if needed

This architecture provides a solid foundation for the AgentHive API that can grow and evolve while maintaining clean, maintainable code.