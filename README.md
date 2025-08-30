# AgentHive [under active development]

Unified multi-agent system built on LangGraph that provides both direct interaction with specialized AI agents and autonomous task delegation to a coordinated team of agents.

## Features

- **Dual Operation Modes**: Direct agent interaction and autonomous orchestration
- **Multi-Model Support**: OpenAI, Anthropic Claude, and Google Gemini
- **Specialized Agents**: Full Stack Engineer, QA Engineer, Product Designer, DevOps Engineer
- **Multimodal Processing**: Support for images, PDFs, and design documents
- **Multiple Interfaces**: Web UI and Slack integration
- **Extensible Architecture**: Built on LangGraph with MCP tool integration

## Quick Start

### Prerequisites

- Python 3.11+
- uv (Python package manager)
- Redis (for state management)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-hive
```

2. Quick setup (recommended):
```bash
make setup
```

This will:
- Copy `.env.example` to `.env`
- Install all dependencies
- Start Redis using Docker
- Run verification checks

3. Manual setup:
```bash
# Install dependencies
make dev-install

# Copy environment configuration
make setup-env

# Configure your API keys in .env
# Edit .env with your API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Start Redis
make redis-start

# Verify setup
make verify
```

4. Run the application:
```bash
make run
```

The API will be available at `http://localhost:8000`

### API Documentation

- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- System status: `http://localhost:8000/status`

## Development

### Available Commands

Run `make help` to see all available commands:

```bash
make help
```

Common development commands:
```bash
make dev-install    # Install development dependencies
make test          # Run test suite
make lint          # Run linting checks
make format        # Format code
make run           # Start the server
make health        # Check server health
make redis-start   # Start Redis
make verify        # Verify setup
```

### Project Structure

```
agent_hive/
├── agent_hive/
│   ├── core/           # Core data structures and configuration
│   ├── agents/         # Agent implementations
│   ├── api/            # FastAPI application and routing
│   ├── integrations/   # External service integrations
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── main.py             # Application entry point
├── Makefile           # Development commands
└── pyproject.toml      # Project configuration
```

### Development Workflow

```bash
# Format and check code
make dev

# Run tests with coverage
make test-coverage

# Check production readiness
make prod-check
```

## License

[Add your license here]
