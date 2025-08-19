# AgentHive Makefile
# Simple commands to work with the application

.PHONY: help install dev-install clean test lint format run verify health docker-build docker-run redis-start redis-stop

# Default target
help:
	@echo "AgentHive Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install production dependencies"
	@echo "  dev-install  Install development dependencies"
	@echo "  clean        Clean build artifacts and cache"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         Run test suite"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  verify       Run setup verification script"
	@echo ""
	@echo "Application Commands:"
	@echo "  run          Start the AgentHive API server"
	@echo "  health       Check application health"
	@echo ""
	@echo "Redis Commands:"
	@echo "  redis-start  Start Redis using Docker"
	@echo "  redis-stop   Stop Redis Docker container"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run application in Docker"

# Setup Commands
install:
	@echo "Installing production dependencies..."
	uv sync --no-dev

dev-install:
	@echo "Installing development dependencies..."
	uv sync --extra dev

clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development Commands
test:
	@echo "Running test suite..."
	uv run pytest -v

test-coverage:
	@echo "Running test suite with coverage..."
	uv run pytest --cov=agent_hive --cov-report=html --cov-report=term

lint:
	@echo "Running linting checks..."
	uv run ruff check .
	uv run mypy agent_hive/

format:
	@echo "Formatting code..."
	uv run black .
	uv run ruff check --fix .

verify:
	@echo "Running setup verification..."
	uv run python verify_setup.py

# Application Commands
run:
	@echo "Starting AgentHive API server..."
	uv run python main.py

run-dev:
	@echo "Starting AgentHive API server in development mode..."
	ENVIRONMENT=development uv run python main.py

health:
	@echo "Checking application health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Server not running or health check failed"

status:
	@echo "Checking application status..."
	@curl -s http://localhost:8000/status | python -m json.tool || echo "Server not running or status check failed"

# Redis Commands
redis-start:
	@echo "Starting Redis using Docker..."
	docker run -d --name agent-hive-redis -p 6379:6379 redis:7-alpine
	@echo "Redis started on port 6379"

redis-stop:
	@echo "Stopping Redis Docker container..."
	docker stop agent-hive-redis || true
	docker rm agent-hive-redis || true

redis-cli:
	@echo "Connecting to Redis CLI..."
	docker exec -it agent-hive-redis redis-cli

# Docker Commands
docker-build:
	@echo "Building Docker image..."
	docker build -t agent-hive:latest .

docker-run:
	@echo "Running application in Docker..."
	docker run -p 8000:8000 --env-file .env agent-hive:latest

# Environment setup
setup-env:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please edit .env with your API keys"; \
	else \
		echo ".env file already exists"; \
	fi

# Complete setup for new developers
setup: setup-env dev-install redis-start verify
	@echo ""
	@echo "🎉 Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run 'make run' to start the server"
	@echo "3. Visit http://localhost:8000/docs for API documentation"

# Quick development workflow
dev: format lint test
	@echo "Development checks complete!"

# Production deployment preparation
prod-check: clean install lint test
	@echo "Production readiness checks complete!"

# Show current configuration
config:
	@echo "Current Configuration:"
	@echo "====================="
	@echo "Python version: $(shell python --version)"
	@echo "UV version: $(shell uv --version)"
	@echo "Project root: $(shell pwd)"
	@echo "Virtual env: $(shell uv run python -c 'import sys; print(sys.prefix)')"
	@echo ""
	@echo "Dependencies:"
	@uv tree --depth 1