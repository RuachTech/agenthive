#!/usr/bin/env python3
"""Verification script for AgentHive foundation setup."""

import sys
from datetime import datetime


def verify_imports():
    """Verify all core imports work correctly."""
    try:
        from agent_hive.core.state import AgentState  # noqa: F401
        from agent_hive.core.config import AgentConfig, SystemConfig  # noqa: F401
        from agent_hive.api import app  # noqa: F401

        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def verify_agent_state():
    """Verify AgentState can be created with all required fields."""
    try:
        from agent_hive.core.state import AgentState

        state: AgentState = {
            "task": "Test verification task",
            "messages": [],
            "next": "test_agent",
            "scratchpad": {"test": "data"},
            "mode": "direct",
            "active_agents": ["test_agent"],
            "multimodal_content": {},
            "session_id": "verify_123",
            "user_id": "test_user",
            "last_updated": datetime.utcnow(),
            "errors": [],
            "task_status": {"status": "in_progress"},
        }

        assert state["task"] == "Test verification task"
        assert state["mode"] == "direct"
        assert len(state["active_agents"]) == 1
        print("✓ AgentState creation and validation successful")
        return True
    except Exception as e:
        print(f"✗ AgentState error: {e}")
        return False


def verify_agent_config():
    """Verify AgentConfig can be created and validated."""
    try:
        from agent_hive.core.config import AgentConfig

        config = AgentConfig(
            name="test_agent",
            display_name="Test Agent",
            description="Test agent for verification",
            system_prompt="You are a test agent.",
            model_provider="openai",
            model_name="gpt-4",
            capabilities=["vision", "code_execution"],
            specialized_for=["testing"],
        )

        assert config.name == "test_agent"
        assert config.model_provider == "openai"
        assert config.enabled is True
        print("✓ AgentConfig creation and validation successful")
        return True
    except Exception as e:
        print(f"✗ AgentConfig error: {e}")
        return False


def verify_system_config():
    """Verify SystemConfig can be created and validated."""
    try:
        from agent_hive.core.config import SystemConfig

        config = SystemConfig(
            api_host="127.0.0.1", api_port=8000, redis_url="redis://localhost:6379"
        )

        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8000
        assert config.session_timeout == 3600  # default
        print("✓ SystemConfig creation and validation successful")
        return True
    except Exception as e:
        print(f"✗ SystemConfig error: {e}")
        return False


def verify_fastapi_app():
    """Verify FastAPI app can be created and basic endpoints work."""
    try:
        from fastapi.testclient import TestClient
        from agent_hive.api import app

        client = TestClient(app)

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AgentHive API"

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

        # Test status endpoint
        response = client.get("/status")
        assert response.status_code == 200
        status_data = response.json()
        assert "system" in status_data

        print("✓ FastAPI app and endpoints working correctly")
        return True
    except Exception as e:
        print(f"✗ FastAPI app error: {e}")
        return False


def verify_project_files():
    """Verify essential project files exist."""
    try:
        import os

        required_files = [
            "pyproject.toml",
            ".gitignore",
            "Makefile",
            "Dockerfile",
            ".dockerignore",
            ".env.example",
            "README.md",
            "main.py",
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            print(f"✗ Missing files: {', '.join(missing_files)}")
            return False

        print("✓ All essential project files present")
        return True
    except Exception as e:
        print(f"✗ Project files check error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("AgentHive Foundation Setup Verification")
    print("=" * 40)

    checks = [
        verify_project_files,
        verify_imports,
        verify_agent_state,
        verify_agent_config,
        verify_system_config,
        verify_fastapi_app,
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        if check():
            passed += 1
        print()

    print("=" * 40)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All foundation components verified successfully!")
        print("\nNext steps:")
        print("1. Run 'make setup' for complete setup")
        print("2. Or manually: 'make setup-env && make redis-start'")
        print("3. Configure API keys in .env file")
        print("4. Run 'make run' to start the application")
        print("5. Visit http://localhost:8000/docs for API documentation")
        print("\nAvailable commands: run 'make help' for full list")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
