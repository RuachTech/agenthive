"""Integration tests for multi-agent orchestrator graph - simplified version."""

from datetime import datetime
from pathlib import Path

import pytest

from agent_hive.core.graphs import OrchestratorGraphFactory, SessionManager
from agent_hive.core.state import AgentState


class TestOrchestratorIntegration:
    """Integration tests for orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_task_analysis_development_task(self) -> None:
        """Test task analysis identifies development agents correctly."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            state: AgentState = {
                "task": "Build a REST API for user management",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": [],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.analyze_task_requirements(state)

            # Should identify full stack engineer for development task
            assert "full_stack_engineer" in result["active_agents"]
            assert result["next"] == "full_stack_engineer"

            # Check task analysis in scratchpad
            task_analysis = result["scratchpad"]["task_analysis"]
            assert (
                task_analysis["original_task"] == "Build a REST API for user management"
            )
            assert "full_stack_engineer" in task_analysis["required_agents"]

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_task_analysis_testing_task(self) -> None:
        """Test task analysis identifies QA agents correctly."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            state: AgentState = {
                "task": "Create comprehensive test suite for the application",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": [],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.analyze_task_requirements(state)

            # Should identify QA engineer for testing task
            assert "qa_engineer" in result["active_agents"]

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_task_analysis_complex_task(self) -> None:
        """Test task analysis identifies multiple agents for complex tasks."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            state: AgentState = {
                "task": "Build, test, and deploy a new web application with proper UI design",
                "messages": [],
                "next": "",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": [],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.analyze_task_requirements(state)

            # Should identify multiple agents
            assert len(result["active_agents"]) >= 2
            assert "full_stack_engineer" in result["active_agents"]

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_router_functionality(self) -> None:
        """Test router node functionality."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Test routing to next agent
            state: AgentState = {
                "task": "Test task",
                "messages": [],
                "next": "full_stack_engineer",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": ["full_stack_engineer", "qa_engineer"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.route_to_appropriate_agents(state)

            # Should route to next agent (qa_engineer)
            assert result["next"] == "qa_engineer"

            # Check routing info
            routing_info = result["scratchpad"]["routing_info"]
            assert routing_info["current_step"] == 1
            assert len(routing_info["routing_history"]) == 1
            assert routing_info["routing_history"][0]["agent"] == "full_stack_engineer"

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_router_to_coordinator(self) -> None:
        """Test routing to coordinator when all agents are processed."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Test routing to coordinator (last agent)
            state: AgentState = {
                "task": "Test task",
                "messages": [],
                "next": "qa_engineer",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": ["full_stack_engineer", "qa_engineer"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.route_to_appropriate_agents(state)

            # Should route to coordinator
            assert result["next"] == "coordinator"

            # Check routing completion
            routing_info = result["scratchpad"]["routing_info"]
            assert routing_info["routing_complete"] is True

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_coordinator_functionality(self) -> None:
        """Test coordinator node functionality."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            state: AgentState = {
                "task": "Build a simple API",
                "messages": [],
                "next": "coordinator",
                "scratchpad": {
                    "full_stack_engineer_analysis": {
                        "response_length": 500,
                        "model_used": "gpt-4",
                        "provider": "openai",
                        "timestamp": datetime.now().isoformat(),
                    }
                },
                "mode": "orchestration",
                "active_agents": ["full_stack_engineer"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            result = await factory.coordinate_agent_responses(state)

            # Check that coordination message was added
            assert len(result["messages"]) == 1
            message_content = result["messages"][0].content
            assert "Multi-Agent Task Completion Report" in message_content
            assert "Full Stack Engineer Analysis" in message_content

            # Check task status
            assert result["task_status"]["status"] == "completed"
            assert result["task_status"]["coordination_complete"] is True

            # Check coordination info
            coordination_info = result["scratchpad"]["coordination_info"]
            assert coordination_info["synthesis_complete"] is True
            assert coordination_info["participating_agents"] == ["full_stack_engineer"]

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_determine_next_agent(self) -> None:
        """Test next agent determination logic."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Load agent configurations

            config_dir = Path("agent_configs")
            factory.agent_factory.load_agent_configurations(config_dir)

            # Test routing to valid agent
            state: AgentState = {
                "task": "Test task",
                "messages": [],
                "next": "full_stack_engineer",
                "scratchpad": {},
                "mode": "orchestration",
                "active_agents": ["full_stack_engineer"],
                "multimodal_content": {},
                "session_id": "test-session",
                "user_id": "test-user",
                "last_updated": None,
                "errors": [],
                "task_status": {},
            }

            next_agent = factory.determine_next_agent(state)
            assert next_agent == "full_stack_engineer"

            # Test routing to coordinator
            state["next"] = "coordinator"
            next_agent = factory.determine_next_agent(state)
            assert next_agent == "coordinator"

            # Test routing to invalid agent (should default to coordinator)
            state["next"] = "invalid_agent"
            next_agent = factory.determine_next_agent(state)
            assert next_agent == "coordinator"

        finally:
            await session_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])
