"""Full workflow integration tests for multi-agent orchestrator graph."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path

from agent_hive.core.graphs import OrchestratorGraphFactory, SessionManager
from agent_hive.core.state import AgentState


class TestOrchestratorFullWorkflow:
    """Test complete orchestrator workflow from start to finish."""

    @pytest.mark.asyncio
    async def test_create_orchestrator_graph(self):
        """Test creation of the complete orchestrator graph."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Load agent configurations
            config_dir = Path("agent_configs")
            factory.agent_factory.load_agent_configurations(config_dir)

            # Mock agent node creation to avoid model dependencies
            original_create_agent_node = factory.agent_factory.create_agent_node

            async def mock_create_agent_node(agent_name):
                # Create a simple mock agent node
                async def mock_agent_node(state: AgentState) -> AgentState:
                    from langchain_core.messages import AIMessage

                    # Add a mock response
                    response = f"Mock response from {agent_name}"
                    state["messages"].append(AIMessage(content=response))

                    # Update scratchpad
                    state["scratchpad"][f"{agent_name}_analysis"] = {
                        "response_length": len(response),
                        "model_used": "mock-model",
                        "provider": "mock",
                        "timestamp": datetime.now().isoformat(),
                    }

                    return state

                return mock_agent_node

            factory.agent_factory.create_agent_node = mock_create_agent_node

            # Create the orchestrator graph
            graph = await factory.create_orchestrator_graph()

            # Verify graph structure
            assert graph is not None

            graph_dict = graph.get_graph().to_json()
            nodes = [node["id"] for node in graph_dict["nodes"]]

            # Should have orchestration nodes
            assert "task_analyzer" in nodes
            assert "router" in nodes
            assert "coordinator" in nodes

            # Should have agent nodes
            assert "full_stack_engineer" in nodes
            assert "qa_engineer" in nodes
            assert "product_designer" in nodes
            assert "devops_engineer" in nodes

            # Restore original method
            factory.agent_factory.create_agent_node = original_create_agent_node

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_execute_orchestration_workflow(self):
        """Test complete orchestration execution workflow."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Load agent configurations
            config_dir = Path("agent_configs")
            factory.agent_factory.load_agent_configurations(config_dir)

            # Mock the graph execution to avoid model dependencies
            with patch.object(
                factory, "create_orchestrator_graph"
            ) as mock_create_graph:
                mock_graph = AsyncMock()

                # Mock the execution result
                mock_result = {
                    "task": "Build a web application",
                    "messages": [
                        Mock(content="Task analysis complete"),
                        Mock(content="Development work complete"),
                        Mock(
                            content="Multi-Agent Task Completion Report\n\n**Original Task:** Build a web application"
                        ),
                    ],
                    "active_agents": ["full_stack_engineer"],
                    "task_status": {
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "participating_agents": ["full_stack_engineer"],
                        "coordination_complete": True,
                    },
                    "scratchpad": {
                        "task_analysis": {
                            "required_agents": ["full_stack_engineer"],
                            "agent_count": 1,
                        },
                        "coordination_info": {
                            "synthesis_complete": True,
                            "participating_agents": ["full_stack_engineer"],
                        },
                    },
                    "last_updated": datetime.now(),
                    "errors": [],
                    "mode": "orchestration",
                    "next": "coordinator",
                    "multimodal_content": {},
                    "session_id": "test-session",
                    "user_id": "test-user",
                }

                mock_graph.ainvoke.return_value = mock_result
                mock_create_graph.return_value = mock_graph

                # Execute orchestration
                result = await factory.execute_orchestration(
                    session_id="test-orchestration-session",
                    task="Build a web application",
                    user_id="test-user",
                )

                # Verify execution result
                assert result["status"] == "success"
                assert result["mode"] == "orchestration"
                assert result["session_id"] == "test-orchestration-session"
                assert "response" in result

                # Verify orchestration info
                orchestration_info = result["orchestration_info"]
                assert orchestration_info["participating_agents"] == [
                    "full_stack_engineer"
                ]
                assert orchestration_info["coordination_complete"] is True

                # Verify state info
                state_info = result["state"]
                assert state_info["message_count"] == 3
                assert state_info["active_agents"] == ["full_stack_engineer"]

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_stream_orchestration_workflow(self):
        """Test streaming orchestration execution workflow."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Load agent configurations
            config_dir = Path("agent_configs")
            factory.agent_factory.load_agent_configurations(config_dir)

            # Mock the graph streaming
            with patch.object(
                factory, "create_orchestrator_graph"
            ) as mock_create_graph:
                mock_graph = AsyncMock()

                # Mock streaming chunks
                async def mock_astream(state, config):
                    # Task analyzer chunk
                    yield {
                        "task_analyzer": {
                            "task": "Create a test application",
                            "active_agents": ["full_stack_engineer"],
                            "scratchpad": {
                                "task_analysis": {
                                    "required_agents": ["full_stack_engineer"],
                                    "agent_count": 1,
                                }
                            },
                        }
                    }

                    # Router chunk
                    yield {
                        "router": {
                            "next": "full_stack_engineer",
                            "scratchpad": {
                                "routing_info": {
                                    "current_step": 1,
                                    "routing_complete": False,
                                }
                            },
                        }
                    }

                    # Agent chunk
                    yield {
                        "full_stack_engineer": {
                            "messages": [Mock(content="Development analysis complete")],
                            "task_status": {"status": "in_progress"},
                            "scratchpad": {
                                "full_stack_engineer_analysis": {
                                    "response_length": 100,
                                    "model_used": "mock-model",
                                }
                            },
                        }
                    }

                    # Coordinator chunk
                    yield {
                        "coordinator": {
                            "messages": [
                                Mock(content="Multi-Agent Task Completion Report")
                            ],
                            "task_status": {
                                "status": "completed",
                                "coordination_complete": True,
                            },
                            "scratchpad": {
                                "coordination_info": {
                                    "synthesis_complete": True,
                                    "participating_agents": ["full_stack_engineer"],
                                }
                            },
                        }
                    }

                mock_graph.astream = mock_astream
                mock_create_graph.return_value = mock_graph

                # Collect streaming results
                results = []
                async for chunk in factory.stream_orchestration(
                    session_id="test-stream-session",
                    task="Create a test application",
                    user_id="test-user",
                ):
                    results.append(chunk)

                # Verify streaming results
                assert len(results) > 0

                # Should have progress updates for each node
                progress_chunks = [r for r in results if r.get("type") == "progress"]
                assert (
                    len(progress_chunks) >= 4
                )  # task_analyzer, router, agent, coordinator

                # Verify progress chunk structure
                for chunk in progress_chunks:
                    assert "node" in chunk
                    assert "session_id" in chunk
                    assert "timestamp" in chunk

                # Should have content updates
                content_chunks = [r for r in results if r.get("type") == "content"]
                assert len(content_chunks) >= 1

                # Verify content chunk structure
                for chunk in content_chunks:
                    assert "content" in chunk
                    assert "node" in chunk
                    assert "session_id" in chunk

                # Should have status updates
                status_chunks = [r for r in results if r.get("type") == "status"]
                assert len(status_chunks) >= 1

                # Should have completion signal
                complete_chunks = [r for r in results if r.get("type") == "complete"]
                assert len(complete_chunks) == 1
                assert complete_chunks[0]["mode"] == "orchestration"
                assert complete_chunks[0]["session_id"] == "test-stream-session"

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_orchestration_error_handling(self):
        """Test error handling during orchestration execution."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Mock graph to raise an exception
            with patch.object(
                factory, "create_orchestrator_graph"
            ) as mock_create_graph:
                mock_graph = AsyncMock()
                mock_graph.ainvoke.side_effect = Exception("Test orchestration error")
                mock_create_graph.return_value = mock_graph

                result = await factory.execute_orchestration(
                    session_id="test-error-session",
                    task="Test task that will fail",
                    user_id="test-user",
                )

                # Verify error handling
                assert result["status"] == "error"
                assert "Test orchestration error" in result["error"]
                assert "Orchestration failed" in result["response"]
                assert result["mode"] == "orchestration"
                assert result["session_id"] == "test-error-session"

        finally:
            await session_manager.stop()

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test error handling during streaming orchestration."""
        session_manager = SessionManager()
        await session_manager.start()

        try:
            factory = OrchestratorGraphFactory(session_manager)

            # Mock graph to raise an exception during streaming
            with patch.object(
                factory, "create_orchestrator_graph"
            ) as mock_create_graph:
                mock_graph = AsyncMock()

                async def mock_astream_error(state, config):
                    yield {"task_analyzer": {"task": "Test streaming error"}}
                    raise Exception("Test streaming error")

                mock_graph.astream = mock_astream_error
                mock_create_graph.return_value = mock_graph

                # Collect streaming results
                results = []
                async for chunk in factory.stream_orchestration(
                    session_id="test-stream-error-session",
                    task="Test streaming error",
                    user_id="test-user",
                ):
                    results.append(chunk)

                # Should have at least one progress chunk and one error chunk
                progress_chunks = [r for r in results if r.get("type") == "progress"]
                assert len(progress_chunks) >= 1

                error_chunks = [r for r in results if r.get("type") == "error"]
                assert len(error_chunks) == 1
                assert "Test streaming error" in error_chunks[0]["error"]
                assert error_chunks[0]["mode"] == "orchestration"
                assert error_chunks[0]["session_id"] == "test-stream-error-session"

        finally:
            await session_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])
