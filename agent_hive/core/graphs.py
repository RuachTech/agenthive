"""Graph factory system for creating LangGraph workflows."""

import logging
from typing import Dict, Any, Optional, List, cast
from datetime import datetime
import asyncio
from dataclasses import dataclass

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig

from .state import AgentState
from .config import SystemConfig
from ..agents.factory import get_agent_factory

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Session management for direct chat conversations."""

    session_id: str
    user_id: str
    mode: str  # "direct" or "orchestration"
    active_agent: Optional[str]
    created_at: datetime
    last_activity: datetime
    state: AgentState
    multimodal_files: List[Dict[str, Any]]

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired based on last activity."""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class SessionManager:
    """Manages chat sessions and state persistence."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Session manager stopped")

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        mode: str = "direct",
        active_agent: Optional[str] = None,
        initial_task: Optional[str] = None,
    ) -> Session:
        """Create a new chat session."""
        now = datetime.now()

        # Initialize agent state
        initial_state: AgentState = {
            "task": initial_task or "",
            "messages": [],
            "next": active_agent or "",
            "scratchpad": {},
            "mode": mode,
            "active_agents": [active_agent] if active_agent else [],
            "multimodal_content": {},
            "session_id": session_id,
            "user_id": user_id,
            "last_updated": now,
            "errors": [],
            "task_status": {"status": "initialized", "timestamp": now.isoformat()},
        }

        session = Session(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            active_agent=active_agent,
            created_at=now,
            last_activity=now,
            state=initial_state,
            multimodal_files=[],
        )

        self.sessions[session_id] = session
        logger.info(
            "Created session %s for user %s in %s mode", session_id, user_id, mode
        )

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session by ID."""
        session = self.sessions.get(session_id)
        if session:
            if session.is_expired(self.config.session_timeout):
                await self.delete_session(session_id)
                return None
            session.update_activity()
        return session

    async def update_session_state(self, session_id: str, state: AgentState) -> bool:
        """Update the state of an existing session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        session.state = state
        session.state["last_updated"] = datetime.now()
        session.update_activity()

        logger.debug("Updated state for session %s", session_id)
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Deleted session %s", session_id)
            return True
        return False

    async def list_sessions(self, user_id: Optional[str] = None) -> List[Session]:
        """List all sessions, optionally filtered by user."""
        sessions = list(self.sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        return sessions

    async def _cleanup_expired_sessions(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if session.is_expired(self.config.session_timeout):
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    await self.delete_session(session_id)

                if expired_sessions:
                    logger.info("Cleaned up %d expired sessions", len(expired_sessions))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in session cleanup: %s", str(e))


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    pass


class DirectChatGraphFactory:
    """Factory for creating single-agent direct chat graphs."""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or SessionManager()
        self.agent_factory = get_agent_factory()
        self._compiled_graphs: dict[str, CompiledStateGraph] = {}

    async def create_direct_chat_graph(
        self, agent_name: str, checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> CompiledStateGraph:
        """
        Create a simple graph for direct agent interaction.

        Args:
            agent_name: Name of the agent to create graph for
            checkpointer: Optional checkpointer for state persistence

        Returns:
            Compiled LangGraph for direct chat with the specified agent

        Raises:
            GraphValidationError: If agent validation fails
        """
        # Check if we already have a compiled graph for this agent
        cache_key = f"direct_{agent_name}"
        if cache_key in self._compiled_graphs:
            return self._compiled_graphs[cache_key]

        # Validate agent exists and can be created
        validation_result = await self.agent_factory.validate_agent(agent_name)
        if not validation_result["valid"]:
            raise GraphValidationError(
                f"Agent '{agent_name}' validation failed: {validation_result['errors']}"
            )

        # Create agent node
        agent_node = await self.agent_factory.create_agent_node(agent_name)

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add the single agent node
        workflow.add_node(agent_name, agent_node)

        # Set entry point and exit
        workflow.set_entry_point(agent_name)
        workflow.add_edge(agent_name, END)

        # Use memory checkpointer if none provided
        if checkpointer is None:
            checkpointer = MemorySaver()

        # Compile the graph
        compiled_graph = workflow.compile(checkpointer=checkpointer)

        # Cache the compiled graph
        self._compiled_graphs[cache_key] = compiled_graph

        logger.info("Created direct chat graph for agent: %s", agent_name)
        return compiled_graph

    async def execute_direct_chat(
        self,
        session_id: str,
        agent_name: str,
        message: str,
        user_id: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Execute a direct chat interaction with an agent.

        Args:
            session_id: Unique session identifier
            agent_name: Name of the agent to chat with
            message: User message
            user_id: User identifier

        Returns:
            Dictionary containing response and session info
        """
        # Get or create session
        session = await self.session_manager.get_session(session_id)
        if not session:
            session = await self.session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                mode="direct",
                active_agent=agent_name,
                initial_task=message,
            )

        # Update session state with new message
        from langchain_core.messages import HumanMessage

        session.state["messages"].append(HumanMessage(content=message))
        session.state["task"] = message

        # Get the compiled graph
        graph = await self.create_direct_chat_graph(agent_name)

        # Execute the graph
        try:
            # Create thread config for checkpointing
            thread_config = RunnableConfig(configurable={"thread_id": session_id})

            # Invoke the graph with current state
            result = await graph.ainvoke(session.state, config=thread_config)

            # Update session with result
            if isinstance(result, dict):
                await self.session_manager.update_session_state(
                    session_id, cast(AgentState, result)
                )

            # Extract the response
            response_content = ""
            if result.get("messages"):
                last_message = result["messages"][-1]
                response_content = getattr(last_message, "content", str(last_message))

            return {
                "response": response_content,
                "session_id": session_id,
                "agent_name": agent_name,
                "status": "success",
                "state": {
                    "message_count": len(result.get("messages", [])),
                    "active_agents": result.get("active_agents", []),
                    "last_updated": result.get("last_updated"),
                    "errors": result.get("errors", []),
                },
            }

        except Exception as e:
            logger.error("Direct chat execution failed: %s", str(e))

            # Update session with error
            error_info = {
                "type": "execution_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
            }
            if isinstance(session.state.get("errors"), list):
                session.state["errors"].append(error_info)
            await self.session_manager.update_session_state(session_id, session.state)

            return {
                "response": f"I encountered an error: {str(e)}",
                "session_id": session_id,
                "agent_name": agent_name,
                "status": "error",
                "error": str(e),
            }

    async def stream_direct_chat(
        self,
        session_id: str,
        agent_name: str,
        message: str,
        user_id: str = "default_user",
    ) -> Any:
        """
        Stream a direct chat interaction with an agent.

        Args:
            session_id: Unique session identifier
            agent_name: Name of the agent to chat with
            message: User message
            user_id: User identifier

        Yields:
            Streaming response chunks
        """
        # Get or create session
        session = await self.session_manager.get_session(session_id)
        if not session:
            session = await self.session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                mode="direct",
                active_agent=agent_name,
                initial_task=message,
            )

        # Update session state with new message
        from langchain_core.messages import HumanMessage

        session.state["messages"].append(HumanMessage(content=message))
        session.state["task"] = message

        # Get the compiled graph
        graph = await self.create_direct_chat_graph(agent_name)

        try:
            # Create thread config for checkpointing
            thread_config = RunnableConfig(configurable={"thread_id": session_id})

            # Stream the graph execution
            async for chunk in graph.astream(session.state, config=thread_config):
                # Extract content from chunk
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            if messages:
                                last_message = messages[-1]
                                content = getattr(
                                    last_message, "content", str(last_message)
                                )
                                if content:
                                    yield {
                                        "type": "content",
                                        "content": content,
                                        "node": node_name,
                                        "session_id": session_id,
                                    }

                # Update session state with latest chunk
                if isinstance(chunk, dict) and agent_name in chunk:
                    chunk_data = chunk[agent_name]
                    if isinstance(chunk_data, dict):
                        await self.session_manager.update_session_state(
                            session_id, cast(AgentState, chunk_data)
                        )

            # Send completion signal
            yield {
                "type": "complete",
                "session_id": session_id,
                "agent_name": agent_name,
            }

        except Exception as e:
            logger.error("Direct chat streaming failed: %s", str(e))
            yield {
                "type": "error",
                "error": str(e),
                "session_id": session_id,
                "agent_name": agent_name,
            }

    def validate_graph_structure(
        self, graph: CompiledStateGraph, agent_name: str
    ) -> Dict[str, Any]:
        """
        Validate the structure and configuration of a compiled graph.

        Args:
            graph: Compiled graph to validate
            agent_name: Expected agent name in the graph

        Returns:
            Validation result with details
        """
        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
        }

        try:
            # Check if graph has the expected structure
            graph_dict = graph.get_graph().to_json()
            nodes = graph_dict.get("nodes", [])
            edges = graph_dict.get("edges", [])

            # Validate nodes
            node_names = [node.get("id") for node in nodes]
            if agent_name not in node_names:
                validation_result["errors"].append(
                    f"Expected agent node '{agent_name}' not found in graph"
                )
                validation_result["valid"] = False

            # Validate edges - should have entry to agent and agent to END
            entry_edges = [e for e in edges if e.get("source") == START]
            exit_edges = [e for e in edges if e.get("target") == END]

            if not entry_edges:
                validation_result["errors"].append("No entry point found in graph")
                validation_result["valid"] = False

            if not exit_edges:
                validation_result["errors"].append("No exit point found in graph")
                validation_result["valid"] = False

            # Add info about graph structure
            validation_result["info"] = {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": node_names,
                "has_checkpointer": hasattr(graph, "checkpointer")
                and graph.checkpointer is not None,
            }

        except Exception as e:
            validation_result["errors"].append(f"Graph validation failed: {str(e)}")
            validation_result["valid"] = False

        return validation_result

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents for direct chat."""
        agents = []
        for agent_name in self.agent_factory.list_agents():
            config = self.agent_factory.get_agent_config(agent_name)
            validation = await self.agent_factory.validate_agent(agent_name)

            agents.append(
                {
                    "name": agent_name,
                    "display_name": config.display_name if config else agent_name,
                    "description": config.description if config else "",
                    "capabilities": config.capabilities if config else [],
                    "available": validation["valid"],
                    "validation_errors": validation.get("errors", []),
                }
            )

        return agents

    def clear_graph_cache(self) -> None:
        """Clear the compiled graph cache."""
        self._compiled_graphs.clear()
        logger.info("Cleared graph cache")


# Global instances
_session_manager: Optional[SessionManager] = None
_graph_factory: Optional[DirectChatGraphFactory] = None


async def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        await _session_manager.start()
    return _session_manager


async def get_graph_factory() -> DirectChatGraphFactory:
    """Get the global graph factory instance."""
    global _graph_factory
    if _graph_factory is None:
        session_manager = await get_session_manager()
        _graph_factory = DirectChatGraphFactory(session_manager)
    return _graph_factory


class OrchestratorGraphFactory:
    """Factory for creating multi-agent orchestrator graphs."""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or SessionManager()
        self.agent_factory = get_agent_factory()
        self._compiled_orchestrator_graph: Optional[CompiledStateGraph] = None

    async def analyze_task_requirements(self, state: AgentState) -> AgentState:
        """
        Analyze the task and determine which agents are needed.

        This node examines the task description and determines:
        - Which specialized agents should be involved
        - The order of agent execution
        - Any special requirements or constraints
        """
        task = state.get("task", "")

        # Initialize task analysis in scratchpad
        state["scratchpad"]["task_analysis"] = {
            "original_task": task,
            "timestamp": datetime.now().isoformat(),
            "analysis_stage": "requirements",
        }

        # Analyze task for different agent requirements
        required_agents = []
        task_lower = task.lower()

        # Check for development-related keywords
        dev_keywords = [
            "code",
            "develop",
            "build",
            "implement",
            "api",
            "database",
            "backend",
            "frontend",
        ]
        if any(keyword in task_lower for keyword in dev_keywords):
            required_agents.append("full_stack_engineer")

        # Check for testing-related keywords
        test_keywords = ["test", "qa", "quality", "bug", "verify", "validate"]
        if any(keyword in task_lower for keyword in test_keywords):
            required_agents.append("qa_engineer")

        # Check for design-related keywords
        design_keywords = [
            "design",
            "ui",
            "ux",
            "interface",
            "wireframe",
            "mockup",
            "figma",
        ]
        if any(keyword in task_lower for keyword in design_keywords):
            required_agents.append("product_designer")

        # Check for DevOps-related keywords
        devops_keywords = [
            "deploy",
            "infrastructure",
            "docker",
            "kubernetes",
            "aws",
            "cloud",
            "ci/cd",
        ]
        if any(keyword in task_lower for keyword in devops_keywords):
            required_agents.append("devops_engineer")

        # If no specific agents identified, default to full stack engineer
        if not required_agents:
            required_agents.append("full_stack_engineer")

        # Update state with analysis results
        state["scratchpad"]["task_analysis"]["required_agents"] = required_agents
        state["scratchpad"]["task_analysis"]["agent_count"] = len(required_agents)
        state["active_agents"] = required_agents.copy()

        # Set the first agent to route to
        state["next"] = required_agents[0] if required_agents else "coordinator"

        logger.info("Task analysis complete. Required agents: %s", required_agents)

        return state

    async def route_to_appropriate_agents(self, state: AgentState) -> AgentState:
        """
        Route to the next appropriate agent based on task analysis and current progress.

        This node manages the workflow between agents, ensuring proper sequencing
        and coordination of multi-agent collaboration.
        """
        # Get current routing information
        current_agent = state.get("next", "")
        active_agents = state.get("active_agents", [])

        # Initialize routing info in scratchpad
        if "routing_info" not in state["scratchpad"]:
            state["scratchpad"]["routing_info"] = {
                "routing_history": [],
                "current_step": 0,
                "total_agents": len(active_agents),
            }

        routing_info = state["scratchpad"]["routing_info"]

        # Record current routing step
        routing_step = {
            "step": routing_info["current_step"],
            "agent": current_agent,
            "timestamp": datetime.now().isoformat(),
            "status": "routing",
        }
        routing_info["routing_history"].append(routing_step)

        # Update routing step counter
        routing_info["current_step"] += 1

        # Determine next agent based on workflow logic
        if current_agent in active_agents:
            current_index = active_agents.index(current_agent)

            # Check if there are more agents to process
            if current_index + 1 < len(active_agents):
                next_agent = active_agents[current_index + 1]
                state["next"] = next_agent
                logger.info("Routing from %s to %s", current_agent, next_agent)
            else:
                # All agents processed, route to coordinator
                state["next"] = "coordinator"
                logger.info("All agents processed, routing to coordinator")
        else:
            # Route to coordinator if current agent not in active list
            state["next"] = "coordinator"
            logger.info("Routing to coordinator (agent not in active list)")

        # Update routing status
        routing_info["next_agent"] = state["next"]
        routing_info["routing_complete"] = state["next"] == "coordinator"

        return state

    async def coordinate_agent_responses(self, state: AgentState) -> AgentState:
        """
        Coordinate and synthesize responses from multiple agents.

        This node takes the outputs from all participating agents and creates
        a comprehensive, unified response for the user.
        """
        # Initialize coordination info
        coordination_info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "participating_agents": state.get("active_agents", []),
            "coordination_stage": "synthesis",
        }

        state["scratchpad"]["coordination_info"] = coordination_info

        # Collect agent contributions from scratchpad
        agent_contributions = {}
        for agent_name in state.get("active_agents", []):
            agent_key = f"{agent_name}_analysis"
            if agent_key in state["scratchpad"]:
                agent_contributions[agent_name] = state["scratchpad"][agent_key]

        # Synthesize final response
        from langchain_core.messages import AIMessage

        synthesis_content = self._synthesize_agent_responses(
            state.get("task", ""), agent_contributions, state.get("active_agents", [])
        )

        # Add synthesized response to messages
        coordinator_message = AIMessage(content=synthesis_content)
        state["messages"].append(coordinator_message)

        # Update task status
        state["task_status"] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "participating_agents": state.get("active_agents", []),
            "coordination_complete": True,
        }

        # Update coordination info
        coordination_info["synthesis_complete"] = True
        coordination_info["final_response_length"] = len(synthesis_content)

        logger.info(
            "Coordination complete for task with %d agents", len(agent_contributions)
        )

        return state

    def _synthesize_agent_responses(
        self,
        original_task: str,
        agent_contributions: Dict[str, Any],
        participating_agents: List[str],
    ) -> str:
        """Synthesize responses from multiple agents into a coherent final response."""

        synthesis_parts = [
            "# Multi-Agent Task Completion Report\n",
            f"**Original Task:** {original_task}\n",
            f"**Participating Agents:** {', '.join(participating_agents)}\n\n",
        ]

        # Add contributions from each agent
        for agent_name in participating_agents:
            if agent_name in agent_contributions:
                contribution = agent_contributions[agent_name]
                synthesis_parts.append(
                    f"## {agent_name.replace('_', ' ').title()} Analysis\n"
                )

                # Extract relevant information from agent contribution
                if isinstance(contribution, dict):
                    if "response_length" in contribution:
                        synthesis_parts.append(
                            f"- Provided detailed analysis ({contribution['response_length']} characters)\n"
                        )
                    if "model_used" in contribution:
                        synthesis_parts.append(
                            f"- Model used: {contribution['model_used']}\n"
                        )
                    if "timestamp" in contribution:
                        synthesis_parts.append(
                            f"- Completed at: {contribution['timestamp']}\n"
                        )

                synthesis_parts.append("\n")

        # Add summary and next steps
        synthesis_parts.extend(
            [
                "## Summary\n",
                f"This task was successfully processed by {len(participating_agents)} specialized agents, ",
                "each contributing their domain expertise to provide a comprehensive solution.\n\n",
                "## Next Steps\n",
                "- Review the individual agent contributions above\n",
                "- Implement any recommended actions\n",
                "- Follow up with specific agents if additional clarification is needed\n",
            ]
        )

        return "".join(synthesis_parts)

    def determine_next_agent(self, state: AgentState) -> str:
        """
        Determine which agent should be executed next based on current state.

        This is used by LangGraph's conditional edges to route between agents.
        """
        next_agent = state.get("next", "coordinator")

        # Validate that the next agent is available
        available_agents = self.agent_factory.list_agents()
        if next_agent not in available_agents and next_agent != "coordinator":
            logger.warning(
                "Next agent '%s' not available, routing to coordinator", next_agent
            )
            return "coordinator"

        return next_agent

    async def create_orchestrator_graph(
        self, checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> CompiledStateGraph:
        """
        Create the complex orchestration graph with conditional routing.

        Flow: START -> Task Analyzer -> Router -> [Agent Nodes] -> Coordinator -> END

        Args:
            checkpointer: Optional checkpointer for state persistence

        Returns:
            Compiled LangGraph for multi-agent orchestration
        """
        # Return cached graph if available
        if self._compiled_orchestrator_graph is not None:
            return self._compiled_orchestrator_graph

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add orchestration nodes
        workflow.add_node("task_analyzer", self.analyze_task_requirements)
        workflow.add_node("router", self.route_to_appropriate_agents)
        workflow.add_node("coordinator", self.coordinate_agent_responses)

        # Add all available agent nodes
        agent_nodes = {}
        for agent_name in self.agent_factory.list_agents():
            try:
                agent_node = await self.agent_factory.create_agent_node(agent_name)
                workflow.add_node(agent_name, agent_node)
                agent_nodes[agent_name] = agent_node
                logger.info("Added agent node: %s", agent_name)
            except Exception as e:
                logger.error("Failed to create agent node %s: %s", agent_name, str(e))

        # Define the workflow edges
        workflow.set_entry_point("task_analyzer")
        workflow.add_edge("task_analyzer", "router")

        # Add conditional edges from router to agents
        workflow.add_conditional_edges(
            "router",
            self.determine_next_agent,
            {
                **{agent_name: agent_name for agent_name in agent_nodes.keys()},
                "coordinator": "coordinator",
            },
        )

        # Add edges from each agent back to router (for multi-agent workflows)
        for agent_name in agent_nodes.keys():
            workflow.add_edge(agent_name, "router")

        # Add final edge from coordinator to end
        workflow.add_edge("coordinator", END)

        # Use memory checkpointer if none provided
        if checkpointer is None:
            checkpointer = MemorySaver()

        # Compile the graph
        self._compiled_orchestrator_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Created orchestrator graph with %d agent nodes", len(agent_nodes))
        return self._compiled_orchestrator_graph

    async def execute_orchestration(
        self, session_id: str, task: str, user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Execute a multi-agent orchestration task.

        Args:
            session_id: Unique session identifier
            task: Task description for orchestration
            user_id: User identifier

        Returns:
            Dictionary containing orchestration results
        """
        # Get or create session
        session = await self.session_manager.get_session(session_id)
        if not session:
            session = await self.session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                mode="orchestration",
                initial_task=task,
            )

        # Update session state with task
        from langchain_core.messages import HumanMessage

        session.state["messages"].append(HumanMessage(content=task))
        session.state["task"] = task
        session.state["mode"] = "orchestration"

        # Get the orchestrator graph
        graph = await self.create_orchestrator_graph()

        try:
            # Create thread config for checkpointing
            thread_config = RunnableConfig(configurable={"thread_id": session_id})

            # Execute the orchestration
            result = await graph.ainvoke(session.state, config=thread_config)

            # Update session with result
            if isinstance(result, dict):
                await self.session_manager.update_session_state(
                    session_id, cast(AgentState, result)
                )

            # Extract the final response
            response_content = ""
            if result.get("messages"):
                last_message = result["messages"][-1]
                response_content = getattr(last_message, "content", str(last_message))

            return {
                "response": response_content,
                "session_id": session_id,
                "mode": "orchestration",
                "status": "success",
                "orchestration_info": {
                    "participating_agents": result.get("active_agents", []),
                    "task_status": result.get("task_status", {}),
                    "coordination_complete": result.get("scratchpad", {})
                    .get("coordination_info", {})
                    .get("synthesis_complete", False),
                },
                "state": {
                    "message_count": len(result.get("messages", [])),
                    "active_agents": result.get("active_agents", []),
                    "last_updated": result.get("last_updated"),
                    "errors": result.get("errors", []),
                },
            }

        except Exception as e:
            logger.error("Orchestration execution failed: %s", str(e))

            # Update session with error
            error_info = {
                "type": "orchestration_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "mode": "orchestration",
            }
            if isinstance(session.state.get("errors"), list):
                session.state["errors"].append(error_info)
            await self.session_manager.update_session_state(session_id, session.state)

            return {
                "response": f"Orchestration failed: {str(e)}",
                "session_id": session_id,
                "mode": "orchestration",
                "status": "error",
                "error": str(e),
            }

    async def stream_orchestration(
        self, session_id: str, task: str, user_id: str = "default_user"
    ) -> Any:
        """
        Stream a multi-agent orchestration execution.

        Args:
            session_id: Unique session identifier
            task: Task description for orchestration
            user_id: User identifier

        Yields:
            Streaming orchestration progress and results
        """

        # Get or create session
        session = await self.session_manager.get_session(session_id)
        if not session:
            session = await self.session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                mode="orchestration",
                initial_task=task,
            )

        # Update session state with task
        from langchain_core.messages import HumanMessage

        session.state["messages"].append(HumanMessage(content=task))
        session.state["task"] = task
        session.state["mode"] = "orchestration"

        # Get the orchestrator graph
        graph = await self.create_orchestrator_graph()

        try:
            # Create thread config for checkpointing
            thread_config = RunnableConfig(configurable={"thread_id": session_id})

            # Stream the orchestration execution
            async for chunk in graph.astream(session.state, config=thread_config):
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        # Send progress updates for each node
                        yield {
                            "type": "progress",
                            "node": node_name,
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                        }

                        # Send content updates if available
                        if isinstance(node_output, dict):
                            if "messages" in node_output and node_output["messages"]:
                                last_message = node_output["messages"][-1]
                                content = getattr(last_message, "content", "")
                                if content:
                                    yield {
                                        "type": "content",
                                        "content": content,
                                        "node": node_name,
                                        "session_id": session_id,
                                    }

                            # Send orchestration status updates
                            if "task_status" in node_output:
                                yield {
                                    "type": "status",
                                    "status": node_output["task_status"],
                                    "node": node_name,
                                    "session_id": session_id,
                                }

                            # Update session state
                            await self.session_manager.update_session_state(
                                session_id, cast(AgentState, node_output)
                            )

            # Send completion signal
            yield {
                "type": "complete",
                "session_id": session_id,
                "mode": "orchestration",
            }

        except Exception as e:
            logger.error("Orchestration streaming failed: %s", str(e))
            yield {
                "type": "error",
                "error": str(e),
                "session_id": session_id,
                "mode": "orchestration",
            }


# Global orchestrator factory instance
_orchestrator_factory: Optional[OrchestratorGraphFactory] = None


async def get_orchestrator_factory() -> OrchestratorGraphFactory:
    """Get the global orchestrator factory instance."""
    global _orchestrator_factory
    if _orchestrator_factory is None:
        session_manager = await get_session_manager()
        _orchestrator_factory = OrchestratorGraphFactory(session_manager)
    return _orchestrator_factory


async def cleanup_global_instances() -> None:
    """Cleanup global instances."""
    global _session_manager, _graph_factory, _orchestrator_factory
    if _session_manager:
        await _session_manager.stop()
        _session_manager = None
    _graph_factory = None
    _orchestrator_factory = None
