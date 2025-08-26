"""Core state management for AgentHive."""

from typing import TypedDict, Annotated, Any, Optional
import operator
from datetime import datetime
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state object that maintains context across all agent interactions."""

    # Primary task description
    task: str

    # Conversation history with automatic message aggregation
    messages: Annotated[list[BaseMessage], operator.add]

    # Next agent to route to in orchestration mode
    next: str

    # Agent working memory for intermediate results and context
    scratchpad: dict[str, Any]

    # Operating mode: "direct" or "orchestration"
    mode: str

    # list of currently involved agents
    active_agents: list[str]

    # Processed files and media content
    multimodal_content: dict[str, Any]

    # Unique session identifier
    session_id: str

    # Optional user identifier for multi-user scenarios
    user_id: Optional[str]

    # Timestamp of last state update
    last_updated: Optional[datetime]

    # Error tracking and recovery information
    errors: list[dict[str, Any]]

    # Task completion status and progress tracking
    task_status: dict[str, Any]
