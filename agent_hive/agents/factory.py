"""Agent node factory system for creating LangGraph-compatible agent functions."""

import logging
from typing import Dict, Any, List, Optional, Sequence
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from ..core.state import AgentState
from ..core.config import AgentConfig
from ..core.models import ModelFactory, get_model_factory, ModelInterface

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """Defines what capabilities an agent has and requires."""
    
    # Required capabilities for the agent to function
    required_capabilities: List[str]
    
    # Optional capabilities that enhance functionality
    optional_capabilities: List[str]
    
    # Model requirements (e.g., vision, function_calling)
    model_requirements: List[str]
    
    # Tool requirements
    required_tools: List[str]
    
    # MCP server requirements
    mcp_requirements: List[str]


class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, agent_name: str, error_type: str, message: str):
        self.agent_name = agent_name
        self.error_type = error_type
        self.message = message
        super().__init__(f"{agent_name}: {error_type} - {message}")


class AgentValidationError(AgentError):
    """Raised when agent validation fails."""
    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


class ErrorRecoveryStrategy:
    """Handles error recovery for agent execution."""
    
    @staticmethod
    async def handle_model_timeout(state: AgentState, agent_name: str) -> AgentState:
        """Handle model timeout by updating state with error info."""
        error_info = {
            "type": "model_timeout",
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "message": "Model request timed out, consider using fallback model"
        }
        
        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_info)
        
        # Update scratchpad with recovery suggestion
        state["scratchpad"][f"{agent_name}_recovery"] = {
            "suggested_action": "retry_with_fallback",
            "error_type": "timeout"
        }
        
        return state
    
    @staticmethod
    async def handle_tool_failure(
        state: AgentState, 
        agent_name: str, 
        tool_name: str, 
        error: Exception
    ) -> AgentState:
        """Handle tool execution failure."""
        error_info = {
            "type": "tool_failure",
            "agent": agent_name,
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "message": str(error)
        }
        
        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_info)
        
        # Update scratchpad with alternative suggestions
        state["scratchpad"][f"{agent_name}_tool_failure"] = {
            "failed_tool": tool_name,
            "suggested_alternatives": "manual_fallback",
            "error_details": str(error)
        }
        
        return state
    
    @staticmethod
    async def handle_validation_failure(
        state: AgentState, 
        agent_name: str, 
        validation_errors: List[str]
    ) -> AgentState:
        """Handle agent validation failure."""
        error_info = {
            "type": "validation_failure",
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "validation_errors": validation_errors
        }
        
        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_info)
        
        # Update next routing to skip this agent
        state["scratchpad"][f"{agent_name}_validation_failed"] = True
        
        return state


class AgentNodeWrapper:
    """Wrapper that handles state updates and error recovery for agent nodes."""
    
    def __init__(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        model: ModelInterface,
        tools: List[BaseTool],
        capabilities: AgentCapabilities
    ):
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.model = model
        self.tools = tools
        self.capabilities = capabilities
        self.recovery_strategy = ErrorRecoveryStrategy()
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute the agent with error handling and state management."""
        try:
            # Validate agent can execute
            await self._validate_execution_requirements(state)
            
            # Update state to show this agent is active
            if self.agent_name not in state.get("active_agents", []):
                state.setdefault("active_agents", []).append(self.agent_name)
            
            # Update last activity timestamp
            state["last_updated"] = datetime.now()
            
            # Execute the agent logic
            updated_state = await self._execute_agent_logic(state)
            
            # Update scratchpad with agent's contribution
            updated_state["scratchpad"][f"{self.agent_name}_last_execution"] = {
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            return updated_state
            
        except asyncio.TimeoutError:
            logger.warning("Agent %s execution timed out", self.agent_name)
            return await self.recovery_strategy.handle_model_timeout(state, self.agent_name)
            
        except AgentValidationError as e:
            logger.error("Agent %s validation failed: %s", self.agent_name, e.message)
            return await self.recovery_strategy.handle_validation_failure(
                state, self.agent_name, [e.message]
            )
            
        except Exception as e:
            logger.error("Agent %s execution failed: %s", self.agent_name, str(e))
            return await self.recovery_strategy.handle_tool_failure(
                state, self.agent_name, "general_execution", e
            )
    
    async def _validate_execution_requirements(self, state: AgentState) -> None:
        """Validate that the agent can execute given current state and requirements."""
        validation_errors = []
        
        # Check model availability
        if not await self.model.is_available():
            validation_errors.append(f"Model {self.model.model_name} is not available")
        
        # Check required capabilities
        for capability in self.capabilities.required_capabilities:
            if capability not in self.agent_config.capabilities:
                validation_errors.append(f"Required capability '{capability}' not available")
        
        # Check multimodal requirements
        if "vision" in self.capabilities.model_requirements:
            if not state.get("multimodal_content"):
                validation_errors.append("Vision capability required but no multimodal content provided")
        
        # Check tool availability
        available_tool_names = [tool.name for tool in self.tools]
        for required_tool in self.capabilities.required_tools:
            if required_tool not in available_tool_names:
                validation_errors.append(f"Required tool '{required_tool}' not available")
        
        if validation_errors:
            raise AgentValidationError(
                self.agent_name, 
                "validation_failed", 
                "; ".join(validation_errors)
            )
    
    async def _execute_agent_logic(self, state: AgentState) -> AgentState:
        """Execute the core agent logic with the model and tools."""
        # Prepare messages for the model
        messages = self._prepare_messages(state)
        
        # Generate response from model
        try:
            response = await asyncio.wait_for(
                self.model.generate(messages),
                timeout=self.agent_config.max_tokens / 100  # Rough timeout estimation
            )
            
            # Create AI message from response
            ai_message = AIMessage(content=response.content)
            
            # Update state with new message
            state["messages"].append(ai_message)
            
            # Update scratchpad with agent's analysis
            state["scratchpad"][f"{self.agent_name}_analysis"] = {
                "response_length": len(response.content),
                "model_used": response.model_name,
                "provider": response.provider,
                "timestamp": datetime.now().isoformat()
            }
            
            return state
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise AgentExecutionError(
                self.agent_name,
                "model_execution_failed",
                f"Model execution failed: {str(e)}"
            )
    
    def _prepare_messages(self, state: AgentState) -> Sequence[BaseMessage]:
        """Prepare messages for model input including system prompt and context."""
        messages: List[BaseMessage] = []
        
        # Add system message with agent's prompt
        system_content = self.agent_config.system_prompt
        
        # Add context from scratchpad if relevant
        if state.get("scratchpad"):
            context_info = []
            for key, value in state["scratchpad"].items():
                if not key.startswith(f"{self.agent_name}_"):  # Don't include own previous work
                    context_info.append(f"{key}: {json.dumps(value, default=str)}")
            
            if context_info:
                system_content += "\n\nContext from other agents:\n" + "\n".join(context_info)
        
        messages.append(SystemMessage(content=system_content))
        
        # Add conversation history
        messages.extend(state.get("messages", []))
        
        # If no human message yet, add the task as initial message
        if not any(isinstance(msg, HumanMessage) for msg in messages):
            if state.get("task"):
                messages.append(HumanMessage(content=state["task"]))
        
        return messages


class AgentFactory:
    """Factory for creating and managing agent nodes."""
    
    def __init__(self, model_factory: Optional[ModelFactory] = None):
        self.model_factory = model_factory or get_model_factory()
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.agent_capabilities: Dict[str, AgentCapabilities] = {}
        self.agent_tools: Dict[str, List[BaseTool]] = {}
    
    def register_agent_config(
        self,
        config: AgentConfig,
        capabilities: AgentCapabilities,
        tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Register an agent configuration with its capabilities and tools."""
        self.agent_configs[config.name] = config
        self.agent_capabilities[config.name] = capabilities
        self.agent_tools[config.name] = tools or []
        
        logger.info("Registered agent: %s", config.name)
    
    async def create_agent_node(
        self,
        agent_name: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None
    ) -> AgentNodeWrapper:
        """
        Create a standardized agent node function compatible with LangGraph.
        
        Args:
            agent_name: Unique identifier for the agent
            system_prompt: Agent's specialized instructions (overrides config)
            tools: Available tools for the agent (overrides registered tools)
            model_config: Model selection and parameters (overrides config)
            capabilities: List of agent capabilities (overrides config)
        
        Returns:
            Callable agent node function compatible with LangGraph
        """
        # Get or validate agent configuration
        if agent_name not in self.agent_configs:
            raise ValueError(f"Agent '{agent_name}' not registered. Call register_agent_config first.")
        
        config = self.agent_configs[agent_name]
        agent_capabilities = self.agent_capabilities[agent_name]
        agent_tools = tools or self.agent_tools[agent_name]
        
        # Override config values if provided
        if system_prompt:
            config.system_prompt = system_prompt
        if capabilities:
            config.capabilities = capabilities
        
        # Get model for this agent
        model_name = f"{agent_name}_model"
        if model_name not in self.model_factory.list_models():
            # Register model if not already registered
            from ..core.models import ModelConfig, ModelProvider
            
            provider_map = {
                "openai": ModelProvider.OPENAI,
                "anthropic": ModelProvider.ANTHROPIC,
                "google": ModelProvider.GOOGLE
            }
            
            model_config_obj = ModelConfig(
                provider=provider_map[config.model_provider],
                model_name=config.model_name,
                api_key="",  # Will be set from environment
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            self.model_factory.register_model(model_name, model_config_obj)
        
        model = await self.model_factory.get_model(model_name)
        
        # Create and return wrapped agent node
        wrapper = AgentNodeWrapper(
            agent_name=agent_name,
            agent_config=config,
            model=model,
            tools=agent_tools,
            capabilities=agent_capabilities
        )
        
        return wrapper
    
    def load_agent_configurations(self, config_dir: Path) -> None:
        """Load agent configurations from JSON files in a directory."""
        if not config_dir.exists():
            logger.warning("Agent config directory does not exist: %s", config_dir)
            return
        
        for config_file in config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Parse agent config
                agent_config = AgentConfig(**config_data.get("config", {}))
                
                # Parse capabilities
                capabilities_data = config_data.get("capabilities", {})
                capabilities = AgentCapabilities(
                    required_capabilities=capabilities_data.get("required_capabilities", []),
                    optional_capabilities=capabilities_data.get("optional_capabilities", []),
                    model_requirements=capabilities_data.get("model_requirements", []),
                    required_tools=capabilities_data.get("required_tools", []),
                    mcp_requirements=capabilities_data.get("mcp_requirements", [])
                )
                
                # Register the agent
                self.register_agent_config(agent_config, capabilities)
                
                logger.info("Loaded agent configuration from %s", config_file.name)
                
            except Exception as e:
                logger.error("Failed to load agent config from %s: %s", config_file, str(e))
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agent_configs.keys())
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agent_configs.get(agent_name)
    
    def get_agent_capabilities(self, agent_name: str) -> Optional[AgentCapabilities]:
        """Get capabilities for a specific agent."""
        return self.agent_capabilities.get(agent_name)
    
    async def validate_agent(self, agent_name: str) -> Dict[str, Any]:
        """Validate that an agent can be created and executed."""
        if agent_name not in self.agent_configs:
            return {"valid": False, "errors": [f"Agent '{agent_name}' not registered"]}
        
        config = self.agent_configs[agent_name]
        capabilities = self.agent_capabilities[agent_name]
        errors = []
        
        # Check model availability
        try:
            model_name = f"{agent_name}_model"
            if model_name in self.model_factory.list_models():
                model = await self.model_factory.get_model(model_name)
                if not await model.is_available():
                    errors.append(f"Model {config.model_name} is not available")
        except Exception as e:
            errors.append(f"Model validation failed: {str(e)}")
        
        # Check tool availability
        available_tools = [tool.name for tool in self.agent_tools.get(agent_name, [])]
        for required_tool in capabilities.required_tools:
            if required_tool not in available_tools:
                errors.append(f"Required tool '{required_tool}' not available")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "config": config,
            "capabilities": capabilities
        }


# Global agent factory instance
agent_factory = AgentFactory()


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance."""
    return agent_factory
