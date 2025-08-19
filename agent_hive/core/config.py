"""Configuration models for AgentHive."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class AgentConfig:
    """Configuration for individual agents with model, tools, and MCP settings."""

    # Agent identification
    name: str
    display_name: str
    description: str

    # Agent behavior and instructions
    system_prompt: str

    # Model configuration
    model_provider: str  # "openai", "anthropic", "google"
    model_name: str
    max_tokens: int = 4000
    temperature: float = 0.7

    # Agent capabilities and specialization
    capabilities: List[str] = field(
        default_factory=list
    )  # ["vision", "code_execution", "web_search"]
    specialized_for: List[str] = field(default_factory=list)  # Domain expertise areas

    # Tool configurations
    core_tools: List[str] = field(default_factory=list)  # Built-in system tools
    composio_tools: List[str] = field(
        default_factory=list
    )  # Composio/MCP tool categories

    # MCP server configurations
    mcp_server_configs: Dict[str, Any] = field(default_factory=dict)

    # Agent metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"

    # Runtime configuration
    enabled: bool = True
    priority: int = 1  # Higher numbers = higher priority in orchestration

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Agent name is required")

        if self.model_provider not in ["openai", "anthropic", "google"]:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class SystemConfig:
    """System-wide configuration for AgentHive."""

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # State management
    redis_url: str = "redis://localhost:6379"
    session_timeout: int = 3600  # seconds

    # File processing
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "application/pdf",
            "text/plain",
            "text/markdown",
        ]
    )

    # Model fallback configuration
    model_timeout: int = 30  # seconds
    max_retries: int = 3

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True

    # Security
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False

    def __post_init__(self) -> None:
        """Validate system configuration."""
        if self.api_port < 1 or self.api_port > 65535:
            raise ValueError("API port must be between 1 and 65535")

        if self.session_timeout <= 0:
            raise ValueError("Session timeout must be positive")

        if self.max_file_size <= 0:
            raise ValueError("Max file size must be positive")
