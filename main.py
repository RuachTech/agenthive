"""Entry point for AgentHive application."""

import uvicorn
import os
from dotenv import load_dotenv

from agent_hive.api import app
from agent_hive.core.config import SystemConfig

# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the application."""
    # Create system configuration from environment variables
    config = SystemConfig(
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_workers=int(os.getenv("API_WORKERS", "1")),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
    )

    # Update app configuration
    app.state.config = config

    # Run the application
    uvicorn.run(
        "agent_hive.api.main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.log_level.lower(),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
    )


if __name__ == "__main__":
    main()
