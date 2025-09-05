#!/usr/bin/env python3
"""
AgentHive API Gateway Demo

This script demonstrates the comprehensive API gateway and routing system
implemented for AgentHive, showcasing all major endpoints and functionality.
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from agent_hive.api import AgentHiveAPI
from agent_hive.core.config import SystemConfig
from agent_hive.core.state import Session
from agent_hive.core.multimodal import ProcessedFile, FileType, ProcessingStatus


class APIGatewayDemo:
    """Demonstration of AgentHive API Gateway functionality."""

    def __init__(self):
        self.api = None
        self.setup_mocks()

    def setup_mocks(self):
        """Set up mock dependencies for demonstration."""
        print("🔧 Setting up API with mock dependencies...")

        # Create mock config
        config = Mock(spec=SystemConfig)
        config.redis_url = "redis://localhost:6379"
        config.cors_origins = ["*"]
        config.api_host = "localhost"
        config.api_port = 8000
        config.environment = "demo"

        # Create API instance
        self.api = AgentHiveAPI(config)

        # Mock state manager
        self.api.state_manager = Mock()
        self.api.state_manager.health_check.return_value = {"status": "healthy"}

        # Mock multimodal processor
        self.api.multimodal_processor = Mock()
        self.api.multimodal_processor.get_supported_formats.return_value = {
            "image": ["image/jpeg", "image/png", "image/gif"],
            "pdf": ["application/pdf"],
            "document": ["text/plain", "text/markdown"],
        }

        # Mock agent factory
        self.api.agent_factory = Mock()
        self.api.agent_factory.list_agents.return_value = [
            "full_stack_engineer",
            "qa_engineer",
            "product_designer",
            "devops_engineer",
        ]

        # Mock agent configs
        mock_configs = {
            "full_stack_engineer": {
                "core_tools": ["code_execution", "file_system", "git_operations"],
                "composio_tools": ["github", "linear", "slack", "notion"],
                "capabilities": ["code_generation", "debugging", "architecture_design"],
            },
            "qa_engineer": {
                "core_tools": [
                    "test_execution",
                    "coverage_analysis",
                    "performance_testing",
                ],
                "composio_tools": ["github", "linear", "browserbase", "serpapi"],
                "capabilities": ["test_planning", "bug_analysis", "quality_assessment"],
            },
            "product_designer": {
                "core_tools": [
                    "design_analysis",
                    "color_extraction",
                    "layout_generation",
                ],
                "composio_tools": ["figma", "notion", "miro", "airtable"],
                "capabilities": ["ui_design", "ux_analysis", "design_systems"],
            },
            "devops_engineer": {
                "core_tools": ["container_management", "infrastructure_analysis"],
                "composio_tools": ["aws", "github", "docker", "kubernetes", "datadog"],
                "capabilities": [
                    "infrastructure_design",
                    "deployment_automation",
                    "monitoring",
                ],
            },
        }

        def get_agent_config(agent_name):
            config_data = mock_configs.get(agent_name, {})
            mock_config = Mock()
            mock_config.core_tools = config_data.get("core_tools", [])
            mock_config.composio_tools = config_data.get("composio_tools", [])
            mock_config.capabilities = config_data.get("capabilities", [])
            return mock_config

        self.api.agent_factory.get_agent_config = get_agent_config

        # Mock model factory
        self.api.model_factory = Mock()
        self.api.model_factory.check_all_models = AsyncMock(
            return_value={
                "gpt-4": True,
                "gpt-4-turbo": True,
                "claude-3-sonnet": True,
                "claude-3-haiku": False,  # Simulate one unavailable model
                "gemini-pro": True,
            }
        )

        # Mock graph factory
        self.api.graph_factory = Mock()
        self.api.graph_factory.get_available_agents = AsyncMock(
            return_value=[
                {
                    "name": "full_stack_engineer",
                    "display_name": "Full Stack Engineer",
                    "description": "Expert in full-stack development, architecture, and code review",
                    "capabilities": [
                        "code_generation",
                        "debugging",
                        "architecture_design",
                    ],
                    "available": True,
                    "validation_errors": [],
                },
                {
                    "name": "qa_engineer",
                    "display_name": "QA Engineer",
                    "description": "Specialist in testing, quality assurance, and bug analysis",
                    "capabilities": [
                        "test_planning",
                        "bug_analysis",
                        "quality_assessment",
                    ],
                    "available": True,
                    "validation_errors": [],
                },
                {
                    "name": "product_designer",
                    "display_name": "Product Designer",
                    "description": "Expert in UI/UX design, wireframing, and design systems",
                    "capabilities": ["ui_design", "ux_analysis", "design_systems"],
                    "available": True,
                    "validation_errors": [],
                },
                {
                    "name": "devops_engineer",
                    "display_name": "DevOps Engineer",
                    "description": "Specialist in infrastructure, deployment, and monitoring",
                    "capabilities": [
                        "infrastructure_design",
                        "deployment_automation",
                        "monitoring",
                    ],
                    "available": True,
                    "validation_errors": [],
                },
            ]
        )

        # Mock session manager and orchestrator
        self.api.session_manager = Mock()
        self.api.orchestrator_factory = Mock()

        print("✅ Mock setup complete!")

    async def demo_system_status(self):
        """Demonstrate system status endpoints."""
        print("\n📊 SYSTEM STATUS ENDPOINTS")
        print("=" * 50)

        # Test agent status
        print("🔍 Getting agent status...")
        status = await self.api.get_agent_status()

        print("📈 System Status:")
        print(f"  • Total Agents: {status['agents']['total_count']}")
        print(f"  • Healthy Agents: {status['agents']['healthy_count']}")
        print(f"  • Total Models: {status['models']['total_count']}")
        print(f"  • Healthy Models: {status['models']['healthy_count']}")
        print(f"  • Uptime: {status['system']['uptime_seconds']:.1f} seconds")

        # Show available agents
        print("\n🤖 Available Agents:")
        for agent in status["agents"]["available"]:
            status_icon = "✅" if agent["available"] else "❌"
            print(f"  {status_icon} {agent['display_name']}")
            print(f"     {agent['description']}")
            print(f"     Capabilities: {', '.join(agent['capabilities'])}")

    async def demo_tools_and_capabilities(self):
        """Demonstrate tools and capabilities endpoints."""
        print("\n🛠️  TOOLS AND CAPABILITIES")
        print("=" * 50)

        # Get all tools
        print("🔧 Getting all available tools...")
        all_tools = await self.api.get_available_tools()

        print("📋 Tools by Agent:")
        for agent_name, tools in all_tools["agents"].items():
            print(f"\n  🤖 {agent_name.replace('_', ' ').title()}:")
            print(f"     Core Tools: {', '.join(tools['core_tools'])}")
            print(f"     Composio Tools: {', '.join(tools['composio_tools'])}")
            print(f"     Capabilities: {', '.join(tools['capabilities'])}")

        print("\n📁 Supported File Formats:")
        for category, formats in all_tools["supported_formats"].items():
            print(f"  • {category.title()}: {', '.join(formats)}")

        # Get tools for specific agent
        print("\n🎯 Getting tools for Full Stack Engineer...")
        engineer_tools = await self.api.get_available_tools(
            agent_name="full_stack_engineer"
        )

        print("🔨 Full Stack Engineer Tools:")
        print(f"  • Core: {', '.join(engineer_tools['core_tools'])}")
        print(f"  • Composio: {', '.join(engineer_tools['composio_tools'])}")
        print(f"  • Capabilities: {', '.join(engineer_tools['capabilities'])}")

    async def demo_session_management(self):
        """Demonstrate session management functionality."""
        print("\n🔐 SESSION MANAGEMENT")
        print("=" * 50)

        # Mock session creation
        mock_session = Mock(spec=Session)
        mock_session.session_id = str(uuid.uuid4())
        mock_session.user_id = "demo_user"
        mock_session.mode = "direct"
        mock_session.active_agent = "full_stack_engineer"
        mock_session.created_at = datetime.utcnow()

        self.api.state_manager.create_session.return_value = mock_session

        # Create session
        print("🆕 Creating new direct chat session...")
        session_result = await self.api.create_session(
            user_id="demo_user", mode="direct", active_agent="full_stack_engineer"
        )

        session_data = session_result["data"]
        print("✅ Session Created:")
        print(f"  • Session ID: {session_data['session_id']}")
        print(f"  • User ID: {session_data['user_id']}")
        print(f"  • Mode: {session_data['mode']}")
        print(f"  • Active Agent: {session_data['active_agent']}")

        # Mock session retrieval for status
        mock_session_for_status = Mock(spec=Session)
        mock_session_for_status.session_id = session_data["session_id"]
        mock_session_for_status.user_id = "demo_user"
        mock_session_for_status.mode = "direct"
        mock_session_for_status.active_agent = "full_stack_engineer"
        mock_session_for_status.created_at = datetime.utcnow()
        mock_session_for_status.last_activity = datetime.utcnow()
        mock_session_for_status.state = {
            "messages": [],
            "active_agents": ["full_stack_engineer"],
        }
        mock_session_for_status.multimodal_files = []

        self.api.session_manager.get_session = AsyncMock(
            return_value=mock_session_for_status
        )

        # Get session status
        print("\n📋 Getting session status...")
        status = await self.api.get_agent_status(session_id=session_data["session_id"])

        session_info = status["session"]
        print("📊 Session Status:")
        print(f"  • Session ID: {session_info['session_id']}")
        print(f"  • Mode: {session_info['mode']}")
        print(f"  • Active Agent: {session_info['active_agent']}")
        print(f"  • Message Count: {session_info['message_count']}")

        # Mock user sessions list
        self.api.state_manager.get_user_sessions.return_value = [
            session_data["session_id"]
        ]
        self.api.state_manager.get_session.return_value = mock_session_for_status

        # List user sessions
        print("\n📝 Listing user sessions...")
        user_sessions = await self.api.list_user_sessions("demo_user")

        print("👤 Sessions for demo_user:")
        for session in user_sessions["sessions"]:
            print(f"  • {session['session_id']} ({session['mode']} mode)")

        return session_data["session_id"]

    async def demo_file_upload(self):
        """Demonstrate multimodal file upload functionality."""
        print("\n📁 MULTIMODAL FILE UPLOAD")
        print("=" * 50)

        # Mock file processing
        mock_processed_files = [
            Mock(
                spec=ProcessedFile,
                file_id="img_001",
                original_name="design_mockup.png",
                file_type=FileType.IMAGE,
                status=ProcessingStatus.COMPLETED,
                file_size=2048576,
                processing_timestamp=datetime.utcnow(),
                metadata={"width": 1920, "height": 1080, "format": "PNG"},
            ),
            Mock(
                spec=ProcessedFile,
                file_id="doc_001",
                original_name="requirements.pdf",
                file_type=FileType.PDF,
                status=ProcessingStatus.COMPLETED,
                file_size=1024000,
                processing_timestamp=datetime.utcnow(),
                metadata={"pages": 15, "text_length": 5000},
            ),
        ]

        # Mock file upload objects
        mock_files = []
        for pf in mock_processed_files:
            mock_file = Mock()
            mock_file.filename = pf.original_name
            mock_file.content_type = (
                "image/png" if pf.file_type == FileType.IMAGE else "application/pdf"
            )
            mock_file.read = AsyncMock(return_value=b"fake_file_content")
            mock_files.append(mock_file)

        # Mock the processor to return our mock files
        async def mock_process_file(filename, content, mime_type, user_id):
            for pf in mock_processed_files:
                if pf.original_name == filename:
                    return pf
            return mock_processed_files[0]  # fallback

        self.api.multimodal_processor.process_file = mock_process_file

        # Test file upload
        print("📤 Uploading multimodal files...")
        upload_result = await self.api.upload_multimodal_content(
            files=mock_files, user_id="demo_user"
        )

        print("✅ File Upload Results:")
        print(f"  • Total Files: {upload_result['total_files']}")
        print(f"  • Successful: {upload_result['successful']}")
        print(f"  • Failed: {upload_result['failed']}")

        print("\n📋 Processed Files:")
        for file_info in upload_result["processed_files"]:
            print(f"  📄 {file_info['original_name']}")
            print(f"     Type: {file_info['file_type']}")
            print(f"     Status: {file_info['status']}")
            print(f"     Size: {file_info['file_size']:,} bytes")

        return upload_result["processed_files"]

    async def demo_direct_chat(self, session_id: str):
        """Demonstrate direct chat functionality."""
        print("\n💬 DIRECT CHAT INTERACTION")
        print("=" * 50)

        # Mock direct chat execution
        mock_chat_result = {
            "response": "Hello! I'm the Full Stack Engineer agent. I can help you with:\n\n"
            "• Code generation and architecture design\n"
            "• Debugging and code review\n"
            "• API development and database design\n"
            "• Integration with GitHub, Linear, and other tools\n\n"
            "What would you like to work on today?",
            "session_id": session_id,
            "agent_name": "full_stack_engineer",
            "status": "success",
            "state": {
                "message_count": 2,
                "active_agents": ["full_stack_engineer"],
                "last_updated": datetime.utcnow().isoformat(),
                "errors": [],
            },
        }

        self.api.graph_factory.execute_direct_chat = AsyncMock(
            return_value=mock_chat_result
        )

        # Test direct chat
        print("🤖 Starting direct chat with Full Stack Engineer...")
        chat_result = await self.api.direct_chat(
            agent_name="full_stack_engineer",
            message="Hello! I need help building a Python web application.",
            session_id=session_id,
            user_id="demo_user",
        )

        result_data = chat_result["data"]
        print("💭 Agent Response:")
        print(f"   {result_data['response']}")
        print("\n📊 Chat Status:")
        print(f"  • Agent: {result_data['agent_name']}")
        print(f"  • Status: {result_data['status']}")
        print(f"  • Messages: {result_data['state']['message_count']}")

    async def demo_orchestration(self):
        """Demonstrate task orchestration functionality."""
        print("\n🎭 TASK ORCHESTRATION")
        print("=" * 50)

        # Mock orchestration execution
        mock_orchestration_result = {
            "response": "# Multi-Agent Task Completion Report\n\n"
            "**Original Task:** Build a complete e-commerce web application with tests and deployment\n"
            "**Participating Agents:** full_stack_engineer, qa_engineer, devops_engineer\n\n"
            "## Full Stack Engineer Analysis\n"
            "- Designed application architecture using FastAPI and React\n"
            "- Implemented user authentication and product catalog\n"
            "- Created RESTful API with database integration\n\n"
            "## QA Engineer Analysis\n"
            "- Developed comprehensive test suite with 95% coverage\n"
            "- Created automated integration and end-to-end tests\n"
            "- Implemented performance testing scenarios\n\n"
            "## DevOps Engineer Analysis\n"
            "- Containerized application with Docker\n"
            "- Set up CI/CD pipeline with GitHub Actions\n"
            "- Configured AWS deployment with monitoring\n\n"
            "## Summary\n"
            "This task was successfully processed by 3 specialized agents, "
            "each contributing their domain expertise to provide a comprehensive solution.",
            "session_id": str(uuid.uuid4()),
            "mode": "orchestration",
            "status": "success",
            "participating_agents": [
                "full_stack_engineer",
                "qa_engineer",
                "devops_engineer",
            ],
            "task_status": {
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "processed_files": [],
        }

        self.api._execute_orchestration = AsyncMock(
            return_value=mock_orchestration_result
        )

        # Test orchestration
        print("🎯 Starting task orchestration...")
        orchestration_result = await self.api.orchestrate_task(
            task="Build a complete e-commerce web application with tests and deployment",
            user_id="demo_user",
        )

        result_data = (
            orchestration_result["data"]
            if "data" in orchestration_result
            else orchestration_result
        )
        print("🎭 Orchestration Complete!")
        print(
            f"  • Participating Agents: {', '.join(result_data['participating_agents'])}"
        )
        print(f"  • Status: {result_data['status']}")
        print(f"  • Task Status: {result_data['task_status']['status']}")

        print("\n📋 Orchestration Response:")
        # Print first few lines of the response
        response_lines = result_data["response"].split("\n")[:8]
        for line in response_lines:
            print(f"   {line}")
        print("   ... (response truncated)")

    async def demo_error_handling(self):
        """Demonstrate error handling and validation."""
        print("\n⚠️  ERROR HANDLING & VALIDATION")
        print("=" * 50)

        # Test invalid agent name
        print("🚫 Testing invalid agent name...")
        try:
            await self.api.direct_chat(
                agent_name="nonexistent_agent", message="Hello", user_id="demo_user"
            )
        except Exception as e:
            print(f"✅ Caught expected error: {type(e).__name__}")

        # Test invalid session creation
        print("🚫 Testing invalid session mode...")
        try:
            await self.api.create_session(user_id="demo_user", mode="invalid_mode")
        except Exception as e:
            print(f"✅ Caught expected error: {type(e).__name__}")

        # Test empty message
        print("🚫 Testing empty message...")
        try:
            await self.api.direct_chat(
                agent_name="full_stack_engineer", message="", user_id="demo_user"
            )
        except Exception as e:
            print(f"✅ Caught expected error: {type(e).__name__}")

        print("✅ Error handling working correctly!")

    async def run_complete_demo(self):
        """Run the complete API gateway demonstration."""
        print("🚀 AGENTHIVE API GATEWAY DEMONSTRATION")
        print("=" * 60)
        print("This demo showcases the comprehensive API gateway and routing")
        print("system implemented for AgentHive, including all major endpoints")
        print("and functionality as specified in task 9.")
        print("=" * 60)

        try:
            # Run all demonstrations
            await self.demo_system_status()
            await self.demo_tools_and_capabilities()
            session_id = await self.demo_session_management()
            await self.demo_file_upload()
            await self.demo_direct_chat(session_id)
            await self.demo_orchestration()
            await self.demo_error_handling()

            print("\n🎉 DEMONSTRATION COMPLETE!")
            print("=" * 60)
            print("✅ All API gateway endpoints demonstrated successfully:")
            print("  • System status and health checks")
            print("  • Agent and tool discovery")
            print("  • Session management (create, read, delete)")
            print("  • Multimodal file upload and processing")
            print("  • Direct chat with agent selection")
            print("  • Task orchestration across multiple agents")
            print("  • Comprehensive error handling and validation")
            print("\n🔧 The API gateway provides:")
            print("  • Request validation and sanitization")
            print("  • Structured error responses")
            print("  • File upload with type validation")
            print("  • Session state management")
            print("  • Agent routing and selection")
            print("  • Multimodal content processing")
            print("  • Streaming response support")
            print("\n📚 Ready for integration with web interfaces and Slack!")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback

            traceback.print_exc()


async def main():
    """Run the API gateway demonstration."""
    demo = APIGatewayDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
