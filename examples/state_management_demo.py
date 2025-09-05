#!/usr/bin/env python3
"""
Demo script showing AgentHive session and state management functionality.

This script demonstrates:
1. Creating and managing user sessions
2. State persistence and recovery
3. Multimodal content handling
4. Session expiration and cleanup
5. Error handling and validation
"""

import json
from datetime import datetime

from agent_hive.core.state import StateManager, Session, ProcessedFile
from langchain_core.messages import HumanMessage, AIMessage


class StateManagementDemo:
    """Demo class for state management functionality."""

    def __init__(self):
        # Initialize with mock Redis for demo (in production, use real Redis)
        self.state_manager = StateManager(
            redis_url="redis://localhost:6379", session_timeout_hours=24
        )
        print("🚀 AgentHive State Management Demo")
        print("=" * 50)

    def demo_session_creation(self):
        """Demonstrate creating different types of sessions."""
        print("\n📝 Creating Sessions")
        print("-" * 30)

        # Create direct mode session
        direct_session = self.state_manager.create_session(
            user_id="developer_alice", mode="direct", active_agent="full_stack_dev"
        )
        print(f"✅ Created direct session: {direct_session.session_id}")
        print(f"   User: {direct_session.user_id}")
        print(f"   Mode: {direct_session.mode}")
        print(f"   Active Agent: {direct_session.active_agent}")

        # Create orchestration mode session
        orchestration_session = self.state_manager.create_session(
            user_id="project_manager_bob", mode="orchestration"
        )
        print(f"✅ Created orchestration session: {orchestration_session.session_id}")
        print(f"   User: {orchestration_session.user_id}")
        print(f"   Mode: {orchestration_session.mode}")

        return direct_session, orchestration_session

    def demo_state_updates(self, session: Session):
        """Demonstrate updating session state with agent interactions."""
        print(f"\n🔄 Updating Session State: {session.session_id}")
        print("-" * 40)

        # Simulate user message
        session.state["task"] = "Create a React login component"
        session.state["messages"].append(
            HumanMessage(
                content="I need a login form with email and password validation"
            )
        )

        # Simulate agent processing
        session.state["scratchpad"]["analysis"] = {
            "requirements": ["email validation", "password strength", "form styling"],
            "approach": "React functional component with hooks",
            "estimated_time": "30 minutes",
        }

        # Simulate agent response
        session.state["messages"].append(
            AIMessage(
                content="I'll create a React login component with proper validation. Let me start with the form structure..."
            )
        )

        # Update session
        success = self.state_manager.update_session(session)
        print(f"✅ Session updated: {success}")
        print(f"   Task: {session.state['task']}")
        print(f"   Messages: {len(session.state['messages'])}")
        print(f"   Analysis: {session.state['scratchpad']['analysis']['approach']}")

    def demo_multimodal_content(self, session: Session):
        """Demonstrate handling multimodal content in sessions."""
        print(f"\n🖼️  Adding Multimodal Content: {session.session_id}")
        print("-" * 45)

        # Create processed file
        design_file = ProcessedFile(
            file_id="design-mockup-001",
            original_name="login_mockup.png",
            file_type="image",
            processed_content={
                "description": "Login form mockup with modern UI design",
                "elements_detected": [
                    "email input field",
                    "password input field",
                    "login button",
                    "forgot password link",
                ],
                "color_scheme": ["#007bff", "#ffffff", "#f8f9fa", "#6c757d"],
                "dimensions": {"width": 400, "height": 300},
            },
            metadata={
                "file_size": 156789,
                "image_format": "PNG",
                "uploaded_by": session.user_id,
            },
            processing_timestamp=datetime.utcnow(),
        )

        # Add to session
        session.multimodal_files.append(design_file)
        session.state["multimodal_content"]["current_design"] = {
            "file_id": design_file.file_id,
            "analysis": "Modern login form with clean, professional design",
            "implementation_notes": "Use similar color scheme and layout structure",
        }

        # Update session
        success = self.state_manager.update_session(session)
        print(f"✅ Multimodal content added: {success}")
        print(f"   File: {design_file.original_name}")
        print(f"   Type: {design_file.file_type}")
        print(f"   Elements: {len(design_file.processed_content['elements_detected'])}")
        print(f"   Colors: {design_file.processed_content['color_scheme']}")

    def demo_session_retrieval(self, session_id: str):
        """Demonstrate retrieving and validating sessions."""
        print(f"\n🔍 Retrieving Session: {session_id}")
        print("-" * 35)

        retrieved_session = self.state_manager.get_session(session_id)

        if retrieved_session:
            print("✅ Session retrieved successfully")
            print(f"   User: {retrieved_session.user_id}")
            print(f"   Mode: {retrieved_session.mode}")
            print(f"   Task: {retrieved_session.state['task']}")
            print(f"   Messages: {len(retrieved_session.state['messages'])}")
            print(f"   Files: {len(retrieved_session.multimodal_files)}")
            print(f"   Last Activity: {retrieved_session.last_activity}")

            # Show state validation
            try:
                self.state_manager._validate_session_state(retrieved_session)
                print("✅ State validation passed")
            except Exception as e:
                print(f"❌ State validation failed: {e}")
        else:
            print("❌ Session not found or expired")

        return retrieved_session

    def demo_user_sessions(self, user_id: str):
        """Demonstrate getting all sessions for a user."""
        print(f"\n👤 User Sessions: {user_id}")
        print("-" * 30)

        session_ids = self.state_manager.get_user_sessions(user_id)
        print(f"✅ Found {len(session_ids)} active sessions")

        for i, session_id in enumerate(session_ids, 1):
            session = self.state_manager.get_session(session_id)
            if session:
                print(f"   {i}. {session_id[:8]}... ({session.mode} mode)")

    def demo_orchestration_workflow(self, session: Session):
        """Demonstrate orchestration mode state management."""
        print(f"\n🎯 Orchestration Workflow: {session.session_id}")
        print("-" * 45)

        # Set up orchestration task
        session.state["task"] = "Build a complete e-commerce website"
        session.state["active_agents"] = [
            "product_designer",
            "full_stack_dev",
            "qa_engineer",
        ]
        session.state["next"] = "task_analyzer"

        # Add orchestration-specific state
        session.state["scratchpad"]["orchestration"] = {
            "task_breakdown": {
                "design_phase": {
                    "agent": "product_designer",
                    "status": "in_progress",
                    "deliverables": ["wireframes", "mockups", "design_system"],
                },
                "development_phase": {
                    "agent": "full_stack_dev",
                    "status": "pending",
                    "dependencies": ["design_phase"],
                },
                "testing_phase": {
                    "agent": "qa_engineer",
                    "status": "pending",
                    "dependencies": ["development_phase"],
                },
            },
            "coordination_notes": "Sequential workflow with design → dev → testing",
        }

        session.state["task_status"] = {
            "overall_progress": "25%",
            "current_phase": "design",
            "estimated_completion": "2024-02-15",
        }

        # Update session
        success = self.state_manager.update_session(session)
        print(f"✅ Orchestration state updated: {success}")
        print(f"   Task: {session.state['task']}")
        print(f"   Active Agents: {len(session.state['active_agents'])}")
        print(f"   Current Phase: {session.state['task_status']['current_phase']}")
        print(f"   Progress: {session.state['task_status']['overall_progress']}")

    def demo_error_handling(self, session: Session):
        """Demonstrate error handling and recovery."""
        print(f"\n⚠️  Error Handling: {session.session_id}")
        print("-" * 35)

        # Simulate agent error
        error_info = {
            "agent": "full_stack_dev",
            "error_type": "model_timeout",
            "message": "OpenAI API timeout after 30 seconds",
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_action": "switched_to_anthropic_claude",
        }

        session.state["errors"].append(error_info)

        # Add recovery information
        session.state["scratchpad"]["error_recovery"] = {
            "original_model": "gpt-4",
            "fallback_model": "claude-3-sonnet",
            "retry_count": 1,
            "recovery_successful": True,
        }

        # Update session
        success = self.state_manager.update_session(session)
        print(f"✅ Error information recorded: {success}")
        print(f"   Error Type: {error_info['error_type']}")
        print(f"   Recovery Action: {error_info['recovery_action']}")
        print(
            f"   Fallback Model: {session.state['scratchpad']['error_recovery']['fallback_model']}"
        )

    def demo_session_cleanup(self):
        """Demonstrate session cleanup and expiration."""
        print("\n🧹 Session Cleanup")
        print("-" * 25)

        # Check system health
        health = self.state_manager.health_check()
        print(f"✅ System Health: {health['status']}")
        print(f"   Total Sessions: {health.get('total_sessions', 'N/A')}")

        # Cleanup expired sessions
        cleaned_count = self.state_manager.cleanup_expired_sessions()
        print(f"✅ Cleaned up {cleaned_count} expired sessions")

    def demo_state_serialization(self, session: Session):
        """Demonstrate state serialization and deserialization."""
        print(f"\n💾 State Serialization: {session.session_id}")
        print("-" * 40)

        # Convert to dictionary
        session_dict = session.to_dict()
        print("✅ Session serialized to dictionary")
        print(f"   Keys: {list(session_dict.keys())}")

        # Convert to JSON
        json_str = json.dumps(session_dict, indent=2, default=str)
        print(f"✅ Session serialized to JSON ({len(json_str)} characters)")

        # Deserialize back
        restored_session = Session.from_dict(session_dict)
        print("✅ Session deserialized successfully")
        print(
            f"   Session ID matches: {restored_session.session_id == session.session_id}"
        )
        print(
            f"   State matches: {restored_session.state['task'] == session.state['task']}"
        )

    def run_demo(self):
        """Run the complete state management demo."""
        try:
            # Create sessions
            direct_session, orchestration_session = self.demo_session_creation()

            # Demo direct mode workflow
            self.demo_state_updates(direct_session)
            self.demo_multimodal_content(direct_session)
            self.demo_error_handling(direct_session)

            # Demo orchestration mode workflow
            self.demo_orchestration_workflow(orchestration_session)

            # Demo session management
            self.demo_session_retrieval(direct_session.session_id)
            self.demo_user_sessions(direct_session.user_id)

            # Demo serialization
            self.demo_state_serialization(direct_session)

            # Demo cleanup
            self.demo_session_cleanup()

            print("\n🎉 Demo completed successfully!")
            print("=" * 50)

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main demo function."""
    print("Starting AgentHive State Management Demo...")
    print("Note: This demo uses mock Redis. In production, ensure Redis is running.")

    demo = StateManagementDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
