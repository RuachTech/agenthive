"""Tests for multimodal integration with agent state."""

import base64
import pytest
from datetime import datetime
from io import BytesIO
from unittest.mock import patch
from PIL import Image

from agent_hive.core.multimodal_integration import MultimodalStateManager
from agent_hive.core.multimodal import ProcessedFile, FileType, ProcessingStatus
from agent_hive.core.state import AgentState
from langchain_core.messages import SystemMessage


class TestMultimodalStateManager:
    """Test MultimodalStateManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MultimodalStateManager()

    @pytest.fixture
    def sample_state(self):
        """Create sample agent state."""
        return AgentState(
            task="Test task",
            messages=[],
            next="",
            scratchpad={},
            mode="direct",
            active_agents=[],
            multimodal_content={},
            session_id="test-session",
            user_id="test-user",
            last_updated=datetime.utcnow(),
            errors=[],
            task_status={},
        )

    @pytest.fixture
    def sample_processed_files(self):
        """Create sample processed files."""
        # Create image file
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        image_file = ProcessedFile(
            file_id="image-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={
                "encoded_image": encoded_image,
                "metadata": {"width": 100, "height": 100, "format": "JPEG"},
            },
            metadata={"user_id": "test_user"},
            processing_timestamp=datetime.utcnow(),
            status=ProcessingStatus.COMPLETED,
        )

        # Create text file
        text_file = ProcessedFile(
            file_id="text-456",
            original_name="document.txt",
            file_type=FileType.DOCUMENT,
            mime_type="text/plain",
            file_size=500,
            processed_content={
                "text_content": "This is a test document with some content.",
                "extraction_available": True,
                "metadata": {"word_count": 8, "line_count": 1},
            },
            metadata={"user_id": "test_user"},
            processing_timestamp=datetime.utcnow(),
            status=ProcessingStatus.COMPLETED,
        )

        return [image_file, text_file]

    def test_add_files_to_state(self, sample_state, sample_processed_files):
        """Test adding processed files to agent state."""
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Verify files were added
        assert "files" in updated_state["multimodal_content"]
        assert len(updated_state["multimodal_content"]["files"]) == 2

        # Verify file data
        files = updated_state["multimodal_content"]["files"]
        assert "image-123" in files
        assert "text-456" in files

        image_data = files["image-123"]
        assert image_data["original_name"] == "test.jpg"
        assert image_data["file_type"] == "image"
        assert image_data["status"] == "completed"

        text_data = files["text-456"]
        assert text_data["original_name"] == "document.txt"
        assert text_data["file_type"] == "document"

        # Verify metadata
        assert updated_state["multimodal_content"]["total_files"] == 2
        assert set(updated_state["multimodal_content"]["file_types"]) == {
            "image",
            "document",
        }

    def test_add_files_to_existing_state(self, sample_state, sample_processed_files):
        """Test adding files to state that already has multimodal content."""
        # Pre-populate state with some content
        sample_state["multimodal_content"] = {
            "files": {"existing-file": {"file_type": "pdf"}},
            "analyses": {},
        }

        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Verify new files were added alongside existing
        assert len(updated_state["multimodal_content"]["files"]) == 3
        assert "existing-file" in updated_state["multimodal_content"]["files"]
        assert "image-123" in updated_state["multimodal_content"]["files"]
        assert "text-456" in updated_state["multimodal_content"]["files"]

    @pytest.mark.asyncio
    async def test_analyze_images_in_state(self, sample_state, sample_processed_files):
        """Test analyzing images in agent state."""
        # Add files to state first
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Mock vision analyzer
        mock_analysis = {
            "analysis": "This is a blue square image",
            "model_used": "gpt-4-vision",
            "analysis_type": "general",
            "confidence": "high",
        }

        with patch.object(
            self.manager.vision_analyzer, "analyze_multiple_images"
        ) as mock_analyze:
            mock_analyze.return_value = [mock_analysis]

            result_state = await self.manager.analyze_images_in_state(updated_state)

        # Verify analysis was added
        assert "analyses" in result_state["multimodal_content"]
        assert "image-123" in result_state["multimodal_content"]["analyses"]

        analysis = result_state["multimodal_content"]["analyses"]["image-123"]
        assert analysis["analysis"] == "This is a blue square image"
        assert analysis["model_used"] == "gpt-4-vision"

    @pytest.mark.asyncio
    async def test_analyze_images_no_new_images(self, sample_state):
        """Test analyzing images when no new images are present."""
        # State with no images
        result_state = await self.manager.analyze_images_in_state(sample_state)

        # State should be unchanged
        assert result_state == sample_state

    @pytest.mark.asyncio
    async def test_analyze_images_error_handling(
        self, sample_state, sample_processed_files
    ):
        """Test error handling during image analysis."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Mock vision analyzer to raise error
        with patch.object(
            self.manager.vision_analyzer, "analyze_multiple_images"
        ) as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            result_state = await self.manager.analyze_images_in_state(updated_state)

        # Verify error was recorded
        assert "analysis_error" in result_state["multimodal_content"]
        assert "Analysis failed" in result_state["multimodal_content"]["analysis_error"]

    def test_create_multimodal_context_message(
        self, sample_state, sample_processed_files
    ):
        """Test creating context message for multimodal content."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Add mock analysis for image
        updated_state["multimodal_content"]["analyses"] = {
            "image-123": {
                "analysis": "This image shows a blue square with clean geometric lines.",
                "error": False,
            }
        }

        message = self.manager.create_multimodal_context_message(updated_state)

        assert isinstance(message, SystemMessage)
        content = message.content

        # Verify content includes file descriptions
        assert "multimodal content is available" in content
        assert "test.jpg" in content
        assert "document.txt" in content
        assert "blue square" in content  # From analysis
        assert "This is a test document" in content  # From text content

    def test_create_context_message_no_content(self, sample_state):
        """Test creating context message when no multimodal content exists."""
        message = self.manager.create_multimodal_context_message(sample_state)
        assert message is None

    def test_get_file_content(self, sample_state, sample_processed_files):
        """Test retrieving specific file content."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Get image file content
        image_content = self.manager.get_file_content(updated_state, "image-123")
        assert image_content is not None
        assert image_content["original_name"] == "test.jpg"
        assert image_content["file_type"] == "image"

        # Get non-existent file
        missing_content = self.manager.get_file_content(updated_state, "missing-file")
        assert missing_content is None

    def test_get_files_by_type(self, sample_state, sample_processed_files):
        """Test retrieving files by type."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Get image files
        image_files = self.manager.get_files_by_type(updated_state, "image")
        assert len(image_files) == 1
        assert image_files[0]["original_name"] == "test.jpg"

        # Get document files
        doc_files = self.manager.get_files_by_type(updated_state, "document")
        assert len(doc_files) == 1
        assert doc_files[0]["original_name"] == "document.txt"

        # Get non-existent type
        pdf_files = self.manager.get_files_by_type(updated_state, "pdf")
        assert len(pdf_files) == 0

    def test_get_text_content_summary(self, sample_state, sample_processed_files):
        """Test getting text content summary."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        summary = self.manager.get_text_content_summary(updated_state)

        # Should include text from document
        assert "document.txt" in summary
        assert "This is a test document" in summary

    def test_clear_multimodal_content(self, sample_state, sample_processed_files):
        """Test clearing multimodal content from state."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )
        assert len(updated_state["multimodal_content"]["files"]) == 2

        # Clear content
        cleared_state = self.manager.clear_multimodal_content(updated_state)

        # Verify content was cleared
        assert cleared_state["multimodal_content"] == {}

    def test_get_content_statistics(self, sample_state, sample_processed_files):
        """Test getting content statistics."""
        # Add files to state
        updated_state = self.manager.add_files_to_state(
            sample_state, sample_processed_files
        )

        # Add mock analysis
        updated_state["multimodal_content"]["analyses"] = {
            "image-123": {"analysis": "test"}
        }

        stats = self.manager.get_content_statistics(updated_state)

        assert stats["total_files"] == 2
        assert stats["file_types"]["image"] == 1
        assert stats["file_types"]["document"] == 1
        assert stats["total_size"] == 1524  # 1024 + 500
        assert stats["analyzed_images"] == 1
        assert stats["text_documents"] == 1
        assert stats["total_text_length"] > 0


class TestMultimodalIntegrationScenarios:
    """Test realistic multimodal integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MultimodalStateManager()

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete multimodal processing workflow."""
        # Create initial state
        state = AgentState(
            task="Analyze uploaded files",
            messages=[],
            next="",
            scratchpad={},
            mode="direct",
            active_agents=["analyst"],
            multimodal_content={},
            session_id="workflow-test",
            user_id="test-user",
            last_updated=datetime.utcnow(),
            errors=[],
            task_status={},
        )

        # Create test files
        img = Image.new("RGB", (150, 100), color="green")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        files = [
            ProcessedFile(
                file_id="workflow-image",
                original_name="chart.png",
                file_type=FileType.IMAGE,
                mime_type="image/png",
                file_size=2048,
                processed_content={
                    "encoded_image": encoded_image,
                    "metadata": {"width": 150, "height": 100, "format": "PNG"},
                },
                metadata={"user_id": "test-user"},
                processing_timestamp=datetime.utcnow(),
                status=ProcessingStatus.COMPLETED,
            ),
            ProcessedFile(
                file_id="workflow-doc",
                original_name="report.txt",
                file_type=FileType.DOCUMENT,
                mime_type="text/plain",
                file_size=1000,
                processed_content={
                    "text_content": "Executive Summary: This report analyzes market trends.",
                    "extraction_available": True,
                    "metadata": {"word_count": 8, "line_count": 1},
                },
                metadata={"user_id": "test-user"},
                processing_timestamp=datetime.utcnow(),
                status=ProcessingStatus.COMPLETED,
            ),
        ]

        # Step 1: Add files to state
        state = self.manager.add_files_to_state(state, files)

        # Step 2: Analyze images
        with patch.object(
            self.manager.vision_analyzer, "analyze_multiple_images"
        ) as mock_analyze:
            mock_analyze.return_value = [
                {
                    "analysis": "This chart shows market growth trends with green bars indicating positive performance.",
                    "model_used": "gpt-4-vision",
                    "analysis_type": "general",
                    "confidence": "high",
                }
            ]

            state = await self.manager.analyze_images_in_state(state)

        # Step 3: Create context message
        context_message = self.manager.create_multimodal_context_message(state)

        # Step 4: Get statistics
        stats = self.manager.get_content_statistics(state)

        # Verify complete workflow
        assert len(state["multimodal_content"]["files"]) == 2
        assert len(state["multimodal_content"]["analyses"]) == 1
        assert context_message is not None
        assert "chart.png" in context_message.content
        assert "market growth trends" in context_message.content
        assert "Executive Summary" in context_message.content

        assert stats["total_files"] == 2
        assert stats["analyzed_images"] == 1
        assert stats["text_documents"] == 1

    def test_mixed_file_processing_results(self):
        """Test handling mixed successful and failed file processing."""
        state = AgentState(
            task="Process mixed files",
            messages=[],
            next="",
            scratchpad={},
            mode="orchestration",
            active_agents=[],
            multimodal_content={},
            session_id="mixed-test",
            user_id=None,
            last_updated=None,
            errors=[],
            task_status={},
        )

        # Create files with mixed success/failure
        files = [
            ProcessedFile(
                file_id="success-file",
                original_name="good.jpg",
                file_type=FileType.IMAGE,
                mime_type="image/jpeg",
                file_size=1024,
                processed_content={"encoded_image": "base64data", "metadata": {}},
                metadata={},
                processing_timestamp=datetime.utcnow(),
                status=ProcessingStatus.COMPLETED,
            ),
            ProcessedFile(
                file_id="failed-file",
                original_name="bad.jpg",
                file_type=FileType.UNKNOWN,
                mime_type="image/jpeg",
                file_size=0,
                processed_content={},
                metadata={},
                processing_timestamp=datetime.utcnow(),
                status=ProcessingStatus.FAILED,
                error_message="File corrupted",
            ),
        ]

        # Add files to state
        updated_state = self.manager.add_files_to_state(state, files)

        # Verify both files are recorded
        assert len(updated_state["multimodal_content"]["files"]) == 2

        # Verify success file
        success_file = updated_state["multimodal_content"]["files"]["success-file"]
        assert success_file["status"] == "completed"
        assert "error_message" not in success_file

        # Verify failed file
        failed_file = updated_state["multimodal_content"]["files"]["failed-file"]
        assert failed_file["status"] == "failed"
        assert failed_file["error_message"] == "File corrupted"

        # Statistics should reflect mixed results
        stats = self.manager.get_content_statistics(updated_state)
        assert stats["total_files"] == 2
