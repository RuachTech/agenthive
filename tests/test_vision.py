"""Tests for vision analysis module."""

import base64
import pytest
from datetime import datetime
from io import BytesIO
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image

from agent_hive.core.vision import VisionAnalyzer
from agent_hive.core.multimodal import ProcessedFile, FileType, ProcessingStatus
from agent_hive.core.models import ModelResponse, ModelFactory


class TestVisionAnalyzer:
    """Test VisionAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model_factory = Mock(spec=ModelFactory)
        self.analyzer = VisionAnalyzer(self.mock_model_factory)

    @pytest.fixture
    def sample_processed_image(self):
        """Create sample processed image file."""
        # Create test image
        img = Image.new("RGB", (200, 150), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        return ProcessedFile(
            file_id="test-image-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={
                "encoded_image": encoded_image,
                "metadata": {"width": 200, "height": 150, "format": "JPEG"},
                "analysis_ready": True,
            },
            metadata={"user_id": "test_user"},
            processing_timestamp=datetime.utcnow(),
            status=ProcessingStatus.COMPLETED,
        )

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, sample_processed_image):
        """Test successful image analysis."""
        # Mock model response
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="This image shows a red rectangle with dimensions 200x150 pixels.",
            provider="openai",
            model_name="gpt-4-vision",
            usage={"tokens": 100},
        )
        mock_model.generate.return_value = mock_response
        self.mock_model_factory.get_model.return_value = mock_model

        # Analyze image
        result = await self.analyzer.analyze_image(sample_processed_image)

        # Verify result
        assert (
            result["analysis"]
            == "This image shows a red rectangle with dimensions 200x150 pixels."
        )
        assert result["model_used"] == "gpt-4-vision"
        assert result["provider"] == "openai"
        assert result["analysis_type"] == "general"
        assert result["confidence"] == "high"
        assert "image_metadata" in result

        # Verify model was called correctly
        self.mock_model_factory.get_model.assert_called_once_with("gpt-4-vision")
        mock_model.generate.assert_called_once()

        # Check message structure
        call_args = mock_model.generate.call_args[0][
            0
        ]  # First positional argument (messages)
        message = call_args[0]
        assert len(message.content) == 2  # Text and image parts
        assert message.content[0]["type"] == "text"
        assert message.content[1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_analyze_image_custom_prompt(self, sample_processed_image):
        """Test image analysis with custom prompt."""
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="Custom analysis result",
            provider="openai",
            model_name="gpt-4-vision",
        )
        mock_model.generate.return_value = mock_response
        self.mock_model_factory.get_model.return_value = mock_model

        custom_prompt = "Describe the colors in this image"
        result = await self.analyzer.analyze_image(
            sample_processed_image, custom_prompt=custom_prompt
        )

        assert result["prompt_used"] == custom_prompt

        # Verify custom prompt was used
        call_args = mock_model.generate.call_args[0][0]
        message = call_args[0]
        assert message.content[0]["text"] == custom_prompt

    @pytest.mark.asyncio
    async def test_analyze_image_design_type(self, sample_processed_image):
        """Test image analysis with design analysis type."""
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="Design analysis result",
            provider="openai",
            model_name="gpt-4-vision",
        )
        mock_model.generate.return_value = mock_response
        self.mock_model_factory.get_model.return_value = mock_model

        result = await self.analyzer.analyze_image(
            sample_processed_image, analysis_type="design"
        )

        assert result["analysis_type"] == "design"

        # Verify design prompt was used
        call_args = mock_model.generate.call_args[0][0]
        message = call_args[0]
        assert "design document" in message.content[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_analyze_non_image_file(self):
        """Test analysis failure for non-image file."""
        non_image_file = ProcessedFile(
            file_id="test-doc-123",
            original_name="test.txt",
            file_type=FileType.DOCUMENT,
            mime_type="text/plain",
            file_size=100,
            processed_content={"text_content": "test"},
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )

        with pytest.raises(ValueError, match="is not an image"):
            await self.analyzer.analyze_image(non_image_file)

    @pytest.mark.asyncio
    async def test_analyze_image_without_encoded_data(self):
        """Test analysis failure for image without encoded data."""
        image_file = ProcessedFile(
            file_id="test-image-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={
                "metadata": {"width": 100, "height": 100}
            },  # No encoded_image
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )

        with pytest.raises(ValueError, match="does not have encoded data"):
            await self.analyzer.analyze_image(image_file)

    @pytest.mark.asyncio
    async def test_analyze_image_model_error(self, sample_processed_image):
        """Test analysis failure when model raises error."""
        mock_model = AsyncMock()
        mock_model.generate.side_effect = Exception("Model API error")
        self.mock_model_factory.get_model.return_value = mock_model

        with pytest.raises(RuntimeError, match="Vision analysis failed"):
            await self.analyzer.analyze_image(sample_processed_image)

    @pytest.mark.asyncio
    async def test_analyze_multiple_images(self):
        """Test analyzing multiple images concurrently."""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=["red", "green", "blue"][i])
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()

            processed_file = ProcessedFile(
                file_id=f"test-image-{i}",
                original_name=f"test{i}.jpg",
                file_type=FileType.IMAGE,
                mime_type="image/jpeg",
                file_size=1024,
                processed_content={
                    "encoded_image": encoded_image,
                    "metadata": {"width": 100, "height": 100, "format": "JPEG"},
                },
                metadata={},
                processing_timestamp=datetime.utcnow(),
                status=ProcessingStatus.COMPLETED,
            )
            images.append(processed_file)

        # Mock model responses
        mock_model = AsyncMock()
        responses = [
            ModelResponse(
                content=f"Analysis of image {i}",
                provider="openai",
                model_name="gpt-4-vision",
            )
            for i in range(3)
        ]
        mock_model.generate.side_effect = responses
        self.mock_model_factory.get_model.return_value = mock_model

        # Analyze multiple images
        results = await self.analyzer.analyze_multiple_images(images)

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Analysis of image {i}" in result["analysis"]
            assert result["model_used"] == "gpt-4-vision"

    @pytest.mark.asyncio
    async def test_analyze_multiple_images_with_non_images(self):
        """Test analyzing multiple files with non-images filtered out."""
        # Create mixed file types
        files = []

        # Add an image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        image_file = ProcessedFile(
            file_id="test-image-1",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={"encoded_image": encoded_image, "metadata": {}},
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )
        files.append(image_file)

        # Add a non-image
        text_file = ProcessedFile(
            file_id="test-doc-1",
            original_name="test.txt",
            file_type=FileType.DOCUMENT,
            mime_type="text/plain",
            file_size=100,
            processed_content={"text_content": "test"},
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )
        files.append(text_file)

        # Mock model response for the image only
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="Analysis of the image",
            provider="openai",
            model_name="gpt-4-vision",
        )
        mock_model.generate.return_value = mock_response
        self.mock_model_factory.get_model.return_value = mock_model

        # Analyze files
        results = await self.analyzer.analyze_multiple_images(files)

        # Should only analyze the image file
        assert len(results) == 1
        assert results[0]["analysis"] == "Analysis of the image"

    @pytest.mark.asyncio
    async def test_extract_text_from_image(self, sample_processed_image):
        """Test text extraction from image."""
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="Extracted text: Hello World\nLine 2: Some more text",
            provider="openai",
            model_name="gpt-4-vision",
        )
        mock_model.generate.return_value = mock_response
        self.mock_model_factory.get_model.return_value = mock_model

        result = await self.analyzer.extract_text_from_image(sample_processed_image)

        assert (
            result["extracted_text"]
            == "Extracted text: Hello World\nLine 2: Some more text"
        )
        assert result["text_length"] > 0
        assert result["extraction_method"] == "vision_model"
        assert result["model_used"] == "gpt-4-vision"
        assert result["confidence"] == "high"

        # Verify OCR-specific prompt was used
        call_args = mock_model.generate.call_args[0][0]
        message = call_args[0]
        assert "extract all text" in message.content[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_extract_text_from_image_error(self, sample_processed_image):
        """Test text extraction error handling."""
        mock_model = AsyncMock()
        mock_model.generate.side_effect = Exception("OCR failed")
        self.mock_model_factory.get_model.return_value = mock_model

        result = await self.analyzer.extract_text_from_image(sample_processed_image)

        assert result["extracted_text"] == ""
        assert result["text_length"] == 0
        assert result["confidence"] == "low"
        assert "error" in result
        assert "OCR failed" in result["error"]

    def test_get_analysis_types(self):
        """Test getting available analysis types."""
        types = self.analyzer.get_analysis_types()

        assert "general" in types
        assert "design" in types
        assert "diagram" in types
        assert "document" in types

    def test_add_analysis_type(self):
        """Test adding custom analysis type."""
        custom_prompt = "Analyze this image for accessibility issues"
        self.analyzer.add_analysis_type("accessibility", custom_prompt)

        types = self.analyzer.get_analysis_types()
        assert "accessibility" in types

        # Verify the prompt was stored
        assert self.analyzer._vision_prompts["accessibility"] == custom_prompt


@pytest.mark.asyncio
async def test_vision_analyzer_integration():
    """Integration test for vision analyzer with real model factory."""
    from agent_hive.core.models import get_model_factory

    # Create real model factory (but we'll mock the model)
    model_factory = get_model_factory()
    analyzer = VisionAnalyzer(model_factory)

    # Create test image
    img = Image.new("RGB", (50, 50), color="yellow")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    processed_file = ProcessedFile(
        file_id="integration-test",
        original_name="integration.jpg",
        file_type=FileType.IMAGE,
        mime_type="image/jpeg",
        file_size=len(buffer.getvalue()),
        processed_content={
            "encoded_image": encoded_image,
            "metadata": {"width": 50, "height": 50, "format": "JPEG"},
        },
        metadata={},
        processing_timestamp=datetime.utcnow(),
        status=ProcessingStatus.COMPLETED,
    )

    # Mock the model to avoid actual API calls
    with patch.object(model_factory, "get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_response = ModelResponse(
            content="Integration test analysis result",
            provider="test",
            model_name="test-vision",
        )
        mock_model.generate.return_value = mock_response
        mock_get_model.return_value = mock_model

        # Perform analysis
        result = await analyzer.analyze_image(processed_file, model_name="test-vision")

        # Verify integration
        assert result["analysis"] == "Integration test analysis result"
        assert result["model_used"] == "test-vision"
        mock_get_model.assert_called_once_with("test-vision")
