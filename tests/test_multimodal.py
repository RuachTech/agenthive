"""Tests for multimodal content processing pipeline."""

import base64
import sys
import pytest
from datetime import datetime
from io import BytesIO
from unittest.mock import Mock, patch
from PIL import Image

from agent_hive.core.multimodal import (
    FileType,
    ProcessingStatus,
    ProcessedFile,
    FileValidator,
    FileValidationError,
    FileProcessingError,
    ImageProcessor,
    PDFProcessor,
    DocumentProcessor,
    MultimodalProcessor,
)


class TestProcessedFile:
    """Test ProcessedFile dataclass."""

    def test_processed_file_creation(self):
        """Test ProcessedFile creation with required fields."""
        processed_file = ProcessedFile(
            file_id="test-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={"test": "data"},
            metadata={"user": "test"},
            processing_timestamp=datetime.utcnow(),
        )

        assert processed_file.file_id == "test-123"
        assert processed_file.original_name == "test.jpg"
        assert processed_file.file_type == FileType.IMAGE
        assert processed_file.status == ProcessingStatus.PENDING
        assert processed_file.file_hash is not None
        assert len(processed_file.file_hash) == 16

    def test_processed_file_hash_generation(self):
        """Test automatic hash generation."""
        processed_file = ProcessedFile(
            file_id="test-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={},
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )

        # Hash should be generated automatically
        assert processed_file.file_hash is not None

        # Same inputs should generate same hash
        processed_file2 = ProcessedFile(
            file_id="test-123",
            original_name="test.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            file_size=1024,
            processed_content={},
            metadata={},
            processing_timestamp=datetime.utcnow(),
        )

        assert processed_file.file_hash == processed_file2.file_hash


class TestFileValidator:
    """Test FileValidator class."""

    def test_validate_supported_image(self):
        """Test validation of supported image file."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        content = buffer.getvalue()

        file_type = FileValidator.validate_file("test.jpg", content, "image/jpeg")
        assert file_type == FileType.IMAGE

    def test_validate_file_too_large(self):
        """Test validation failure for oversized file."""
        # Create content larger than max size
        large_content = b"x" * (FileValidator.MAX_FILE_SIZE + 1)

        with pytest.raises(FileValidationError, match="exceeds maximum allowed size"):
            FileValidator.validate_file("large.txt", large_content, "text/plain")

    def test_validate_unsupported_mime_type(self):
        """Test validation failure for unsupported MIME type."""
        content = b"test content"

        with pytest.raises(FileValidationError, match="Unsupported file type"):
            FileValidator.validate_file("test.xyz", content, "application/xyz")

    def test_validate_pdf_file(self):
        """Test validation of PDF file."""
        # Create minimal PDF content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"

        file_type = FileValidator.validate_file(
            "test.pdf", pdf_content, "application/pdf"
        )
        assert file_type == FileType.PDF

    def test_validate_invalid_pdf(self):
        """Test validation failure for invalid PDF."""
        invalid_pdf = b"not a pdf file"

        with pytest.raises(FileValidationError, match="Invalid PDF file"):
            FileValidator.validate_file("test.pdf", invalid_pdf, "application/pdf")

    def test_validate_text_document(self):
        """Test validation of text document."""
        content = b"This is a test document"

        file_type = FileValidator.validate_file("test.txt", content, "text/plain")
        assert file_type == FileType.DOCUMENT

    def test_validate_invalid_image(self):
        """Test validation failure for invalid image."""
        invalid_image = b"not an image"

        with pytest.raises(FileValidationError, match="Invalid image file"):
            FileValidator.validate_file("test.jpg", invalid_image, "image/jpeg")


class TestImageProcessor:
    """Test ImageProcessor class."""

    @pytest.mark.asyncio
    async def test_process_valid_image(self):
        """Test processing of valid image."""
        # Create test image
        img = Image.new("RGB", (200, 150), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        content = buffer.getvalue()

        result = await ImageProcessor.process_image(content, "test.jpg")

        assert "encoded_image" in result
        assert "metadata" in result
        assert result["metadata"]["width"] == 200
        assert result["metadata"]["height"] == 150
        assert result["metadata"]["format"] == "JPEG"
        assert result["analysis_ready"] is True
        assert "test.jpg" in result["description"]

    @pytest.mark.asyncio
    async def test_process_rgba_image(self):
        """Test processing of RGBA image."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        content = buffer.getvalue()

        result = await ImageProcessor.process_image(content, "test.png")

        assert result["metadata"]["has_transparency"] is True
        assert result["metadata"]["mode"] == "RGBA"

    @pytest.mark.asyncio
    async def test_process_invalid_image(self):
        """Test processing failure for invalid image."""
        invalid_content = b"not an image"

        with pytest.raises(FileProcessingError, match="Failed to process image"):
            await ImageProcessor.process_image(invalid_content, "test.jpg")


class TestPDFProcessor:
    """Test PDFProcessor class."""

    @pytest.mark.asyncio
    async def test_process_pdf_without_pypdf2(self):
        """Test PDF processing when PyPDF2 is not available."""
        pdf_content = b"%PDF-1.4\ntest content"

        # Temporarily remove PyPDF2 from sys.modules if it exists
        original_pypdf2 = sys.modules.pop("PyPDF2", None)

        def mock_import(name, *args, **kwargs):
            if name == "PyPDF2":
                raise ImportError("No module named 'PyPDF2'")
            return __import__(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                result = await PDFProcessor.process_pdf(pdf_content, "test.pdf")
        finally:
            # Restore PyPDF2 if it was there
            if original_pypdf2:
                sys.modules["PyPDF2"] = original_pypdf2

        assert result["extraction_available"] is False
        assert result["page_count"] == 0
        assert result["metadata"]["processing_method"] == "basic"

    @pytest.mark.asyncio
    async def test_process_pdf_with_pypdf2(self):
        """Test PDF processing with PyPDF2 available."""
        pdf_content = b"%PDF-1.4\ntest content"

        # Mock PyPDF2
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test page content"

        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {"/Title": "Test PDF", "/Author": "Test Author"}

        mock_pypdf2 = Mock()
        mock_pypdf2.PdfReader.return_value = mock_reader

        with patch.dict("sys.modules", {"PyPDF2": mock_pypdf2}):
            result = await PDFProcessor.process_pdf(pdf_content, "test.pdf")

        assert result["extraction_available"] is True
        assert result["metadata"]["page_count"] == 1
        assert "Test page content" in result["text_content"]
        assert result["metadata"]["title"] == "Test PDF"
        assert result["metadata"]["author"] == "Test Author"

    @pytest.mark.asyncio
    async def test_process_pdf_extraction_error(self):
        """Test PDF processing with extraction error."""
        pdf_content = b"invalid pdf"

        mock_pypdf2 = Mock()
        mock_pypdf2.PdfReader.side_effect = Exception("PDF read error")

        with patch.dict("sys.modules", {"PyPDF2": mock_pypdf2}):
            with pytest.raises(FileProcessingError, match="Failed to process PDF"):
                await PDFProcessor.process_pdf(pdf_content, "test.pdf")


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    @pytest.mark.asyncio
    async def test_process_text_document(self):
        """Test processing of plain text document."""
        content = b"This is a test document\nwith multiple lines\nand some words."

        result = await DocumentProcessor.process_document(
            content, "test.txt", "text/plain"
        )

        assert result["extraction_available"] is True
        assert (
            result["text_content"]
            == "This is a test document\nwith multiple lines\nand some words."
        )
        assert result["metadata"]["line_count"] == 3
        assert result["metadata"]["word_count"] == 11
        assert "test.txt" in result["description"]

    @pytest.mark.asyncio
    async def test_process_markdown_document(self):
        """Test processing of markdown document."""
        content = b"# Test Markdown\n\nThis is **bold** text."

        result = await DocumentProcessor.process_document(
            content, "test.md", "text/markdown"
        )

        assert result["extraction_available"] is True
        assert "# Test Markdown" in result["text_content"]
        assert "**bold**" in result["text_content"]

    @pytest.mark.asyncio
    async def test_process_unsupported_document(self):
        """Test processing of unsupported document type."""
        content = b"binary document content"

        result = await DocumentProcessor.process_document(
            content,
            "test.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        assert result["extraction_available"] is False
        assert result["text_content"] == ""
        assert "advanced text extraction not available" in result["description"]

    @pytest.mark.asyncio
    async def test_process_document_encoding_fallback(self):
        """Test processing document with encoding fallback."""
        # Create content that requires fallback encoding (latin-1 can handle any byte)
        content = b"\xe9\xe8\xe7"  # Some non-UTF8 bytes that latin-1 can handle

        result = await DocumentProcessor.process_document(
            content, "test.txt", "text/plain"
        )

        # Should succeed with fallback encoding
        assert result["extraction_available"] is True
        assert len(result["text_content"]) > 0


class TestMultimodalProcessor:
    """Test MultimodalProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MultimodalProcessor()

    @pytest.mark.asyncio
    async def test_process_image_file(self):
        """Test processing of image file."""
        # Create test image
        img = Image.new("RGB", (100, 100), color="green")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        content = buffer.getvalue()

        result = await self.processor.process_file(
            "test.jpg", content, "image/jpeg", "user123"
        )

        assert result.file_type == FileType.IMAGE
        assert result.status == ProcessingStatus.COMPLETED
        assert result.original_name == "test.jpg"
        assert result.metadata["user_id"] == "user123"
        assert "encoded_image" in result.processed_content

    @pytest.mark.asyncio
    async def test_process_text_file(self):
        """Test processing of text file."""
        content = b"This is a test document with some content."

        result = await self.processor.process_file("test.txt", content, "text/plain")

        assert result.file_type == FileType.DOCUMENT
        assert result.status == ProcessingStatus.COMPLETED
        assert result.processed_content["extraction_available"] is True
        assert "test document" in result.processed_content["text_content"]

    @pytest.mark.asyncio
    async def test_process_invalid_file(self):
        """Test processing of invalid file."""
        content = b"x" * (FileValidator.MAX_FILE_SIZE + 1)  # Too large

        with pytest.raises(FileValidationError):
            await self.processor.process_file("large.txt", content, "text/plain")

    @pytest.mark.asyncio
    async def test_process_multiple_files(self):
        """Test processing multiple files."""
        # Create test files
        img = Image.new("RGB", (50, 50), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_content = buffer.getvalue()

        text_content = b"Test document content"

        files = [
            ("image.jpg", image_content, "image/jpeg"),
            ("document.txt", text_content, "text/plain"),
        ]

        results = await self.processor.process_multiple_files(files, "user123")

        assert len(results) == 2

        # Find image and text results
        image_result = next(r for r in results if r.file_type == FileType.IMAGE)
        text_result = next(r for r in results if r.file_type == FileType.DOCUMENT)

        assert image_result.status == ProcessingStatus.COMPLETED
        assert text_result.status == ProcessingStatus.COMPLETED
        assert image_result.metadata["user_id"] == "user123"
        assert text_result.metadata["user_id"] == "user123"

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()

        assert "image" in formats
        assert "pdf" in formats
        assert "document" in formats

        assert "image/jpeg" in formats["image"]
        assert "application/pdf" in formats["pdf"]
        assert "text/plain" in formats["document"]


@pytest.fixture
def sample_image_content():
    """Create sample image content for testing."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_text_content():
    """Create sample text content for testing."""
    return b"This is a sample text document for testing purposes."


@pytest.fixture
def sample_processed_file(sample_image_content):
    """Create sample ProcessedFile for testing."""
    return ProcessedFile(
        file_id="test-file-123",
        original_name="test.jpg",
        file_type=FileType.IMAGE,
        mime_type="image/jpeg",
        file_size=len(sample_image_content),
        processed_content={
            "encoded_image": base64.b64encode(sample_image_content).decode(),
            "metadata": {"width": 100, "height": 100, "format": "JPEG"},
        },
        metadata={"user_id": "test_user"},
        processing_timestamp=datetime.utcnow(),
        status=ProcessingStatus.COMPLETED,
    )


class TestIntegration:
    """Integration tests for multimodal processing."""

    @pytest.mark.asyncio
    async def test_end_to_end_image_processing(self, sample_image_content):
        """Test complete image processing pipeline."""
        processor = MultimodalProcessor()

        # Process the image
        result = await processor.process_file(
            "test_image.jpg", sample_image_content, "image/jpeg"
        )

        # Verify all processing steps completed
        assert result.status == ProcessingStatus.COMPLETED
        assert result.file_type == FileType.IMAGE
        assert "encoded_image" in result.processed_content
        assert "metadata" in result.processed_content
        assert result.processed_content["analysis_ready"] is True

        # Verify metadata
        metadata = result.processed_content["metadata"]
        assert metadata["width"] == 100
        assert metadata["height"] == 100
        assert metadata["format"] == "JPEG"

    @pytest.mark.asyncio
    async def test_end_to_end_text_processing(self, sample_text_content):
        """Test complete text processing pipeline."""
        processor = MultimodalProcessor()

        # Process the text file
        result = await processor.process_file(
            "test_doc.txt", sample_text_content, "text/plain"
        )

        # Verify all processing steps completed
        assert result.status == ProcessingStatus.COMPLETED
        assert result.file_type == FileType.DOCUMENT
        assert result.processed_content["extraction_available"] is True
        assert "sample text document" in result.processed_content["text_content"]

        # Verify metadata
        metadata = result.processed_content["metadata"]
        assert metadata["word_count"] > 0
        assert metadata["line_count"] >= 1
