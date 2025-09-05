"""Multimodal content processing pipeline for AgentHive."""

import base64
import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types for multimodal processing."""

    IMAGE = "image"
    PDF = "pdf"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Processing status for uploaded files."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessedFile:
    """Processed file with extracted content and metadata."""

    file_id: str
    original_name: str
    file_type: FileType
    mime_type: str
    file_size: int
    processed_content: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_timestamp: datetime
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    file_hash: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate file hash if not provided."""
        if self.file_hash is None:
            # Generate hash from file_id and original_name for uniqueness
            content = f"{self.file_id}:{self.original_name}:{self.file_size}"
            self.file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


class FileValidationError(Exception):
    """Raised when file validation fails."""

    pass


class FileProcessingError(Exception):
    """Raised when file processing fails."""

    pass


class FileValidator:
    """Validates uploaded files against supported formats and constraints."""

    # Maximum file size in bytes (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Supported MIME types
    SUPPORTED_MIME_TYPES = {
        # Images
        "image/jpeg": FileType.IMAGE,
        "image/jpg": FileType.IMAGE,
        "image/png": FileType.IMAGE,
        "image/gif": FileType.IMAGE,
        "image/webp": FileType.IMAGE,
        "image/bmp": FileType.IMAGE,
        "image/tiff": FileType.IMAGE,
        # PDFs
        "application/pdf": FileType.PDF,
        # Documents
        "text/plain": FileType.DOCUMENT,
        "text/markdown": FileType.DOCUMENT,
        "application/msword": FileType.DOCUMENT,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCUMENT,
        "application/rtf": FileType.DOCUMENT,
    }

    @classmethod
    def validate_file(
        cls, filename: str, content: bytes, mime_type: Optional[str] = None
    ) -> FileType:
        """
        Validate uploaded file and determine its type.

        Args:
            filename: Original filename
            content: File content as bytes
            mime_type: Optional MIME type hint

        Returns:
            FileType enum value

        Raises:
            FileValidationError: If file is invalid or unsupported
        """
        # Check file size
        if len(content) > cls.MAX_FILE_SIZE:
            raise FileValidationError(
                f"File size {len(content)} bytes exceeds maximum allowed size {cls.MAX_FILE_SIZE} bytes"
            )

        # Determine MIME type if not provided
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(filename)

        if mime_type is None:
            raise FileValidationError(
                f"Could not determine MIME type for file: {filename}"
            )

        # Check if MIME type is supported
        if mime_type not in cls.SUPPORTED_MIME_TYPES:
            supported_types = ", ".join(cls.SUPPORTED_MIME_TYPES.keys())
            raise FileValidationError(
                f"Unsupported file type: {mime_type}. Supported types: {supported_types}"
            )

        file_type = cls.SUPPORTED_MIME_TYPES[mime_type]

        # Additional validation based on file type
        if file_type == FileType.IMAGE:
            cls._validate_image(content)
        elif file_type == FileType.PDF:
            cls._validate_pdf(content)

        return file_type

    @classmethod
    def _validate_image(cls, content: bytes) -> None:
        """Validate image file content."""
        try:
            with Image.open(BytesIO(content)) as img:
                # Verify image can be opened and read
                img.verify()
        except Exception as e:
            raise FileValidationError(f"Invalid image file: {e}")

    @classmethod
    def _validate_pdf(cls, content: bytes) -> None:
        """Validate PDF file content."""
        # Check PDF magic number
        if not content.startswith(b"%PDF-"):
            raise FileValidationError("Invalid PDF file: missing PDF header")


class ImageProcessor:
    """Processes image files for vision model analysis."""

    @staticmethod
    async def process_image(content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process image file and extract metadata and analysis.

        Args:
            content: Image file content as bytes
            filename: Original filename

        Returns:
            Dictionary containing processed image data
        """
        try:
            # Open and analyze image
            with Image.open(BytesIO(content)) as img:
                # Extract basic metadata
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.size[0],
                    "height": img.size[1],
                    "has_transparency": img.mode in ("RGBA", "LA")
                    or "transparency" in img.info,
                }

                # Convert to RGB if necessary for analysis
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Encode image for vision model analysis
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                encoded_image = base64.b64encode(buffer.getvalue()).decode()

                return {
                    "encoded_image": encoded_image,
                    "metadata": metadata,
                    "analysis_ready": True,
                    "description": f"Image file: {filename} ({metadata['width']}x{metadata['height']}, {metadata['format']})",
                }

        except Exception as e:
            logger.error("Image processing failed: %s", e)
            raise FileProcessingError(f"Failed to process image: {e}")


class PDFProcessor:
    """Processes PDF files for text extraction."""

    @staticmethod
    async def process_pdf(content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process PDF file and extract text content.

        Args:
            content: PDF file content as bytes
            filename: Original filename

        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Try to import PyPDF2 for PDF processing
            try:
                import PyPDF2
            except ImportError:
                logger.warning("PyPDF2 not available, using basic PDF processing")
                return {
                    "text_content": "",
                    "page_count": 0,
                    "metadata": {"processing_method": "basic"},
                    "description": f"PDF file: {filename} (text extraction not available)",
                    "extraction_available": False,
                }

            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))

            text_content = ""
            page_count = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(
                        "Failed to extract text from page %d: %s", page_num + 1, e
                    )

            # Extract metadata
            metadata = {
                "page_count": page_count,
                "processing_method": "PyPDF2",
                "text_length": len(text_content),
            }

            # Add PDF info if available
            if pdf_reader.metadata:
                metadata.update(
                    {
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                        "creator": pdf_reader.metadata.get("/Creator", ""),
                    }
                )

            return {
                "text_content": text_content.strip(),
                "metadata": metadata,
                "description": f"PDF file: {filename} ({page_count} pages, {len(text_content)} characters)",
                "extraction_available": True,
            }

        except Exception as e:
            logger.error("PDF processing failed: %s", e)
            raise FileProcessingError(f"Failed to process PDF: {e}")


class DocumentProcessor:
    """Processes document files for text extraction."""

    @staticmethod
    async def process_document(
        content: bytes, filename: str, mime_type: str
    ) -> Dict[str, Any]:
        """
        Process document file and extract text content.

        Args:
            content: Document file content as bytes
            filename: Original filename
            mime_type: MIME type of the document

        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            if mime_type == "text/plain" or mime_type == "text/markdown":
                # Handle plain text files
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    # Try other encodings
                    for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                        try:
                            text_content = content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise FileProcessingError(
                            "Could not decode text file with any supported encoding"
                        )

                metadata = {
                    "encoding": "utf-8",
                    "line_count": text_content.count("\n") + 1,
                    "character_count": len(text_content),
                    "word_count": len(text_content.split()),
                }

                return {
                    "text_content": text_content,
                    "metadata": metadata,
                    "description": f"Text file: {filename} ({metadata['word_count']} words, {metadata['line_count']} lines)",
                    "extraction_available": True,
                }

            else:
                # For other document types, we'd need additional libraries
                # For now, return basic info
                return {
                    "text_content": "",
                    "metadata": {"processing_method": "basic", "mime_type": mime_type},
                    "description": f"Document file: {filename} (advanced text extraction not available)",
                    "extraction_available": False,
                }

        except Exception as e:
            logger.error("Document processing failed: %s", e)
            raise FileProcessingError(f"Failed to process document: {e}")


class MultimodalProcessor:
    """Main processor for handling multimodal content."""

    def __init__(self) -> None:
        self.validator = FileValidator()
        self.image_processor = ImageProcessor()
        self.pdf_processor = PDFProcessor()
        self.document_processor = DocumentProcessor()

    async def process_file(
        self,
        filename: str,
        content: bytes,
        mime_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ProcessedFile:
        """
        Process uploaded file and return ProcessedFile object.

        Args:
            filename: Original filename
            content: File content as bytes
            mime_type: Optional MIME type hint
            user_id: Optional user identifier

        Returns:
            ProcessedFile object with processed content

        Raises:
            FileValidationError: If file validation fails
            FileProcessingError: If file processing fails
        """
        file_id = str(uuid4())
        processing_timestamp = datetime.utcnow()

        try:
            # Validate file
            file_type = self.validator.validate_file(filename, content, mime_type)

            # Determine actual MIME type
            if mime_type is None:
                mime_type, _ = mimetypes.guess_type(filename)

            # Create initial ProcessedFile object
            processed_file = ProcessedFile(
                file_id=file_id,
                original_name=filename,
                file_type=file_type,
                mime_type=mime_type or "application/octet-stream",
                file_size=len(content),
                processed_content={},
                metadata={"user_id": user_id} if user_id else {},
                processing_timestamp=processing_timestamp,
                status=ProcessingStatus.PROCESSING,
            )

            # Process based on file type
            if file_type == FileType.IMAGE:
                processed_content = await self.image_processor.process_image(
                    content, filename
                )
            elif file_type == FileType.PDF:
                processed_content = await self.pdf_processor.process_pdf(
                    content, filename
                )
            elif file_type == FileType.DOCUMENT:
                processed_content = await self.document_processor.process_document(
                    content, filename, processed_file.mime_type
                )
            else:
                raise FileProcessingError(f"Unsupported file type: {file_type}")

            # Update processed file with results
            processed_file.processed_content = processed_content
            processed_file.status = ProcessingStatus.COMPLETED

            logger.info(
                "Successfully processed file: %s (type: %s)", filename, file_type.value
            )
            return processed_file

        except (FileValidationError, FileProcessingError) as e:
            # Create failed ProcessedFile object
            processed_file = ProcessedFile(
                file_id=file_id,
                original_name=filename,
                file_type=FileType.UNKNOWN,
                mime_type=mime_type or "application/octet-stream",
                file_size=len(content),
                processed_content={},
                metadata={"user_id": user_id} if user_id else {},
                processing_timestamp=processing_timestamp,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
            )

            logger.error("File processing failed for %s: %s", filename, e)
            raise

    async def process_multiple_files(
        self,
        files: List[tuple[str, bytes, Optional[str]]],
        user_id: Optional[str] = None,
    ) -> List[ProcessedFile]:
        """
        Process multiple files concurrently.

        Args:
            files: List of (filename, content, mime_type) tuples
            user_id: Optional user identifier

        Returns:
            List of ProcessedFile objects
        """
        import asyncio

        tasks = []
        for filename, content, mime_type in files:
            task = asyncio.create_task(
                self.process_file(filename, content, mime_type, user_id)
            )
            tasks.append(task)

        results = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except (FileValidationError, FileProcessingError) as e:
                # Create failed result for this file
                logger.error("Failed to process file in batch: %s", e)
                # We could add the failed file to results if needed
                continue

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported file formats organized by category.

        Returns:
            Dictionary mapping categories to supported MIME types
        """
        formats: Dict[str, List[str]] = {}
        for mime_type, file_type in self.validator.SUPPORTED_MIME_TYPES.items():
            category = file_type.value
            if category not in formats:
                formats[category] = []
            formats[category].append(mime_type)

        return formats


# Global multimodal processor instance
multimodal_processor = MultimodalProcessor()


def get_multimodal_processor() -> MultimodalProcessor:
    """Get the global multimodal processor instance."""
    return multimodal_processor
