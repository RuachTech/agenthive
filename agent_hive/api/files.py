"""File upload and multimodal content handling."""

from typing import Dict, Any, List, TYPE_CHECKING
import structlog

from fastapi import UploadFile

from agent_hive.core.multimodal import FileValidationError, FileProcessingError
from agent_hive.api.validation import (
    RequestValidator,
    ResponseFormatter,
    APIError,
    ValidationError,
    ServiceUnavailableError,
    create_http_exception,
)

if TYPE_CHECKING:
    from agent_hive.api.core import AgentHiveAPI

logger = structlog.get_logger()


class FileService:
    """Service class for file upload and processing operations."""

    def __init__(self, api_instance: "AgentHiveAPI") -> None:
        self.api = api_instance

    async def upload_multimodal_content(
        self, files: List[UploadFile], user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Upload and process multimodal content.

        Args:
            files: List of uploaded files
            user_id: User identifier

        Returns:
            Processing results for all files
        """
        try:
            if not files:
                raise ValidationError("No files provided", field="files")

            # Validate user ID
            user_id = RequestValidator.validate_user_id(user_id)

            results = []
            errors = []

            for file in files:
                try:
                    content = await file.read()

                    # Validate file
                    RequestValidator.validate_file_upload(
                        filename=file.filename or "unknown",
                        content_type=file.content_type,
                        file_size=len(content),
                    )

                    processed_file = await self.api.multimodal_processor.process_file(
                        filename=file.filename or "unknown",
                        content=content,
                        mime_type=file.content_type,
                        user_id=user_id,
                    )

                    results.append(
                        {
                            "file_id": processed_file.file_id,
                            "original_name": processed_file.original_name,
                            "file_type": processed_file.file_type.value,
                            "status": processed_file.status.value,
                            "file_size": processed_file.file_size,
                            "processing_timestamp": processed_file.processing_timestamp.isoformat(),
                            "metadata": processed_file.metadata,
                        }
                    )

                except (FileValidationError, FileProcessingError, ValidationError) as e:
                    errors.append(
                        {
                            "filename": file.filename,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )

            upload_result = {
                "processed_files": results,
                "errors": errors,
                "total_files": len(files),
                "successful": len(results),
                "failed": len(errors),
            }

            return ResponseFormatter.success_response(
                data=upload_result,
                message=f"Processed {len(results)}/{len(files)} files successfully",
            )

        except APIError as e:
            logger.error("File upload validation failed", error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error("File upload failed", error=str(e))
            error = ServiceUnavailableError("file_upload", str(e))
            raise create_http_exception(error)

    def get_supported_formats(self) -> Dict[str, Any]:
        """
        Get supported file formats.

        Returns:
            Dictionary of supported formats by category
        """
        try:
            formats = self.api.multimodal_processor.get_supported_formats()

            return ResponseFormatter.success_response(
                data={
                    "supported_formats": formats,
                    "max_file_size_bytes": 10 * 1024 * 1024,  # 10MB
                    "max_file_size_human": "10 MB",
                },
                message="Supported file formats retrieved successfully",
            )

        except Exception as e:
            logger.error("Failed to get supported formats", error=str(e))
            error = ServiceUnavailableError("file_formats", str(e))
            raise create_http_exception(error)
