"""Integration utilities for multimodal content with agent state."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage

from .multimodal import ProcessedFile, FileType, get_multimodal_processor
from .state import AgentState
from .vision import get_vision_analyzer

logger = logging.getLogger(__name__)


class MultimodalStateManager:
    """Manages multimodal content integration with agent state."""

    def __init__(self) -> None:
        self.processor = get_multimodal_processor()
        self.vision_analyzer = get_vision_analyzer()

    def add_files_to_state(
        self,
        state: AgentState,
        processed_files: List[ProcessedFile],
        auto_analyze_images: bool = True,
    ) -> AgentState:
        """
        Add processed files to agent state.

        Args:
            state: Current agent state
            processed_files: List of processed files to add
            auto_analyze_images: Whether to automatically analyze images

        Returns:
            Updated agent state
        """
        # Initialize multimodal_content if not present
        if "files" not in state["multimodal_content"]:
            state["multimodal_content"]["files"] = {}

        if "analyses" not in state["multimodal_content"]:
            state["multimodal_content"]["analyses"] = {}

        # Add files to state
        for processed_file in processed_files:
            file_data = {
                "file_id": processed_file.file_id,
                "original_name": processed_file.original_name,
                "file_type": processed_file.file_type.value,
                "mime_type": processed_file.mime_type,
                "file_size": processed_file.file_size,
                "status": processed_file.status.value,
                "processing_timestamp": processed_file.processing_timestamp.isoformat(),
                "processed_content": processed_file.processed_content,
                "metadata": processed_file.metadata,
            }

            if processed_file.error_message:
                file_data["error_message"] = processed_file.error_message

            state["multimodal_content"]["files"][processed_file.file_id] = file_data

        # Update state metadata
        state["multimodal_content"]["total_files"] = len(
            state["multimodal_content"]["files"]
        )
        state["multimodal_content"]["file_types"] = list(
            set(f["file_type"] for f in state["multimodal_content"]["files"].values())
        )

        logger.info("Added %d files to agent state", len(processed_files))
        return state

    async def analyze_images_in_state(
        self,
        state: AgentState,
        analysis_type: str = "general",
        model_name: str = "gpt-4-vision",
    ) -> AgentState:
        """
        Analyze all images in the agent state.

        Args:
            state: Current agent state
            analysis_type: Type of analysis to perform
            model_name: Vision model to use

        Returns:
            Updated agent state with analysis results
        """
        if "files" not in state["multimodal_content"]:
            return state

        # Find image files that haven't been analyzed
        image_files = []
        for file_id, file_data in state["multimodal_content"]["files"].items():
            if file_data["file_type"] == FileType.IMAGE.value and file_id not in state[
                "multimodal_content"
            ].get("analyses", {}):

                # Reconstruct ProcessedFile object
                processed_file = ProcessedFile(
                    file_id=file_data["file_id"],
                    original_name=file_data["original_name"],
                    file_type=FileType(file_data["file_type"]),
                    mime_type=file_data["mime_type"],
                    file_size=file_data["file_size"],
                    processed_content=file_data["processed_content"],
                    metadata=file_data["metadata"],
                    processing_timestamp=file_data["processing_timestamp"],
                )
                image_files.append(processed_file)

        if not image_files:
            logger.info("No new images to analyze in state")
            return state

        try:
            # Analyze images
            analyses = await self.vision_analyzer.analyze_multiple_images(
                image_files, analysis_type, model_name
            )

            # Add analyses to state
            for i, analysis in enumerate(analyses):
                if i < len(image_files):
                    file_id = image_files[i].file_id
                    state["multimodal_content"]["analyses"][file_id] = analysis

            logger.info("Analyzed %d images in state", len(analyses))

        except Exception as e:
            logger.error("Failed to analyze images in state: %s", e)
            # Add error information to state
            state["multimodal_content"]["analysis_error"] = str(e)

        return state

    def create_multimodal_context_message(
        self, state: AgentState
    ) -> Optional[SystemMessage]:
        """
        Create a context message describing multimodal content for agents.

        Args:
            state: Current agent state

        Returns:
            SystemMessage with multimodal context or None if no content
        """
        if not state.get("multimodal_content") or not state["multimodal_content"].get(
            "files"
        ):
            return None

        context_parts = [
            "The following multimodal content is available in this conversation:"
        ]

        # Describe files
        for file_id, file_data in state["multimodal_content"]["files"].items():
            file_type = file_data["file_type"]
            filename = file_data["original_name"]

            if file_type == "image":
                # Include image analysis if available
                analysis = state["multimodal_content"].get("analyses", {}).get(file_id)
                if analysis and not analysis.get("error"):
                    context_parts.append(
                        f"- Image: {filename}\n  Analysis: {analysis['analysis'][:200]}..."
                    )
                else:
                    context_parts.append(f"- Image: {filename} (analysis pending)")

            elif file_type == "pdf":
                content = file_data["processed_content"]
                if content.get("extraction_available"):
                    text_preview = content.get("text_content", "")[:200]
                    context_parts.append(
                        f"- PDF: {filename}\n  Content preview: {text_preview}..."
                    )
                else:
                    context_parts.append(
                        f"- PDF: {filename} (text extraction not available)"
                    )

            elif file_type == "document":
                content = file_data["processed_content"]
                if content.get("extraction_available"):
                    text_preview = content.get("text_content", "")[:200]
                    context_parts.append(
                        f"- Document: {filename}\n  Content preview: {text_preview}..."
                    )
                else:
                    context_parts.append(f"- Document: {filename}")

        context_message = "\n".join(context_parts)
        context_message += (
            "\n\nYou can reference this content in your responses and analysis."
        )

        return SystemMessage(content=context_message)

    def get_file_content(
        self, state: AgentState, file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get processed content for a specific file.

        Args:
            state: Current agent state
            file_id: ID of the file to retrieve

        Returns:
            File content dictionary or None if not found
        """
        files = state.get("multimodal_content", {}).get("files", {})
        file_content = files.get(file_id)
        return file_content if isinstance(file_content, dict) else None

    def get_files_by_type(
        self, state: AgentState, file_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get all files of a specific type from state.

        Args:
            state: Current agent state
            file_type: Type of files to retrieve ("image", "pdf", "document")

        Returns:
            List of file data dictionaries
        """
        files = state.get("multimodal_content", {}).get("files", {})
        return [
            file_data
            for file_data in files.values()
            if isinstance(file_data, dict) and file_data.get("file_type") == file_type
        ]

    def get_text_content_summary(self, state: AgentState) -> str:
        """
        Get a summary of all text content from processed files.

        Args:
            state: Current agent state

        Returns:
            Combined text content summary
        """
        text_parts = []
        files = state.get("multimodal_content", {}).get("files", {})

        for file_data in files.values():
            if not isinstance(file_data, dict):
                continue
            filename = file_data.get("original_name", "unknown")
            content = file_data.get("processed_content", {})

            if file_data.get("file_type") == "pdf" and content.get("text_content"):
                text_parts.append(f"=== {filename} ===\n{content['text_content']}")

            elif file_data.get("file_type") == "document" and content.get(
                "text_content"
            ):
                text_parts.append(f"=== {filename} ===\n{content['text_content']}")

        return "\n\n".join(text_parts)

    def clear_multimodal_content(self, state: AgentState) -> AgentState:
        """
        Clear all multimodal content from state.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        state["multimodal_content"] = {}
        logger.info("Cleared multimodal content from state")
        return state

    def get_content_statistics(self, state: AgentState) -> Dict[str, Any]:
        """
        Get statistics about multimodal content in state.

        Args:
            state: Current agent state

        Returns:
            Dictionary with content statistics
        """
        files = state.get("multimodal_content", {}).get("files", {})
        analyses = state.get("multimodal_content", {}).get("analyses", {})

        stats: Dict[str, Any] = {
            "total_files": len(files),
            "file_types": {},
            "total_size": 0,
            "analyzed_images": len(analyses),
            "text_documents": 0,
            "total_text_length": 0,
        }

        for file_data in files.values():
            if not isinstance(file_data, dict):
                continue
            file_type = file_data.get("file_type", "unknown")
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["total_size"] += file_data.get("file_size", 0)

            if file_type in ["pdf", "document"]:
                content = file_data.get("processed_content", {})
                if isinstance(content, dict) and content.get("text_content"):
                    stats["text_documents"] += 1
                    text_content = content.get("text_content", "")
                    if isinstance(text_content, str):
                        stats["total_text_length"] += len(text_content)

        return stats


# Global multimodal state manager instance
multimodal_state_manager = MultimodalStateManager()


def get_multimodal_state_manager() -> MultimodalStateManager:
    """Get the global multimodal state manager instance."""
    return multimodal_state_manager
