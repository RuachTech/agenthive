"""Vision analysis integration for multimodal content."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .models import ModelFactory, get_model_factory
from .multimodal import ProcessedFile, FileType

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analyzes visual content using vision-capable models."""

    def __init__(self, model_factory: Optional[ModelFactory] = None):
        self.model_factory = model_factory or get_model_factory()
        self._vision_prompts = {
            "general": "Analyze this image and describe what you see in detail. Include objects, people, text, colors, composition, and any other relevant details.",
            "design": "Analyze this design document or interface. Describe the layout, UI elements, typography, color scheme, and overall design principles used.",
            "diagram": "Analyze this diagram or technical drawing. Describe the components, relationships, flow, and technical details shown.",
            "document": "Analyze this document image. Extract and describe any text, tables, charts, or other structured information visible.",
        }

    async def analyze_image(
        self,
        processed_file: ProcessedFile,
        analysis_type: str = "general",
        custom_prompt: Optional[str] = None,
        model_name: str = "gpt-4-vision",
    ) -> Dict[str, Any]:
        """
        Analyze an image using a vision-capable model.

        Args:
            processed_file: ProcessedFile object containing image data
            analysis_type: Type of analysis ("general", "design", "diagram", "document")
            custom_prompt: Optional custom analysis prompt
            model_name: Name of the vision model to use

        Returns:
            Dictionary containing analysis results

        Raises:
            ValueError: If file is not an image or doesn't have encoded data
            RuntimeError: If vision analysis fails
        """
        if processed_file.file_type != FileType.IMAGE:
            raise ValueError(f"File {processed_file.original_name} is not an image")

        if "encoded_image" not in processed_file.processed_content:
            raise ValueError(
                f"Image {processed_file.original_name} does not have encoded data"
            )

        try:
            # Get vision-capable model
            model = await self.model_factory.get_model(model_name)

            # Prepare prompt
            prompt = custom_prompt or self._vision_prompts.get(
                analysis_type, self._vision_prompts["general"]
            )

            # Create message with image
            encoded_image = processed_file.processed_content["encoded_image"]

            # For OpenAI vision models, we need to structure the message properly
            message_content: List[Dict[Any, Any]] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]

            message = HumanMessage(content=message_content)  # type: ignore[arg-type]

            # Generate analysis
            response = await model.generate([message])

            # Extract analysis results
            analysis_result = {
                "analysis": response.content,
                "model_used": response.model_name,
                "provider": response.provider,
                "analysis_type": analysis_type,
                "prompt_used": prompt,
                "image_metadata": processed_file.processed_content.get("metadata", {}),
                "confidence": "high",  # Could be extracted from model response if available
                "timestamp": processed_file.processing_timestamp.isoformat(),
            }

            logger.info(
                "Vision analysis completed for %s using %s",
                processed_file.original_name,
                response.model_name,
            )

            return analysis_result

        except Exception as e:
            logger.error(
                "Vision analysis failed for %s: %s", processed_file.original_name, e
            )
            raise RuntimeError(f"Vision analysis failed: {e}")

    async def analyze_multiple_images(
        self,
        processed_files: List[ProcessedFile],
        analysis_type: str = "general",
        model_name: str = "gpt-4-vision",
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images concurrently.

        Args:
            processed_files: List of ProcessedFile objects containing image data
            analysis_type: Type of analysis to perform
            model_name: Name of the vision model to use

        Returns:
            List of analysis results
        """

        # Filter for image files only
        image_files = [f for f in processed_files if f.file_type == FileType.IMAGE]

        if not image_files:
            logger.warning("No image files found for batch analysis")
            return []

        # Create analysis tasks
        tasks = []
        for processed_file in image_files:
            task = asyncio.create_task(
                self.analyze_image(processed_file, analysis_type, model_name=model_name)
            )
            tasks.append(task)

        # Execute analyses concurrently
        results = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                logger.error("Failed to analyze image in batch: %s", e)
                # Add error result
                results.append(
                    {
                        "analysis": f"Analysis failed: {e}",
                        "error": True,
                        "analysis_type": analysis_type,
                    }
                )

        return results

    def get_analysis_types(self) -> List[str]:
        """Get available analysis types."""
        return list(self._vision_prompts.keys())

    def add_analysis_type(self, name: str, prompt: str) -> None:
        """Add a custom analysis type."""
        self._vision_prompts[name] = prompt

    async def extract_text_from_image(
        self,
        processed_file: ProcessedFile,
        model_name: str = "gpt-4-vision",
    ) -> Dict[str, Any]:
        """
        Extract text content from an image using OCR-like analysis.

        Args:
            processed_file: ProcessedFile object containing image data
            model_name: Name of the vision model to use

        Returns:
            Dictionary containing extracted text and metadata
        """
        ocr_prompt = """
        Extract all text visible in this image. Provide the text exactly as it appears, 
        maintaining formatting and structure where possible. If there are multiple text 
        sections, separate them clearly. Also identify any tables, lists, or structured 
        data and preserve their format.
        """

        try:
            result = await self.analyze_image(
                processed_file,
                analysis_type="document",
                custom_prompt=ocr_prompt,
                model_name=model_name,
            )

            # Process the result to extract structured text
            extracted_text = result["analysis"]

            return {
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "extraction_method": "vision_model",
                "model_used": result["model_used"],
                "confidence": result.get("confidence", "medium"),
                "timestamp": result["timestamp"],
            }

        except Exception as e:
            logger.error(
                "Text extraction failed for %s: %s", processed_file.original_name, e
            )
            return {
                "extracted_text": "",
                "text_length": 0,
                "extraction_method": "vision_model",
                "error": str(e),
                "confidence": "low",
            }


# Global vision analyzer instance
vision_analyzer = VisionAnalyzer()


def get_vision_analyzer() -> VisionAnalyzer:
    """Get the global vision analyzer instance."""
    return vision_analyzer
