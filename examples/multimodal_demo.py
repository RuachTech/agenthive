#!/usr/bin/env python3
"""
Demo script showing multimodal content processing capabilities.

This script demonstrates:
1. File upload and validation
2. Image processing and analysis
3. PDF text extraction
4. Document processing
5. Integration with agent state
6. Vision analysis of images
"""

import asyncio
from datetime import datetime
from io import BytesIO
from PIL import Image

from agent_hive.core.multimodal import (
    get_multimodal_processor,
    FileType,
)
from agent_hive.core.multimodal_integration import get_multimodal_state_manager
from agent_hive.core.state import AgentState

# from agent_hive.core.vision import get_vision_analyzer  # Not used in demo


def create_sample_files():
    """Create sample files for demonstration."""
    files = []

    # Create a sample image
    img = Image.new("RGB", (300, 200), color="lightblue")
    # Add some simple shapes
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 100], fill="red", outline="black")
    draw.ellipse([200, 75, 250, 125], fill="green", outline="black")
    draw.text((100, 150), "Sample Chart", fill="black")

    # Save to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_content = buffer.getvalue()

    files.append(("sample_chart.png", image_content, "image/png"))

    # Create a sample text document
    text_content = """# Project Report

## Executive Summary
This document outlines the key findings from our recent analysis.

## Key Metrics
- Revenue: $1.2M
- Growth: 15%
- Customer Satisfaction: 92%

## Recommendations
1. Expand marketing efforts
2. Improve customer support
3. Invest in new technology

## Conclusion
The project shows promising results and should be continued.
""".encode(
        "utf-8"
    )

    files.append(("report.md", text_content, "text/markdown"))

    # Create a simple PDF-like content (just text for demo)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj

xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
173
%%EOF"""

    files.append(("sample.pdf", pdf_content, "application/pdf"))

    return files


async def demo_basic_processing():
    """Demonstrate basic file processing."""
    print("🔄 Demo: Basic File Processing")
    print("=" * 50)

    processor = get_multimodal_processor()

    # Get supported formats
    formats = processor.get_supported_formats()
    print("📋 Supported file formats:")
    for category, mime_types in formats.items():
        print(f"  {category.upper()}: {len(mime_types)} types")
        for mime_type in mime_types[:3]:  # Show first 3
            print(f"    - {mime_type}")
        if len(mime_types) > 3:
            print(f"    ... and {len(mime_types) - 3} more")
    print()

    # Create and process sample files
    sample_files = create_sample_files()

    print("📁 Processing sample files...")
    processed_files = []

    for filename, content, mime_type in sample_files:
        try:
            print(f"  Processing: {filename} ({len(content)} bytes)")
            processed_file = await processor.process_file(
                filename, content, mime_type, user_id="demo_user"
            )
            processed_files.append(processed_file)

            print(f"    ✅ Status: {processed_file.status.value}")
            print(f"    📊 Type: {processed_file.file_type.value}")

            if processed_file.file_type == FileType.IMAGE:
                metadata = processed_file.processed_content.get("metadata", {})
                print(
                    f"    🖼️  Dimensions: {metadata.get('width')}x{metadata.get('height')}"
                )
            elif processed_file.file_type == FileType.DOCUMENT:
                content_info = processed_file.processed_content
                if content_info.get("extraction_available"):
                    print(
                        f"    📄 Text length: {len(content_info.get('text_content', ''))}"
                    )
                    print(
                        f"    📊 Word count: {content_info.get('metadata', {}).get('word_count', 0)}"
                    )

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print(f"\n✅ Processed {len(processed_files)} files successfully\n")
    return processed_files


async def demo_vision_analysis(processed_files):
    """Demonstrate vision analysis of images."""
    print("👁️  Demo: Vision Analysis")
    print("=" * 50)

    # Note: This demo uses mock analysis since we don't have real API keys
    print("📝 Note: Using mock vision analysis for demo purposes")
    print("    In production, this would use real vision models like GPT-4V\n")

    # vision_analyzer = get_vision_analyzer()  # Commented out as it's not used in demo

    # Find image files
    image_files = [f for f in processed_files if f.file_type == FileType.IMAGE]

    if not image_files:
        print("❌ No image files found for analysis")
        return

    print(f"🔍 Analyzing {len(image_files)} image(s)...")

    for image_file in image_files:
        print(f"\n📸 Analyzing: {image_file.original_name}")

        # Mock analysis (in real usage, this would call the actual model)
        mock_analysis = {
            "analysis": f"This image shows a sample chart with geometric shapes including a red rectangle and green circle. The image has dimensions {image_file.processed_content['metadata']['width']}x{image_file.processed_content['metadata']['height']} pixels and contains the text 'Sample Chart'.",
            "model_used": "mock-gpt-4-vision",
            "provider": "mock",
            "analysis_type": "general",
            "confidence": "high",
            "image_metadata": image_file.processed_content.get("metadata", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }

        print(f"  🤖 Model: {mock_analysis['model_used']}")
        print(f"  📊 Analysis: {mock_analysis['analysis'][:100]}...")
        print(f"  🎯 Confidence: {mock_analysis['confidence']}")

    print("\n✅ Vision analysis completed\n")


async def demo_state_integration(processed_files):
    """Demonstrate integration with agent state."""
    print("🔗 Demo: Agent State Integration")
    print("=" * 50)

    state_manager = get_multimodal_state_manager()

    # Create sample agent state
    state = AgentState(
        task="Analyze uploaded documents and images",
        messages=[],
        next="",
        scratchpad={},
        mode="direct",
        active_agents=["analyst"],
        multimodal_content={},
        session_id="demo-session-123",
        user_id="demo_user",
        last_updated=datetime.utcnow(),
        errors=[],
        task_status={},
    )

    print("📋 Initial state:")
    print(f"  Task: {state['task']}")
    print(f"  Mode: {state['mode']}")
    print(f"  Session: {state['session_id']}")
    print(f"  Multimodal content: {len(state['multimodal_content'])} items")

    # Add files to state
    print(f"\n📁 Adding {len(processed_files)} files to state...")
    updated_state = state_manager.add_files_to_state(state, processed_files)

    # Get statistics
    stats = state_manager.get_content_statistics(updated_state)
    print("📊 Content statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  File types: {stats['file_types']}")
    print(f"  Total size: {stats['total_size']} bytes")
    print(f"  Text documents: {stats['text_documents']}")
    print(f"  Total text length: {stats['total_text_length']} characters")

    # Create context message
    context_message = state_manager.create_multimodal_context_message(updated_state)
    if context_message:
        print("\n💬 Context message created:")
        print(f"  Length: {len(context_message.content)} characters")
        print(f"  Preview: {context_message.content[:150]}...")

    # Get text summary
    text_summary = state_manager.get_text_content_summary(updated_state)
    if text_summary:
        print("\n📄 Text content summary:")
        print(f"  Length: {len(text_summary)} characters")
        print(f"  Preview: {text_summary[:100]}...")

    print("\n✅ State integration completed\n")
    return updated_state


async def demo_file_retrieval(state):
    """Demonstrate file retrieval from state."""
    print("🔍 Demo: File Retrieval")
    print("=" * 50)

    state_manager = get_multimodal_state_manager()

    # Get files by type
    image_files = state_manager.get_files_by_type(state, "image")
    document_files = state_manager.get_files_by_type(state, "document")
    pdf_files = state_manager.get_files_by_type(state, "pdf")

    print(f"📸 Image files: {len(image_files)}")
    for file_data in image_files:
        print(f"  - {file_data['original_name']} ({file_data['file_size']} bytes)")

    print(f"📄 Document files: {len(document_files)}")
    for file_data in document_files:
        print(f"  - {file_data['original_name']} ({file_data['file_size']} bytes)")

    print(f"📋 PDF files: {len(pdf_files)}")
    for file_data in pdf_files:
        print(f"  - {file_data['original_name']} ({file_data['file_size']} bytes)")

    # Get specific file content
    if state["multimodal_content"]["files"]:
        file_id = list(state["multimodal_content"]["files"].keys())[0]
        file_content = state_manager.get_file_content(state, file_id)
        if file_content:
            print(f"\n🔍 Sample file content (ID: {file_id}):")
            print(f"  Name: {file_content['original_name']}")
            print(f"  Type: {file_content['file_type']}")
            print(f"  Status: {file_content['status']}")

    print("\n✅ File retrieval completed\n")


async def main():
    """Run the complete multimodal processing demo."""
    print("🚀 AgentHive Multimodal Processing Demo")
    print("=" * 60)
    print("This demo showcases the multimodal content processing pipeline")
    print("including file validation, processing, vision analysis, and")
    print("integration with agent state management.\n")

    try:
        # Demo 1: Basic file processing
        processed_files = await demo_basic_processing()

        # Demo 2: Vision analysis
        await demo_vision_analysis(processed_files)

        # Demo 3: State integration
        updated_state = await demo_state_integration(processed_files)

        # Demo 4: File retrieval
        await demo_file_retrieval(updated_state)

        print("🎉 Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ File validation and type detection")
        print("✅ Image processing and metadata extraction")
        print("✅ Document text extraction")
        print("✅ PDF processing (basic)")
        print("✅ Vision analysis integration")
        print("✅ Agent state management")
        print("✅ Content retrieval and statistics")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
