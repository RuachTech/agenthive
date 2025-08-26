"""Demo script showing how to use the model abstraction layer."""

import asyncio
import os
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

from agent_hive.core.models import ModelProvider, ModelConfig, get_model_factory


async def demo_model_abstraction() -> None:
    """Demonstrate the model abstraction layer functionality."""
    print("🤖 AgentHive Model Abstraction Layer Demo")
    print("=" * 50)

    # Get the global model factory
    factory = get_model_factory()

    # Configure models (using dummy API keys for demo)
    models_to_register: List[Dict[str, Any]] = [
        {
            "name": "gpt-4-primary",
            "config": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
                temperature=0.7,
                max_tokens=1000,
            ),
            "fallbacks": ["claude-fallback"],
        },
        {
            "name": "claude-fallback",
            "config": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key"),
                temperature=0.5,
                max_tokens=1000,
            ),
            "fallbacks": [],
        },
        {
            "name": "gemini-experimental",
            "config": ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                api_key=os.getenv("GOOGLE_API_KEY", "dummy-key"),
                temperature=0.8,
                max_tokens=1000,
            ),
            "fallbacks": [],
        },
    ]

    # Register all models
    print("\n📝 Registering models...")
    for model_info in models_to_register:
        try:
            factory.register_model(
                model_info["name"], model_info["config"], model_info["fallbacks"]
            )
            print(
                f"✅ Registered: {model_info['name']} ({model_info['config'].provider.value})"
            )
        except Exception as e:
            print(f"❌ Failed to register {model_info['name']}: {e}")

    # List all registered models
    print(f"\n📋 Available models: {factory.list_models()}")

    # Check model availability (will fail with dummy keys, but shows the mechanism)
    print("\n🔍 Checking model availability...")
    availability = await factory.check_all_models()
    for model_name, is_available in availability.items():
        status = "✅ Available" if is_available else "❌ Unavailable"
        print(f"  {model_name}: {status}")

    # Demonstrate getting a model (with fallback logic)
    print("\n🎯 Getting model with fallback logic...")
    try:
        model = await factory.get_model("gpt-4-primary")
        print(f"✅ Got model: {model.model_name} (provider: {model.provider})")

        # Try to generate (will fail with dummy keys, but shows the interface)
        print("\n💬 Attempting generation (will fail with dummy keys)...")
        messages = [HumanMessage(content="Hello, how are you?")]

        try:
            response = await model.generate(messages)
            print(f"✅ Response: {response.content}")
            print(f"📊 Usage: {response.usage}")
        except Exception as e:
            print(
                f"❌ Generation failed (expected with dummy keys): {type(e).__name__}"
            )

    except Exception as e:
        print(f"❌ Failed to get model: {e}")

    # Demonstrate streaming (conceptual)
    print("\n🌊 Streaming interface example...")
    try:
        model = await factory.get_model("claude-fallback")
        print(f"✅ Got streaming model: {model.model_name}")

        messages = [HumanMessage(content="Tell me a short story")]
        print("📡 Starting stream (will fail with dummy keys)...")

        try:
            async for chunk in model.stream_generate(messages):
                print(f"📝 Chunk: {chunk}", end="", flush=True)
            print()  # New line after streaming
        except Exception as e:
            print(f"❌ Streaming failed (expected with dummy keys): {type(e).__name__}")

    except Exception as e:
        print(f"❌ Failed to get streaming model: {e}")

    print("\n🎉 Demo completed!")
    print("\n💡 To use with real API keys:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("   export GOOGLE_API_KEY='your-google-key'")


if __name__ == "__main__":
    asyncio.run(demo_model_abstraction())
