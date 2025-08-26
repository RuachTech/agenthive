#!/usr/bin/env python3
"""
Setup script for AgentHive Direct Chat Demo environment variables.

This script helps you set up the required API keys for the demo.
"""

import os
from pathlib import Path


def main():
    """Set up environment variables for the demo."""
    print("AgentHive Direct Chat Demo - Environment Setup")
    print("=" * 50)
    
    # Check current environment
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print("\nCurrent environment status:")
    print(f"OPENAI_API_KEY: {'✓ Set' if openai_key else '✗ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✓ Set' if anthropic_key else '✗ Not set'}")
    
    if openai_key or anthropic_key:
        print("\n✓ At least one API key is configured!")
        print("You can run the demo with real AI models.")
        return
    
    print("\n⚠ No API keys found in environment variables.")
    print("\nTo run the demo with real AI models, you need to set at least one API key:")
    print("\nOption 1: Set environment variables in your shell:")
    print("  export OPENAI_API_KEY='your-openai-api-key-here'")
    print("  export ANTHROPIC_API_KEY='your-anthropic-api-key-here'")
    
    print("\nOption 2: Create a .env file in the project root:")
    print("  OPENAI_API_KEY=your-openai-api-key-here")
    print("  ANTHROPIC_API_KEY=your-anthropic-api-key-here")
    
    print("\nOption 3: Set them temporarily for this session:")
    
    # Interactive setup
    response = input("\nWould you like to set API keys interactively for this session? (y/n): ")
    
    if response.lower().startswith('y'):
        setup_interactive()
    else:
        print("\nYou can still run the demo in mock mode without API keys.")
        print("Run: python examples/direct_chat_demo.py")


def setup_interactive():
    """Interactive API key setup."""
    print("\nInteractive API Key Setup")
    print("-" * 30)
    
    # OpenAI API Key
    openai_key = input("Enter your OpenAI API Key (or press Enter to skip): ").strip()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        print("✓ OpenAI API key set for this session")
    
    # Anthropic API Key
    anthropic_key = input("Enter your Anthropic API Key (or press Enter to skip): ").strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        print("✓ Anthropic API key set for this session")
    
    if openai_key or anthropic_key:
        print("\n✓ API keys configured! You can now run the demo:")
        print("  python examples/direct_chat_demo.py")
        
        # Offer to create .env file
        save_env = input("\nWould you like to save these keys to a .env file? (y/n): ")
        if save_env.lower().startswith('y'):
            create_env_file(openai_key, anthropic_key)
    else:
        print("\nNo API keys provided. Demo will run in mock mode.")


def create_env_file(openai_key, anthropic_key):
    """Create a .env file with the API keys."""
    env_path = Path(".env")
    
    # Read existing .env if it exists
    existing_content = ""
    if env_path.exists():
        with open(env_path, 'r') as f:
            existing_content = f.read()
    
    # Prepare new content
    new_lines = []
    
    if openai_key and "OPENAI_API_KEY" not in existing_content:
        new_lines.append(f"OPENAI_API_KEY={openai_key}")
    
    if anthropic_key and "ANTHROPIC_API_KEY" not in existing_content:
        new_lines.append(f"ANTHROPIC_API_KEY={anthropic_key}")
    
    if new_lines:
        with open(env_path, 'a') as f:
            if existing_content and not existing_content.endswith('\n'):
                f.write('\n')
            f.write('\n'.join(new_lines) + '\n')
        
        print(f"✓ API keys saved to {env_path}")
        print("Note: Make sure .env is in your .gitignore file!")
    else:
        print("API keys already exist in .env file")


if __name__ == "__main__":
    main()