# AgentHive Direct Chat Demo

This demo showcases the single-agent direct chat graph functionality implemented in Task 5.

## Features Demonstrated

- **Session Management**: Create, manage, and clean up chat sessions
- **Graph Creation**: Build LangGraph workflows for direct agent interaction
- **Real AI Integration**: Chat with OpenAI GPT-4o-mini or Anthropic Claude-3-Haiku
- **State Persistence**: Maintain conversation context across interactions
- **Streaming Responses**: Real-time streaming chat responses
- **Error Handling**: Robust error recovery and validation
- **Graph Validation**: Ensure proper graph structure and configuration

## Setup

### Option 1: Quick Setup with Script

```bash
python examples/setup_demo_env.py
```

This interactive script will help you configure API keys.

### Option 2: Manual Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

### Option 3: .env File

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**Note**: You only need ONE of the API keys to run the demo with real AI models.

## Running the Demo

```bash
python examples/direct_chat_demo.py
```

## Demo Modes

### With API Keys (Recommended)
- Creates real agents powered by OpenAI or Anthropic models
- Executes actual conversations with AI models
- Demonstrates streaming responses
- Shows real error handling and recovery

### Without API Keys (Mock Mode)
- Shows the system architecture and flow
- Demonstrates session management
- Explains what would happen with real API keys
- Still validates the core functionality

## What You'll See

1. **Environment Setup**: Agent and model registration
2. **Session Management**: Creating and managing chat sessions
3. **Graph Creation**: Building and validating LangGraph workflows
4. **Real Chat Execution**: Actual conversations with AI agents
5. **Error Handling**: Validation and error recovery demonstrations

## Expected Output

```
Starting AgentHive Direct Chat Graph Demo
==================================================
✓ OpenAI API key found
Setting up demo environment...
Registered OpenAI model: gpt-4o-mini
Demo environment setup complete with 1 agents: ['openai_assistant']

=== Session Management Demo ===
Created session: demo_session_001
Session mode: direct
Active agent: openai_assistant
...

=== Graph Creation Demo ===
Processing agent: openai_assistant
Step 1: Agent validation
  Agent validation result: True
Step 2: Graph creation
  ✓ Successfully created graph for openai_assistant
...

=== Direct Chat Execution Demo ===
Attempting real chat execution with available agents...
Testing direct chat with openai_assistant...
Executing direct chat...
✓ Direct chat execution successful!
  Status: success
  Response: Hello! I'm an AI assistant created by OpenAI...
...
```

## Cost Considerations

The demo uses cost-efficient models:
- **OpenAI**: GPT-4o-mini (very low cost)
- **Anthropic**: Claude-3-Haiku (low cost)

Each demo run typically costs less than $0.01.

## Troubleshooting

### "No API keys found"
- Set at least one API key using the methods above
- The demo will run in mock mode without API keys

### "Model validation failed"
- Check that your API key is valid and has sufficient credits
- Ensure you have access to the specified models

### "Graph creation failed"
- This is expected without API keys
- With valid API keys, this should work

### Rate Limiting
- If you hit rate limits, wait a moment and try again
- The demo uses minimal requests to avoid this

## Architecture Overview

The demo showcases these key components:

```
DirectChatGraphFactory
├── SessionManager (session lifecycle)
├── AgentFactory (agent creation and validation)
├── ModelFactory (AI model integration)
└── LangGraph (workflow execution)
```

## Next Steps

After running the demo, you can:

1. Modify agent configurations in the demo script
2. Add your own custom agents
3. Integrate with the full AgentHive API
4. Build your own applications using the direct chat functionality

## API Keys Security

**Important**: Never commit API keys to version control!

- Use environment variables or .env files
- Add .env to your .gitignore
- Use different keys for development and production
- Rotate keys regularly

## Support

If you encounter issues:

1. Check that your API keys are valid
2. Ensure you have sufficient API credits
3. Verify your internet connection
4. Check the logs for detailed error messages

The demo is designed to be robust and will provide helpful error messages to guide you through any issues.