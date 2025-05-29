# Civ6-AI System Architecture

## System Overview

The Civ6-AI project is designed to create an AI agent that can play Civilization VI using multimodal vision models and multi-agent collaboration. The system takes screenshots of the game, analyzes them using vision models, discusses the game state through multiple specialized agents, makes decisions, and executes actions in the game.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Civ6-AI System                              │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Main Application                             │
└─────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Capture    │  │    Vision    │  │    Agents    │  │     Game     │
│    Module    │  │    Module    │  │    Module    │  │   Interface  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Screenshot  │  │    Image     │  │    Agent     │  │     Game     │
│   Capturer   │  │   Analyzer   │  │   Manager    │  │  Controller  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                                             │
                                             ▼
                                   ┌──────────────────────┐
                                   │    Multi-Agent System │
                                   └──────────────────────┘
                                             │
                   ┌───────────────────────────────────────────────┐
                   ▼                 ▼                 ▼           ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │  Strategic   │  │   Military   │  │   Economic   │  │  Diplomatic  │
         │   Advisor    │  │  Commander   │  │   Advisor    │  │    Envoy     │
         └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                                                                       │
                                                                       ▼
                                                             ┌──────────────────┐
                                                             │      Chief       │
                                                             │    Executive     │
                                                             └──────────────────┘
          │
          ▼
┌──────────────────┐
│       TTS        │
│      Module      │
└──────────────────┘
          │
          ▼
┌──────────────────┐
│      Voice       │
│   Synthesizer    │
└──────────────────┘
```

## Sequence Diagram

```
┌─────────┐          ┌───────────┐          ┌───────────┐          ┌──────────┐          ┌─────────┐          ┌────────┐
│  Main   │          │ Screenshot │          │   Image   │          │  Agent   │          │  Game   │          │  TTS   │
│ Process │          │  Capturer  │          │  Analyzer │          │ Manager  │          │Interface │          │ Module │
└────┬────┘          └─────┬─────┘          └─────┬─────┘          └────┬─────┘          └────┬────┘          └────┬───┘
     │                     │                      │                     │                     │                     │                    │
     │ Initialize          │                      │                     │                     │                     │                    │
     │─────────────────────>                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Initialize          │                      │                     │                     │                     │                    │
     │────────────────────────────────────────────>                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Initialize          │                      │                     │                     │                     │                    │
     │───────────────────────────────────────────────────────────────────>                    │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Initialize          │                      │                     │                     │                     │                    │
     │──────────────────────────────────────────────────────────────────────────────────────────>                    │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Initialize (if TTS enabled)                │                     │                     │                     │                    │
     │─────────────────────────────────────────────────────────────────────────────────────────────────────────────>                    │
     │                     │                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Start Main Loop     │                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Capture Screenshot  │                      │                     │                     │                     │                    │
     │─────────────────────>                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │                     │ Return Screenshot    │                     │                     │                     │                    │
     │<─────────────────────                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Analyze Screenshot  │                      │                     │                     │                     │                    │
     │────────────────────────────────────────────>                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │                     │                      │ Return Game State   │                     │                     │                    │
     │<────────────────────────────────────────────                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Process Game State  │                      │                     │                     │                     │                    │
     │───────────────────────────────────────────────────────────────────>                    │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │                     │                      │                     │ Return Action &     │                     │                    │
     │                     │                      │                     │ Agent Thoughts      │                     │                     │
     │<───────────────────────────────────────────────────────────────────                    │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Vocalize Thoughts (if TTS enabled)         │                     │                     │                     │                    │
     │─────────────────────────────────────────────────────────────────────────────────────────────────────────────>                    │
     │                     │                      │                     │                     │                     │                    │
     │ Execute Action      │                      │                     │                     │                     │                    │
     │──────────────────────────────────────────────────────────────────────────────────────────>                    │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Loop continues...   │                      │                     │                     │                     │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Cleanup (on exit)   │                      │                     │                     │                     │                    │
     │──────────────────────────────────────────────────────────────────────────────────────────>                    │                    │
     │                     │                      │                     │                     │                     │                    │
     │ Stop TTS (if enabled)                      │                     │                     │                     │                    │
     │─────────────────────────────────────────────────────────────────────────────────────────────────────────────>                    │
┌────┴────┐          ┌─────┴─────┐          ┌─────┴─────┐          ┌────┴─────┐          ┌────┴────┐          ┌────┴───┐
│  Main   │          │ Screenshot │          │   Image   │          │  Agent   │          │  Game   │          │  TTS   │
│ Process │          │  Capturer  │          │  Analyzer │          │ Manager  │          │Interface │          │ Module │
└─────────┘          └───────────┘          └───────────┘          └──────────┘          └─────────┘          └────────┘
```

## Component Descriptions

### 1. Screenshot Capture Module (`src/capture/`)

**Purpose**: Captures screenshots of the Civilization VI game window.

**Key Components**:
- `ScreenCapturer`: Takes screenshots of the game window, either full screen or a specific region.

**Functionality**:
- Captures screenshots at regular intervals
- Optionally saves screenshots to disk
- Provides different output formats (PIL Image, numpy array, OpenCV format)

### 2. Vision Module (`src/vision/`)

**Purpose**: Analyzes game screenshots to interpret the game state.

**Key Components**:
- `ImageAnalyzer`: Processes screenshots using vision models to extract game state information.

**Functionality**:
- Supports multiple vision model backends (Ollama, Qwen V3)
- Converts images to base64 for API requests
- Parses model responses into structured game state information

### 3. Agents Module (`src/agents/`)

**Purpose**: Manages multiple AI agents that collaborate to analyze the game and make decisions.

**Key Components**:
- `AgentManager`: Coordinates multiple specialized agents using crew.ai.
- Agent roles: Strategic Advisor, Military Commander, Economic Advisor, Diplomatic Envoy, Chief Executive

**Functionality**:
- Initializes agents with specific roles and goals
- Creates tasks for each agent
- Processes game state through the agent crew
- Collects agent thoughts and decisions
- Provides fallback processing when crew.ai is not available

### 4. Game Interface Module (`src/game_interface/`)

**Purpose**: Interacts with the Civilization VI game by executing actions.

**Key Components**:
- `GameController`: Executes actions in the game through keyboard and mouse inputs.

**Functionality**:
- Translates high-level actions to game inputs
- Handles keyboard and mouse interactions
- Manages game window focus

### 5. TTS Module (`src/tts/`)

**Purpose**: Vocalizes agent thoughts and actions with different voices.

**Key Components**:
- `VoiceSynthesizer`: Converts text to speech with different voices for each agent.

**Functionality**:
- Supports multiple TTS engines (edge-tts, pyttsx3)
- Provides unique voice profiles for different agents
- Processes TTS requests asynchronously



## Data Flow

1. The main process captures a screenshot of the Civilization VI game.
2. The screenshot is sent to the vision module for analysis.
3. The vision module returns a structured representation of the game state.
4. The game state is sent to the agent manager for processing.
5. The agent manager distributes the game state to specialized agents.
6. Agents analyze the game state from their perspective and provide recommendations.
7. The Chief Executive agent makes the final decision based on all recommendations.
8. Agent thoughts are vocalized through the TTS module (if enabled).
9. The final action is executed in the game through the game interface.
10. The process repeats for the next game state.

## Configuration

The system is highly configurable through YAML files:

- `config/default.yaml`: Main configuration for all components
- `config/agents.yaml`: Configuration for agent roles and responsibilities

## Command-Line Interface

The main application provides several command-line options:

- `--config`: Path to configuration file
- `--debug`: Enable debug mode
- `--model`: Vision model to use (ollama, qwen)
- `--agent-config`: Path to agent configuration file
- `--tts`: Enable text-to-speech for agent vocalization
