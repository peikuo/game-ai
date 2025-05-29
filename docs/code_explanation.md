# Civ6-AI Code Explanation

This document provides a detailed explanation of the Civ6-AI project's code structure, key components, and how they interact with each other.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Main Application Flow](#main-application-flow)
3. [Screenshot Capture Module](#screenshot-capture-module)
4. [Vision Module](#vision-module)
5. [Agent System](#agent-system)
6. [Game Interface](#game-interface)
7. [Text-to-Speech Module](#text-to-speech-module)
8. [Configuration System](#configuration-system)
9. [Example Workflow](#example-workflow)

## Project Overview

The Civ6-AI project creates an AI agent that can play Civilization VI by:
1. Taking screenshots of the game
2. Using visual models to interpret the game state
3. Employing multiple specialized agents to analyze the game from different perspectives
4. Making decisions based on the collective analysis
5. Executing actions in the game

The project is organized in a modular structure where each component has a specific responsibility:

```
civ6-ai/
├── config/              # Configuration files
├── docs/                # Documentation
├── src/                 # Source code
│   ├── capture/         # Screenshot capture functionality
│   ├── vision/          # Visual model integration
│   ├── agents/          # Multi-agent system implementation
│   ├── game_interface/  # Interface to interact with Civilization VI
│   ├── tts/             # Text-to-speech functionality
│   └── utils/           # Utility functions
├── main.py              # Entry point
└── requirements.txt     # Dependencies
```

## Main Application Flow

The main application (`main.py`) orchestrates the entire system. Here's a breakdown of its functionality:

```python
# Simplified main.py flow
def main():
    # 1. Parse command-line arguments
    args = parse_args()
    
    # 2. Load configuration
    config = config_loader.load_config(args.config)
    
    # 3. Initialize components
    screen_capturer = screenshot.ScreenCapturer(config["capture"])
    analyzer = image_analyzer.ImageAnalyzer(model=args.model, config=config["vision"])
    game_interface = game_controller.GameController(config["game"])
    
    # 4. Initialize TTS if enabled
    tts_engine = None
    if args.tts:
        tts_engine = voice_synthesizer.VoiceSynthesizer(config.get("tts", {}))
    
    # 5. Initialize agent manager
    agent_mgr = agent_manager.AgentManager(
        config_path=args.agent_config,
        vision_model=analyzer,
        tts_engine=tts_engine
    )
    
    # 6. Main loop
    while True:
        # a. Capture screenshot
        image = screen_capturer.capture()
        
        # b. Analyze screenshot
        game_state = analyzer.analyze(image)
        
        # c. Process with agents
        action, agent_thoughts = agent_mgr.process(game_state)
        
        # d. Vocalize agent thoughts if TTS is enabled
        if args.tts and tts_engine and agent_thoughts:
            for agent_name, thought in agent_thoughts.items():
                tts_engine.synthesize(thought, agent_name=agent_name)
        
        # e. Execute action
        game_interface.execute_action(action)
```

The main loop continuously:
1. Captures a screenshot of the game
2. Analyzes the screenshot to understand the game state
3. Processes the game state through the agent system
4. Vocalizes agent thoughts (if TTS is enabled)
5. Executes the decided action in the game

## Screenshot Capture Module

The screenshot capture module (`src/capture/screenshot.py`) is responsible for taking screenshots of the Civilization VI game window.

### Key Classes and Methods

#### `ScreenCapturer` Class

```python
class ScreenCapturer:
    def __init__(self, config):
        self.region = config.get("region", None)
        self.save_path = Path(config.get("save_path", "screenshots"))
        self.save_screenshots = config.get("save_screenshots", False)
    
    def capture(self):
        """Capture a screenshot of the game window."""
        # Capture full screen or specific region
        if self.region:
            screenshot = ImageGrab.grab(bbox=self.region)
        else:
            screenshot = ImageGrab.grab()
        
        # Save screenshot if enabled
        if self.save_screenshots:
            # Save logic...
        
        return screenshot
    
    def capture_to_array(self):
        """Convert screenshot to numpy array."""
        # ...
    
    def capture_to_cv2(self):
        """Convert screenshot to OpenCV format."""
        # ...
```

The `ScreenCapturer` can:
- Capture the entire screen or a specific region
- Save screenshots to disk (optional)
- Convert screenshots to different formats (PIL Image, numpy array, OpenCV)

## Vision Module

The vision module (`src/vision/image_analyzer.py`) analyzes screenshots using visual models to interpret the game state.

### Key Classes and Methods

#### `ImageAnalyzer` Class

```python
class ImageAnalyzer:
    def __init__(self, model="ollama", config=None):
        self.model_type = model.lower()
        self.config = config or {}
        
        # Set up model-specific configurations
        if self.model_type == "ollama":
            self.model_name = self.config.get("ollama_model", "llava:latest")
            self.api_url = self.config.get("ollama_api", "http://localhost:11434/api/generate")
        elif self.model_type == "qwen":
            self.api_key = self.config.get("qwen_api_key", os.environ.get("QWEN_API_KEY"))
            self.api_url = self.config.get("qwen_api", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    
    def analyze(self, image, prompt=None):
        """Analyze a screenshot of Civilization VI."""
        if prompt is None:
            prompt = """
            Analyze this Civilization VI game screenshot in detail. Describe:
            1. Current game phase and turn number
            2. Player's civilization and resources
            3. Visible units and their positions
            4. Cities and their status
            5. Terrain features and strategic resources
            6. Any notifications or alerts
            7. Current research and civic progress
            8. Diplomatic status with other civilizations
            9. Any other important game state information
            """
        
        # Call appropriate model API based on model_type
        if self.model_type == "ollama":
            return self._analyze_with_ollama(image, prompt)
        elif self.model_type == "qwen":
            return self._analyze_with_qwen(image, prompt)
```

The `ImageAnalyzer`:
- Supports multiple vision model backends (Ollama, Qwen V3)
- Sends screenshots to the vision model with a detailed prompt
- Parses the model's response into a structured game state representation

### Model Integration

The vision module can integrate with:

1. **Ollama** (local):
   - Uses the Ollama API to run models like LLaVA locally
   - Sends base64-encoded images and receives text descriptions

2. **Qwen V3** (API):
   - Uses the Qwen API for cloud-based vision analysis
   - Requires an API key for authentication

## Agent System

The agent system (`src/agents/agent_manager.py`) manages multiple specialized agents that collaborate to analyze the game and make decisions.

### Key Classes and Methods

#### `AgentManager` Class

```python
class AgentManager:
    def __init__(self, config_path, vision_model=None, tts_engine=None):
        self.config_path = Path(config_path)
        self.vision_model = vision_model
        self.tts_engine = tts_engine
        self.config = self._load_config()
        self.agents = []
        self.tasks = []
        self.crew = None
        self.agent_thoughts = {}
        
        # Initialize agents if crew.ai is available
        if CREWAI_AVAILABLE:
            self._initialize_agents()
            self._initialize_tasks()
            self._initialize_crew()
    
    def process(self, game_state):
        """Process the game state and return recommended actions."""
        # Clear previous agent thoughts
        self.agent_thoughts = {}
        
        if CREWAI_AVAILABLE and self.crew:
            action = self._process_with_crew(game_state)
        else:
            action = self._process_fallback(game_state)
            
        return action, self.agent_thoughts
```

The `AgentManager`:
- Loads agent configurations from a YAML file
- Creates specialized agents with different roles and goals
- Processes game state through the agent crew
- Collects agent thoughts and decisions
- Provides a fallback mechanism when crew.ai is not available

### Agent Roles

The system uses multiple specialized agents:

1. **Strategic Advisor**: Analyzes the overall game state and provides strategic recommendations
2. **Military Commander**: Analyzes military situations and recommends unit movements
3. **Economic Advisor**: Analyzes economic situations and recommends production priorities
4. **Diplomatic Envoy**: Analyzes diplomatic relations and recommends diplomatic actions
5. **Chief Executive**: Reviews all analyses and makes final decisions

### CrewAI Integration

The agent system uses the CrewAI framework to orchestrate the collaboration between agents:

```python
def _initialize_crew(self):
    """Initialize the crew with agents and tasks."""
    process = Process.SEQUENTIAL
    if self.config.get("process", "sequential").lower() == "hierarchical":
        process = Process.HIERARCHICAL
    
    self.crew = Crew(
        agents=self.agents,
        tasks=self.tasks,
        verbose=2,
        process=process
    )
```

The crew can be configured to work in either:
- **Sequential** mode: Agents work one after another
- **Hierarchical** mode: Agents can delegate tasks to each other

## Game Interface

The game interface module (`src/game_interface/game_controller.py`) interacts with the Civilization VI game by executing actions.

### Key Classes and Methods

#### `GameController` Class

```python
class GameController:
    def __init__(self, config):
        self.config = config
        self.key_bindings = config.get("key_bindings", {})
        self.window_title = config.get("window_title", "Sid Meier's Civilization VI")
    
    def execute_action(self, action):
        """Execute an action in the game."""
        action_type = action.get("action", "")
        
        # Focus the game window
        self._focus_game_window()
        
        # Execute the appropriate action
        if action_type == "move_unit":
            self._move_unit(action.get("direction", ""))
        elif action_type == "build_unit":
            self._build_unit(action.get("unit_type", ""))
        elif action_type == "research_technology":
            self._research_technology(action.get("technology", ""))
        elif action_type == "skip_turn":
            self._skip_turn()
        # ... other action types
```

The `GameController`:
- Focuses the game window
- Translates high-level actions to game inputs
- Handles keyboard and mouse interactions
- Manages game window focus

## Text-to-Speech Module

The text-to-speech module (`src/tts/voice_synthesizer.py`) vocalizes agent thoughts with different voices.

### Key Classes and Methods

#### `VoiceSynthesizer` Class

```python
class VoiceSynthesizer:
    def __init__(self, config=None):
        self.config = config or {}
        self.engine = self.config.get("engine", "edge-tts")
        self.output_dir = Path(self.config.get("output_dir", "tts_output"))
        
        # Voice profiles for different agents
        self.voice_profiles = self.config.get("voice_profiles", {
            "StrategicAdvisor": {
                "voice": "en-US-ChristopherNeural",
                "rate": "+0%",
                "pitch": "+0Hz",
                "style": "serious"
            },
            # ... other agent profiles
        })
        
        # Queue for asynchronous TTS processing
        self.tts_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
    
    def synthesize(self, text, agent_name="default", output_file=None, blocking=False):
        """Synthesize speech for the given text using the agent's voice profile."""
        # Get voice profile for the agent
        profile = self.voice_profiles.get(agent_name, self.voice_profiles["default"])
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = int(time.time())
            output_file = self.output_dir / f"{agent_name}_{timestamp}.mp3"
        
        # Add to queue for processing
        task = {
            "text": text,
            "profile": profile,
            "output_file": str(output_file),
            "agent_name": agent_name
        }
        
        if blocking:
            return self._process_tts_task(task)
        else:
            self.tts_queue.put(task)
            self._ensure_processing_thread()
            return str(output_file)
```

The `VoiceSynthesizer`:
- Supports multiple TTS engines (edge-tts, pyttsx3)
- Provides unique voice profiles for different agents
- Processes TTS requests asynchronously
- Saves audio files to disk

### Voice Profiles

Each agent has a unique voice profile:

```yaml
voice_profiles:
  StrategicAdvisor:
    voice: "en-US-ChristopherNeural"
    rate: "+0%"
    pitch: "+0Hz"
    style: "serious"
  MilitaryCommander:
    voice: "en-US-GuyNeural"
    rate: "+10%"
    pitch: "-2Hz"
    style: "commanding"
  # ... other profiles
```

## Configuration System

The configuration system uses YAML files to store settings for all components:

### Main Configuration (`config/default.yaml`)

```yaml
# Screenshot capture configuration
capture:
  region: null  # null for full screen, or [x, y, width, height]
  save_path: "screenshots"
  save_screenshots: true

# Vision model configuration
vision:
  # Ollama configuration
  ollama_model: "llava:latest"
  ollama_api: "http://localhost:11434/api/generate"
  
  # Qwen configuration
  qwen_api_key: ""  # Set this or use environment variable QWEN_API_KEY
  qwen_api: "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# Game interface configuration
game:
  window_title: "Sid Meier's Civilization VI"
  key_bindings:
    next_turn: "end"
    escape: "esc"
    # ... other key bindings

# Text-to-speech configuration
tts:
  engine: "edge-tts"  # "edge-tts" or "pyttsx3"
  output_dir: "tts_output"
  voice_profiles:
    # ... agent voice profiles
```

### Agent Configuration (`config/agents.yaml`)

```yaml
# Agent definitions
agents:
  - name: "StrategicAdvisor"
    role: "Strategic advisor for Civilization VI"
    goal: "Analyze the game state and provide strategic advice for long-term planning"
    backstory: "An experienced Civilization VI player with deep knowledge of game mechanics and strategies."
  
  # ... other agent definitions

# Process configuration
process: "sequential"  # or "hierarchical"
```

## Example Workflow

Here's a detailed example of how the system processes a single game turn:

1. **Screenshot Capture**:
   ```python
   # In main.py
   image = screen_capturer.capture()
   ```
   The system takes a screenshot of the Civilization VI game window.

2. **Vision Analysis**:
   ```python
   # In main.py
   game_state = analyzer.analyze(image)
   ```
   The screenshot is sent to the vision model (Ollama or Qwen V3), which analyzes it and returns a structured representation of the game state.

3. **Agent Processing**:
   ```python
   # In main.py
   action, agent_thoughts = agent_mgr.process(game_state)
   ```
   
   Inside the agent manager:
   ```python
   # In agent_manager.py
   def _process_with_crew(self, game_state):
       # Format game state for the crew
       game_state_str = self._format_game_state(game_state)
       
       # Update task contexts with the current game state
       for task in self.tasks:
           task.context = game_state_str
       
       # Run the crew
       result = self.crew.kickoff()
       
       # Collect agent thoughts
       for agent in self.agents:
           if hasattr(agent, 'last_output') and agent.last_output:
               self.agent_thoughts[agent.name] = agent.last_output
       
       # Parse the result
       return self._parse_crew_result(result)
   ```
   
   The game state is processed by multiple specialized agents:
   - The **Strategic Advisor** analyzes the overall game state
   - The **Military Commander** focuses on military aspects
   - The **Economic Advisor** analyzes resources and production
   - The **Diplomatic Envoy** evaluates diplomatic relations
   - The **Chief Executive** makes the final decision

4. **Text-to-Speech** (if enabled):
   ```python
   # In main.py
   if args.tts and tts_engine and agent_thoughts:
       for agent_name, thought in agent_thoughts.items():
           tts_engine.synthesize(thought, agent_name=agent_name)
   ```
   
   The system vocalizes the thoughts of each agent using their unique voice profile.

5. **Action Execution**:
   ```python
   # In main.py
   game_interface.execute_action(action)
   ```
   
   The system executes the decided action in the game:
   ```python
   # In game_controller.py
   def execute_action(self, action):
       action_type = action.get("action", "")
       
       # Focus the game window
       self._focus_game_window()
       
       # Execute the appropriate action
       if action_type == "move_unit":
           self._move_unit(action.get("direction", ""))
       elif action_type == "build_unit":
           self._build_unit(action.get("unit_type", ""))
       # ... other action types
   ```

This cycle repeats for each turn of the game, allowing the AI agent to play Civilization VI autonomously.

## Conclusion

The Civ6-AI project is a sophisticated system that combines computer vision, multi-agent AI, and game automation to create an autonomous player for Civilization VI. The modular architecture allows for easy extension and customization of different components.

Key strengths of the system include:
- Modular design with clear separation of concerns
- Support for multiple vision models (local and cloud-based)
- Multi-agent collaboration for complex decision-making
- Customizable agent roles and voice profiles
- Flexible configuration system

For further details on specific components, refer to the source code and configuration files in the project repository.
