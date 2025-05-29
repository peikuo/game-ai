# Civ6-AI

An AI agent that plays Civilization VI using multimodal vision models and multi-agent collaboration.

## Overview

This project creates an AI agent that can play Civilization VI by:
1. Taking screenshots of the game
2. Using visual models to interpret the game state
3. Employing multiple agents to analyze and discuss the game from different perspectives
4. Planning and executing actions in the game

## Architecture

The system consists of several key components:
- **Screenshot Capture**: Takes screenshots of the Civilization VI game
- **Visual Processing**: Interprets game state using vision models
- **Multi-agent System**: Multiple specialized agents collaborate to analyze the game
- **Action Planning**: Decision-making based on agent discussions
- **Game Interaction**: Executes actions in the game

## Getting Started

### Prerequisites

- Python 3.9+
- Civilization VI installed
- Access to vision models (local or API)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/civ6-ai.git
cd civ6-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the main agent
python main.py
```

## Project Structure

```
civ6-ai/
├── config/              # Configuration files
├── docs/                # Documentation
├── examples/            # Example scripts
├── notebooks/           # Jupyter notebooks for experimentation
├── specs/               # Project specifications
├── src/                 # Source code
│   ├── capture/         # Screenshot capture functionality
│   ├── vision/          # Visual model integration
│   ├── agents/          # Multi-agent system implementation
│   ├── game_interface/  # Interface to interact with Civilization VI
│   └── utils/           # Utility functions
├── tests/               # Unit and integration tests
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── setup.py             # Package installation
```

## References

This project is built with inspiration from:
- [AGUVIS](https://github.com/xlang-ai/aguvis): A unified pure vision-based framework for autonomous GUI agents
- [OSWorld](https://github.com/xlang-ai/OSWorld): A desktop environment simulator for testing AI agents
- [Crew.AI](https://docs.crewai.com): An open-source Python framework for developing multi-agent AI systems

## License

[MIT License](LICENSE)
