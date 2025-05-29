I want to made a game player agent controlled by AI model.

The game I want to play is Civilization VI.

The AI MultiModal agent will screenshot the game and send it to visual model, then get the detail description of the game screen.

Then multiagent will discuss the game screen from different perspectives, plan the action, then make a decision.

I have two projects for reference:
1. ~/Workspace/ai/aguvis
2. ~/Workspace/ai/OSWorld
aguvis has a VLLM model for describing image, and a LLM model for reasoning. 
OSWorld is for reference, it is a simulator for game, it can simulate the game environment, and return the game screen to AI agent, we do not use it.

We can use aguvis with ollama locally or Qwen V3 API, we need multiple models for test which have best efficiency.

we can use crew.ai for multiagent.

This project language is majorly python.

Game playing will be living on bilibili. and we need tts to speak out the thinking and action for agent. And multi agent can have different voice and speaking style.