"""
Agent manager for coordinating multiple AI agents using crew.ai
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Crew, Process, Task

    CREWAI_AVAILABLE = True
except ImportError:
    logger.warning(
        "crewai not installed. Multi-agent functionality will be limited.")
    CREWAI_AVAILABLE = False


class AgentManager:
    """
    Manages multiple AI agents for analyzing and playing Civilization VI.
    Uses crew.ai for agent orchestration.
    """

    def __init__(self, config_path, vision_model=None, tts_engine=None):
        """
        Initialize the AgentManager.

        Args:
            config_path (str): Path to agent configuration file
            vision_model: Vision model for image analysis
        """
        self.config_path = Path(config_path)
        self.vision_model = vision_model
        self.tts_engine = tts_engine
        self.config = self._load_config()
        self.agents = []
        self.tasks = []
        self.crew = None
        self.agent_thoughts = {}

        if CREWAI_AVAILABLE:
            self._initialize_agents()
            self._initialize_tasks()
            self._initialize_crew()
        else:
            logger.warning(
                "CrewAI not available. Using fallback single-agent mode.")

    def _load_config(self):
        """
        Load agent configuration from YAML file.

        Returns:
            dict: Agent configuration
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self):
        """
        Create default configuration for agents.

        Returns:
            dict: Default agent configuration
        """
        return {
            "agents": [
                {
                    "name": "StrategicAdvisor",
                    "role": "Strategic advisor for Civilization VI",
                    "goal": "Analyze the game state and provide strategic advice",
                    "backstory": "An experienced Civilization VI player with deep knowledge of game mechanics and strategies.",
                },
                {
                    "name": "MilitaryCommander",
                    "role": "Military commander for Civilization VI",
                    "goal": "Analyze military situations and recommend unit movements and combat strategies",
                    "backstory": "A tactical genius with expertise in military operations and warfare in Civilization VI.",
                },
                {
                    "name": "EconomicAdvisor",
                    "role": "Economic advisor for Civilization VI",
                    "goal": "Optimize resource usage, production, and economic growth",
                    "backstory": "An economic expert who understands how to build a thriving civilization economy.",
                },
                {
                    "name": "DiplomaticEnvoy",
                    "role": "Diplomatic envoy for Civilization VI",
                    "goal": "Manage relationships with other civilizations and city-states",
                    "backstory": "A skilled diplomat who knows how to navigate complex international relations.",
                },
                {
                    "name": "ChiefExecutive",
                    "role": "Chief executive for decision making",
                    "goal": "Make final decisions based on input from all advisors",
                    "backstory": "A decisive leader who can weigh different perspectives and make optimal choices.",
                },
            ],
            "process": "sequential",  # or "hierarchical"
        }

    def _initialize_agents(self):
        """
        Initialize crew.ai agents based on configuration.
        """
        if not CREWAI_AVAILABLE:
            return

        self.agents = []
        for agent_config in self.config.get("agents", []):
            agent = Agent(
                name=agent_config.get(
                    "name",
                    "Advisor"),
                role=agent_config.get(
                    "role",
                    "Game advisor"),
                goal=agent_config.get(
                    "goal",
                    "Analyze the game and provide advice"),
                backstory=agent_config.get(
                    "backstory",
                    "An experienced Civilization VI player"),
                verbose=True,
                allow_delegation=True,
            )
            self.agents.append(agent)
            logger.debug(f"Initialized agent: {agent.name}")

    def _initialize_tasks(self):
        """
        Initialize tasks for the agents.
        """
        if not CREWAI_AVAILABLE or not self.agents:
            return

        # Create tasks based on agent roles
        self.tasks = []

        # Strategic Advisor Task
        if len(self.agents) > 0:
            self.tasks.append(
                Task(
                    description="Analyze the overall game state and provide strategic recommendations",
                    expected_output="Strategic analysis and recommendations",
                    agent=self.agents[0],
                ))

        # Military Commander Task
        if len(self.agents) > 1:
            self.tasks.append(
                Task(
                    description="Analyze military situation and recommend unit movements",
                    expected_output="Military analysis and tactical recommendations",
                    agent=self.agents[1],
                ))

        # Economic Advisor Task
        if len(self.agents) > 2:
            self.tasks.append(
                Task(
                    description="Analyze economic situation and recommend production priorities",
                    expected_output="Economic analysis and production recommendations",
                    agent=self.agents[2],
                ))

        # Diplomatic Envoy Task
        if len(self.agents) > 3:
            self.tasks.append(
                Task(
                    description="Analyze diplomatic relations and recommend diplomatic actions",
                    expected_output="Diplomatic analysis and recommendations",
                    agent=self.agents[3],
                ))

        # Chief Executive Task
        if len(self.agents) > 4:
            self.tasks.append(
                Task(
                    description="Review all analyses and make final decisions",
                    expected_output="Final decision on actions to take",
                    agent=self.agents[4],
                    context=[task.description for task in self.tasks],
                )
            )

    def _initialize_crew(self):
        """
        Initialize the crew with agents and tasks.
        """
        if not CREWAI_AVAILABLE or not self.agents or not self.tasks:
            return

        process = Process.SEQUENTIAL
        if self.config.get("process", "sequential").lower() == "hierarchical":
            process = Process.HIERARCHICAL

        self.crew = Crew(
            agents=self.agents, tasks=self.tasks, verbose=2, process=process
        )
        logger.info(
            f"Initialized crew with {len(self.agents)} agents and {len(self.tasks)} tasks"
        )

    def process(self, game_state):
        """
            Process the game state with the agent crew and
        return recommended actions.

            Args:
                game_state (dict): Current game state information

            Returns:
                tuple: (dict, dict) - Recommended actions to take and
        agent thoughts
        """
        # Clear previous agent thoughts
        self.agent_thoughts = {}

        if CREWAI_AVAILABLE and self.crew:
            action = self._process_with_crew(game_state)
        else:
            action = self._process_fallback(game_state)

        return action, self.agent_thoughts

    def _process_with_crew(self, game_state):
        """
        Process game state using crew.ai.

        Args:
            game_state (dict): Current game state information

        Returns:
            dict: Recommended actions from the crew
        """
        try:
            # Convert game state to a format the crew can use
            game_state_str = self._format_game_state(game_state)

            # Update task contexts with the current game state
            for task in self.tasks:
                task.context = game_state_str

            # Run the crew
            result = self.crew.kickoff()

            # Collect agent thoughts during processing
            for agent in self.agents:
                if hasattr(agent, "last_output") and agent.last_output:
                    self.agent_thoughts[agent.name] = agent.last_output

            # Parse the result
            return self._parse_crew_result(result)

        except Exception as e:
            logger.exception(f"Error processing with crew: {e}")
            return {"error": str(e), "action": "skip_turn"}

    def _process_fallback(self, game_state):
        """
        Fallback processing when crew.ai is not available.

        Args:
            game_state (dict): Current game state information

        Returns:
            dict: Simple recommended action
        """
        logger.info("Using fallback processing (single-agent mode)")

        # Extract basic information from game state
        phase = game_state.get("structured_data", {}).get("phase", "")
        raw_description = game_state.get("raw_description", "")

        # Generate a thought for the fallback agent
        thought = f"Analyzing game state in {phase} phase. {raw_description[:200]}..."
        self.agent_thoughts["FallbackAgent"] = thought

        # Simple rule-based decision making
        if "production" in phase.lower():
            action = {"action": "build_unit", "unit_type": "warrior"}
            self.agent_thoughts[
                "FallbackAgent"
            ] += "\nI recommend building a warrior unit for early defense and exploration."
        elif "research" in phase.lower():
            action = {"action": "research_technology", "technology": "pottery"}
            self.agent_thoughts[
                "FallbackAgent"
            ] += "\nI recommend researching pottery to enable granaries and improve food production."
        elif "movement" in phase.lower():
            action = {"action": "move_unit", "direction": "explore"}
            self.agent_thoughts[
                "FallbackAgent"
            ] += "\nI recommend exploring the map to discover resources and other civilizations."
        else:
            action = {"action": "skip_turn"}
            self.agent_thoughts[
                "FallbackAgent"
            ] += "\nI recommend skipping the turn as there are no critical actions needed."

        return action

    def _format_game_state(self, game_state):
        """
        Format game state as a string for the crew.

        Args:
            game_state (dict): Game state information

        Returns:
            str: Formatted game state
        """
        if isinstance(game_state, str):
            return game_state

        formatted = "Current Game State:\n"

        # Add raw description if available
        if "raw_description" in game_state:
            formatted += game_state["raw_description"]
            return formatted

        # Otherwise, format structured data
        structured = game_state.get("structured_data", {})
        for key, value in structured.items():
            formatted += f"{key.capitalize()}: {value}\n"

        return formatted

    def _parse_crew_result(self, result):
        """
        Parse the result from the crew into a structured action.

        Args:
            result (str): Result from the crew

        Returns:
            dict: Structured action
        """
        # Try to parse as JSON
        try:
            import json

            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass

        # Simple parsing based on keywords
        action = {}

        if "build" in result.lower() or "produce" in result.lower():
            action["action"] = "build"
            if "unit" in result.lower():
                action["type"] = "unit"
                # Try to extract unit type
                unit_types = [
                    "warrior",
                    "settler",
                    "builder",
                    "slinger",
                    "archer"]
                for unit in unit_types:
                    if unit in result.lower():
                        action["unit_type"] = unit
                        break
            elif "building" in result.lower():
                action["type"] = "building"
                # Try to extract building type
                building_types = ["monument", "granary", "walls", "library"]
                for building in building_types:
                    if building in result.lower():
                        action["building_type"] = building
                        break

        elif "research" in result.lower():
            action["action"] = "research"
            # Try to extract technology
            tech_types = [
                "pottery",
                "animal husbandry",
                "mining",
                "sailing",
                "astrology",
            ]
            for tech in tech_types:
                if tech in result.lower():
                    action["technology"] = tech
                    break

        elif "move" in result.lower():
            action["action"] = "move"
            # Try to extract direction
            directions = ["north", "south", "east", "west"]
            for direction in directions:
                if direction in result.lower():
                    action["direction"] = direction
                    break

        elif "skip" in result.lower() or "end turn" in result.lower():
            action["action"] = "skip_turn"

        else:
            # Default action if we can't parse
            action["action"] = "skip_turn"
            action["raw_result"] = result

        return action
