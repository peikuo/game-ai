#!/usr/bin/env python
"""
Game analyzer module for analyzing game state and processing monologues.
"""

import logging
from typing import Any, Dict, Optional, Union

from src.game_player.game_state import GameStateObject, dict_to_game_state_object
from src.utils.log_utils import log_monologue

# Set up logger
logger = logging.getLogger(__name__)


class GameAnalyzer:
    """
    Analyzes game state and processes monologues.
    """

    def __init__(self, config: Dict[str, Any], tts_manager=None):
        """
        Initialize the game analyzer.

        Args:
            config: Configuration dictionary
            tts_manager: Optional TTS manager instance
        """
        self.config = config
        self.tts_manager = tts_manager

    def analyze(
        self, game_state: Dict[str, Any], turn_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the game state and determine the next action.

        Args:
            game_state: Game state dictionary from the vision model

        Returns:
            Action dictionary to be executed by the game controller
        """
        logger.info("Analyzing game state to determine next action")

        # Convert dictionary to GameStateObject for better type safety
        game_state_obj = dict_to_game_state_object(game_state)

        if game_state_obj.monologue:
            self._process_monologue(game_state_obj, turn_number)

        # For now, just return a simple action based on the game state
        # determined by the vision model
        if game_state_obj.action_analysis and game_state_obj.action_analysis.simple:
            if game_state_obj.action_analysis.action:
                logger.info(
                    f"Using simple action from vision model: "
                    f"{game_state_obj.action_analysis.action}"
                )
                # Convert Action object to dictionary for GameController
                return game_state_obj.action_analysis.action.to_dict()

        # If no simple action is available, we need more complex decision making
        # This could be expanded with more sophisticated game logic
        logger.info("No simple action available, using default action")

        return None

    def _process_monologue(
        self,
        game_state: Union[Dict[str, Any], GameStateObject],
        turn_number: Optional[int] = None,
    ) -> None:
        """
        Process monologue text from game state, play TTS audio, and log it.

        Args:
            game_state: Game state containing monologue (either dict or GameStateObject)
            turn_number: Current turn number
        """
        # Handle both dictionary and GameStateObject inputs
        if isinstance(game_state, dict):
            monologue = game_state.get("monologue")
            raw_response = game_state.get("raw_response")
        else:
            monologue = game_state.monologue
            raw_response = game_state.raw_response

        if monologue:
            logger.info(f"Processing monologue: {monologue}")

            # Log monologue using centralized logging function
            log_monologue(
                monologue,
                raw_response=raw_response,
                turn=turn_number)
            logger.info("Monologue recorded in session log")

            # If the game state has a monologue, process it (will handle TTS if
            # available)
            if self.tts_manager and self.tts_manager.is_available():
                logger.info("Playing monologue audio")
                self.tts_manager.speak_monologue(monologue)
            else:
                logger.info(
                    "TTS manager not available, skipping audio playback")
