#!/usr/bin/env python
"""
Game analyzer module for processing game state and generating responses.
Handles monologue generation and TTS integration.
"""

import logging
from typing import Any, Dict, Optional

from src.tts.tts_manager import TTSManager
from src.utils.log_utils import log_monologue
from src.game_player.game_state import GameStateObject, dict_to_game_state_object

logger = logging.getLogger(__name__)


class GameAnalyzer:
    """
    Analyzes game state and processes monologues.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the game analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize TTS if enabled in config
        self.tts_manager = None
        if config.get("tts", {}).get("enabled", False):
            logger.info("Initializing text-to-speech for GameAnalyzer")
            self.tts_manager = TTSManager(config.get("tts", {}))
        else:
            logger.info("TTS disabled in config")

    # Removed _convert_to_object method as we now use the dedicated GameStateObject class
    
    def process_game_state(self, game_state: Dict[str, Any], turn_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Process the game state and return any updates.
        
        Args:
            game_state: Game state dictionary from the vision model
            turn_number: Optional turn number for logging
            
        Returns:
            Updated game state dictionary with added GameStateObject
        """
        logger.debug(f"Processing game state for turn {turn_number}")
        
        # Convert game state to a proper GameStateObject for easier access
        game_state_obj = dict_to_game_state_object(game_state)
        logger.debug(f"Converted game state to GameStateObject with attributes: {dir(game_state_obj)}")
        
        # Process monologue if available
        if game_state_obj.monologue:
            self._process_monologue(game_state, turn_number)
        
        # Add the object as a property of the dictionary for easier access
        game_state['_obj'] = game_state_obj
        
        # Additional game state processing can be added here
        
        return game_state

    def _process_monologue(self, game_state: Dict[str, Any], turn_number: Optional[int] = None) -> None:
        """
        Process monologue text from game state, play TTS audio, and log it.
        
        Args:
            game_state: Game state containing monologue
            turn_number: Current turn number
        """
        if "monologue" in game_state and game_state["monologue"]:
            monologue = game_state["monologue"]
            logger.info(f"Processing monologue: {monologue}")
            
            # Log monologue using centralized logging function
            raw_response = game_state.get("raw_response")
            log_monologue(monologue, raw_response=raw_response, turn=turn_number)
            logger.info("Monologue recorded in session log")
            
            # Play monologue audio if TTS is available
            if self.tts_manager and self.tts_manager.is_available():
                logger.info("Playing monologue audio")
                self.tts_manager.speak_monologue(monologue)
            else:
                logger.info("TTS manager not available, skipping audio playback")
