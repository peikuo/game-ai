#!/usr/bin/env python
"""
Game analyzer module for analyzing game state and processing monologues.
"""

import logging
from typing import Dict, Any, Optional

from src.utils.log_utils import log_monologue
from src.tts.tts_manager import TTSManager

# Set up logger
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

    def _convert_to_object(self, data: Dict[str, Any]) -> Any:
        """
        Convert a dictionary to an object with attributes for easier access.
        Recursively converts nested dictionaries and lists.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            Object with attributes corresponding to dictionary keys
        """
        if isinstance(data, dict):
            # Create a new object
            class GameStateObject:
                def __repr__(self):
                    attrs = [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
                    return f"GameStateObject({', '.join(attrs)})"
            
            obj = GameStateObject()
            
            # Add each key-value pair as an attribute
            for key, value in data.items():
                setattr(obj, key, self._convert_to_object(value))
            
            return obj
        elif isinstance(data, list):
            # Convert each item in the list
            return [self._convert_to_object(item) for item in data]
        else:
            # Return primitive values as is
            return data
    
    def process_game_state(self, game_state: Dict[str, Any], turn_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Process the game state and return any updates.
        
        Args:
            game_state: Game state dictionary from the vision model
            turn_number: Optional turn number for logging
            
        Returns:
            Updated game state dictionary
        """
        logger.debug(f"Processing game state for turn {turn_number}")
        
        # Convert game state to object for easier access
        game_state_obj = self._convert_to_object(game_state)
        logger.debug(f"Converted game state to object with attributes: {dir(game_state_obj)}")
        
        # Process monologue if available
        if hasattr(game_state_obj, 'monologue') and game_state_obj.monologue:
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
