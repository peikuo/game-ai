#!/usr/bin/env python
"""
Game controller module for interacting with Civilization VI.
Handles execution of game actions through input simulation.
"""

import logging
import time
from typing import Any, Dict

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("PyAutoGUI not available. Install with: pip install pyautogui")

logger = logging.getLogger(__name__)


class GameController:
    """
    Controller class for interacting with Civilization VI game.
    Handles mouse clicks, keyboard inputs, and other game interactions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the game controller with configuration.

        Args:
            config: Dictionary containing game configuration parameters
        """
        self.config = config
        self.game_name = config.get("current_game", "civ6")
        # Delay between actions in seconds
        self.action_delay = config.get("action_delay", 0.5)
        self.input_config = config.get("input", {})
        
        # Check if PyAutoGUI is available
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("PyAutoGUI not available. Game control will be simulated.")
        else:
            # Configure PyAutoGUI for safety
            pyautogui.PAUSE = 0.1  # Add small pause between PyAutoGUI commands
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            screen_width, screen_height = pyautogui.size()
            logger.info(f"PyAutoGUI detected screen size: {screen_width}x{screen_height}")
        
        logger.info("GameController initialized for %s", self.game_name)

    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a game action based on the provided action dictionary.

        Args:
            action: Dictionary containing action type and parameters

        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        action_type = action.get("type")
        if not action_type:
            logger.error("No action type specified")
            return False

        logger.info("Executing action: %s", action_type)

        # Execute different actions based on action type
        if action_type == "click":
            return self._execute_click(action)
        elif action_type == "key":
            return self._execute_key_press(action)
        elif action_type == "wait":
            return self._execute_wait(action)
        elif action_type == "drag":
            return self._execute_drag(action)
        elif action_type == "select":
            return self._execute_select(action)
        else:
            logger.warning("Unknown action type: %s", action_type)
            return False

    def _execute_click(self, action: Dict[str, Any]) -> bool:
        """
        Execute a mouse click action.

        Args:
            action: Dictionary with x, y coordinates and optional button parameter

        Returns:
            bool: True if successful
        """
        x = action.get("x")
        y = action.get("y")
        button = action.get("button", "left")

        if x is None or y is None:
            logger.error("Click action missing x or y coordinates")
            return False

        logger.debug("Clicking at (%s, %s) with %s button", x, y, button)
        
        if PYAUTOGUI_AVAILABLE:
            try:
                # Move to position first (more reliable)
                pyautogui.moveTo(x, y, duration=0.2)
                # Then click
                pyautogui.click(x=x, y=y, button=button)
                logger.info(f"PyAutoGUI clicked at ({x}, {y}) with {button} button")
            except Exception as e:
                logger.error(f"PyAutoGUI click failed: {e}")
                return False
        else:
            logger.info(f"Simulated click at ({x}, {y}) with {button} button")

        # Simulate action delay
        time.sleep(self.action_delay)
        return True

    def _execute_key_press(self, action: Dict[str, Any]) -> bool:
        """
        Execute a keyboard key press action.

        Args:
            action: Dictionary with key parameter

        Returns:
            bool: True if successful
        """
        key = action.get("key")
        if not key:
            logger.error("Key action missing key parameter")
            return False

        logger.debug("Pressing key: %s", key)
        
        if PYAUTOGUI_AVAILABLE:
            try:
                # Handle special key combinations (ctrl+key, shift+key, etc.)
                if '+' in key:
                    parts = key.split('+')
                    modifiers = parts[:-1]
                    key_to_press = parts[-1]
                    
                    # Handle hotkeys with modifiers
                    keys_to_press = [mod.lower().strip() for mod in modifiers] + [key_to_press.lower().strip()]
                    pyautogui.hotkey(*keys_to_press)
                    logger.info(f"PyAutoGUI pressed hotkey: {'+'.join(keys_to_press)}")
                else:
                    # Handle single key press
                    pyautogui.press(key)
                    logger.info(f"PyAutoGUI pressed key: {key}")
            except Exception as e:
                logger.error(f"PyAutoGUI key press failed: {e}")
                return False
        else:
            logger.info(f"Simulated key press: {key}")

        # Simulate action delay
        time.sleep(self.action_delay)
        return True

    def _execute_wait(self, action: Dict[str, Any]) -> bool:
        """
        Execute a wait action.

        Args:
            action: Dictionary with duration parameter in seconds

        Returns:
            bool: True if successful
        """
        duration = action.get("duration", 1.0)
        logger.debug("Waiting for %s seconds", duration)
        time.sleep(duration)
        return True

    def _execute_drag(self, action: Dict[str, Any]) -> bool:
        """
        Execute a mouse drag action.

        Args:
            action: Dictionary with start_x, start_y, end_x, end_y coordinates

        Returns:
            bool: True if successful
        """
        start_x = action.get("start_x")
        start_y = action.get("start_y")
        end_x = action.get("end_x")
        end_y = action.get("end_y")

        if None in (start_x, start_y, end_x, end_y):
            logger.error("Drag action missing required coordinates")
            return False

        logger.debug(
            "Dragging from (%s, %s) to (%s, %s)",
            start_x,
            start_y,
            end_x,
            end_y)
            
        if PYAUTOGUI_AVAILABLE:
            try:
                # Move to start position first
                pyautogui.moveTo(start_x, start_y, duration=0.2)
                # Then drag to end position
                pyautogui.dragTo(end_x, end_y, duration=0.5, button='left')
                logger.info(f"PyAutoGUI dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            except Exception as e:
                logger.error(f"PyAutoGUI drag failed: {e}")
                return False
        else:
            logger.info(f"Simulated drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")

        # Simulate action delay
        time.sleep(self.action_delay)
        return True

    def _execute_select(self, action: Dict[str, Any]) -> bool:
        """
        Execute a selection action.

        Args:
            action: Dictionary with selection parameters

        Returns:
            bool: True if successful
        """
        selection_type = action.get("selection_type")
        target = action.get("target")

        if not selection_type or not target:
            logger.error("Select action missing required parameters")
            return False

        logger.debug("Selecting %s with type %s", target, selection_type)
        # TODO: Implement actual selection logic

        # Simulate action delay
        time.sleep(self.action_delay)
        return True

    def cleanup(self) -> None:
        """
        Perform cleanup operations when shutting down the controller.
        """
        logger.info("Cleaning up game controller resources")
        # Release any resources, close connections, etc.
