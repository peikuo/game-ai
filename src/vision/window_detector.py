"""
Module for detecting the location and identity of a game window on the screen using a vision model.
"""

import json
import logging
import re
from typing import Optional, Tuple, Dict

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.utils.image_utils import process_screenshot, encode_image_to_base64
from src.utils import model_call
from src.utils.config_loader import get_api_config

logger = logging.getLogger(__name__)


class WindowLocation(BaseModel):
    """Pydantic model for game window location"""

    x: int = Field(..., description="X-coordinate of the top-left corner")
    y: int = Field(..., description="Y-coordinate of the top-left corner")
    width: int = Field(..., description="Width of the window")
    height: int = Field(..., description="Height of the window")
    game_name: Optional[str] = Field(
        None, description="Name of the detected game")

    @model_validator(mode="after")
    def validate_dimensions(self) -> "WindowLocation":
        """Validate that dimensions are positive"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Width and height must be positive (got width={self.width}, height={self.height})"
            )
        return self

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple"""
        return (self.x, self.y, self.width, self.height)


class WindowDetector:
    """Class for detecting game windows using vision models"""

    def __init__(self, model="qwen", config=None):
        """
        Initialize the window detector
        
        Args:
            model (str): Model type to use (qwen, ollama)
            config (dict): Configuration for the vision model
        """
        self.model_type = model
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load prompt templates from config
        self.prompts = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates from config
        
        Returns:
            Dict of prompt templates
        """
        # Default prompt templates
        default_prompts = {
            "window_detection": (
                "Locate any game window on this screen. "
                "Analyze the image and identify the coordinates of the game window. "
                "Return your answer in the following JSON format:\n"
                "{\n"
                '  "x": [left coordinate in pixels],\n'
                '  "y": [top coordinate in pixels],\n'
                '  "width": [width in pixels],\n'
                '  "height": [height in pixels],\n'
                '  "game_name": [name of the game if visible]\n'
                "}\n\n"
                'If no game window is found, reply with: {"not_found": true}\n'
                "IMPORTANT: Ensure your response includes this exact JSON format."
            )
        }
        
        # Try to load from config
        api_config = get_api_config()
        config_prompts = api_config.get("prompts", {}).get("window_detection", {})
        
        # Merge with defaults (config takes precedence)
        return {**default_prompts, **config_prompts}
    
    def detect_game_window(self, screenshot, game_name="", prompt_context=""):
        """
        Detect the game window in a screenshot
        
        Args:
            screenshot: PIL Image of the screenshot
            game_name: Name of the game to look for
            prompt_context: Additional context for the prompt
            
        Returns:
            Tuple of (window_region, detected_game_name)
        """
        if game_name:
            # Add game name to prompt context if provided
            additional_context = f"Look for a {game_name} game window. "
            prompt_context = additional_context + prompt_context
            
        # Process the screenshot directly without capturing again
        return self._detect_game_window_impl(screenshot, prompt_context)
    
    def _detect_game_window_impl(
        self, screenshot, prompt_context: str = ""
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
        """
        Detect the location and identity of the game window on the screen.

        Args:
            screenshot: PIL Image of the screenshot
            prompt_context (str): Optional extra context for the vision model prompt

        Returns:
            (window_region, game_name):
                window_region: (x, y, width, height) tuple if found, else None
                game_name: Detected game name if available, else None
        """
        # Use the provided screenshot directly
        if screenshot is None:
            logger.error("No screenshot provided for window detection.")
            return None, None

        # Get the window detection prompt from config
        window_prompt = self.prompts.get("window_detection")

        # Combine with context if provided
        full_prompt = (
            prompt_context +
            "\n" +
            window_prompt if prompt_context else window_prompt)

        # Process and optimize the screenshot
        processed_image = process_screenshot(screenshot)
        
        # Encode the image to base64
        base64_image = encode_image_to_base64(processed_image, max_width=1024, optimize=True)

        # Send to model using the centralized model_call module
        model_result = model_call.call_vision_model(base64_image, full_prompt)
        
        # Get the response text
        response_text = ""
        if model_result.get("status") == "success":
            response_text = model_result.get("response_text", "")
        else:
            logger.error(f"Error in model call: {model_result.get('error')}")
            return None, None
            
        # Create a result dict similar to what image_analyzer would return
        result = {"raw_description": response_text}

        # Parse model response for [x, y, width, height] and game name
        window_region = None
        game_name = None

        try:
            # Extract raw text from result
            raw_text = result.get("raw_description", "")
            
            # Find JSON blocks in the response
            json_matches = [
                s.strip()
                for s in raw_text.split("\n")
                if s.strip().startswith("{") and s.strip().endswith("}")
            ]
            
            # Try each JSON block with Pydantic
            for json_str in json_matches:
                try:
                    # Parse JSON and validate with Pydantic in one step
                    window_loc = WindowLocation.model_validate_json(json_str)
                    window_region = window_loc.to_tuple()
                    game_name = window_loc.game_name
                    logger.info(f"Detected window: {window_region}, game: {game_name}")
                    break
                except (json.JSONDecodeError, ValidationError):
                    # Continue to next JSON block if this one fails
                    continue

            # If we didn't find valid JSON, try to extract coordinates from text
            if not window_region:
                # Look for patterns like [x, y, width, height] or coordinates:
                # x, y, width, height
                coord_patterns = [
                    # [x, y, width, height]
                    r"\[(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)\]",
                    # coordinates: x, y, width, height
                    r"coordinates:?\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)",
                    # window: x, y, width, height
                    r"window:?\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)",
                ]

                # Try each pattern
                for pattern in coord_patterns:
                    match = re.search(pattern, raw_text, re.IGNORECASE)
                    if match:
                        try:
                            x, y, width, height = map(int, match.groups())
                            window_loc = WindowLocation(
                                x=x, y=y, width=width, height=height
                            )
                            window_region = window_loc.to_tuple()
                            break
                        except (ValueError, ValidationError):
                            continue

                # Try to extract game name if we found coordinates
                if window_region and not game_name:
                    game_name_patterns = [
                        r"game\s*name:?\s*([\w\s\-:]+)",
                        r"detected\s*game:?\s*([\w\s\-:]+)",
                        r"game\s*is:?\s*([\w\s\-:]+)",
                    ]

                    for pattern in game_name_patterns:
                        match = re.search(pattern, raw_text, re.IGNORECASE)
                        if match:
                            game_name = match.group(1).strip()
                            break
        except Exception as e:
            logger.error("Error parsing window location: %s", e)

        if not window_region:
            logger.warning(
                "Could not extract window coordinates from model response. Using full screen."
            )
        if window_region:
            logger.info("Detected game window region: %s", window_region)
        if game_name:
            logger.info("Detected game name: %s", game_name)

        # Enhance prompt to get better structured output next time
        if not window_region:
            logger.info("Adding structured output hint to prompt for next detection attempt")

        return window_region, game_name
