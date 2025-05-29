"""
Module for detecting the location and identity of a game window on the screen using a vision model.
"""

import json
import logging
import os
import re
from typing import Optional, Tuple, Dict
import io
from PIL import Image

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.utils.image_utils import process_screenshot, encode_image_to_base64, scale_coordinates
from src.utils import model_call
from src.utils.config_loader import get_api_config, load_api_config, load_config, PROJECT_ROOT

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
        # Default prompt templates (minimal fallback in case config is missing)
        default_prompts = {
            "window_detection": "Locate any game window and return JSON with x, y, width, height, and game_name."
        }
        
        # Try to load from config
        api_config = load_api_config() if self.config is None else self.config
        
        # Load the main config file
        config_path = os.path.join(PROJECT_ROOT, "config", "default.yaml")
        main_config = load_config(config_path)
        
        # Get prompts from the vision section of the config
        vision_prompts = main_config.get("vision", {}).get("prompts", {})
        window_detection_prompt = vision_prompts.get("window_detection", "")
        
        # If we found a prompt in the config, use it
        if window_detection_prompt:
            logger.info("Loaded window detection prompt from config file")
            default_prompts["window_detection"] = window_detection_prompt
        else:
            logger.warning("No window detection prompt found in config, using fallback")
        
        return default_prompts
    
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
        # Add game name to prompt context if provided
        if game_name:
            prompt_context = f"The game is {game_name}. " + prompt_context
            
        return self._detect_game_window_impl(screenshot, prompt_context)
    
    def _detect_game_window_impl(
        self, screenshot, prompt_context: str = ""
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
        """
        Detect the location and identity of the game window on the screen.

        Args:
            screenshot: PIL Image of the screenshot
            prompt_context (str): Optional extra context for the vision model prompt
            wait_seconds (float): Time to wait before processing the screenshot

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

        # No waiting here - the main.py file already handles the waiting
        
        # Save the original screenshot for inspection
        from datetime import datetime
        from pathlib import Path
        
        # Create screenshots directory if it doesn't exist
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        # Save the original screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = screenshots_dir / f"original_{timestamp}.png"
        screenshot.save(original_path)
        logger.info(f"Original screenshot saved to: {original_path}")
        
        # Process and optimize the screenshot
        processed_image = process_screenshot(screenshot)
        
        # Save the processed image before resizing
        processed_path = screenshots_dir / f"processed_{timestamp}.png"
        processed_image.save(processed_path)
        logger.info(f"Processed screenshot saved to: {processed_path}")
        
        # Get original image dimensions before resizing
        original_size = processed_image.size
        logger.info(f"Original image dimensions: {original_size[0]}x{original_size[1]}")
        
        # Encode the image to base64 (this will resize it)
        max_width = 1024  # Store this for scaling coordinates later
        base64_image = encode_image_to_base64(processed_image, max_width=max_width, optimize=True)
        
        # Save the resized/optimized image used for the model
        # First, decode the base64 image back to a PIL Image
        import base64
        import io
        
        try:
            # Decode base64 to image
            image_data = base64.b64decode(base64_image)
            optimized_image = Image.open(io.BytesIO(image_data))
            
            # Save the optimized image
            optimized_path = screenshots_dir / f"optimized_{timestamp}.png"
            optimized_image.save(optimized_path)
            logger.info(f"Optimized screenshot saved to: {optimized_path}")
        except Exception as e:
            logger.error(f"Failed to save optimized image: {e}")

        
        # Store the resized dimensions for later coordinate scaling
        # For simplicity, we'll estimate the resized dimensions based on max_width
        if original_size[0] > max_width:
            scale_ratio = max_width / original_size[0]
            resized_width = max_width
            resized_height = int(original_size[1] * scale_ratio)
            resized_size = (resized_width, resized_height)
        else:
            # No resizing occurred
            resized_size = original_size

        # Send to model using the centralized model_call module
        model_result = model_call.call_vision_model(base64_image, full_prompt)
        
        # Parse model response directly to Pydantic model
        window_region = None
        game_name = None
        
        if model_result.get("status") != "success":
            logger.error(f"Error in model call: {model_result.get('error')}")
            return None, None
            
        # Get the response text
        response_text = model_result.get("response_text", "")
        
        # Parse JSON and validate with Pydantic in one step
        try:
            # Add more detailed logging for debugging
            logger.info(f"Attempting to parse JSON: {response_text}")
            
            # Try to clean the response text if it's not valid JSON
            try:
                # First attempt direct parsing
                window_loc = WindowLocation.model_validate_json(response_text)
            except Exception as parse_error:
                logger.warning(f"Initial JSON parsing failed: {parse_error}")
                return None, None
            
            # Get the window region as a tuple from the model (based on resized image)
            model_window_region = window_loc.to_tuple()
            game_name = window_loc.game_name
            
            # Log detailed information about the detected window (from model)
            logger.info(f"Model detected window coordinates: x={window_loc.x}, y={window_loc.y}, width={window_loc.width}, height={window_loc.height}")
            
            # Scale the coordinates back to the original image size if needed
            if original_size != resized_size:
                # Log detailed information before scaling
                logger.info(f"Original window coordinates: {model_window_region}")
                logger.info(f"Original image size: {original_size}")
                logger.info(f"Resized image size: {resized_size}")
                
                # Use the scale_coordinates function from image_utils
                window_region = scale_coordinates(
                    coords=model_window_region,
                    original_size=original_size,
                    resized_size=resized_size
                )
                
                # Log the scaled coordinates
                logger.info(f"Scaled window coordinates: {window_region}")
            else:
                # No scaling needed
                window_region = model_window_region
                logger.info("No scaling needed for window coordinates")
                
            logger.info(f"Detected game: {game_name}")
            
            # Verify the coordinates make sense
            if window_loc.x < 0 or window_loc.y < 0:
                logger.warning(f"Detected negative coordinates: ({window_loc.x}, {window_loc.y}). This may cause issues.")
                
            if window_loc.width <= 0 or window_loc.height <= 0:
                logger.warning(f"Invalid window dimensions: width={window_loc.width}, height={window_loc.height}")
                return None, None
                
            # Check if the window size is reasonable
            if window_loc.width < 100 or window_loc.height < 100:
                logger.warning(f"Window dimensions seem too small: {window_loc.width}x{window_loc.height}")
            
            return window_region, game_name
        except Exception as e:
            logger.error(f"Failed to parse window location: {e}")
            return None, None
        return window_region, game_name

