"""
Image analyzer for game screenshots using vision models.

This module provides functionality to analyze screenshots from various games
using multimodal vision models. It supports different model backends and
can be configured for different games.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from src.utils.image_utils import encode_image_to_base64
from src.utils import model_call
from src.vision.frame_extractor import FrameExtractor

# Pydantic models for structured response parsing
class Position(BaseModel):
    x: int
    y: int

class UIElement(BaseModel):
    name: str
    position: List[int] = Field(..., description="[x, y] coordinates")
    state: str
    function: str

class GameArea(BaseModel):
    board: List[int] = Field(..., description="[x, y, width, height]")
    player_area: Optional[List[int]] = Field(None, description="[x, y, width, height]")
    controls: Optional[List[int]] = Field(None, description="[x, y, width, height]")
    # Allow additional areas to be dynamically added
    additional_areas: Optional[Dict[str, List[int]]] = Field(None, description="Additional game areas")

class PlayerStatus(BaseModel):
    position: Optional[Union[List[int], str]] = None
    cash: Optional[int] = None
    properties: Optional[List[str]] = None
    # Allow additional fields
    additional_info: Optional[Dict[str, Any]] = None

class Action(BaseModel):
    type: str  # e.g., "click", "drag", "press"
    position: Optional[List[int]] = None  # [x, y] coordinates if applicable
    key: Optional[str] = None  # Key to press if applicable
    parameters: Optional[Dict[str, Any]] = None  # Additional parameters

class ActionAnalysis(BaseModel):
    simple: bool
    explanation: str
    action: Optional[Action] = None

class GameState(BaseModel):
    phase: Optional[str] = None
    current_turn: Optional[str] = None
    player_status: Optional[PlayerStatus] = None
    board_state: Optional[List[Dict[str, Any]]] = None
    resources: Optional[Dict[str, Any]] = None
    other_players: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Dict[str, Any]]] = None

class GameAnalysis(BaseModel):
    ui_elements: List[UIElement]
    game_areas: Optional[GameArea] = None
    game_state: GameState
    action_analysis: ActionAnalysis
    raw_description: Optional[str] = None
    monologue: Optional[str] = None

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Class for analyzing game screenshots using vision models.
    Supports multiple model backends including Ollama and Qwen V3.
    Can be configured for different games.
    """

    def __init__(
            self,
            model="qwen",
            config=None,
            game_config=None,
            screenshot_capture=None):
        """
        Initialize the ImageAnalyzer.

        Args:
            model (str): Model backend to use ('ollama', 'qwen')
            config (dict): Configuration for the analyzer
        """
        self.model_type = model.lower()
        self.config = config or {}
        self.game_config = game_config or {}
        self.screenshot_capture = screenshot_capture

        # Initialize frame extractor if screenshot capture is provided
        self.frame_extractor = None
        if self.screenshot_capture:
            frame_config = self.config.get("frame_extraction", {})
            self.frame_extractor = FrameExtractor(
                self.screenshot_capture, config=frame_config
            )

        # Get current game configuration
        self.current_game = game_config.get("name", "vanity_fair")  # Use the game_name passed from main.py
        logger.info(f"Initializing image analyzer for game: {self.current_game}")
        
        # Get the games dictionary
        self.games = config.get("games", {})
        
        # Log available games for debugging
        logger.info(f"Available games in config: {list(self.games.keys())}")
        
        # Load game-specific configuration
        self.game_settings = self.games.get(self.current_game, {}) or self.games.get("vanity_fair", {})
        
        # Explicitly log the game settings to help debug
        logger.info(f"Loaded game settings for: {self.current_game}")
        
        self.game_name = self.game_settings.get("name", "Vanity Fair: The Pursuit")
        self.game_prompts = self.game_settings.get("prompts", {})
        
        # Log available prompts for this game
        logger.info(f"Available prompts for {self.game_name}: {list(self.game_prompts.keys())}")

        # No need to initialize model_call as we're using the module directly
        
        # Store model type and name for logging
        self.model_name = None
        logger.info(f"Using {self.model_type} model")

    def _prepare_image(self, image):
        """
        Prepare image for analysis by processing it if needed.

        Args:
            image (PIL.Image or str): Image to process or path to image

        Returns:
            PIL.Image: Processed image ready for analysis
        """
        # Handle string paths to images
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                logger.error(f"Error opening image from path: {e}")
                return None
                
        # Process the image for analysis if not already optimized
        # if isinstance(image, Image.Image) and not image.info.get('optimized', False):
        #     logger.debug("Processing image for analysis")
        #     # Use image_utils.py for all image processing
        #     image = optimize_image(image)
        #     image = process_screenshot(image)
            
        return image

    def analyze(
        self,
        image,
        prompt=None,
        analysis_type="game_state",
        check_animation=False,
        region_name="full_screen",
    ):
        """
        Analyze a game screenshot.

        Args:
            image (PIL.Image or str): Screenshot to analyze or path to screenshot
            prompt (str, optional): Custom prompt to use for analysis
            analysis_type (str, optional): Type of analysis to perform,
                corresponds to prompt keys in config

        Returns:
            dict: Analysis results including game state information
        """
        # Check for animation if requested and frame extractor is available
        if check_animation and self.frame_extractor and self.screenshot_capture:
            # First check if the screen is animating
            if (
                self.screenshot_capture.previous_frames and
                len(self.screenshot_capture.frame_diffs) >= 2
            ):
                avg_diff = sum(self.screenshot_capture.frame_diffs) / len(
                    self.screenshot_capture.frame_diffs
                )
                if avg_diff > self.screenshot_capture.change_threshold:
                    logger.info(
                        "Animation detected with average difference: %.4f", avg_diff
                    )

                    # Process the animation
                    animation_prompt = self.game_prompts.get("animation", None)
                    animation_result = self.frame_extractor.process_animation(
                        region_name, prompt=animation_prompt
                    )

                    if animation_result.get("status") == "success":
                        logger.info("Successfully analyzed animation sequence")
                        return {
                            "type": "animation",
                            "analysis": animation_result.get("analysis"),
                            "key_frame_count": animation_result.get(
                                "key_frame_count", 0
                            ),
                            "total_frame_count": animation_result.get(
                                "total_frame_count", 0
                            ),
                        }

        try:
            # Prepare image for analysis
            processed_image = self._prepare_image(image)
            
            # Encode the image to base64 without resizing
            base64_image = encode_image_to_base64(processed_image)
            
            # If no prompt is provided, use the default game state prompt
            if not prompt:
                prompt = self.game_prompts.get(analysis_type, "Describe what you see in this game screenshot.")
                if prompt == "Describe what you see in this game screenshot.":
                    logger.warning(f"⚠️ USING FALLBACK PROMPT - No {analysis_type} prompt found in config for game: {self.current_game}")
                    logger.warning(f"Available prompts in config: {list(self.game_prompts.keys())}")
                else:
                    logger.info(f"Using game-specific prompt for {analysis_type} from config")
            else:
                logger.info(f"Using custom prompt for {analysis_type} (passed directly to analyze method)")
            
            # Log the full prompt
            logger.info(f"===== PROMPT FOR {analysis_type.upper()} =====\n{prompt}\n===== END PROMPT =====")
            
            # Validate that the prompt isn't empty or too basic
            if len(prompt) < 50:
                logger.warning(f"Prompt seems too short (only {len(prompt)} chars). This may not provide enough context for good analysis.")
            
            # Log the first 100 characters of the prompt for verification
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            logger.info(f"Prompt preview: {prompt_preview}")
            
            # Prepare and call the vision model
            try:
                # Ensure prompt includes the game name for better context
                if self.game_name and self.game_name not in prompt:
                    prompt = f"Analyze this {self.game_name} screenshot: {prompt}"
                    logger.info(f"Added game name to prompt: {self.game_name}")
                
                # Call vision model with prepared image and prompt
                response = model_call.call_vision_model(base64_image, prompt)
            except Exception as e:
                logger.error(f"Error calling vision model: {str(e)}")
                return {"status": "error", "message": str(e)}
            
            if response.get("status") == "success":
                # Parse the model's response
                return self._parse_model_response(response.get("response_text", ""))
            else:
                logger.error(f"Error in model call: {response.get('error')}")
                return {"error": response.get("error", "Unknown error in model call")}

        except Exception as e:
            logger.exception(f"Error analyzing image: {e}")
            return {"error": str(e)}

    # The _analyze_with_ollama and _analyze_with_qwen methods have been replaced by the centralized model_call.py

    def _parse_model_response(self, response_text):
        """
        Parse the model's response text into a structured format using Pydantic models.

        Args:
            response_text (str): Raw text response from the model, expected to be in JSON format

        Returns:
            dict: Structured game state information with UI elements, game state, and action analysis
        """
        # Clean up response text to ensure it's valid JSON
        # Sometimes the LLM might add markdown code blocks or extra text
        cleaned_text = self._clean_json_response(response_text)
        
        # Try to parse as JSON
        try:
            logger.info("Attempting to parse response as JSON")
            json_data = json.loads(cleaned_text)
            
            try:
                # Try to validate with our Pydantic model
                logger.info("Validating JSON with GameAnalysis model")
                game_analysis = GameAnalysis(**json_data)
                logger.info(f"Successfully parsed response as GameAnalysis: {len(json_data)} fields found")
                
                # Add the raw response for debugging
                result = game_analysis.model_dump()
                result["raw_response"] = response_text
                
                # Log whether this is a simple action
                if game_analysis.action_analysis.simple:
                    logger.info(f"Simple action detected: {game_analysis.action_analysis.action.type} at position {game_analysis.action_analysis.action.position if game_analysis.action_analysis.action.position else 'N/A'}")
                else:
                    logger.info(f"Complex action: {game_analysis.action_analysis.explanation}")
                
                return result
                
            except Exception as e:
                logger.warning(f"Error validating JSON with Pydantic model: {str(e)}")
                # If validation fails, return the raw JSON
                return {
                    "raw_response": response_text,
                    "parsed_json": json_data,
                    "validation_error": str(e)
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"Response is not valid JSON: {str(e)}")
            # Fall back to basic text parsing
            return self._parse_text_response(response_text)
    
    def _clean_json_response(self, text):
        """Clean up the response text to ensure it's valid JSON."""
        # Remove markdown code blocks if present
        if "```json" in text or "```" in text:
            logger.info("Removing markdown code blocks from response")
            lines = text.split('\n')
            cleaned_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip() in ['```json', '```']:
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not in_code_block and line.strip():
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
        
        # Find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx+1]
        
        return text
    
    def _parse_text_response(self, response_text):
        """Fallback method to parse plain text responses."""
        import re
        logger.warning("Falling back to text-based parsing since response is not JSON")
        
        # Create a structure similar to what we expect from JSON but built from text parsing
        result = {
            "raw_description": response_text,
            "ui_elements": [],
            "game_state": {
                "phase": None,
                "current_turn": None,
                "player_status": {},
                "board_state": [],
                "resources": {},
                "events": []
            },
            "action_analysis": {
                "simple": False,  # Default to complex since we can't reliably determine
                "explanation": "Unable to determine if this is a simple action",
                "action": None
            }
        }
        
        # Extract key information using simple heuristics
        sections = {
            "ui_elements": ["button", "menu", "icon", "clickable", "element"],
            "game_state": ["phase", "turn", "player", "board", "resources"],
            "action_analysis": ["action", "simple", "complex", "click", "press"],
            "game_areas": ["board", "player area", "controls"]
        }
        
        # Look for specific patterns in the text
        lines = response_text.split("\n")
        
        # Try to identify if the response mentions a simple action
        for line in lines:
            if ("simple" in line.lower() and 
                    ("true" in line.lower() or "yes" in line.lower())):
                result["action_analysis"]["simple"] = True
                result["action_analysis"]["explanation"] = line.strip()
            
            # Try to extract coordinates for clicking
            if "click" in line.lower() and any(c.isdigit() for c in line):
                # Look for x=123, y=456 pattern or similar
                coords = re.findall(r'\d+', line)
                if len(coords) >= 2:
                    result["action_analysis"]["action"] = {
                        "type": "click",
                        "position": [int(coords[0]), int(coords[1])]
                    }
                    result["action_analysis"]["simple"] = True
        
        # Try to extract UI elements
        for line in lines:
            element_keywords = ["button", "icon", "menu", "element"]
            if (any(kw in line.lower() for kw in element_keywords) and 
                    any(c.isdigit() for c in line)):
                # Try to extract a name and position
                element_name = None
                position = None
                
                # Look for a name pattern like "Button: X" or "X button" 
                name_pattern = (r'["\']([^"\'\.]+)["\']|'
                               r'Button:\s*([^\,\.]+)|'
                               r'([\w\s]+)\s+button')
                name_match = re.search(name_pattern, line, re.IGNORECASE)
                if name_match:
                    groups = [g for g in name_match.groups() if g is not None]
                    if groups:
                        element_name = groups[0].strip()
                
                # Look for coordinates
                coords = re.findall(r'\d+', line)
                if len(coords) >= 2:
                    position = [int(coords[0]), int(coords[1])]
                
                if element_name and position:
                    result["ui_elements"].append({
                        "name": element_name,
                        "position": position,
                        "state": "enabled",  # Default assumption
                        "function": "Unknown"  # Default
                    })
        
        # If we found no UI elements but have action coordinates, create a default
        action = result["action_analysis"].get("action")
        if (not result["ui_elements"] and action and 
                action.get("position")):
            position = action["position"]
            result["ui_elements"].append({
                "name": "Detected Element",
                "position": position,
                "state": "enabled",
                "function": "Target for recommended action"
            })
        
        # Log what we found
        ui_count = len(result["ui_elements"])
        logger.info(f"Extracted {ui_count} UI elements from text response")
        if result["action_analysis"]["simple"]:
            logger.info("Detected a simple action recommendation from text")
            
        return result
