"""
Image analyzer for game screenshots using vision models.

This module provides functionality to analyze screenshots from various games
using multimodal vision models. It supports different model backends and
can be configured for different games.
"""

import json
import logging

from PIL import Image
import requests
from openai import OpenAI

from src.utils.config_loader import get_api_config
from src.utils.image_utils import (encode_image_to_base64, process_screenshot)
from src.vision.frame_extractor import FrameExtractor

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
        self.current_game = self.game_config.get("current_game", "civ6")
        self.games = self.game_config.get("games", {})

        # Load game-specific configuration
        self.game_settings = self.games.get(self.current_game, {})
        self.game_name = self.game_settings.get("name", "Civilization VI")
        self.game_prompts = self.game_settings.get("prompts", {})

        # Set up model-specific configurations
        if self.model_type == "ollama":
            # Load API configuration from centralized config loader
            api_config = get_api_config()
            ollama_config = api_config["ollama"]
            
            # Get API URL from config or centralized API config
            self.api_url = self.config.get("ollama_api") or ollama_config["api_url"]
            self.model_name = self.config.get("ollama_model") or ollama_config["model"]
            logger.info("Using model: %s", self.model_name)

        elif self.model_type == "qwen":
            # Load API configuration from centralized config loader
            api_config = get_api_config()
            qwen_config = api_config["qwen"]
            
            # Get API key from config or centralized API config
            self.api_key = self.config.get("qwen_api_key") or qwen_config["api_key"]
            
            # Get model name and base URL
            self.model_name = self.config.get("qwen_model") or qwen_config["model"]
            self.base_url = self.config.get("qwen_base_url") or qwen_config["base_url"]
            
            if not self.api_key:
                logger.warning(
                    "Qwen API key not found. Set in config or .env file."
                )

            # Initialize OpenAI client for Qwen
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            logger.info(
                f"Using Qwen model: {self.model_name} via OpenAI-compatible API"
            )

        else:
            logger.warning(
                f"Unknown model type:{self.model_type}, falling back to Ollama"
            )
            self.model_type = "ollama"
            self.model_name = "llava:latest"
            self.api_url = "http://localhost:11434/api/generate"

    def _encode_image(self, image, max_width=1024):
        """
        Resize and encode image to base64.
        Resizing reduces API costs and improves response times.
        Checks if the image is already optimized to avoid double processing.

        Args:
            image (PIL.Image or str): Image to encode or path to image
            max_width (int): Maximum width for the resized image

        Returns:
            str: Base64 encoded image
        """
        # Process the image for analysis if not already optimized
        if isinstance(image, Image.Image) and not image.info.get('optimized', False):
            logger.debug("Processing image for analysis")
            image = process_screenshot(image)

        # Use the centralized image utility function
        # We still need to resize for the API even if the image is already optimized
        # But we don't need to re-optimize colors or compression
        return encode_image_to_base64(image, max_width=max_width, optimize=True, force_optimize=False)

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
                image (PIL.Image or str): Screenshot to analyze or
        path to screenshot
                prompt (str, optional): Custom prompt to use for analysis
                analysis_type (str, optional): Type of analysis to perform,
        corresponds to prompt keys in config

            Returns:
                dict: Analysis results including game state
                information
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
            if self.model_type == "ollama":
                return self._analyze_with_ollama(image, prompt)
            elif self.model_type == "qwen":
                return self._analyze_with_qwen(image, prompt)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {"error": f"Unsupported model type: {self.model_type}"}

        except Exception as e:
            logger.exception(f"Error analyzing image: {e}")
            return {"error": str(e)}

    def _analyze_with_ollama(self, image, prompt):
        """
        Analyze image using Ollama API.

        Args:
            image (PIL.Image or str): Screenshot to analyze
            prompt (str): Prompt for analysis

        Returns:
            dict: Analysis results
        """
        encoded_image = self._encode_image(image)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "images": [encoded_image],
        }

        logger.debug("Sending request to Ollama API: %s", self.api_url)
        response = requests.post(self.api_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            return self._parse_model_response(result.get("response", ""))
        else:
            logger.error(
                f"Ollama API error: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code}"}

    def _analyze_with_qwen(self, image, prompt=None):
        """
        Analyze image using Qwen V3 API via OpenAI-compatible interface.

        Args:
            image (PIL.Image or str): Screenshot to analyze
            prompt (str): Prompt for analysis

        Returns:
            dict: Analysis results
        """
        # Check if API key is available
        if not self.api_key:
            logger.error(
                "Qwen API key not found. Cannot proceed with analysis.")
            return {"error": "API key not found"}

        # Encode the image to base64
        encoded_image = self._encode_image(image)
        
        # Check if encoding was successful
        if encoded_image is None:
            logger.error("Failed to encode image for Qwen API")
            return {"error": "Failed to encode image", "status": "error"}

        try:
            # Qwen model doesn't accept system prompts, so we'll combine the context with the user prompt
            # Create the completion request using the OpenAI-compatible API
            logger.debug(
                "Sending request to Qwen API using model %s", self.model_name)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}" # Base64 string truncated in logs
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            # Extract the response content
            response_text = completion.choices[0].message.content
            logger.debug(
                "Received response from Qwen API: %.100s...", response_text)

            # Parse the response
            return self._parse_model_response(response_text)

        except Exception as e:
            logger.error("Error using Qwen API: %s", str(e))
            return {"error": f"API error: {str(e)}", "status": "error"}

    def _parse_model_response(self, response_text):
        """
        Parse the model's response text into a structured format.

        Args:
            response_text (str): Raw text response from the model

        Returns:
            dict: Structured game state information
        """
        # Try to parse as JSON if the response is in JSON format
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Otherwise, create a structured format from the text
        game_state = {"raw_description": response_text, "structured_data": {}}

        # Extract key information using simple heuristics
        sections = {
            "phase": ["phase", "turn"],
            "civilization": ["civilization", "civ"],
            "resources": ["resources", "gold", "science", "culture", "faith"],
            "units": ["units", "military", "troops"],
            "cities": ["cities", "city", "population"],
            "terrain": ["terrain", "features", "resources"],
            "notifications": ["notifications", "alerts", "warnings"],
            "research": ["research", "technology", "civic"],
            "diplomacy": ["diplomacy", "relations", "other civilizations"],
        }

        lines = response_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new section
            for section, keywords in sections.items():
                if any(line.lower().startswith(keyword.lower())
                        for keyword in keywords):
                    current_section = section
                    game_state["structured_data"][current_section] = line
                    break

            # Add to current section if we're in one
            if current_section and not line.lower().startswith(
                tuple(k.lower() for k in sections.keys())
            ):
                if isinstance(
                        game_state["structured_data"].get(current_section),
                        str):
                    game_state["structured_data"][current_section] += f" {line}"
                else:
                    game_state["structured_data"][current_section] = line

        return game_state
