"""
Image analyzer for game screenshots using vision models.

This module provides functionality to analyze screenshots from various games
using multimodal vision models. It supports different model backends and
can be configured for different games.
"""

import json
import logging

from PIL import Image

from src.utils.config_loader import get_api_config
from src.utils.image_utils import process_screenshot, encode_image_to_base64, optimize_image
from src.utils import model_call
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
        if isinstance(image, Image.Image) and not image.info.get('optimized', False):
            logger.debug("Processing image for analysis")
            # Use image_utils.py for all image processing
            image = optimize_image(image)
            image = process_screenshot(image)
            
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
            # Prepare image for analysis
            processed_image = self._prepare_image(image)
            
            # Encode the image to base64
            base64_image = encode_image_to_base64(processed_image, max_width=1024, optimize=True)
            
            # If no prompt is provided, use the default game state prompt
            if not prompt:
                prompt = self.game_prompts.get(analysis_type, "Describe what you see in this game screenshot.")
            
            # Use the centralized model call module
            response = model_call.call_vision_model(base64_image, prompt)
            
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
