"""
Centralized model calling utilities for the Game AI project.

This module provides a unified interface for making calls to various AI models,
abstracting away the specific implementation details of each model provider.
Currently supports Qwen and Ollama models.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import requests
from openai import OpenAI

from src.utils.config_loader import get_api_config
from src.utils.image_utils import truncate_base64
from src.utils.log_utils import get_session_logger

logger = logging.getLogger(__name__)

# Module-level variables for singleton instance
_model_type = "qwen"
_config = None
_model_instance = None


class _ModelCall:
    """
    Private class for handling all model API calls.
    Not meant to be instantiated directly outside of this module.
    """

    def __init__(self, model_type: str = "qwen",
                 config: Optional[Dict] = None):
        """
        Initialize the _ModelCall with the specified model type and configuration.

        Args:
            model_type (str): Type of model to use ('qwen', 'ollama')
            config (dict, optional): Configuration for the model
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        self.client = None
        self.api_key = None
        self.base_url = None
        self.model_name = None

        # Set up model-specific configurations
        self._setup_model()

    def _setup_model(self):
        """Set up the model based on the model type."""
        # Load API configuration from centralized config loader
        api_config = get_api_config()

        if self.model_type == "ollama":
            ollama_config = api_config.get("ollama", {})

            # Get API URL from config or centralized API config
            self.api_url = self.config.get("ollama_api") or ollama_config.get(
                "api_url", "http://localhost:11434/api/generate"
            )
            self.model_name = self.config.get(
                "ollama_model") or ollama_config.get("model", "llava:latest")
            logger.info("Using Ollama model: %s", self.model_name)

        elif self.model_type == "qwen":
            qwen_config = api_config.get("qwen", {})

            # Get API key from config or centralized API config
            self.api_key = self.config.get(
                "qwen_api_key") or qwen_config.get("api_key")

            # Get model name and base URL
            self.model_name = self.config.get("qwen_model") or qwen_config.get(
                "model", "qwen-vl-max-latest"
            )
            self.base_url = self.config.get("qwen_base_url") or qwen_config.get(
                "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            if not self.api_key:
                logger.warning(
                    "Qwen API key not found. Set in config or .env file.")
            else:
                # Initialize OpenAI client for Qwen
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url)
                logger.info(
                    "Using Qwen model: %s via OpenAI-compatible API",
                    self.model_name)

        else:
            logger.warning(
                f"Unknown model type: {self.model_type}, falling back to Qwen"
            )
            self.model_type = "qwen"
            self._setup_model()  # Recursive call to set up with the default model

    def call_vision_model(self, base64_image: str,
                          prompt: str) -> Dict[str, Any]:
        """
        Call a vision model with a base64-encoded image and prompt.

        Args:
            base64_image (str): Base64-encoded image string
            prompt (str): Prompt for analysis

        Returns:
            dict: Response from the model
        """
        if self.model_type == "ollama":
            return self._call_ollama_vision(base64_image, prompt)
        elif self.model_type == "qwen":
            return self._call_qwen_vision(base64_image, prompt)
        else:
            error_msg = f"Unsupported model type: {self.model_type}"
            logger.error(error_msg)
            return {"error": error_msg}

    def call_video_model(
            self, base64_frames: List[str], prompt: str) -> Dict[str, Any]:
        """
        Call a model with a sequence of base64-encoded frames (video) and prompt.

        Args:
            base64_frames (list): List of base64-encoded frame strings
            prompt (str): Prompt for analysis

        Returns:
            dict: Response from the model
        """
        if not base64_frames or len(base64_frames) < 1:
            error_msg = "No frames provided for analysis"
            logger.error(error_msg)
            return {"error": error_msg}

        if self.model_type == "qwen":
            return self._call_qwen_video(base64_frames, prompt)
        else:
            error_msg = (
                f"Video analysis not supported for model type: {self.model_type}"
            )
            logger.error(error_msg)
            return {"error": error_msg}

    def call_text_model(self, prompt: str) -> Dict[str, Any]:
        """
        Call a text-only model with a prompt.

        Args:
            prompt (str): Prompt for the model

        Returns:
            dict: Response from the model
        """
        if self.model_type == "ollama":
            return self._call_ollama_text(prompt)
        elif self.model_type == "qwen":
            return self._call_qwen_text(prompt)
        else:
            error_msg = f"Unsupported model type: {self.model_type}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _call_ollama_vision(self, base64_image: str,
                            prompt: str) -> Dict[str, Any]:
        """
        Call Ollama vision model with a base64-encoded image and prompt.

        Args:
            base64_image (str): Base64-encoded image string
            prompt (str): Prompt for analysis

        Returns:
            dict: Response from the model
        """
        # Log truncated base64 string for debugging
        logger.debug("Using base64 image: %s", truncate_base64(base64_image))

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "images": [base64_image],
        }

        logger.debug("Sending request to Ollama API: %s", self.api_url)
        try:
            response = requests.post(self.api_url, json=payload)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                logger.debug(
                    "Received response from Ollama API: %.100s...",
                    response_text)
                return {"response_text": response_text, "status": "success"}
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                logger.error(error_msg)
                return {"error": error_msg, "status": "error"}
        except Exception as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

    def _call_qwen_vision(self, base64_image: str,
                          prompt: str) -> Dict[str, Any]:
        """
        Call Qwen vision model with a base64-encoded image and prompt.

        Args:
            base64_image (str): Base64-encoded image string
            prompt (str): Prompt for analysis

        Returns:
            dict: Response from the model
        """
        # Check if API key and client are available
        if not self.api_key or not self.client:
            error_msg = "Qwen API key not found or client not initialized"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

        # Log truncated base64 string for debugging
        logger.debug("Using base64 image: %s", truncate_base64(base64_image))

        try:
            # Create the completion request using the OpenAI-compatible API
            logger.debug(
                "Sending request to Qwen API using model %s",
                self.model_name)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # Extract the response content
            response_text = completion.choices[0].message.content
            logger.info("Received response from Qwen API: %s", response_text)

            return {"response_text": response_text, "status": "success"}

        except Exception as e:
            error_msg = f"Error calling Qwen API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

    def _call_qwen_video(
            self, base64_frames: List[str], prompt: str) -> Dict[str, Any]:
        """
        Call Qwen model with a sequence of base64-encoded frames (video) and prompt.

        Args:
            base64_frames (list): List of base64-encoded frame strings
            prompt (str): Prompt for analysis

        Returns:
            dict: Response from the model
        """
        # Check if API key and client are available
        if not self.api_key or not self.client:
            error_msg = "Qwen API key not found or client not initialized"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

        try:
            # Log truncated base64 strings for debugging
            for i, frame in enumerate(base64_frames):
                logger.debug("Frame %d base64: %s", i, truncate_base64(frame))

            # Prepare the video frames for the API request
            video_frames = [
                f"data:image/jpeg;base64,{img}" for img in base64_frames]

            # Create the completion request
            logger.debug(
                "Sending frame sequence analysis request to Qwen API using model: %s",
                self.model_name,
            )

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_frames},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )

            # Extract the response content
            response_text = completion.choices[0].message.content
            logger.debug(
                "Received response from Qwen API: %.100s...",
                response_text)

            return {
                "response_text": response_text,
                "frame_count": len(base64_frames),
                "status": "success",
            }

        except Exception as e:
            error_msg = f"Error calling Qwen API for video analysis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

    def _call_ollama_text(self, prompt: str) -> Dict[str, Any]:
        """
        Call Ollama text model with a prompt.

        Args:
            prompt (str): Prompt for the model

        Returns:
            dict: Response from the model
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        logger.debug("Sending text request to Ollama API: %s", self.api_url)
        try:
            response = requests.post(self.api_url, json=payload)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                logger.debug(
                    "Received response from Ollama API: %.100s...",
                    response_text)
                return {"response_text": response_text, "status": "success"}
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                logger.error(error_msg)
                return {"error": error_msg, "status": "error"}
        except Exception as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

    def _call_qwen_text(self, prompt: str) -> Dict[str, Any]:
        """
        Call Qwen text model with a prompt.

        Args:
            prompt (str): Prompt for the model

        Returns:
            dict: Response from the model
        """
        # Check if API key and client are available
        if not self.api_key or not self.client:
            error_msg = "Qwen API key not found or client not initialized"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}

        try:
            # Create the completion request using the OpenAI-compatible API
            logger.debug(
                "Sending text request to Qwen API using model %s",
                self.model_name)
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )

            # Extract the response content
            response_text = completion.choices[0].message.content
            logger.debug(
                "Received response from Qwen API: %.100s...",
                response_text)

            return {"response_text": response_text, "status": "success"}

        except Exception as e:
            error_msg = f"Error calling Qwen API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "error"}


# Module initialization function
def initialize(model_type: str = "qwen",
               config: Optional[Dict] = None) -> None:
    """
    Initialize the model call module with the specified model type and configuration.
    This should be called once at the start of the application.

    Args:
        model_type (str): Type of model to use ('qwen', 'ollama')
        config (dict, optional): Configuration for the model
    """
    global _model_type, _config, _model_instance
    _model_type = model_type.lower()
    _config = config or {}
    _model_instance = None  # Reset instance to force recreation with new settings
    logger.info(
        f"Model call module initialized with model type: {_model_type}")


def _get_model_instance():
    """
    Get or create the singleton model instance.

    Returns:
        _ModelCall: The singleton model instance
    """
    global _model_instance, _model_type, _config
    if _model_instance is None:
        _model_instance = _ModelCall(model_type=_model_type, config=_config)
    return _model_instance


def call_vision_model(base64_image: str, prompt: str) -> Dict[str, Any]:
    """
    Call a vision model with a base64-encoded image and prompt.

    Args:
        base64_image (str): Base64-encoded image string
        prompt (str): Prompt for analysis

    Returns:
        dict: Response from the model with status and response_text or error
    """
    # Log the prompt being used for debugging
    logger.info(
        f"===== VISION MODEL PROMPT START =====\n{prompt}\n===== VISION MODEL PROMPT END ====="
    )

    # Log image size for debugging
    image_size_kb = len(base64_image) / 1024 if base64_image else 0
    logger.info(f"Image size: {image_size_kb:.2f} KB")

    model = _get_model_instance()
    return model.call_vision_model(base64_image, prompt)


def call_video_model(base64_frames: List[str], prompt: str) -> Dict[str, Any]:
    """
    Call a model with a sequence of base64-encoded frames (video) and prompt.

    Args:
        base64_frames (list): List of base64-encoded frame strings
        prompt (str): Prompt for analysis

    Returns:
        dict: Response from the model with status and response_text or error
    """
    model = _get_model_instance()
    return model.call_video_model(base64_frames, prompt)


def call_text_model(prompt: str) -> Dict[str, Any]:
    """
    Call a text-only model with a prompt.

    Args:
        prompt (str): Prompt for the model

    Returns:
        dict: Response from the model with status and response_text or error
    """
    model = _get_model_instance()
    return model.call_text_model(prompt)
