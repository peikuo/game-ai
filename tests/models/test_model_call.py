#!/usr/bin/env python3
"""
Test script for the model_call.py module.
This script verifies that the centralized model calling functionality works correctly.
"""

import logging
import sys

from PIL import Image

from src.utils.config_loader import get_api_config
from src.utils.model_call import ModelCall, call_text_model, call_vision_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_vision_model():
    """Test the vision model call functionality."""
    logger.info("Testing vision model call...")

    # Load a test image
    try:
        # Try to load a screenshot from the screenshots directory
        image = Image.open("screenshots/latest.png")
        logger.info("Loaded test image from screenshots/latest.png")
    except Exception as e:
        logger.warning(f"Could not load test image: {e}")
        # Create a simple test image
        image = Image.new("RGB", (100, 100), color="red")
        logger.info("Created a simple test image")

    # Test direct model call
    prompt = "Describe what you see in this image briefly."

    # Get API config
    api_config = get_api_config()
    model_type = "qwen"  # Default to qwen

    # Check if Ollama is configured and use it if available
    if "ollama" in api_config and api_config["ollama"].get("api_url"):
        model_type = "ollama"
        logger.info("Using Ollama for vision test")

    # Create model call instance
    model_call = ModelCall(model_type=model_type)

    # Test the model call
    result = model_call.call_vision_model(image, prompt)

    if result.get("status") == "success":
        logger.info("Vision model call successful!")
        logger.info(f"Response: {result.get('response_text', '')[:100]}...")
    else:
        logger.error(f"Vision model call failed: {result.get('error')}")

    # Test convenience function
    logger.info("Testing vision convenience function...")
    response = call_vision_model(image, prompt, model_type=model_type)
    logger.info(f"Convenience function response: {response[:100]}...")


def test_text_model():
    """Test the text model call functionality."""
    logger.info("Testing text model call...")

    prompt = "What is the capital of France? Keep your answer very brief."

    # Get API config
    api_config = get_api_config()
    model_type = "qwen"  # Default to qwen

    # Check if Ollama is configured and use it if available
    if "ollama" in api_config and api_config["ollama"].get("api_url"):
        model_type = "ollama"
        logger.info("Using Ollama for text test")

    # Create model call instance
    model_call = ModelCall(model_type=model_type)

    # Test the model call
    result = model_call.call_text_model(prompt)

    if result.get("status") == "success":
        logger.info("Text model call successful!")
        logger.info(f"Response: {result.get('response_text', '')}")
    else:
        logger.error(f"Text model call failed: {result.get('error')}")

    # Test convenience function
    logger.info("Testing text convenience function...")
    response = call_text_model(prompt, model_type=model_type)
    logger.info(f"Convenience function response: {response}")


if __name__ == "__main__":
    logger.info("Starting model_call.py tests...")

    # Test vision model
    test_vision_model()

    # Test text model
    test_text_model()

    logger.info("All tests completed.")
