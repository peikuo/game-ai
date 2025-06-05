#!/usr/bin/env python3
"""
Test script to verify the integration of the centralized model_call.py module.
"""

import logging
import sys

from PIL import Image

from src.utils.image_utils import encode_image_to_base64
from src.utils.model_call import ModelCall
from src.vision.image_analyzer import ImageAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_image_analyzer():
    """Test the image analyzer with the centralized model call."""
    logger.info("Testing image analyzer with centralized model call...")

    # Create an image analyzer
    analyzer = ImageAnalyzer(model="qwen")

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

    # Test the image analyzer
    prompt = "Describe what you see in this image briefly."
    result = analyzer.analyze(image, prompt=prompt)

    if "error" in result:
        logger.error(f"Image analysis failed: {result.get('error')}")
    else:
        logger.info("Image analysis successful!")
        logger.info(f"Analysis result: {result}")


def test_direct_model_call():
    """Test the direct model call with base64 encoding."""
    logger.info("Testing direct model call with base64 encoding...")

    # Create a model call instance
    model_call = ModelCall(model_type="qwen")

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

    # Encode the image to base64
    base64_image = encode_image_to_base64(image, max_width=1024, optimize=True)

    # Test the model call
    prompt = "Describe what you see in this image briefly."
    result = model_call.call_vision_model(base64_image, prompt)

    if result.get("status") == "success":
        logger.info("Direct model call successful!")
        logger.info(f"Response: {result.get('response_text', '')[:100]}...")
    else:
        logger.error(f"Direct model call failed: {result.get('error')}")


if __name__ == "__main__":
    logger.info("Starting model integration tests...")

    # Test the image analyzer
    test_image_analyzer()

    # Test direct model call
    test_direct_model_call()

    logger.info("All tests completed.")
