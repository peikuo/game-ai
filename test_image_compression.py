#!/usr/bin/env python
"""
Test script to verify the enhanced image compression logging in encode_image_to_base64.
This script creates test images with different characteristics and tests the compression
to verify the logging output.
"""

import os
import logging
import numpy as np
from PIL import Image

from src.utils.image_utils import encode_image_to_base64
from src.utils.log_utils import setup_logging

# Set up logging
setup_logging(log_level=logging.INFO, log_file="test_image_compression.log", cleanup=True)
logger = logging.getLogger("test_image_compression")

def create_test_image(width, height, mode="RGB", has_transparency=False):
    """Create a test image with specified dimensions and characteristics."""
    if mode == "RGB":
        # Create a colorful gradient image
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                array[y, x] = [r, g, b]
        img = Image.fromarray(array)
    elif mode == "RGBA" and has_transparency:
        # Create an image with transparency
        array = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                # Add transparency pattern
                alpha = 255 if (x + y) % 2 == 0 else 128
                array[y, x] = [r, g, b, alpha]
        img = Image.fromarray(array)
    else:
        # Create a simple RGB image
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                r = g = b = (x + y) % 256
                array[y, x] = [r, g, b]
        img = Image.fromarray(array)
    
    return img

def test_compression_logging():
    """Test image compression with different image types and verify logging."""
    logger.info("Starting image compression logging test")
    
    # Test case 1: Small RGB image
    logger.info("\n\nTest Case 1: Small RGB image")
    img1 = create_test_image(200, 150, "RGB")
    encode_image_to_base64(img1)
    
    # Test case 2: Medium RGB image
    logger.info("\n\nTest Case 2: Medium RGB image")
    img2 = create_test_image(800, 600, "RGB")
    encode_image_to_base64(img2)
    
    # Test case 3: Image with transparency
    logger.info("\n\nTest Case 3: Image with transparency")
    img3 = create_test_image(400, 300, "RGBA", has_transparency=True)
    encode_image_to_base64(img3)
    
    # Test case 4: Large image
    logger.info("\n\nTest Case 4: Large image")
    img4 = create_test_image(1920, 1080, "RGB")
    encode_image_to_base64(img4)
    
    logger.info("Image compression logging test completed")

if __name__ == "__main__":
    test_compression_logging()
