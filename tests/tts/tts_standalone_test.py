#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone TTS test script that doesn't rely on the TTSManager class.
This script directly uses the kokoro TTS library to generate speech
and implements NumPy-free audio playback methods.
"""

import argparse
import logging
import sys

import torch

from src.tts.tts_manager import TTSManager

# Add the project root to the path so we can import our modules
sys.path.append(".")  # Add the current directory to the path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tts_standalone_test")

# TTS Configuration
TTS_CONFIG = {
    "tts_model": "hexgrad/Kokoro-82M-v1.1-zh",
    "voice": "zf_001",
    "temp_folder": "temp_audio",
    "sample_rate": 24000,
}


# These functions are now handled by TTSManager


def test_tts_standalone():
    """Test TTS functionality using TTSManager"""
    # Initialize TTS Manager
    tts_manager = TTSManager(TTS_CONFIG)
    if not tts_manager.is_available():
        logger.error("Failed to initialize TTS model")
        return False

    # Test text (Chinese)
    test_text = "这是一个测试，用于检查文本转语音功能。"
    logger.info(f"Testing TTS with text: '{test_text}'")

    try:
        # Use TTSManager to speak the text
        success = tts_manager.speak_monologue(test_text)
        return success

    except Exception as e:
        logger.error(f"Error in TTS test: {e}")
        return False


def test_tts_english():
    """Test TTS with English text"""
    # Initialize TTS Manager
    tts_manager = TTSManager(TTS_CONFIG)
    if not tts_manager.is_available():
        logger.error("Failed to initialize TTS model")
        return False

    # Test text (English)
    test_text = "This is a test for English text in Chinese TTS."
    logger.info(f"Testing TTS with English text: '{test_text}'")

    try:
        # Use TTSManager to speak the text
        success = tts_manager.speak_monologue(test_text)
        return success

    except Exception as e:
        logger.error(f"Error in TTS test: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TTS test")
    parser.add_argument(
        "--text",
        type=str,
        help="Text to convert to speech",
        default="这是一个测试，用于检查文本转语音功能。",
    )
    parser.add_argument("--voice", type=str, help="Voice ID", default="zf_001")
    args = parser.parse_args()

    # Update config with command line arguments
    if args.voice:
        TTS_CONFIG["voice"] = args.voice

    # Test TTS with Chinese text
    success = test_tts_standalone()
    logger.info(f"Chinese TTS test {'succeeded' if success else 'failed'}")

    # Test TTS with English text
    success = test_tts_english()
    logger.info(f"English TTS test {'succeeded' if success else 'failed'}")
