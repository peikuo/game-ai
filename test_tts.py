#!/usr/bin/env python
"""
Test script to verify TTS functionality in the Game AI.
This script initializes the TTS components directly to verify proper configuration.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.tts.tts_manager import TTSManager
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tts_test")

def test_tts_initialization():
    """Test TTS initialization with different config structures."""
    # Load the default config
    config = load_config("config/default.yaml")
    
    # Ensure audio config exists and is enabled
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["enabled"] = True
    
    logger.info(f"Testing TTS with full config: {config.get('audio', {})}")
    
    # Test 1: Initialize with full config
    try:
        tts_manager1 = TTSManager(config)
        logger.info("✅ TTS initialized successfully with full config")
        logger.info(f"Voice: {tts_manager1.voice_id}, Model: {tts_manager1.repo_id}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS with full config: {e}")
    
    # Test 2: Initialize with just audio section
    try:
        audio_config = config.get("audio", {})
        tts_manager2 = TTSManager(audio_config)
        logger.info("✅ TTS initialized successfully with audio config section")
        logger.info(f"Voice: {tts_manager2.voice_id}, Model: {tts_manager2.repo_id}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS with audio config section: {e}")
    
    # Test 3: Test TTS with a sample text
    try:
        test_text = "This is a test of the text to speech system."
        logger.info(f"Testing TTS speech generation with text: '{test_text}'")
        
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate and play speech directly
        success = tts_manager2.speak_monologue(test_text)
        if success:
            logger.info("✅ TTS generated and played speech successfully")
        else:
            logger.warning("⚠️ TTS failed to generate or play speech")
    except Exception as e:
        logger.error(f"❌ Failed to generate or play speech: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TTS functionality")
    args = parser.parse_args()
    
    test_tts_initialization()
