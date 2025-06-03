#!/usr/bin/env python
"""
Test script to verify TTS functionality in the GameAnalyzer.
This script directly tests the GameAnalyzer with TTS enabled.
"""

import logging
import sys
from pathlib import Path

from src.game_player.game_analyzer import GameAnalyzer
from src.tts.tts_manager import TTSManager
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("game_analyzer_tts_test")

def test_game_analyzer_tts():
    """Test GameAnalyzer with TTS enabled"""
    # Load the default config
    config = load_config("config/default.yaml")
    
    # Ensure audio config exists and is enabled
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["enabled"] = True
    
    logger.info(f"Audio config: {config.get('audio', {})}")
    
    # Create a GameAnalyzer with TTS enabled
    analyzer = GameAnalyzer(config=config)
    
    # Check if TTS manager was created
    if analyzer.tts_manager and analyzer.tts_manager.is_available():
        logger.info("✅ GameAnalyzer created TTS manager successfully")
        logger.info(f"Voice: {analyzer.tts_manager.voice_id}, Model: {analyzer.tts_manager.repo_id}")
        
        # Test TTS with a sample monologue
        test_monologue = "This is a test monologue from the game analyzer."
        logger.info(f"Testing TTS with monologue: '{test_monologue}'")
        
        # Directly use the TTS manager to speak the monologue
        success = analyzer.tts_manager.speak_monologue(test_monologue)
        if success:
            logger.info("✅ TTS played monologue successfully")
        else:
            logger.warning("⚠️ TTS failed to play monologue")
    else:
        logger.error("❌ GameAnalyzer failed to create TTS manager")

if __name__ == "__main__":
    test_game_analyzer_tts()
