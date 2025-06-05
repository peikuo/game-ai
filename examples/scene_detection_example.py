#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to integrate scene detection with voice transcription
into the main game loop.
"""

from src.game_interface.scene_detect import SceneDetector
import logging
import os
import sys
import time

import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def capture_screenshot():
    """
    Capture a screenshot from the game window.
    In a real implementation, this would capture from the actual game window.
    For this example, we'll simulate it with a simple image.

    Returns:
        numpy.ndarray: Screenshot as a numpy array (RGB format)
    """
    # Create a simulated game screen (gray background with some elements)
    height, width = 480, 640
    screen = np.ones((height, width, 3), dtype=np.uint8) * \
        200  # Gray background

    # Add some game elements that might change between frames
    # This is just for demonstration
    current_time = time.time()

    # Moving element - create a red rectangle
    pos_x = int((current_time % 5) / 5 * (width - 100))
    screen[200:300, pos_x: pos_x + 100] = [255, 0, 0]  # Red rectangle

    return screen


def process_scene_results(scene_result):
    """
    Process the results from scene detection.

    Args:
        scene_result (dict): Results from SceneDetector
    """
    if not scene_result:
        logger.info("No scene results available")
        return

    # Log animation detection
    if scene_result.get("animation_detected", False):
        logger.info("Animation detected in scene")

    # Log voice detection and transcription
    if scene_result.get("voice_detected", False):
        logger.info("Voice detected in scene")

        # Process dialogue if available
        dialogue = scene_result.get("dialogue")
        if dialogue:
            logger.info("Transcribed dialogue:")
            for segment in dialogue.get("dialogue", []):
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment.get("text", "")
                logger.info(f"{speaker}: {text}")

                # In a real game, you might use this text for:
                # 1. Decision making based on NPC dialogue
                # 2. Storing conversation history
                # 3. Triggering responses or actions
        else:
            logger.info("No dialogue transcribed")

    # Process key frames
    key_frames = scene_result.get("animation_frames", [])
    if key_frames:
        logger.info(f"Captured {len(key_frames)} key frames from animation")
        # In a real game, you might use these frames for:
        # 1. Visual analysis of important animation moments
        # 2. Detecting specific events or transitions
        # 3. Saving as reference images


def main():
    """
    Main function demonstrating scene detection integration with game loop.
    """
    # Create audio directory if it doesn't exist
    audio_dir = "temp_audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        logger.info(f"Created directory: {audio_dir}")

    # Initialize scene detector with configuration
    config = {
        "animation_diff_threshold": 0.02,  # Sensitivity for detecting animation
        "animation_consecutive_frames": 3,  # Frames needed to confirm animation
        "animation_interval": 0.1,  # Time between animation checks
        "voice_detection_enabled": True,  # Enable voice detection
        "voice_silence_threshold": 1.0,  # Seconds of silence to stop recording
        "whisper_model": "tiny",  # Speech recognition model size
        "whisper_language": "zh",  # Default language (Chinese)
        "audio_recording_path": audio_dir,  # Where to save audio recordings
    }

    detector = SceneDetector(config)
    logger.info("Scene detector initialized")

    # Simulate game loop
    logger.info("Starting game loop simulation")
    logger.info("Press Ctrl+C to stop")

    try:
        scene_active = False
        scene_result = None

        # Main game loop
        while True:
            # Capture screenshot (in real game, this would be from the game
            # window)
            screenshot = capture_screenshot()

            # Process the frame with scene detector
            if scene_active:
                # Continue recording the active scene
                scene_result = detector.record(screen=screenshot)

                # Check if scene is finished
                if scene_result.get("finished", False):
                    logger.info("Scene completed")
                    process_scene_results(scene_result)
                    scene_active = False
            else:
                # Check for new scene
                scene_result = detector.record(screen=screenshot)

                # If animation or voice is detected, a scene has started
                if scene_result.get(
                        "animation_detected",
                        False) or scene_result.get(
                        "voice_detected",
                        False):
                    logger.info("New scene detected")
                    scene_active = True

            # In a real game, you would:
            # 1. Update game state based on scene detection
            # 2. Make decisions based on transcribed dialogue
            # 3. Analyze key frames for important visual information

            # Simulate game frame rate
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Game loop stopped by user")

    logger.info("Example completed")


if __name__ == "__main__":
    main()
