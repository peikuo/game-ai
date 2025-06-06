#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for scene detection with voice transcription.
This script demonstrates the use of the SceneDetector class
to detect animation and voice in a simulated game scene.
"""

import logging
import os
import sys
import time

import numpy as np
from PIL import Image

from src.game_interface.scene_detect import SceneDetector

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_frames(num_frames=30, width=640, height=480):
    """
    Create test frames with some simulated animation.

    Args:
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height

    Returns:
        List of numpy arrays representing frames
    """
    frames = []

    # Create base frame (gray background)
    base_frame = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Generate frames with a moving object
    for i in range(num_frames):
        # Copy base frame
        frame = base_frame.copy()

        # Add a moving rectangle
        pos_x = int((i / num_frames) * (width - 100))
        pos_y = height // 2 - 50

        # Draw rectangle
        frame[pos_y: pos_y + 100, pos_x: pos_x +
              100] = [255, 0, 0]  # Red rectangle

        # Add frame to list (as numpy array)
        frames.append(frame)

    return frames


def test_scene_detection():
    """
    Test the SceneDetector class with simulated frames and real audio recording.
    """
    # Create audio directory if it doesn't exist
    audio_dir = "temp_audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        logger.info(f"Created directory: {audio_dir}")

    # Create a SceneDetector instance
    config = {
        "animation_diff_threshold": 0.02,
        "animation_consecutive_frames": 3,
        "animation_interval": 0.1,
        "voice_detection_enabled": True,
        "voice_silence_threshold": 2.0,  # 2 seconds of silence to stop
        "whisper_model": "tiny",  # Use tiny model for faster testing
        "whisper_language": "zh",  # Default to Chinese
        "audio_recording_path": audio_dir,
    }

    detector = SceneDetector(config)

    # Create test frames
    logger.info("Generating test frames...")
    frames = create_test_frames(num_frames=30)

    # Simulate a scene with animation and voice
    logger.info("Starting scene detection test...")
    logger.info("Speak into your microphone to test voice detection")
    logger.info("Press Ctrl+C to stop the test")

    result = None

    try:
        # Process frames one by one with a delay to simulate real-time
        for i, frame in enumerate(frames):
            logger.info(f"Processing frame {i+1}/{len(frames)}")

            # Record the frame
            result = detector.record(screen=frame)

            # Check if scene is finished
            if result and result.get("finished", False):
                logger.info("Scene detection finished")
                break

            # Wait a bit to simulate real-time processing
            time.sleep(0.2)

        # If we've gone through all frames but scene isn't finished yet,
        # wait for voice recording to complete
        if result and not result.get("finished", False):
            logger.info(
                "Animation finished, waiting for voice recording to complete..."
            )

            # Wait for up to 10 seconds for voice recording to complete
            for _ in range(50):  # 10 seconds (50 * 0.2)
                # Use the last frame for continued recording
                result = detector.record(frames[-1])

                if result and result.get("finished", False):
                    logger.info("Scene detection finished")
                    break

                time.sleep(0.2)

        # Display results
        logger.info("Scene detection results:")
        logger.info(
            f"Animation detected: {result.get('animation_detected', False)}")
        logger.info(f"Voice detected: {result.get('voice_detected', False)}")

        # Display dialogue if available
        dialogue = result.get("dialogue")
        if dialogue:
            logger.info("Transcribed dialogue:")
            for segment in dialogue.get("dialogue", []):
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment.get("text", "")
                logger.info(f"{speaker}: {text}")
        else:
            logger.info("No dialogue transcribed")

        # Display key frames
        key_frames = result.get("animation_frames", [])
        logger.info(f"Captured {len(key_frames)} key frames")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # No need to explicitly stop the detector
        logger.info("Test completed")


if __name__ == "__main__":
    test_scene_detection()
