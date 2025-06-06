#!/usr/bin/env python
"""
Test script for scene detection functionality.
This script waits for 5 seconds and then starts scene detection,
allowing time to play a video or audio for testing.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

# Import project modules
from src.capture.screenshot import ScreenCapturer
from src.game_interface.scene_detect import SceneDetector
from src.utils.session_logger import SessionLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test scene detection functionality")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum duration to monitor for scenes in seconds",
    )
    parser.add_argument(
        "--animation-threshold",
        type=float,
        default=0.02,
        help="Threshold for detecting animation changes",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=3,
        help="Number of consecutive frames needed to detect animation",
    )
    parser.add_argument(
        "--voice-enabled",
        action="store_true",
        help="Enable voice detection",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for speech recognition",
    )
    parser.add_argument(
        "--whisper-language",
        type=str,
        default="zh",
        help="Language code for speech recognition",
    )
    parser.add_argument(
        "--session-log",
        action="store_true",
        help="Enable session logging",
    )
    return parser.parse_args()


def main():
    """Main function to test scene detection."""
    args = parse_args()
    
    # Initialize screenshot capture with configuration
    screenshot_config = {
        "use_full_screen": True,
        "save_screenshots": False,
        "save_path": "screenshots",
    }
    screenshot_capture = ScreenCapturer(screenshot_config)
    logger.info("Screenshot capture initialized")
    
    # Create scene detector
    scene_detector = SceneDetector({
        "animation_diff_threshold": args.animation_threshold,
        "animation_consecutive_frames": args.consecutive_frames,
        "voice_detection_enabled": args.voice_enabled,
        "whisper_model": args.whisper_model,
        "whisper_language": args.whisper_language,
        "audio_recording_path": "temp_audio",
    })
    
    # Reset the scene detector completely for a new session
    scene_detector.reset()
    
    logger.info("Scene detector initialized")
    
    # Initialize session logger if enabled
    session_log = None
    if args.session_log:
        session_log = SessionLogger(log_dir="session_logs", game_name="Scene Detection Test")
        logger.info("Session logging enabled")
    
    # Wait for 5 seconds before starting
    logger.info("Waiting 5 seconds before starting scene detection...")
    logger.info("Please prepare to play your video/audio for testing")
    for i in range(5, 0, -1):
        logger.info(f"Starting in {i} seconds...")
        time.sleep(1)
    
    logger.info("Starting scene detection now!")
    
    # Main monitoring loop
    start_time = time.time()
    turn_count = 0
    scene_active = False
    
    try:
        while time.time() - start_time < args.max_duration:
            # Increment turn counter
            turn_count += 1
            logger.info(f"Turn {turn_count}")
            
            # Capture screenshot
            image = screenshot_capture.capture()
            
            # Log screenshot if enabled
            if session_log:
                session_log.log_screenshot(image, turn=turn_count)
            
            # Check for scene detection
            scene_result = scene_detector.record(image)
            
            if not scene_result:
                logger.error("No scene results available, check logs!")
                continue
            
            if not scene_result.get("finished", True):
                logger.info("Scene detection in progress, waiting...")
                scene_active = True
                time.sleep(1)  # Wait 1 second before continuing
                continue
            
            # If we were in an active scene and now it's finished
            if scene_active and scene_result.get("finished", True):
                logger.info("Scene detection finished!")
                scene_active = False
                
                # Log scene detection results
                if scene_result.get("animation_detected", False):
                    logger.info("Animation detected in scene")
                    frames_count = len(scene_result.get("animation_frames", []))
                    logger.info(f"Captured {frames_count} animation frames")
                
                if scene_result.get("voice_detected", False):
                    logger.info("Voice detected in scene")
                    dialogue = scene_result.get("dialogue", "No dialogue transcribed")
                    logger.info(f"Transcribed dialogue: {dialogue}")
                
                # Log scene detection results if session logging is enabled
                if args.session_log:
                    # Debug: print scene_result structure
                    logger.info(f"Scene result keys: {list(scene_result.keys())}")
                    for key, value in scene_result.items():
                        if key != 'animation_frames':
                            logger.info(f"Scene result {key}: {type(value)} = {value}")
                    
                    # Force animation_detected to True for testing base64 embedding
                    if not scene_result.get("animation_detected") and not scene_result.get("animation_frames"):
                        # Create a simple test frame for demonstration
                        logger.info("Creating test animation frame for base64 embedding demonstration")
                        test_frame = np.zeros((200, 300, 3), dtype=np.uint8)
                        test_frame[:100, :150] = [255, 0, 0]  # Red rectangle
                        test_frame[100:, 150:] = [0, 0, 255]  # Blue rectangle
                        scene_result["animation_detected"] = True
                        scene_result["animation_frames"] = [test_frame]
                    
                    session_log.log_scene_detection(scene_result, turn=turn_count)
                    
                # Break out of the loop if we've detected and finished a scene
                logger.info("Scene detection test completed successfully!")
                break
            
            # Short delay between captures
            time.sleep(0.5)
        
        # If we exit the loop without detecting a scene
        if not scene_active and time.time() - start_time >= args.max_duration:
            logger.warning(f"No scene detected within {args.max_duration} seconds")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during scene detection test: {e}")
    
    finally:
        logger.info("Scene detection test completed")
        if session_log:
            session_log.log_summary(
                total_turns=turn_count,
                additional_notes="Scene detection test summary"
            )


if __name__ == "__main__":
    main()
