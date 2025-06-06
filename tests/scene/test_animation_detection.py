#!/usr/bin/env python3
"""
Test script for animation detection using existing screenshots.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.game_interface.scene_detect import SceneDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_images_from_directory(directory):
    """Load all images from a directory and sort them by name."""
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(Path(directory).glob(ext))
    
    # Sort files by name
    image_files = sorted(image_files)
    
    images = {}
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            images[img_path.name] = img_array
            logger.info(f"Loaded image: {img_path.name}, shape: {img_array.shape}")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
    
    logger.info(f"Loaded {len(images)} images")
    return images

def test_animation_detection(images, threshold=None):
    """Test animation detection on a series of images."""
    logger = logging.getLogger(__name__)
    
    # Initialize scene detector with default config
    config = {
        "animation_diff_threshold": 0.01,
        "animation_consecutive_frames": 3,
        "min_frames": 3,
        "voice_detection_enabled": False,
        "whisper_model": "tiny"
    }
    
    if threshold is not None:
        config["animation_diff_threshold"] = threshold
    
    scene_detector = SceneDetector(config)
    
    # Process each image
    results = []
    animated_frames = 0
    consecutive_changes = 0
    animation_detected = False
    
    # Initialize with first frame
    if len(images) > 0:
        first_img_name = list(images.keys())[0]
        first_img = images[first_img_name]
        logger.info(f"Initialized with first frame: {first_img_name}")
        scene_detector.prev_frame = first_img  # Set initial frame
        
    # Process subsequent frames
    for i, img_name in enumerate(list(images.keys())[1:], 1):
        img_array = images[img_name]
        
        # Skip if image is None
        if img_array is None:
            logger.warning(f"Skipping frame {i}: {img_name} - Image is None")
            continue
        
        # Check for animation between consecutive frames
        is_frame_changed = scene_detector._detect_screen_change(img_array, threshold)
        diff_value = scene_detector.frame_diffs[-1] if scene_detector.frame_diffs else 0
        
        # Update consecutive changes counter (simulating SceneDetector logic)
        if is_frame_changed:
            consecutive_changes += 1
            if consecutive_changes >= config["min_frames"] and not animation_detected:
                animation_detected = True
                logger.info(f"Animation detected after {consecutive_changes} consecutive changes")
        else:
            consecutive_changes = 0
            
        # Update previous frame
        scene_detector.prev_frame = img_array
        
        logger.info(f"Frame {i}: {img_name} - Frame changed: {is_frame_changed}, Difference: {diff_value:.6f}, "
                   f"Threshold: {scene_detector.animation_diff_threshold:.6f}, "
                   f"Consecutive changes: {consecutive_changes}, Animation detected: {animation_detected}")
        
        results.append({
            "frame": i,
            "name": img_name,
            "frame_changed": is_frame_changed,
            "difference": diff_value,
            "consecutive_changes": consecutive_changes,
            "animation_detected": animation_detected
        })
        
        if is_frame_changed:
            animated_frames += 1
    
    # Calculate statistics
    animated_frames = sum(1 for r in results if r["frame_changed"])
    logger.info(f"Animation detected in {animated_frames} out of {len(results)} frame pairs")
    
    # Print detailed results for frames with highest differences
    sorted_results = sorted(results, key=lambda x: x["difference"], reverse=True)
    logger.info("\nTop 3 frames with highest differences:")
    for r in sorted_results[:3]:
        logger.info(f"Frame {r['frame']}: {r['name']} - Difference: {r['difference']:.6f}, Animated: {r['frame_changed']}")
        
    # Return final animation detection status
    return {"animation_detected": animation_detected, "frame_changes": animated_frames, "results": results}

def main():
    parser = argparse.ArgumentParser(description="Test animation detection on a sequence of images")
    parser.add_argument("--dir", default="session_logs/screenshots/20250605_195424", help="Directory containing images")
    parser.add_argument("--threshold", type=float, help="Animation difference threshold (default: 0.01)")
    parser.add_argument("--auto-adjust", action="store_true", help="Automatically adjust threshold if no animation detected")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Loading images from {args.dir}")
    images = load_images_from_directory(args.dir)
    logger.info(f"Loaded {len(images)} images")
    
    if not images:
        logger.error("No images found in the specified directory")
        return
    
    logger.info(f"Testing animation detection with threshold: {args.threshold or 'default (0.01)'}")
    result_data = test_animation_detection(images, args.threshold)
    
    # Print summary
    if not result_data["animation_detected"]:
        logger.warning("No animation detected in any frames!")
    
    # Try with a lower threshold if no animation was detected
    if hasattr(args, 'auto_adjust') and args.auto_adjust and not result_data["animation_detected"]:
        new_threshold = args.threshold / 2 if args.threshold else 0.005
        logger.info(f"\nAutomatically adjusting threshold to {new_threshold}")
        test_animation_detection(images, new_threshold)

if __name__ == "__main__":
    main()
