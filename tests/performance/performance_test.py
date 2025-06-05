#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance test for scene_detect.py
"""

import logging
import os
import time

import numpy as np
import psutil

from src.game_interface.scene_detect import SceneDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def main():
    # Initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Create audio directory if it doesn't exist
    audio_dir = "temp_audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Initialize scene detector with configuration
    config = {
        "animation_diff_threshold": 0.02,  # Sensitivity for detecting animation
        "animation_consecutive_frames": 3,  # Frames needed to confirm animation
        "animation_interval": 0.1,  # Time between animation checks
        "voice_detection_enabled": True,  # Enable voice detection
        "voice_silence_threshold": 1.0,  # Seconds of silence to stop recording
        # Speech recognition model size (use tiny for performance test)
        "whisper_model": "tiny",
        "whisper_language": "zh",  # Default language (Chinese)
        "audio_recording_path": audio_dir,  # Where to save audio recordings
    }

    start_time = time.time()
    detector = SceneDetector(config)
    init_time = time.time() - start_time
    logger.info(f"SceneDetector initialization time: {init_time:.4f} seconds")

    # Memory after initialization
    after_init_memory = get_memory_usage()
    logger.info(
        f"Memory after initialization: {after_init_memory:.2f} MB (delta: {after_init_memory - initial_memory:.2f} MB)"
    )

    # Generate test frames (720p resolution)
    logger.info("Generating test frames...")
    frame_count = 30
    frames = []
    for i in range(frame_count):
        # Create random frame with some patterns to simulate animation
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Add some moving elements to simulate animation
        offset = i * 10
        frame[100 + offset: 300 + offset, 200:400,
              0] = 255  # Red rectangle that moves
        frame[
            200:400, 300 + offset: 500 + offset, 1
        ] = 255  # Green rectangle that moves

        frames.append(frame)

    # Memory after frame generation
    after_frames_memory = get_memory_usage()
    logger.info(
        f"Memory after frame generation: {after_frames_memory:.2f} MB (delta: {after_frames_memory - after_init_memory:.2f} MB)"
    )

    # Process frames and measure performance
    logger.info("Processing frames...")
    frame_times = []
    frame_memory = []

    for i, frame in enumerate(frames):
        frame_start = time.time()
        result = detector.record(screen=frame)
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        # Calculate memory usage of the result frames if they exist
        frames = result.get("frames", [])
        result_memory = (sum(f.nbytes if hasattr(f, "nbytes")
                             else 0 for f in frames) / 1024 / 1024)  # MB
        frame_memory.append(result_memory)

        # Log the results with animation and voice detection status
        animation_detected = result.get("animation_detected", False)
        voice_detected = result.get("voice_detected", False)
        logger.info(
            f"Frame {i+1}/{frame_count} processed in {frame_time:.4f}s, result memory: {result_memory:.2f} MB, animation: {animation_detected}, voice: {voice_detected}"
        )

    # Final memory usage
    final_memory = get_memory_usage()

    # Performance summary
    logger.info("\n--- Performance Summary ---")
    logger.info(
        f"Average frame processing time: {sum(frame_times) / len(frame_times):.4f} seconds"
    )
    logger.info(f"Min frame processing time: {min(frame_times):.4f} seconds")
    logger.info(f"Max frame processing time: {max(frame_times):.4f} seconds")
    logger.info(
        f"Total memory usage: {final_memory:.2f} MB (delta from start: {final_memory - initial_memory:.2f} MB)"
    )
    logger.info(
        f"Average result frame memory: {sum(frame_memory) / len(frame_memory):.2f} MB"
    )

    # Check if animation was detected
    animation_detected = any(frame_time > 0.1 for frame_time in frame_times)
    logger.info(f"Animation detection triggered: {animation_detected}")

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    logger.info(f"CPU usage: {cpu_percent:.1f}%")

    logger.info("Performance test completed")


if __name__ == "__main__":
    main()
