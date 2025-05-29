"""
Frame extractor for game AI.

This module provides functionality to detect animations in games,
extract key frames, and process them for analysis by vision models.
"""

import base64
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from src.utils.image_utils import encode_image_to_base64
from src.utils import model_call

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Class for extracting key frames from animations in game screenshots.
    Detects animations, extracts representative frames,
    and sends them to vision models.
    """

    def __init__(self, screenshot_capture, config=None):
        """
        Initialize the FrameExtractor.

        Args:
            screenshot_capture: ScreenshotCapture instance for capturing frames
            config (dict, optional): Configuration for frame extraction
        """
        self.screenshot_capture = screenshot_capture
        self.config = config or {}

        # Frame extraction configuration
        self.max_frames = self.config.get("max_frames", 4)
        # Maximum frames to extract
        self.min_frames = self.config.get("min_frames", 10)
        # Minimum frames to analyze
        self.frame_interval = self.config.get("frame_interval", 0.2)  # seconds
        self.animation_threshold = self.config.get("animation_threshold", 0.05)
        self.max_capture_time = self.config.get(
            "max_capture_time", 10)  # seconds
        self.diff_threshold = self.config.get(
            "diff_threshold", 0.1
        )  # Difference threshold for key frame extraction
        self.max_image_size = self.config.get(
            "max_image_size", (640, 480)
        )  # Max size for API requests

        # Temp directory for storing frame files
        self.temp_dir = Path(tempfile.gettempdir()) / "civ6_ai_frames"
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # No need to initialize model_call as we're using the module directly
        # Note: only Qwen supports video analysis
        
        # Model call initialization is now handled in main.py
        # No need to check for API key here as that's handled by the model_call module

    def detect_and_capture_frames(
            self,
            region_name="full_screen",
            max_frames=None):
        """
        Detect if an animation is occurring and capture frames.

        Args:
            region_name (str): Name of the region to monitor
            max_frames (int, optional): Maximum number of frames to capture

        Returns:
            list: List of captured frames if animation detected, None otherwise
        """
        if max_frames is None:
            max_frames = self.max_frames * 5  # Capture more frames than we'll use

        logger.info(f"Monitoring for animation in region: {region_name}")

        # Initialize variables
        frames = []
        consecutive_changes = 0
        start_time = time.time()
        animation_detected = False

        # Take initial screenshot
        initial_frame = self.screenshot_capture.capture(region_name)
        frames.append(np.array(initial_frame))

        # Monitor for animation
        while (
            time.time() - start_time < self.max_capture_time
            and len(frames) < max_frames
        ):
            # Wait for the frame interval
            time.sleep(self.frame_interval)

            # Capture next frame
            next_frame = self.screenshot_capture.capture(region_name)

            # Check if screen has changed
            if self.screenshot_capture.detect_screen_change(
                threshold=self.animation_threshold
            ):
                consecutive_changes += 1
                frames.append(np.array(next_frame))

                # If we detect consistent changes, it's likely an animation
                if consecutive_changes >= self.min_frames and not animation_detected:
                    logger.info(
                        f"Animation detected after {consecutive_changes} consecutive changes"
                    )
                    animation_detected = True
            else:
                consecutive_changes = 0

                # If animation was previously detected but has now stopped,
                # break the loop
                if animation_detected:
                    logger.info(
                        f"Animation appears to have stopped after {len(frames)} frames"
                    )
                    break

        if animation_detected:
            logger.info(
                f"Captured {len(frames)} frames of animation over {time.time() - start_time:.2f} seconds"
            )
            return frames
        else:
            logger.info("No animation detected")
            return None

    def extract_key_frames(self, frames, max_frames=None):
        """
        Extract key frames from a sequence of frames.

        Args:
            frames (list): List of frames to analyze
            max_frames (int, optional): Maximum number of key frames to extract

        Returns:
            list: List of key frames
        """
        if max_frames is None:
            max_frames = self.max_frames

        if not frames or len(frames) < 2:
            logger.warning("Not enough frames to extract key frames")
            return frames

        # If we only need a few frames, use simple sampling
        if len(frames) <= max_frames:
            return frames

        # Convert frames to grayscale for comparison
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:  # Color image
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            else:  # Already grayscale
                gray_frames.append(frame)

        # Calculate differences between consecutive frames
        frame_diffs = []
        for i in range(1, len(gray_frames)):
            diff = np.sum(cv2.absdiff(gray_frames[i - 1], gray_frames[i])) / (
                gray_frames[i].shape[0] * gray_frames[i].shape[1] * 255.0
            )
            frame_diffs.append((i, diff))

        # Sort frames by difference (descending)
        frame_diffs.sort(key=lambda x: x[1], reverse=True)

        # Always include first and last frame
        key_indices = [0, len(frames) - 1]

        # Add frames with highest differences
        for idx, diff in frame_diffs:
            if idx not in key_indices and diff > self.diff_threshold:
                key_indices.append(idx)
                if len(key_indices) >= max_frames:
                    break

        # If we still don't have enough frames, add evenly spaced frames
        if len(key_indices) < max_frames:
            remaining = max_frames - len(key_indices)
            step = len(frames) // (remaining + 1)
            for i in range(1, remaining + 1):
                idx = i * step
                if idx not in key_indices and idx < len(frames):
                    key_indices.append(idx)

        # Sort indices to maintain temporal order
        key_indices.sort()

        # Extract key frames
        key_frames = [frames[i] for i in key_indices]
        logger.info(
            f"Extracted {len(key_frames)} key frames from {len(frames)} total frames"
        )

        return key_frames

    def resize_frame(self, frame, max_size=None):
        """
        Resize a frame to be within the maximum size limit.

        Args:
            frame (numpy.ndarray): Frame to resize
            max_size (tuple, optional): Maximum (width, height)

        Returns:
            PIL.Image: Resized image
        """
        if max_size is None:
            max_size = self.max_image_size

        # Convert to PIL Image if it's a numpy array
        if isinstance(frame, np.ndarray):
            if frame.shape[2] == 3:  # RGB
                img = Image.fromarray(frame)
            else:  # RGBA or other format
                img = Image.fromarray(frame).convert("RGB")
        else:
            img = frame

        # Check if resizing is needed
        if img.width > max_size[0] or img.height > max_size[1]:
            # Calculate new size while preserving aspect ratio
            ratio = min(max_size[0] / img.width, max_size[1] / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug(
                f"Resized image from {img.width}x{img.height} to {new_size[0]}x{new_size[1]}"
            )
        return img

    def save_frame(self, frame, index=0):
        """
        Save a frame to a temporary file.

        Args:
            frame: Frame to save (numpy array or PIL Image)
            index (int): Frame index

        Returns:
            str: Path to the saved frame
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}_{index}.jpg"
        file_path = str(self.temp_dir / filename)

        # Resize the frame
        img = self.resize_frame(frame)

        # Save the image
        img.save(file_path, format="JPEG", quality=85)
        logger.debug(f"Saved frame to {file_path}")

        return file_path

    def encode_image(self, image_path):
        """
        Encode an image file to base64.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Base64-encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_frames(self, frames, prompt=None):
        """
        Analyze a sequence of frames using the Qwen model.

        Args:
            frames (list): List of frames to analyze
            prompt (str, optional): Custom prompt for analysis

        Returns:
            dict: Analysis results
        """
        # We no longer need to check if model_call is initialized
        # The model_call module will handle errors internally

        if not frames or len(frames) < 1:
            logger.error("No frames provided for analysis")
            return {"error": "No frames provided"}

        # Default prompt if none provided
        if prompt is None:
            prompt = """
            Analyze this sequence of game frames in detail. Describe:
            1. What is happening in the animation sequence
            2. Any important game events or state changes
            3. Any UI elements that appear or change
            4. Any notifications or alerts
            5. The significance of these changes in the game context

            Provide a detailed description of what's happening and its meaning in the game.
            """

        try:
            # Save frames to files and encode them
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = self.save_frame(frame, i)
                frame_paths.append(frame_path)

            # Load images as PIL Images and encode to base64
            base64_frames = []
            for path in frame_paths:
                img = Image.open(path)
                base64_img = encode_image_to_base64(img, max_width=640, optimize=True)
                base64_frames.append(base64_img)
            
            # Use the centralized model call utility for video analysis
            # Use the model_call module directly instead of self.model_call
            result = model_call.call_video_model(base64_frames, prompt)
            
            # Clean up temporary files
            for path in frame_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {path}: {e}")

            if result.get("status") == "success":
                return {
                    "analysis": result.get("response_text", ""),
                    "frame_count": len(frames)
                }
            else:
                logger.error(f"Error in model call: {result.get('error')}")
                return {"error": result.get("error", "Unknown error in model call")}

        except Exception as e:
            logger.exception(f"Error analyzing frames: {e}")
            return {"error": str(e)}

    def process_animation(self, region_name="full_screen", prompt=None):
        """
            Complete process to detect, capture, extract key frames,
        and analyze an animation.

            Args:
                region_name (str): Name of the region to monitor
                prompt (str, optional): Custom prompt for analysis

            Returns:
                dict: Analysis results
        """
        # Detect and capture animation frames
        logger.info(f"Detecting animation in region: {region_name}")
        frames = self.detect_and_capture_frames(region_name)

        if not frames or len(frames) < 2:
            logger.info("No animation detected or insufficient frames captured")
            return {"status": "no_animation", "message": "No animation detected"}

        # Extract key frames
        logger.info(f"Extracting key frames from {len(frames)} captured frames")
        key_frames = self.extract_key_frames(frames)

        if not key_frames or len(key_frames) < 1:
            logger.info("No key frames extracted")
            return {
                "status": "no_key_frames",
                "message": "No key frames could be extracted"
            }

        # Analyze the key frames
        logger.info(f"Analyzing {len(key_frames)} key frames")
        analysis_result = self.analyze_frames(key_frames, prompt)

        if "error" in analysis_result:
            logger.error(f"Error analyzing frames: {analysis_result['error']}")
            return {
                "status": "error",
                "message": f"Error analyzing frames: {analysis_result['error']}"
            }

        return {
            "status": "success",
            "analysis": analysis_result.get("analysis", ""),
            "key_frame_count": len(key_frames),
            "total_frame_count": len(frames)
        }
