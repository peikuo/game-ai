"""
Screenshot capture functionality for Civilization VI
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from PIL import Image

from src.utils.image_utils import process_screenshot

logger = logging.getLogger(__name__)


class ScreenCapturer:
    """
    Class for capturing screenshots of the Civilization VI game window.
    """

    def __init__(self, config):
        """
        Initialize the ScreenCapturer.

        Args:
            config (dict): Configuration for the screen capturer.
                - region (tuple): (x, y, width, height) of the game window
                - regions (dict): Dictionary of named regions to capture
                - save_path (str): Path to save screenshots
                - save_screenshots (bool): Whether to save screenshots to disk
        """
        self.region = config.get("region", None)
        self.regions = config.get("regions", {})
        self.save_path = Path(config.get("save_path", "screenshots"))
        self.save_screenshots = config.get("save_screenshots", False)
        self.previous_frames = []
        self.frame_diffs = []
        self.min_frames = 5
        self.change_threshold = 0.05

        if self.save_screenshots:
            self.save_path.mkdir(exist_ok=True, parents=True)
            logger.info("Screenshots will be saved to %s", self.save_path)

    def capture(self):
        """
        Capture a screenshot of the game window.

        Returns:
            PIL.Image: The captured screenshot
        """
        try:
            with mss.mss() as sct:
                if self.region:
                    # Capture specific region
                    x, y, width, height = self.region
                    region = {
                        "left": x,
                        "top": y,
                        "width": width,
                        "height": height}
                    sct_img = sct.grab(region)
                else:
                    # Capture full screen (monitor 1 is usually the primary
                    # monitor)
                    sct_img = sct.grab(sct.monitors[1])

                # Convert to PIL Image
                screenshot = Image.frombytes(
                    "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"
                )

                # Process the screenshot using the centralized utility function
                # This handles all optimization in one place - all images are resized to 1280x720
                screenshot = process_screenshot(screenshot)

                # Mark as optimized for downstream components
                screenshot.info['optimized'] = True

                # Log basic screenshot info in debug mode
                if logger.level <= logging.DEBUG:
                    logger.debug(
                        "Captured and optimized screenshot: %dx%d pixels",
                        screenshot.width, screenshot.height
                    )

                if self.save_screenshots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = self.save_path / f"screenshot_{timestamp}.png"
                    screenshot.save(save_path)
                    logger.debug("Screenshot saved: %s", save_path)

                return screenshot

        except Exception as e:
            logger.error("Error capturing screenshot: %s", e)
            return None

    def capture_to_array(self, region_name="full_screen"):
        """
        Capture a screenshot and convert to numpy array.

        Args:
            region_name (str): Name of the region to capture

        Returns:
            numpy.ndarray: The captured screenshot as an array (RGB format)
        """
        try:
            with mss.mss() as sct:
                # Get region coordinates
                region_coords = (
                    self.regions.get(region_name)
                    if region_name != "full_screen"
                    else self.region
                )
                if region_coords:
                    # Capture specific region
                    x, y, width, height = region_coords
                    region = {
                        "left": x,
                        "top": y,
                        "width": width,
                        "height": height}
                    sct_img = sct.grab(region)
                else:
                    # Capture full screen
                    sct_img = sct.grab(sct.monitors[1])

                # Convert to numpy array directly
                img_array = np.array(sct_img)

                # Store frame for change detection
                self._update_frame_history(img_array)

                # Save screenshot if enabled
                if self.save_screenshots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = self.save_path / \
                        f"{region_name}_{timestamp}.png"
                    Image.frombytes(
                        "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"
                    ).save(save_path)
                    logger.debug(f"Screenshot saved to {save_path}")

                return img_array
        except Exception as e:
            logger.error(f"Error capturing screenshot as array: {e}")
            return None

    def capture_to_cv2(self, region_name="full_screen"):
        """
        Capture a screenshot and convert to OpenCV format.

        Args:
            region_name (str): Name of the region to capture

        Returns:
            numpy.ndarray: The captured screenshot in BGR format for OpenCV
        """
        screenshot_array = self.capture_to_array(region_name)
        if screenshot_array is not None:
            # Convert RGB to BGR for OpenCV (requires cv2)
            import cv2
            return cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2BGR)
        return None

    def _update_frame_history(self, new_frame):
        """
        Update the frame history with a new frame and calculate differences.

        Args:
            new_frame (numpy.ndarray): New frame to add to history
        """
        # Convert to grayscale for simpler comparison
        if len(new_frame.shape) == 3:  # Color image
            import cv2
            gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray_frame = new_frame

        # Add to history
        self.previous_frames.append(gray_frame)

        # Keep only the last N frames
        if len(self.previous_frames) > self.min_frames:
            self.previous_frames.pop(0)

        # Calculate differences if we have at least 2 frames
        if len(self.previous_frames) >= 2:
            prev = self.previous_frames[-2]
            curr = self.previous_frames[-1]

            # Ensure frames are the same size
            if prev.shape == curr.shape:
                # Calculate normalized difference
                pixel_count = prev.shape[0] * prev.shape[1]
                # Calculate absolute difference between frames
                diff = np.sum(np.abs(prev - curr)) / (pixel_count * 255.0)
                self.frame_diffs.append(diff)

                # Keep only the last N-1 differences
                if len(self.frame_diffs) > self.min_frames - 1:
                    self.frame_diffs.pop(0)

    def detect_screen_change(self, threshold=None):
        """
        Detect if the screen has changed significantly from the previous frame.

        Args:
            threshold (float, optional): Custom threshold for change detection

        Returns:
            bool: True if the screen has changed significantly, False otherwise
        """
        if not self.frame_diffs:
            return False

        # Use custom threshold if provided, otherwise use configured threshold
        thresh = threshold if threshold is not None else self.change_threshold

        # Check if the most recent difference exceeds the threshold
        return self.frame_diffs[-1] > thresh

    def wait_for_screen_change(
        self, timeout=30, check_interval=0.5, region_name="full_screen"
    ):
        """
        Wait for the screen to change significantly.

        Args:
            timeout (float): Maximum time to wait in seconds
            check_interval (float): Time between checks in seconds
            region_name (str): Name of the region to capture

        Returns:
            bool: True if screen changed, False if timeout occurred
        """
        start_time = time.time()

        # Take initial screenshot
        self.capture_to_array(region_name)

        while time.time() - start_time < timeout:
            # Wait for the specified interval
            time.sleep(check_interval)

            # Take another screenshot
            self.capture_to_array(region_name)

            # Check if screen has changed
            if self.detect_screen_change():
                logger.info(
                    "Screen change detected after %.2f seconds",
                    time.time() - start_time
                )
                return True

        logger.warning("Timeout (%ss) waiting for screen change", timeout)
        return False

    def capture_sequence(
            self,
            num_frames=5,
            interval=0.5,
            region_name="full_screen"):
        """
        Capture a sequence of screenshots.

        Args:
            num_frames (int): Number of frames to capture
            interval (float): Time interval between frames in seconds
            region_name (str): Name of the region to capture

        Returns:
            list: List of captured screenshots
        """
        frames = []
        for i in range(num_frames):
            screenshot = self.capture()
            if screenshot:
                frames.append(screenshot)
            time.sleep(interval)

        return frames
