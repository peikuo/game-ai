"""
Scene detection module for game AI.

This module provides functionality to detect scene changes in games,
including visual animations and audio (human voice) detection.
"""

import logging
import time
import numpy as np
import threading
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

# Voice detection imports
try:
    import pyaudio
    import webrtcvad
    VOICE_DETECTION_AVAILABLE = True
except ImportError:
    VOICE_DETECTION_AVAILABLE = False
    logging.warning("Voice detection dependencies not available. Install with: pip install pyaudio webrtcvad")

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Class for detecting scene changes in games, including visual animations and audio.
    """

    def __init__(self, config=None, screenshot_capture=None):
        """
        Initialize the SceneDetector.

        Args:
            config (dict, optional): Configuration for scene detection
            screenshot_capture: ScreenshotCapture instance for capturing frames
        """
        self.config = config or {}
        self.screenshot_capture = screenshot_capture
        
        # Animation detection configuration
        self.max_frames = self.config.get("max_frames", 4)
        self.min_frames = self.config.get("min_frames", 10)
        self.frame_interval = self.config.get("frame_interval", 0.2)  # seconds
        self.animation_threshold = self.config.get("animation_threshold", 0.05)
        self.max_capture_time = self.config.get("max_capture_time", 10)  # seconds
        
        # Voice detection configuration
        self.voice_detection_enabled = self.config.get("voice_detection_enabled", VOICE_DETECTION_AVAILABLE)
        self.vad_mode = self.config.get("vad_mode", 3)  # 0-3, 3 is most aggressive
        self.vad_frame_duration = self.config.get("vad_frame_duration", 30)  # ms (10, 20, or 30)
        self.vad_sample_rate = self.config.get("vad_sample_rate", 16000)  # Hz
        self.voice_silence_threshold = self.config.get("voice_silence_threshold", 1.0)  # seconds
        
        # State variables
        self.previous_frames = []
        self.frame_diffs = []
        self.animation_detected = False
        self.voice_detected = False
        self.voice_detection_thread = None
        self.stop_voice_detection = threading.Event()
        
        # Initialize voice detector if available
        self.vad = None
        if self.voice_detection_enabled and VOICE_DETECTION_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(self.vad_mode)
                logger.info(f"Voice Activity Detection initialized with mode {self.vad_mode}")
            except Exception as e:
                logger.error(f"Error initializing voice detector: {e}")
                self.voice_detection_enabled = False

    def record(self, max_duration=10.0, region_name="full_screen") -> Dict[str, Any]:
        """
        Record and analyze scene changes, including visual animations and audio.
        
        Args:
            max_duration (float): Maximum duration to record in seconds
            region_name (str): Name of the region to monitor
            
        Returns:
            dict: Scene detection results including animation and voice status
        """
        logger.info(f"Starting scene recording for up to {max_duration} seconds")
        
        # Start voice detection in a separate thread if enabled
        if self.voice_detection_enabled and self.vad:
            self.stop_voice_detection.clear()
            self.voice_detected = False
            self.voice_detection_thread = threading.Thread(
                target=self._monitor_voice,
                args=(max_duration,)
            )
            self.voice_detection_thread.daemon = True
            self.voice_detection_thread.start()
            logger.info("Voice detection started")
        
        # Monitor for animation
        animation_result = self._detect_animation(region_name, max_duration)
        
        # Wait for voice detection to complete if it's running
        if self.voice_detection_enabled and self.voice_detection_thread and self.voice_detection_thread.is_alive():
            self.stop_voice_detection.set()
            self.voice_detection_thread.join(timeout=1.0)
            logger.info("Voice detection stopped")
        
        # Prepare result
        result = {
            "finished": True,  # Default to finished
            "animation_detected": animation_result.get("animation_detected", False),
            "animation_frames": animation_result.get("frames", []),
            "voice_detected": self.voice_detected,
            "duration": animation_result.get("duration", 0),
        }
        
        # If either animation or voice is still active, mark as not finished
        if (animation_result.get("animation_detected", False) and not animation_result.get("animation_finished", True)) or \
           (self.voice_detected and not animation_result.get("voice_finished", True)):
            result["finished"] = False
        
        logger.info(f"Scene recording completed: animation={result['animation_detected']}, voice={result['voice_detected']}, finished={result['finished']}")
        return result

    def _detect_animation(self, region_name="full_screen", max_duration=10.0) -> Dict[str, Any]:
        """
        Detect if an animation is occurring and capture frames.
        
        Args:
            region_name (str): Name of the region to monitor
            max_duration (float): Maximum duration to monitor in seconds
            
        Returns:
            dict: Animation detection results
        """
        logger.info(f"Monitoring for animation in region: {region_name}")
        
        # Initialize variables
        frames = []
        consecutive_changes = 0
        start_time = time.time()
        animation_detected = False
        animation_finished = False
        
        # Take initial screenshot
        if self.screenshot_capture:
            initial_frame = self.screenshot_capture.capture(region_name)
            if initial_frame:
                frames.append(np.array(initial_frame))
        else:
            logger.error("Screenshot capture not available")
            return {
                "animation_detected": False,
                "animation_finished": True,
                "frames": [],
                "duration": 0
            }
        
        # Monitor for animation
        while (time.time() - start_time < max_duration):
            # Wait for the frame interval
            time.sleep(self.frame_interval)
            
            # Capture next frame
            next_frame = self.screenshot_capture.capture(region_name)
            if next_frame is None:
                continue
                
            # Check if screen has changed
            if self._detect_screen_change(np.array(next_frame)):
                consecutive_changes += 1
                frames.append(np.array(next_frame))
                
                # If we detect consistent changes, it's likely an animation
                if consecutive_changes >= self.min_frames and not animation_detected:
                    logger.info(f"Animation detected after {consecutive_changes} consecutive changes")
                    animation_detected = True
            else:
                consecutive_changes = 0
                
                # If animation was previously detected but has now stopped,
                # break the loop
                if animation_detected:
                    logger.info(f"Animation appears to have stopped after {len(frames)} frames")
                    animation_finished = True
                    break
        
        duration = time.time() - start_time
        
        if animation_detected:
            logger.info(f"Captured {len(frames)} frames of animation over {duration:.2f} seconds")
            if time.time() - start_time >= max_duration:
                logger.info("Animation recording reached maximum duration")
                animation_finished = False
        else:
            logger.info("No animation detected")
            animation_finished = True
        
        return {
            "animation_detected": animation_detected,
            "animation_finished": animation_finished,
            "frames": frames,
            "duration": duration
        }

    def _detect_screen_change(self, new_frame, threshold=None) -> bool:
        """
        Detect if the screen has changed significantly from the previous frame.
        
        Args:
            new_frame (numpy.ndarray): New frame to compare
            threshold (float, optional): Custom threshold for change detection
            
        Returns:
            bool: True if the screen has changed significantly, False otherwise
        """
        # Convert to grayscale for simpler comparison
        if len(new_frame.shape) == 3:  # Color image
            try:
                import cv2
                gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
            except ImportError:
                # Fallback to simple averaging if OpenCV is not available
                gray_frame = np.mean(new_frame, axis=2).astype(np.uint8)
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
                diff = np.sum(np.abs(prev.astype(float) - curr.astype(float))) / (pixel_count * 255.0)
                self.frame_diffs.append(diff)
                
                # Keep only the last N-1 differences
                if len(self.frame_diffs) > self.min_frames - 1:
                    self.frame_diffs.pop(0)
                
                # Use custom threshold if provided, otherwise use configured threshold
                thresh = threshold if threshold is not None else self.animation_threshold
                
                # Check if the most recent difference exceeds the threshold
                return diff > thresh
        
        return False

    def _monitor_voice(self, max_duration=10.0):
        """
        Monitor for human voice activity.
        
        Args:
            max_duration (float): Maximum duration to monitor in seconds
        """
        if not VOICE_DETECTION_AVAILABLE or not self.vad:
            logger.error("Voice detection dependencies not available")
            return
        
        try:
            # Initialize PyAudio
            audio = pyaudio.PyAudio()
            
            # Open audio stream
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.vad_sample_rate,
                input=True,
                frames_per_buffer=int(self.vad_sample_rate * self.vad_frame_duration / 1000)
            )
            
            # Start monitoring
            start_time = time.time()
            last_voice_time = 0
            silence_duration = 0
            voice_active = False
            
            logger.info("Voice monitoring started")
            
            while not self.stop_voice_detection.is_set() and (time.time() - start_time < max_duration):
                # Read audio frame
                frame = stream.read(int(self.vad_sample_rate * self.vad_frame_duration / 1000), exception_on_overflow=False)
                
                # Check if frame contains voice
                try:
                    is_speech = self.vad.is_speech(frame, self.vad_sample_rate)
                    
                    if is_speech:
                        if not voice_active:
                            logger.info("Voice detected")
                            voice_active = True
                            self.voice_detected = True
                        
                        last_voice_time = time.time()
                        silence_duration = 0
                    elif voice_active:
                        # Calculate silence duration
                        silence_duration = time.time() - last_voice_time
                        
                        # If silence exceeds threshold, consider voice stopped
                        if silence_duration > self.voice_silence_threshold:
                            logger.info(f"Voice stopped after {silence_duration:.2f}s of silence")
                            voice_active = False
                except Exception as e:
                    logger.error(f"Error in voice detection: {e}")
            
            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            logger.info(f"Voice monitoring completed after {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in voice monitoring: {e}")

    def extract_key_frames(self, frames, max_frames=None):
        """
        Extract key frames from a sequence of frames.
        
        Args:
            frames (list): List of frames to extract key frames from
            max_frames (int, optional): Maximum number of key frames to extract
            
        Returns:
            list: List of key frames
        """
        if not frames:
            return []
        
        if max_frames is None:
            max_frames = self.max_frames
        
        # If we have fewer frames than max_frames, return all frames
        if len(frames) <= max_frames:
            return frames
        
        # Simple approach: take evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]
