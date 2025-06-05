"""
Scene detection module for game AI.

This module provides functionality to detect scene changes in games,
including visual animations and audio (human voice) detection with
transcription.
"""

import logging
import os
import threading
import time
import wave
from typing import Any, Dict

import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Voice detection imports
try:
    import pyaudio
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    logger.warning(
        "WebRTCVAD not available. Voice detection will be disabled.")
    VAD_AVAILABLE = False

# Speech recognition imports
try:
    import torch
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "faster-whisper not available. Speech-to-text will be disabled.")
    FASTER_WHISPER_AVAILABLE = False


class SceneDetector:
    """Detects animations and voice activity in games.

    This class is responsible for detecting when animations are playing
    and when voice activity is present in the game. It can record audio
    during animations and transcribe the speech to text.

    Attributes:
        config (Dict): Configuration for scene detection
        animation_diff_threshold (float): Threshold for detecting animation
        animation_consecutive_frames (int): Frames needed to confirm animation
        animation_interval (float): Time between animation checks
        voice_detection_enabled (bool): Whether to enable voice detection
        voice_silence_threshold (float): Seconds of silence before stopping
        whisper_model (str): Model size for speech recognition
        whisper_language (str): Language code for speech recognition
        audio_recording_path (str): Directory to save audio recordings
    """

    def __init__(self, config: Dict):
        """Initialize the SceneDetector.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config

        # Animation detection parameters
        self.animation_diff_threshold = config.get(
            "animation_diff_threshold", 0.02)
        self.animation_consecutive_frames = config.get(
            "animation_consecutive_frames", 3
        )
        self.animation_interval = config.get("animation_interval", 0.1)
        self.screenshot_capture = config.get("screenshot_capture", False)
        self.min_frames = config.get(
            "min_frames", 3
        )  # Minimum frames for animation detection

        # Voice detection parameters
        self.voice_detection_enabled = config.get(
            "voice_detection_enabled", True)
        self.voice_silence_threshold = config.get(
            "voice_silence_threshold", 1.0)

        # Speech recognition parameters
        self.whisper_model = config.get("whisper_model", "medium")
        self.whisper_language = config.get("whisper_language", "zh")

        # Audio recording path
        self.audio_recording_path = config.get(
            "audio_recording_path", "temp_audio")
        os.makedirs(self.audio_recording_path, exist_ok=True)

        # Initialize state
        self._reset_state()

        # Initialize threading events
        self.stop_voice_detection = threading.Event()

        # Initialize voice activity detection
        if self.voice_detection_enabled:
            self._init_vad()
            logger.info("Voice Activity Detection initialized with mode 3")

    def _init_vad(self):
        """Initialize voice activity detection."""
        if not VAD_AVAILABLE:
            logger.warning(
                "WebRTCVAD not available. Voice detection will be disabled.")
            self.voice_detection_enabled = False
            self.vad = None
            return

        try:
            # Initialize WebRTCVAD with mode 3 (most aggressive)
            self.vad = webrtcvad.Vad(3)

            # Set up audio parameters for voice detection
            self.vad_sample_rate = 16000  # Hz
            self.vad_frame_duration = 30  # ms
            self.vad_buffer = []
            self.audio_frames = []
            self.voice_detected = False
            self.last_voice_time = 0
            self.temp_audio_file = os.path.join(
                self.audio_recording_path, f"audio_{int(time.time())}.wav"
            )
        except Exception as e:
            logger.error(f"Error initializing VAD: {e}")
            self.voice_detection_enabled = False
            self.vad = None

    def _reset_state(self):
        """Reset the internal state."""
        # Animation detection state
        self.prev_frame = None
        self.frame_diffs = []
        self.previous_frames = []
        self.animation_detected = False
        self.animation_frames = []
        self.consecutive_changes = 0
        self.animation_start_time = None
        self.animation_end_time = None

        # Voice detection state
        self.voice_detected = False
        self.voice_recording = False
        self.voice_recording_thread = None
        self.voice_data = bytearray()
        self.voice_start_time = None
        self.voice_end_time = None
        self.last_voice_time = None

        # Scene state
        self.scene_active = False
        self.scene_finished = False
        self.scene_result = {}

        # Audio processing state
        self.audio_file_path = None
        self.dialogue = None

    def record(self, screen, max_duration=10.0) -> Dict[str, Any]:
        """
        Record and analyze scene changes, including visual animations and audio.

        Args:
            screen: The screen image captured by main.py (numpy array)
            max_duration (float): Maximum duration to record in seconds
            region_name (str): Name of the region to monitor

        Returns:
            dict: Scene detection results including animation and voice status
        """
        logger.info(
            f"Starting scene recording for up to {max_duration} seconds")

        # Start voice detection in a separate thread if enabled
        if self.voice_detection_enabled and self.vad:
            # Check if voice detection thread is already running
            if (
                hasattr(self, "voice_detection_thread")
                and self.voice_detection_thread
                and self.voice_detection_thread.is_alive()
            ):
                # If thread is daemon, check if we need to stop it
                if self.voice_detection_thread.daemon:
                    # Check if we need to stop voice detection
                    if not self.stop_voice_detection.is_set():
                        logger.info("Voice detection already running")
                    else:
                        # Reset and start a new thread
                        self.stop_voice_detection.clear()
                        self.voice_detected = False
                        self.voice_detection_thread = threading.Thread(
                            target=self._monitor_voice, args=(max_duration,)
                        )
                        self.voice_detection_thread.daemon = True
                        self.voice_detection_thread.start()
                        logger.info("Voice detection restarted")
            else:
                # Start a new voice detection thread
                self.stop_voice_detection.clear()
                self.voice_detected = False
                self.voice_detection_thread = threading.Thread(
                    target=self._monitor_voice, args=(max_duration,)
                )
                self.voice_detection_thread.daemon = True
                self.voice_detection_thread.start()
                logger.info("Voice detection started")

        # Animation detection
        animation_result = self._detect_animation(screen, max_duration)

        # Wait for voice detection to complete if it's running
        if (
            self.voice_detection_enabled
            and self.voice_detection_thread
            and self.voice_detection_thread.is_alive()
        ):
            self.stop_voice_detection.set()
            self.voice_detection_thread.join(timeout=1.0)
            logger.info("Voice detection stopped")

        # Process audio if voice was detected
        dialogue = None
        if self.voice_detected and self.audio_recorded:
            dialogue = self._process_audio_to_text()

        # Prepare result
        result = {
            "finished": True,  # Default to finished
            "animation_detected": animation_result.get("animation_detected", False),
            "animation_frames": animation_result.get("frames", []),
            "voice_detected": self.voice_detected,
            "duration": animation_result.get("duration", 0),
            "dialogue": dialogue,
        }

        # If either animation or voice is still active, mark as not finished
        if (
            animation_result.get("animation_detected", False)
            and not animation_result.get("animation_finished", True)
        ) or (self.voice_detected and not self.audio_recorded):
            result["finished"] = False

        logger.info(
            f"Scene recording completed: animation={result['animation_detected']}, voice={result['voice_detected']}, finished={result['finished']}"
        )
        return result

    def _detect_animation(self, screen, max_duration=10.0) -> Dict[str, Any]:
        """
        Detect if an animation is occurring and capture frames.

        Args:
            screen: The screen image captured by main.py (numpy array)
            max_duration (float): Maximum duration to monitor in seconds

        Returns:
            dict: Animation detection results
        """
        logger.info(f"Monitoring for animation in region: full_screen")

        # Initialize variables
        start_time = time.time()
        animation_detected = False
        animation_finished = False

        # Initialize frames list if not already initialized
        if not hasattr(
                self,
                "animation_frames") or self.animation_frames is None:
            self.animation_frames = []

        # Use the provided screen as the current frame
        if screen is not None:
            initial_frame = np.array(screen)
            self.animation_frames.append(initial_frame)
            self.prev_frame = initial_frame
        else:
            logger.error("No screen provided")
            return {
                "animation_detected": False,
                "animation_finished": True,
                "frames": [],
                "duration": 0,
            }

        # Since we're now receiving frames from main.py, we can't monitor for animation in a loop
        # Instead, we'll analyze the single frame we received and compare it to
        # the previous frame

        # Check if screen has changed from previous frame
        if self.prev_frame is not None and self._detect_screen_change(
                np.array(screen)):
            self.consecutive_changes += 1

            # If we detect consistent changes across multiple calls, it's
            # likely an animation
            if self.consecutive_changes >= self.min_frames and not animation_detected:
                logger.info(
                    f"Animation detected after {self.consecutive_changes} consecutive changes"
                )
                animation_detected = True
        else:
            self.consecutive_changes = 0

            # If animation was previously detected but has now stopped
            if animation_detected:
                logger.info(
                    f"Animation appears to have stopped after {len(self.animation_frames)} frames"
                )
                animation_finished = True

        # Update the previous frame for the next comparison
        self.prev_frame = np.array(screen)

        # Store the frame if we're tracking an animation or if this is the
        # first frame
        if animation_detected and not animation_finished:
            self.animation_frames.append(np.array(screen))

        duration = time.time() - start_time

        # Sample key frames if animation is detected and finished
        key_frames = []
        if animation_detected:
            logger.info(
                f"Captured {len(self.animation_frames)} frames of animation over {duration:.2f} seconds"
            )
            if time.time() - start_time >= max_duration:
                logger.info("Animation recording reached maximum duration")
                animation_finished = True
                # Sample key frames when animation is finished
                if len(self.animation_frames) > 0:
                    # Sample at most 5 key frames evenly distributed
                    num_frames = min(5, len(self.animation_frames))
                    indices = np.linspace(
                        0, len(self.animation_frames) - 1, num_frames, dtype=int
                    )
                    key_frames = [self.animation_frames[i] for i in indices]
                    logger.info(
                        f"Sampled {len(key_frames)} key frames from animation")
                # Clear animation frames for next detection
                self.animation_frames = []
            else:
                # Animation is still in progress
                animation_finished = False
        else:
            logger.info("No animation detected")
            animation_finished = True
            # For non-animation cases, keep at least the initial frame if
            # available
            if len(self.animation_frames) > 0:
                key_frames = [self.animation_frames[0]]
            # Clear animation frames for next detection
            self.animation_frames = []

        return {
            "animation_detected": animation_detected,
            "animation_finished": animation_finished,
            "frames": key_frames,  # Return sampled key frames or initial frame
            "duration": time.time() - start_time,
        }

    def _detect_screen_change(self, new_frame, threshold=None) -> bool:
        """
                Detect if the screen has changed significantly from the previous frame.
        {{ ... }}

                Args:
                    new_frame (numpy.ndarray): New frame to compare
                    threshold (float, optional): Custom threshold for change detection

                Returns:
                    bool: True if the screen has changed significantly, False otherwise
        """
        # Convert to grayscale for simpler comparison using pure NumPy
        # (avoiding OpenCV)
        if len(new_frame.shape) == 3:  # Color image
            # Use simple averaging for grayscale conversion
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
                diff = np.sum(np.abs(prev.astype(float) -
                              curr.astype(float))) / (pixel_count * 255.0)
                self.frame_diffs.append(diff)

                # Keep only the last N-1 differences
                if len(self.frame_diffs) > self.min_frames - 1:
                    self.frame_diffs.pop(0)

                # Use custom threshold if provided, otherwise use configured
                # threshold
                thresh = (
                    threshold
                    if threshold is not None
                    else self.animation_diff_threshold
                )

                # Check if the most recent difference exceeds the threshold
                return diff > thresh

        return False

    def _monitor_voice(self, max_duration=10.0):
        """
        Monitor for human voice activity and record audio when detected.

        Args:
            max_duration (float): Maximum duration to monitor in seconds
        """
        if not VAD_AVAILABLE or not self.vad:
            logger.error("Voice detection dependencies not available")
            return

        try:
            # Initialize PyAudio
            audio = pyaudio.PyAudio()

            # Parameters for audio recording
            format = pyaudio.paInt16
            channels = 1
            rate = self.vad_sample_rate
            chunk = int(rate * self.vad_frame_duration / 1000)

            # Open audio stream
            stream = audio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )

            # Start monitoring
            start_time = time.time()
            last_voice_time = 0
            silence_duration = 0
            voice_active = False
            self.audio_frames = []
            self.is_recording_audio = True
            self.audio_recorded = False

            logger.info("Voice monitoring started")

            while not self.stop_voice_detection.is_set() and (
                time.time() - start_time < max_duration
            ):
                # Read audio frame
                frame = stream.read(chunk, exception_on_overflow=False)

                # Always store audio frames for processing later
                self.audio_frames.append(frame)

                # Check if frame contains voice
                try:
                    is_speech = self.vad.is_speech(frame, rate)

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
                            logger.info(
                                f"Voice stopped after {silence_duration:.2f}s of silence"
                            )
                            voice_active = False

                            # If we've detected voice and then silence, we can
                            # stop recording
                            if (self.voice_detected and time.time() - start_time >
                                    2.0):  # Ensure at least 2 seconds of audio
                                logger.info("Voice recording complete")
                                break
                except Exception as e:
                    logger.error(f"Error in voice detection: {e}")

            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save the recorded audio
            if self.voice_detected and len(self.audio_frames) > 0:
                self._save_audio_frames()
                self.audio_recorded = True
                logger.info(f"Audio saved to {self.temp_audio_file}")

            self.is_recording_audio = False
            logger.info(
                f"Voice monitoring completed after {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error in voice monitoring: {e}")
            self.is_recording_audio = False

    def _save_audio_frames(self):
        """
        Save recorded audio frames to a WAV file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.temp_audio_file), exist_ok=True)

            # Save as WAV file
            with wave.open(self.temp_audio_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for 'int16'
                wf.setframerate(self.vad_sample_rate)
                wf.writeframes(b"".join(self.audio_frames))

            logger.info(f"Audio saved to {self.temp_audio_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False

    def _process_audio_to_text(self):
        """
        Process recorded audio for transcription.
        Uses faster-whisper for speech-to-text conversion.

        Returns:
            dict: Transcription results
        """

        if not os.path.exists(self.temp_audio_file):
            logger.error(f"Audio file not found: {self.temp_audio_file}")
            return None

        # Use faster-whisper for transcription
        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(
                    f"Processing audio with faster-whisper (model: {self.whisper_model})"
                )

                # Initialize model
                compute_type = "float16" if torch.cuda.is_available() else "int8"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = WhisperModel(
                    self.whisper_model,
                    device=device,
                    compute_type=compute_type)

                # Transcribe
                segments, info = model.transcribe(
                    self.temp_audio_file,
                    language=self.whisper_language,
                    beam_size=5,
                    word_timestamps=True,
                )

                # Process segments
                segments_list = list(segments)

                # Basic transcription
                dialogue = [
                    {"speaker": "UNKNOWN", "text": segment.text}
                    for segment in segments_list
                ]

                # Try to group consecutive segments
                if len(dialogue) > 1:
                    grouped_dialogue = []
                    current_text = dialogue[0]["text"]

                    for i in range(1, len(dialogue)):
                        # If time gap is small, consider it the same utterance
                        if (
                            i < len(segments_list)
                            and segments_list[i].start - segments_list[i - 1].end < 1.0
                        ):
                            current_text += " " + dialogue[i]["text"]
                        else:
                            grouped_dialogue.append(
                                {"speaker": "UNKNOWN", "text": current_text.strip()}
                            )
                            current_text = dialogue[i]["text"]

                    # Add the last segment
                    if current_text:
                        grouped_dialogue.append(
                            {"speaker": "UNKNOWN", "text": current_text.strip()}
                        )

                    dialogue = grouped_dialogue

                logger.info(
                    f"faster-whisper transcription completed with {len(dialogue)} segments"
                )
                return {
                    "dialogue": dialogue,
                    "raw_segments": [
                        {"text": s.text, "start": s.start, "end": s.end}
                        for s in segments_list
                    ],
                }

            except Exception as e:
                logger.error(f"Error in faster-whisper processing: {e}")

        logger.error("Speech-to-text failed")
        return None

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
