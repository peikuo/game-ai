#!/usr/bin/env python3
"""
Isolated test for voice monitoring functionality.
This script tests only the voice detection and recording capabilities.
"""

import argparse
import logging
import os
import sys
import threading
import time
import wave
from typing import Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Voice detection imports
try:
    import pyaudio
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    logger.warning("WebRTCVAD not available. Voice detection will be disabled.")
    VAD_AVAILABLE = False


class VoiceMonitor:
    """Simple class to test voice monitoring functionality."""

    def __init__(self, config=None):
        """Initialize the voice monitor with configuration."""
        self.config = config or {}
        
        # Voice detection settings
        self.vad_mode = self.config.get("vad_mode", 1)
        self.vad_sample_rate = 16000  # WebRTCVAD requires 8000, 16000, 32000, or 48000 Hz
        self.vad_frame_duration = 30  # ms
        self.voice_silence_threshold = self.config.get("voice_silence_threshold", 1.0)
        self.min_consecutive_voice_frames = self.config.get("min_consecutive_voice_frames", 3)
        
        # State variables
        self.audio_frames = []
        self.voice_detected = False
        self.audio_recorded = False
        self.is_recording_audio = False
        self.stop_voice_detection = threading.Event()
        
        # Output file
        timestamp = int(time.time())
        self.temp_audio_dir = self.config.get("temp_audio_dir", "temp_audio_test")
        self.temp_audio_file = os.path.join(
            self.temp_audio_dir, f"audio_{timestamp}.wav"
        )
        
        # Initialize VAD
        self._init_vad()

    def _init_vad(self):
        """Initialize the Voice Activity Detection module."""
        if not VAD_AVAILABLE:
            logger.warning("Voice detection dependencies not available")
            self.vad = None
            return False

        try:
            logger.info(f"Initializing WebRTCVAD with mode {self.vad_mode}")
            self.vad = webrtcvad.Vad(self.vad_mode)
            logger.info(f"Voice detection initialized with sample rate {self.vad_sample_rate}Hz")
            logger.info(f"Audio will be saved to {self.temp_audio_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            self.vad = None
            return False

    def monitor_voice(self, max_duration=10.0):
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
            
            # Log available audio devices to help with debugging
            logger.info("Available audio input devices:")
            info = audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    logger.info(f"Device {i}: {device_info.get('name')}")
            
            # Use specified device index if provided in config
            device_index = self.config.get("audio_device_index", None)
            if device_index is not None:
                logger.info(f"Using specified audio device index: {device_index}")

            # Open audio stream with specified device if provided
            stream_kwargs = {
                "format": format,
                "channels": channels,
                "rate": rate,
                "input": True,
                "frames_per_buffer": chunk,
            }
            
            if device_index is not None:
                stream_kwargs["input_device_index"] = device_index
                
            stream = audio.open(**stream_kwargs)

            # Start monitoring
            start_time = time.time()
            last_voice_time = 0
            silence_duration = 0
            voice_active = False
            self.audio_frames = []
            self.is_recording_audio = True
            self.audio_recorded = False
            
            # For debugging: record a few seconds regardless of voice detection
            debug_record_seconds = self.config.get("debug_record_seconds", 0)
            if debug_record_seconds > 0:
                logger.info(f"Debug mode: Recording {debug_record_seconds} seconds regardless of voice detection")

            logger.info("Voice monitoring started")
            
            # Track consecutive voice frames for more reliable detection
            consecutive_voice_frames = 0
            min_consecutive_frames = self.config.get("min_consecutive_voice_frames", 3)
            
            # Keep track of volume levels for debugging
            volume_levels = []

            while not self.stop_voice_detection.is_set() and (
                time.time() - start_time < max_duration
            ):
                try:
                    # Read audio frame
                    frame = stream.read(chunk, exception_on_overflow=False)
                    
                    # Calculate audio volume for debugging
                    if len(frame) > 0:
                        audio_data = np.frombuffer(frame, dtype=np.int16)
                        volume = np.abs(audio_data).mean()
                        volume_levels.append(volume)
                        if len(volume_levels) % 10 == 0:  # Log every 10 frames
                            logger.debug(f"Current audio volume: {volume}")

                    # Always store audio frames for processing later
                    self.audio_frames.append(frame)

                    # Check if frame contains voice
                    try:
                        is_speech = self.vad.is_speech(frame, rate)

                        if is_speech:
                            consecutive_voice_frames += 1
                            
                            # Only consider it voice if we have enough consecutive frames
                            if consecutive_voice_frames >= min_consecutive_frames and not voice_active:
                                logger.info(f"Voice detected after {consecutive_voice_frames} consecutive frames")
                                voice_active = True
                                self.voice_detected = True

                            last_voice_time = time.time()
                            silence_duration = 0
                        else:
                            consecutive_voice_frames = 0
                            
                            if voice_active:
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
                        logger.error(f"Error in voice detection frame analysis: {e}")
                        
                except IOError as e:
                    logger.error(f"IOError reading from audio stream: {e}")
                    # Try to recover by sleeping briefly
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error reading audio frame: {e}")
                    
            # For debug recording mode, always mark as detected if we recorded enough frames
            if debug_record_seconds > 0 and len(self.audio_frames) > 0:
                self.voice_detected = True

            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save the recorded audio
            if (self.voice_detected or debug_record_seconds > 0) and len(self.audio_frames) > 0:
                self._save_audio_frames()
                self.audio_recorded = True
                logger.info(f"Audio saved to {self.temp_audio_file}")
                
                # Log audio statistics
                if volume_levels:
                    logger.info(f"Audio volume stats - Min: {min(volume_levels)}, Max: {max(volume_levels)}, Avg: {sum(volume_levels)/len(volume_levels):.2f}")

            self.is_recording_audio = False
            logger.info(
                f"Voice monitoring completed after {time.time() - start_time:.2f}s"
            )

            return {
                "voice_detected": self.voice_detected,
                "audio_recorded": self.audio_recorded,
                "audio_file": self.temp_audio_file if self.audio_recorded else None,
                "duration": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Error in voice monitoring: {e}")
            self.is_recording_audio = False
            return {
                "voice_detected": False,
                "audio_recorded": False,
                "error": str(e),
            }

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
                wf.setsampwidth(2)  # 2 bytes for 'int16' format
                wf.setframerate(self.vad_sample_rate)
                wf.writeframes(b"".join(self.audio_frames))
                
            return True
        except Exception as e:
            logger.error(f"Error saving audio frames: {e}")
            return False


def test_voice_detection_dependencies():
    """Test if the required dependencies for voice detection are available."""
    print("\n=== Testing Voice Detection Dependencies ===")
    print(f"WebRTCVAD available: {VAD_AVAILABLE}")
    
    try:
        import faster_whisper
        print(f"Faster-Whisper available: True")
    except ImportError:
        print(f"Faster-Whisper available: False")


def test_audio_devices():
    """Test and list available audio input devices."""
    print("\n=== Testing Audio Input Devices ===")
    
    if not VAD_AVAILABLE:
        print("PyAudio not available, cannot list audio devices")
        return
    
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        print("Available audio input devices:")
        for i in range(num_devices):
            device_info = audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"Device {i}: {device_info.get('name')}")
                print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
        
        audio.terminate()
    except Exception as e:
        print(f"Error listing audio devices: {e}")


def test_vad_sensitivity():
    """Test different VAD sensitivity levels."""
    print("\n=== Testing VAD Sensitivity Levels ===")
    
    if not VAD_AVAILABLE:
        print("WebRTCVAD not available, cannot test sensitivity levels")
        return
    
    print("Available VAD modes (0=least aggressive, 3=most aggressive):")
    for mode in range(4):
        try:
            vad = webrtcvad.Vad(mode)
            print(f"Mode {mode} initialized successfully")
        except Exception as e:
            print(f"Mode {mode} initialization failed: {e}")


def test_voice_recording(duration=5, device_index=None, vad_mode=1, debug_mode=False, min_consecutive_frames=3):
    """
    Test voice recording for a specified duration.
    
    Args:
        duration (int): Recording duration in seconds
        device_index (int): Audio device index to use (None = default device)
        vad_mode (int): VAD sensitivity mode (0=least aggressive, 3=most aggressive)
        debug_mode (bool): If True, always record audio regardless of voice detection
        min_consecutive_frames (int): Minimum consecutive voice frames to trigger detection
    """
    print(f"\n=== Testing Voice Recording ({duration} seconds) ===")
    print(f"Device index: {device_index if device_index is not None else 'Default'}, VAD mode: {vad_mode}")
    
    if debug_mode:
        print("DEBUG MODE: Will record audio regardless of voice detection")
    
    if not VAD_AVAILABLE:
        print("WebRTCVAD not available, cannot test voice recording")
        return
    
    # Configure the voice monitor
    config = {
        "vad_mode": vad_mode,
        "audio_device_index": device_index,
        "min_consecutive_voice_frames": min_consecutive_frames,
        "debug_record_seconds": duration if debug_mode else 0,
    }
    
    monitor = VoiceMonitor(config)
    
    print("Please speak into your microphone...")
    result = monitor.monitor_voice(max_duration=duration)
    
    if result:
        print("\nTest Results:")
        print(f"Voice detected: {result.get('voice_detected', False)}")
        print(f"Audio recorded: {result.get('audio_recorded', False)}")
        if result.get('audio_recorded'):
            print(f"Audio saved to: {result.get('audio_file')}")
        print(f"Duration: {result.get('duration', 0):.2f} seconds")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
    else:
        print("Test failed - no result returned")


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(
        description="Test voice detection functionality")
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Duration for voice recording test in seconds",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Audio device index to use for recording",
    )
    parser.add_argument(
        "--skip-recording",
        action="store_true",
        help="Skip the voice recording test",
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="VAD sensitivity mode (0=least aggressive, 3=most aggressive)",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode to record audio regardless of voice detection",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=3,
        help="Minimum consecutive voice frames to trigger detection",
    )
    parser.add_argument(
        "--test-all-devices",
        action="store_true",
        help="Test recording on all available input devices",
    )
    parser.add_argument(
        "--test-all-vad-modes",
        action="store_true",
        help="Test recording with all VAD sensitivity modes",
    )

    args = parser.parse_args()

    # Run dependency tests
    test_voice_detection_dependencies()
    
    # Test audio devices
    test_audio_devices()
    
    # Test VAD sensitivity levels
    test_vad_sensitivity()
    
    # Test all devices if requested
    if args.test_all_devices and VAD_AVAILABLE:
        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"\nTesting device {i}: {device_info.get('name')}")
                test_voice_recording(
                    duration=args.duration,
                    device_index=i,
                    vad_mode=args.vad_mode,
                    debug_mode=args.debug_mode,
                    min_consecutive_frames=args.min_consecutive_frames
                )
        audio.terminate()
    
    # Test all VAD modes if requested
    elif args.test_all_vad_modes:
        for mode in range(4):
            print(f"\nTesting VAD mode {mode}")
            test_voice_recording(
                duration=args.duration,
                device_index=args.device,
                vad_mode=mode,
                debug_mode=args.debug_mode,
                min_consecutive_frames=args.min_consecutive_frames
            )

    # Run standard recording test if not skipped
    elif not args.skip_recording:
        test_voice_recording(args.duration, args.device, 
                            args.vad_mode, 
                            args.debug_mode,
                            args.min_consecutive_frames)


if __name__ == "__main__":
    main()
