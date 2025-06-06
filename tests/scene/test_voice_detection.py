#!/usr/bin/env python3
"""
Test script for voice detection functionality.
This script tests the voice detection capabilities independently from other components.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the SceneDetector class
from src.game_interface.scene_detect import SceneDetector, VAD_AVAILABLE, FASTER_WHISPER_AVAILABLE

def test_voice_detection_dependencies():
    """Test if the required dependencies for voice detection are available."""
    print("\n=== Testing Voice Detection Dependencies ===")
    print(f"WebRTCVAD available: {VAD_AVAILABLE}")
    print(f"Faster-Whisper available: {FASTER_WHISPER_AVAILABLE}")
    
    if not VAD_AVAILABLE:
        print("\nWebRTCVAD is not available. Please install it with:")
        print("pip install webrtcvad pyaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn")
    
    if not FASTER_WHISPER_AVAILABLE:
        print("\nFaster-Whisper is not available. Please install it with:")
        print("pip install faster-whisper --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn")

def test_audio_devices():
    """Test and list available audio input devices."""
    print("\n=== Testing Audio Input Devices ===")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("Available audio input devices:")
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"Device {i}: {device_info.get('name')}")
                print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
        
        p.terminate()
    except Exception as e:
        print(f"Error testing audio devices: {e}")

def test_vad_sensitivity():
    """Test different VAD sensitivity levels."""
    print("\n=== Testing VAD Sensitivity Levels ===")
    if not VAD_AVAILABLE:
        print("WebRTCVAD not available. Skipping sensitivity test.")
        return
    
    try:
        import webrtcvad
        
        print("Available VAD modes (0=least aggressive, 3=most aggressive):")
        for mode in range(4):
            vad = webrtcvad.Vad(mode)
            print(f"Mode {mode} initialized successfully")
    except Exception as e:
        print(f"Error testing VAD sensitivity: {e}")

def test_voice_recording(duration=5, device_index=None, vad_mode=1, debug_mode=False, min_consecutive_frames=3):
    """Test voice recording for a specified duration.
    
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
    
    print("Please speak into your microphone...")
    
    # Create a configuration with custom settings
    config = {
        "voice_detection_enabled": True,
        "voice_silence_threshold": 0.5,  # More sensitive silence detection
        "whisper_model": "tiny",  # Use tiny model for faster testing
        "whisper_language": "en",  # Use English for testing
        "audio_recording_path": "temp_audio_test",
        "vad_mode": vad_mode,
        "min_consecutive_voice_frames": min_consecutive_frames
    }
    
    # If debug mode is enabled, set debug_record_seconds
    if debug_mode:
        config["debug_record_seconds"] = duration
    
    # If device_index is specified, add it to the config
    if device_index is not None:
        config["audio_device_index"] = device_index
    
    # Initialize the SceneDetector
    detector = SceneDetector(config)
    
    # Start voice detection in a separate thread
    detector.stop_voice_detection.clear()
    detector.voice_detected = False
    detector.audio_recorded = False
    
    # Create a dummy screen (black image)
    import numpy as np
    dummy_screen = np.zeros((100, 100, 3), dtype=np.uint8)
    
    print(f"Starting voice detection for {duration} seconds...")
    start_time = time.time()
    
    # Record for the specified duration
    result = detector.record(dummy_screen, max_duration=duration)
    
    print(f"Recording completed in {time.time() - start_time:.2f} seconds")
    print(f"Voice detected: {result['voice_detected']}")
    
    # If voice was detected, process the audio
    if result['voice_detected']:
        print("Voice was detected! Processing audio to text...")
        if result.get('dialogue'):
            print(f"Transcribed text: {result['dialogue']}")
        elif result.get('voice_content'):
            print(f"Voice content: {result['voice_content']}")
        else:
            print("No transcription available")
            
        # Show the audio file path
        if hasattr(detector, 'temp_audio_file') and detector.temp_audio_file:
            print(f"Audio saved to: {detector.temp_audio_file}")
    else:
        print("No voice was detected. Try speaking louder or check your microphone settings.")
    
    return result

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

    # Run the tests
    test_voice_detection_dependencies()
    test_audio_devices()
    test_vad_sensitivity()

    if not args.skip_recording:
        if args.test_all_devices:
            # Test all available audio devices
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                info = p.get_host_api_info_by_index(0)
                num_devices = info.get('deviceCount')
                
                for i in range(num_devices):
                    device_info = p.get_device_info_by_host_api_device_index(0, i)
                    if device_info.get('maxInputChannels') > 0:
                        print(f"\n\nTesting device {i}: {device_info.get('name')}")
                        test_voice_recording(
                            duration=args.duration,
                            device_index=i,
                            vad_mode=args.vad_mode,
                            debug_mode=args.debug_mode,
                            min_consecutive_frames=args.min_consecutive_frames
                        )
                p.terminate()
            except Exception as e:
                print(f"Error testing all devices: {e}")
        elif args.test_all_vad_modes:
            # Test all VAD sensitivity modes
            for mode in range(4):
                print(f"\n\nTesting VAD mode {mode}")
                test_voice_recording(
                    duration=args.duration,
                    device_index=args.device,
                    vad_mode=mode,
                    debug_mode=args.debug_mode,
                    min_consecutive_frames=args.min_consecutive_frames
                )
        else:
            # Run a single test with specified parameters
            test_voice_recording(
                duration=args.duration,
                device_index=args.device,
                vad_mode=args.vad_mode,
                debug_mode=args.debug_mode,
                min_consecutive_frames=args.min_consecutive_frames
            )

    
    if not args.skip_recording:
        test_voice_recording(args.duration, args.device)

if __name__ == "__main__":
    main()
