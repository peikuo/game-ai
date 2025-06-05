# Scene Detection with Voice Transcription

This document explains the scene detection system with voice transcription capabilities implemented in the game AI project.

## Overview

The scene detection system is designed to detect animations and voice activity in games, record audio during these scenes, and transcribe spoken dialogue. This is particularly useful for:

1. Detecting when cutscenes or animations are playing
2. Recording and transcribing NPC dialogue
3. Capturing key frames from animations
4. Providing structured dialogue data for game analysis and decision-making

## Features

- **Animation Detection**: Detects significant visual changes in the game screen
- **Voice Activity Detection**: Uses WebRTCVAD to detect human speech
- **Audio Recording**: Records audio during detected scenes
- **Speech-to-Text Transcription**: Converts recorded audio to text using faster-whisper
- **Key Frame Extraction**: Captures important frames during animations

## Dependencies

The system requires the following dependencies:

```bash
pip install numpy pyaudio webrtcvad torch faster-whisper soundfile pydub --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

## Usage

### Basic Usage

```python
from src.game_interface.scene_detect import SceneDetector

# Initialize the detector with configuration
config = {
    "animation_diff_threshold": 0.02,      # Sensitivity for detecting animation
    "animation_consecutive_frames": 3,      # Frames needed to confirm animation
    "animation_interval": 0.1,             # Time between animation checks
    "voice_detection_enabled": True,       # Enable voice detection
    "voice_silence_threshold": 1.0,        # Seconds of silence to stop recording
    "whisper_model": "tiny",              # Speech recognition model size
    "whisper_language": "zh",             # Default language (Chinese)
    "audio_recording_path": "temp_audio"   # Where to save audio recordings
}

detector = SceneDetector(config)

# In your game loop:
screenshot = capture_screenshot()  # Get current game screen
result = detector.record(screenshot)

# Check if scene is finished
if result.get("finished", False):
    # Process scene results
    if result.get("dialogue"):
        for segment in result["dialogue"]["dialogue"]:
            speaker = segment["speaker"]
            text = segment["text"]
            print(f"{speaker}: {text}")
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `animation_diff_threshold` | Threshold for detecting significant frame changes | 0.02 |
| `animation_consecutive_frames` | Number of consecutive frames needed to confirm animation | 3 |
| `animation_interval` | Time between animation detection checks (seconds) | 0.1 |
| `voice_detection_enabled` | Whether to enable voice detection | True |
| `voice_silence_threshold` | Seconds of silence before stopping voice recording | 1.0 |
| `whisper_model` | Model size for faster-whisper ("tiny", "base", "small", "medium", "large") | "medium" |
| `whisper_language` | Language code for transcription | "zh" (Chinese) |
| `audio_recording_path` | Directory to save recorded audio files | "temp_audio" |

### Result Structure

The `record()` method returns a dictionary with the following structure:

```python
{
    "animation_detected": bool,    # Whether animation was detected
    "voice_detected": bool,        # Whether voice was detected
    "finished": bool,              # Whether the scene is finished
    "animation_frames": [],        # List of key frames from the animation
    "dialogue": {                  # Transcription results (if available)
        "dialogue": [
            {
                "speaker": "UNKNOWN",  # Speaker identifier
                "text": "..."          # Transcribed text
            },
            # More dialogue segments...
        ],
        "raw_segments": [...]      # Raw transcription segments with timestamps
    }
}
```

## Examples

See the example scripts in the `examples` directory:

- `test_scene_detect.py`: Simple test script for scene detection
- `scene_detection_example.py`: Integration example with simulated game loop

## Integration with Main Game Loop

To integrate scene detection into the main game loop:

1. Initialize the `SceneDetector` with appropriate configuration
2. In each game frame, pass the current screenshot to `detector.record()`
3. Check if a scene is active or finished
4. Process dialogue and key frames when a scene is finished

See `examples/scene_detection_example.py` for a complete integration example.

## Limitations

- Voice detection requires a microphone and may be affected by background noise
- Speech-to-text accuracy depends on the chosen model size and audio quality
- Speaker diarization (identifying different speakers) is basic and may not always be accurate
- Animation detection is based on frame differences and may need tuning for specific games
