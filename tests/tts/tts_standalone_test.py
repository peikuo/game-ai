#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone TTS test script that doesn't rely on the TTSManager class.
This script directly uses the kokoro TTS library to generate speech
and implements NumPy-free audio playback methods.
"""

import argparse
import logging
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tts_standalone_test")

# TTS Configuration
TTS_CONFIG = {
    "tts_model": "hexgrad/Kokoro-82M-v1.1-zh",
    "voice": "zf_001",
    "temp_folder": "temp_audio",
    "sample_rate": 24000,
}


def initialize_tts_model():
    """Initialize the TTS model directly without TTSManager"""
    try:
        from kokoro import KModel, KPipeline

        logger.info(f"Initializing TTS model {TTS_CONFIG['tts_model']}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Initialize the TTS model
        model = KModel(repo_id=TTS_CONFIG["tts_model"]).to(device).eval()

        # Helper function for English words in Chinese text
        def en_callable(text):
            # Simple handling of common English words
            return text

        # Create Chinese pipeline
        zh_pipeline = KPipeline(
            lang_code="z",
            repo_id=TTS_CONFIG["tts_model"],
            model=model,
            en_callable=en_callable,
        )

        logger.info("TTS model initialized successfully")
        return zh_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {e}")
        return None


def play_audio(audio_tensor, sample_rate=24000):
    """
    Play audio directly using NumPy and sounddevice

    Args:
        audio_tensor: PyTorch tensor containing audio data
        sample_rate: Sample rate of the audio

    Returns:
        bool: True if playback was successful, False otherwise
    """
    try:
        import numpy as np
        import sounddevice as sd

        # Convert PyTorch tensor to NumPy array if needed
        if torch.is_tensor(audio_tensor):
            audio = audio_tensor.detach().cpu().numpy()
        else:
            audio = audio_tensor

        # Ensure the audio is in the correct format (float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Play the audio
        sd.play(audio, sample_rate)
        sd.wait()
        logger.info("Audio playback completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        return False


def test_tts_standalone():
    """Test TTS functionality without relying on TTSManager"""
    # Initialize TTS model
    zh_pipeline = initialize_tts_model()
    if not zh_pipeline:
        logger.error("Failed to initialize TTS model")
        return False

    # Test text (Chinese)
    test_text = "这是一个测试，用于检查文本转语音功能。"
    logger.info(f"Testing TTS with text: '{test_text}'")

    # Generate audio
    try:
        audio_tensor = None
        for result in zh_pipeline(test_text, voice=TTS_CONFIG["voice"]):
            gs, ps, audio = result
            audio_tensor = audio
            break

        if audio_tensor is None:
            logger.error("Failed to generate audio tensor")
            return False

        # Print tensor shape and type information for debugging
        logger.info(f"Audio tensor type: {type(audio_tensor)}")
        logger.info(f"Audio tensor shape: {audio_tensor.shape}")
        logger.info(f"Audio tensor dtype: {audio_tensor.dtype}")

        # Save audio to file
        audio_file = save_audio_to_file(
            audio_tensor, TTS_CONFIG["sample_rate"])
        if not audio_file:
            logger.error("Failed to save audio to file")
            return False

        # Play audio file
        success = play_audio_file(audio_file)
        if success:
            logger.info("✅ TTS test completed successfully")
            return True
        else:
            logger.error("❌ Failed to play audio file")
            return False
    except Exception as e:
        logger.error(f"Error in TTS test: {e}")
        return False


def test_tts_english():
    """Test TTS with English text"""
    # Initialize TTS model
    zh_pipeline = initialize_tts_model()
    if not zh_pipeline:
        logger.error("Failed to initialize TTS model")
        return False

    # Test text (English)
    test_text = "This is a test of the text to speech functionality."
    logger.info(f"Testing TTS with English text: '{test_text}'")

    # Generate audio
    try:
        audio_tensor = None
        for result in zh_pipeline(test_text, voice=TTS_CONFIG["voice"]):
            gs, ps, audio = result
            audio_tensor = audio
            break

        if audio_tensor is None:
            logger.error("Failed to generate audio tensor")
            return False

        # Save audio to file
        audio_file = save_audio_to_file(
            audio_tensor, TTS_CONFIG["sample_rate"])
        if not audio_file:
            logger.error("Failed to save audio to file")
            return False

        # Play audio file
        success = play_audio_file(audio_file)
        if success:
            logger.info("✅ English TTS test completed successfully")
            return True
        else:
            logger.error("❌ Failed to play audio file")
            return False
    except Exception as e:
        logger.error(f"Error in English TTS test: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TTS test")
    parser.add_argument(
        "--english",
        action="store_true",
        help="Test English TTS")
    args = parser.parse_args()

    if args.english:
        test_tts_english()
    else:
        test_tts_standalone()
