#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS Manager module for text-to-speech functionality.
Uses kokoro TTS library with direct NumPy-based audio playback.
Compatible with NumPy 2.x and PyTorch 2.7.1+.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

# Configure logging
logger = logging.getLogger(__name__)


class TTSManager:
    """
    Manages text-to-speech functionality using kokoro TTS library.
    Provides direct NumPy-based audio playback without intermediate file saving.
    """

    # Voice IDs for different speakers
    VOICES = {
        "female": "zf_001",
        "male": "zm_001",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS manager.

        Args:
            config: Configuration dictionary with TTS settings
        """
        self.config = config
        self.model = None
        self.zh_pipeline = None
        self.voice_id = config.get(
            "voice", "zf_001")  # Default to female voice
        self.sample_rate = config.get("sample_rate", 24000)
        self.temp_folder = config.get("temp_folder", "temp_audio")
        self.model_name = config.get("tts_model", "hexgrad/Kokoro-82M-v1.1-zh")

        # Create temp folder if it doesn't exist
        os.makedirs(self.temp_folder, exist_ok=True)

        # Initialize TTS model
        self._initialize_model()

    def _initialize_model(self) -> bool:
        """
        Initialize the TTS model.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            from kokoro import KModel, KPipeline

            logger.info(f"Initializing TTS model {self.model_name}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Initialize the TTS model
            self.model = KModel(repo_id=self.model_name).to(device).eval()

            # Helper function for English words in Chinese text
            def en_callable(text):
                # Simple handling of common English words
                return text

            # Create Chinese pipeline
            self.zh_pipeline = KPipeline(
                lang_code="z",
                repo_id=self.model_name,
                model=self.model,
                en_callable=en_callable,
            )

            logger.info("TTS model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            self.model = None
            self.zh_pipeline = None
            return False

    def is_available(self) -> bool:
        """
        Check if TTS functionality is available.

        Returns:
            bool: True if TTS is available, False otherwise
        """
        return self.zh_pipeline is not None and self.model is not None

    def play_audio(
            self,
            audio_tensor,
            sample_rate: Optional[int] = None) -> bool:
        """
        Play audio directly using NumPy and sounddevice.

        Args:
            audio_tensor: PyTorch tensor containing audio data
            sample_rate: Sample rate of the audio (defaults to self.sample_rate)

        Returns:
            bool: True if playback was successful, False otherwise
        """
        try:
            import numpy as np
            import sounddevice as sd

            # Use instance sample rate if not specified
            sr = sample_rate or self.sample_rate

            # Convert PyTorch tensor to NumPy array if needed
            if torch.is_tensor(audio_tensor):
                audio = audio_tensor.detach().cpu().numpy()
            else:
                audio = audio_tensor

            # Ensure the audio is in the correct format (float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Play the audio
            sd.play(audio, sr)
            sd.wait()
            logger.info("Audio playback completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False

    def speak_monologue(
        self, text: str, voice_id: Optional[str] = None, cleanup: bool = False
    ) -> bool:
        """
        Convert text to speech and play it.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (defaults to self.voice_id)
            cleanup: Whether to clean up temporary files

        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.debug("No monologue text provided")
            return False

        if not self.is_available():
            logger.warning("TTS functionality is not available")
            return False

        try:
            voice = voice_id or self.voice_id
            voice_name = next(
                (k for k, v in self.VOICES.items() if v == voice), "unknown"
            )
            logger.info(
                f"Speaking monologue with voice: {voice_name} ({voice})")

            # Preprocess text (remove excessive whitespace, etc.)
            text = " ".join(text.split())

            # Generate audio
            audio_tensor = None
            for result in self.zh_pipeline(text, voice=voice):
                gs, ps, audio = result
                audio_tensor = audio
                break

            if audio_tensor is None:
                logger.error("Failed to generate audio tensor")
                return False

            # Print tensor shape and type information for debugging
            logger.debug(f"Audio tensor type: {type(audio_tensor)}")
            logger.debug(f"Audio tensor shape: {audio_tensor.shape}")
            logger.debug(f"Audio tensor dtype: {audio_tensor.dtype}")

            # Play audio directly
            success = self.play_audio(audio_tensor, self.sample_rate)

            return success
        except Exception as e:
            logger.error(f"Error in speak_monologue: {e}")
            return False
