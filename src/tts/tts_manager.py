#!/usr/bin/env python
"""
Text-to-Speech manager module for game AI monologues.
Uses Kokoro TTS library to convert text to speech.
"""

import os
import logging
import warnings
import soundfile as sf
import sounddevice as sd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Kokoro after filtering warnings
from kokoro import KModel, KPipeline

# Set up logging
logger = logging.getLogger(__name__)

class TTSManager:
    """
    Manager class for Text-to-Speech functionality.
    Handles the conversion of monologue text to speech.
    """

    # Available voices
    VOICES = {
        'female_1': 'zf_001',  # Chinese female voice 1
        'male_1': 'zm_010',    # Chinese male voice 1
    }
    
    # Default voice
    DEFAULT_VOICE = VOICES['female_1']
    
    # Default sample rate for audio generation
    SAMPLE_RATE = 24000
    
    # Default model repository
    DEFAULT_REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS Manager with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config

        # Handle both full config and audio-only config
        if "audio" in config:
            # Full config with audio section
            self.audio_config = config.get("audio", {})
        else:
            # Directly passed audio config
            self.audio_config = config
            
        logger.info(f"TTS config: {self.audio_config}")
        
        # Extract configuration values with defaults
        self.voice_id = self.audio_config.get("voice", self.DEFAULT_VOICE)
        self.repo_id = self.audio_config.get("tts_model", self.DEFAULT_REPO_ID)
        self.temp_folder = Path(self.audio_config.get("temp_folder", "temp_audio"))
        self.sample_rate = self.audio_config.get("sample_rate", self.SAMPLE_RATE)
        
        # Create temp folder if it doesn't exist
        if not self.temp_folder.exists():
            self.temp_folder.mkdir(parents=True, exist_ok=True)
            
        # Initialize model and pipeline
        self._initialize_tts()
        
    def _initialize_tts(self):
        """Initialize the TTS model and pipeline"""
        try:
            logger.info("Initializing TTS model...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Initialize the TTS model
            self.model = KModel(repo_id=self.repo_id).to(device).eval()
            
            # Helper function for English words in Chinese text
            def en_callable(text):
                # Simple handling of common English words
                return text
                
            # Create Chinese pipeline
            self.zh_pipeline = KPipeline(
                lang_code='z', 
                repo_id=self.repo_id, 
                model=self.model, 
                en_callable=en_callable
            )
            logger.info("TTS model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            self.model = None
            self.zh_pipeline = None

    def is_available(self) -> bool:
        """Check if TTS functionality is available"""
        return self.model is not None and self.zh_pipeline is not None
        
    def _play_audio_stream(self, audio) -> bool:
        """Play audio directly without saving to a file
        
        Args:
            audio: Audio data (PyTorch tensor or NumPy array)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert PyTorch tensor to NumPy array if needed
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().numpy()
            
            # Ensure the audio is in the correct format (float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Play the audio
            sd.play(audio, self.sample_rate)
            sd.wait()
            return True
            
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS to handle newlines and other special characters
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text ready for TTS
        """
        if not text:
            return ""
            
        # Replace multiple newlines with a space
        processed = text.replace('\n\n', '。 ')
        
        # Replace single newlines with a pause marker
        processed = processed.replace('\n', '，')
        
        # Replace any remaining multiple spaces with a single space
        processed = ' '.join(processed.split())
        
        logger.debug(f"Preprocessed text for TTS: '{processed[:50]}...'" if len(processed) > 50 else f"'{processed}'")
        
        return processed

    def speak_monologue(self, text: str, voice_id: Optional[str] = None, cleanup: bool = False) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: The monologue text to speak
            voice_id: Voice ID to use (uses default if not specified)
            cleanup: Whether to delete the audio file after playback (default: False)
            
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
            # Use specified voice or default
            voice = voice_id or self.voice_id
            voice_name = next(
                (k for k, v in self.VOICES.items() if v == voice), 
                "unknown"
            )
            logger.info(f"Speaking monologue with voice: {voice_name}")
            
            # Preprocess text to handle newlines and other special characters
            processed_text = self._preprocess_text(text)
            
            # Process the text using the pipeline
            for result in self.zh_pipeline(processed_text, voice=voice):
                gs, ps, audio = result
                
                # Stream the audio directly without saving to file
                logger.debug("Streaming audio...")
                success = self._play_audio_stream(audio)
                
                # Optionally save to file if needed for debugging
                if self.audio_config.get("save_audio_files", False):
                    temp_audio_file = self.temp_folder / "temp_monologue.wav"
                    sf.write(str(temp_audio_file), audio, self.sample_rate)
                    logger.debug(f"Saved audio to {temp_audio_file}")
                    
                    # Clean up if requested and file was saved
                    if cleanup:
                        logger.debug("Cleaning up audio file after playback")
                        try:
                            if temp_audio_file.exists():
                                os.remove(temp_audio_file)
                        except Exception as e:
                            logger.warning(f"Failed to clean up audio file: {e}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error speaking monologue: {e}")
            return False
            
        return False
