#!/usr/bin/env python
"""
Test script for reading a file and converting its content to speech using TTS.
This script reads a text file and plays its content using the TTSManager.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from src.tts.tts_manager import TTSManager
from src.utils.config_loader import load_config


# Filter warnings to suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to path so we can import project modules
sys.path.append(str(Path(__file__).parent.parent))


def read_file_content(file_path):
    """Read content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def process_file_with_tts(file_path, voice_id=None, cleanup=True):
    """
    Read a file and convert its content to speech
    
    Args:
        file_path: Path to the text file
        voice_id: Optional voice ID to use
        cleanup: Whether to clean up audio files after playback
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    config = load_config(config_path)
    
    # Initialize TTSManager
    tts_manager = TTSManager(config)
    
    # Check if TTS is available
    if not tts_manager.is_available():
        print("TTS functionality is not available. Please check your configuration.")
        return False
    
    # Read file content
    content = read_file_content(file_path)
    if not content:
        return False
    
    # Process content by chunks if it's too long (optional)
    # TTS might work better with smaller chunks of text
    MAX_CHUNK_SIZE = 5000  # characters per chunk
    
    # Break content into sentences or paragraphs if needed
    chunks = []
    if len(content) > MAX_CHUNK_SIZE:
        # Simple splitting by new lines
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                chunks.append(paragraph.strip())
    else:
        chunks = [content]
    
    # Play each chunk
    for i, chunk in enumerate(chunks):
        print(f"\n--- Playing chunk {i+1}/{len(chunks)} ---")
        # print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        success = tts_manager.speak_monologue(chunk, voice_id=voice_id, cleanup=cleanup)
        if not success:
            print(f"Failed to speak chunk {i+1}")
    
    return True


def main():
    """Main function to parse arguments and process the file"""
    parser = argparse.ArgumentParser(description="Convert text file content to speech using TTS")
    parser.add_argument("file", help="Path to the text file to read")
    parser.add_argument("--voice", help="Voice ID to use (e.g., 'zf_001' for female, 'zm_010' for male)")
    parser.add_argument("--keep-files", action="store_true", help="Keep temporary audio files after playback")
    
    args = parser.parse_args()
    
    # Get voice ID if specified
    voice_id = args.voice
    
    # Process the file
    success = process_file_with_tts(args.file, voice_id=voice_id, cleanup=not args.keep_files)
    
    if success:
        print("\nFile processing completed successfully")
    else:
        print("\nFile processing failed")


if __name__ == "__main__":
    main()
