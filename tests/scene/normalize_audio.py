#!/usr/bin/env python3
"""
Script to normalize audio files to make them audible.
"""

import argparse
import os
import wave
import numpy as np

def normalize_audio_file(input_file, output_file=None, gain_factor=10.0):
    """
    Normalize an audio file by increasing its volume.
    
    Args:
        input_file (str): Path to input WAV file
        output_file (str): Path to output WAV file (if None, will append '_normalized' to input filename)
        gain_factor (float): Factor to multiply the audio data by
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_normalized{ext}"
    
    print(f"Normalizing {input_file} to {output_file} with gain factor {gain_factor}")
    
    # Open input file
    with wave.open(input_file, 'rb') as wf:
        # Get parameters
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read all frames
        frames = wf.readframes(n_frames)
    
    # Convert to numpy array
    audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # Print audio stats before normalization
    print(f"Before normalization - Min: {np.min(audio_data)}, Max: {np.max(audio_data)}, Mean: {np.mean(audio_data):.2f}")
    
    # Apply gain factor
    audio_data = audio_data.astype(np.float32) * gain_factor
    
    # Clip to valid range for int16
    audio_data = np.clip(audio_data, -32768, 32767)
    
    # Convert back to int16
    audio_data = audio_data.astype(np.int16)
    
    # Print audio stats after normalization
    print(f"After normalization - Min: {np.min(audio_data)}, Max: {np.max(audio_data)}, Mean: {np.mean(audio_data):.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Write to output file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(framerate)
        wf.writeframes(audio_data.tobytes())
    
    print(f"Normalized audio saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Normalize WAV audio files")
    parser.add_argument("input_file", help="Input WAV file path")
    parser.add_argument("-o", "--output-file", help="Output WAV file path")
    parser.add_argument("-g", "--gain", type=float, default=10.0, help="Gain factor (default: 10.0)")
    
    args = parser.parse_args()
    normalize_audio_file(args.input_file, args.output_file, args.gain)

if __name__ == "__main__":
    main()
