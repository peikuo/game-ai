#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import soundfile as sf
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import only what we need for TTS
from kokoro import KModel, KPipeline
import torch

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
SAMPLE_RATE = 24000
REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'

# Available voices
VOICES = {
    'female_1': 'zf_001',  # Chinese female voice 1
    'male_1': 'zm_010',    # Chinese male voice 1
    # Add more voices if available in the model
}

# Default voice
DEFAULT_VOICE = VOICES['female_1']

def setup_tts():
    """Set up the TTS system"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the TTS model and pipeline
    model = KModel(repo_id=REPO_ID).to(device).eval()
    
    # Helper function for English words in Chinese text
    def en_callable(text):
        if text == 'Kokoro':
            return 'kˈOkəɹO'
        return text  # Default simple handling
    
    # Create Chinese pipeline
    zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=model, en_callable=en_callable)
    
    return zh_pipeline

def say_monologue(text, zh_pipeline, voice=DEFAULT_VOICE):
    """Convert text to speech and play it"""
    voice_name = next((k for k, v in VOICES.items() if v == voice), "unknown")
    print(f"\n--- Monologue (Voice: {voice_name}) ---\n{text}\n")
    
    if not text:
        print("No monologue to speak.")
        return
    
    # Process the text using the pipeline
    for result in zh_pipeline(text, voice=voice):
        gs, ps, audio = result
        
        # Save audio to a temporary file
        temp_audio_file = "temp_monologue.wav"
        sf.write(temp_audio_file, audio, SAMPLE_RATE)
        
        # Play the audio using system command
        print(f"Speaking monologue...")
        os.system(f"afplay {temp_audio_file}")
        
        # Clean up
        os.remove(temp_audio_file)

def test_monologue(selected_voice=None):
    """Test monologue generation and TTS"""
    # Set up TTS
    zh_pipeline = setup_tts()
    
    # Determine which voice to use
    voice = selected_voice or DEFAULT_VOICE
    
    # Example monologues to test
    monologues = [
        "我现在看到了游戏的主菜单，有开始游戏、设置和退出三个按钮。游戏的主角站在屏幕右侧，看起来是在等待玩家的指令。天气晴朗，似乎是白天的场景。",
        "角色正在战斗中，生命值只剩下30%了，应该立即使用恢复药水。敌人似乎有火属性，使用水系技能可能会更有效。",
        "我注意到游戏界面中有一个隐藏的宝箱，在右下角的灌木丛后面，玩家可能没有注意到。这可能包含重要的物品，建议去检查一下。"
    ]
    
    # Test each monologue
    print("\n========== 测试游戏AI的中文独白功能 ==========\n")
    
    if selected_voice is None:
        # Test with all available voices
        print("Testing with all available voices...\n")
        for voice_name, voice_id in VOICES.items():
            print(f"\n==== 使用 {voice_name} 声音 ====")
            # Just test the first monologue with each voice
            say_monologue(monologues[0], zh_pipeline, voice=voice_id)
            time.sleep(1)  # Pause between voices
    else:
        # Test all monologues with the selected voice
        for i, monologue in enumerate(monologues):
            print(f"\n---- 独白 {i+1} ----")
            say_monologue(monologue, zh_pipeline, voice=voice)
            time.sleep(1)  # Pause between monologues
    
    print("\n测试完成！所有示例独白已成功转换为语音。")
    print("在实际游戏中，这些独白将根据游戏画面实时生成。")

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test the monologue TTS feature with different voices.')
    parser.add_argument('--voice', type=str, choices=list(VOICES.keys()), 
                        help='Voice to use for the monologue')
    
    args = parser.parse_args()
    
    # If a voice is specified, use that one, otherwise test all voices
    selected_voice = VOICES.get(args.voice) if args.voice else None
    test_monologue(selected_voice)
