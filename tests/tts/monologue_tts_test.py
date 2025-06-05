#!/usr/bin/env python
"""
Test script for monologue TTS feature.
This script demonstrates how the monologue feature works with the TTSManager.
"""

from src.utils.config_loader import load_config
from src.tts.tts_manager import TTSManager
import os
import sys
import warnings
from pathlib import Path

# Filter warnings to suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to path so we can import project modules
sys.path.append(str(Path(__file__).parent.parent))


def test_monologue_tts():
    """Test the monologue TTS feature with different voices"""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    config = load_config(config_path)

    # Initialize TTSManager
    tts_manager = TTSManager(config)

    # Check if TTS is available
    if not tts_manager.is_available():
        print("TTS functionality is not available. Please check your configuration.")
        return

    # Example monologues in Chinese
    monologues = [
        "我看到角色正在主城中心，有几个NPC在等待交互。右侧有一个任务面板显示有新的任务可接取。建议先与任务NPC对话，了解当前的主线任务。",
        "战斗场景中，敌人似乎有火属性，我注意到他们的攻击带有火焰效果。角色的生命值还有70%，但魔法值较低。建议使用水系技能进行克制，并注意补充魔法值。",
        "这个宝箱看起来有陷阱，周围地面上有可疑的压力板。我建议先使用远程工具或技能触发陷阱，然后再安全地打开宝箱。宝箱可能包含稀有物品。",
    ]

    # Test with default voice
    print("\n===== 测试默认声音 =====")
    for i, monologue in enumerate(monologues):
        print(f"\n--- 测试独白 {i+1} ---")
        print(monologue)
        # Use cleanup=True to remove the audio file after playback
        tts_manager.speak_monologue(monologue, cleanup=True)

    # Test with male voice if available
    if "male_1" in tts_manager.VOICES.keys():
        male_voice = tts_manager.VOICES["male_1"]
        print("\n===== 测试男声 =====")
        for i, monologue in enumerate(monologues):
            print(f"\n--- 测试独白 {i+1} ---")
            print(monologue)
            # Use cleanup=True to remove the audio file after playback
            tts_manager.speak_monologue(
                monologue, voice_id=male_voice, cleanup=True)

    print("\n测试完成！")


if __name__ == "__main__":
    print("Testing monologue TTS feature...")
    test_monologue_tts()
