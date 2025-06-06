#!/usr/bin/env python
"""
Main entry point for the Game AI Agent.
"""
import argparse
import logging
import os
import time

# Import project modules
from src.capture.screenshot import ScreenCapturer
from src.game_interface.game_controller import GameController
from src.game_interface.scene_detect import SceneDetector
from src.game_player.game_analyzer import GameAnalyzer
from src.tts.tts_manager import TTSManager
from src.utils import model_call
from src.utils.config_loader import load_config
from src.utils.log_utils import get_session_logger, setup_logging
from src.vision.image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Game AI Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["ollama", "qwen"],
        help="Vision model to use",
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Game to play (overrides config)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="session_logs",
        help="Directory to store session logs",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable session logging",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum number of turns to play",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Configure logging using our utility module
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level, log_file="game_ai.log", cleanup=True)

    # Load configuration
    config = load_config(args.config)

    # Initialize the model_call module
    model_call.initialize(model_type=args.model, config=config)

    # Override game from command line if provided
    game_name = args.game or config.get("game", "vanity_fair")
    game_config = config.get("games", {}).get(game_name, {})

    logger.info(
        f"Starting AI agent for game: {game_name} with {args.model} model")

    # Initialize screenshot capturer with full screen mode enabled
    screenshot_config = {
        "save_screenshots": not args.no_logging,
        "save_path": os.path.join(args.log_dir, "screenshots"),
        "use_full_screen": True,  # Enable full screen capture
    }
    screenshot_capture = ScreenCapturer(config=screenshot_config)

    # Add the game_name to the game_config to ensure it's properly identified
    game_config["name"] = game_name

    # Initialize core components
    image_analyzer = ImageAnalyzer(
        model=args.model,
        config=config,
        game_config=game_config,
        screenshot_capture=screenshot_capture,
    )
    logger.info("Image analyzer inited")

    # Initialize session logger if enabled
    session_log = get_session_logger(log_dir=args.log_dir, game_name="Game AI")
    logger.info("Session logger inited")

    # Initialize TTS Manager if enabled
    tts_manager = TTSManager(config.get("audio", {}))
    logger.info("TTS initialized and enabled")

    # Initialize Scene Detector
    scene_config = {
        "animation_diff_threshold": 0.02,
        "animation_consecutive_frames": 3,
        "animation_interval": 0.1,
        "voice_detection_enabled": True,
        "voice_silence_threshold": 2.0,
        "whisper_model": "tiny",
        "whisper_language": "zh",
        "audio_recording_path": "temp_audio",
    }
    scene_detector = SceneDetector(scene_config)
    logger.info("Scene detector initialized")

    # Create game controller and analyzer
    game_controller = GameController(config=game_config)
    logger.info("Game controller inited")
    game_analyzer = GameAnalyzer(config=game_config, tts_manager=tts_manager)
    logger.info("Game analyzer inited")

    # Wait for 3 seconds to allow the game to start
    logger.info("Waiting for 5 seconds to allow the game to start")
    time.sleep(5)
    logger.info("OK， Start working!")
    # Main game loop
    try:
        turn_count = 0
        # Run for specified number of turns or until interrupted
        while turn_count < args.max_turns:
            # Increment turn counter
            turn_count += 1
            logger.info(f"Turn {turn_count}/{args.max_turns}")

            # Capture screenshot and analyze game state
            image = screenshot_capture.capture()

            # Check for scene detection (animations and voice)
            scene_result = scene_detector.record(image)

            if not scene_result:
                logger.error("No scene results available, need log check!")
                continue

            if not scene_result.get("finished", True):
                logger.info("Scene detection in progress, waiting...")
                time.sleep(1)  # Wait 1 second before continuing
                continue

            if scene_result.get("animation_detected", False):
                logger.info("Animation detected in scene")
                # Key frames are available in scene_result["animation_frames"]

            if scene_result.get("voice_detected", False):
                logger.info("Voice detected in scene")
                # Dialogue is available in scene_result["dialogue"]

            # Log scene detection results if session logging is enabled
            if session_log:
                session_log.log_scene_detection(scene_result, turn=turn_count)

            # Log screenshot if enabled
            if session_log:
                session_log.log_screenshot(image, turn=turn_count)

            # Analyze game state
            game_state = image_analyzer.analyze(
                image, analysis_type="game_state", region_name="game"
            )

            # Log game state if enabled
            if session_log:
                session_log.log_game_state(game_state, turn=turn_count)

            # Display monologue if available (TTS handled by GameAnalyzer)
            if "monologue" in game_state and game_state["monologue"]:
                monologue = game_state["monologue"]
                # Print with color to make it stand out in the terminal
                print(f"\nMonologue: {monologue}\n")

                # Log the raw LLM response if available for debugging
                if "raw_response" in game_state:
                    logger.debug(
                        f"Raw LLM Response: {game_state['raw_response']}")

                # Add monologue to session log if enabled
                if session_log:
                    # Create a properly structured game state object for
                    # logging
                    monologue_data = {
                        "raw_description": monologue,  # Required by the logger
                        "monologue": monologue,
                        "type": "monologue_only",
                    }
                    session_log.log_game_state(monologue_data, turn=turn_count)

            # Use game analyzer to determine next action
            action = game_analyzer.analyze(game_state, turn_number=turn_count)

            # Execute the determined action
            if action:
                logger.info(f"Executing action: {action}")
                game_controller.execute_action(action)
            else:
                logger.warning("No action determined from game state")

            # Wait for next turn (simulate taking an action)
            logger.debug("Waiting for next turn...")
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Game interrupted by user")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        logger.info("Game ended")
        # Cleanup code here if needed


if __name__ == "__main__":
    main()
