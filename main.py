#!/usr/bin/env python
"""
Main entry point for the Game AI Agent.
"""
import argparse
import logging
import os
import time
import mss

# Import project modules
from src.capture.screenshot import ScreenCapturer
from src.game_interface import game_controller
from src.tts import voice_synthesizer
from src.utils.config_loader import load_config
from src.utils.log_utils import setup_logging, get_session_logger
from src.utils import model_call
from src.vision.image_analyzer import ImageAnalyzer
# WindowDetector removed - using full screen capture instead

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
    
    # Log debug mode if enabled
    if args.debug:
        logger.debug("Debug mode enabled")

    # Load configuration
    config = load_config(args.config)
    
    # Initialize the model_call module
    model_call.initialize(model_type=args.model, config=config)

    # Override game from command line if provided
    game_name = args.game or config.get("game", "vanity_fair")
    game_config = config.get("games", {}).get(game_name, {})

    logger.info(f"Starting AI agent for game: {game_name}")
    logger.info(f"Using vision model: {args.model}")

    # Initialize screenshot capturer
    logger.info("Initializing screenshot capturer")
    screenshot_config = {
        "save_screenshots": not args.no_logging,
        "save_path": os.path.join(args.log_dir, "screenshots"),
    }
    screenshot_capture = ScreenCapturer(config=screenshot_config)

    # --- Use full screen instead of window detection ---
    logger.info("Using full screen for gameplay (window detection disabled)")
    
    # Set up full screen region
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        window_region = (monitor['left'], monitor['top'], monitor['width'], monitor['height'])
    
    logger.info(f"Using full screen region: {window_region}")
    detected_game_name = game_name  # Use the configured game name
    
    # Wait for 3 seconds to allow the game to start
    logger.info("Waiting for 3 seconds...")
    time.sleep(3)
    
    # Capture initial screenshot using full screen
    logger.info("Capturing initial full screen screenshot...")
    initial_screenshot = screenshot_capture.capture()
    
    # Configure screenshot capturer to use full screen region
    screenshot_capture.region = window_region
    logger.info(f"Screenshot region set to full screen: {window_region}")
    # --- End full screen setup ---

    # Initialize image analyzer with screenshot capture for frame extraction
    logger.info(f"Setting up image analyzer for game: {game_name}")
    
    # Add the game_name to the game_config to ensure it's properly identified
    game_config["name"] = game_name
    
    image_analyzer = ImageAnalyzer(
        model=args.model,
        config=config,  # Pass the full config instead of just the vision section
        game_config=game_config,
        screenshot_capture=screenshot_capture,
    )

    # Initialize session logger if enabled
    session_log = None
    if not args.no_logging:
        logger.info(f"Initializing session logger in {args.log_dir}")
        session_log = get_session_logger(log_dir=args.log_dir, game_name="Game AI")

    # Initialize game interface
    logger.info("Initializing game controller")
    # Create game controller but don't use it yet (will be used in future implementation)
    _ = game_controller.GameController(config=game_config)

    # Initialize TTS if enabled
    tts_engine = None
    if args.tts:
        logger.info("Initializing text-to-speech")
        tts_engine = voice_synthesizer.VoiceSynthesizer(config.get("tts", {}))

    # Main game loop
    try:
        turn_count = 0
        # Run for specified number of turns or until interrupted
        while turn_count < args.max_turns:
            # Increment turn counter
            turn_count += 1
            logger.info(f"Turn {turn_count}/{args.max_turns}")

            # Capture screenshot
            logger.debug("Capturing screenshot")
            image = screenshot_capture.capture()

            # Log screenshot if enabled
            if session_log:
                session_log.log_screenshot(image, turn=turn_count)

            # Analyze game state
            logger.debug("Analyzing game state")
            game_state = image_analyzer.analyze(
                image,
                analysis_type="game_state",
                check_animation=True,
                region_name="game"
            )

            # Log game state if enabled
            if session_log:
                session_log.log_game_state(game_state, turn=turn_count)

            # TODO: Add agent decision making here

            # Speak decision if TTS enabled
            if tts_engine:
                # TODO: Change this to actual agent thoughts/decisions
                tts_engine.speak(f"Turn {turn_count} analysis complete.")

            # Wait for next turn (simulate taking an action)
            logger.debug("Waiting for next turn...")
            time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Game interrupted by user")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        logger.info("Game ended")
        # Cleanup code here if needed


if __name__ == "__main__":
    main()