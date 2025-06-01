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
from src.game_player.game_analyzer import GameAnalyzer
from src.utils.config_loader import load_config
from src.utils.log_utils import setup_logging, get_session_logger, log_screenshot, log_game_state
from src.utils import model_call
from src.vision.image_analyzer import ImageAnalyzer
from src.vision.window_detector import WindowDetector

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
        help="Enable text-to-speech for monologues",
        default=True,
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
    
    # Debug log the configuration
    if args.debug:
        logger.debug(f"Loaded config keys: {list(config.keys())}")
        if "games" in config:
            logger.debug(f"Games in config: {list(config.get('games', {}).keys())}")
            for game_key, game_data in config.get('games', {}).items():
                logger.debug(f"Game '{game_key}' config keys: {list(game_data.keys())}")
                if 'prompts' in game_data:
                    logger.debug(f"Game '{game_key}' prompts: {list(game_data.get('prompts', {}).keys())}")
        
        # Log game_config
        logger.debug(f"Game config: {game_config}")
        logger.debug(f"TTS config: {game_config.get('tts', {})}")
    
    # Initialize the model_call module
    model_call.initialize(model_type=args.model, config=config)

    # Override game from command line if provided
    game_name = args.game or config.get("current_game", "vanity_fair")
    game_config = config.get("games", {}).get(game_name, {})
    
    # Add the game name to the config to ensure it's available for all components
    game_config["current_game"] = game_name
    
    # Add TTS configuration from command line arguments
    if "tts" not in game_config:
        game_config["tts"] = {}
    game_config["tts"]["enabled"] = args.tts
    
    logger.info(f"Starting AI agent for game: {game_name}")
    logger.info(f"Using vision model: {args.model}")
    logger.info(f"TTS enabled: {args.tts}")

    # Initialize screenshot capturer
    logger.info("Initializing screenshot capturer")
    screenshot_config = {
        "save_screenshots": not args.no_logging,
        "save_path": os.path.join(args.log_dir, "screenshots"),
    }
    screenshot_capture = ScreenCapturer(config=screenshot_config)

    # --- Game window detection ---
    window_region = None
    detected_game_name = None

    # Initialize window detector if window detection is enabled
    if config.get("detect_window", True):
        logger.info("Detecting game window...")
        window_detector = WindowDetector(
            model=args.model,
            config=config.get("vision", {}),
        )

        # Wait for 3 seconds to allow the game to process
        logger.info("Waiting for 3 seconds...")
        time.sleep(3)
        logger.info("Capturing screenshot for window detection...")

        # Capture a screenshot for window detection
        initial_screenshot = screenshot_capture.capture()

        # Detect game window
        window_region, detected_game_name = window_detector.detect_game_window(
            initial_screenshot,
            game_name=game_name,
            prompt_context="",
        )
        if window_region:
            screenshot_capture.region = window_region
            logger.info(f"Game window region set to: {window_region}")
        else:
            logger.warning(
                "Could not detect game window region. Using full screen.")

        if detected_game_name:
            logger.info(f"Detected game name: {detected_game_name}")
            # Optionally update config or runtime state with detected game name
    # --- End game window detection ---

    # Initialize image analyzer with screenshot capture for frame extraction
    image_analyzer = ImageAnalyzer(
        model=args.model,
        config=config.get("vision", {}),
        game_config=game_config,
        screenshot_capture=screenshot_capture,
    )

    # Initialize session logger if enabled
    if not args.no_logging:
        logger.info(f"Initializing session logger in {args.log_dir}")
        get_session_logger(log_dir=args.log_dir, game_name=game_name)

    # Initialize game analyzer
    logger.info("Initializing game analyzer")
    game_analyzer = GameAnalyzer(config=game_config)
    logger.debug(f"TTS configuration: {game_config.get('tts', {})}")
    
    # Initialize game controller
    logger.info("Initializing game controller")
    game_controller = GameController(config=game_config)

    logger.info("Waiting for 5 seconds...")
    time.sleep(5)
    logger.info("OK, Start Capture!")

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

            # Log command line arguments once per turn for debugging
            logger.debug(f"Command line arguments: {args}")

            # Log screenshot if enabled
            if not args.no_logging:
                log_screenshot(image, turn=turn_count)

            # Analyze game state
            logger.debug("Analyzing game state")
            game_state = image_analyzer.analyze(
                image,
                analysis_type="game_state",
                check_animation=True,
                region_name="game"
            )

            # Log game state if enabled
            log_game_state(game_state, turn=turn_count)
            
            # Process game state through the game analyzer
            logger.debug("Processing game state through GameAnalyzer")
            game_state = game_analyzer.process_game_state(game_state, turn_number=turn_count)
                
            # Execute game action if available in the game state using the object-oriented approach
            # Access the game state through the _obj property
            game_obj = game_state.get('_obj')
            if game_obj and hasattr(game_obj, 'action_analysis'):
                action_analysis = game_obj.action_analysis
                if hasattr(action_analysis, 'simple') and action_analysis.simple:
                    if hasattr(action_analysis, 'action') and action_analysis.action:
                        action = action_analysis.action
                        # Convert action object back to dict for the controller
                        action_dict = {}
                        for attr in dir(action):
                            if not attr.startswith('_'):
                                action_dict[attr] = getattr(action, attr)
                        
                        logger.info(f"Executing game action from analysis: {action_dict}")
                        game_controller.execute_action(action_dict)
                    else:
                        logger.warning("Simple action indicated but no action object found")
                else:
                    logger.info("No simple action to execute in this turn")
            else:
                logger.info("No action_analysis found in game state object")
            
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