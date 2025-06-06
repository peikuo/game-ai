"""
Session logger for recording game sessions with screenshots and
    model responses.
"""

import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.utils.image_utils import optimize_image

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Class for logging game sessions with screenshots and model responses.
    """

    def __init__(self, log_dir="session_logs", game_name="Game AI"):
        """
        Initialize the SessionLogger.

        Args:
            log_dir (str): Directory to save session logs
            game_name (str): Name of the game being played
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.game_name = game_name

        # Create a new session log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.log_file = self.log_dir / f"session_{timestamp}.html"
        
        # Create screenshots directory
        self.screenshots_dir = self.log_dir / "screenshots" / self.session_id
        self.screenshots_dir.mkdir(exist_ok=True, parents=True)

        # Initialize the HTML log file with header
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"<!DOCTYPE html>\n")
            f.write(f"<html lang=\"en\">\n")
            f.write(f"<head>\n")
            f.write(f"  <meta charset=\"UTF-8\">\n")
            f.write(f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
            f.write(f"  <title>{self.game_name} AI Session Log</title>\n")
            f.write(f"  <style>\n")
            f.write(f"    body {{ font-family: Arial, sans-serif; margin: 20px; }}\n")
            f.write(f"    h1, h2, h3 {{ color: #333; }}\n")
            f.write(f"    .screenshot {{ max-width: 800px; margin: 10px 0; border: 1px solid #ddd; }}\n")
            f.write(f"    .scene-frame {{ max-width: 800px; margin: 10px 0; border: 1px solid #ddd; }}\n")
            f.write(f"    .scene-section {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-radius: 5px; }}\n")
            f.write(f"    pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }}\n")
            f.write(f"    .summary {{ background-color: #e6f7ff; padding: 15px; margin-top: 20px; border-radius: 5px; }}\n")
            f.write(f"  </style>\n")
            f.write(f"</head>\n")
            f.write(f"<body>\n")
            f.write(f"  <h1>{self.game_name} AI Session Log</h1>\n")
            f.write(f"  <p>Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"  <p>This file contains the log of a {self.game_name} AI gameplay session, including screenshots, ")
            f.write(f"AI analysis, and decisions made at each turn.</p>\n")
            f.write(f"  <hr>\n")

        logger.info(f"Session log initialized at {self.log_file}")

    def log_turn(
        self,
        turn_number,
        screenshot,
        game_state,
        action,
        agent_thoughts=None,
        prompt=None,
    ):
        """
        Log a turn to the session log file.

        Args:
            turn_number (int): Current turn number
            screenshot: PIL Image of the game screen
            game_state (dict): Game state analysis from the vision model
            action (dict): Action decided by the agent
            agent_thoughts (dict, optional): Thoughts from different agents
            prompt (str, optional): The prompt sent to the LLM
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                # Write turn header with timestamp
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"## Turn {turn_number} - {current_time}\n\n")

                # Save and embed screenshot
                f.write("### Screenshot\n\n")
                img_path = self._save_screenshot(screenshot, turn_number)
                f.write(f"![Turn {turn_number} Screenshot]({img_path})\n\n")

                # Also embed as base64 for self-contained MD file
                buffered = BytesIO()
                screenshot.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                f.write(f"<details>\n<summary>Embedded Screenshot</summary>\n\n")
                f.write(
                    f'<img src="data:image/png;base64,{img_str}" alt="Turn {turn_number} Screenshot" width="800">\n'
                )
                f.write(f"</details>\n\n")

                # Write the prompt sent to the LLM if available
                if prompt:
                    f.write("### Prompt Sent to LLM\n\n")
                    f.write("```\n")
                    f.write(prompt)
                    f.write("\n```\n\n")

                # Write game state analysis
                f.write("### Game State Analysis\n\n")

                # Write raw description if available
                if isinstance(game_state,
                              dict) and "raw_description" in game_state:
                    f.write("#### LLM Response\n\n")
                    f.write(game_state["raw_description"])
                    f.write("\n\n")

                # Write structured analysis if available
                f.write("#### Structured Analysis\n\n")
                f.write("```json\n")
                if isinstance(game_state, dict):
                    # Format nicely if it's a dict
                    structured_data = {
                        k: v for k, v in game_state.items() if k != "raw_description"}
                    f.write(json.dumps(structured_data, indent=2))
                else:
                    # Otherwise just write as string
                    f.write(str(game_state))
                f.write("\n```\n\n")

                # Write agent thoughts if available
                if agent_thoughts:
                    f.write("### Agent Thoughts\n\n")
                    for agent_name, thought in agent_thoughts.items():
                        f.write(f"#### {agent_name}\n\n")
                        f.write(thought)
                        f.write("\n\n")

                # Write decided action
                f.write("### Action Taken\n\n")
                f.write("```json\n")
                f.write(json.dumps(action, indent=2))
                f.write("\n```\n\n")

                # Add separator for next turn
                f.write("---\n\n")

            logger.debug(f"Logged turn {turn_number} to session log")
            return True

        except Exception as e:
            logger.error(f"Error logging turn {turn_number}: {e}")
            return False

    def _save_screenshot(self, screenshot, turn_number):
        """
        Save screenshot to file and return relative path.
        Optimizes the image to reduce file size.

        Args:
            screenshot: PIL Image
            turn_number: Current turn number

        Returns:
            str: Relative path to the saved screenshot
        """
        # Create screenshots directory if it doesn't exist
        screenshots_dir = self.log_dir / "screenshots" / self.session_id
        screenshots_dir.mkdir(exist_ok=True, parents=True)

        # Optimize the screenshot using the centralized utility
        # Target width of 800px is a good balance for log files
        # We don't force optimization if the image is already optimized
        optimized_screenshot, stats = optimize_image(
            screenshot, max_width=800, optimize_colors=True, quality=85, force=False)

        # Log optimization results if significant and not already optimized
        if (
            not stats.get("already_optimized", False)
            and stats.get("compression_ratio", 0) > 2.0
        ):
            logger.debug(
                f"Screenshot for turn {turn_number} optimized: "
                f"{stats.get('compression_ratio', 0):.1f}x smaller"
            )
        elif stats.get("already_optimized", False):
            logger.debug(
                f"Screenshot for turn {turn_number} was already optimized")

        # Save the optimized screenshot
        img_path = screenshots_dir / f"turn_{turn_number:03d}.png"
        optimized_screenshot.save(img_path)

        # Return relative path from the log file
        return os.path.relpath(img_path, self.log_dir)

    def log_screenshot(self, screenshot, turn=None):
        """
        Log just a screenshot to the session log file without additional data.

        Args:
            screenshot: PIL Image of the game screen
            turn: Optional turn number to associate with the screenshot
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                # Write header with timestamp
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if turn is None:
                    header_text = "Screenshot"
                else:
                    header_text = f"Screenshot - Turn {turn}"
                
                f.write("  <div class=\"scene-section\">\n")
                f.write(f"    <h2>{header_text} - {current_time}</h2>\n")

                # Save screenshot to the screenshots directory
                if turn is not None:
                    screenshot_name = f"screenshot_turn_{turn}.png"
                else:
                    screenshot_name = f"screenshot_{int(datetime.now().timestamp())}.png"
                
                # Save the actual file to the screenshots directory
                screenshot_path = self.screenshots_dir / screenshot_name
                screenshot.save(screenshot_path, format="PNG")
                
                # Embed as base64 directly in the HTML
                try:
                    buffered = BytesIO()
                    screenshot.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    f.write(f"    <img src=\"data:image/png;base64,{img_str}\" ")
                    f.write(f"alt=\"{header_text}\" class=\"screenshot\">\n")
                except Exception as e:
                    logger.error(f"Error embedding base64 image: {e}")
                    
                f.write("  </div>\n")
                f.write("  <hr>\n")

            logger.debug(
                "Logged screenshot%s", " for turn " + str(turn) if turn else ""
            )
            return True

        except Exception as e:
            logger.error("Error logging screenshot: %s", e)
            return False
            
    def log_game_state(self, game_state, turn=None):
        """Log the current game state.

        Args:
            game_state: Game state data to log
            turn: Optional turn number to associate with the game state

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                # Write turn header with timestamp if turn is provided
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if turn:
                    f.write("  <div class=\"scene-section\">\n")
                    f.write(f"    <h2>Game State - Turn {turn} - {current_time}</h2>\n")
                else:
                    f.write("  <div class=\"scene-section\">\n")
                    f.write(f"    <h2>Game State - {current_time}</h2>\n")

                # Write structured analysis
                f.write("    <h3>Structured Analysis</h3>\n")
                f.write("    <pre>\n")
                if isinstance(game_state, dict):
                    # Format nicely if it's a dict
                    f.write(json.dumps(game_state, indent=2))
                else:
                    # Otherwise just write as string
                    f.write(str(game_state))
                f.write("\n    </pre>\n")
                f.write("  </div>\n")
                f.write("  <hr>\n")
                
            logger.debug("Logged game state to session log")
            return True
            
        except Exception as e:
            logger.error(f"Error logging game state: {e}")
            return False

    def log_scene_detection(self, scene_result, turn=None):
        """Log scene detection results including animation frames and dialogue.

        Args:
            scene_result (dict): Scene detection results from SceneDetector
            turn: Optional turn number to associate with the scene
        """
        try:
            if not scene_result:
                logger.warning("Empty scene result, nothing to log")
                return False

            # Always log scene detection results, even if no animation or voice detected
            # This ensures we can see the base64-encoded animation frames in the log
            if not scene_result.get("animation_detected") and not scene_result.get("voice_detected"):
                logger.info("No animation or voice detected, but logging anyway for testing")

            with open(self.log_file, "a", encoding="utf-8") as f:
                # Write turn header with timestamp if turn is provided
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if turn is not None:
                    f.write(f"  <div class=\"scene-section\">\n")
                    f.write(f"    <h2>Scene Detection - Turn {turn} - {current_time}</h2>\n")
                else:
                    f.write(f"  <div class=\"scene-section\">\n")
                    f.write(f"    <h2>Scene Detection - {current_time}</h2>\n")

                # Write scene detection summary
                f.write("    <h3>Scene Detection Summary</h3>\n")
                f.write("    <ul>\n")
                f.write(f"      <li>Animation Detected: {scene_result.get('animation_detected', False)}</li>\n")
                f.write(f"      <li>Voice Detected: {scene_result.get('voice_detected', False)}</li>\n")
                
                # Ensure duration is a float before formatting
                duration = scene_result.get('duration', 0)
                if isinstance(duration, str):
                    try:
                        duration = float(duration)
                    except (ValueError, TypeError):
                        duration = 0.0
                f.write(f"      <li>Scene Duration: {duration:.2f} seconds</li>\n")
                f.write("    </ul>\n")

                # Log dialogue if available
                if scene_result.get("dialogue"):
                    f.write("    <h3>Detected Dialogue</h3>\n")
                    f.write("    <pre>\n")
                    f.write(scene_result["dialogue"])
                    f.write("\n    </pre>\n")
                    
                # Log voice content if available but dialogue is not
                elif scene_result.get("voice_detected") and scene_result.get("voice_content"):
                    f.write("    <h3>Detected Voice Content</h3>\n")
                    f.write("    <pre>\n")
                    f.write(scene_result["voice_content"])
                    f.write("\n    </pre>\n")

                # Log key frames if available
                animation_frames = scene_result.get("animation_frames", [])
                if animation_frames and len(animation_frames) > 0:
                    f.write(f"    <h3>Animation Key Frames ({len(animation_frames)} frames)</h3>\n")

                    # Save up to 5 key frames to avoid excessive logging
                    max_frames_to_log = min(5, len(animation_frames))
                    frame_indices = [0]  # Always include first frame

                    # Add middle frames if we have more than 2 frames
                    if max_frames_to_log > 2:
                        step = len(animation_frames) // (max_frames_to_log - 1)
                        frame_indices.extend(
                            [step * i for i in range(1, max_frames_to_log - 1)]
                        )

                    # Always include last frame if we have more than 1 frame
                    if len(animation_frames) > 1:
                        frame_indices.append(len(animation_frames) - 1)

                    # Save and embed selected frames
                    for i, frame_idx in enumerate(frame_indices):
                        if frame_idx < len(animation_frames):
                            frame = Image.fromarray(animation_frames[frame_idx])
                            
                            # Save screenshot to the screenshots directory
                            if turn is not None:
                                frame_name = f"{turn}_scene_{i}.png"
                            else:
                                frame_name = f"scene_{i}.png"
                            
                            # Save the actual file to the screenshots directory
                            screenshot_path = self.screenshots_dir / frame_name
                            frame.save(screenshot_path, format="PNG")
                            
                            # Embed as base64 directly in the HTML
                            try:
                                buffered = BytesIO()
                                frame.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                f.write(f"    <div>\n")
                                f.write(f"      <h4>Animation Frame {i}</h4>\n")
                                f.write(f"      <img src=\"data:image/png;base64,{img_str}\" ")
                                f.write(f"alt=\"Scene Frame {i}\" class=\"scene-frame\">\n")
                                f.write(f"    </div>\n")
                            except Exception as e:
                                logger.error(f"Error embedding base64 image: {e}")

                # Write full scene data in JSON format
                f.write("    <h3>Full Scene Data</h3>\n")
                f.write("    <pre>\n")
                # Remove the actual frame data to avoid huge log files
                scene_data = {}
                for k, v in scene_result.items():
                    if k != "animation_frames":
                        # Ensure all values are JSON serializable
                        if isinstance(v, (bool, int, float, str, list, dict)) or v is None:
                            scene_data[k] = v
                        else:
                            # Convert non-serializable types to string
                            scene_data[k] = str(v)
                            
                if "animation_frames" in scene_result:
                    scene_data["animation_frames_count"] = len(scene_result["animation_frames"])
                    
                f.write(json.dumps(scene_data, indent=2))
                f.write("\n    </pre>\n")
                f.write("  </div>\n")
                f.write("  <hr>\n")

            logger.debug(f"Logged scene detection results to session log")
            return True

        except Exception as e:
            logger.error(f"Error logging scene detection: {e}")
            return False

    def log_summary(
            self,
            total_turns,
            final_score=None,
            victory_type=None,
            additional_notes=None):
        """
            Log a summary of the session.

            Args:
                total_turns (int): Total number of turns played
                final_score (int, optional): Final score
                victory_type (str, optional): Type of victory achieved
                additional_notes (str, optional):
        Additional notes about the session
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("  <div class=\"summary\">\n")
                f.write("    <h2>Session Summary</h2>\n")
                f.write(f"    <p><strong>Total turns played:</strong> {total_turns}</p>\n")

                if final_score is not None:
                    f.write(f"    <p><strong>Final score:</strong> {final_score}</p>\n")

                if victory_type is not None:
                    f.write(f"    <p><strong>Victory type:</strong> {victory_type}</p>\n")

                if additional_notes is not None:
                    f.write("    <h3>Additional Notes</h3>\n")
                    f.write(f"    <p>{additional_notes}</p>\n")

                f.write(f"    <p><em>Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>\n")
                f.write("  </div>\n")

            logger.info(f"Session summary logged to {self.log_file}")
            return True

        except Exception as e:
            logger.error(f"Error logging session summary: {e}")
            return False
