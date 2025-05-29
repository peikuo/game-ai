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
        self.log_file = self.log_dir / f"session_{timestamp}.md"

        # Initialize the log file with header
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# {self.game_name} AI Session Log\n\n")
            f.write(
                f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(
                f"This file contains the log of a {self.game_name} AI gameplay session, including screenshots,"
            )
            f.write("AI analysis, and decisions made at each turn.\n\n")
            f.write("---\n\n")

        logger.info(f"Session log initialized at {self.log_file}")

    def log_turn(
        self, turn_number, screenshot, game_state, action, agent_thoughts=None, prompt=None
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
                f.write(f'<img src="data:image/png;base64,{img_str}" alt="Turn {turn_number} Screenshot" width="800">\n')
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
                if isinstance(game_state, dict) and "raw_description" in game_state:
                    f.write("#### LLM Response\n\n")
                    f.write(game_state["raw_description"])
                    f.write("\n\n")
                
                # Write structured analysis if available
                f.write("#### Structured Analysis\n\n")
                f.write("```json\n")
                if isinstance(game_state, dict):
                    # Format nicely if it's a dict
                    structured_data = {k: v for k, v in game_state.items() if k != "raw_description"}
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
            screenshot, max_width=800, optimize_colors=True, quality=85, force=False
        )
        
        # Log optimization results if significant and not already optimized
        if not stats.get("already_optimized", False) and stats.get("compression_ratio", 0) > 2.0:
            logger.debug(
                f"Screenshot for turn {turn_number} optimized: "
                f"{stats.get('compression_ratio', 0):.1f}x smaller"
            )
        elif stats.get("already_optimized", False):
            logger.debug(f"Screenshot for turn {turn_number} was already optimized")

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
                header = "## Screenshot" if turn is None else f"## Screenshot - Turn {turn}"
                f.write(f"{header} - {current_time}\n\n")

                # Save and embed screenshot
                img_path = self._save_screenshot(screenshot, turn or 0)
                f.write(f"![{header}]({img_path})\n\n")

                # Add separator
                f.write("---\n\n")
                
            logger.debug("Logged screenshot%s", ' for turn ' + str(turn) if turn else '')
            return True

        except Exception as e:
            logger.error("Error logging screenshot: %s", e)
            return False
            
    def log_game_state(self, game_state, turn=None):
        """
        Log just the game state analysis to the session log file without screenshots or actions.
        
        Args:
            game_state (dict): Game state analysis from the vision model
            turn: Optional turn number to associate with the game state
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                # Write header with timestamp
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = "## Game State" if turn is None else f"## Game State - Turn {turn}"
                f.write(f"{header} - {current_time}\n\n")

                # Write game state analysis
                f.write("### Game State Analysis\n\n")
                
                # Write raw description if available
                if isinstance(game_state, dict) and "raw_description" in game_state:
                    f.write("#### LLM Response\n\n")
                    f.write(game_state["raw_description"])
                    f.write("\n\n")
                
                # Write structured analysis if available
                f.write("#### Structured Analysis\n\n")
                f.write("```json\n")
                if isinstance(game_state, dict):
                    # Format nicely if it's a dict
                    structured_data = {k: v for k, v in game_state.items() if k != "raw_description"}
                    f.write(json.dumps(structured_data, indent=2))
                else:
                    # Otherwise just write as string
                    f.write(str(game_state))
                f.write("\n```\n\n")

                # Add separator
                f.write("---\n\n")
                
            logger.debug("Logged game state to session log" + (f" for turn {turn}" if turn else ""))
            return True
        except Exception as e:
            logger.error("Error logging game state%s: %s", 
                        f" for turn {turn}" if turn else "", e)
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
                f.write("## Session Summary\n\n")
                f.write(f"Total turns played: {total_turns}\n\n")

                if final_score is not None:
                    f.write(f"Final score: {final_score}\n\n")

                if victory_type is not None:
                    f.write(f"Victory type: {victory_type}\n\n")

                if additional_notes is not None:
                    f.write("### Additional Notes\n\n")
                    f.write(additional_notes)
                    f.write("\n\n")

                f.write(
                    f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

            logger.info(f"Session summary logged to {self.log_file}")
            return True

        except Exception as e:
            logger.error(f"Error logging session summary: {e}")
            return False
