"""
HTML Session logger for recording game sessions with screenshots and
model responses in an interactive HTML format.
"""

import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from src.utils.image_utils import optimize_image, truncate_base64

logger = logging.getLogger(__name__)


class HTMLSessionLogger:
    """
    Class for logging game sessions with screenshots and model responses in HTML format.
    """

    def __init__(self, log_dir="session_logs", game_name="Game AI"):
        """
        Initialize the HTMLSessionLogger.

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
        self.screenshot_count = 0
        self.turn_count = 0

        # Initialize the log file with HTML template
        self._initialize_html_file()
        
        logger.info(f"HTML Session log initialized at {self.log_file}")

    def _initialize_html_file(self):
        """Initialize the HTML log file with header and CSS styling."""
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.game_name} AI Session Log</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 40px;
        }}
        .turn-container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            padding: 20px;
            overflow: hidden;
        }}
        .screenshot {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        code {{
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
        }}
        .json {{
            white-space: pre-wrap;
        }}
        .collapsible {{
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .active, .collapsible:hover {{
            background-color: #e0e0e0;
        }}
        .content {{
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: white;
            border-radius: 0 0 4px 4px;
        }}
        .llm-response {{
            background-color: #f0f7fb;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .agent-thought {{
            background-color: #fdf5e6;
            border-left: 5px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .action {{
            background-color: #f0fff0;
            border-left: 5px solid #2ecc71;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .summary {{
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            margin-top: 40px;
        }}
        /* Tabs for organizing content */
        .tab {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 4px 4px 0 0;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: white;
            border-bottom: 2px solid #3498db;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 4px 4px;
            background-color: white;
        }}
    </style>
</head>
<body>
    <h1>{self.game_name} AI Session Log</h1>
    <div class="timestamp">
        Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    <p>
        This file contains the log of a {self.game_name} AI gameplay session, including screenshots,
        AI analysis, and decisions made at each turn.
    </p>
    <div id="session-content">
    <!-- Session content will be appended here -->
    </div>
    
    <script>
    // JavaScript for collapsible sections
    document.addEventListener('DOMContentLoaded', function() {{
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {{
            coll[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                }}
            }});
        }}
        
        // Function for tab navigation
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Expose the function globally
        window.openTab = openTab;
        
        // Activate the first tab by default for each tab group
        var tabGroups = document.getElementsByClassName("tab");
        for (var i = 0; i < tabGroups.length; i++) {{
            var firstTab = tabGroups[i].getElementsByClassName("tablinks")[0];
            if (firstTab) {{
                firstTab.click();
            }}
        }}
    }});
    </script>
</body>
</html>
"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(html_template)

    def _append_to_html(self, html_content):
        """
        Append HTML content to the log file.
        
        Args:
            html_content (str): HTML content to append
        """
        try:
            # Read the current file
            with open(self.log_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Replace the placeholder with the new content and the placeholder again
            # This effectively inserts the new content before the closing tags
            content = content.replace("</div>\n    \n    <script>", 
                                     f"{html_content}</div>\n    \n    <script>")
                
            # Write the updated content back
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(content)
                
            return True
        except Exception as e:
            logger.error(f"Error appending to HTML log: {e}")
            return False

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
            self.turn_count = turn_number
            
            # Start building HTML content
            turn_html = []
            turn_html.append(f'<div class="turn-container">')
            turn_html.append(f'<h2>Turn {turn_number} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>')
            
            # Create tab navigation
            turn_html.append('<div class="tab">')
            turn_html.append(f'<button class="tablinks" onclick="openTab(event, \'screenshot-{turn_number}\')">Screenshot</button>')
            turn_html.append(f'<button class="tablinks" onclick="openTab(event, \'analysis-{turn_number}\')">Analysis</button>')
            turn_html.append(f'<button class="tablinks" onclick="openTab(event, \'action-{turn_number}\')">Action</button>')
            
            # Add optional tab buttons
            if prompt:
                turn_html.append(f'<button class="tablinks" onclick="openTab(event, \'prompt-{turn_number}\')">Prompt</button>')
            if agent_thoughts:
                turn_html.append(f'<button class="tablinks" onclick="openTab(event, \'thoughts-{turn_number}\')">Agent Thoughts</button>')
            
            turn_html.append('</div>')
            
            # Screenshot tab content
            turn_html.append(f'<div id="screenshot-{turn_number}" class="tabcontent">')
            turn_html.append('<h3>Screenshot</h3>')
            
            # Save and embed screenshot
            img_path = self._save_screenshot(screenshot, turn_number)
            
            # Encode image to base64 for embedding
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            turn_html.append(f'<img src="data:image/png;base64,{img_str}" alt="Turn {turn_number} Screenshot" class="screenshot">')
            turn_html.append(f'<p><small>Saved to: {img_path}</small></p>')
            turn_html.append('</div>')
            
            # Analysis tab content
            turn_html.append(f'<div id="analysis-{turn_number}" class="tabcontent">')
            turn_html.append('<h3>Game State Analysis</h3>')
            
            # Add raw description if available
            if isinstance(game_state, dict) and "raw_description" in game_state:
                turn_html.append('<div class="llm-response">')
                turn_html.append('<h4>LLM Response</h4>')
                turn_html.append(f'<p>{game_state["raw_description"].replace("\n", "<br>")}</p>')
                turn_html.append('</div>')
            
            # Add structured analysis
            turn_html.append('<div>')
            turn_html.append('<h4>Structured Analysis</h4>')
            turn_html.append('<pre class="json"><code>')
            
            if isinstance(game_state, dict):
                # Format nicely if it's a dict
                structured_data = {k: v for k, v in game_state.items() if k != "raw_description"}
                turn_html.append(json.dumps(structured_data, indent=2))
            else:
                # Otherwise just write as string
                turn_html.append(str(game_state))
            
            turn_html.append('</code></pre>')
            turn_html.append('</div>')
            turn_html.append('</div>')
            
            # Action tab content
            turn_html.append(f'<div id="action-{turn_number}" class="tabcontent">')
            turn_html.append('<h3>Action Taken</h3>')
            turn_html.append('<div class="action">')
            turn_html.append('<pre class="json"><code>')
            turn_html.append(json.dumps(action, indent=2))
            turn_html.append('</code></pre>')
            turn_html.append('</div>')
            turn_html.append('</div>')
            
            # Add prompt tab if available
            if prompt:
                turn_html.append(f'<div id="prompt-{turn_number}" class="tabcontent">')
                turn_html.append('<h3>Prompt Sent to LLM</h3>')
                turn_html.append('<pre><code>')
                turn_html.append(prompt)
                turn_html.append('</code></pre>')
                turn_html.append('</div>')
            
            # Add agent thoughts tab if available
            if agent_thoughts:
                turn_html.append(f'<div id="thoughts-{turn_number}" class="tabcontent">')
                turn_html.append('<h3>Agent Thoughts</h3>')
                
                for agent_name, thought in agent_thoughts.items():
                    turn_html.append('<div class="agent-thought">')
                    turn_html.append(f'<h4>{agent_name}</h4>')
                    turn_html.append(f'<p>{thought.replace("\n", "<br>")}</p>')
                    turn_html.append('</div>')
                
                turn_html.append('</div>')
            
            # Close the turn container
            turn_html.append('</div>')
            
            # Join all HTML parts and append to the file
            self._append_to_html('\n'.join(turn_html))
            
            logger.debug(f"Logged turn {turn_number} to HTML session log")
            return True

        except Exception as e:
            logger.error(f"Error logging turn {turn_number} to HTML: {e}")
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
            self.screenshot_count += 1
            turn_num = turn if turn is not None else f"S{self.screenshot_count}"
            
            # Create screenshot container
            screenshot_html = []
            screenshot_html.append('<div class="turn-container">')
            screenshot_html.append(f'<h2>Screenshot {turn_num} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>')
            
            # Save and embed screenshot
            img_path = self._save_screenshot(screenshot, turn or self.screenshot_count)
            
            # Encode image to base64 for embedding
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            screenshot_html.append(f'<img src="data:image/png;base64,{img_str}" alt="Screenshot {turn_num}" class="screenshot">')
            screenshot_html.append(f'<p><small>Saved to: {img_path}</small></p>')
            screenshot_html.append('</div>')
            
            # Join all HTML parts and append to the file
            self._append_to_html('\n'.join(screenshot_html))
            
            logger.debug(f"Logged screenshot {turn_num} to HTML session log")
            return True

        except Exception as e:
            logger.error(f"Error logging screenshot to HTML: {e}")
            return False
            
    def log_game_state(self, game_state, turn=None):
        """
        Log just the game state analysis to the session log file without screenshots or actions.
        
        Args:
            game_state (dict): Game state analysis from the vision model
            turn: Optional turn number to associate with the game state
        """
        try:
            turn_num = turn if turn is not None else self.turn_count + 1
            
            # Create game state container
            game_state_html = []
            game_state_html.append('<div class="turn-container">')
            game_state_html.append(f'<h2>Game State Analysis {turn_num} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>')
            
            # Add raw description if available
            if isinstance(game_state, dict) and "raw_description" in game_state:
                game_state_html.append('<div class="llm-response">')
                game_state_html.append('<h3>LLM Response</h3>')
                game_state_html.append(f'<p>{game_state["raw_description"].replace("\n", "<br>")}</p>')
                game_state_html.append('</div>')
            
            # Add structured analysis
            game_state_html.append('<div>')
            game_state_html.append('<h3>Structured Analysis</h3>')
            game_state_html.append('<pre class="json"><code>')
            
            if isinstance(game_state, dict):
                # Format nicely if it's a dict
                structured_data = {k: v for k, v in game_state.items() if k != "raw_description"}
                game_state_html.append(json.dumps(structured_data, indent=2))
            else:
                # Otherwise just write as string
                game_state_html.append(str(game_state))
            
            game_state_html.append('</code></pre>')
            game_state_html.append('</div>')
            game_state_html.append('</div>')
            
            # Join all HTML parts and append to the file
            self._append_to_html('\n'.join(game_state_html))
            
            logger.debug(f"Logged game state {turn_num} to HTML session log")
            return True

        except Exception as e:
            logger.error(f"Error logging game state to HTML: {e}")
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
            additional_notes (str, optional): Additional notes about the session
        """
        try:
            # Create summary container
            summary_html = []
            summary_html.append('<div class="summary">')
            summary_html.append('<h2>Session Summary</h2>')
            summary_html.append(f'<p><strong>Total turns played:</strong> {total_turns}</p>')
            
            if final_score is not None:
                summary_html.append(f'<p><strong>Final score:</strong> {final_score}</p>')
            
            if victory_type is not None:
                summary_html.append(f'<p><strong>Victory type:</strong> {victory_type}</p>')
            
            if additional_notes is not None:
                summary_html.append('<h3>Additional Notes</h3>')
                summary_html.append(f'<p>{additional_notes.replace("\n", "<br>")}</p>')
            
            summary_html.append(f'<p><strong>Session ended:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
            summary_html.append('</div>')
            
            # Join all HTML parts and append to the file
            self._append_to_html('\n'.join(summary_html))
            
            logger.info(f"Session summary logged to HTML file {self.log_file}")
            return True

        except Exception as e:
            logger.error(f"Error logging session summary to HTML: {e}")
            return False
