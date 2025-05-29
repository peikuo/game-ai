"""
Logging utilities for the Civ6-AI project.
"""

import json
import logging
import os
import re
from pathlib import Path

from src.utils.image_utils import truncate_base64

logger = logging.getLogger(__name__)


class Base64TruncationFilter(logging.Filter):
    """
    A logging filter that truncates base64 strings in log messages.
    Specifically targets OpenAI client logs to prevent massive base64 strings
    from cluttering the logs.
    """
    
    def __init__(self, max_length=50):
        super().__init__()
        self.max_length = max_length
        self.base64_pattern = re.compile(r'"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)"')
        self.json_pattern = re.compile(r'"url": "(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)"')
        
    def filter(self, record):
        if not hasattr(record, 'msg'):
            return True
            
        # Skip if not a string
        if not isinstance(record.msg, str):
            return True
            
        # Check if this is an OpenAI client log
        if 'openai._base_client' in record.name and 'Request options' in record.msg:
            try:
                # Try to parse the log message as JSON
                if hasattr(record, 'args') and record.args:
                    # If we have args, they might contain the request options
                    for i, arg in enumerate(record.args):
                        if isinstance(arg, dict):
                            self._truncate_base64_in_dict(arg)
                
                # Also check the message itself for base64 strings
                record.msg = self._truncate_base64_in_string(record.msg)
                
            except Exception as e:
                # If anything goes wrong, just leave the log message as is
                logger.debug(f"Error truncating base64 in log message: {e}")
                
        return True
        
    def _truncate_base64_in_string(self, text):
        """Truncate base64 strings in a text string."""
        # Replace base64 strings in the format "data:image/...;base64,..."
        text = self.base64_pattern.sub(
            lambda m: f'"{truncate_base64(m.group(1), self.max_length)}"', 
            text
        )
        
        # Replace base64 strings in JSON format
        text = self.json_pattern.sub(
            lambda m: f'"url": "{truncate_base64(m.group(1), self.max_length)}"', 
            text
        )
        
        return text
        
    def _truncate_base64_in_dict(self, data):
        """Recursively truncate base64 strings in a dictionary."""
        if not isinstance(data, dict):
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                self._truncate_base64_in_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._truncate_base64_in_dict(item)
            elif isinstance(value, str) and ';base64,' in value:
                data[key] = truncate_base64(value, self.max_length)


def setup_logging(log_level=logging.INFO, log_file="game-ai.log", cleanup=True):
    """
    Set up logging configuration with optional log file cleanup.
    
    Args:
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Path to log file, or None for console-only logging
        cleanup: Whether to remove the old log file if it exists
        
    Returns:
        None
    """
    # Clean up old log file if requested
    if cleanup and log_file:
        log_path = Path(log_file)
        if log_path.exists():
            try:
                log_path.unlink()
                print(f"Removed old log file: {log_path}")
            except Exception as e:
                print(f"Failed to remove old log file: {e}")
    
    # Configure root logger
    handlers = []
    
    # Console handler (always added)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Apply the Base64TruncationFilter to the OpenAI client logger
    openai_logger = logging.getLogger('openai._base_client')
    openai_logger.addFilter(Base64TruncationFilter(max_length=50))
    
    logger.debug("Logging configured successfully")
    if log_file:
        logger.debug("Logs will be written to %s", log_file)
