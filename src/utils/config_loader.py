"""
Configuration loader for the Civ6-AI project.
This module implements a singleton pattern to ensure configurations are loaded only once.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Find the project root directory (where .env file is located)
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ENV_FILE = PROJECT_ROOT / ".env"

# Configuration cache
_CONFIG_CACHE = {}
_ENV_LOADED = False

# Load environment variables from .env file (only once)
def _ensure_env_loaded():
    global _ENV_LOADED
    if not _ENV_LOADED:
        if ENV_FILE.exists():
            load_dotenv(dotenv_path=ENV_FILE)
            logger.info("Loaded environment variables from %s", ENV_FILE)
        else:
            logger.warning(".env file not found at %s", ENV_FILE)
        _ENV_LOADED = True


def load_config(config_path):
    """
    Load configuration from a YAML file. Implements a singleton pattern
    to ensure each config file is loaded only once.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    # Ensure environment variables are loaded
    _ensure_env_loaded()
    
    config_path = Path(config_path)
    cache_key = str(config_path.absolute())
    
    # Return cached config if available
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]
    
    if not config_path.exists():
        logger.warning("Configuration file not found: %s", config_path)
        _CONFIG_CACHE[cache_key] = {}
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info("Loaded configuration from %s", config_path)
        _CONFIG_CACHE[cache_key] = config
        return config

    except Exception as e:
        logger.error("Error loading configuration from %s: %s", config_path, e)
        _CONFIG_CACHE[cache_key] = {}
        return {}


def get_env_var(var_name, default=None):
    """
    Get environment variable with fallback to default.

    Args:
        var_name (str): Name of the environment variable
        default: Default value if environment variable is not set

    Returns:
        Value of the environment variable or default
    """
    # Ensure environment variables are loaded
    _ensure_env_loaded()
    
    value = os.environ.get(var_name, default)
    if value is not None:
        logger.debug("Using environment variable %s", var_name)
    return value


def load_api_config():
    """
    Load API configuration from environment variables.
    Implements a singleton pattern to ensure the API config is loaded only once.

    Returns:
        dict: API configuration dictionary
    """
    cache_key = 'api_config'
    
    # Return cached config if available
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]
        
    api_config = {
        "qwen": {
            "api_key": get_env_var("QWEN_API_KEY"),
            "model": get_env_var("QWEN_MODEL", "qwen-vl-max-latest"),
            "base_url": get_env_var(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        },
        "ollama": {
            "api_url": get_env_var("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
            "model": get_env_var("OLLAMA_MODEL", "llava:latest"),
        }
    }
    
    # Log API configuration status (without exposing sensitive information)
    if api_config["qwen"]["api_key"]:
        logger.info("Qwen API key loaded from environment variables")
    else:
        logger.warning("Qwen API key not found in environment variables")
    
    # Cache the result
    _CONFIG_CACHE[cache_key] = api_config
    return api_config


# Alias for backward compatibility
get_api_config = load_api_config
