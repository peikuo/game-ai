#!/usr/bin/env python
"""
Test script to verify PyAutoGUI click functionality.
This script tests basic mouse movement and clicking to ensure PyAutoGUI is working correctly.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pyautogui_test")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    logger.info("PyAutoGUI is available")
    # Set safety features
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.5      # Add small pause between commands
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.error("PyAutoGUI is not installed. Install with: pip install pyautogui")
    sys.exit(1)


def get_screen_info():
    """Get and display screen information"""
    width, height = pyautogui.size()
    logger.info(f"Screen resolution: {width}x{height}")
    return width, height


def test_190_394_info():
    time.sleep(3)
    pyautogui.moveTo(1, 1, duration=1)
    pyautogui.moveTo(190, 394, duration=2)
    pyautogui.moveTo(260, 455, duration=2)
    pyautogui.click(x=260, y=455, button="left")


def test_mouse_movement():
    """Test basic mouse movement"""
    logger.info("Testing mouse movement...")
    
    # Get current mouse position
    start_x, start_y = pyautogui.position()
    logger.info(f"Current mouse position: ({start_x}, {start_y})")
    
    # Move to center of screen
    width, height = pyautogui.size()
    center_x, center_y = width // 2, height // 2
    
    logger.info(f"Moving mouse to center of screen: ({center_x}, {center_y})")
    pyautogui.moveTo(center_x, center_y, duration=1)
    time.sleep(1)
    
    # Move to each corner with 50px offset
    corners = [
        (50, 50, "top-left"),
        (width - 50, 50, "top-right"),
        (width - 50, height - 50, "bottom-right"),
        (50, height - 50, "bottom-left")
    ]
    
    for x, y, position in corners:
        logger.info(f"Moving mouse to {position} corner: ({x}, {y})")
        pyautogui.moveTo(x, y, duration=1)
        time.sleep(0.5)
    
    # Return to original position
    logger.info(f"Returning to original position: ({start_x}, {start_y})")
    pyautogui.moveTo(start_x, start_y, duration=1)
    
    logger.info("Mouse movement test completed")

def test_mouse_click(click_test=False):
    """
    Test mouse clicking
    
    Args:
        click_test: If True, actually perform clicks. If False, just simulate.
    """
    logger.info("Testing mouse clicking...")
    
    # Create a test area
    width, height = pyautogui.size()
    center_x, center_y = width // 2, height // 2
    
    # Define test points (relative to center)
    test_points = [
        (0, 0, "center"),
        (-100, -100, "top-left of center"),
        (100, -100, "top-right of center"),
        (100, 100, "bottom-right of center"),
        (-100, 100, "bottom-left of center")
    ]
    
    for dx, dy, position in test_points:
        x, y = center_x + dx, center_y + dy
        logger.info(f"Moving to {position}: ({x}, {y})")
        pyautogui.moveTo(x, y, duration=1)
        
        if click_test:
            logger.info(f"Clicking at {position}")
            pyautogui.click(x=x, y=y)
            time.sleep(0.5)
        else:
            logger.info(f"Simulating click at {position} (no actual click)")
            time.sleep(0.5)
    
    logger.info("Mouse click test completed")

def create_test_window():
    """Create a simple test window to click on (requires tkinter)"""
    try:
        import tkinter as tk
        from tkinter import ttk
        
        logger.info("Creating test window...")
        
        root = tk.Tk()
        root.title("PyAutoGUI Click Test")
        root.geometry("400x300")
        
        # Center the window
        width, height = 400, 300
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add some buttons to click
        frame = ttk.Frame(root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        click_count = {"value": 0}
        
        def increment_counter():
            click_count["value"] += 1
            counter_label.config(text=f"Click count: {click_count['value']}")
        
        ttk.Button(frame, text="Click Me!", command=increment_counter).pack(pady=10)
        counter_label = ttk.Label(frame, text="Click count: 0")
        counter_label.pack(pady=10)
        
        ttk.Label(frame, text="Move your mouse here and click\nto test PyAutoGUI functionality").pack(pady=10)
        
        # Add a quit button
        ttk.Button(frame, text="Quit", command=root.destroy).pack(pady=10)
        
        logger.info("Test window created. Close the window when testing is complete.")
        root.mainloop()
        
    except ImportError:
        logger.error("Tkinter is not available. Cannot create test window.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyAutoGUI functionality")
    parser.add_argument("--window", action="store_true", help="Create a test window to click on")
    parser.add_argument("--click", action="store_true", help="Perform actual clicks (use with caution)")
    args = parser.parse_args()
    
    logger.info("Starting PyAutoGUI test")

    test_190_394_info()
    
    # # Get screen information
    # screen_width, screen_height = get_screen_info()
    
    # # Test mouse movement
    # test_mouse_movement()
    
    # # Test mouse clicking
    # test_mouse_click(click_test=args.click)
    
    # # Create test window if requested
    # if args.window:
    #     create_test_window()
    
    logger.info("PyAutoGUI test completed")
