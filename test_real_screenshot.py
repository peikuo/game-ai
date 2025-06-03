#!/usr/bin/env python3
"""
Test script to verify image compression with a more realistic game screenshot.
"""

import io
import base64
import random
import numpy as np
from PIL import Image, ImageDraw

# Import our image utilities
from src.utils.image_utils import encode_image_to_base64

def create_game_screenshot(width=1920, height=1080):
    """Create a simulated game screenshot with UI elements and graphics."""
    # Create base image with gradient background (sky-like)
    img = Image.new('RGB', (width, height), color=(100, 150, 255))
    pixels = np.array(img)
    
    # Add gradient
    for y in range(height):
        factor = y / height
        pixels[y, :, 0] = 100 - int(50 * factor)  # R
        pixels[y, :, 1] = 150 - int(50 * factor)  # G
        pixels[y, :, 2] = 255 - int(100 * factor)  # B
    
    # Convert back to PIL Image
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    # Add ground/terrain (green area at bottom)
    draw.rectangle([(0, int(height * 0.7)), (width, height)], 
                  fill=(30, 120, 30))
    
    # Add some "buildings" or structures
    for i in range(10):
        x = random.randint(0, width)
        w = random.randint(100, 300)
        h = random.randint(100, 400)
        y = int(height * 0.7) - h
        draw.rectangle([(x, y), (x + w, int(height * 0.7))], 
                      fill=(100, 100, 100))
        
        # Add windows to buildings
        for j in range(5):
            for k in range(8):
                if random.random() > 0.3:  # Some windows are dark
                    wx = x + 20 + j * 50
                    wy = y + 30 + k * 40
                    if wx < x + w - 10 and wy < int(height * 0.7) - 10:
                        draw.rectangle([(wx, wy), (wx + 20, wy + 25)], 
                                      fill=(255, 255, 200))
    
    # Add UI elements (HUD-like elements)
    
    # Health bar
    draw.rectangle([(50, 50), (300, 80)], fill=(0, 0, 0), outline=(255, 255, 255))
    draw.rectangle([(55, 55), (55 + 240, 75)], fill=(255, 50, 50))
    
    # Mini-map
    draw.rectangle([(width - 220, 50), (width - 50, 220)], 
                  fill=(30, 30, 30), outline=(255, 255, 255))
    
    # Add some dots on minimap
    for i in range(10):
        mx = width - 220 + random.randint(10, 160)
        my = 50 + random.randint(10, 160)
        dot_size = random.randint(3, 8)
        dot_color = (0, 255, 0) if random.random() > 0.7 else (255, 0, 0)
        draw.ellipse([(mx, my), (mx + dot_size, my + dot_size)], fill=dot_color)
    
    # Add text-like elements (menu items)
    menu_items = ["Inventory", "Skills", "Map", "Quests", "Settings"]
    for i, item in enumerate(menu_items):
        y_pos = 300 + i * 40
        # Text background
        draw.rectangle([(width - 150, y_pos), (width - 20, y_pos + 30)], 
                      fill=(50, 50, 50, 128))
        # We can't easily draw text, so simulate with lines
        draw.line([(width - 130, y_pos + 15), (width - 40, y_pos + 15)], 
                 fill=(255, 255, 255), width=2)
    
    # Add a "character" in the center
    center_x, center_y = width // 2, height // 2
    # Character body
    draw.ellipse([(center_x - 50, center_y - 100), 
                 (center_x + 50, center_y + 100)], 
                fill=(200, 150, 100))
    # Character head
    draw.ellipse([(center_x - 30, center_y - 160), 
                 (center_x + 30, center_y - 100)], 
                fill=(200, 150, 100))
    
    return img

def main():
    # Create a simulated game screenshot
    print("Creating simulated game screenshot...")
    screenshot = create_game_screenshot()
    
    # Measure original size (PNG format)
    original_buffer = io.BytesIO()
    screenshot.save(original_buffer, format='PNG')
    original_size = len(original_buffer.getvalue())
    
    # Measure original size (JPEG format with quality=95)
    jpeg_buffer = io.BytesIO()
    screenshot.save(jpeg_buffer, format='JPEG', quality=95)
    jpeg_size = len(jpeg_buffer.getvalue())
    
    # Measure original size (JPEG format with quality=40)
    jpeg_low_buffer = io.BytesIO()
    screenshot.save(jpeg_low_buffer, format='JPEG', quality=40)
    jpeg_low_size = len(jpeg_low_buffer.getvalue())
    
    # Use our compression function
    compressed_base64 = encode_image_to_base64(screenshot)
    compressed_bytes = base64.b64decode(compressed_base64)
    compressed_size = len(compressed_bytes)
    
    # Print results
    print(f'Original PNG size: {original_size/1024:.2f} KB')
    print(f'Original JPEG (q=95) size: {jpeg_size/1024:.2f} KB')
    print(f'Original JPEG (q=40) size: {jpeg_low_size/1024:.2f} KB')
    print(f'Our compressed size: {compressed_size/1024:.2f} KB')
    print(f'Size reduction vs PNG: {(1 - compressed_size/original_size) * 100:.2f}%')
    print(f'Size reduction vs JPEG (q=95): {(1 - compressed_size/jpeg_size) * 100:.2f}%')
    print(f'Size reduction vs JPEG (q=40): {(1 - compressed_size/jpeg_low_size) * 100:.2f}%')
    print(f'Dimensions preserved: {screenshot.size}')
    
    # Save the test image for visual inspection
    screenshot.save("test_screenshot.png")
    
    # Save compressed version for comparison
    compressed_img = Image.open(io.BytesIO(compressed_bytes))
    compressed_img.save("test_screenshot_compressed.png")
    print("Saved test images to test_screenshot.png and test_screenshot_compressed.png")

if __name__ == "__main__":
    main()
