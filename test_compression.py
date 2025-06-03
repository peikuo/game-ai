#!/usr/bin/env python3
"""
Test script to verify image compression without dimension reduction.
"""

import io
import base64
from PIL import Image, ImageDraw

# Import our image utilities
from src.utils.image_utils import encode_image_to_base64

def main():
    # Create a test image with some patterns (to better simulate a game screenshot)
    print("Creating test image...")
    test_img = Image.new('RGB', (1920, 1080), color='white')
    draw = ImageDraw.Draw(test_img)
    
    # Add a grid pattern
    for i in range(0, 1920, 20):
        draw.line([(i, 0), (i, 1080)], fill=(200, 200, 200), width=1)
    for i in range(0, 1080, 20):
        draw.line([(0, i), (1920, i)], fill=(200, 200, 200), width=1)
    
    # Add some colored shapes
    for i in range(10):
        draw.rectangle(
            [100*i, 100*i, 100*i+400, 100*i+400], 
            fill=(255, 0, 0)
        )
    draw.ellipse([500, 200, 900, 600], fill=(0, 255, 0))
    draw.polygon([(1000, 200), (1400, 200), (1200, 600)], fill=(0, 0, 255))
    
    # Measure original size (PNG format)
    original_buffer = io.BytesIO()
    test_img.save(original_buffer, format='PNG')
    original_size = len(original_buffer.getvalue())
    
    # Measure original size (JPEG format with quality=95)
    jpeg_buffer = io.BytesIO()
    test_img.save(jpeg_buffer, format='JPEG', quality=95)
    jpeg_size = len(jpeg_buffer.getvalue())
    
    # Use our compression function
    compressed_base64 = encode_image_to_base64(test_img)
    compressed_bytes = base64.b64decode(compressed_base64)
    compressed_size = len(compressed_bytes)
    
    # Print results
    print(f'Original PNG size: {original_size/1024:.2f} KB')
    print(f'Original JPEG (q=95) size: {jpeg_size/1024:.2f} KB')
    print(f'Our compressed size: {compressed_size/1024:.2f} KB')
    print(f'Size reduction vs PNG: {(1 - compressed_size/original_size) * 100:.2f}%')
    print(f'Size reduction vs JPEG (q=95): {(1 - compressed_size/jpeg_size) * 100:.2f}%')
    print(f'Dimensions preserved: {test_img.size}')

if __name__ == "__main__":
    main()
