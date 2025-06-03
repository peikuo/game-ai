"""
Utility functions for image processing and optimization.

This module provides centralized image processing functions to be used
across the codebase, ensuring consistent image handling.
"""

import io
import logging
import base64
from typing import Tuple, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Standard image dimensions for processing throughout the application
STANDARD_MAX_WIDTH = 1280
STANDARD_MAX_HEIGHT = 720


def optimize_image(
    image: Image.Image,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    optimize_colors: bool = True,
    compression_level: int = 9,
    # quality parameter is unused but kept for backwards compatibility
    quality: int = 85,
    force: bool = False,
    # Option to convert to JPEG for even smaller size
    convert_to_jpeg: bool = False,
    # JPEG quality setting
    jpeg_quality: int = 80,
) -> Tuple[Image.Image, dict]:
    """
    Optimize an image for size without significant quality loss.
    
    This function applies multiple optimization techniques:
    1. Resizing (if max_width is provided)
    2. Color palette optimization (if optimize_colors is True and image has <256 colors)
    3. Lossless PNG compression
    
    Args:
        image: PIL Image to optimize
        max_width: Maximum width to resize to (preserves aspect ratio)
        optimize_colors: Whether to optimize colors using palette conversion
        compression_level: PNG compression level (1-9)
        quality: Quality level for compression (0-100)
        
    Returns:
        Tuple containing:
        - Optimized PIL Image
        - Dictionary with optimization statistics
    """
    # Check if the image is already optimized (unless force=True)
    if not force and hasattr(image, 'info') and image.info.get('optimized', False):
        # Image is already optimized, return it as is with minimal stats
        logger.debug("Image already optimized, skipping optimization")
        return image, {
            "already_optimized": True,
            "original_size": {"width": image.width, "height": image.height},
            "compression_ratio": 1.0  # No additional compression
        }
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    stats = {
        "original_size": {"width": img.width, "height": img.height},
        "original_mode": img.mode,
        "already_optimized": False
    }
    
    # Get original dimensions
    width, height = img.size
    
    # Step 1: Resize if needed
    resize_needed = False
    new_width, new_height = width, height
    
    # Check if width exceeds max_width
    if max_width and width > max_width:
        resize_needed = True
        new_width = max_width
        new_height = int(height * (max_width / width))
    
    # Check if height exceeds max_height (after potential width adjustment)
    if max_height and new_height > max_height:
        resize_needed = True
        # Recalculate width to maintain aspect ratio
        ratio = max_height / new_height
        new_height = max_height
        new_width = int(new_width * ratio)
    
    # Perform resize if needed
    if resize_needed:
        # Resize with high-quality resampling
        img = img.resize((new_width, new_height), Image.LANCZOS)
        stats["resized"] = True
        stats["resized_size"] = {"width": new_width, "height": new_height}
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    else:
        stats["resized"] = False
    
    # Step 2: Optimize color palette if possible
    if optimize_colors and img.mode in ("RGB", "RGBA"):
        # Check if the image can be converted to P mode (palette) without quality loss
        # This works well for games with limited color palettes
        try:
            # Sample a subset of pixels for faster color counting (for large images)
            # Completely skip color sampling for high-resolution images
            # This avoids the IndexError warnings and improves performance
            if width * height > 1000000:  # For images larger than 1 megapixel
                logger.info(f"Skipping color optimization for large image ({width}x{height})")
                unique_colors = 1000  # Assume many colors for large images
                stats["color_sampling_skipped"] = True
            else:
                # For smaller images, do a full color analysis
                try:
                    unique_colors = len(set(img.getdata()))
                    stats["unique_colors"] = unique_colors
                    logger.debug(f"Image has {unique_colors} unique colors")
                except Exception as e:
                    logger.warning(f"Error analyzing colors: {e}")
                    unique_colors = 1000  # Assume many colors
            
            if unique_colors < 256:
                # Use median cut for better quality
                img = img.quantize(colors=min(unique_colors + 16, 256), method=2)
                stats["color_optimized"] = True
                stats["new_mode"] = img.mode
                logger.debug(f"Converted image to palette mode (unique colors: {unique_colors})")
            else:
                stats["color_optimized"] = False
        except Exception as e:
            logger.warning(f"Color optimization failed: {e}")
            stats["color_optimized"] = False
    else:
        stats["color_optimized"] = False
    
    # Step 3: Apply compression
    original_bytes = width * height * (4 if img.mode == 'RGBA' else 3)  # Approximate size
    stats["estimated_original_bytes"] = original_bytes
    
    # Determine best format based on image content and settings
    if convert_to_jpeg and img.mode in ("RGB", "L") and not stats.get("color_optimized", False):
        # Convert to JPEG for photos or complex images without transparency
        optimized = io.BytesIO()
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save as JPEG with specified quality
        img.save(optimized, format="JPEG", optimize=True, quality=jpeg_quality)
        stats["format"] = "JPEG"
        optimized.seek(0)
        
        # Check if JPEG is actually smaller
        jpeg_size = len(optimized.getvalue())
        
        # Try PNG as well to compare
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG", optimize=True, compress_level=compression_level)
        png_size = len(png_buffer.getvalue())
        
        # If PNG is smaller, use that instead
        if png_size < jpeg_size:
            optimized = png_buffer
            stats["format"] = "PNG"
            optimized.seek(0)
    else:
        # Use PNG for graphics, screenshots, or images with transparency
        optimized = io.BytesIO()
        img.save(optimized, format="PNG", optimize=True, compress_level=compression_level)
        stats["format"] = "PNG"
        optimized.seek(0)
    
    # Get optimized size
    optimized_bytes = len(optimized.getvalue())
    stats["optimized_bytes"] = optimized_bytes
    stats["compression_ratio"] = original_bytes / optimized_bytes if optimized_bytes > 0 else 0
    
    logger.info(
        f"Image optimization: ~{original_bytes/1024:.1f}KB → {optimized_bytes/1024:.1f}KB "
        f"(ratio: {stats['compression_ratio']:.1f}x)"
    )
    
    # Return the optimized image with metadata
    result_img = Image.open(optimized)
    # Mark the image as optimized
    result_img.info['optimized'] = True
    return result_img, stats


def resize_image(
    image: Image.Image,
    max_width: int,
    preserve_aspect_ratio: bool = True,
) -> Image.Image:
    """
    Resize an image to a maximum width while preserving aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_width: Maximum width for the resized image
        preserve_aspect_ratio: Whether to preserve the aspect ratio
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if width <= max_width:
        return image.copy()
    
    if preserve_aspect_ratio:
        new_height = int(height * (max_width / width))
        return image.resize((max_width, new_height), Image.LANCZOS)
    else:
        return image.resize((max_width, height), Image.LANCZOS)


def truncate_base64(base64_str, max_length=50):
    """
    Truncate a base64 string for logging purposes.
    
    Args:
        base64_str: Base64 string to truncate
        max_length: Maximum length to show before truncating
        
    Returns:
        Truncated string with format: "first_N_chars...last_10_chars (total_length)"
    """
    if not base64_str:
        return ""
        
    total_length = len(base64_str)
    if total_length <= max_length:
        return base64_str
        
    # Show first part, ellipsis, and last 10 chars
    first_part = base64_str[:max_length]
    last_part = base64_str[-10:] if len(base64_str) > 10 else ""
    return f"{first_part}...{last_part} (length: {total_length})"


def encode_image_to_base64(image, max_width=None, optimize=False, force_optimize=False):
    """
    Encode an image to base64 string with intelligent compression without resizing dimensions.
    Automatically selects the best compression format based on image characteristics.
    
    Args:
        image: PIL Image or path to image
        max_width: No longer used (kept for backward compatibility)
        optimize: No longer used (kept for backward compatibility)
        force_optimize: No longer used (kept for backward compatibility)
        
    Returns:
        Base64 encoded image string
        
    Raises:
        ValueError: If image is None
        IOError: If image file cannot be opened
        Exception: For any other errors during encoding
    """
    if image is None:
        error_msg = "Cannot encode None image"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # If image is a string, assume it's a path and open the image
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                error_msg = f"Error opening image from path: {str(e)}"
                logger.error(error_msg)
                raise IOError(error_msg) from e
        
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            error_msg = f"Expected PIL Image, got {type(image)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Calculate original image dimensions and size
        original_width = img.width
        original_height = img.height
        original_pixels = original_width * original_height
        
        # Estimate original uncompressed size (assuming 3 bytes per pixel for RGB)
        original_bytes = original_pixels * (4 if img.mode == 'RGBA' else 3)
        original_size_kb = original_bytes / 1024
        
        logger.info(f"Original image: {original_width}x{original_height} pixels, ~{original_size_kb:.2f} KB uncompressed")
        
        # Try both PNG and JPEG compression to determine which is smaller
        png_buffer = io.BytesIO()
        jpeg_buffer = io.BytesIO()
        
        # Check if image has alpha channel (transparency)
        has_transparency = img.mode == 'RGBA' or 'transparency' in img.info
        
        # Always try PNG compression (good for screenshots with text/UI elements)
        img.save(png_buffer, format="PNG", optimize=True, compress_level=9)
        png_size = len(png_buffer.getvalue())
        
        # Try JPEG if no transparency
        jpeg_size = float('inf')  # Default to infinity if JPEG not used
        if not has_transparency:
            # Convert to RGB if needed (JPEG doesn't support other modes)
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img
                
            # Use quality=40 for game screenshots (good balance for game visuals)
            rgb_img.save(jpeg_buffer, format="JPEG", optimize=True, quality=40)
            jpeg_size = len(jpeg_buffer.getvalue())
        
        # Choose the smaller format
        if jpeg_size < png_size and not has_transparency:
            buffer = jpeg_buffer
            format_used = "JPEG"
            img_bytes = buffer.getvalue()
        else:
            buffer = png_buffer
            format_used = "PNG"
            img_bytes = buffer.getvalue()
        
        # Get size information for logging
        compressed_size_bytes = len(img_bytes)
        compressed_size_kb = compressed_size_bytes / 1024
        
        # Calculate compression ratio based on original byte estimate
        compression_ratio = original_bytes / compressed_size_bytes
        size_reduction_percent = (1 - (compressed_size_bytes / original_bytes)) * 100
        
        # Log the image size and compression details
        logger.info(f"Before compression: {original_width}x{original_height} pixels, ~{original_size_kb:.2f} KB")
        logger.info(f"After compression: {format_used}, {compressed_size_kb:.2f} KB, "
                   f"reduced by {size_reduction_percent:.1f}%, ratio: {compression_ratio:.2f}x")
        
        # Encode to base64
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_str
    
    except Exception as e:
        error_msg = f"Error encoding image to base64: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


# scale_coordinates function has been removed as it's no longer used
# The project now uses full screen capture without window detection
