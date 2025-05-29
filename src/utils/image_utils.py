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


def process_screenshot(screenshot):
    """
    Process a screenshot immediately after capture.
    This is the main entry point for screenshot processing that should be used
    by other modules.

    Args:
        screenshot: PIL Image of the screenshot

    Returns:
        The original image without optimization
        
    Raises:
        ValueError: If screenshot is None or invalid
        Exception: For any other errors during processing
    """
    if screenshot is None:
        error_msg = "Cannot process None screenshot"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    try:
        # Log original size
        if logger.level <= logging.DEBUG:
            logger.debug("Original screenshot size: %dx%d", screenshot.width, screenshot.height)
    
        # MODIFIED: Skip optimization and return original image
        logger.info(f"Skipping image optimization - returning original image {screenshot.width}x{screenshot.height}")
        
        # Keep track of image dimensions for later reference
        screenshot.original_size = (screenshot.width, screenshot.height)
    
        return screenshot
    except AttributeError as e:
        error_msg = f"Invalid screenshot object: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error processing screenshot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


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


def encode_image_to_base64(image, max_width=1024, optimize=False, force_optimize=False):
    """
    Encode an image to base64 string with resizing but no other optimization.
    
    Args:
        image: PIL Image or path to image
        max_width: Maximum width to resize to (default: 1024 for API compatibility)
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
        error_msg = "Cannot encode None image to base64"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Load image if it's a path
        if isinstance(image, str) or hasattr(image, 'read'):
            try:
                image = Image.open(image)
            except Exception as e:
                error_msg = f"Failed to open image: {str(e)}"
                logger.error(error_msg)
                raise IOError(error_msg) from e
        
        # Make a copy to avoid modifying the original
        img = image.copy()
        original_size = img.size
        
        # MODIFIED: Only resize the image for API compatibility, but no other optimization
        # We still need to resize because the API has size limits
        if max_width and img.width > max_width:
            # Calculate new height maintaining aspect ratio
            aspect_ratio = img.height / img.width
            new_height = int(max_width * aspect_ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {original_size[0]}x{original_size[1]} to {img.width}x{img.height} for API compatibility")
        else:
            logger.info(f"Using original image size: {img.width}x{img.height}")
        
        # Encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Log a truncated version of the base64 string
        logger.debug("Encoded image to base64: %s", truncate_base64(base64_str))
        
        return base64_str
    except (ValueError, IOError) as e:
        # Let these specific exceptions propagate up
        raise
    except Exception as e:
        error_msg = f"Error encoding image to base64: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def scale_coordinates(coords, original_size, resized_size):
    """
    Scale coordinates from a resized image back to the original image size.
    
    Args:
        coords: Tuple of (x, y, width, height) coordinates in the resized image
        original_size: Tuple of (width, height) of the original image
        resized_size: Tuple of (width, height) of the resized image
        
    Returns:
        Tuple of (x, y, width, height) coordinates scaled to the original image
        
    Raises:
        ValueError: If coordinates are invalid
        Exception: For any other errors during scaling
    """
    if coords is None:
        error_msg = "Cannot scale None coordinates"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Extract coordinates
        x, y, width, height = coords
        
        # Make sure coordinates are integers
        if not all(isinstance(coord, int) for coord in [x, y, width, height]):
            logger.warning(f"Non-integer coordinates detected: {coords}")
            # Convert to integers
            x, y, width, height = int(x), int(y), int(width), int(height)
        
        # Ensure width and height are positive
        if width <= 0 or height <= 0:
            error_msg = f"Invalid dimensions detected: width={width}, height={height}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # MODIFIED: Apply a fixed 2x scaling factor for Retina displays
        fixed_scale = 2.0
        
        # Scale coordinates
        scaled_x = int(x * fixed_scale)
        scaled_y = int(y * fixed_scale)
        scaled_width = int(width * fixed_scale)
        scaled_height = int(height * fixed_scale)
        
        # Ensure scaled coordinates are valid
        scaled_x = max(0, scaled_x)  # Ensure x is not negative
        scaled_y = max(0, scaled_y)  # Ensure y is not negative
        
        # Get original_width and original_height if available
        if original_size:
            original_width, original_height = original_size
            
            # Ensure width and height are reasonable
            scaled_width = min(scaled_width, original_width - scaled_x)  # Ensure width doesn't exceed image
            scaled_height = min(scaled_height, original_height - scaled_y)  # Ensure height doesn't exceed image
        
        logger.info(f"Scaling coordinates: ({x}, {y}, {width}, {height}) → "
                  f"({scaled_x}, {scaled_y}, {scaled_width}, {scaled_height}), "
                  f"using fixed scale factor: {fixed_scale:.2f}")
        
        return (scaled_x, scaled_y, scaled_width, scaled_height)
    except ValueError as e:
        # Let ValueError propagate up
        raise
    except Exception as e:
        error_msg = f"Error scaling coordinates: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
