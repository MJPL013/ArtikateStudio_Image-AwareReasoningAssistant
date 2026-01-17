"""
Helper Utilities Module
General utility functions for image handling, file operations, and data formatting.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(str(__file__).rsplit('src', 1)[0])
from src.config import APP_CONFIG


def get_image_files(
    directory: Union[str, Path],
    recursive: bool = False
) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to directory
        recursive: If True, search subdirectories
        
    Returns:
        List of Path objects for image files
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    image_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in APP_CONFIG.supported_extensions:
            image_files.append(path)
    
    # Sort by filename
    image_files.sort(key=lambda x: x.name.lower())
    
    return image_files


def is_valid_image(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid, readable image.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in APP_CONFIG.supported_extensions:
            return False
        
        # Try to open with PIL
        with Image.open(path) as img:
            img.verify()
        
        return True
    except Exception:
        return False


def resize_image(
    image: np.ndarray,
    max_dimension: Optional[int] = None,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize an image to fit within maximum dimensions.
    
    Args:
        image: Input image (BGR numpy array)
        max_dimension: Maximum size for largest dimension
        maintain_aspect: If True, maintain aspect ratio
        
    Returns:
        Resized image
    """
    if max_dimension is None:
        max_dimension = APP_CONFIG.max_image_dimension
    
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dimension:
        return image
    
    if maintain_aspect:
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
    else:
        new_w = new_h = max_dimension
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_for_display(
    image: np.ndarray,
    max_width: int = 800,
    max_height: int = 600
) -> np.ndarray:
    """
    Resize image for UI display.
    
    Args:
        image: Input image
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        Resized image for display
    """
    h, w = image.shape[:2]
    
    # Calculate scale factors
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    if scale >= 1.0:
        return image
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_relative_path(file_path: Union[str, Path], base_path: Union[str, Path]) -> str:
    """
    Get relative path from base directory.
    
    Args:
        file_path: Full file path
        base_path: Base directory path
        
    Returns:
        Relative path as string
    """
    try:
        return str(Path(file_path).relative_to(Path(base_path)))
    except ValueError:
        return str(file_path)


def get_image_info(file_path: Union[str, Path]) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    path = Path(file_path)
    
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            format_type = img.format
    except Exception as e:
        return {
            "error": str(e),
            "filename": path.name
        }
    
    file_size = path.stat().st_size
    
    return {
        "filename": path.name,
        "path": str(path),
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 2) if height > 0 else 0,
        "mode": mode,
        "format": format_type,
        "file_size": file_size,
        "file_size_formatted": format_file_size(file_size)
    }


def create_image_grid(
    images: List[np.ndarray],
    cols: int = 3,
    cell_size: Tuple[int, int] = (200, 200),
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of images to display
        cols: Number of columns
        cell_size: Size of each cell (width, height)
        padding: Padding between cells
        background_color: Background color (BGR)
        
    Returns:
        Grid image
    """
    n_images = len(images)
    if n_images == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    rows = (n_images + cols - 1) // cols
    
    cell_w, cell_h = cell_size
    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    
    grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # Resize image to fit cell
        img_resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        
        # Calculate position
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        
        # Place image
        grid[y:y+cell_h, x:x+cell_w] = img_resized
    
    return grid


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for display in Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR for OpenCV processing."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
