"""
OpenCV Operations Module
Implements Tier 3 technical quality features using fast OpenCV math.

Features:
- Sharpness detection (Laplacian variance) with region-aware analysis
- Exposure analysis (histogram-based)
- Text/watermark detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import sys
sys.path.append(str(__file__).rsplit('src', 1)[0])
from src.config import QUALITY_THRESHOLDS


@dataclass
class SharpnessResult:
    """Result of sharpness analysis."""
    score: float           # Normalized 0-100
    category: str          # "Sharp", "Soft", "Blurry"
    raw_variance: float    # Raw Laplacian variance
    is_fail_fast: bool     # Should reject immediately?


@dataclass
class ExposureResult:
    """Result of exposure analysis."""
    category: str          # "Well-Exposed", "Over-Exposed", "Under-Exposed"
    mean_brightness: float
    histogram_spread: float


def calculate_laplacian_variance(image: np.ndarray) -> float:
    """
    Calculate Laplacian variance as a measure of image sharpness.
    Higher values indicate sharper images.
    
    Args:
        image: BGR or grayscale image
        
    Returns:
        Laplacian variance value
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return float(variance)


def calculate_sharpness_in_region(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> SharpnessResult:
    """
    Calculate sharpness score, optionally focused on a specific region.
    This allows us to distinguish between intentional bokeh (sharp subject,
    blurry background) and genuinely blurry images.
    
    Args:
        image: BGR image
        bbox: Optional (x1, y1, x2, y2) bounding box for region of interest
        
    Returns:
        SharpnessResult with score, category, and fail-fast flag
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Ensure valid coordinates
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            region = image[y1:y2, x1:x2]
        else:
            region = image
    else:
        # Default: analyze center 60% of image (where product usually is)
        h, w = image.shape[:2]
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        region = image[margin_y:h-margin_y, margin_x:w-margin_x]
    
    raw_variance = calculate_laplacian_variance(region)
    
    # Normalize to 0-100 scale
    normalized_score = min(100.0, (raw_variance / QUALITY_THRESHOLDS.sharpness_max_value) * 100)
    
    # Categorize
    if raw_variance >= QUALITY_THRESHOLDS.sharpness_sharp:
        category = "Sharp"
    elif raw_variance >= QUALITY_THRESHOLDS.sharpness_soft:
        category = "Soft"
    else:
        category = "Blurry"
    
    # Check fail-fast condition
    is_fail_fast = raw_variance < QUALITY_THRESHOLDS.fail_fast_sharpness
    
    return SharpnessResult(
        score=round(normalized_score, 2),
        category=category,
        raw_variance=round(raw_variance, 2),
        is_fail_fast=is_fail_fast
    )


def analyze_exposure(image: np.ndarray) -> ExposureResult:
    """
    Analyze image exposure using histogram analysis.
    
    Args:
        image: BGR image
        
    Returns:
        ExposureResult with category and metrics
    """
    # Convert to grayscale for brightness analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate mean brightness
    mean_brightness = float(np.mean(gray))
    
    # Calculate histogram spread (standard deviation)
    histogram_spread = float(np.std(gray))
    
    # Categorize exposure
    if mean_brightness < QUALITY_THRESHOLDS.exposure_dark_threshold:
        category = "Under-Exposed"
    elif mean_brightness > QUALITY_THRESHOLDS.exposure_bright_threshold:
        category = "Over-Exposed"
    else:
        category = "Well-Exposed"
    
    return ExposureResult(
        category=category,
        mean_brightness=round(mean_brightness, 2),
        histogram_spread=round(histogram_spread, 2)
    )


def detect_text_regions(image: np.ndarray) -> bool:
    """
    Detect if image contains text/watermarks using MSER (Maximally Stable
    Extremal Regions) - a fast method that doesn't require deep learning.
    
    Args:
        image: BGR image
        
    Returns:
        Boolean indicating if text was detected
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create MSER detector
    mser = cv2.MSER_create()
    mser.setMinArea(100)
    mser.setMaxArea(5000)
    
    # Detect regions
    regions, _ = mser.detectRegions(gray)
    
    # Filter for text-like regions (aspect ratio, size patterns)
    text_like_count = 0
    for region in regions:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(region)
        
        # Text characters typically have specific aspect ratios
        aspect_ratio = w / max(h, 1)
        
        # Text-like if: reasonable aspect ratio and not too large
        if 0.1 < aspect_ratio < 3.0 and w * h < gray.shape[0] * gray.shape[1] * 0.1:
            text_like_count += 1
    
    # Threshold: if many text-like regions clustered together
    # This is a heuristic - adjust based on your dataset
    return text_like_count > 20


def analyze_image_quality(
    image: np.ndarray,
    subject_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, Any]:
    """
    Comprehensive image quality analysis using OpenCV.
    
    Args:
        image: BGR image
        subject_bbox: Optional bounding box of main subject for region-aware blur detection
        
    Returns:
        Dictionary with all Tier 3 quality features
    """
    # Sharpness analysis (region-aware)
    sharpness = calculate_sharpness_in_region(image, subject_bbox)
    
    # Exposure analysis
    exposure = analyze_exposure(image)
    
    # Text detection
    has_text = detect_text_regions(image)
    
    return {
        "sharpness_score": sharpness.score,
        "sharpness_category": sharpness.category,
        "exposure_category": exposure.category,
        "text_detected": has_text,
        "_raw_sharpness_variance": sharpness.raw_variance,
        "_is_fail_fast": sharpness.is_fail_fast,
        "_mean_brightness": exposure.mean_brightness
    }
