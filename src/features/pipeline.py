"""
Feature Extraction Pipeline Module
Implements the FACADE pattern - exposes a single simple function that
orchestrates the complex multi-model extraction process.

Key Feature: FAIL-FAST logic to reject blurry images before running
expensive vision models (YOLO, CLIP).
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image
import logging

import sys
sys.path.append(str(__file__).rsplit('src', 1)[0])
from src.config import QUALITY_THRESHOLDS, APP_CONFIG
from src.features.cv_ops import analyze_image_quality, calculate_sharpness_in_region
from src.features.vision_models import get_yolo_detector, get_clip_classifier

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionResult:
    """
    Complete feature extraction result matching the exact JSON schema.
    Now includes context_signals for fashion/model detection.
    """
    # Status
    status: str  # "SUCCESS", "REJECTED", "ERROR"
    rejection_reason: Optional[str] = None
    
    # Tier 1: Object & Content (YOLO)
    objects_detected: Optional[list] = None
    object_count: Optional[int] = None
    has_people: Optional[bool] = None
    primary_object_area_percent: Optional[float] = None
    
    # Tier 2: Semantic Understanding (CLIP)
    clip_scene_type: Optional[str] = None
    clip_style: Optional[str] = None
    background_complexity: Optional[str] = None
    
    # Tier 3: Technical Quality (OpenCV)
    sharpness_score: Optional[float] = None
    sharpness_category: Optional[str] = None
    exposure_category: Optional[str] = None
    text_detected: Optional[bool] = None
    
    # Context Signals (Derived)
    is_fashion_context: Optional[bool] = None  # Person + Clothing/Studio = Fashion
    is_product_text: Optional[bool] = None     # Text on product, not watermark
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and internal fields."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and not key.startswith('_'):
                result[key] = value
        return result
    
    def to_llm_context(self) -> Dict[str, Any]:
        """
        Convert to dictionary optimized for LLM consumption.
        Removes status fields and internal metadata.
        Converts numpy types to native Python types for JSON serialization.
        """
        import numpy as np
        
        def convert_value(v):
            """Convert numpy types to Python native types."""
            if isinstance(v, (np.floating, np.float32, np.float64)):
                return float(v)
            elif isinstance(v, (np.integer, np.int32, np.int64)):
                return int(v)
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            return v
        
        exclude_keys = {'status', 'rejection_reason'}
        result = {}
        for key, value in asdict(self).items():
            if value is not None and not key.startswith('_') and key not in exclude_keys:
                result[key] = convert_value(value)
        return result


def load_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load image from various input types into BGR numpy array.
    
    Args:
        image_input: File path, numpy array, or PIL Image
        
    Returns:
        BGR numpy array
    """
    if isinstance(image_input, np.ndarray):
        # Already numpy array
        if len(image_input.shape) == 2:
            # Grayscale, convert to BGR
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            # RGBA, convert to BGR
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        return image_input
    
    elif isinstance(image_input, Image.Image):
        # PIL Image
        rgb_array = np.array(image_input.convert('RGB'))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    elif isinstance(image_input, (str, Path)):
        # File path
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        return image
    
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_for_processing(image: np.ndarray, max_dim: int = None) -> np.ndarray:
    """
    Resize image if it exceeds maximum dimension (for performance).
    
    Args:
        image: BGR numpy array
        max_dim: Maximum dimension (uses config default if not specified)
        
    Returns:
        Resized image (or original if already small enough)
    """
    if max_dim is None:
        max_dim = APP_CONFIG.max_image_dimension
    
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dim:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def consolidate_features(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    run_full_analysis: bool = True
) -> FeatureExtractionResult:
    """
    FACADE FUNCTION: Main entry point for feature extraction.
    
    Orchestrates the complete pipeline:
    1. Load and preprocess image
    2. FAIL-FAST: Quick blur check using OpenCV
    3. If passed: Run YOLO object detection
    4. Run CLIP semantic analysis
    5. Complete OpenCV quality metrics
    6. Consolidate into exact JSON schema
    
    Args:
        image_input: Image as file path, numpy array, or PIL Image
        run_full_analysis: If False, only run fail-fast check (for testing)
        
    Returns:
        FeatureExtractionResult with all extracted features
    """
    try:
        # Step 1: Load image
        image = load_image(image_input)
        original_shape = image.shape[:2]
        
        # Step 2: Resize for processing (performance optimization)
        image = resize_for_processing(image)
        
        # Step 3: FAIL-FAST - Quick blur check on center region
        # We check center first to save compute on obviously blurry images
        initial_sharpness = calculate_sharpness_in_region(image, bbox=None)
        
        if initial_sharpness.is_fail_fast:
            logger.info(f"Image rejected by fail-fast (sharpness: {initial_sharpness.score})")
            return FeatureExtractionResult(
                status="REJECTED",
                rejection_reason="Image is too blurry (failed quality threshold)",
                sharpness_score=initial_sharpness.score,
                sharpness_category=initial_sharpness.category
            )
        
        if not run_full_analysis:
            # Early exit for testing fail-fast only
            return FeatureExtractionResult(
                status="SUCCESS",
                sharpness_score=initial_sharpness.score,
                sharpness_category=initial_sharpness.category
            )
        
        # Step 4: Run YOLO object detection
        logger.info("Running YOLO object detection...")
        yolo = get_yolo_detector()
        yolo_results = yolo.detect(image)
        
        # Step 5: Region-aware sharpness (use detected object bbox)
        # This distinguishes intentional bokeh from blur
        subject_bbox = yolo_results.get("_primary_object_bbox")
        if subject_bbox is not None:
            # Re-calculate sharpness focused on the detected subject
            subject_sharpness = calculate_sharpness_in_region(image, bbox=subject_bbox)
            
            # Check if subject itself is blurry (even if background is sharp)
            if subject_sharpness.is_fail_fast:
                logger.info(f"Subject is blurry (sharpness: {subject_sharpness.score})")
                return FeatureExtractionResult(
                    status="REJECTED",
                    rejection_reason="Main subject is blurry",
                    sharpness_score=subject_sharpness.score,
                    sharpness_category=subject_sharpness.category,
                    objects_detected=yolo_results.get("objects_detected"),
                    object_count=yolo_results.get("object_count")
                )
            sharpness_result = subject_sharpness
        else:
            sharpness_result = initial_sharpness
        
        # Step 6: Run CLIP semantic analysis
        logger.info("Running CLIP semantic analysis...")
        clip = get_clip_classifier()
        clip_results = clip.analyze(image)
        
        # Step 7: Complete OpenCV quality analysis
        logger.info("Running OpenCV quality analysis...")
        cv_results = analyze_image_quality(image, subject_bbox)
        
        # Step 8: CONTEXT DETECTION - Determine if this is fashion/model photography
        detected_objects = set(obj.lower() for obj in yolo_results.get("objects_detected", []))
        has_people = yolo_results.get("has_people", False)
        clip_scene = clip_results.get("clip_scene_type", "unknown")
        clip_style = clip_results.get("clip_style", "unknown")
        text_detected = cv_results.get("text_detected", False)
        sharpness_cat = sharpness_result.category
        
        # Fashion/Apparel items that indicate model photography
        fashion_items = {
            'tie', 'backpack', 'handbag', 'suitcase', 'umbrella',
            'shirt', 'dress', 'coat', 'jacket', 'shoe', 'sandal',
            'boot', 'hat', 'cap', 'glasses', 'watch', 'belt'
        }
        
        # Detect fashion context: Person + (Fashion Item OR Studio Setting)
        is_fashion_context = False
        if has_people:
            # Check if any fashion items detected
            has_fashion_items = len(detected_objects.intersection(fashion_items)) > 0
            # Check if studio/professional setting
            is_studio = clip_scene in ['studio_product', 'lifestyle']
            is_professional = clip_style == 'professional'
            
            if has_fashion_items or is_studio or is_professional:
                is_fashion_context = True
                logger.info("Fashion context detected: Model/apparel photography")
        
        # Detect product text: Text on product (not watermark)
        # Text is allowed if: image is sharp + studio setting + not on edges
        is_product_text = False
        if text_detected:
            if sharpness_cat in ['Sharp', 'Soft'] and clip_scene == 'studio_product':
                is_product_text = True
                logger.info("Product text detected: Likely brand/design, not watermark")
        
        # Step 9: Consolidate all results with context signals
        return FeatureExtractionResult(
            status="SUCCESS",
            rejection_reason=None,
            
            # Tier 1: YOLO
            objects_detected=yolo_results.get("objects_detected", []),
            object_count=yolo_results.get("object_count", 0),
            has_people=has_people,
            primary_object_area_percent=yolo_results.get("primary_object_area_percent", 0.0),
            
            # Tier 2: CLIP
            clip_scene_type=clip_scene,
            clip_style=clip_style,
            background_complexity=clip_results.get("background_complexity", "unknown"),
            
            # Tier 3: OpenCV
            sharpness_score=sharpness_result.score,
            sharpness_category=sharpness_cat,
            exposure_category=cv_results.get("exposure_category", "unknown"),
            text_detected=text_detected,
            
            # Context Signals
            is_fashion_context=is_fashion_context,
            is_product_text=is_product_text
        )
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        return FeatureExtractionResult(
            status="ERROR",
            rejection_reason=f"File not found: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return FeatureExtractionResult(
            status="ERROR",
            rejection_reason=f"Processing error: {str(e)}"
        )


def batch_consolidate_features(
    image_paths: list,
    progress_callback: callable = None
) -> list:
    """
    Process multiple images and return consolidated features for each.
    
    Args:
        image_paths: List of image file paths
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of (path, FeatureExtractionResult) tuples
    """
    results = []
    total = len(image_paths)
    
    for i, path in enumerate(image_paths):
        try:
            result = consolidate_features(path)
            results.append((str(path), result))
        except Exception as e:
            results.append((str(path), FeatureExtractionResult(
                status="ERROR",
                rejection_reason=str(e)
            )))
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results
