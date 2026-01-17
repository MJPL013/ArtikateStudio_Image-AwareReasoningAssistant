"""
Vision Models Module
Implements YOLO and CLIP model wrappers with SINGLETON pattern.

Critical: Models are loaded ONLY ONCE into memory to avoid reloading
for every image, which would be extremely slow and memory-inefficient.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append(str(__file__).rsplit('src', 1)[0])
from src.config import MODEL_CONFIG, get_scene_type_mapping, get_style_mapping, get_background_mapping


# ============================================================================
# YOLO DETECTOR - SINGLETON PATTERN
# ============================================================================

class YOLODetector:
    """
    YOLOv8 object detector with Singleton pattern.
    Ensures model is loaded only once across all usages.
    """
    
    _instance: Optional['YOLODetector'] = None
    _model = None
    _initialized: bool = False
    
    def __new__(cls) -> 'YOLODetector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not YOLODetector._initialized:
            self._load_model()
            YOLODetector._initialized = True
    
    def _load_model(self):
        """Load YOLOv8 model (happens only once)."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {MODEL_CONFIG.yolo_model}")
            self._model = YOLO(MODEL_CONFIG.yolo_model)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run object detection on an image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Dictionary with detection results matching the required schema
        """
        if self._model is None:
            raise RuntimeError("YOLO model not loaded")
        
        # Run inference
        results = self._model(
            image,
            conf=MODEL_CONFIG.yolo_confidence,
            iou=MODEL_CONFIG.yolo_iou_threshold,
            verbose=False
        )[0]
        
        # Extract detected objects
        objects_detected: List[str] = []
        has_people: bool = False
        primary_object_bbox: Optional[Tuple[int, int, int, int]] = None
        primary_object_area: float = 0.0
        
        image_area = image.shape[0] * image.shape[1]
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get class name
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]
                objects_detected.append(class_name)
                
                # Check for people
                if class_name.lower() in ["person", "people", "human"]:
                    has_people = True
                
                # Calculate area and track primary (largest) object
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area > primary_object_area:
                    primary_object_area = box_area
                    primary_object_bbox = (int(x1), int(y1), int(x2), int(y2))
        
        # Calculate primary object area percentage
        primary_object_area_percent = (primary_object_area / image_area * 100) if image_area > 0 else 0.0
        
        return {
            "objects_detected": list(set(objects_detected)),  # Deduplicate
            "object_count": len(objects_detected),
            "has_people": has_people,
            "primary_object_area_percent": round(primary_object_area_percent, 2),
            "_primary_object_bbox": primary_object_bbox  # Internal use for region-aware blur
        }


# ============================================================================
# CLIP CLASSIFIER - SINGLETON PATTERN
# ============================================================================

class CLIPClassifier:
    """
    CLIP model for semantic image classification with Singleton pattern.
    Ensures model is loaded only once across all usages.
    """
    
    _instance: Optional['CLIPClassifier'] = None
    _model = None
    _processor = None
    _device = None  # Store the actual device being used
    _initialized: bool = False
    
    def __new__(cls) -> 'CLIPClassifier':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not CLIPClassifier._initialized:
            self._load_model()
            CLIPClassifier._initialized = True
    
    @staticmethod
    def _get_device() -> str:
        """Auto-detect the best available device."""
        device_config = MODEL_CONFIG.clip_device
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_config
    
    def _load_model(self):
        """Load CLIP model and processor (happens only once)."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Determine the actual device to use
            CLIPClassifier._device = self._get_device()
            
            logger.info(f"Loading CLIP model: {MODEL_CONFIG.clip_model}")
            logger.info(f"Using device: {CLIPClassifier._device}")
            
            # Force safetensors format to bypass torch.load security check (CVE-2025-32434)
            self._model = CLIPModel.from_pretrained(
                MODEL_CONFIG.clip_model,
                use_safetensors=True
            )
            self._processor = CLIPProcessor.from_pretrained(MODEL_CONFIG.clip_model)
            
            # Move to detected device
            self._model = self._model.to(CLIPClassifier._device)
            self._model.eval()
            
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _classify(self, image: np.ndarray, labels: List[str]) -> Tuple[str, float]:
        """
        Classify image against a list of text labels.
        
        Args:
            image: BGR numpy array
            labels: List of text descriptions to classify against
            
        Returns:
            Tuple of (best matching label, confidence score)
        """
        from PIL import Image
        
        # Convert BGR to RGB PIL Image
        rgb_image = image[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(rgb_image)
        
        # Process inputs
        inputs = self._processor(
            text=labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(CLIPClassifier._device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get best match
        best_idx = probs.argmax().item()
        best_label = labels[best_idx]
        confidence = probs[0][best_idx].item()
        
        return best_label, confidence
    
    def classify_scene(self, image: np.ndarray) -> str:
        """Classify the scene type of an image."""
        label, _ = self._classify(image, MODEL_CONFIG.scene_labels)
        mapping = get_scene_type_mapping()
        return mapping.get(label, "unknown")
    
    def classify_style(self, image: np.ndarray) -> str:
        """Classify the photography style of an image."""
        label, _ = self._classify(image, MODEL_CONFIG.style_labels)
        mapping = get_style_mapping()
        return mapping.get(label, "unknown")
    
    def classify_background(self, image: np.ndarray) -> str:
        """Classify the background complexity of an image."""
        label, _ = self._classify(image, MODEL_CONFIG.background_labels)
        mapping = get_background_mapping()
        return mapping.get(label, "unknown")
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run all CLIP classifications on an image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Dictionary with semantic classification results
        """
        if self._model is None:
            raise RuntimeError("CLIP model not loaded")
        
        return {
            "clip_scene_type": self.classify_scene(image),
            "clip_style": self.classify_style(image),
            "background_complexity": self.classify_background(image)
        }


# ============================================================================
# MODEL CACHE MANAGEMENT
# ============================================================================

def get_yolo_detector() -> YOLODetector:
    """Get or create the singleton YOLO detector instance."""
    return YOLODetector()


def get_clip_classifier() -> CLIPClassifier:
    """Get or create the singleton CLIP classifier instance."""
    return CLIPClassifier()


def preload_models():
    """
    Preload all models into memory.
    Call this at application startup to avoid cold-start delays.
    """
    logger.info("Preloading vision models...")
    get_yolo_detector()
    get_clip_classifier()
    logger.info("All models preloaded successfully")


def clear_model_cache():
    """
    Clear the model cache to free memory.
    Useful for testing or when switching model configurations.
    """
    YOLODetector._instance = None
    YOLODetector._model = None
    YOLODetector._initialized = False
    
    CLIPClassifier._instance = None
    CLIPClassifier._model = None
    CLIPClassifier._processor = None
    CLIPClassifier._initialized = False
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model cache cleared")
