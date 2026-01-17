"""
Configuration Module for Blind Image Reasoning System
Central location for all thresholds, model paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # YOLOv8 Configuration
    yolo_model: str = "yolov8n.pt"  # Nano version for speed
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.45
    
    # CLIP Configuration
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"  # Using CPU - upgrade PyTorch to 2.6+ with CUDA for GPU
    
    # Scene type labels for CLIP classification
    scene_labels: List[str] = field(default_factory=lambda: [
        "studio product photography",
        "outdoor scene",
        "indoor room",
        "cluttered background",
        "white background",
        "lifestyle photography"
    ])
    
    # Style labels for CLIP classification
    style_labels: List[str] = field(default_factory=lambda: [
        "professional photography",
        "amateur photography",
        "artistic photography",
        "casual snapshot"
    ])
    
    # Background complexity labels
    background_labels: List[str] = field(default_factory=lambda: [
        "minimal clean background",
        "simple background",
        "moderate background",
        "complex busy background"
    ])


@dataclass
class QualityThresholds:
    """Thresholds for image quality assessment."""
    
    # Sharpness thresholds (Laplacian variance)
    sharpness_sharp: float = 100.0      # Above this = "Sharp"
    sharpness_soft: float = 50.0        # Above this = "Soft", below = "Blurry"
    
    # Fail-fast threshold (reject immediately if below)
    fail_fast_sharpness: float = 30.0
    
    # Exposure thresholds (histogram analysis)
    exposure_dark_threshold: float = 50.0    # Mean brightness below = under-exposed
    exposure_bright_threshold: float = 200.0  # Mean brightness above = over-exposed
    
    # Object area thresholds
    min_subject_area_percent: float = 10.0   # Subject too small if below
    max_subject_area_percent: float = 95.0   # Subject too cropped if above
    
    # Normalization for sharpness score (0-100)
    sharpness_max_value: float = 500.0  # Laplacian variance cap for normalization


@dataclass
class AppConfig:
    """Application-wide configuration."""
    
    # Image processing
    max_image_dimension: int = 1024  # Resize large images for performance
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    ])
    
    # LLM Configuration
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30
    
    # UI Configuration
    batch_progress_update_interval: int = 1  # Update progress every N images


@dataclass
class APIConfig:
    """API keys and LLM configuration."""
    
    # Gemini API Key (set via environment variable GEMINI_API_KEY)
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    
    # OpenAI API Key (set via environment variable OPENAI_API_KEY)  
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Default LLM provider
    default_llm_provider: str = "gemini"  # Options: "mock", "gemini", "openai"


# Global config instances
MODEL_CONFIG = ModelConfig()
QUALITY_THRESHOLDS = QualityThresholds()
APP_CONFIG = AppConfig()
API_CONFIG = APIConfig()


def get_scene_type_mapping() -> dict:
    """Map CLIP labels to simplified scene types."""
    return {
        "studio product photography": "studio_product",
        "outdoor scene": "outdoors",
        "indoor room": "indoor",
        "cluttered background": "cluttered_room",
        "white background": "studio_product",
        "lifestyle photography": "lifestyle"
    }


def get_style_mapping() -> dict:
    """Map CLIP labels to simplified style categories."""
    return {
        "professional photography": "professional",
        "amateur photography": "amateur",
        "artistic photography": "artistic",
        "casual snapshot": "amateur"
    }


def get_background_mapping() -> dict:
    """Map CLIP labels to background complexity categories."""
    return {
        "minimal clean background": "minimal",
        "simple background": "simple",
        "moderate background": "moderate",
        "complex busy background": "complex"
    }
