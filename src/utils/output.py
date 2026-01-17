"""
Output Utilities Module
Handles saving analysis results to JSON files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_analysis_result(
    image_path: str,
    extraction_result: 'FeatureExtractionResult',
    verdict_result: 'VerdictResult',
    output_dir: str = "outputs",
    filename: Optional[str] = None
) -> str:
    """
    Save a complete analysis result to a JSON file.
    
    Args:
        image_path: Path to the original image
        extraction_result: Feature extraction result
        verdict_result: LLM verdict result
        output_dir: Directory to save outputs
        filename: Optional custom filename (auto-generated if not provided)
        
    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem if image_path else "analysis"
        filename = f"{image_name}_{timestamp}.json"
    
    # Build the comprehensive output structure
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path) if image_path else None,
            "image_filename": Path(image_path).name if image_path else None,
            "version": "2.0",
            "llm_retries": verdict_result.retry_count
        },
        
        "classification": {
            "image_type": verdict_result.image_type,
            "quality_tier": verdict_result.quality_tier,
            "confidence": verdict_result.confidence
        },
        
        "features": {
            "tier1_objects": {
                "detected_classes": extraction_result.objects_detected or [],
                "count": extraction_result.object_count or 0,
                "has_people": extraction_result.has_people or False,
                "primary_subject_size_percent": extraction_result.primary_object_area_percent or 0.0
            },
            "tier2_semantics": {
                "clip_scene_type": extraction_result.clip_scene_type or "unknown",
                "clip_style": extraction_result.clip_style or "unknown",
                "background_complexity": extraction_result.background_complexity or "unknown"
            },
            "tier3_technical": {
                "sharpness_score": extraction_result.sharpness_score or 0.0,
                "sharpness_category": extraction_result.sharpness_category or "unknown",
                "exposure_category": extraction_result.exposure_category or "unknown",
                "text_detected": extraction_result.text_detected or False
            },
            "context_signals": {
                "is_fashion_context": extraction_result.is_fashion_context or False,
                "is_product_text": extraction_result.is_product_text or False
            }
        },
        
        "suitability_assessment": {
            "ecommerce_product_page": verdict_result.ecommerce_suitable,
            "social_media_marketing": verdict_result.social_media_suitable,
            "professional_website": verdict_result.professional_suitable
        },
        
        "trust_evaluation": {
            "trust_score": verdict_result.trust_score,
            "trust_factors": verdict_result.trust_factors,
            "distrust_factors": verdict_result.distrust_factors
        },
        
        "quality_scores": {
            "overall": verdict_result.quality_score,
            "technical": verdict_result.technical_score,
            "composition": verdict_result.composition_score,
            "commercial_viability": verdict_result.commercial_score
        },
        
        "risks_detected": verdict_result.risks,
        "quality_issues": verdict_result.quality_issues,
        
        "recommendations": {
            "critical_actions": verdict_result.critical_actions,
            "improvements": verdict_result.improvements,
            "recommendation_summary": verdict_result.recommendation
        },
        
        "verdict": {
            "decision": verdict_result.verdict,
            "confidence": verdict_result.confidence,
            "primary_reason": verdict_result.primary_reason,
            "warnings": verdict_result.warnings,
            "bonuses": verdict_result.bonuses
        },
        
        "status": extraction_result.status,
        "rejection_reason": extraction_result.rejection_reason
    }
    
    # Convert numpy types
    output_data = convert_numpy_types(output_data)
    
    # Save to file
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return str(output_file)


def save_batch_results(
    results: list,
    output_dir: str = "outputs",
    filename: Optional[str] = None
) -> str:
    """
    Save batch processing results to a JSON file.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save outputs
        filename: Optional custom filename
        
    Returns:
        Path to the saved JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{timestamp}.json"
    
    # Convert numpy types
    results = convert_numpy_types(results)
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "version": "1.0"
        },
        "results": results
    }
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return str(output_file)
