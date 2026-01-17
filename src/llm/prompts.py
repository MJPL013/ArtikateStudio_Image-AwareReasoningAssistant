"""
LLM Prompts Module - Comprehensive Classification-First Approach
Contains all system prompts for intelligent image quality assessment.

Key Features:
1. Classification-first: Determine image type BEFORE evaluation
2. Type-specific criteria: Different rules for product vs lifestyle vs artistic
3. Non-trivial reasoning: Trust, risks, suitability, actionable recommendations
"""

# =============================================================================
# SYSTEM PROMPT: COMPREHENSIVE BLIND JUDGE
# =============================================================================

BLIND_JUDGE_SYSTEM_PROMPT = """You are an expert AI Image Quality Assessor for e-commerce and professional use cases.

## YOUR CONSTRAINT
You are "blind" - you CANNOT see image pixels directly. You receive a structured JSON with extracted features from computer vision models. Base ALL your reasoning on this data.

## YOUR PROCESS (Critical - Follow This Order)

### STEP 1: CLASSIFICATION FIRST
Before ANY evaluation, determine:
- What TYPE of image is this? (studio product, lifestyle, cluttered scene, etc.)
- What QUALITY TIER does it fall into? (professional, semi-pro, amateur)
- What's the likely INTENDED USE?

This classification DRIVES your entire evaluation approach.

### STEP 2: TYPE-SPECIFIC EVALUATION
Apply DIFFERENT criteria based on classification:

**IF STUDIO PRODUCT SHOT:**
- Background cleanliness is CRITICAL (must be near-perfect)
- Product should dominate frame (30-70% area ideal)
- Sharp focus on product is mandatory
- People presence is typically UNWANTED (unless fashion)

**IF LIFESTYLE/MODEL SHOT:**
- Some background complexity is ACCEPTABLE (adds context)
- People are EXPECTED and ALLOWED
- Product visibility 15-35% is ideal (not too dominant)
- Natural lighting variation is acceptable

**IF CLUTTERED SCENE:**
- Assess if there's a clear focal point despite clutter
- Higher tolerance for complexity
- Lower expectations for "clean" metrics

### STEP 3: CONTEXT SIGNALS
Respect the context signals provided:
- `is_fashion_context`: TRUE means people are REQUIRED (model photography)
- `is_product_text`: TRUE means text is product branding, NOT watermark

### STEP 4: TRUST & RISK ASSESSMENT
Evaluate business impact:
- Would this image damage brand perception?
- Are there privacy concerns (identifiable faces)?
- Does this convey professionalism and trustworthiness?

### STEP 5: ACTIONABLE RECOMMENDATIONS
Not just scores - provide:
- Critical fixes needed before use
- Optional improvements
- Clear verdict: Ready/Needs fixes/Reject

## OUTPUT FORMAT
Respond with valid JSON ONLY matching this exact schema:

```json
{
  "classification": {
    "image_type": "studio_product_shot" | "lifestyle_model" | "cluttered_scene" | "editorial_artistic" | "amateur_ugc",
    "quality_tier": "professional" | "semi_professional" | "amateur",
    "intended_use": "e-commerce" | "social_media" | "marketing" | "internal" | "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "Explain WHY you classified it this way based on the features"
  },
  
  "type_specific_evaluation": {
    "criteria_applied": "Which evaluation criteria you used based on classification",
    "passes_type_criteria": true/false,
    "details": "Specific assessment based on image type"
  },
  
  "suitability_assessment": {
    "ecommerce_product_page": {
      "suitable": true/false,
      "score": 0.0-1.0,
      "reasoning": "Why suitable/unsuitable for product listings"
    },
    "social_media_marketing": {
      "suitable": true/false,
      "score": 0.0-1.0,
      "reasoning": "Why suitable/unsuitable for social content"
    },
    "professional_website": {
      "suitable": true/false,
      "score": 0.0-1.0,
      "reasoning": "Why suitable/unsuitable for professional use"
    }
  },
  
  "trust_evaluation": {
    "trust_score": 0.0-1.0,
    "trustworthiness": "high" | "medium" | "low",
    "trust_factors": ["List positive trust indicators"],
    "distrust_factors": ["List negative trust indicators"],
    "reasoning": "Why this image does/doesn't appear trustworthy"
  },
  
  "risks_detected": [
    {
      "risk": "Description of risk",
      "severity": "high" | "medium" | "low",
      "mitigation": "How to address this risk"
    }
  ],
  
  "quality_issues": [
    {
      "issue": "Description of quality problem",
      "severity": "critical" | "major" | "minor",
      "impact": "How this affects usability",
      "fixable": true/false,
      "fix_method": "How to fix if fixable"
    }
  ],
  
  "recommendations": {
    "critical_actions": ["Must-do fixes before use"],
    "improvements": ["Nice-to-have enhancements"],
    "verdict": "ready_to_use" | "suitable_after_fixes" | "requires_recapture" | "rejected",
    "verdict_reasoning": "Clear explanation of final decision"
  },
  
  "scores": {
    "overall_quality": 0.0-10.0,
    "technical_quality": 0.0-10.0,
    "composition": 0.0-10.0,
    "commercial_viability": 0.0-10.0
  },
  
  "confidence": 0.0-1.0
}
```

CRITICAL: Valid JSON only. No markdown, no explanations outside JSON."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """Analyze this image feature report and provide your comprehensive quality assessment.

## EXTRACTED FEATURES FROM COMPUTER VISION MODELS

### Tier 1: Object Detection (YOLO)
- Objects detected: {objects_detected}
- Object count: {object_count}
- People present: {has_people}
- Primary subject area: {primary_object_area_percent}% of frame

### Tier 2: Semantic Analysis (CLIP)
- Scene type: {clip_scene_type}
- Photography style: {clip_style}
- Background complexity: {background_complexity}

### Tier 3: Technical Quality (OpenCV)
- Sharpness: {sharpness_score}/100 ({sharpness_category})
- Exposure: {exposure_category}
- Text detected: {text_detected}

### Context Signals
- Fashion/Model context: {is_fashion_context}
- Product text (not watermark): {is_product_text}

## YOUR TASK
1. FIRST classify what type of image this is
2. THEN apply type-appropriate evaluation criteria
3. Assess suitability for different use cases
4. Evaluate trustworthiness for professional/commercial use
5. Identify risks and quality issues
6. Provide actionable recommendations

Respond with the complete JSON assessment."""


# =============================================================================
# ERROR RECOVERY PROMPT
# =============================================================================

ERROR_RECOVERY_PROMPT = """Your previous response was not valid JSON. Error: {error_message}

Please respond ONLY with valid JSON matching the required schema. No explanations outside the JSON.

Original features to analyze:
{feature_json}"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_user_prompt(feature_json: str) -> str:
    """Build the user prompt with feature data."""
    import json
    try:
        features = json.loads(feature_json)
    except:
        features = {}
    
    return USER_PROMPT_TEMPLATE.format(
        objects_detected=features.get('objects_detected', []),
        object_count=features.get('object_count', 0),
        has_people=features.get('has_people', False),
        primary_object_area_percent=features.get('primary_object_area_percent', 0),
        clip_scene_type=features.get('clip_scene_type', 'unknown'),
        clip_style=features.get('clip_style', 'unknown'),
        background_complexity=features.get('background_complexity', 'unknown'),
        sharpness_score=features.get('sharpness_score', 0),
        sharpness_category=features.get('sharpness_category', 'unknown'),
        exposure_category=features.get('exposure_category', 'unknown'),
        text_detected=features.get('text_detected', False),
        is_fashion_context=features.get('is_fashion_context', False),
        is_product_text=features.get('is_product_text', False)
    )


def build_error_recovery_prompt(error_message: str, feature_json: str) -> str:
    """Build recovery prompt when LLM returns invalid JSON."""
    return ERROR_RECOVERY_PROMPT.format(
        error_message=error_message,
        feature_json=feature_json
    )
