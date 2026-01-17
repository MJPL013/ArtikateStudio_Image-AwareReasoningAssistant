"""
LLM Judge Module
Implements the "Blind Judge" that makes quality decisions based on extracted features.

Key Patterns:
- RETRY PATTERN: Uses tenacity for automatic retries on failures
- RECURSIVE ERROR RECOVERY: Injects error context for malformed JSON responses
- Pydantic validation for strict schema enforcement
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

import sys
sys.path.append(str(__file__).rsplit('src', 1)[0])
from src.config import APP_CONFIG, API_CONFIG
from src.llm.prompts import (
    BLIND_JUDGE_SYSTEM_PROMPT,
    build_user_prompt,
    build_error_recovery_prompt
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC SCHEMA FOR VERDICT VALIDATION
# =============================================================================

class VerdictSchema(BaseModel):
    """Strict schema for LLM verdict response."""
    
    verdict: str = Field(..., description="APPROVED or REJECTED")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    primary_reason: str = Field(..., description="Main reason for decision")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    bonuses: List[str] = Field(default_factory=list, description="Positive indicators")
    quality_score: int = Field(..., ge=0, le=100, description="Overall quality score")
    recommendation: str = Field(default="", description="Improvement suggestion")
    
    @validator('verdict')
    def validate_verdict(cls, v):
        if v.upper() not in ['APPROVED', 'REJECTED']:
            raise ValueError('Verdict must be APPROVED or REJECTED')
        return v.upper()
    
    class Config:
        extra = 'ignore'  # Ignore extra fields from LLM


@dataclass
class VerdictResult:
    """Final verdict result with metadata."""
    verdict: str
    confidence: float
    primary_reason: str
    warnings: List[str]
    bonuses: List[str]
    quality_score: int
    recommendation: str
    
    # Metadata
    retry_count: int = 0
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# LLM INTERFACE (Mock/Placeholder)
# =============================================================================

class LLMInterface:
    """
    Interface for LLM calls.
    This is a mock implementation - replace with actual API calls.
    """
    
    def __init__(self, provider: str = "mock"):
        self.provider = provider
        self._call_count = 0
    
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make an LLM call. Override this for real implementations.
        
        Args:
            system_prompt: System context
            user_prompt: User message with feature data
            
        Returns:
            Raw string response from LLM
        """
        self._call_count += 1
        
        if self.provider == "mock":
            return self._mock_response(user_prompt)
        elif self.provider == "openai":
            return self._openai_call(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._gemini_call(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def _mock_response(self, user_prompt: str) -> str:
        """Generate a mock LLM response based on feature data."""
        # Extract features from prompt for mock logic
        try:
            # Find the JSON in the prompt
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', user_prompt, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group(1))
            else:
                features = {}
        except:
            features = {}
        
        # Context-aware rulebook logic
        verdict = "APPROVED"
        warnings = []
        bonuses = []
        primary_reason = "Image meets quality standards"
        quality_score = 75
        context_detected = "product_only"
        
        # Extract context signals
        is_fashion = features.get("is_fashion_context", False)
        is_product_text = features.get("is_product_text", False)
        has_people = features.get("has_people", False)
        
        if is_fashion:
            context_detected = "fashion_model"
        elif has_people:
            context_detected = "lifestyle"
        
        # Check rejection criteria with context awareness
        if features.get("sharpness_category") == "Blurry":
            verdict = "REJECTED"
            primary_reason = "Subject is not sharp enough"
            quality_score = 25
        elif has_people and not is_fashion:
            # Only reject people if NOT a fashion context
            verdict = "REJECTED"
            primary_reason = "People detected in non-fashion product image"
            quality_score = 30
        elif features.get("primary_object_area_percent", 100) < 10:
            verdict = "REJECTED"
            primary_reason = "Product occupies too little of the frame"
            quality_score = 20
        elif features.get("text_detected") and not is_product_text:
            # Only reject text if it's NOT product text (watermark)
            verdict = "REJECTED"
            primary_reason = "Watermark or overlay text detected"
            quality_score = 35
        else:
            # Check for bonuses
            if is_fashion and features.get("clip_style") == "professional":
                bonuses.append("👗 Professional model/apparel photography")
                quality_score = 92
            
            if features.get("clip_scene_type") == "studio_product":
                if features.get("clip_style") == "professional":
                    bonuses.append("Professional studio photography")
                    quality_score = max(quality_score, 90)
            
            if features.get("background_complexity") == "minimal":
                bonuses.append("Clean, distraction-free background")
                quality_score = min(quality_score + 5, 100)
            
            if is_product_text:
                bonuses.append("Product branding/design text detected (allowed)")
            
            # Check for warnings
            if features.get("sharpness_category") == "Soft":
                warnings.append("Image could be sharper")
                quality_score = max(quality_score - 10, 0)
            
            if features.get("exposure_category") not in ["Well-Exposed", None]:
                warnings.append(f"Exposure issue: {features.get('exposure_category', 'unknown')}")
                quality_score = max(quality_score - 5, 0)
            
            if features.get("background_complexity") == "complex":
                warnings.append("Background may distract from product")
            
            if features.get("clip_style") == "amateur":
                warnings.append("Image appears non-professional")
                quality_score = max(quality_score - 15, 0)
        
        # Build recommendation
        if verdict == "APPROVED" and warnings:
            recommendation = "Consider addressing the warnings for higher quality"
        elif verdict == "REJECTED":
            recommendation = "Please retake the image addressing the rejection reason"
        elif is_fashion:
            recommendation = "Excellent fashion/model photography!"
        else:
            recommendation = "Great image quality!"
        
        response = {
            "verdict": verdict,
            "confidence": 0.85 if verdict == "APPROVED" else 0.92,
            "primary_reason": primary_reason,
            "context_detected": context_detected,
            "warnings": warnings,
            "bonuses": bonuses,
            "quality_score": quality_score,
            "recommendation": recommendation
        }
        
        return json.dumps(response)
    
    def _openai_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make actual OpenAI API call."""
        # Placeholder - implement with actual API
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("OpenAI not installed, falling back to mock")
            return self._mock_response(user_prompt)
    
    def _gemini_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make actual Gemini API call."""
        try:
            import google.generativeai as genai
            
            # Configure with API key from environment or config
            api_key = API_CONFIG.gemini_api_key
            if not api_key:
                logger.warning("GEMINI_API_KEY not set, falling back to mock")
                return self._mock_response(user_prompt)
            
            genai.configure(api_key=api_key)
            
            # Use gemini-1.5-flash for faster responses
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                system_instruction=system_prompt
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent decisions
                    max_output_tokens=1000
                )
            )
            return response.text
            
        except ImportError:
            logger.warning("google-generativeai not installed, falling back to mock")
            return self._mock_response(user_prompt)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


# =============================================================================
# VERDICT PARSER WITH ERROR RECOVERY
# =============================================================================

class JSONParseError(Exception):
    """Raised when LLM returns invalid JSON."""
    pass


def parse_verdict_response(response: str) -> VerdictSchema:
    """
    Parse and validate LLM response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Validated VerdictSchema
        
    Raises:
        JSONParseError: If response is not valid JSON or doesn't match schema
    """
    # Try to extract JSON from response (handle markdown code blocks)
    import re
    
    # Remove markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response.strip()
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Invalid JSON: {str(e)}")
    
    try:
        return VerdictSchema(**data)
    except Exception as e:
        raise JSONParseError(f"Schema validation failed: {str(e)}")


# =============================================================================
# MAIN JUDGE FUNCTION WITH RETRY LOGIC
# =============================================================================

_llm_interface: Optional[LLMInterface] = None


def get_llm_interface(provider: str = "mock") -> LLMInterface:
    """Get or create LLM interface."""
    global _llm_interface
    if _llm_interface is None or _llm_interface.provider != provider:
        _llm_interface = LLMInterface(provider)
    return _llm_interface


@retry(
    stop=stop_after_attempt(APP_CONFIG.llm_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(JSONParseError),
    reraise=True
)
def _call_llm_with_retry(
    feature_json: str,
    llm: LLMInterface,
    previous_error: Optional[str] = None
) -> str:
    """
    Call LLM with automatic retry on failures.
    
    Implements the RETRY PATTERN using tenacity.
    """
    if previous_error:
        # Use error recovery prompt with context
        user_prompt = build_error_recovery_prompt(previous_error, feature_json)
    else:
        user_prompt = build_user_prompt(feature_json)
    
    response = llm.call(BLIND_JUDGE_SYSTEM_PROMPT, user_prompt)
    
    # Validate response (will raise JSONParseError if invalid)
    parse_verdict_response(response)
    
    return response


def get_verdict(
    features: Dict[str, Any],
    llm_provider: str = "mock"
) -> VerdictResult:
    """
    Main entry point: Get quality verdict for extracted features.
    
    This function:
    1. Converts features to JSON
    2. Calls LLM with retry logic
    3. Parses and validates response
    4. Returns structured VerdictResult
    
    Args:
        features: Dictionary of extracted image features
        llm_provider: LLM provider to use ("mock", "openai", "gemini")
        
    Returns:
        VerdictResult with verdict and metadata
    """
    llm = get_llm_interface(llm_provider)
    feature_json = json.dumps(features, indent=2)
    retry_count = 0
    last_error = None
    
    # Try to get valid response with recursive error recovery
    for attempt in range(APP_CONFIG.llm_max_retries):
        try:
            response = _call_llm_with_retry(
                feature_json,
                llm,
                previous_error=last_error
            )
            
            # Parse successful response
            verdict_schema = parse_verdict_response(response)
            
            return VerdictResult(
                verdict=verdict_schema.verdict,
                confidence=verdict_schema.confidence,
                primary_reason=verdict_schema.primary_reason,
                warnings=verdict_schema.warnings,
                bonuses=verdict_schema.bonuses,
                quality_score=verdict_schema.quality_score,
                recommendation=verdict_schema.recommendation,
                retry_count=retry_count,
                raw_response=response
            )
            
        except JSONParseError as e:
            retry_count += 1
            last_error = str(e)
            logger.warning(f"LLM returned invalid JSON (attempt {attempt + 1}): {e}")
            
            if attempt == APP_CONFIG.llm_max_retries - 1:
                # Final attempt failed, return error verdict
                return VerdictResult(
                    verdict="ERROR",
                    confidence=0.0,
                    primary_reason=f"LLM failed to return valid response after {retry_count} attempts",
                    warnings=[f"Last error: {last_error}"],
                    bonuses=[],
                    quality_score=0,
                    recommendation="Manual review required",
                    retry_count=retry_count,
                    raw_response=None
                )
    
    # Should not reach here, but just in case
    return VerdictResult(
        verdict="ERROR",
        confidence=0.0,
        primary_reason="Unknown error in verdict processing",
        warnings=[],
        bonuses=[],
        quality_score=0,
        recommendation="Manual review required",
        retry_count=retry_count
    )


def get_verdict_from_result(
    extraction_result: 'FeatureExtractionResult',
    llm_provider: str = "mock"
) -> VerdictResult:
    """
    Convenience function to get verdict directly from FeatureExtractionResult.
    
    Args:
        extraction_result: Result from consolidate_features()
        llm_provider: LLM provider to use
        
    Returns:
        VerdictResult
    """
    # Handle pre-rejected images
    if extraction_result.status == "REJECTED":
        return VerdictResult(
            verdict="REJECTED",
            confidence=1.0,
            primary_reason=extraction_result.rejection_reason or "Failed quality checks",
            warnings=[],
            bonuses=[],
            quality_score=0,
            recommendation="Image did not pass initial quality filters",
            retry_count=0
        )
    
    if extraction_result.status == "ERROR":
        return VerdictResult(
            verdict="ERROR",
            confidence=0.0,
            primary_reason=extraction_result.rejection_reason or "Processing error",
            warnings=[],
            bonuses=[],
            quality_score=0,
            recommendation="Please try a different image",
            retry_count=0
        )
    
    # Get LLM verdict for successful extractions
    features = extraction_result.to_llm_context()
    return get_verdict(features, llm_provider)
