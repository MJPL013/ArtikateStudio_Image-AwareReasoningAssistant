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
# PYDANTIC SCHEMA FOR COMPREHENSIVE VERDICT VALIDATION
# =============================================================================

class ClassificationSchema(BaseModel):
    """Image classification result."""
    image_type: str = Field(default="unknown")
    quality_tier: str = Field(default="unknown")
    intended_use: str = Field(default="unknown")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")

class SuitabilityScore(BaseModel):
    """Suitability assessment for a use case."""
    suitable: bool = Field(default=False)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")

class TrustEvaluation(BaseModel):
    """Trust and professionalism assessment."""
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    trustworthiness: str = Field(default="medium")
    trust_factors: List[str] = Field(default_factory=list)
    distrust_factors: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

class RiskItem(BaseModel):
    """Individual risk detected."""
    risk: str = Field(default="")
    severity: str = Field(default="low")
    mitigation: str = Field(default="")

class QualityIssue(BaseModel):
    """Individual quality issue."""
    issue: str = Field(default="")
    severity: str = Field(default="minor")
    impact: str = Field(default="")
    fixable: bool = Field(default=True)
    fix_method: str = Field(default="")

class Recommendations(BaseModel):
    """Actionable recommendations."""
    critical_actions: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    verdict: str = Field(default="ready_to_use")
    verdict_reasoning: str = Field(default="")

class Scores(BaseModel):
    """Quality scores."""
    overall_quality: float = Field(default=5.0, ge=0.0, le=10.0)
    technical_quality: float = Field(default=5.0, ge=0.0, le=10.0)
    composition: float = Field(default=5.0, ge=0.0, le=10.0)
    commercial_viability: float = Field(default=5.0, ge=0.0, le=10.0)

class ComprehensiveVerdictSchema(BaseModel):
    """Comprehensive schema for LLM verdict response."""
    
    classification: ClassificationSchema = Field(default_factory=ClassificationSchema)
    type_specific_evaluation: Dict[str, Any] = Field(default_factory=dict)
    suitability_assessment: Dict[str, SuitabilityScore] = Field(default_factory=dict)
    trust_evaluation: TrustEvaluation = Field(default_factory=TrustEvaluation)
    risks_detected: List[RiskItem] = Field(default_factory=list)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    recommendations: Recommendations = Field(default_factory=Recommendations)
    scores: Scores = Field(default_factory=Scores)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    class Config:
        extra = 'ignore'


@dataclass
class VerdictResult:
    """Final verdict result with comprehensive metadata."""
    # Core verdict
    verdict: str
    confidence: float
    primary_reason: str
    quality_score: int
    
    # Classification
    image_type: str = "unknown"
    quality_tier: str = "unknown"
    
    # Suitability
    ecommerce_suitable: bool = False
    social_media_suitable: bool = False
    professional_suitable: bool = False
    
    # Trust
    trust_score: float = 0.5
    trust_factors: List[str] = None
    distrust_factors: List[str] = None
    
    # Issues and recommendations
    warnings: List[str] = None
    bonuses: List[str] = None
    risks: List[Dict] = None
    quality_issues: List[Dict] = None
    critical_actions: List[str] = None
    improvements: List[str] = None
    recommendation: str = ""
    
    # Scores
    technical_score: float = 5.0
    composition_score: float = 5.0
    commercial_score: float = 5.0
    
    # Metadata
    retry_count: int = 0
    raw_response: Optional[str] = None
    
    def __post_init__(self):
        # Initialize mutable defaults
        if self.warnings is None:
            self.warnings = []
        if self.bonuses is None:
            self.bonuses = []
        if self.trust_factors is None:
            self.trust_factors = []
        if self.distrust_factors is None:
            self.distrust_factors = []
        if self.risks is None:
            self.risks = []
        if self.quality_issues is None:
            self.quality_issues = []
        if self.critical_actions is None:
            self.critical_actions = []
        if self.improvements is None:
            self.improvements = []
    
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
        """Generate a comprehensive mock LLM response based on feature data."""
        # Extract features from prompt
        features = {}
        try:
            # Parse features from the user prompt format
            import re
            for line in user_prompt.split('\n'):
                if 'Objects detected:' in line:
                    features['objects_detected'] = eval(line.split(':')[1].strip()) if '[' in line else []
                elif 'People present:' in line:
                    features['has_people'] = 'True' in line
                elif 'Primary subject area:' in line:
                    match = re.search(r'([\d.]+)%', line)
                    features['primary_object_area_percent'] = float(match.group(1)) if match else 0
                elif 'Scene type:' in line:
                    features['clip_scene_type'] = line.split(':')[1].strip()
                elif 'Photography style:' in line:
                    features['clip_style'] = line.split(':')[1].strip()
                elif 'Background complexity:' in line:
                    features['background_complexity'] = line.split(':')[1].strip()
                elif 'Sharpness:' in line:
                    match = re.search(r'([\d.]+)/100', line)
                    features['sharpness_score'] = float(match.group(1)) if match else 50
                    if 'Sharp' in line:
                        features['sharpness_category'] = 'Sharp'
                    elif 'Soft' in line:
                        features['sharpness_category'] = 'Soft'
                    elif 'Blurry' in line:
                        features['sharpness_category'] = 'Blurry'
                elif 'Exposure:' in line:
                    features['exposure_category'] = line.split(':')[1].strip()
                elif 'Text detected:' in line:
                    features['text_detected'] = 'True' in line
                elif 'Fashion/Model context:' in line:
                    features['is_fashion_context'] = 'True' in line
                elif 'Product text' in line:
                    features['is_product_text'] = 'True' in line
        except:
            pass
        
        # ===== STEP 1: CLASSIFICATION =====
        is_fashion = features.get('is_fashion_context', False)
        has_people = features.get('has_people', False)
        clip_scene = features.get('clip_scene_type', 'unknown')
        clip_style = features.get('clip_style', 'unknown')
        
        # Determine image type
        if is_fashion or (has_people and clip_scene in ['studio_product', 'lifestyle']):
            image_type = "lifestyle_model"
        elif clip_scene == 'studio_product':
            image_type = "studio_product_shot"
        elif features.get('background_complexity') == 'complex':
            image_type = "cluttered_scene"
        elif clip_style == 'amateur':
            image_type = "amateur_ugc"
        else:
            image_type = "studio_product_shot"
        
        # Determine quality tier
        sharpness = features.get('sharpness_score', 50)
        if clip_style == 'professional' and sharpness > 70:
            quality_tier = "professional"
        elif sharpness > 50:
            quality_tier = "semi_professional"
        else:
            quality_tier = "amateur"
        
        # ===== STEP 2: EVALUATE BASED ON TYPE =====
        warnings = []
        bonuses = []
        risks = []
        quality_issues = []
        critical_actions = []
        improvements = []
        trust_factors = []
        distrust_factors = []
        
        overall_score = 7.0
        technical_score = 7.0
        composition_score = 7.0
        commercial_score = 7.0
        
        # Technical quality checks
        sharpness_cat = features.get('sharpness_category', 'Soft')
        if sharpness_cat == 'Blurry':
            quality_issues.append({
                "issue": "Severe blur detected",
                "severity": "critical",
                "impact": "Unusable for professional purposes",
                "fixable": False,
                "fix_method": "Recapture with proper focus"
            })
            critical_actions.append("Recapture image - blur too severe")
            distrust_factors.append("Poor technical quality")
            technical_score = 2.0
            overall_score = 3.0
        elif sharpness_cat == 'Soft':
            quality_issues.append({
                "issue": "Soft focus detected",
                "severity": "major",
                "impact": "Reduces perceived professionalism",
                "fixable": True,
                "fix_method": "Apply sharpening in post-processing"
            })
            improvements.append("Apply light sharpening")
            technical_score = 5.5
        else:
            trust_factors.append("Sharp, clear focus")
            technical_score = 8.5
        
        # Exposure check
        exposure = features.get('exposure_category', 'Well-Exposed')
        if exposure == 'Under-Exposed':
            quality_issues.append({
                "issue": "Image is too dark",
                "severity": "major",
                "impact": "Product details may not be visible",
                "fixable": True,
                "fix_method": "Increase brightness/exposure"
            })
            improvements.append("Brighten the image")
            technical_score -= 1.5
        elif exposure == 'Over-Exposed':
            quality_issues.append({
                "issue": "Image is too bright",
                "severity": "major",
                "impact": "Loss of detail in highlights",
                "fixable": True,
                "fix_method": "Reduce exposure/brightness"
            })
            improvements.append("Reduce brightness")
            technical_score -= 1.5
        else:
            trust_factors.append("Well-balanced exposure")
        
        # People/fashion context
        if has_people and not is_fashion:
            risks.append({
                "risk": "Unintended people in product shot",
                "severity": "high",
                "mitigation": "Crop or retake without people"
            })
            critical_actions.append("Remove people from product shot")
            distrust_factors.append("Unprofessional composition")
            commercial_score = 3.0
        elif is_fashion:
            bonuses.append("👗 Professional model photography - people appropriate")
            trust_factors.append("Fashion/model context correctly identified")
            commercial_score = 8.5
        
        # Background assessment
        bg_complexity = features.get('background_complexity', 'simple')
        if image_type == "studio_product_shot":
            if bg_complexity == 'minimal':
                bonuses.append("Clean, professional background")
                trust_factors.append("Distraction-free background")
                composition_score = 9.0
            elif bg_complexity == 'complex':
                quality_issues.append({
                    "issue": "Cluttered background for product shot",
                    "severity": "major",
                    "impact": "Distracts from product",
                    "fixable": True,
                    "fix_method": "Remove background or reshoot"
                })
                distrust_factors.append("Unprofessional background")
                composition_score = 4.0
        
        # Subject size
        subject_area = features.get('primary_object_area_percent', 30)
        if subject_area < 10:
            quality_issues.append({
                "issue": "Product too small in frame",
                "severity": "critical",
                "impact": "Product not visible enough",
                "fixable": True,
                "fix_method": "Crop tighter or retake closer"
            })
            critical_actions.append("Increase product prominence in frame")
            composition_score = 3.0
        elif subject_area > 70 and image_type == "studio_product_shot":
            bonuses.append("Product prominently featured")
        
        # Text/watermark
        if features.get('text_detected') and not features.get('is_product_text'):
            risks.append({
                "risk": "Possible watermark or overlay text",
                "severity": "medium",
                "mitigation": "Verify text is intentional branding"
            })
            distrust_factors.append("Unexplained text overlay")
        elif features.get('is_product_text'):
            bonuses.append("Product branding detected (appropriate)")
        
        # Calculate overall score
        overall_score = (technical_score * 0.35 + composition_score * 0.30 + commercial_score * 0.35)
        
        # ===== STEP 3: DETERMINE VERDICT =====
        if any(qi.get('severity') == 'critical' for qi in quality_issues):
            verdict = "requires_recapture" if sharpness_cat == 'Blurry' else "suitable_after_fixes"
            final_verdict = "REJECTED" if verdict == "requires_recapture" else "APPROVED"
        elif critical_actions:
            verdict = "suitable_after_fixes"
            final_verdict = "APPROVED"
        elif warnings or quality_issues:
            verdict = "suitable_after_fixes"
            final_verdict = "APPROVED"
        else:
            verdict = "ready_to_use"
            final_verdict = "APPROVED"
        
        # Suitability
        ecom_suitable = overall_score >= 6.0 and not any(qi.get('severity') == 'critical' for qi in quality_issues)
        social_suitable = overall_score >= 5.0
        prof_suitable = overall_score >= 7.0 and quality_tier in ['professional', 'semi_professional']
        
        # Trust score
        trust_score = min(1.0, max(0.0, (len(trust_factors) - len(distrust_factors) * 0.5 + 5) / 10))
        
        # Build comprehensive response
        response = {
            "classification": {
                "image_type": image_type,
                "quality_tier": quality_tier,
                "intended_use": "e-commerce" if image_type == "studio_product_shot" else "marketing",
                "confidence": 0.85,
                "reasoning": f"Classified as {image_type} based on scene type ({clip_scene}), style ({clip_style}), and composition"
            },
            "type_specific_evaluation": {
                "criteria_applied": f"Evaluated using {image_type} criteria",
                "passes_type_criteria": overall_score >= 6.0,
                "details": f"Applied type-specific standards for {image_type}"
            },
            "suitability_assessment": {
                "ecommerce_product_page": {
                    "suitable": ecom_suitable,
                    "score": min(1.0, overall_score / 10),
                    "reasoning": "Suitable for product listings" if ecom_suitable else "Quality issues prevent e-commerce use"
                },
                "social_media_marketing": {
                    "suitable": social_suitable,
                    "score": min(1.0, (overall_score + 1) / 10),
                    "reasoning": "Acceptable for social content" if social_suitable else "Below social media standards"
                },
                "professional_website": {
                    "suitable": prof_suitable,
                    "score": min(1.0, overall_score / 10),
                    "reasoning": "Professional quality" if prof_suitable else "Not professional enough"
                }
            },
            "trust_evaluation": {
                "trust_score": trust_score,
                "trustworthiness": "high" if trust_score > 0.7 else ("medium" if trust_score > 0.4 else "low"),
                "trust_factors": trust_factors,
                "distrust_factors": distrust_factors,
                "reasoning": f"Trust score {trust_score:.2f} based on {len(trust_factors)} positive and {len(distrust_factors)} negative factors"
            },
            "risks_detected": risks,
            "quality_issues": quality_issues,
            "recommendations": {
                "critical_actions": critical_actions,
                "improvements": improvements,
                "verdict": verdict,
                "verdict_reasoning": f"Overall score {overall_score:.1f}/10 - {verdict.replace('_', ' ')}"
            },
            "scores": {
                "overall_quality": round(overall_score, 1),
                "technical_quality": round(technical_score, 1),
                "composition": round(composition_score, 1),
                "commercial_viability": round(commercial_score, 1)
            },
            "confidence": 0.88
        }
        
        return json.dumps(response, indent=2)
    
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


def parse_verdict_response(response: str) -> ComprehensiveVerdictSchema:
    """
    Parse and validate LLM response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Validated ComprehensiveVerdictSchema
        
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
        return ComprehensiveVerdictSchema(**data)
    except Exception as e:
        raise JSONParseError(f"Schema validation failed: {str(e)}")


def create_verdict_from_comprehensive(data: dict, raw_response: str, retry_count: int = 0) -> VerdictResult:
    """Convert comprehensive LLM response to VerdictResult."""
    
    # Extract classification
    classification = data.get('classification', {})
    image_type = classification.get('image_type', 'unknown')
    quality_tier = classification.get('quality_tier', 'unknown')
    
    # Extract suitability
    suitability = data.get('suitability_assessment', {})
    ecom = suitability.get('ecommerce_product_page', {})
    social = suitability.get('social_media_marketing', {})
    prof = suitability.get('professional_website', {})
    
    # Extract trust
    trust = data.get('trust_evaluation', {})
    
    # Extract scores
    scores = data.get('scores', {})
    overall_quality = scores.get('overall_quality', 5.0)
    
    # Extract recommendations
    recommendations = data.get('recommendations', {})
    verdict_str = recommendations.get('verdict', 'ready_to_use')
    
    # Map verdict to APPROVED/REJECTED
    if verdict_str in ['ready_to_use', 'suitable_after_fixes']:
        final_verdict = "APPROVED"
    else:
        final_verdict = "REJECTED"
    
    # Build primary reason
    primary_reason = recommendations.get('verdict_reasoning', classification.get('reasoning', 'Quality assessment complete'))
    
    # Extract warnings (from quality issues)
    warnings = [qi.get('issue', '') for qi in data.get('quality_issues', []) if qi.get('severity') != 'critical']
    
    # Extract bonuses (from trust factors)
    bonuses = trust.get('trust_factors', [])
    
    # Extract risks and quality issues
    risks = data.get('risks_detected', [])
    quality_issues = data.get('quality_issues', [])
    
    # Build recommendation text
    critical = recommendations.get('critical_actions', [])
    improvements = recommendations.get('improvements', [])
    if critical:
        recommendation = f"Critical: {critical[0]}"
    elif improvements:
        recommendation = f"Suggestion: {improvements[0]}"
    else:
        recommendation = "Image meets quality standards"
    
    return VerdictResult(
        verdict=final_verdict,
        confidence=data.get('confidence', 0.8),
        primary_reason=primary_reason,
        quality_score=int(overall_quality * 10),  # Convert 0-10 to 0-100
        
        # Classification
        image_type=image_type,
        quality_tier=quality_tier,
        
        # Suitability
        ecommerce_suitable=ecom.get('suitable', False),
        social_media_suitable=social.get('suitable', False),
        professional_suitable=prof.get('suitable', False),
        
        # Trust
        trust_score=trust.get('trust_score', 0.5),
        trust_factors=trust.get('trust_factors', []),
        distrust_factors=trust.get('distrust_factors', []),
        
        # Issues and recommendations
        warnings=warnings,
        bonuses=bonuses,
        risks=risks,
        quality_issues=quality_issues,
        critical_actions=critical,
        improvements=improvements,
        recommendation=recommendation,
        
        # Scores
        technical_score=scores.get('technical_quality', 5.0),
        composition_score=scores.get('composition', 5.0),
        commercial_score=scores.get('commercial_viability', 5.0),
        
        # Metadata
        retry_count=retry_count,
        raw_response=raw_response
    )


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
            parse_verdict_response(response)  # Validate schema
            
            # Extract JSON data and create comprehensive VerdictResult
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            return create_verdict_from_comprehensive(data, response, retry_count)
            
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
