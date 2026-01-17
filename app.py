"""
Blind Image Reasoning System - Streamlit Application
Professional dashboard for image quality analysis with batch processing support.

Features:
- Single image upload and analysis
- Batch folder processing with progress tracking
- Configuration sidebar with threshold controls
- Transparency view showing raw feature JSON
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from PIL import Image
import io
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.pipeline import consolidate_features, FeatureExtractionResult
from src.llm.judge import get_verdict_from_result, VerdictResult
from src.utils.helpers import get_image_files, get_image_info, is_valid_image
from src.utils.output import save_analysis_result, save_batch_results
from src.config import QUALITY_THRESHOLDS, MODEL_CONFIG, APP_CONFIG, API_CONFIG


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Blind Image Reasoning System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .verdict-approved {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .verdict-rejected {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .verdict-error {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .feature-category {
        font-weight: 600;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### Quality Thresholds")
    
    sharpness_threshold = st.slider(
        "Sharpness Threshold (Fail-Fast)",
        min_value=10.0,
        max_value=100.0,
        value=QUALITY_THRESHOLDS.fail_fast_sharpness,
        step=5.0,
        help="Images below this sharpness score are immediately rejected"
    )
    
    min_subject_area = st.slider(
        "Minimum Subject Area (%)",
        min_value=5.0,
        max_value=50.0,
        value=QUALITY_THRESHOLDS.min_subject_area_percent,
        step=5.0,
        help="Product must occupy at least this percentage of the image"
    )
    
    st.markdown("### Model Selection")
    
    # Get default index based on API_CONFIG
    llm_options = ["gemini", "mock", "openai"]
    default_idx = llm_options.index(API_CONFIG.default_llm_provider) if API_CONFIG.default_llm_provider in llm_options else 0
    
    llm_provider = st.selectbox(
        "LLM Provider",
        options=llm_options,
        index=default_idx,
        help="Select the LLM to use for quality verdicts"
    )
    
    st.markdown("### Output Settings")
    
    auto_save_json = st.checkbox(
        "📥 Auto-save JSON output",
        value=True,
        help="Automatically save detailed JSON results to outputs/ folder"
    )
    
    st.markdown("### About")
    st.markdown("""
    **Blind Image Reasoning System**
    
    This system uses a "blind" architecture where the 
    LLM judge cannot see image pixels directly. Instead:
    
    1. 🔍 **OpenCV** checks technical quality
    2. 🎯 **YOLOv8** detects objects
    3. 🧠 **CLIP** understands semantics
    4. ⚖️ **LLM** makes the final verdict
    
    The LLM receives only structured JSON features.
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Reload Models", use_container_width=True):
        from src.features.vision_models import preload_models, clear_model_cache
        with st.spinner("Reloading models..."):
            clear_model_cache()
            preload_models()
            st.session_state.models_loaded = True
        st.success("Models reloaded!")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">🔍 Blind Image Reasoning System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">E-Commerce Image Quality Control with AI-Powered Analysis</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📷 Single Image", "📁 Batch Processing"])


# =============================================================================
# TAB 1: SINGLE IMAGE ANALYSIS
# =============================================================================

with tab1:
    st.markdown("### Upload an Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload a product image to analyze its quality"
    )
    
    if uploaded_file is not None:
        # Create two columns: image preview and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image info
            info = {
                "Filename": uploaded_file.name,
                "Size": f"{image.width} × {image.height}",
                "Format": image.format or "Unknown",
                "File Size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.json(info)
        
        with col2:
            st.markdown("#### 🔬 Analysis Results")
            
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Extracting features..."):
                    # Convert uploaded file to PIL Image for processing
                    image_pil = Image.open(uploaded_file)
                    
                    # Run feature extraction
                    start_time = time.time()
                    extraction_result = consolidate_features(image_pil)
                    extraction_time = time.time() - start_time
                    
                    # Get LLM verdict
                    with st.spinner(f"Getting {llm_provider.upper()} verdict..."):
                        verdict_result = get_verdict_from_result(extraction_result, llm_provider)
                    
                    # Auto-save JSON if enabled
                    saved_path = None
                    if auto_save_json:
                        saved_path = save_analysis_result(
                            image_path=uploaded_file.name,
                            extraction_result=extraction_result,
                            verdict_result=verdict_result
                        )
                    
                    st.session_state.analysis_results = {
                        "extraction": extraction_result,
                        "verdict": verdict_result,
                        "time": extraction_time,
                        "saved_path": saved_path
                    }
                    
                    if saved_path:
                        st.success(f"📥 JSON saved: `{saved_path}`")
            
            # Display results if available
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                extraction = results["extraction"]
                verdict = results["verdict"]
                
                # Fashion Mode Badge
                if extraction.is_fashion_context:
                    st.markdown("""
                    <div style="background: linear-gradient(90deg, #EC4899, #8B5CF6); 
                                color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                                display: inline-block; margin-bottom: 1rem; font-weight: 600;">
                        👗 Fashion Mode Detected
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("People are ALLOWED in this context (model/apparel photography)")
                
                # Product Text Badge
                if extraction.is_product_text:
                    st.markdown("""
                    <div style="background: #3B82F6; color: white; padding: 0.3rem 0.8rem; 
                                border-radius: 15px; display: inline-block; margin-bottom: 0.5rem; 
                                font-size: 0.85rem;">
                        🏷️ Product Text (Not Watermark)
                    </div>
                    """, unsafe_allow_html=True)
                
                # Verdict display
                verdict_class = {
                    "APPROVED": "verdict-approved",
                    "REJECTED": "verdict-rejected",
                    "ERROR": "verdict-error"
                }.get(verdict.verdict, "verdict-error")
                
                verdict_emoji = {
                    "APPROVED": "✅",
                    "REJECTED": "❌",
                    "ERROR": "⚠️"
                }.get(verdict.verdict, "⚠️")
                
                st.markdown(f"""
                <div class="{verdict_class}">
                    <h2>{verdict_emoji} {verdict.verdict}</h2>
                    <p><strong>{verdict.primary_reason}</strong></p>
                    <p>Quality Score: <strong>{verdict.quality_score}/100</strong></p>
                    <p>Confidence: {verdict.confidence:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Warnings and Bonuses
                if verdict.warnings:
                    st.markdown("##### ⚠️ Warnings")
                    for warning in verdict.warnings:
                        st.warning(warning)
                
                if verdict.bonuses:
                    st.markdown("##### 🌟 Quality Bonuses")
                    for bonus in verdict.bonuses:
                        st.success(bonus)
                
                # Recommendation
                if verdict.recommendation:
                    st.info(f"💡 **Recommendation:** {verdict.recommendation}")
                
                # Processing time
                st.caption(f"⏱️ Processing time: {results['time']:.2f}s")
    
    # Feature transparency section
    if st.session_state.analysis_results:
        with st.expander("🔍 View Extracted Features (What the Blind Judge Saw)", expanded=False):
            extraction = st.session_state.analysis_results["extraction"]
            
            st.markdown("This is the exact JSON data sent to the LLM judge:")
            
            # Format as JSON
            features_dict = extraction.to_llm_context()
            st.json(features_dict)
            
            st.markdown("---")
            st.markdown("#### Feature Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📦 Object Detection (YOLO)**")
                st.write(f"- Objects: {extraction.objects_detected or []}")
                st.write(f"- Count: {extraction.object_count or 0}")
                people_status = "Yes 👗 (Fashion)" if extraction.is_fashion_context else ("Yes ⚠️" if extraction.has_people else "No ✓")
                st.write(f"- Has People: {people_status}")
                st.write(f"- Subject Area: {extraction.primary_object_area_percent or 0:.1f}%")
            
            with col2:
                st.markdown("**🧠 Semantic Analysis (CLIP)**")
                st.write(f"- Scene Type: {extraction.clip_scene_type or 'N/A'}")
                st.write(f"- Style: {extraction.clip_style or 'N/A'}")
                st.write(f"- Background: {extraction.background_complexity or 'N/A'}")
            
            with col3:
                st.markdown("**📊 Technical Quality (OpenCV)**")
                st.write(f"- Sharpness: {extraction.sharpness_score or 0:.1f}/100 ({extraction.sharpness_category})")
                st.write(f"- Exposure: {extraction.exposure_category or 'N/A'}")
                text_status = "Yes 🏷️ (Product)" if extraction.is_product_text else ("Yes ⚠️" if extraction.text_detected else "No ✓")
                st.write(f"- Text/Watermarks: {text_status}")


# =============================================================================
# TAB 2: BATCH PROCESSING
# =============================================================================

with tab2:
    st.markdown("### Batch Process Multiple Images")
    
    folder_path = st.text_input(
        "Enter folder path containing images",
        placeholder="C:/path/to/your/images",
        help="Enter the full path to a folder containing product images"
    )
    
    include_subfolders = st.checkbox("Include subfolders", value=False)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_button = st.button(
            "🚀 Process Folder",
            type="primary",
            use_container_width=True,
            disabled=not folder_path
        )
    
    with col2:
        if folder_path and Path(folder_path).exists():
            try:
                image_files = get_image_files(folder_path, recursive=include_subfolders)
                st.info(f"📁 Found {len(image_files)} images in the folder")
            except Exception as e:
                st.error(f"Error reading folder: {e}")
                image_files = []
        else:
            image_files = []
    
    if process_button and image_files:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_results = []
        total = len(image_files)
        
        for i, image_path in enumerate(image_files):
            status_text.text(f"Processing: {image_path.name} ({i+1}/{total})")
            
            try:
                # Extract features
                extraction = consolidate_features(image_path)
                
                # Get verdict
                verdict = get_verdict_from_result(extraction, llm_provider)
                
                batch_results.append({
                    "Filename": image_path.name,
                    "Path": str(image_path),
                    "Status": extraction.status,
                    "Verdict": verdict.verdict,
                    "Score": verdict.quality_score,
                    "Reason": verdict.primary_reason,
                    "Sharpness": extraction.sharpness_score or 0,
                    "Scene": extraction.clip_scene_type or "N/A",
                    "Style": extraction.clip_style or "N/A"
                })
                
            except Exception as e:
                batch_results.append({
                    "Filename": image_path.name,
                    "Path": str(image_path),
                    "Status": "ERROR",
                    "Verdict": "ERROR",
                    "Score": 0,
                    "Reason": str(e),
                    "Sharpness": 0,
                    "Scene": "N/A",
                    "Style": "N/A"
                })
            
            # Update progress
            progress_bar.progress((i + 1) / total)
        
        status_text.text("✅ Processing complete!")
        st.session_state.batch_results = batch_results
    
    # Display batch results
    if st.session_state.batch_results:
        st.markdown("---")
        st.markdown("### 📊 Batch Results")
        
        results_df = pd.DataFrame(st.session_state.batch_results)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(results_df)
        approved = len(results_df[results_df['Verdict'] == 'APPROVED'])
        rejected = len(results_df[results_df['Verdict'] == 'REJECTED'])
        errors = len(results_df[results_df['Verdict'] == 'ERROR'])
        
        with col1:
            st.metric("Total Images", total)
        with col2:
            st.metric("Approved ✅", approved, f"{approved/total*100:.0f}%")
        with col3:
            st.metric("Rejected ❌", rejected, f"{rejected/total*100:.0f}%")
        with col4:
            st.metric("Errors ⚠️", errors)
        
        # Filter options
        st.markdown("#### Filter Results")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            verdict_filter = st.multiselect(
                "Filter by Verdict",
                options=["APPROVED", "REJECTED", "ERROR"],
                default=["APPROVED", "REJECTED", "ERROR"]
            )
        
        with filter_col2:
            min_score = st.slider("Minimum Score", 0, 100, 0)
        
        # Apply filters
        filtered_df = results_df[
            (results_df['Verdict'].isin(verdict_filter)) &
            (results_df['Score'] >= min_score)
        ]
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%d"
                ),
                "Sharpness": st.column_config.NumberColumn(
                    "Sharpness",
                    format="%.1f"
                )
            }
        )
        
        # Export option
        st.markdown("#### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "batch_results.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = results_df.to_json(orient="records", indent=2)
            st.download_button(
                "📥 Download JSON",
                json_data,
                "batch_results.json",
                "application/json",
                use_container_width=True
            )
        
        # Rejection reasons breakdown
        rejected_df = results_df[results_df['Verdict'] == 'REJECTED']
        if len(rejected_df) > 0:
            with st.expander("📋 Rejection Reasons Breakdown", expanded=False):
                reason_counts = rejected_df['Reason'].value_counts()
                st.bar_chart(reason_counts)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
    <p>Blind Image Reasoning System v1.0</p>
    <p>Built with Streamlit • YOLOv8 • CLIP • OpenCV</p>
</div>
""", unsafe_allow_html=True)
