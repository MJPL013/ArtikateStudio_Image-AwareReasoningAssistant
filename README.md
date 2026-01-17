# Blind Image Reasoning System

<div align="center">

🔍 **AI-Powered E-Commerce Image Quality Control**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)

</div>

---

## 🎯 Overview

The **Blind Image Reasoning System** is a production-ready quality control system for e-commerce images. It uses a unique "Blind Reasoning" architecture where the LLM Judge **cannot see image pixels directly** - instead, it receives a structured JSON report of extracted features and applies a rulebook to render quality verdicts.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BLIND IMAGE REASONING SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────────────────────────────────────────────┐   │
│  │          │    │              SMART FEATURE EXTRACTION                │   │
│  │  INPUT   │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐                │   │
│  │  IMAGE   │───▶│  │ OpenCV  │ │  YOLO   │ │  CLIP   │                │   │
│  │          │    │  │(Tier 3) │ │(Tier 1) │ │(Tier 2) │                │   │
│  └──────────┘    │  └────┬────┘ └────┬────┘ └────┬────┘                │   │
│                  │       │           │           │                      │   │
│                  │       └───────────┴───────────┘                      │   │
│                  │                   │                                  │   │
│                  │           ┌───────▼───────┐                          │   │
│                  │           │ JSON FEATURES │                          │   │
│                  │           └───────┬───────┘                          │   │
│                  └───────────────────┼──────────────────────────────────┘   │
│                                      │                                      │
│                              ┌───────▼───────┐                              │
│                              │               │                              │
│                              │   LLM JUDGE   │ ◀── Cannot see pixels!       │
│                              │  (The Blind)  │     Only sees JSON           │
│                              │               │                              │
│                              └───────┬───────┘                              │
│                                      │                                      │
│                              ┌───────▼───────┐                              │
│                              │    VERDICT    │                              │
│                              │ ✅ APPROVED   │                              │
│                              │ ❌ REJECTED   │                              │
│                              └───────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🚀 Fail-Fast Architecture
- Blurry images are rejected **immediately** using cheap OpenCV math
- Saves compute by not running expensive models on obviously bad images

### 🧠 Smart Feature Extraction
| Tier | Technology | Features |
|------|------------|----------|
| **Tier 1** | YOLOv8 | Object detection, people detection, subject area |
| **Tier 2** | CLIP | Scene type, photography style, background complexity |
| **Tier 3** | OpenCV | Sharpness, exposure, text/watermark detection |

### 🎯 Region-Aware Blur Detection
- Distinguishes **intentional bokeh** (sharp subject, blurry background) from **actual blur**
- Analyzes the detected subject region, not the whole image

### ⚖️ Intelligent LLM Judge
- Applies a comprehensive **Rulebook** for consistent decisions
- Automatic retry with error recovery for malformed responses
- Mock mode for testing without API costs

## 📁 Project Structure

```
├── .idx/
│   └── dev.nix              # Nix config for Google Project IDX
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration (thresholds, model settings)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── cv_ops.py        # OpenCV operations (blur, exposure, text)
│   │   ├── vision_models.py # SINGLETON YOLO & CLIP wrappers
│   │   └── pipeline.py      # FACADE orchestrator (consolidate_features)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py       # System prompts for Blind Judge
│   │   └── judge.py         # RETRY logic & LLM caller
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Image utilities, file handling
├── app.py                   # Streamlit UI
└── README.md
```

## 🛠️ Installation

### Option 1: Google Project IDX (Recommended)

1. Open project in IDX - the Nix environment will auto-configure
2. Wait for the workspace to initialize
3. The Streamlit preview will start automatically

### Option 2: Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Artikate-Studio

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🚀 Usage

### Single Image Analysis

1. Upload an image using the file uploader
2. Click **"Analyze Image"**
3. View the verdict, warnings, and quality score
4. Expand **"View Extracted Features"** to see the JSON the LLM received

### Batch Processing

1. Enter a folder path containing product images
2. Optionally check "Include subfolders"
3. Click **"Process Folder"**
4. View results in the interactive table
5. Export as CSV or JSON

## 🔧 Configuration

### Quality Thresholds

Adjust in `src/config.py` or via the Streamlit sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fail_fast_sharpness` | 30.0 | Minimum sharpness to proceed |
| `sharpness_sharp` | 100.0 | Threshold for "Sharp" category |
| `sharpness_soft` | 50.0 | Threshold for "Soft" category |
| `min_subject_area_percent` | 10.0 | Minimum product area |

### LLM Providers

Set in the sidebar or code:

- `mock` - Local rule-based mock (no API costs)
- `openai` - OpenAI GPT-4 (requires API key)
- `gemini` - Google Gemini (requires API key)

## 📊 JSON Feature Schema

The system extracts exactly this schema:

```json
{
  "objects_detected": ["shoe", "shoebox"],
  "object_count": 2,
  "has_people": false,
  "primary_object_area_percent": 45.2,
  
  "clip_scene_type": "studio_product",
  "clip_style": "professional",
  "background_complexity": "minimal",
  
  "sharpness_score": 78.5,
  "sharpness_category": "Sharp",
  "exposure_category": "Well-Exposed",
  "text_detected": false
}
```

## 🏗️ Design Patterns

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Singleton** | `vision_models.py` | Load YOLO/CLIP once, reuse for all images |
| **Facade** | `pipeline.py` | Simple `consolidate_features()` hides complexity |
| **Retry** | `judge.py` | Automatic retry with error context for LLM |

## 📝 The Rulebook

The LLM Judge applies these rules in order:

### Automatic Rejection
- ❌ Blurry subject (`sharpness_category == "Blurry"`)
- ❌ People in product image (`has_people == true`)
- ❌ Subject too small (`primary_object_area_percent < 10%`)
- ❌ Watermarks detected (`text_detected == true`)

### Quality Warnings
- ⚠️ Soft focus
- ⚠️ Exposure issues
- ⚠️ Complex background
- ⚠️ Amateur style

### Quality Bonuses
- 🌟 Professional studio photography
- 🌟 Clean, minimal background

## 🔒 License

MIT License - See LICENSE file for details.

---

<div align="center">

**Built with ❤️ using Streamlit • YOLOv8 • CLIP • OpenCV**

</div>
