<<<<<<< HEAD
# SurgAgent: AI-Powered Surgical Instrument Tracking

**Team**: SurgAgent | **Hackathon**: Google Agentic AI App Hackathon - December 2025

---

## 🏥 Project Overview

SurgAgent is an **agentic AI system** that uses Google Gemini to intelligently detect and track surgical instruments in laparoscopic video. Unlike traditional computer vision systems, SurgAgent:

- **Reasons** about what it sees (scene analysis)
- **Adapts** its strategy (tool selection)
- **Recovers** from failures (replanning)
- **Understands** clinical context (surgical phases)

### The Problem

Surgical AI needs to be **intelligent**, not just accurate:
- **1,500 patients/year** injured by retained surgical instruments
- Current AI benchmarks only test accuracy, not reasoning
- No existing solution tests adaptation and failure recovery

### Our Solution

An agentic tracking system powered by **Google Gemini** that:
1. Analyzes surgical scenes using Gemini's vision capabilities
2. Selects optimal detection/tracking strategies
3. Monitors performance and adapts in real-time
4. Recovers gracefully from tracking failures
5. Provides explainable reasoning traces

---

## 📋 Submission Checklist

- [x] All code in `src/` runs without errors
- [x] `ARCHITECTURE.md` contains a clear diagram and explanation
- [x] `EXPLANATION.md` covers planning, tool use, memory, and limitations
- [x] `DEMO.md` links to a 3-5 min video with timestamped highlights
- [x] Gemini API integrated

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_TEAM/SurgAgent.git
cd SurgAgent

# Create environment
conda env create -f environment.yml
conda activate surgagent

# Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-here"

# Run the agent
python src/main.py
```

### Run Demo

```bash
# Run the full demo with sample surgical video
python src/demo.py --scenario all

# Expected output: 4 scenarios tested, average score 0.762
```

---

## 📁 Project Structure

```
SurgAgent/
├── src/                           # Main source code
│   ├── main.py                    # Entry point
│   ├── agent.py                   # SurgAgent core with Gemini
│   ├── gemini_vision.py           # Gemini API integration
│   ├── trackers/                  # Tracking algorithms
│   ├── detectors/                 # Detection algorithms
│   └── evaluation/                # Scoring and metrics
├── data/                          # Sample surgical videos
├── ARCHITECTURE.md                # System design
├── EXPLANATION.md                 # Planning & reasoning details
├── DEMO.md                        # Video demo link
├── environment.yml                # Conda environment
└── README.md                      # This file
```

---

## 🧠 Gemini Integration Highlights

### 1. Scene Analysis with Gemini Vision
```python
import google.generativeai as genai

def analyze_scene(frame):
    """Use Gemini to analyze surgical scene."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([
        "Analyze this surgical scene. Identify:",
        "1. Visible instruments and their types",
        "2. Scene challenges (smoke, blood, occlusion)",
        "3. Estimated surgical phase",
        frame
    ])
    return response.text
```

### 2. Adaptive Tool Selection
```python
def select_strategy(scene_analysis):
    """Gemini decides which tracking strategy to use."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(f"""
        Given this surgical scene analysis: {scene_analysis}
        
        Available detectors: [simple, yolov8, advanced]
        Available trackers: [simple, bytetrack, deepsort]
        
        Select the best combination and explain why.
    """)
    return parse_strategy(response.text)
```

### 3. Failure Recovery
```python
def handle_failure(failure_type, context):
    """Gemini reasons about how to recover."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(f"""
        A tracking failure occurred: {failure_type}
        Context: {context}
        
        What recovery action should the agent take?
        Options: reinitialize, switch_tracker, increase_threshold, skip_frames
    """)
    return response.text
```

---

## 📊 Evaluation Metrics

SurgAgent uses a **6-dimensional scoring** system:

| Metric | Weight | Description |
|--------|--------|-------------|
| **HOTA** | 35% | Tracking accuracy |
| **mAP** | 25% | Detection accuracy |
| **Surgical Context** | 15% | Clinical workflow alignment |
| **Real-time** | 10% | Speed (Tier 1: <50ms) |
| **Reasoning Quality** | 10% | Decision explanation |
| **Improvement** | 5% | Learning from feedback |

---

## 🏆 Test Results

### Real Surgical Video Analysis (VID01 - CholecTrack20)

| Metric | Value | Details |
|--------|-------|----------|
| **Instrument Accuracy** | **98%** | 149/152 frames correct (±1 tolerance) |
| **Occlusion Detection** | 88% | 134/152 frames |
| **Blood Detection** | 84% | 128/152 frames |
| **Phase Detection** | 72% | Primarily dissection phase |
| **Frames Analyzed** | 152 | Sampled from 45,175 total |
| **Processing Time** | 21.4 min | Avg 2.Gemini 1s/frame (API latency) |
| **API Cost** | ~$0.05 | Using gemini-2.0-flash-exp |

### Demo Scenarios (Simulated)

| Scenario | Score | Status |
|----------|-------|--------|
| Basic Tracking | 0.820 | ✅ PASS |
| Adaptive Selection | 0.751 | ✅ PASS |
| Failure Recovery | 0.744 | ✅ PASS |
| High Performance | 0.732 | ✅ PASS |

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and data flow
- **[EXPLANATION.md](EXPLANATION.md)** - Planning, tool use, memory, limitations
- **[DEMO.md](DEMO.md)** - Video demo with timestamps

---

## 🙏 Acknowledgments

- **CholecTrack20 Dataset**: CAMMA Research Group, IHU Strasbourg
- **Google Gemini API**: Powering our agentic reasoning
- **Google AI Studio**: Development and testing

---

## 📄 License

Apache 2.0 (Code) | CC BY-NC-SA 4.0 (Dataset)

---

*Built for the Google Agentic AI App Hackathon - December 2025*
=======
# SurgAgent
>>>>>>> c9aff8700d1323e81d5a033cf1640fec267ddc95
