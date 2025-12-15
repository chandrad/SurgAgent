# SurgAgent Architecture

## System Overview

SurgAgent is an **agentic AI system** that uses Google Gemini to power intelligent surgical instrument tracking. The architecture follows a multi-stage reasoning pipeline with adaptive tool selection.

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SURGAGENT SYSTEM                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────┐     ┌─────────────────────────────────────────────┐    │
│   │                 │     │            GEMINI REASONING CORE            │    │
│   │  Surgical Video │     │  ┌─────────────────────────────────────┐   │    │
│   │    Input        │────▶│  │  1️⃣ Scene Analysis                  │   │    │
│   │                 │     │  │     "What am I seeing?"              │   │    │
│   └─────────────────┘     │  └─────────────┬───────────────────────┘   │    │
│                           │                │                            │    │
│                           │  ┌─────────────▼───────────────────────┐   │    │
│   ┌─────────────────┐     │  │  2️⃣ Tool Selection                  │   │    │
│   │  Tool Catalog   │────▶│  │     "Which strategy should I use?"  │   │    │
│   │  - Detectors    │     │  └─────────────┬───────────────────────┘   │    │
│   │  - Trackers     │     │                │                            │    │
│   └─────────────────┘     │  ┌─────────────▼───────────────────────┐   │    │
│                           │  │  3️⃣ Tracking Execution              │   │    │
│                           │  │     "Process each frame"            │   │    │
│                           │  └─────────────┬───────────────────────┘   │    │
│                           │                │                            │    │
│                           │  ┌─────────────▼───────────────────────┐   │    │
│                           │  │  4️⃣ Quality Monitoring              │   │    │
│                           │  │     "How am I doing?"               │   │    │
│   ┌─────────────────┐     │  └─────────────┬───────────────────────┘   │    │
│   │  Confidence     │◀────│                │                            │    │
│   │  < Threshold?   │     │  ┌─────────────▼───────────────────────┐   │    │
│   │  ↓              │     │  │  5️⃣ Failure Recovery (if needed)    │   │    │
│   │  Trigger        │────▶│  │     "How do I fix this?"            │   │    │
│   │  Recovery       │     │  └─────────────┬───────────────────────┘   │    │
│   └─────────────────┘     │                │                            │    │
│                           │  ┌─────────────▼───────────────────────┐   │    │
│                           │  │  6️⃣ Validation                      │   │    │
│                           │  │     "Final quality check"           │   │    │
│                           │  └─────────────────────────────────────┘   │    │
│                           └─────────────────────────────────────────────┘    │
│                                              │                                │
│   ┌──────────────────────────────────────────▼───────────────────────────┐   │
│   │                         OUTPUT                                        │   │
│   │  • Tracking Predictions (bounding boxes, track IDs)                  │   │
│   │  • Reasoning Trace (decisions + explanations)                        │   │
│   │  • Confidence Scores (per-frame)                                     │   │
│   │  • Quality Checkpoints                                               │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Details

### 1. Gemini Vision Module (`gemini_vision.py`)

Uses **Gemini 2.0 Flash (Experimental)** for fast scene understanding:

```python
# Scene Analysis with Gemini
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content([frame_image, prompt])
```

**Capabilities**:
- Instrument identification (grasper, bipolar, hook, scissors, clipper, irrigator, specimen bag)
- Scene challenge detection (smoke, blood, occlusion)
- Surgical phase estimation
- Visibility quality assessment (1-10 scale)

### 2. Strategy Selector (`agent.py`)

Uses **Gemini 2.0 Flash** for strategic reasoning:

```python
# Tool Selection with Gemini
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content(f"""
    Scene: {scene_analysis}
    Select optimal detector + tracker combination.
""")
```

**Decision Factors**:
- Scene complexity
- Visibility conditions
- Available computational budget
- Historical performance

### 3. Tracker Pool (`trackers/`)

| Tracker | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| SimpleTracker | Fast | Low | Clear scenes |
| ByteTrack | Medium | High | General use |
| DeepSORT | Slow | Very High | Occlusions |

### 4. Detector Pool (`detectors/`)

| Detector | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| SimpleDetector | Fast | Low | Testing |
| YOLOv8 Surgical | Medium | High | Production |
| AdvancedDetector | Slow | Very High | Smoke/blood |

### 5. Evaluation Module (`evaluation/`)

**Multi-dimensional scoring**:
- HOTA (tracking accuracy)
- mAP (detection accuracy)
- Surgical Context (clinical relevance)
- Real-time Performance
- Reasoning Quality

---

## 🔄 Data Flow

```
Input Video
    │
    ▼
┌─────────────────┐
│ Frame Extraction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Gemini Vision  │────▶│  Scene Analysis │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Gemini Reasoning│────▶│ Strategy Choice │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Detection       │─────┐
└────────┬────────┘     │
         │              │
         ▼              │ Per-frame
┌─────────────────┐     │ loop
│ Tracking        │     │
└────────┬────────┘     │
         │              │
         ▼              │
┌─────────────────┐     │
│ Quality Check   │─────┘
└────────┬────────┘
         │
         ▼ (if confidence < threshold)
┌─────────────────┐
│ Recovery        │
│ (Gemini decides)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Results  │
└─────────────────┘
```

---

## 🧠 Gemini API Usage

### Models Used

| Model | Purpose | Why |
|-------|---------|-----|
| `gemini-2.0-flash-exp` | Scene analysis | Fast, good vision, available |
| `gemini-2.0-flash-exp` | Strategy selection | Reasoning + speed |
| `gemini-2.0-flash-exp` | Failure recovery | Consistent performance |

> **Note**: Initially planned to use `gemini-1.5-pro` for reasoning, but `gemini-2.0-flash-exp` provided excellent performance for all tasks while staying within rate limits (10 req/min).

### API Calls Per Video

| Stage | Calls | Tokens (est.) |
|-------|-------|---------------|
| Scene Analysis | 1 | ~1,000 |
| Tool Selection | 1 | ~500 |
| Quality Checks | 3-6 | ~300 each |
| Recovery (if any) | 0-2 | ~400 each |
| **Total** | ~8 | ~3,000 |

---

## 📊 Surgical Context Awareness

### Phase-Tool Constraints

The system understands surgical workflow:

```python
PHASE_CONSTRAINTS = {
    "clipping_cutting": {
        "expected": ["clipper", "scissors", "grasper"],
        "forbidden": ["hook"]  # Safety rule
    },
    "gallbladder_dissection": {
        "expected": ["grasper", "hook", "bipolar"],
        "forbidden": ["clipper"]  # Comes later
    }
}
```

### Tool Interaction Detection

Recognizes when instruments work together:
- Grasper + Hook (dissection)
- Grasper + Clipper (clipping)
- Grasper + Scissors (cutting)

---

## 🔐 Security Considerations

- API keys stored in environment variables
- No patient data transmitted to Gemini
- Video frames processed locally when possible
- Gemini only receives anonymized images

---

## 📈 Scalability

| Metric | Current | With Optimization |
|--------|---------|-------------------|
| Videos/hour | 10 | 50+ |
| Latency | ~2s/frame | ~200ms/frame |
| API cost | ~$0.01/video | ~$0.005/video |

---

*See [EXPLANATION.md](EXPLANATION.md) for detailed reasoning and planning documentation.*
