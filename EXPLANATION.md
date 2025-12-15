# SurgAgent Explanation

This document explains the **planning**, **tool use**, **memory**, and **limitations** of SurgAgent — our agentic surgical instrument tracking system powered by Google Gemini.

---

## 🧠 1. Planning: How the Agent Reasons

### Multi-Stage Reasoning Pipeline

SurgAgent doesn't just run detection algorithms blindly. It follows a structured planning process:

```
Stage 1: Scene Analysis    → "What am I looking at?"
Stage 2: Tool Selection    → "What's the best approach?"
Stage 3: Execution         → "Run tracking on each frame"
Stage 4: Quality Check     → "Am I doing well?"
Stage 5: Recovery          → "How do I fix problems?"
Stage 6: Validation        → "Final quality assessment"
```

### Gemini-Powered Planning

**Scene Analysis** (Gemini Vision):
```python
prompt = """
Analyze this surgical frame:
1. What instruments are visible?
2. What challenges exist (smoke, blood, occlusion)?
3. What surgical phase is this?
4. What's the visibility quality (1-10)?
"""
scene_analysis = gemini_vision.analyze(frame, prompt)
```

**Strategy Selection** (Gemini Pro):
```python
prompt = f"""
Given scene analysis: {scene_analysis}
Available strategies:
- Simple: fast but low accuracy
- Standard: balanced
- Advanced: slow but handles occlusion

Which strategy should I use and why?
"""
strategy = gemini_pro.decide(prompt)
```

### Why This Matters

Traditional CV systems are **reactive** — they process each frame independently. SurgAgent is **proactive**:

| Traditional | SurgAgent |
|-------------|-----------|
| Fixed algorithm | Adaptive strategy |
| No scene understanding | Gemini analyzes context |
| Fails on occlusion | Plans for recovery |
| No explanation | Full reasoning trace |

---

## 🔧 2. Tool Use: What the Agent Can Do

### Available Tools

SurgAgent has access to multiple detection and tracking algorithms:

#### Detectors

| Tool | When Used | Gemini's Reasoning |
|------|-----------|-------------------|
| `SimpleDetector` | Testing | "Fast enough for clear scenes" |
| `YOLOv8Surgical` | 80% of cases | "Best balance of speed/accuracy" |
| `AdvancedDetector` | Smoke/occlusion | "Handles difficult conditions" |

#### Trackers

| Tool | When Used | Gemini's Reasoning |
|------|-----------|-------------------|
| `SimpleTracker` | Testing | "Basic IoU matching" |
| `ByteTrack` | Standard | "Good with temporary occlusions" |
| `DeepSORT` | Complex | "Best ID preservation" |

### Dynamic Tool Switching

The agent can switch tools mid-video based on conditions:

```python
# Example: Smoke detected at frame 150
ToolSwitch(
    frame=150,
    from_tool="yolov8_surgical",
    to_tool="advanced_detector",
    reason="occlusion_detected",
    confidence_before=0.68,
    confidence_after=0.85,
    reasoning="Smoke detected. Switching to smoke-robust detector."
)
```

### How Gemini Selects Tools

```python
# Gemini reasoning example
"""
Input: Scene has moderate smoke, 2 instruments visible
Available: simple_detector, yolov8_surgical, advanced_detector

Gemini's reasoning:
- Smoke present → need robust detector
- Only 2 instruments → don't need fastest option
- Decision: advanced_detector
- Tracker: bytetrack (handles occlusions from smoke)
"""
```

---

## 💾 3. Memory: What the Agent Remembers

### Short-Term Memory (Per-Video)

The agent maintains context throughout each video:

```python
class AgentMemory:
    # Current state
    current_strategy: str
    active_tracks: Dict[int, Track]
    confidence_history: List[float]
    
    # Reasoning history
    reasoning_trace: List[ReasoningStep]
    tool_switches: List[ToolSwitch]
    recovery_events: List[RecoveryEvent]
    quality_checkpoints: List[QualityCheckpoint]
```

### Quality Checkpoints

Every 15 frames, the agent evaluates its performance:

```python
QualityCheckpoint(
    frame=30,
    metrics={"avg_confidence": 0.82, "track_continuity": 0.95},
    decision="continue",  # or "replan", "switch_tool"
    threshold_used=0.65,
    reasoning="Confidence 0.82 > threshold 0.65, continuing"
)
```

### Failure Memory

When failures occur, the agent remembers what worked:

```python
RecoveryEvent(
    frame=100,
    failure_type="track_loss",
    recovery_action="Increased IoU threshold from 0.5 to 0.7",
    success=True,
    frames_to_recover=5,
    tools_used=["bytetrack"]
)
# Agent now knows: higher IoU threshold works for track loss
```

---

## ⚠️ 4. Limitations: What We Don't Claim

### Current Technical Limitations

| Limitation | Description | Future Work |
|------------|-------------|-------------|
| **Placeholder Metrics** | HOTA/mAP use simulated values | Integrate TrackEval |
| **Single Procedure** | Only cholecystectomy tested | Expand to other surgeries |
| **No Real-Time** | Demo runs offline | Optimize for 30fps |
| **Limited Dataset** | 8 episode clips | Extract more from dataset |

### Gemini-Specific Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **API Latency** | ~500ms per call | Cache scene analysis |
| **Rate Limits** | 60 RPM on free tier | Batch requests |
| **No Fine-Tuning** | Can't adapt to new instruments | Use detailed prompts |
| **Cost** | ~$0.01 per video | Optimize token usage |

### What We Don't Claim

❌ **Not Clinical-Ready**: This is a research prototype, not FDA-approved  
❌ **Not Real-Time**: Current implementation is for offline analysis  
❌ **Not Fully Autonomous**: Designed for human-AI collaboration  
❌ **Not Validated**: No clinical trials or surgeon evaluation yet  

### Honest Assessment

| Aspect | Status | Evidence |
|--------|--------|----------|
| Core functionality | ✅ Working | Demo runs successfully |
| Gemini integration | ✅ Novel | Multi-stage reasoning |
| Surgical context | ✅ Innovative | Phase-aware evaluation |
| Production-ready | ❌ Not yet | Placeholder metrics |
| Clinically validated | ❌ No | Research prototype only |

---

## 🎯 5. Why This Approach?

### The Agentic Advantage

Traditional tracking systems fail because they can't:
1. Understand **why** a failure occurred
2. Decide **how** to recover
3. Explain **what** they did

SurgAgent addresses all three with Gemini-powered reasoning.

### Alignment with Hackathon Criteria

| Criterion | How SurgAgent Addresses It |
|-----------|---------------------------|
| **Technical Excellence** | Multi-stage pipeline, robust architecture |
| **Innovative Gemini Use** | Vision + reasoning for adaptive tracking |
| **Societal Impact** | Surgical safety (1,500 injuries/year) |
| **Architecture** | Clean separation, documented reasoning |

---

## 📚 Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design diagrams
- [DEMO.md](DEMO.md) - Video demo with timestamps
- [README.md](README.md) - Quick start guide

---

*SurgAgent: Bringing reasoning and adaptation to surgical AI*
