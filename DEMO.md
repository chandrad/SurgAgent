# SurgAgent Demo

## 🎬 Video Demo: Google Drive

**▶️ [Watch the Full Demo Video](https://drive.google.com/drive/u/0/folders/1p3yydztFGrbtazQMfyxIK7Q_gIHbVajB)**


---

## ⏱️ Video Timestamps

| Time | Section | Description |
|------|---------|-------------|
| **0:00** | Introduction | Project overview and problem statement |
| **0:30** | The Problem | Why surgical AI needs to be intelligent |
| **1:00** | Architecture | System design and Gemini integration |
| **1:45** | Gemini Integration | Scene analysis and tool selection demos |
| **2:30** | Live Demo | Real surgical image analysis with Gemini |
| **3:30** | Results | Comprehensive test summary |
| **4:30** | Conclusion | Impact and future work |

---

## 🎬 Full Video Tracking Test (VID01)

**Successfully analyzed a 30-minute surgical video with Gemini Vision!**

### Video Details

| Property | Value |
|----------|-------|
| Video | VID01.mp4 |
| Duration | 1807s (30 min) |
| Size | 1.20 GB |
| Resolution | 854×480 |
| FPS | 25 |
| Total Frames | 45,175 |

### Frame-by-Frame Analysis

| Frame | Timestamp | Visibility | Instruments | Challenges | Phase |
|-------|-----------|------------|-------------|------------|-------|
| 0 | 0:00 | 2/10 | 0 | - | preparation |
| 9035 | 6:01 | 6/10 | 1 | blood, occlusion | dissection |
| 18070 | 12:03 | 6/10 | 2 | blood, occlusion | dissection |
| 27105 | 18:04 | 6/10 | 1 | blood, occlusion | dissection |
| 36140 | 24:05 | 6/10 | 1 | occlusion, blood | clipping |

### Surgical Progression Detected

```
0:00  ─────────────────────────────────────────────────────  30:07
  │                                                            │
  ├─ PREPARATION (0:00)                                       │
  │   Visibility: 2/10, no instruments visible                │
  │                                                            │
  ├─ DISSECTION (6:01 - 18:04)                                │
  │   Visibility: 6/10, 1-2 instruments                       │
  │   Challenges: blood, occlusion                            │
  │                                                            │
  └─ CLIPPING (24:05)                                         │
      Visibility: 6/10, 1 instrument                          │
      Phase transition detected by Gemini                     │
```

### Gemini's Strategy Recommendation

```
Detector: yolov8_surgical
Tracker: byte_track
Reasoning: "Given the presence of occlusion and blood, a balanced 
           approach is needed. yolov8_surgical offers good accuracy, 
           and byte_track provides good occlusion handling without 
           sacrificing too much speed."
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Frames Analyzed | 5 |
| Total API Time | 10.7s |
| Avg per Frame | 2.1s |
| API Calls | 6 |

---

## 🧪 Live Test Results (Real Gemini API)

### Scene Analysis - 3 Diverse Surgical Frames

| Image | Visibility | Instruments | Challenges | Phase | Time |
|-------|------------|-------------|------------|-------|------|
| 006701.png | 7/10 | 1 | occlusion | dissection | 4.24s |
| 059176.png | 6/10 | 3 | blood, occlusion | clipping | 3.29s |
| 088926.png | 6/10 | 3 | blood, occlusion | clipping | 2.93s |

### Strategy Selection

**Input Scene**: visibility=6, challenges=[blood, occlusion]

**Gemini's Decision**:
```
Detector: yolov8_surgical
Tracker: byte_track
Reasoning: "Given the visibility of 6/10 and the presence of blood 
           and occlusion, a balanced approach is needed. yolov8_surgical 
           offers a good balance between speed and accuracy for detection, 
           while byte_track provides good occlusion handling without being 
           as slow as deep_sort."
```

### Failure Recovery Scenarios

| Failure Type | Context | Gemini's Recovery | Reasoning |
|--------------|---------|-------------------|-----------|
| **track_loss** | frame 50, confidence 0.3 | `reinitialize` | "Track loss after only 5 frames suggests the tracker is struggling. Reinitializing allows re-acquisition without changing components." |
| **low_confidence** | heavy smoke, conf 0.45 | `switch_detector` | "The primary issue is low confidence due to heavy smoke. Switching to a smoke-robust detector is the optimal strategy." |

---

## 📊 Test Summary

```
============================================================
📋 COMPREHENSIVE TEST SUMMARY
============================================================

📈 Tests Completed:
   Scene Analyses: 3
   Strategy Selections: 1
   Failure Recoveries: 2
   Total API Calls: 6

🔍 Scene Diversity:
   006701.png: visibility=7, instruments=1, phase=dissection
   059176.png: visibility=6, instruments=3, phase=clipping
   088926.png: visibility=6, instruments=3, phase=clipping

✅ All tests passed successfully!
```

---

## 🖥️ Run Tests Yourself

### Prerequisites

```bash
# Set your Gemini API key
export GOOGLE_API_KEY="your-key-here"

# Navigate to src directory
cd AgenticAIHackathon_Dec2025/src
```

### Quick Test (3 API calls)

```bash
python test_gemini_e2e.py
```

### Comprehensive Test (6 API calls)

```bash
python test_comprehensive.py
```

### Expected Output

```
🖼️ Image 1/3: 006701.png
   ⏱️ Time: 4.24s
   👁️ Visibility: 7/10
   🔧 Instruments: 1
   ⚠️ Challenges: ['occlusion']
   📍 Phase: dissection

🎯 Selected Strategy:
   Detector: yolov8_surgical
   Tracker: byte_track
   
✅ Comprehensive testing complete!
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `gemini_test_results.json` | Basic test results |
| `comprehensive_test_results.json` | Full test results |
| `demo_results.json` | Demo scenario results |

---

## 💰 API Cost Estimate

| Test | API Calls | Estimated Cost |
|------|-----------|----------------|
| Basic (test_gemini_e2e.py) | 3 | ~$0.001 |
| Comprehensive (test_comprehensive.py) | 6 | ~$0.002 |
| Full Demo (demo.py) | 0 (simulated) | Free |

*Using gemini-2.0-flash-exp (generous free tier)*

---

*Demo created for the Google Agentic AI App Hackathon - December 2025*
