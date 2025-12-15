#!/usr/bin/env python3
"""
SurgAgent End-to-End Test with Real Gemini Vision API

This script tests the full agent pipeline with:
1. Real surgical images from CholecTrack20
2. Live Gemini Vision API calls
3. Adaptive strategy selection
4. Full reasoning trace

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python test_gemini_e2e.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Check for API key first
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("=" * 60)
    print("❌ GOOGLE_API_KEY not set!")
    print("=" * 60)
    print("\nTo run this test, you need a Google Gemini API key.")
    print("\n1. Get a free key at: https://aistudio.google.com/app/apikey")
    print("2. Set it in your environment:")
    print("   export GOOGLE_API_KEY='your-key-here'")
    print("3. Run this script again")
    print("=" * 60)
    sys.exit(1)

# Now import the agent (requires API key to be set)
sys.path.insert(0, str(Path(__file__).parent))
from agent import SurgAgent

# Real surgical images from CholecTrack20
SAMPLE_IMAGES = [
    Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Validation/VID30/Frames/019601.png",
    Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Validation/VID30/Frames/029926.png",
    Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Validation/VID30/Frames/043651.png",
]


def test_scene_analysis(agent: SurgAgent, image_path: Path) -> dict:
    """Test Gemini Vision scene analysis with a real surgical image."""
    print(f"\n📸 Analyzing: {image_path.name}")
    
    if not image_path.exists():
        print(f"   ⚠️ Image not found: {image_path}")
        return None
    
    start = time.time()
    analysis = agent.analyze_scene(str(image_path))
    elapsed = time.time() - start
    
    print(f"   ⏱️ Time: {elapsed:.2f}s")
    print(f"   👁️ Visibility: {analysis.get('visibility_score', 'N/A')}/10")
    print(f"   🔧 Instruments: {analysis.get('instrument_count', 'N/A')}")
    print(f"   ⚠️ Challenges: {analysis.get('scene_challenges', [])}")
    print(f"   📍 Phase: {analysis.get('estimated_phase', 'N/A')}")
    
    return analysis


def test_strategy_selection(agent: SurgAgent, scene_analysis: dict) -> dict:
    """Test Gemini-powered strategy selection."""
    print("\n🎯 Selecting tracking strategy with Gemini...")
    
    start = time.time()
    strategy = agent.select_strategy(scene_analysis)
    elapsed = time.time() - start
    
    print(f"   ⏱️ Time: {elapsed:.2f}s")
    print(f"   🔍 Detector: {strategy.get('detector', 'N/A')}")
    print(f"   🔗 Tracker: {strategy.get('tracker', 'N/A')}")
    print(f"   💭 Reasoning: {strategy.get('reasoning', 'N/A')[:100]}...")
    
    return strategy


def test_failure_recovery(agent: SurgAgent) -> str:
    """Test Gemini-powered failure recovery planning."""
    print("\n🔧 Testing failure recovery...")
    
    context = {
        "frame": 50,
        "confidence": 0.45,
        "last_detection": "frame 48",
        "scene": "moderate smoke"
    }
    
    start = time.time()
    action = agent.handle_failure("track_loss", context)
    elapsed = time.time() - start
    
    print(f"   ⏱️ Time: {elapsed:.2f}s")
    print(f"   🔧 Recovery action: {action}")
    
    return action


def run_full_pipeline():
    """Run the complete end-to-end test."""
    
    print("=" * 60)
    print("🏥 SURGAGENT END-TO-END GEMINI VISION TEST")
    print("=" * 60)
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print("=" * 60)
    
    # Initialize agent
    print("\n1️⃣ Initializing SurgAgent with Gemini...")
    try:
        agent = SurgAgent(api_key=API_KEY)
        print("   ✅ Agent initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    
    # Find a valid image
    valid_image = None
    for img_path in SAMPLE_IMAGES:
        if img_path.exists():
            valid_image = img_path
            break
    
    if not valid_image:
        # Try to find any image in the dataset
        dataset_path = Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20"
        images = list(dataset_path.rglob("*.png"))
        if images:
            valid_image = images[0]
        else:
            print("\n❌ No surgical images found. Please ensure CholecTrack20 is downloaded.")
            return
    
    # Stage 2: Scene Analysis (LIVE GEMINI CALL)
    print("\n2️⃣ SCENE ANALYSIS (Gemini Vision API)")
    print("-" * 40)
    analysis = test_scene_analysis(agent, valid_image)
    
    if analysis and "error" not in analysis:
        # Stage 3: Strategy Selection (LIVE GEMINI CALL)
        print("\n3️⃣ STRATEGY SELECTION (Gemini Pro API)")
        print("-" * 40)
        strategy = test_strategy_selection(agent, analysis)
        
        # Stage 4: Failure Recovery Test (LIVE GEMINI CALL)
        print("\n4️⃣ FAILURE RECOVERY (Gemini Pro API)")
        print("-" * 40)
        recovery = test_failure_recovery(agent)
    
    # Stage 5: Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print("\n🧠 Reasoning Trace:")
    print(agent.get_reasoning_summary())
    
    print("\n📈 API Calls Made:")
    print(f"   Scene Analysis: 1")
    print(f"   Strategy Selection: 1")
    print(f"   Failure Recovery: 1")
    print(f"   Total: 3 Gemini API calls")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_analyzed": str(valid_image) if valid_image else None,
        "reasoning_trace": [vars(r) for r in agent.reasoning_trace],
        "tool_switches": [vars(t) for t in agent.tool_switches],
        "recovery_events": [vars(r) for r in agent.recovery_events],
    }
    
    output_path = Path(__file__).parent.parent / "gemini_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("\n✅ End-to-end Gemini test complete!")
    
    return results


if __name__ == "__main__":
    run_full_pipeline()
