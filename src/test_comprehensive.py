#!/usr/bin/env python3
"""
SurgAgent Comprehensive Scenario Testing

Tests the Gemini-powered agent across multiple surgical images
with different scene characteristics to demonstrate adaptive behavior.

Cost Warning: Uses ~6-8 Gemini API calls total (minimal cost)

Usage:
    export GOOGLE_API_KEY="your-key"
    python test_comprehensive.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Check API key
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Set GOOGLE_API_KEY environment variable")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from agent import SurgAgent

# Find available surgical images
DATA_PATH = Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20"


def find_test_images(max_images: int = 3) -> List[Path]:
    """Find surgical images for testing (limit to save API costs)."""
    images = []
    if DATA_PATH.exists():
        # Get a diverse set of images from different frame numbers
        all_images = sorted(DATA_PATH.rglob("*.png"))
        if len(all_images) >= max_images:
            # Take sparse samples for diversity
            step = len(all_images) // max_images
            images = [all_images[i * step] for i in range(max_images)]
        else:
            images = all_images[:max_images]
    return images


def run_comprehensive_test():
    """Run comprehensive tests across multiple scenarios."""
    
    print("=" * 70)
    print("🏥 SURGAGENT COMPREHENSIVE SCENARIO TESTING")
    print("   Powered by Google Gemini 2.0 Flash")
    print("=" * 70)
    
    # Initialize agent
    print("\n🔧 Initializing SurgAgent...")
    agent = SurgAgent(api_key=API_KEY)
    
    # Find test images
    test_images = find_test_images(max_images=3)  # Limit to 3 for cost
    if not test_images:
        print("❌ No test images found. Please download CholecTrack20 dataset.")
        return None
    
    print(f"📸 Found {len(test_images)} test images")
    
    results = []
    
    # Test 1: Scene Analysis across multiple images
    print("\n" + "=" * 70)
    print("📊 TEST 1: SCENE ANALYSIS ACROSS DIVERSE FRAMES")
    print("=" * 70)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n🖼️ Image {i}/{len(test_images)}: {img_path.name}")
        
        start = time.time()
        analysis = agent.analyze_scene(str(img_path))
        elapsed = time.time() - start
        
        result = {
            "image": img_path.name,
            "analysis": analysis,
            "time_s": round(elapsed, 2)
        }
        results.append(result)
        
        # Display results
        print(f"   ⏱️ Time: {elapsed:.2f}s")
        print(f"   👁️ Visibility: {analysis.get('visibility_score', 'N/A')}/10")
        print(f"   🔧 Instruments: {analysis.get('instrument_count', 'N/A')}")
        print(f"   ⚠️ Challenges: {analysis.get('scene_challenges', [])}")
        print(f"   📍 Phase: {analysis.get('estimated_phase', 'N/A')}")
    
    # Test 2: Strategy Adaptation
    print("\n" + "=" * 70)
    print("🎯 TEST 2: STRATEGY ADAPTATION")
    print("=" * 70)
    
    # Use the last analysis for strategy selection
    if results:
        last_analysis = results[-1]["analysis"]
        print(f"\nUsing scene: {results[-1]['image']}")
        print(f"Scene characteristics: visibility={last_analysis.get('visibility_score')}, "
              f"challenges={last_analysis.get('scene_challenges', [])}")
        
        strategy = agent.select_strategy(last_analysis)
        
        print(f"\n🔍 Selected Strategy:")
        print(f"   Detector: {strategy.get('detector', 'N/A')}")
        print(f"   Tracker: {strategy.get('tracker', 'N/A')}")
        print(f"   Reasoning: {strategy.get('reasoning', 'N/A')[:200]}...")
    
    # Test 3: Multiple Failure Scenarios
    print("\n" + "=" * 70)
    print("🔧 TEST 3: FAILURE RECOVERY SCENARIOS")
    print("=" * 70)
    
    failure_scenarios = [
        {"type": "track_loss", "context": {"frame": 50, "confidence": 0.3, "last_seen": "frame 45"}},
        {"type": "low_confidence", "context": {"frame": 100, "confidence": 0.45, "scene": "heavy smoke"}},
    ]
    
    for scenario in failure_scenarios:
        print(f"\n⚠️ Failure: {scenario['type']}")
        print(f"   Context: {scenario['context']}")
        
        action = agent.handle_failure(scenario['type'], scenario['context'])
        print(f"   ✅ Recovery: {action}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    total_api_calls = len(results) + 1 + len(failure_scenarios)  # scenes + strategy + recoveries
    
    print(f"\n📈 Tests Completed:")
    print(f"   Scene Analyses: {len(results)}")
    print(f"   Strategy Selections: 1")
    print(f"   Failure Recoveries: {len(failure_scenarios)}")
    print(f"   Total API Calls: {total_api_calls}")
    
    print(f"\n🔍 Scene Diversity:")
    for r in results:
        print(f"   {r['image']}: visibility={r['analysis'].get('visibility_score')}, "
              f"instruments={r['analysis'].get('instrument_count')}, "
              f"phase={r['analysis'].get('estimated_phase')}")
    
    print("\n🧠 Full Reasoning Trace:")
    print(agent.get_reasoning_summary())
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_api_calls": total_api_calls,
        "scene_analyses": results,
        "reasoning_trace": [vars(r) for r in agent.reasoning_trace],
        "tool_switches": [vars(t) for t in agent.tool_switches],
        "recovery_events": [vars(r) for r in agent.recovery_events],
    }
    
    output_path = Path(__file__).parent.parent / "comprehensive_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("\n✅ Comprehensive testing complete!")
    
    return output


if __name__ == "__main__":
    run_comprehensive_test()
