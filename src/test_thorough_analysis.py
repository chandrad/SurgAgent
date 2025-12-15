#!/usr/bin/env python3
"""
SurgAgent Thorough Video Analysis with Ground Truth Comparison

Performs comprehensive analysis of VID01:
1. Parses CholecTrack20 JSON annotations
2. Extracts and analyzes 15 key frames with Gemini
3. Compares Gemini's analysis to ground truth
4. Generates detailed report

Usage:
    export GOOGLE_API_KEY="your-key"
    python test_thorough_analysis.py --video VID01 --frames 15
"""

import os
import sys
import json
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Check API key
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Set GOOGLE_API_KEY environment variable")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from agent import SurgAgent

DATA_PATH = Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Testing"

# CholecTrack20 instrument mapping
INSTRUMENT_NAMES = {
    0: "grasper",
    1: "bipolar", 
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "specimen-bag"
}

# Phase mapping (from CholecT50)
PHASE_NAMES = {
    0: "preparation",
    1: "calot_triangle_dissection",
    2: "clipping_and_cutting",
    3: "gallbladder_dissection",
    4: "gallbladder_packaging",
    5: "cleaning_and_coagulation",
    6: "gallbladder_retraction"
}


def load_annotations(json_path: Path) -> Dict:
    """Load and parse CholecTrack20 annotations."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def analyze_ground_truth(annotations: Dict) -> Dict:
    """Analyze ground truth annotations comprehensively."""
    frames_data = annotations.get("annotations", {})
    video_info = annotations.get("video", {})
    
    stats = {
        "total_annotated_frames": len(frames_data),
        "instruments_by_type": Counter(),
        "phases_by_type": Counter(),
        "challenges": {
            "occluded": 0,
            "bleeding": 0,
            "smoke": 0,
            "blurred": 0,
            "reflection": 0,
            "stainedlens": 0
        },
        "instruments_per_frame": [],
        "frame_numbers": []
    }
    
    for frame_num, frame_annotations in frames_data.items():
        stats["frame_numbers"].append(int(frame_num))
        stats["instruments_per_frame"].append(len(frame_annotations))
        
        for ann in frame_annotations:
            # Count instruments
            inst_id = ann.get("instrument", -1)
            if inst_id in INSTRUMENT_NAMES:
                stats["instruments_by_type"][INSTRUMENT_NAMES[inst_id]] += 1
            
            # Count phases
            phase_id = ann.get("phase", -1)
            if phase_id in PHASE_NAMES:
                stats["phases_by_type"][PHASE_NAMES[phase_id]] += 1
            
            # Count challenges
            if ann.get("occluded"):
                stats["challenges"]["occluded"] += 1
            if ann.get("bleeding"):
                stats["challenges"]["bleeding"] += 1
            if ann.get("smoke"):
                stats["challenges"]["smoke"] += 1
            if ann.get("blurred"):
                stats["challenges"]["blurred"] += 1
            if ann.get("reflection"):
                stats["challenges"]["reflection"] += 1
            if ann.get("stainedlens"):
                stats["challenges"]["stainedlens"] += 1
    
    # Calculate averages
    if stats["instruments_per_frame"]:
        stats["avg_instruments_per_frame"] = sum(stats["instruments_per_frame"]) / len(stats["instruments_per_frame"])
    
    return stats


def get_gt_for_frame(annotations: Dict, frame_num: int) -> Dict:
    """Get ground truth for a specific frame."""
    frames_data = annotations.get("annotations", {})
    
    # Find closest annotated frame
    annotated_frames = [int(f) for f in frames_data.keys()]
    if not annotated_frames:
        return {}
    
    closest = min(annotated_frames, key=lambda x: abs(x - frame_num))
    frame_annotations = frames_data.get(str(closest), [])
    
    instruments = []
    phase = None
    challenges = []
    
    for ann in frame_annotations:
        inst_id = ann.get("instrument", -1)
        if inst_id in INSTRUMENT_NAMES:
            instruments.append(INSTRUMENT_NAMES[inst_id])
        
        phase_id = ann.get("phase", -1)
        if phase_id in PHASE_NAMES:
            phase = PHASE_NAMES[phase_id]
        
        if ann.get("occluded"):
            challenges.append("occlusion")
        if ann.get("bleeding"):
            challenges.append("blood")
        if ann.get("smoke"):
            challenges.append("smoke")
        if ann.get("blurred"):
            challenges.append("blur")
    
    return {
        "frame": closest,
        "instruments": list(set(instruments)),
        "instrument_count": len(frame_annotations),
        "phase": phase,
        "challenges": list(set(challenges))
    }


def extract_frame(video_path: Path, frame_idx: int, output_path: Path, fps: int = 25) -> bool:
    """Extract a single frame from video."""
    timestamp = frame_idx / fps
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1", "-q:v", "2",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def run_thorough_analysis(video_name: str = "VID01", num_frames: int = 15):
    """Run thorough analysis with ground truth comparison."""
    
    print("=" * 70)
    print("🔬 SURGAGENT THOROUGH VIDEO ANALYSIS")
    print("   With Ground Truth Comparison")
    print("=" * 70)
    
    # Load files
    video_dir = DATA_PATH / video_name
    video_path = video_dir / f"{video_name}.mp4"
    json_path = video_dir / f"{video_name}.json"
    
    if not video_path.exists() or not json_path.exists():
        print(f"❌ Video or annotations not found: {video_dir}")
        return None
    
    print(f"\n📹 Video: {video_name}")
    print(f"   Size: {video_path.stat().st_size / (1024**3):.2f} GB")
    
    # Load and analyze ground truth
    print("\n📊 GROUND TRUTH ANALYSIS")
    print("-" * 40)
    annotations = load_annotations(json_path)
    gt_stats = analyze_ground_truth(annotations)
    
    print(f"   Annotated frames: {gt_stats['total_annotated_frames']}")
    print(f"   Avg instruments/frame: {gt_stats.get('avg_instruments_per_frame', 0):.1f}")
    print(f"\n   Instruments found:")
    for inst, count in gt_stats["instruments_by_type"].most_common():
        print(f"      {inst}: {count}")
    print(f"\n   Phases found:")
    for phase, count in gt_stats["phases_by_type"].most_common():
        print(f"      {phase}: {count}")
    print(f"\n   Challenges:")
    for challenge, count in gt_stats["challenges"].items():
        if count > 0:
            print(f"      {challenge}: {count}")
    
    # Initialize agent
    print("\n🤖 GEMINI ANALYSIS")
    print("-" * 40)
    agent = SurgAgent(api_key=API_KEY)
    
    # Select frames evenly across video
    frame_numbers = sorted(gt_stats["frame_numbers"])
    if len(frame_numbers) >= num_frames:
        step = len(frame_numbers) // num_frames
        target_frames = [frame_numbers[i * step] for i in range(num_frames)]
    else:
        target_frames = frame_numbers[:num_frames]
    
    print(f"   Analyzing {len(target_frames)} frames across the video...")
    
    results = []
    comparisons = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, frame_idx in enumerate(target_frames):
            frame_path = temp_path / f"frame_{frame_idx:06d}.png"
            
            # Extract frame
            if not extract_frame(video_path, frame_idx, frame_path):
                print(f"   ⚠️ Failed to extract frame {frame_idx}")
                continue
            
            # Get ground truth
            gt = get_gt_for_frame(annotations, frame_idx)
            
            # Gemini analysis
            start = time.time()
            gemini_analysis = agent.analyze_scene(str(frame_path))
            elapsed = time.time() - start
            
            # Compare
            comparison = {
                "frame": frame_idx,
                "timestamp_s": frame_idx / 25,
                "ground_truth": gt,
                "gemini_analysis": gemini_analysis,
                "processing_time": elapsed,
                "matches": {
                    "instrument_count": abs(
                        gt.get("instrument_count", 0) - 
                        gemini_analysis.get("instrument_count", 0)
                    ) <= 1,
                    "phase": (
                        gt.get("phase", "").replace("_", " ").lower() in 
                        gemini_analysis.get("estimated_phase", "").lower() or
                        gemini_analysis.get("estimated_phase", "").lower() in 
                        gt.get("phase", "").replace("_", " ").lower()
                    ) if gt.get("phase") else True
                }
            }
            comparisons.append(comparison)
            
            # Print progress
            progress = f"[{i+1}/{len(target_frames)}]"
            match_status = "✅" if comparison["matches"]["instrument_count"] else "❌"
            print(f"   {progress} Frame {frame_idx}: GT={gt.get('instrument_count',0)} inst, "
                  f"Gemini={gemini_analysis.get('instrument_count',0)} inst {match_status} ({elapsed:.1f}s)")
    
    # Analysis comparison summary
    print("\n" + "=" * 70)
    print("📈 COMPARISON SUMMARY: GEMINI vs GROUND TRUTH")
    print("=" * 70)
    
    inst_matches = sum(1 for c in comparisons if c["matches"]["instrument_count"])
    phase_matches = sum(1 for c in comparisons if c["matches"]["phase"])
    total = len(comparisons)
    
    print(f"\n   Frames analyzed: {total}")
    print(f"   Instrument count matches (±1): {inst_matches}/{total} ({100*inst_matches/total:.0f}%)")
    print(f"   Phase matches: {phase_matches}/{total} ({100*phase_matches/total:.0f}%)")
    
    # Detailed comparison table
    print("\n📋 DETAILED FRAME-BY-FRAME COMPARISON")
    print("-" * 70)
    print(f"{'Frame':<8} {'Time':<8} {'GT Inst':<10} {'Gemini':<10} {'GT Phase':<20} {'Gemini Phase':<15}")
    print("-" * 70)
    
    for c in comparisons:
        gt = c["ground_truth"]
        gem = c["gemini_analysis"]
        frame_str = str(c["frame"])
        time_str = f"{c['timestamp_s']:.0f}s"
        gt_inst = str(gt.get("instrument_count", "?"))
        gem_inst = str(gem.get("instrument_count", "?"))
        gt_phase = gt.get("phase", "?")[:18] if gt.get("phase") else "?"
        gem_phase = gem.get("estimated_phase", "?")[:13]
        
        print(f"{frame_str:<8} {time_str:<8} {gt_inst:<10} {gem_inst:<10} {gt_phase:<20} {gem_phase:<15}")
    
    # Challenges detected
    print("\n🔍 CHALLENGES DETECTED BY GEMINI")
    print("-" * 40)
    all_challenges = []
    for c in comparisons:
        all_challenges.extend(c["gemini_analysis"].get("scene_challenges", []))
    challenge_counts = Counter(all_challenges)
    for challenge, count in challenge_counts.most_common():
        print(f"   {challenge}: {count} frames")
    
    # Strategy recommendation
    print("\n🎯 FINAL STRATEGY RECOMMENDATION")
    print("-" * 40)
    
    # Use most recent analysis for strategy
    if comparisons:
        last_analysis = comparisons[-1]["gemini_analysis"]
        strategy = agent.select_strategy(last_analysis)
        print(f"\n   Detector: {strategy.get('detector', 'N/A')}")
        print(f"   Tracker: {strategy.get('tracker', 'N/A')}")
        print(f"   Reasoning: {strategy.get('reasoning', 'N/A')[:200]}...")
    
    # Performance stats
    total_time = sum(c["processing_time"] for c in comparisons)
    avg_time = total_time / len(comparisons) if comparisons else 0
    
    print(f"\n⚡ PERFORMANCE")
    print("-" * 40)
    print(f"   Total API calls: {len(comparisons) + 1}")
    print(f"   Total API time: {total_time:.1f}s")
    print(f"   Avg per frame: {avg_time:.1f}s")
    print(f"   Estimated cost: ~${len(comparisons) * 0.0003:.3f}")
    
    # Save results
    output = {
        "video": video_name,
        "ground_truth_stats": {
            "total_frames": gt_stats["total_annotated_frames"],
            "instruments": dict(gt_stats["instruments_by_type"]),
            "phases": dict(gt_stats["phases_by_type"]),
            "challenges": gt_stats["challenges"]
        },
        "gemini_analysis": {
            "frames_analyzed": len(comparisons),
            "instrument_accuracy": f"{100*inst_matches/total:.0f}%",
            "phase_accuracy": f"{100*phase_matches/total:.0f}%"
        },
        "comparisons": comparisons,
        "strategy": strategy if comparisons else {},
        "api_stats": {
            "total_calls": len(comparisons) + 1,
            "total_time_s": total_time
        }
    }
    
    output_path = Path(__file__).parent.parent / f"thorough_analysis_{video_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("\n✅ Thorough analysis complete!")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SurgAgent Thorough Analysis")
    parser.add_argument("--video", type=str, default="VID01", help="Video name")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to analyze")
    args = parser.parse_args()
    
    run_thorough_analysis(args.video, args.frames)
