#!/usr/bin/env python3
"""
SurgAgent Sampled Full Video Analysis

Analyzes every 10th annotated frame with rate limiting to stay within API quotas.
Saves progress incrementally to allow resumption.

Estimated time: ~15-20 minutes for ~152 frames
Rate limit: 10 req/min (6 second delay between requests)

Usage:
    export GOOGLE_API_KEY="your-key"
    python test_sampled_full_analysis.py --video VID01
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

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Set GOOGLE_API_KEY environment variable")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from agent import SurgAgent

DATA_PATH = Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Testing"

INSTRUMENT_NAMES = {0: "grasper", 1: "bipolar", 2: "hook", 3: "scissors", 4: "clipper", 5: "irrigator", 6: "specimen-bag"}
PHASE_NAMES = {0: "preparation", 1: "calot_triangle_dissection", 2: "clipping_and_cutting", 
               3: "gallbladder_dissection", 4: "gallbladder_packaging", 5: "cleaning_and_coagulation", 6: "gallbladder_retraction"}

def load_annotations(json_path: Path) -> Dict:
    with open(json_path) as f:
        return json.load(f)

def get_gt_for_frame(annotations: Dict, frame_num: int) -> Dict:
    frames_data = annotations.get("annotations", {})
    frame_annotations = frames_data.get(str(frame_num), [])
    
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
        if ann.get("occluded"): challenges.append("occlusion")
        if ann.get("bleeding"): challenges.append("blood")
        if ann.get("smoke"): challenges.append("smoke")
    
    return {"frame": frame_num, "instruments": list(set(instruments)), 
            "instrument_count": len(frame_annotations), "phase": phase, "challenges": list(set(challenges))}

def extract_frame(video_path: Path, frame_idx: int, output_path: Path, fps: int = 25) -> bool:
    timestamp = frame_idx / fps
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path), "-vframes", "1", "-q:v", "2", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()

def run_sampled_analysis(video_name: str = "VID01", sample_rate: int = 10):
    """Run sampled analysis with rate limiting."""
    
    print("=" * 70)
    print("🔬 SURGAGENT SAMPLED FULL VIDEO ANALYSIS")
    print("   Rate-Limited for API Quota Compliance")
    print("=" * 70)
    
    # Setup paths
    video_dir = DATA_PATH / video_name
    video_path = video_dir / f"{video_name}.mp4"
    json_path = video_dir / f"{video_name}.json"
    output_path = Path(__file__).parent.parent / f"sampled_analysis_{video_name}.json"
    
    if not video_path.exists() or not json_path.exists():
        print(f"❌ Video or annotations not found")
        return None
    
    # Load annotations
    annotations = load_annotations(json_path)
    frames_data = annotations.get("annotations", {})
    all_frames = sorted([int(f) for f in frames_data.keys()])
    
    # Sample every Nth frame
    sampled_frames = all_frames[::sample_rate]
    total_frames = len(sampled_frames)
    
    print(f"\n📹 Video: {video_name}")
    print(f"📊 Total annotated frames: {len(all_frames)}")
    print(f"📊 Sampled frames (1/{sample_rate}): {total_frames}")
    print(f"⏱️ Estimated time: {total_frames * 7 / 60:.1f} minutes")
    print(f"💰 Estimated cost: ~${total_frames * 0.0003:.2f}")
    
    # Check for existing progress
    existing_results = []
    completed_frames = set()
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
            existing_results = data.get("comparisons", [])
            completed_frames = {r["frame"] for r in existing_results}
        print(f"\n📂 Resuming from {len(completed_frames)} completed frames")
    
    remaining_frames = [f for f in sampled_frames if f not in completed_frames]
    print(f"📊 Remaining frames: {len(remaining_frames)}")
    
    if not remaining_frames:
        print("✅ All frames already analyzed!")
        return None
    
    # Initialize agent
    print("\n🤖 Initializing SurgAgent...")
    agent = SurgAgent(api_key=API_KEY)
    
    results = list(existing_results)
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("🔍 ANALYZING FRAMES (with 6s delay for rate limiting)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, frame_idx in enumerate(remaining_frames):
            frame_path = temp_path / f"frame_{frame_idx:06d}.png"
            
            # Extract frame
            if not extract_frame(video_path, frame_idx, frame_path):
                print(f"   ⚠️ Failed to extract frame {frame_idx}")
                continue
            
            # Get ground truth
            gt = get_gt_for_frame(annotations, frame_idx)
            
            # Gemini analysis
            api_start = time.time()
            gemini_analysis = agent.analyze_scene(str(frame_path))
            api_time = time.time() - api_start
            
            # Compare
            inst_match = abs(gt.get("instrument_count", 0) - gemini_analysis.get("instrument_count", 0)) <= 1
            
            comparison = {
                "frame": frame_idx,
                "timestamp_s": frame_idx / 25,
                "ground_truth": gt,
                "gemini_analysis": gemini_analysis,
                "processing_time": api_time,
                "inst_match": inst_match
            }
            results.append(comparison)
            
            # Progress
            total_done = len(results)
            progress_pct = 100 * total_done / total_frames
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (len(remaining_frames) - i - 1) if i > 0 else 0
            
            match_icon = "✅" if inst_match else "❌"
            print(f"   [{total_done}/{total_frames}] Frame {frame_idx}: "
                  f"GT={gt.get('instrument_count',0)}, Gemini={gemini_analysis.get('instrument_count','?')} {match_icon} "
                  f"({api_time:.1f}s) | ETA: {remaining/60:.1f}min")
            
            # Save progress after each batch of 10
            if total_done % 10 == 0:
                save_results(output_path, results, annotations, sampled_frames)
            
            # Rate limiting: wait 6 seconds between requests
            if i < len(remaining_frames) - 1:
                time.sleep(6)
    
    # Final save
    save_results(output_path, results, annotations, sampled_frames)
    
    # Summary
    print_summary(results, total_frames, time.time() - start_time)
    
    return results

def save_results(output_path: Path, results: List, annotations: Dict, sampled_frames: List):
    """Save results incrementally."""
    inst_matches = sum(1 for r in results if r.get("inst_match", False))
    
    output = {
        "video": "VID01",
        "total_sampled_frames": len(sampled_frames),
        "analyzed_frames": len(results),
        "instrument_accuracy": f"{100*inst_matches/len(results):.1f}%" if results else "0%",
        "comparisons": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

def print_summary(results: List, total_frames: int, elapsed: float):
    """Print analysis summary."""
    print("\n" + "=" * 70)
    print("📊 SAMPLED ANALYSIS SUMMARY")
    print("=" * 70)
    
    inst_matches = sum(1 for r in results if r.get("inst_match", False))
    
    print(f"\n   Frames analyzed: {len(results)}/{total_frames}")
    print(f"   Instrument accuracy (±1): {100*inst_matches/len(results):.1f}%")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Avg per frame: {elapsed/len(results):.1f}s")
    
    # Phase distribution
    phases = Counter()
    challenges = Counter()
    for r in results:
        phase = r["gemini_analysis"].get("estimated_phase", "unknown")
        phases[phase] += 1
        for c in r["gemini_analysis"].get("scene_challenges", []):
            challenges[c] += 1
    
    print(f"\n   Phases detected:")
    for phase, count in phases.most_common():
        print(f"      {phase}: {count} ({100*count/len(results):.0f}%)")
    
    print(f"\n   Challenges detected:")
    for challenge, count in challenges.most_common():
        print(f"      {challenge}: {count} ({100*count/len(results):.0f}%)")
    
    print(f"\n📁 Results saved incrementally")
    print("\n✅ Sampled analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="VID01")
    parser.add_argument("--sample-rate", type=int, default=10, help="Sample every Nth frame")
    args = parser.parse_args()
    
    run_sampled_analysis(args.video, args.sample_rate)
