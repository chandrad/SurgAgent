#!/usr/bin/env python3
"""
SurgAgent Real-Time Video Tracking Test

Tests the Gemini-powered agent on actual surgical video by:
1. Extracting frames from a test video
2. Analyzing key frames with Gemini Vision
3. Simulating tracking across frame sequence
4. Generating performance report

Cost-Optimized: Only analyzes 5 key frames to minimize API costs

Usage:
    export GOOGLE_API_KEY="your-key"
    python test_video_tracking.py --video VID01
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check API key
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Set GOOGLE_API_KEY environment variable")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from agent import SurgAgent

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / "surgagent-track/data/raw/cholectrack20/Testing"


def get_video_info(video_path: Path) -> Dict:
    """Get video metadata using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), {})
            return {
                "duration": float(info.get("format", {}).get("duration", 0)),
                "fps": eval(stream.get("r_frame_rate", "25/1")),
                "width": stream.get("width", 1920),
                "height": stream.get("height", 1080),
                "total_frames": int(float(info.get("format", {}).get("duration", 0)) * eval(stream.get("r_frame_rate", "25/1")))
            }
    except Exception as e:
        print(f"⚠️ Could not get video info: {e}")
    return {"duration": 0, "fps": 25, "width": 1920, "height": 1080, "total_frames": 0}


def extract_frames(video_path: Path, output_dir: Path, frame_indices: List[int], fps: int = 25) -> List[Path]:
    """Extract specific frames from video using ffmpeg."""
    import subprocess
    
    extracted = []
    for frame_idx in frame_indices:
        timestamp = frame_idx / fps
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp), 
            "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            extracted.append(output_path)
        else:
            print(f"⚠️ Failed to extract frame {frame_idx}")
    
    return extracted


def load_annotations(json_path: Path) -> Dict:
    """Load CholecTrack20 annotations."""
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def run_video_tracking_test(video_name: str = "VID01", num_frames: int = 5):
    """Run end-to-end video tracking test."""
    
    print("=" * 70)
    print("🎬 SURGAGENT REAL-TIME VIDEO TRACKING TEST")
    print("   Powered by Google Gemini 2.0 Flash")
    print("=" * 70)
    
    # Find video files
    video_dir = DATA_PATH / video_name
    video_path = video_dir / f"{video_name}.mp4"
    json_path = video_dir / f"{video_name}.json"
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        available = [d.name for d in DATA_PATH.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"📁 Available videos: {available}")
        return None
    
    print(f"\n📹 Video: {video_name}")
    print(f"   Path: {video_path}")
    print(f"   Size: {video_path.stat().st_size / (1024**3):.2f} GB")
    
    # Get video info
    print("\n🎥 Analyzing video metadata...")
    video_info = get_video_info(video_path)
    print(f"   Duration: {video_info['duration']:.1f}s")
    print(f"   FPS: {video_info['fps']}")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   Total Frames: {video_info['total_frames']}")
    
    # Load annotations
    print("\n📝 Loading ground truth annotations...")
    annotations = load_annotations(json_path)
    if annotations:
        print(f"   Loaded annotations for {len(annotations.get('frames', []))} frames")
    else:
        print("   ⚠️ No annotations found")
    
    # Initialize agent
    print("\n🤖 Initializing SurgAgent with Gemini...")
    agent = SurgAgent(api_key=API_KEY)
    
    # Select frames to analyze (evenly distributed)
    total_frames = video_info['total_frames'] or 7500  # Default to 5 minutes at 25fps
    frame_step = max(1, total_frames // num_frames)
    target_frames = [i * frame_step for i in range(num_frames)]
    
    print(f"\n📸 Extracting {num_frames} key frames for analysis...")
    print(f"   Frame indices: {target_frames}")
    
    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract frames
        extracted_frames = extract_frames(
            video_path, temp_path, target_frames, 
            fps=int(video_info['fps'])
        )
        
        if not extracted_frames:
            print("❌ Failed to extract frames. Ensure ffmpeg is installed.")
            return None
        
        print(f"   ✅ Extracted {len(extracted_frames)} frames")
        
        # Analyze each frame with Gemini
        print("\n" + "=" * 70)
        print("🔍 FRAME-BY-FRAME ANALYSIS WITH GEMINI VISION")
        print("=" * 70)
        
        results = []
        for i, frame_path in enumerate(extracted_frames):
            frame_idx = target_frames[i]
            print(f"\n🖼️ Frame {frame_idx} ({i+1}/{len(extracted_frames)})")
            
            start = time.time()
            analysis = agent.analyze_scene(str(frame_path))
            elapsed = time.time() - start
            
            result = {
                "frame_index": frame_idx,
                "timestamp_s": frame_idx / video_info['fps'],
                "analysis": analysis,
                "processing_time_s": round(elapsed, 2)
            }
            results.append(result)
            
            # Display
            print(f"   ⏱️ Time: {elapsed:.2f}s")
            print(f"   ⏰ Video timestamp: {result['timestamp_s']:.1f}s")
            print(f"   👁️ Visibility: {analysis.get('visibility_score', 'N/A')}/10")
            print(f"   🔧 Instruments: {analysis.get('instrument_count', 'N/A')}")
            print(f"   ⚠️ Challenges: {analysis.get('scene_challenges', [])}")
            print(f"   📍 Phase: {analysis.get('estimated_phase', 'N/A')}")
        
        # Strategy recommendation based on final frame
        print("\n" + "=" * 70)
        print("🎯 TRACKING STRATEGY RECOMMENDATION")
        print("=" * 70)
        
        last_analysis = results[-1]["analysis"]
        strategy = agent.select_strategy(last_analysis)
        
        print(f"\n   Detector: {strategy.get('detector', 'N/A')}")
        print(f"   Tracker: {strategy.get('tracker', 'N/A')}")
        print(f"   Reasoning: {strategy.get('reasoning', 'N/A')[:300]}...")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 VIDEO TRACKING TEST SUMMARY")
    print("=" * 70)
    
    avg_visibility = sum(r["analysis"].get("visibility_score", 5) for r in results) / len(results)
    avg_instruments = sum(r["analysis"].get("instrument_count", 0) for r in results) / len(results)
    total_time = sum(r["processing_time_s"] for r in results)
    
    phases = [r["analysis"].get("estimated_phase", "unknown") for r in results]
    challenges = []
    for r in results:
        challenges.extend(r["analysis"].get("scene_challenges", []))
    
    print(f"\n📹 Video Analyzed: {video_name}")
    print(f"   Duration: {video_info['duration']:.1f}s")
    print(f"   Frames Analyzed: {len(results)}")
    print(f"   Total API Time: {total_time:.1f}s")
    
    print(f"\n📈 Scene Statistics:")
    print(f"   Avg Visibility: {avg_visibility:.1f}/10")
    print(f"   Avg Instruments: {avg_instruments:.1f}")
    print(f"   Phases Detected: {list(set(phases))}")
    print(f"   Challenges Found: {list(set(challenges))}")
    
    print(f"\n🎯 Recommended Strategy:")
    print(f"   {strategy.get('detector', 'N/A')} + {strategy.get('tracker', 'N/A')}")
    
    print(f"\n🧠 Reasoning Trace:")
    print(agent.get_reasoning_summary())
    
    # Estimate real-time performance
    fps_actual = len(results) / total_time if total_time > 0 else 0
    print(f"\n⚡ Performance Estimate:")
    print(f"   Analysis FPS: {fps_actual:.2f} (Gemini API)")
    print(f"   For real-time (25fps): Would need local inference + Gemini for decisions")
    
    # Save results
    output = {
        "video": video_name,
        "video_info": video_info,
        "frames_analyzed": len(results),
        "frame_results": results,
        "summary": {
            "avg_visibility": avg_visibility,
            "avg_instruments": avg_instruments,
            "phases": list(set(phases)),
            "challenges": list(set(challenges)),
        },
        "strategy": strategy,
        "total_api_time_s": total_time,
        "total_api_calls": len(results) + 1,  # frames + strategy
        "reasoning_trace": [vars(r) for r in agent.reasoning_trace],
    }
    
    output_path = Path(__file__).parent.parent / f"video_tracking_results_{video_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("\n✅ Video tracking test complete!")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SurgAgent Video Tracking Test")
    parser.add_argument("--video", type=str, default="VID01", help="Video name (e.g., VID01)")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    args = parser.parse_args()
    
    run_video_tracking_test(args.video, args.frames)
