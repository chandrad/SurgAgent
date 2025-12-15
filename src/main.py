#!/usr/bin/env python3
"""
SurgAgent: Main Entry Point

An agentic AI system for surgical instrument tracking powered by Google Gemini.

Usage:
    python main.py --mode demo
    python main.py --mode track --video path/to/video.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from agent import SurgAgent
from demo import run_demo


def main():
    parser = argparse.ArgumentParser(
        description="SurgAgent: AI-Powered Surgical Instrument Tracking"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "track", "analyze"],
        default="demo",
        help="Operating mode"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to surgical video (for track mode)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Demo scenario: all, basic, adaptive, failure_recovery"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google Gemini API key (or set GOOGLE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("❌ Error: Google API key required")
        print("   Set GOOGLE_API_KEY environment variable or use --api-key")
        print("   Get a key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    os.environ["GOOGLE_API_KEY"] = args.api_key
    
    if args.mode == "demo":
        print("\n🏥 SurgAgent Demo Mode")
        print("=" * 50)
        scenarios = [args.scenario] if args.scenario != "all" else None
        run_demo(scenarios)
        
    elif args.mode == "track":
        if not args.video:
            print("❌ Error: --video required for track mode")
            sys.exit(1)
            
        print(f"\n🎬 Tracking video: {args.video}")
        agent = SurgAgent(api_key=args.api_key)
        results = agent.track_video(args.video)
        print(f"✅ Tracking complete: {results}")
        
    elif args.mode == "analyze":
        print("\n🔍 Scene Analysis Mode")
        agent = SurgAgent(api_key=args.api_key)
        # Interactive analysis mode
        while True:
            frame_path = input("Enter frame path (or 'quit'): ")
            if frame_path.lower() == 'quit':
                break
            analysis = agent.analyze_scene(frame_path)
            print(f"\n📊 Analysis:\n{analysis}\n")


if __name__ == "__main__":
    main()
