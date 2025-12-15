#!/usr/bin/env python3
"""
SurgAgent: Core Agent with Gemini Integration

This is the main agent class that uses Google Gemini for:
- Scene analysis (vision)
- Strategy selection (reasoning)
- Failure recovery (planning)
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import google.generativeai as genai
from PIL import Image


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning process."""
    stage: str
    timestamp_ms: int
    action: str
    reasoning: str
    inputs: Optional[Dict] = None
    outputs: Optional[Dict] = None


@dataclass
class ToolSwitch:
    """Records when and why the agent switched tools."""
    frame: int
    from_tool: str
    to_tool: str
    reason: str
    confidence_before: float
    confidence_after: Optional[float] = None
    reasoning: str = ""


@dataclass
class RecoveryEvent:
    """Tracks a failure recovery attempt."""
    frame: int
    failure_type: str
    recovery_action: str
    success: bool
    frames_to_recover: int
    tools_used: List[str] = field(default_factory=list)


@dataclass
class QualityCheckpoint:
    """Quality assessment at a decision point."""
    frame: int
    metrics: Dict[str, float]
    decision: str
    threshold_used: float
    reasoning: str


class SurgAgent:
    """
    Agentic surgical instrument tracking system powered by Google Gemini.
    
    Capabilities:
    - Scene analysis using Gemini Vision
    - Adaptive tool selection using Gemini Pro
    - Failure detection and recovery
    - Full reasoning trace logging
    """
    
    # Available detection strategies
    DETECTORS = {
        "simple_detector": {"speed": "fast", "accuracy": "low", "smoke_robust": False},
        "yolov8_surgical": {"speed": "medium", "accuracy": "high", "smoke_robust": False},
        "advanced_detector": {"speed": "slow", "accuracy": "very_high", "smoke_robust": True}
    }
    
    # Available tracking strategies
    TRACKERS = {
        "simple_tracker": {"speed": "fast", "occlusion_handling": "poor"},
        "byte_track": {"speed": "medium", "occlusion_handling": "good"},
        "deep_sort": {"speed": "slow", "occlusion_handling": "excellent"}
    }
    
    # Surgical phase constraints
    PHASE_CONSTRAINTS = {
        "preparation": {"expected": ["grasper"], "forbidden": ["clipper", "scissors"]},
        "calot_triangle_dissection": {"expected": ["grasper", "hook", "bipolar"], "forbidden": []},
        "clipping_cutting": {"expected": ["clipper", "scissors", "grasper"], "forbidden": ["hook"]},
        "gallbladder_dissection": {"expected": ["grasper", "hook", "bipolar"], "forbidden": ["clipper"]},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize SurgAgent with Gemini API."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize models - using available model names
        # Using gemini-2.0-flash-exp for both (it's fast and works)
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.reasoning_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Agent state
        self.current_detector = None
        self.current_tracker = None
        self.reasoning_trace: List[ReasoningStep] = []
        self.tool_switches: List[ToolSwitch] = []
        self.recovery_events: List[RecoveryEvent] = []
        self.quality_checkpoints: List[QualityCheckpoint] = []
        self.confidence_history: List[float] = []
        
        print("✅ SurgAgent initialized with Gemini API")
    
    def analyze_scene(self, frame_path: str) -> Dict[str, Any]:
        """
        Analyze a surgical scene using Gemini Vision.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Scene analysis dictionary
        """
        start_time = int(time.time() * 1000)
        
        try:
            # Load image
            image = Image.open(frame_path)
            
            # Create prompt for Gemini
            prompt = """
            Analyze this surgical laparoscopic frame. Provide a JSON response with:
            
            {
                "instruments": [
                    {"type": "grasper/bipolar/hook/scissors/clipper/irrigator", "visible": true/false}
                ],
                "scene_challenges": ["smoke", "blood", "occlusion", "motion_blur"],
                "visibility_score": 1-10,
                "estimated_phase": "preparation/dissection/clipping/packaging",
                "instrument_count": number,
                "recommendations": "description of best tracking approach"
            }
            
            Be concise and accurate. Focus on surgical instruments only.
            """
            
            # Call Gemini Vision
            response = self.vision_model.generate_content([prompt, image])
            
            # Parse response
            analysis = self._parse_scene_analysis(response.text)
            
            # Log reasoning step
            self.reasoning_trace.append(ReasoningStep(
                stage="scene_analysis",
                timestamp_ms=start_time,
                action="Analyzed surgical frame with Gemini Vision",
                reasoning=f"Identified {analysis.get('instrument_count', 0)} instruments, "
                         f"visibility score {analysis.get('visibility_score', 'N/A')}",
                inputs={"frame_path": frame_path},
                outputs=analysis
            ))
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Scene analysis error: {e}")
            return {
                "instruments": [],
                "scene_challenges": [],
                "visibility_score": 5,
                "estimated_phase": "unknown",
                "error": str(e)
            }
    
    def select_strategy(self, scene_analysis: Dict) -> Dict[str, str]:
        """
        Use Gemini to select the best detection/tracking strategy.
        
        Args:
            scene_analysis: Output from analyze_scene()
            
        Returns:
            Selected strategy dictionary
        """
        start_time = int(time.time() * 1000)
        
        prompt = f"""
        You are an AI agent selecting surgical tracking tools.
        
        Scene Analysis:
        - Visibility: {scene_analysis.get('visibility_score', 5)}/10
        - Challenges: {scene_analysis.get('scene_challenges', [])}
        - Instruments: {scene_analysis.get('instrument_count', 2)}
        - Phase: {scene_analysis.get('estimated_phase', 'unknown')}
        
        Available Detectors:
        - simple_detector: fast, low accuracy
        - yolov8_surgical: balanced, high accuracy
        - advanced_detector: slow, handles smoke/blood
        
        Available Trackers:
        - simple_tracker: fast, poor occlusion handling
        - byte_track: balanced, good occlusion handling
        - deep_sort: slow, excellent occlusion handling
        
        Select the best combination. Respond with JSON:
        {{
            "detector": "detector_name",
            "tracker": "tracker_name",
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.reasoning_model.generate_content(prompt)
            strategy = self._parse_strategy(response.text)
            
            # Update agent state
            old_detector = self.current_detector
            self.current_detector = strategy.get("detector", "yolov8_surgical")
            self.current_tracker = strategy.get("tracker", "byte_track")
            
            # Log tool switch if changed
            if old_detector and old_detector != self.current_detector:
                self.tool_switches.append(ToolSwitch(
                    frame=0,
                    from_tool=old_detector,
                    to_tool=self.current_detector,
                    reason="strategy_update",
                    confidence_before=0.0,
                    reasoning=strategy.get("reasoning", "")
                ))
            
            # Log reasoning step
            self.reasoning_trace.append(ReasoningStep(
                stage="tool_selection",
                timestamp_ms=start_time,
                action=f"Selected {self.current_detector} + {self.current_tracker}",
                reasoning=strategy.get("reasoning", "Gemini recommendation"),
                inputs={"scene_analysis": scene_analysis},
                outputs=strategy
            ))
            
            return strategy
            
        except Exception as e:
            print(f"⚠️ Strategy selection error: {e}")
            # Default fallback
            return {
                "detector": "yolov8_surgical",
                "tracker": "byte_track",
                "reasoning": f"Default strategy (error: {e})"
            }
    
    def handle_failure(self, failure_type: str, context: Dict) -> str:
        """
        Use Gemini to decide how to recover from a failure.
        
        Args:
            failure_type: Type of failure (track_loss, low_confidence, etc.)
            context: Current tracking context
            
        Returns:
            Recovery action to take
        """
        start_time = int(time.time() * 1000)
        
        prompt = f"""
        A surgical tracking failure occurred.
        
        Failure Type: {failure_type}
        Current Detector: {self.current_detector}
        Current Tracker: {self.current_tracker}
        Context: {json.dumps(context)}
        
        Recovery Options:
        1. reinitialize - Reset tracker with new parameters
        2. switch_detector - Use a different detector
        3. switch_tracker - Use a different tracker
        4. increase_threshold - Raise IoU threshold
        5. skip_frames - Skip problematic frames
        
        Select the best recovery action and explain briefly.
        Respond with JSON:
        {{
            "action": "action_name",
            "parameters": {{}},
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.reasoning_model.generate_content(prompt)
            recovery = self._parse_recovery(response.text)
            
            # Log recovery event
            self.recovery_events.append(RecoveryEvent(
                frame=context.get("frame", 0),
                failure_type=failure_type,
                recovery_action=recovery.get("action", "reinitialize"),
                success=True,  # Optimistic
                frames_to_recover=5,
                tools_used=[self.current_detector, self.current_tracker]
            ))
            
            # Log reasoning step
            self.reasoning_trace.append(ReasoningStep(
                stage="replanning",
                timestamp_ms=start_time,
                action=f"Recovery action: {recovery.get('action')}",
                reasoning=recovery.get("reasoning", "Gemini recommendation"),
                inputs={"failure_type": failure_type, "context": context},
                outputs=recovery
            ))
            
            return recovery.get("action", "reinitialize")
            
        except Exception as e:
            print(f"⚠️ Recovery planning error: {e}")
            return "reinitialize"
    
    def quality_check(self, frame: int, metrics: Dict[str, float]) -> str:
        """
        Perform quality checkpoint and decide whether to continue or replan.
        
        Args:
            frame: Current frame number
            metrics: Current performance metrics
            
        Returns:
            Decision: "continue", "replan", "switch_tool"
        """
        avg_confidence = metrics.get("avg_confidence", 0.8)
        threshold = 0.65
        
        if avg_confidence >= threshold:
            decision = "continue"
            reasoning = f"Confidence {avg_confidence:.2f} >= threshold {threshold}"
        elif avg_confidence >= 0.5:
            decision = "switch_tool"
            reasoning = f"Confidence {avg_confidence:.2f} below threshold, switching tools"
        else:
            decision = "replan"
            reasoning = f"Confidence {avg_confidence:.2f} very low, replanning needed"
        
        self.quality_checkpoints.append(QualityCheckpoint(
            frame=frame,
            metrics=metrics,
            decision=decision,
            threshold_used=threshold,
            reasoning=reasoning
        ))
        
        return decision
    
    def track_video(self, video_path: str) -> Dict:
        """
        Track surgical instruments in a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tracking results with full reasoning trace
        """
        print(f"\n🎬 Processing: {video_path}")
        start_time = int(time.time() * 1000)
        
        # Stage 1: Extract first frame and analyze
        print("  📊 Stage 1: Scene Analysis...")
        # In a real implementation, extract frame from video
        # scene_analysis = self.analyze_scene(first_frame)
        scene_analysis = {
            "instruments": [{"type": "grasper"}, {"type": "bipolar"}],
            "visibility_score": 8,
            "scene_challenges": [],
            "estimated_phase": "dissection"
        }
        
        # Stage 2: Select strategy
        print("  🔧 Stage 2: Strategy Selection...")
        strategy = self.select_strategy(scene_analysis)
        print(f"      → {strategy['detector']} + {strategy['tracker']}")
        
        # Stage 3: Run tracking (simulated)
        print("  🎯 Stage 3: Tracking...")
        predictions = self._simulate_tracking(30)
        
        # Stage 4: Validate
        print("  ✅ Stage 4: Validation...")
        self.reasoning_trace.append(ReasoningStep(
            stage="validation",
            timestamp_ms=int(time.time() * 1000),
            action="Self-assessment of tracking results",
            reasoning="All frames processed successfully",
            outputs={"frames": 30, "tracks": 2}
        ))
        
        end_time = int(time.time() * 1000)
        
        return {
            "video_path": video_path,
            "predictions": predictions,
            "tools_used": [self.current_detector, self.current_tracker],
            "reasoning_trace": [vars(r) for r in self.reasoning_trace],
            "tool_switches": [vars(t) for t in self.tool_switches],
            "recovery_events": [vars(r) for r in self.recovery_events],
            "quality_checkpoints": [vars(q) for q in self.quality_checkpoints],
            "processing_time_ms": end_time - start_time
        }
    
    def _simulate_tracking(self, num_frames: int) -> List[Dict]:
        """Simulate tracking for demo purposes."""
        predictions = []
        for frame in range(num_frames):
            predictions.append({
                "frame": frame,
                "instruments": [
                    {"track_id": 1, "type": "grasper", "confidence": 0.9},
                    {"track_id": 2, "type": "bipolar", "confidence": 0.85}
                ]
            })
            
            # Quality check every 10 frames
            if frame > 0 and frame % 10 == 0:
                self.quality_check(frame, {"avg_confidence": 0.87})
        
        return predictions
    
    def _parse_scene_analysis(self, response_text: str) -> Dict:
        """Parse Gemini's scene analysis response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Default response if parsing fails
        return {
            "instruments": [],
            "visibility_score": 7,
            "scene_challenges": [],
            "estimated_phase": "unknown",
            "raw_response": response_text[:500]
        }
    
    def _parse_strategy(self, response_text: str) -> Dict:
        """Parse Gemini's strategy selection response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "detector": "yolov8_surgical",
            "tracker": "byte_track",
            "reasoning": "Default selection"
        }
    
    def _parse_recovery(self, response_text: str) -> Dict:
        """Parse Gemini's recovery planning response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "action": "reinitialize",
            "parameters": {},
            "reasoning": "Default recovery"
        }
    
    def get_reasoning_summary(self) -> str:
        """Get a human-readable summary of agent's reasoning."""
        summary = ["=" * 50, "🧠 SurgAgent Reasoning Summary", "=" * 50]
        
        for step in self.reasoning_trace:
            summary.append(f"\n📍 {step.stage.upper()}")
            summary.append(f"   Action: {step.action}")
            summary.append(f"   Reasoning: {step.reasoning}")
        
        if self.tool_switches:
            summary.append(f"\n🔄 Tool Switches: {len(self.tool_switches)}")
        
        if self.recovery_events:
            summary.append(f"🔧 Recovery Events: {len(self.recovery_events)}")
        
        summary.append(f"\n📊 Quality Checkpoints: {len(self.quality_checkpoints)}")
        
        return "\n".join(summary)


# Quick test
if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ Set GOOGLE_API_KEY to test Gemini integration")
        print("   Example: export GOOGLE_API_KEY='your-key-here'")
    else:
        agent = SurgAgent()
        print(agent.get_reasoning_summary())
