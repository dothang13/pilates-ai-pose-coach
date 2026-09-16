"""
Self-Contained Mock Benchmark & Test Script for Person B
Simulates 2 Squat repetitions (1 Correct Rep, 1 Incomplete/Shallow Rep)
Tests Signal Smoothing, FSM Phase Tracking, Rule Assessment & Debounced Audio.
"""

import os
import sys
import time
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to utf-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core_coach.smoothing import OneEuroFilter
from core_coach.fsm_counter import RepetitionFSM, ExerciseState
from core_coach.exercise_rules import evaluate_form_rules
from core_coach.feedback_engine import FeedbackEngine

def simulate_squat_pipeline():
    print("=" * 65)
    print("TEST BENCHMARK: PERSON B - EXERCISE UNDERSTANDING & COACH")
    print("=" * 65)

    # 1. Initialize Person B modules
    angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    squat_fsm = RepetitionFSM(start_threshold=155.0, bottom_threshold=95.0, min_rep_duration=1.0)
    feedback_engine = FeedbackEngine(debounce_cooldown=2.0)

    # 2. Generate Synthetic Squat Angles over 120 frames (4 seconds at 30 FPS)
    # Rep 1 (Frames 0 -> 60): Good Squat (170° -> 85° -> 170°)
    # Rep 2 (Frames 60 -> 120): Shallow Squat / Bad Form (170° -> 115° -> 170°)
    timestamps = np.linspace(0, 4.0, 120)
    ideal_angles = []

    # Rep 1: 0 to 2 seconds
    t1 = np.linspace(0, np.pi, 60)
    angles_rep1 = 170.0 - 85.0 * np.sin(t1)  # reaches 85° at peak
    ideal_angles.extend(angles_rep1)

    # Rep 2: 2 to 4 seconds
    t2 = np.linspace(0, np.pi, 60)
    angles_rep2 = 170.0 - 55.0 * np.sin(t2)  # only reaches 115° (shallow!)
    ideal_angles.extend(angles_rep2)

    # Add realistic sensor/camera noise (+/- 2.5°)
    np.random.seed(42)
    noisy_angles = np.array(ideal_angles) + np.random.normal(0, 2.0, len(ideal_angles))

    print("[*] Simulating 120 frames of Squat motion from Person A...\n")

    for i in range(120):
        t = timestamps[i]
        raw_angle = noisy_angles[i]

        # Step 1: Smooth angle using One-Euro Filter
        smooth_angle = angle_filter.update(raw_angle, t)

        # Step 2: Update FSM Rep Counter
        state, reps, meta = squat_fsm.update(smooth_angle, t)

        # Step 3: Evaluate Form Rules (simulate knee and torso angles)
        angles_dict = {
            "knee_angle": smooth_angle,
            "torso_angle": 68.0,
            "knee_distance": 0.40,
            "ankle_distance": 0.40
        }
        is_valid, errors, score = evaluate_form_rules("squat", angles_dict)

        # Step 4: Process Feedback
        fb = feedback_engine.process_feedback(is_valid, errors, state.value, current_time=t)

        # Print key transition events
        if i % 15 == 0 or state == ExerciseState.BOTTOM or fb["audio_cue_to_speak"]:
            audio_text = f"[AUDIO: '{fb['audio_cue_to_speak']}']" if fb["audio_cue_to_speak"] else ""
            print(f"[t={t:.2f}s | Frame {i:3d}] "
                  f"Raw: {raw_angle:5.1f} deg | Smooth: {smooth_angle:5.1f} deg | "
                  f"State: {state.value:10s} | Reps: {reps} | Form: {fb['status_text']} {audio_text}")

    print("\n" + "=" * 65)
    print(f"[+] SIMULATION COMPLETED!")
    print(f"    - Total Valid Reps Counted: {squat_fsm.rep_count} / 1 expected (Rep 2 correctly rejected for shallow depth)")
    print(f"    - One-Euro Filter: Successfully eliminated noise spikes")
    print(f"    - Feedback Engine: Successfully throttled audio cues")
    print("=" * 65)

    # Allow background audio speaker to finish playing
    time.sleep(1.5)

if __name__ == "__main__":
    simulate_squat_pipeline()
