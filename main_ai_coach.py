"""
Pilates & Fitness AI Pose Coach - Complete Realtime Application
Integrates Person A (RTMPose 2D/3D Vision) + Person B (FSM Rep Counter, Biomechanics, ML Error Classifier & Studio Voice)

Usage:
  python main_ai_coach.py --exercise squat --source 0
  python main_ai_coach.py --exercise squat --source data/raw/squat_video.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Reconfigure stdout to utf-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core_coach.smoothing import OneEuroFilter
from core_coach.fsm_counter import RepetitionFSM, HoldTimer, ExerciseState
from core_coach.exercise_rules import EXERCISE_CONFIGS, evaluate_form_rules
from core_coach.feedback_engine import FeedbackEngine
from core_coach.error_classifier import PostureErrorClassifier

# 17 COCO Keypoints index
KEYPOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Knee",
    "L_Ankle", "R_Hip", "R_Knee", "R_Ankle"
]

def calculate_angle(p1, p2, p3):
    """Calculates interior angle (degrees) between p1 - p2 - p3 with p2 as vertex."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 180.0
    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def select_primary_user(keypoints, scores, frame_w, frame_h, kpt_thr=0.3):
    """
    Intelligently selects the primary workout user when multiple people are detected.
    Prioritizes:
      1. Bounding box area (user closest to camera is largest).
      2. Centrality (distance from horizontal center of the frame).
    Returns (primary_kpts, primary_scores, person_idx)
    """
    if keypoints is None or len(keypoints) == 0:
        return None, None, -1
    if len(keypoints) == 1:
        return keypoints[0], scores[0], 0

    best_idx = 0
    best_rank = -1.0
    frame_cx = frame_w / 2.0

    for i in range(len(keypoints)):
        kpts = keypoints[i]
        scs = scores[i]
        valid_mask = scs > kpt_thr
        if np.sum(valid_mask) < 4:
            continue

        valid_kpts = kpts[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)

        box_area = float((max_x - min_x) * (max_y - min_y))
        user_cx = float((min_x + max_x) / 2.0)
        dist_to_center = abs(user_cx - frame_cx) / (frame_cx + 1e-5)

        # Centrality weight: 1.0 at dead center, 0.4 at edge
        centrality = max(0.4, 1.0 - 0.6 * dist_to_center)
        rank_score = box_area * centrality

        if rank_score > best_rank:
            best_rank = rank_score
            best_idx = i

    return keypoints[best_idx], scores[best_idx], best_idx

def draw_hud(frame, exercise_name, state_text, reps_or_time, score, is_valid, error_msgs, fps, current_angle=None, num_persons=1):
    """
    Renders professional translucent HUD overlays, rep count, depth bar, and error alerts.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top Status Banner (Translucent Dark Slate)
    cv2.rectangle(overlay, (15, 15), (420, 145), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border color based on posture validity
    border_color = (0, 230, 115) if is_valid else (0, 60, 255)
    cv2.rectangle(frame, (15, 15), (420, 145), border_color, 2)

    # Exercise & Phase Title
    cv2.putText(frame, f"{exercise_name.upper()} | {state_text}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Primary Metric (Rep Count / Time)
    metric_color = (0, 255, 127) if is_valid else (0, 200, 255)
    cv2.putText(frame, f"{reps_or_time}", (30, 95),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, metric_color, 3)

    # Form Score, FPS & Multi-person status
    user_str = f" | Nguoi: 1/{num_persons}" if num_persons > 1 else ""
    cv2.putText(frame, f"Form Score: {score}% | FPS: {fps:.1f}{user_str}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # 2. Side Depth Progress Bar (for Squat depth visual feedback)
    if current_angle is not None:
        bar_x = w - 60
        bar_y_top = 100
        bar_height = 280
        bar_y_bottom = bar_y_top + bar_height

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 30, bar_y_bottom), (30, 30, 30), -1)
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 30, bar_y_bottom), (150, 150, 150), 1)

        # Progress calculation (from 160° standing down to 80° deep squat)
        progress = np.clip((160.0 - current_angle) / (160.0 - 80.0), 0.0, 1.0)
        fill_h = int(progress * bar_height)
        fill_color = (0, 255, 127) if current_angle <= 95.0 else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, bar_y_bottom - fill_h), (bar_x + 30, bar_y_bottom), fill_color, -1)

        # 90° target line indicator
        target_y = bar_y_bottom - int(((160.0 - 90.0) / (160.0 - 80.0)) * bar_height)
        cv2.line(frame, (bar_x - 10, target_y), (bar_x + 40, target_y), (0, 255, 255), 2)
        cv2.putText(frame, "90 deg", (bar_x - 70, target_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, f"{int(current_angle)} deg", (bar_x - 65, bar_y_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 3. Bottom Hotkey Navigation Bar
    cv2.rectangle(frame, (0, h - 26), (w, h), (15, 15, 20), -1)
    hotkey_guide = "[1]Squat  [2]Lunge  [3]Bridge  [4]Plank  [5]Hundred  [6]BirdDog  [7]LegRaise  |  [Q]Thoat"
    cv2.putText(frame, hotkey_guide, (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 255), 1)

    # 4. Floating Error Banner (if error detected, placed above the hotkey bar)
    if not is_valid and error_msgs:
        err_text = error_msgs[0]
        banner_w = int(len(err_text) * 16) + 40
        start_x = max(20, (w - banner_w) // 2)
        cv2.rectangle(frame, (start_x, h - 75), (start_x + banner_w, h - 33), (0, 0, 180), -1)
        cv2.rectangle(frame, (start_x, h - 75), (start_x + banner_w, h - 33), (0, 0, 255), 2)
        cv2.putText(frame, f"[!] {err_text}", (start_x + 20, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame

def run_ai_coach(source=0, exercise="squat", mode="balanced", device="cpu", kpt_thr=0.4):
    print("=" * 65)
    print("PILATES & FITNESS AI POSE COACH - LIVE COACHING PIPELINE")
    print("=" * 65)
    print(f"[*] Selected Exercise : {exercise.upper()}")
    print(f"[*] Video Source       : {source}")
    print(f"[*] Pose Model Engine  : RTMPose-{mode[0].upper()} (SimCC)")
    print(f"[*] Execution Device   : {device.upper()}")
    print("=" * 65)

    # 1. Initialize Person A: Vision Pose Engine
    body_estimator = Body(mode=mode, backend='onnxruntime', device=device)

    # 2. Initialize Person B: Coaching Modules
    angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    rep_fsm = RepetitionFSM(start_threshold=155.0, bottom_threshold=95.0, min_rep_duration=1.0)
    hold_timer = HoldTimer()
    feedback_engine = FeedbackEngine(debounce_cooldown=2.5, enable_voice=True)
    ai_classifier = PostureErrorClassifier()

    # Welcome voice greeting
    feedback_engine.speaker.speak("Chào bạn, hệ thống AI Huấn luyện viên đã sẵn sàng!")

    # Video Capture
    video_src = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {source}")
        return

    if isinstance(video_src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_history = []
    prev_time = time.time()

    print("[+] Camera stream started! Press 'q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[*] Video stream ended.")
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        # -------------------------------------------------------------
        # 👤 PERSON A: 2D Pose Estimation
        # -------------------------------------------------------------
        keypoints, scores = body_estimator(frame)
        num_persons = len(keypoints) if keypoints is not None else 0

        # Draw skeleton
        vis_frame = draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr)

        # -------------------------------------------------------------
        # 👤 PERSON B: Kinematic Analysis & AI Coaching
        # -------------------------------------------------------------
        current_angle = None
        state_text = "STANDBY"
        metric_display = "REPS: 0"
        is_form_valid = True
        error_msgs = []
        form_score = 100

        if num_persons > 0:
            h_f, w_f, _ = frame.shape
            user_kpts, user_scores, active_idx = select_primary_user(keypoints, scores, w_f, h_f, kpt_thr=kpt_thr)

            if user_kpts is None:
                continue

            # If multiple persons detected, draw lock tag above primary user's head
            if num_persons > 1 and user_scores[0] > kpt_thr:
                nose_x, nose_y = int(user_kpts[0][0]), int(user_kpts[0][1])
                cv2.putText(vis_frame, "NGUOI CHINH [LOCKED]", (max(10, nose_x - 75), max(25, nose_y - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2)
                cv2.circle(vis_frame, (nose_x, nose_y), 6, (0, 255, 127), -1)

            # Check if lower body (Hips & Knees) is actually visible in camera frame
            has_hips = (user_scores[11] > kpt_thr or user_scores[12] > kpt_thr)
            has_knees = (user_scores[13] > kpt_thr or user_scores[14] > kpt_thr)
            is_full_body_visible = has_hips and has_knees

            if not is_full_body_visible:
                state_text = "LUI RA XA (2M)"
                metric_display = "CHO TOAN THAN"
                is_form_valid = True
                error_msgs = ["Vui lòng lùi ra xa 2 mét để camera thấy từ đầu đến chân"]
                form_score = 100
            else:
                # Use most confident leg
                l_conf = (user_scores[11] + user_scores[13] + user_scores[15]) / 3.0
                r_conf = (user_scores[12] + user_scores[14] + user_scores[16]) / 3.0

                if l_conf > r_conf and l_conf > kpt_thr:
                    raw_knee = calculate_angle(user_kpts[11], user_kpts[13], user_kpts[15])
                    raw_hip = calculate_angle(user_kpts[5], user_kpts[11], user_kpts[13])
                elif r_conf > kpt_thr:
                    raw_knee = calculate_angle(user_kpts[12], user_kpts[14], user_kpts[16])
                    raw_hip = calculate_angle(user_kpts[6], user_kpts[12], user_kpts[14])
                else:
                    raw_knee, raw_hip = 180.0, 180.0

                # Step 1: Smooth knee angle
                smooth_knee = angle_filter.update(raw_knee, curr_time)
                current_angle = smooth_knee

                # Step 2: FSM Rep Counter / Timer
                ex_cfg = EXERCISE_CONFIGS.get(exercise.lower(), {})
                if ex_cfg.get("type") == "REPETITION":
                    state, reps, meta = rep_fsm.update(smooth_knee, curr_time)
                    state_text = state.value
                    metric_display = f"REPS: {reps}"
                else:
                    # Isometric hold (e.g. Plank)
                    hold_sec = hold_timer.update(is_form_valid, curr_time)
                    state_text = "HOLDING" if hold_timer.is_holding else "PAUSED"
                    metric_display = f"TIME: {int(hold_sec)}s"

                # Step 3: Biomechanical Form Evaluation (Phase-Aware)
                shoulder_mid = (user_kpts[5] + user_kpts[6]) / 2.0
                hip_mid = (user_kpts[11] + user_kpts[12]) / 2.0
                torso_dx = shoulder_mid[0] - hip_mid[0]
                torso_dy = hip_mid[1] - shoulder_mid[1]
                torso_lean = float(np.degrees(np.arctan2(torso_dy, abs(torso_dx) + 1e-5)))

                angles_dict = {
                    "knee_angle": smooth_knee,
                    "hip_angle": raw_hip,
                    "torso_angle": torso_lean,
                    "knee_distance": np.linalg.norm(user_kpts[13] - user_kpts[14]),
                    "ankle_distance": np.linalg.norm(user_kpts[15] - user_kpts[16])
                }
                
                # Only check depth when user is in BOTTOM or ASCENDING phase
                is_form_valid, error_msgs, form_score = evaluate_form_rules(exercise, angles_dict, phase=state_text)

                # Step 4: Multi-label AI Model Prediction (only active during movement)
                if ai_classifier.is_loaded and state_text in ["BOTTOM", "ASCENDING"]:
                    feature_row = np.array([
                        smooth_knee, smooth_knee, raw_hip, raw_hip,
                        90.0, 90.0, torso_lean, torso_lean,
                        0.08, 0.08
                    ], dtype=np.float32)
                    has_ai_err, ai_err_msgs, _ = ai_classifier.predict_errors(feature_row)
                    if has_ai_err and not is_form_valid:
                        error_msgs.extend(ai_err_msgs)

                # Step 5: Realtime Studio Voice Feedback Trigger
                feedback_engine.process_feedback(is_form_valid, error_msgs, state_text, current_time=curr_time)

        # 3. Render Professional HUD
        vis_frame = draw_hud(
            vis_frame, exercise, state_text, metric_display,
            form_score, is_form_valid, error_msgs, avg_fps, current_angle, num_persons
        )

        cv2.imshow("Pilates AI Pose Coach - Real-Time Alignment HUD", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Hotkeys 1-7 to switch exercises dynamically without restarting
        elif key == ord('1') and exercise != "squat":
            exercise = "squat"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Squat")
        elif key == ord('2') and exercise != "forward_lunge":
            exercise = "forward_lunge"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Lunge")
        elif key == ord('3') and exercise != "glute_bridge":
            exercise = "glute_bridge"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Cầu mông")
        elif key == ord('4') and exercise != "forearm_plank":
            exercise = "forearm_plank"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Plank")
        elif key == ord('5') and exercise != "pilates_hundred":
            exercise = "pilates_hundred"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Hundred")
        elif key == ord('6') and exercise != "bird_dog":
            exercise = "bird_dog"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Bird Dog")
        elif key == ord('7') and exercise != "side_leg_raise":
            exercise = "side_leg_raise"
            rep_fsm.reset(); hold_timer.reset()
            feedback_engine.speaker.speak("Chuyển sang bài Nâng chân ngang")

    cap.release()
    cv2.destroyAllWindows()
    feedback_engine.speaker.stop()
    print("[+] AI Coach closed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilates AI Pose Coach - Complete Realtime Application")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) or path to video file (.mp4)")
    parser.add_argument("--exercise", type=str, default="squat", choices=list(EXERCISE_CONFIGS.keys()), help="Exercise to assess")
    parser.add_argument("--mode", type=str, default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_ai_coach(source=args.source, exercise=args.exercise, mode=args.mode, device=args.device)
