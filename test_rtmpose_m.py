"""
Real-time RTMPose Pose Estimation & Kinematics Benchmark
Model Architecture:
  - Human Detector: RTMDet (RTMDet-m / RTMDet-nano)
  - Pose Estimator: RTMPose (RTMPose-m / RTMPose-t/s) with SimCC Head
  - Backend: ONNX Runtime (CPU / CUDA)
  - Keypoints: 17 COCO Standard Keypoints (Expandable to WholeBody 26/133)
"""

import argparse
import time
import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# 17 COCO Keypoints definition
KEYPOINT_NAMES = [
    "Nose",         # 0
    "L_Eye",        # 1
    "R_Eye",        # 2
    "L_Ear",        # 3
    "R_Ear",        # 4
    "L_Shoulder",   # 5
    "R_Shoulder",   # 6
    "L_Elbow",      # 7
    "R_Elbow",      # 8
    "L_Wrist",      # 9
    "R_Wrist",      # 10
    "L_Hip",        # 11
    "R_Hip",       # 12
    "L_Knee",      # 13
    "R_Knee",        # 14
    "L_Ankle",       # 15
    "R_Ankle"       # 16
]

def calculate_angle(p1, p2, p3):
    """
    Calculate the interior angle (in degrees) between three 2D keypoints: p1 - p2 - p3.
    p2 is the apex/vertex of the angle (e.g. Hip -> Knee -> Ankle).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)

def run_rtmpose_benchmark(source=0, mode='balanced', device='cpu', kpt_thr=0.4):
    """
    Run the RTMPose pipeline benchmark with ONNX Runtime backend.
    """
    mode_info = {
        'lightweight': 'RTMDet-nano + RTMPose-t (Ultra Fast ~120+ FPS, Edge/Mobile)',
        'balanced': 'RTMDet-m + RTMPose-m (High Precision ~75.3% AP, ~90 FPS, Pilates Standard)',
        'performance': 'RTMDet-l + RTMPose-l (Max Precision Research)'
    }

    print("=" * 65)
    print("PILATES AI POSE COACH - BENCHMARK PIPELINE")
    print("=" * 65)
    print(f"[*] Model Pipeline : {mode_info.get(mode, mode)}")
    print(f"[*] Inference Engine: ONNX Runtime ({device.upper()})")
    print(f"[*] Video Source    : {source}")
    print(f"[*] Confidence Thr  : {kpt_thr}")
    print("=" * 65)
    print("[*] Loading ONNX checkpoint from rtmlib cache...")

    # Initialize body pose estimator via rtmlib
    body_estimator = Body(
        mode=mode,
        backend='onnxruntime',
        device=device
    )

    # Convert source to int if integer string passed
    if isinstance(source, str) and source.isdigit():
        video_src = int(source)
    else:
        video_src = source

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {source}")
        return

    # Set default resolution if using webcam
    if isinstance(video_src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_history = []
    print("[+] Pipeline ready! Press 'q' on the video window to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[*] Video stream ended.")
            break

        start_time = time.time()

        # 1. Pipeline Inference: Detect human & predict 17 keypoints with SimCC
        # keypoints: ndarray (N, 17, 2), scores: ndarray (N, 17)
        keypoints, scores = body_estimator(frame)
        print("keypoints shape:", keypoints.shape)
        print("scores shape:", scores.shape)
        print("first person:", keypoints[0])
        inf_time = time.time() - start_time
        curr_fps = 1.0 / inf_time if inf_time > 0 else 0
        fps_history.append(curr_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        # 2. Draw skeleton overlay
        vis_frame = draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr)

        # 3. Draw HUD with Model Specs & Performance metrics
        num_persons = len(keypoints) if keypoints is not None else 0
        h, w, _ = vis_frame.shape

        cv2.rectangle(vis_frame, (10, 10), (380, 125), (20, 20, 20), -1)
        cv2.rectangle(vis_frame, (10, 10), (380, 125), (0, 255, 200), 2)

        cv2.putText(vis_frame, f"AI Model: RTMPose-{mode[0].upper()} (SimCC Head)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Backend : ONNX Runtime [{device.upper()}]", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(vis_frame, f"FPS: {avg_fps:.1f} ({inf_time*1000:.1f} ms)", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Persons Detected: {num_persons}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # 4. Extract kinematic angles for primary person
        if num_persons > 0:
            user_kpts = keypoints[0]
            user_scores = scores[0]

            # Mark confident keypoints
            for i, name in enumerate(KEYPOINT_NAMES):
                if user_scores[i] > kpt_thr:
                    kx, ky = user_kpts[i]
                    cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 255, 255), -1)

            # Left Knee Angle: Hip(11) -> Knee(13) -> Ankle(15)
            if user_scores[11] > kpt_thr and user_scores[13] > kpt_thr and user_scores[15] > kpt_thr:
                left_knee_angle = calculate_angle(user_kpts[11], user_kpts[13], user_kpts[15])
                knee_pos = user_kpts[13]
                cv2.putText(vis_frame, f"L-Knee: {left_knee_angle:.1f} deg", 
                            (int(knee_pos[0]) + 12, int(knee_pos[1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Right Knee Angle: Hip(12) -> Knee(14) -> Ankle(16)
            if user_scores[12] > kpt_thr and user_scores[14] > kpt_thr and user_scores[16] > kpt_thr:
                right_knee_angle = calculate_angle(user_kpts[12], user_kpts[14], user_kpts[16])
                knee_pos = user_kpts[14]
                cv2.putText(vis_frame, f"R-Knee: {right_knee_angle:.1f} deg", 
                            (int(knee_pos[0]) + 12, int(knee_pos[1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Pilates AI Pose Coach - RTMPose Benchmark", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Benchmark terminated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilates AI Pose Coach - RTMPose Real-time Test")
    parser.add_argument("--source", type=str, default="0", 
                        help="Video source: webcam index (0, 1) or path to mp4 video file")
    parser.add_argument("--mode", type=str, default="balanced", 
                        choices=["lightweight", "balanced", "performance"],
                        help="Model preset: lightweight (nano/tiny), balanced (m - recommended), performance (l)")
    parser.add_argument("--device", type=str, default="cpu", 
                        choices=["cpu", "cuda"],
                        help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument("--kpt-thr", type=float, default=0.4, 
                        help="Keypoint detection confidence threshold (default: 0.4)")
    
    args = parser.parse_args()
    run_rtmpose_benchmark(source=args.source, mode=args.mode, device=args.device, kpt_thr=args.kpt_thr)
