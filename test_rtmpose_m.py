"""
Real-time RTMPose Pose Estimation & Kinematics Benchmark

Model Architecture:
  - Human Detector: RTMDet
  - Pose Estimator: RTMPose with SimCC Head
  - Backend: ONNX Runtime (CPU / CUDA)
  - Keypoints: 17 COCO Standard Keypoints

Additional:
  - Collect 2D keypoints for VideoPose3D
  - Export keypoints_2d.npy: (T, 17, 2)
  - Export keypoints_metadata.npz
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# ============================================================
# 1. COCO 17 KEYPOINTS
# ============================================================

KEYPOINT_NAMES = [
    "Nose",  # 0
    "L_Eye",  # 1
    "R_Eye",  # 2
    "L_Ear",  # 3
    "R_Ear",  # 4
    "L_Shoulder",  # 5
    "R_Shoulder",  # 6
    "L_Elbow",  # 7
    "R_Elbow",  # 8
    "L_Wrist",  # 9
    "R_Wrist",  # 10
    "L_Hip",  # 11
    "R_Hip",  # 12
    "L_Knee",  # 13
    "R_Knee",  # 14
    "L_Ankle",  # 15
    "R_Ankle",  # 16
]


# ============================================================
# 2. CALCULATE 2D JOINT ANGLE
# ============================================================


def calculate_angle(p1, p2, p3):
    """
    Calculate interior angle (degrees) between 3 keypoints.
    p2 is the vertex of the angle.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


# ============================================================
# 3. SAVE 2D KEYPOINT SEQUENCE
# ============================================================


def save_keypoints(keypoints_history, output_dir, width, height, fps, valid_history):
    """
    Save 2D keypoints and video metadata for VideoPose3D.

    keypoints_2d.npy:
        Shape (T, 17, 2)
        Pixel coordinates in original video resolution.

    keypoints_metadata.npz:
        width, height, fps, valid_frames
    """
    if not keypoints_history:
        print("[!] No valid keypoints to save.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_2d = np.stack(keypoints_history, axis=0).astype(np.float32)

    valid_frames = np.asarray(valid_history, dtype=np.bool_)

    assert poses_2d.ndim == 3
    assert poses_2d.shape[1:] == (17, 2)
    assert len(valid_frames) == len(poses_2d)

    npy_path = output_dir / "keypoints_2d.npy"
    metadata_path = output_dir / "keypoints_metadata.npz"

    np.save(npy_path, poses_2d)

    np.savez(
        metadata_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        valid_frames=valid_frames,
        keypoint_format="COCO-17",
    )

    print("\n" + "=" * 60)
    print("[+] 2D KEYPOINT EXPORT COMPLETED")
    print(f"[+] File: {npy_path}")
    print(f"[+] Shape: {poses_2d.shape}")
    print(f"[+] Data type: {poses_2d.dtype}")
    print(f"[+] Width: {width}")
    print(f"[+] Height: {height}")
    print(f"[+] FPS: {fps:.2f}")
    print(
        f"[+] Frames with detected person: "
        f"{int(valid_frames.sum())}/{len(valid_frames)}"
    )
    print(f"[+] Metadata: {metadata_path}")
    print("=" * 60)


# ============================================================
# 4. MAIN RTMPOSE PIPELINE
# ============================================================


def run_rtmpose_benchmark(
    source=0, mode="balanced", device="cpu", kpt_thr=0.4, output_dir="outputs"
):
    mode_info = {
        "lightweight": "RTMDet-nano + RTMPose-t",
        "balanced": "RTMDet-m + RTMPose-m",
        "performance": "RTMDet-l + RTMPose-l",
    }

    print("=" * 65)
    print("PILATES AI POSE COACH - RTMPOSE 2D PIPELINE")
    print("=" * 65)
    print(f"[*] Model: {mode_info.get(mode, mode)}")
    print(f"[*] Backend: ONNX Runtime ({device.upper()})")
    print(f"[*] Source: {source}")
    print(f"[*] Confidence threshold: {kpt_thr}")
    print("=" * 65)

    # Initialize RTMDet + RTMPose
    print("[*] Loading RTMPose models...")

    body_estimator = Body(mode=mode, backend="onnxruntime", device=device)

    # Parse webcam index or video file
    if isinstance(source, str) and source.isdigit():
        video_src = int(source)
    else:
        video_src = source

    cap = cv2.VideoCapture(video_src)

    if not cap.isOpened():
        print(f"[!] Cannot open video source: {source}")
        return

    if isinstance(video_src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    print(f"[*] Video resolution: {width}x{height}")
    print(f"[*] Video FPS: {fps:.2f}")

    # Tracking data for VideoPose3D
    keypoints_history = []
    valid_history = []

    fps_history = []
    frame_count = 0

    print("[+] Pipeline ready!")
    print("[+] Press 'q' to stop and export keypoints.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("[*] Video stream ended.")
                break

            frame_count += 1
            start_time = time.time()

            # ------------------------------------------------
            # STEP 1: RTMDET + RTMPOSE INFERENCE
            # ------------------------------------------------

            keypoints, scores = body_estimator(frame)

            num_persons = len(keypoints) if keypoints is not None else 0

            # ------------------------------------------------
            # STEP 2: COLLECT 2D KEYPOINTS
            # ------------------------------------------------

            if num_persons > 0:
                user_kpts = np.asarray(keypoints[0], dtype=np.float32)

                user_scores = np.asarray(scores[0], dtype=np.float32)

                if user_kpts.shape == (17, 2) and np.isfinite(user_kpts).all():
                    keypoints_history.append(user_kpts.copy())
                    valid_history.append(True)
                elif keypoints_history:
                    keypoints_history.append(keypoints_history[-1].copy())
                    valid_history.append(False)

            elif keypoints_history:
                # Forward-fill on missing detection.
                # Suitable only for a simple single-person test.
                keypoints_history.append(keypoints_history[-1].copy())
                valid_history.append(False)

            # ------------------------------------------------
            # STEP 3: PERFORMANCE METRICS
            # ------------------------------------------------

            inf_time = time.time() - start_time

            curr_fps = 1.0 / inf_time if inf_time > 0 else 0.0

            fps_history.append(curr_fps)

            if len(fps_history) > 30:
                fps_history.pop(0)

            avg_fps = float(np.mean(fps_history))

            # ------------------------------------------------
            # STEP 4: DRAW 2D SKELETON
            # ------------------------------------------------

            if keypoints is not None and num_persons > 0:
                vis_frame = draw_skeleton(
                    frame.copy(), keypoints, scores, kpt_thr=kpt_thr
                )
            else:
                vis_frame = frame.copy()

            # ------------------------------------------------
            # STEP 5: DRAW HUD
            # ------------------------------------------------

            cv2.rectangle(vis_frame, (10, 10), (420, 145), (20, 20, 20), -1)

            cv2.rectangle(vis_frame, (10, 10), (420, 145), (0, 255, 200), 2)

            cv2.putText(
                vis_frame,
                f"Model: RTMPose ({mode})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis_frame,
                f"Backend: ONNX Runtime [{device.upper()}]",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.putText(
                vis_frame,
                f"FPS: {avg_fps:.1f} ({inf_time * 1000:.1f} ms)",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                vis_frame,
                f"Persons: {num_persons}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                1,
            )

            cv2.putText(
                vis_frame,
                f"Collected frames: {len(keypoints_history)}",
                (20, 137),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # ------------------------------------------------
            # STEP 6: CALCULATE 2D KINEMATICS
            # ------------------------------------------------

            if num_persons > 0:
                user_kpts = keypoints[0]
                user_scores = scores[0]

                for i, name in enumerate(KEYPOINT_NAMES):
                    if user_scores[i] > kpt_thr:
                        kx, ky = user_kpts[i]

                        cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 255, 255), -1)

                # Left knee: Hip(11)-Knee(13)-Ankle(15)
                if np.all(user_scores[[11, 13, 15]] > kpt_thr):
                    angle = calculate_angle(user_kpts[11], user_kpts[13], user_kpts[15])

                    x, y = user_kpts[13]

                    cv2.putText(
                        vis_frame,
                        f"L-Knee: {angle:.1f} deg",
                        (int(x) + 12, int(y) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # Right knee: Hip(12)-Knee(14)-Ankle(16)
                if np.all(user_scores[[12, 14, 16]] > kpt_thr):
                    angle = calculate_angle(user_kpts[12], user_kpts[14], user_kpts[16])

                    x, y = user_kpts[14]

                    cv2.putText(
                        vis_frame,
                        f"R-Knee: {angle:.1f} deg",
                        (int(x) + 12, int(y) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 200, 0),
                        2,
                    )

            # ------------------------------------------------
            # STEP 7: DISPLAY
            # ------------------------------------------------

            cv2.imshow("Pilates AI Pose Coach - RTMPose 2D", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[*] Stop requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ----------------------------------------------------
        # STEP 8: EXPORT 2D KEYPOINT SEQUENCE
        # ----------------------------------------------------

        save_keypoints(
            keypoints_history=keypoints_history,
            output_dir=output_dir,
            width=width,
            height=height,
            fps=fps,
            valid_history=valid_history,
        )

        print(f"[*] Total video frames processed: {frame_count}")
        print("[+] RTMPose benchmark finished.")


# ============================================================
# 5. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Pilates AI Pose Coach - RTMPose 2D Export for VideoPose3D")
    )

    parser.add_argument(
        "--source", type=str, default="0", help="Webcam index or MP4 video path"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
    )

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--kpt-thr", type=float, default=0.4)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for exported keypoints",
    )

    args = parser.parse_args()

    run_rtmpose_benchmark(
        source=args.source,
        mode=args.mode,
        device=args.device,
        kpt_thr=args.kpt_thr,
        output_dir=args.output_dir,
    )
