"""
Live RTMPose 2D + VideoPose3D 3D

Camera 2D and XYZ skeleton side by side.

Controls:
    Q / ESC : Quit
    A / D   : Rotate 3D left/right
    W / S   : Rotate 3D up/down

Run:
    python live_3d.py --device cpu
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import torch
from rtmlib import Body, draw_skeleton

# ============================================
# 1. PATHS
# ============================================

ROOT = Path(__file__).resolve().parent
VP3D_DIR = ROOT / "third_party" / "VideoPose3D"

sys.path.insert(0, str(VP3D_DIR))

from third_party.VideoPose3D.common.camera import normalize_screen_coordinates
from third_party.VideoPose3D.common.model import TemporalModelOptimized1f

CHECKPOINT = VP3D_DIR / "checkpoint" / "pretrained_h36m_detectron_coco.bin"


# ============================================
# 2. CONFIGURATION
# ============================================

WIDTH = 640
HEIGHT = 480

# Run VideoPose3D every N webcam frames.
# Increase this on slow CPUs.
LIFT_INTERVAL = 2

# The pretrained non-causal model uses
# 243 temporal frames.
RECEPTIVE_FIELD = 243

JOINT_NAMES = [
    "Pelvis",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "Spine",
    "Thorax",
    "Neck",
    "Head",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]

BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


# ============================================
# 3. VIDEO FRAME PREPARATION
# ============================================


def prepare_frame(frame):
    """
    Resize with original aspect ratio preserved,
    then pad to 640x480.
    """
    h, w = frame.shape[:2]

    scale = min(WIDTH / w, HEIGHT / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    x = (WIDTH - new_w) // 2
    y = (HEIGHT - new_h) // 2

    canvas[y : y + new_h, x : x + new_w] = resized

    return canvas


# ============================================
# 4. LOAD VIDEOPOSE3D
# ============================================


def load_videopose3d(device):

    model = TemporalModelOptimized1f(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
    ).to(device)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_pos"])
    model.eval()

    assert model.receptive_field() == RECEPTIVE_FIELD

    print("[OK] VideoPose3D loaded")

    return model


# ============================================
# 5. TEMPORAL INPUT
# ============================================


def build_temporal_input(history):
    """
    Approximate current-frame prediction using
    past frames plus repeated future boundary.

    121 past + current + 121 repeated current.

    This is a low-latency approximation,
    not exact offline non-causal inference.
    """
    frames = np.asarray(list(history), dtype=np.float32)

    # Last 122 frames = 121 past + current.
    frames = frames[-122:]

    if len(frames) < 122:
        missing = 122 - len(frames)
        left = np.repeat(frames[:1], missing, axis=0)
        frames = np.concatenate([left, frames], axis=0)

    # Future frames are unavailable live.
    right = np.repeat(frames[-1:], 121, axis=0)

    sequence = np.concatenate([frames, right], axis=0)

    assert sequence.shape == (243, 17, 2)

    return sequence


# ============================================
# 6. ASYNCHRONOUS 3D WORKER
# ============================================


class Pose3DWorker:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.tasks = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_pose = None
        self.latest_frame_id = -1
        self.latest_latency_ms = 0.0
        self.error = None

        self.thread = threading.Thread(target=self._run, daemon=True)

        self.thread.start()

    def submit(self, sequence, frame_id):
        """
        Keep the newest pending request.
        Do not accumulate stale frames.
        """
        item = (sequence.copy(), frame_id)

        try:
            self.tasks.put_nowait(item)
        except Full:
            try:
                self.tasks.get_nowait()
            except Empty:
                pass

            try:
                self.tasks.put_nowait(item)
            except Full:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                sequence, frame_id = self.tasks.get(timeout=0.1)
            except Empty:
                continue

            try:
                start = time.perf_counter()

                inputs = torch.from_numpy(sequence[None]).to(self.device)

                with torch.inference_mode():
                    prediction = self.model(inputs)

                pose = prediction[0, 0].cpu().numpy()

                # Pelvis-relative coordinates
                pose -= pose[0:1]

                latency = (time.perf_counter() - start) * 1000

                with self.lock:
                    self.latest_pose = pose.copy()
                    self.latest_frame_id = frame_id
                    self.latest_latency_ms = latency

            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                self.stop_event.set()

    def get_result(self):
        with self.lock:
            pose = None if self.latest_pose is None else self.latest_pose.copy()

            return (pose, self.latest_frame_id, self.latest_latency_ms, self.error)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


# ============================================
# 7. 3D SKELETON RENDERER
# ============================================


def project_3d(points, azimuth, elevation, scale=170):
    """
    Rotate and orthographically project XYZ
    coordinates to a 2D canvas.
    """

    az = np.radians(azimuth)
    el = np.radians(elevation)

    ry = np.array(
        [[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]],
        dtype=np.float32,
    )

    rx = np.array(
        [[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]],
        dtype=np.float32,
    )

    rotated = points @ ry.T @ rx.T

    # Orthographic projection
    pixels = np.column_stack(
        [WIDTH / 2 + rotated[:, 0] * scale, HEIGHT / 2 - rotated[:, 1] * scale]
    )

    return pixels.astype(np.int32)


def draw_3d_grid(
    canvas,
    azimuth,
    elevation,
    scale=170,
    extent=1.0,
    step=0.25
):
    """
    Draw a 3D coordinate box with grid lines.
    X: horizontal
    Y: vertical
    Z: depth
    """

    grid_color = (65, 65, 65)
    edge_color = (115, 115, 115)

    def to_pixel(point):
        point = np.asarray([point], dtype=np.float32)
        return tuple(
            project_3d(
                point, azimuth, elevation, scale
            )[0]
        )

    def line(a, b, color=grid_color, thickness=1):
        cv2.line(
            canvas,
            to_pixel(a),
            to_pixel(b),
            color,
            thickness,
            cv2.LINE_AA
        )

    ticks = np.linspace(-extent, extent,
                        int(2 * extent / step) + 1)

    # Floor grid: XZ plane at Y = -extent
    for t in ticks:
        line(
            [-extent, -extent, t],
            [extent, -extent, t]
        )
        line(
            [t, -extent, -extent],
            [t, -extent, extent]
        )

    # Back wall: XY plane at Z = extent
    for t in ticks:
        line(
            [-extent, t, extent],
            [extent, t, extent]
        )
        line(
            [t, -extent, extent],
            [t, extent, extent]
        )

    # Side wall: YZ plane at X = -extent
    for t in ticks:
        line(
            [-extent, t, -extent],
            [-extent, t, extent]
        )
        line(
            [-extent, -extent, t],
            [-extent, extent, t]
        )

    # Outer box edges
    corners = [
        [-extent, -extent, -extent],
        [extent, -extent, -extent],
        [extent, extent, -extent],
        [-extent, extent, -extent],
        [-extent, -extent, extent],
        [extent, -extent, extent],
        [extent, extent, extent],
        [-extent, extent, extent]
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for a, b in edges:
        line(
            corners[a],
            corners[b],
            edge_color,
            1
        )


def draw_3d_panel(pose, azimuth=0, elevation=0, frame_lag=0, latency=0):
    canvas = np.full((HEIGHT, WIDTH, 3), (25, 25, 30), dtype=np.uint8)

    cv2.putText(
        canvas,
        "VideoPose3D - Live XYZ",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if pose is None:
        cv2.putText(
            canvas,
            "Waiting for 3D prediction...",
            (100, HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
        )
        return canvas
    # Draw 3D coordinate grid
    draw_3d_grid(
        canvas,
        azimuth,
        elevation,
        scale=170,
        extent=0.9,
        step=0.3
    )

    # Draw XYZ reference axes
    axis_points = np.array(
        [[0, 0, 0], [0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]], dtype=np.float32
    )

    axes = project_3d(axis_points, azimuth, elevation)

    colors = [
        (0, 0, 255),  # X: red
        (0, 255, 0),  # Y: green
        (255, 120, 0),  # Z: blue
    ]

    for i in range(3):
        cv2.line(canvas, tuple(axes[0]), tuple(axes[i + 1]), colors[i], 2)

        cv2.putText(
            canvas,
            "XYZ"[i],
            tuple(axes[i + 1] + [5, -5]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colors[i],
            2,
        )
    # Convert camera coordinates to display coordinates
    pose_view = pose.copy()

    # Camera Y points downward; display Y points upward
    pose_view[:, 1] *= -1

    # Project skeleton joints
    points = project_3d(
        pose_view, azimuth, elevation
    )


    # Draw bones
    for a, b in BONES:
        cv2.line(
            canvas, tuple(points[a]), tuple(points[b]), (255, 180, 50), 2, cv2.LINE_AA
        )

    # Draw joints
    for i, point in enumerate(points):
        cv2.circle(canvas, tuple(point), 5, (0, 210, 255), -1, cv2.LINE_AA)

    # Display coordinates for selected joints
    selected = [(0, "Pelvis"), (2, "R_Knee"), (5, "L_Knee")]

    y = 355

    for idx, name in selected:
        x3, y3, z3 = pose_view[idx]

        text = f"{name}: {x3:+.2f}, {y3:+.2f}, {z3:+.2f}"

        cv2.putText(
            canvas, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1
        )

        y += 24

    cv2.putText(
        canvas,
        f"3D inference: {latency:.0f} ms",
        (15, 447),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 180),
        1,
    )

    cv2.putText(
        canvas,
        f"3D frame lag: {frame_lag}",
        (15, 468),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return canvas


# ============================================
# 8. REAL-TIME MAIN PIPELINE
# ============================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="0")

    parser.add_argument(
        "--mode", default="balanced", choices=["lightweight", "balanced", "performance"]
    )

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    device = torch.device(args.device)

    print("[INFO] Loading RTMPose...")

    body = Body(mode=args.mode, backend="onnxruntime", device=args.device)

    print("[INFO] Loading VideoPose3D...")

    model_3d = load_videopose3d(device)
    worker = Pose3DWorker(model_3d, device)

    source = int(args.source) if str(args.source).isdigit() else args.source

    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        worker.stop()
        raise RuntimeError("Cannot open camera/video")

    # Maximum history needed for current frame:
    # 121 previous frames + current frame.
    history = deque(maxlen=122)

    azimuth = 0
    elevation = 0

    frame_id = 0
    fps = 0.0
    prev_time = time.perf_counter()

    print("[OK] Live pipeline started")
    print("Q: quit | A/D: rotate | W/S: tilt")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            frame = prepare_frame(frame)

            # ====================================
            # RTMPOSE 2D
            # ====================================

            keypoints, scores = body(frame)

            valid = keypoints is not None and len(keypoints) > 0

            if valid:
                person = np.asarray(keypoints[0], dtype=np.float32)

                confidence = np.asarray(scores[0], dtype=np.float32)

                valid = (
                    person.shape == (17, 2)
                    and confidence.shape == (17,)
                    and np.isfinite(person).all()
                )

            if valid:
                # Normalize using the exact frame
                # dimensions passed to RTMPose.
                normalized = normalize_screen_coordinates(
                    person, w=WIDTH, h=HEIGHT
                ).astype(np.float32)

                history.append(normalized)

            elif history:
                history.append(history[-1].copy())

            # ====================================
            # VIDEOPOSE3D ASYNC INFERENCE
            # ====================================

            if history and frame_id % LIFT_INTERVAL == 0:
                sequence = build_temporal_input(history)

                worker.submit(sequence, frame_id)

            pose_3d, pose_frame, latency, error = worker.get_result()

            if error is not None:
                raise RuntimeError(f"VideoPose3D worker error: {error}")

            # ====================================
            # 2D CAMERA PANEL
            # ====================================

            if valid:
                vis_2d = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.4)
            else:
                vis_2d = frame.copy()

            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = 0.85 * fps + 0.15 * instant_fps if fps > 0 else instant_fps

            cv2.putText(
                vis_2d,
                "RTMPose 2D - Live Camera",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 200),
                2,
            )

            cv2.putText(
                vis_2d,
                f"Display FPS: {fps:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            # ====================================
            # 3D PANEL
            # ====================================

            frame_lag = frame_id - pose_frame if pose_frame >= 0 else 0

            vis_3d = draw_3d_panel(
                pose_3d,
                azimuth=azimuth,
                elevation=elevation,
                frame_lag=frame_lag,
                latency=latency,
            )

            # Both panels in one window
            combined = np.hstack([vis_2d, vis_3d])

            cv2.imshow("Pilates AI Coach | 2D + Live 3D", combined)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("a"):
                azimuth -= 10

            elif key == ord("d"):
                azimuth += 10

            elif key == ord("w"):
                elevation += 10

            elif key == ord("s"):
                elevation -= 10

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
