"""
VideoPose3D 3D Pose Lifting Module

Input:
    RTMPose COCO-17 2D keypoints
    Shape: (17, 2)

Processing:
    2D COCO-17
        ↓
    Screen-coordinate normalization
        ↓
    Temporal buffer: 243 frames
        ↓
    VideoPose3D
        ↓
    3D pose: 17 joints

Output:
    3D H36M reduced skeleton
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEPOSE3D_ROOT = PROJECT_ROOT / "third_party" / "VideoPose3D"
CHECKPOINT_PATH = (
    VIDEPOSE3D_ROOT
    / "checkpoint"
    / "pretrained_h36m_detectron_coco.bin"
)

# Allow importing VideoPose3D official code
if str(VIDEPOSE3D_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEPOSE3D_ROOT))

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates


# ============================================================
# VIDEOPOSE3D CONFIGURATION
# ============================================================

NUM_JOINTS_2D = 17
NUM_JOINTS_3D = 17
IN_FEATURES = 2

FILTER_WIDTHS = [3, 3, 3, 3, 3]

CHANNELS = 1024
DROPOUT = 0.25

RECEPTIVE_FIELD = 243


# ============================================================
# H36M REDUCED SKELETON
# ============================================================
#
# This is the 17-joint skeleton used by VideoPose3D after
# removing static joints from the original Human3.6M skeleton.
#
# Important:
# This is NOT the COCO-17 order.
#
# Joint order:
#   0  Hip / Root
#   1  Right Hip
#   2  Right Knee
#   3  Right Foot
#   4  Left Hip
#   5  Left Knee
#   6  Left Foot
#   7  Spine
#   8  Thorax
#   9  Neck
#   10 Head
#   11 Left Shoulder
#   12 Left Elbow
#   13 Left Wrist
#   14 Right Shoulder
#   15 Right Elbow
#   16 Right Wrist
#

H36M_JOINT_NAMES = [
    "Root",
    "R_Hip",
    "R_Knee",
    "R_Foot",
    "L_Hip",
    "L_Knee",
    "L_Foot",
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


# Parent relationships after VideoPose3D reduces H36M
H36M_PARENTS = [
    -1,   # 0 Root
    0,    # 1 R_Hip
    1,    # 2 R_Knee
    2,    # 3 R_Foot
    0,    # 4 L_Hip
    4,    # 5 L_Knee
    5,    # 6 L_Foot
    0,    # 7 Spine
    7,    # 8 Thorax
    8,    # 9 Neck
    9,    # 10 Head
    8,    # 11 L_Shoulder
    11,   # 12 L_Elbow
    12,   # 13 L_Wrist
    8,    # 14 R_Shoulder
    14,   # 15 R_Elbow
    15,   # 16 R_Wrist
]


H36M_EDGES = [
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


# ============================================================
# VIDEOPOSE3D LIFTER
# ============================================================

class VideoPose3DLifter:
    """
    Convert a temporal sequence of RTMPose COCO-17 2D poses
    into a 3D pose using the official VideoPose3D model.
    """

    def __init__(
        self,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
        use_padding=True,
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.use_padding = use_padding

        # ----------------------------------------------------
        # Validate checkpoint
        # ----------------------------------------------------

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"\nVideoPose3D checkpoint not found:\n"
                f"{self.checkpoint_path}\n\n"
                f"Expected:\n"
                f"third_party/VideoPose3D/checkpoint/"
                f"pretrained_h36m_detectron_coco.bin"
            )

        # ----------------------------------------------------
        # Build official VideoPose3D model
        # ----------------------------------------------------

        self.model = TemporalModel(
            NUM_JOINTS_2D,
            IN_FEATURES,
            NUM_JOINTS_3D,
            filter_widths=FILTER_WIDTHS,
            causal=False,
            dropout=DROPOUT,
            channels=CHANNELS,
            dense=False,
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        print("=" * 65)
        print("VIDEOPOSE3D INITIALIZATION")
        print("=" * 65)
        print(f"[*] Device     : {self.device}")
        print(f"[*] Checkpoint : {self.checkpoint_path}")
        print(f"[*] Architecture: {FILTER_WIDTHS}")
        print(f"[*] Receptive field: {RECEPTIVE_FIELD} frames")
        print("[*] Loading checkpoint...")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_pos"])

        self.model.to(self.device)
        self.model.eval()

        self.buffer = deque(maxlen=RECEPTIVE_FIELD)

        print("[+] VideoPose3D loaded successfully.")
        print("=" * 65)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_keypoints(keypoints_2d, frame_width, frame_height):
        """
        Convert pixel coordinates from RTMPose into the same
        normalized screen coordinate system used by VideoPose3D.

        Input:
            keypoints_2d:
                shape (17, 2)
                pixel coordinates

        Output:
            shape (17, 2)
        """

        keypoints_2d = np.asarray(
            keypoints_2d,
            dtype=np.float32
        )

        if keypoints_2d.shape != (17, 2):
            raise ValueError(
                f"Expected keypoints shape (17, 2), "
                f"got {keypoints_2d.shape}"
            )

        normalized = normalize_screen_coordinates(
            keypoints_2d,
            w=frame_width,
            h=frame_height,
        )

        return normalized.astype(np.float32)

    # ========================================================
    # ADD FRAME
    # ========================================================

    def process_frame(
        self,
        keypoints_2d,
        frame_width,
        frame_height,
    ):
        """
        Add one RTMPose 2D frame into the temporal buffer.

        Returns:
            None
                if there is not enough data.

            np.ndarray
                shape (17, 3)
                when VideoPose3D produces a prediction.
        """

        normalized = self.normalize_keypoints(
            keypoints_2d,
            frame_width,
            frame_height,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Always append the frame.
        #
        # Do NOT remove the frame just because some keypoints
        # have low confidence.
        # ----------------------------------------------------

        self.buffer.append(normalized)

        # ----------------------------------------------------
        # Not enough frames yet
        # ----------------------------------------------------

        if len(self.buffer) < RECEPTIVE_FIELD:

            if not self.use_padding:
                return None

            # During warm-up, repeat the latest frame
            # until reaching 243 frames.
            sequence = self._get_padded_sequence()

        else:
            sequence = np.asarray(
                self.buffer,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Run VideoPose3D
        # ----------------------------------------------------

        pose_3d = self.predict(sequence)

        return pose_3d

    # ========================================================
    # GET PADDED SEQUENCE
    # ========================================================

    def _get_padded_sequence(self):
        """
        Create exactly 243 frames.

        If only N real frames are available, repeat the latest
        real frame to fill the remaining frames.

        This is only a warm-up strategy.
        """

        if len(self.buffer) == 0:
            return None

        sequence = np.asarray(
            self.buffer,
            dtype=np.float32
        )

        missing = RECEPTIVE_FIELD - len(sequence)

        if missing > 0:
            last_frame = sequence[-1:]

            padding = np.repeat(
                last_frame,
                missing,
                axis=0,
            )

            sequence = np.concatenate(
                [sequence, padding],
                axis=0,
            )

        return sequence

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, sequence_2d):
        """
        Run VideoPose3D inference.

        Input:
            sequence_2d:
                (243, 17, 2)

        Output:
            (17, 3)
        """

        sequence_2d = np.asarray(
            sequence_2d,
            dtype=np.float32
        )

        if sequence_2d.shape != (
            RECEPTIVE_FIELD,
            NUM_JOINTS_2D,
            IN_FEATURES,
        ):
            raise ValueError(
                "Invalid sequence shape: "
                f"{sequence_2d.shape}. "
                f"Expected "
                f"({RECEPTIVE_FIELD}, "
                f"{NUM_JOINTS_2D}, "
                f"{IN_FEATURES})"
            )

        # (243, 17, 2)
        #       ↓
        # (1, 243, 17, 2)

        input_tensor = torch.from_numpy(
            sequence_2d
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Official model returns:
        # (batch, output_frames, 17, 3)
        #
        # With 243-frame input and this architecture:
        # output_frames = 1

        prediction = prediction.detach().cpu().numpy()

        pose_3d = prediction[0, -1]

        return pose_3d.astype(np.float32)

    # ========================================================
    # BUFFER INFO
    # ========================================================

    def buffer_size(self):
        return len(self.buffer)

    def is_ready(self):
        return len(self.buffer) >= RECEPTIVE_FIELD

    def reset(self):
        self.buffer.clear()


# ============================================================
# 3D VISUALIZER
# ============================================================

class Pose3DVisualizer:
    """
    Simple persistent Matplotlib 3D skeleton visualizer.
    """

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt

        plt.ion()

        self.fig = plt.figure(
            "VideoPose3D - 3D Pose"
        )

        self.ax = self.fig.add_subplot(
            111,
            projection="3d"
        )

        self.ax.set_title(
            "VideoPose3D - Estimated 3D Pose"
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)

        self.ax.set_box_aspect((1, 1, 1))

        self.lines = []

        for _ in H36M_EDGES:
            line, = self.ax.plot(
                [],
                [],
                [],
                linewidth=2
            )
            self.lines.append(line)

        self.scatter = None

    def update(self, pose_3d):
        """
        Update the 3D skeleton.

        pose_3d shape:
            (17, 3)
        """

        if pose_3d is None:
            return

        pose = np.asarray(
            pose_3d,
            dtype=np.float32
        )

        if pose.shape != (17, 3):
            return

        # ----------------------------------------------------
        # Root-center the skeleton for visualization.
        #
        # VideoPose3D prediction is root-relative.
        # ----------------------------------------------------

        pose = pose - pose[0]

        # ----------------------------------------------------
        # Update bones
        # ----------------------------------------------------

        for line, (parent, child) in zip(
            self.lines,
            H36M_EDGES
        ):
            x = [
                pose[parent, 0],
                pose[child, 0]
            ]

            y = [
                pose[parent, 1],
                pose[child, 1]
            ]

            z = [
                pose[parent, 2],
                pose[child, 2]
            ]

            line.set_data(x, y)
            line.set_3d_properties(z)

        # ----------------------------------------------------
        # Update joints
        # ----------------------------------------------------

        if self.scatter is not None:
            self.scatter.remove()

        self.scatter = self.ax.scatter(
            pose[:, 0],
            pose[:, 1],
            pose[:, 2],
            s=20,
        )

        # ----------------------------------------------------
        # Dynamic axis range
        # ----------------------------------------------------

        max_range = np.max(
            np.ptp(pose, axis=0)
        )

        if max_range < 0.1:
            max_range = 1.0

        center = np.mean(
            pose,
            axis=0
        )

        half_range = max_range * 0.8

        self.ax.set_xlim(
            center[0] - half_range,
            center[0] + half_range
        )

        self.ax.set_ylim(
            center[1] - half_range,
            center[1] + half_range
        )

        self.ax.set_zlim(
            center[2] - half_range,
            center[2] + half_range
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.plt.pause(0.001)

    def close(self):
        self.plt.close(self.fig)


# ============================================================
# STANDALONE TEST
# ============================================================

def main():
    """
    Standalone test for the VideoPose3D module.

    This test uses RTMPose directly so that the file can be
    executed by itself.

    The integrated application should use main.py.
    """

    import cv2
    from rtmlib import Body

    print("\n")
    print("=" * 65)
    print("VIDEOPOSE3D STANDALONE TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # RTMPose
    # --------------------------------------------------------

    body_estimator = Body(
        mode="balanced",
        backend="onnxruntime",
        device="cpu"
    )

    # --------------------------------------------------------
    # VideoPose3D
    # --------------------------------------------------------

    lifter = VideoPose3DLifter(
        device="cpu",
        use_padding=True
    )

    # --------------------------------------------------------
    # 3D visualization
    # --------------------------------------------------------

    visualizer = Pose3DVisualizer()

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Could not open webcam.")
        visualizer.close()
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print("[+] Webcam started.")
    print("[+] Press 'q' to exit.")
    print()

    frame_count = 0

    try:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            h, w = frame.shape[:2]

            # ------------------------------------------------
            # RTMPose
            # ------------------------------------------------

            keypoints, scores = body_estimator(frame)

            # ------------------------------------------------
            # Draw 2D skeleton
            # ------------------------------------------------

            from rtmlib import draw_skeleton

            vis_frame = draw_skeleton(
                frame,
                keypoints,
                scores,
                kpt_thr=0.4
            )

            # ------------------------------------------------
            # VideoPose3D
            # ------------------------------------------------

            if (
                keypoints is not None
                and len(keypoints) > 0
            ):

                user_kpts = keypoints[0]

                pose_3d = lifter.process_frame(
                    user_kpts,
                    w,
                    h
                )

                # --------------------------------------------
                # 3D visualization
                # --------------------------------------------

                if pose_3d is not None:
                    visualizer.update(pose_3d)

            # ------------------------------------------------
            # HUD
            # ------------------------------------------------

            buffer_count = lifter.buffer_size()

            cv2.rectangle(
                vis_frame,
                (10, 10),
                (420, 105),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                vis_frame,
                "RTMPose -> VideoPose3D",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vis_frame,
                f"2D Buffer: {buffer_count}/{RECEPTIVE_FIELD}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            status = (
                "3D: READY"
                if lifter.is_ready()
                else "3D: WARM-UP"
            )

            cv2.putText(
                vis_frame,
                status,
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "VideoPose3D Standalone Test",
                vis_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()
        visualizer.close()

        print("[*] VideoPose3D test terminated.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()