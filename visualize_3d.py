"""
Interactive 3D Skeleton Visualization

Input:
    data/processed/keypoints_3d.npy

Features:
    - Human3.6M 17-joint skeleton
    - XYZ axes
    - Interactive rotation
    - Frame slider
    - Joint selector
    - XYZ coordinates

Run:
    python visualize_3d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "data" / "processed" / "keypoints_3d.npy"


# Human3.6M 17 joint ordering
JOINT_NAMES = [
    "Pelvis",  # 0
    "R_Hip",  # 1
    "R_Knee",  # 2
    "R_Ankle",  # 3
    "L_Hip",  # 4
    "L_Knee",  # 5
    "L_Ankle",  # 6
    "Spine",  # 7
    "Thorax",  # 8
    "Neck",  # 9
    "Head",  # 10
    "L_Shoulder",  # 11
    "L_Elbow",  # 12
    "L_Wrist",  # 13
    "R_Shoulder",  # 14
    "R_Elbow",  # 15
    "R_Wrist",  # 16
]


# Human3.6M skeleton connections
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


# =====================================================
# 2. LOAD 3D KEYPOINTS
# =====================================================


def load_3d_keypoints():

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"3D file not found: {INPUT_PATH}")

    poses = np.load(INPUT_PATH, allow_pickle=False).astype(np.float32)

    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(f"Expected (T, 17, 3), got {poses.shape}")

    if len(poses) == 0:
        raise ValueError("No frames found")

    if not np.isfinite(poses).all():
        raise ValueError("3D data contains NaN or Inf")

    print("[INFO] Loaded 3D:", poses.shape)

    return poses


# =====================================================
# 3. VISUALIZATION
# =====================================================


def visualize(poses):

    num_frames = len(poses)

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_axes([0.08, 0.23, 0.72, 0.68], projection="3d")

    # Frame slider
    frame_ax = fig.add_axes([0.18, 0.12, 0.60, 0.025])

    frame_slider = Slider(
        ax=frame_ax,
        label="Frame",
        valmin=0,
        valmax=max(1, num_frames - 1),
        valinit=0,
        valstep=1,
    )

    # Joint selector
    joint_ax = fig.add_axes([0.18, 0.06, 0.60, 0.025])

    joint_slider = Slider(
        ax=joint_ax, label="Joint", valmin=0, valmax=16, valinit=0, valstep=1
    )

    # Coordinate text area
    info_text = fig.text(
        0.80, 0.67, "", fontsize=10, family="monospace", verticalalignment="top"
    )

    # Global symmetric axis limits.
    # Keep the same scale while switching frames.
    max_abs = float(np.max(np.abs(poses)))
    limit = max(max_abs * 1.1, 0.1)

    def draw_frame(frame_index, selected_joint):

        # Preserve current view when changing frames.
        elevation = ax.elev
        azimuth = ax.azim

        ax.cla()

        pose = poses[frame_index]

        # =================================================
        # PLOT BONES
        # =================================================

        for start, end in BONES:
            xs = pose[[start, end], 0]
            ys = pose[[start, end], 1]
            zs = pose[[start, end], 2]

            ax.plot(xs, ys, zs, color="royalblue", linewidth=2.5)

        # =================================================
        # PLOT JOINTS
        # =================================================

        ax.scatter(
            pose[:, 0], pose[:, 1], pose[:, 2], c="dodgerblue", s=35, depthshade=False
        )

        # Highlight selected joint
        x, y, z = pose[selected_joint]

        ax.scatter([x], [y], [z], c="red", s=110, depthshade=False)

        ax.text(x, y, z, f" {selected_joint}", color="red", fontsize=11)

        # =================================================
        # DRAW XYZ REFERENCE AXES
        # =================================================

        ax.plot([-limit, limit], [0, 0], [0, 0], color="red", alpha=0.35)

        ax.plot([0, 0], [-limit, limit], [0, 0], color="green", alpha=0.35)

        ax.plot([0, 0], [0, 0], [-limit, limit], color="blue", alpha=0.35)

        # =================================================
        # AXIS CONFIGURATION
        # =================================================

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"RTMPose + VideoPose3D | Frame {frame_index}", fontsize=13)

        ax.view_init(elev=elevation, azim=azimuth)

        # =================================================
        # DISPLAY SELECTED JOINT XYZ
        # =================================================

        info_text.set_text(
            f"JOINT {selected_joint}\n"
            f"{JOINT_NAMES[selected_joint]}\n\n"
            f"X: {x:+.4f}\n"
            f"Y: {y:+.4f}\n"
            f"Z: {z:+.4f}\n\n"
            f"Frame:\n"
            f"{frame_index}/{num_frames - 1}"
        )

        fig.canvas.draw_idle()

    # =====================================================
    # 4. SLIDER CALLBACKS
    # =====================================================

    def update(_):
        frame_index = min(int(frame_slider.val), num_frames - 1)

        selected_joint = int(joint_slider.val)

        draw_frame(frame_index, selected_joint)

    frame_slider.on_changed(update)
    joint_slider.on_changed(update)

    draw_frame(0, 0)

    plt.show()


# =====================================================
# 5. MAIN
# =====================================================


def main():

    print("=" * 60)
    print("VIDEOPOSE3D - XYZ SKELETON VIEWER")
    print("=" * 60)

    poses = load_3d_keypoints()

    print("[INFO] Number of frames:", len(poses))
    print("[INFO] Joint count:", len(JOINT_NAMES))
    print("[INFO] Drag mouse to rotate the 3D view")
    print("[INFO] Use sliders to select frame and joint")

    visualize(poses)


if __name__ == "__main__":
    main()
