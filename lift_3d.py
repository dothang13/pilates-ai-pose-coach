"""
RTMPose 2D to VideoPose3D Lifting

Input:
    data/processed/keypoints_2d.npy
    data/processed/keypoints_metadata.npz

Output:
    data/processed/keypoints_3d.npy

Run:
    python lift_3d.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

VIDEOPOSE3D_DIR = ROOT / "third_party" / "VideoPose3D"

sys.path.insert(0, str(VIDEOPOSE3D_DIR))

from common.camera import normalize_screen_coordinates
from common.model import TemporalModel

DATA_DIR = ROOT / "data" / "processed"

INPUT_2D = DATA_DIR / "keypoints_2d.npy"
METADATA_PATH = DATA_DIR / "keypoints_metadata.npz"
OUTPUT_3D = DATA_DIR / "keypoints_3d.npy"

CHECKPOINT_PATH = VIDEOPOSE3D_DIR / "checkpoint" / "pretrained_h36m_detectron_coco.bin"


# =====================================================
# 2. LOAD 2D KEYPOINTS
# =====================================================


def load_2d_keypoints():

    if not INPUT_2D.is_file():
        raise FileNotFoundError(f"2D keypoints not found: {INPUT_2D}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Video metadata not found: {METADATA_PATH}")

    poses_2d = np.load(INPUT_2D, allow_pickle=False).astype(np.float32)

    if poses_2d.ndim != 3 or poses_2d.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T, 17, 2), got {poses_2d.shape}")

    if len(poses_2d) == 0:
        raise ValueError("No 2D frames available")

    if not np.isfinite(poses_2d).all():
        raise ValueError("2D keypoints contain NaN or Inf")

    with np.load(METADATA_PATH, allow_pickle=False) as metadata:
        width = int(metadata["width"])
        height = int(metadata["height"])

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video size: {width}x{height}")

    print("[INFO] 2D shape:", poses_2d.shape)
    print("[INFO] Resolution:", width, "x", height)

    # Match the normalization used in VideoPose3D.
    poses_2d = normalize_screen_coordinates(poses_2d, w=width, h=height).astype(
        np.float32
    )

    return poses_2d


# =====================================================
# 3. LOAD PRETRAINED VIDEOPOSE3D
# =====================================================


def load_model(device):

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_pos"])
    model.eval()

    print("[SUCCESS] VideoPose3D loaded")
    print("[INFO] Receptive field:", model.receptive_field())

    return model


# =====================================================
# 4. 2D TO 3D LIFTING
# =====================================================


def lift_to_3d(model, poses_2d, device, batch_size=8):

    receptive_field = model.receptive_field()

    if receptive_field % 2 != 1:
        raise ValueError("Expected an odd receptive field")

    pad = (receptive_field - 1) // 2

    # Repeat boundary frames for temporal context.
    padded = np.pad(poses_2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")

    outputs = []

    with torch.inference_mode():
        for start in range(0, len(poses_2d), batch_size):
            end = min(start + batch_size, len(poses_2d))

            windows = np.stack(
                [padded[i : i + receptive_field] for i in range(start, end)], axis=0
            )

            inputs = torch.from_numpy(windows).to(device)

            predictions = model(inputs)

            # Expected: (B, 1, 17, 3)
            if predictions.shape[1:] != (1, 17, 3):
                raise RuntimeError(f"Unexpected output: {predictions.shape}")

            batch_3d = predictions[:, 0].cpu().numpy()

            outputs.append(batch_3d)

            print(f"[INFO] Processed {end}/{len(poses_2d)} frames", end="\r")

    poses_3d = np.concatenate(outputs, axis=0).astype(np.float32)

    # Human3.6M root joint is pelvis (index 0).
    # Express all joint positions relative to pelvis.
    poses_3d -= poses_3d[:, 0:1, :]

    if not np.isfinite(poses_3d).all():
        raise ValueError("VideoPose3D produced invalid coordinates")

    return poses_3d


# =====================================================
# 5. MAIN
# =====================================================


def main():

    print("=" * 60)
    print("RTMPOSE 2D -> VIDEOPOSE3D 3D")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Device:", device)

    poses_2d = load_2d_keypoints()

    model = load_model(device)

    poses_3d = lift_to_3d(model, poses_2d, device)

    np.save(OUTPUT_3D, poses_3d)

    print("\n[SUCCESS] 3D keypoints saved")
    print("[INFO] Output:", OUTPUT_3D)
    print("[INFO] Shape:", poses_3d.shape)

    print("\nFirst frame, pelvis:", poses_3d[0, 0])
    print("First frame, right knee:", poses_3d[0, 2])

    print("=" * 60)


if __name__ == "__main__":
    main()
