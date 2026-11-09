
"""
Test loading VideoPose3D pretrained checkpoint.

Run:
    python test_load_3d.py
"""

import sys
from pathlib import Path

import torch


# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent

VIDEOPOSE3D_DIR = (
    ROOT / "third_party" / "VideoPose3D"
)

CHECKPOINT_PATH = (
    VIDEOPOSE3D_DIR
    / "checkpoint"
    / "pretrained_h36m_detectron_coco.bin"
)

sys.path.insert(0, str(VIDEOPOSE3D_DIR))

from common.model import TemporalModel


# =====================================================
# 2. INITIALIZE MODEL
# =====================================================

def main():

    print("=" * 60)
    print("VIDEOPOSE3D - MODEL LOADING TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)
    print("[INFO] Checkpoint:", CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024
    ).to(device)

    print("[INFO] Model initialized")

    # Only load a checkpoint from a trusted source
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_pos" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_pos'"
        )

    model.load_state_dict(checkpoint["model_pos"])
    model.eval()

    print("[SUCCESS] Checkpoint loaded!")

    # =================================================
    # 3. TEST FORWARD PASS
    # =================================================

    receptive_field = model.receptive_field()

    print(
        "[INFO] Receptive field:",
        receptive_field
    )

    dummy_input = torch.zeros(
        1,
        receptive_field,
        17,
        2,
        device=device
    )

    with torch.inference_mode():
        output = model(dummy_input)

    print("[INFO] Input:", tuple(dummy_input.shape))
    print("[INFO] Output:", tuple(output.shape))

    assert output.shape == (1, 1, 17, 3)
    assert torch.isfinite(output).all()

    print("=" * 60)
    print("[SUCCESS] VIDEOPOSE3D IS READY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
