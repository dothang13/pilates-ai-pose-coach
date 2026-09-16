"""
Biomechanical Form Rules & Threshold Configurations for 7 Exercises
Defines correct posture thresholds and detects joint-level deviations.
"""

from typing import Dict, Any, List, Tuple

# Dictionary of configurations for all 7 standard exercises
EXERCISE_CONFIGS = {
    "squat": {
        "name": "Squat",
        "type": "REPETITION",
        "primary_angle_key": "knee_angle",
        "start_angle": 160.0,
        "target_depth_angle": 90.0,
        "min_rep_duration": 1.2,
        "rules": {
            "knee_depth": {
                "desc": "Độ sâu khớp gối",
                "check": lambda angles: angles.get("knee_angle", 180) <= 95.0,
                "error_msg": "Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa."
            },
            "torso_lean": {
                "desc": "Độ nghiêng thân người",
                "check": lambda angles: angles.get("torso_angle", 70) >= 50.0,
                "error_msg": "Gập người quá sâu! Hãy thẳng ngực lên."
            },
            "knee_valgus": {
                "desc": "Trục đầu gối",
                "check": lambda angles: angles.get("knee_distance", 0.4) >= (angles.get("ankle_distance", 0.4) * 0.85),
                "error_msg": "Đầu gối đang bị chụm vào trong! Đẩy gối ra ngoài."
            }
        }
    },
    "forward_lunge": {
        "name": "Forward Lunge",
        "type": "REPETITION",
        "primary_angle_key": "front_knee_angle",
        "start_angle": 160.0,
        "target_depth_angle": 90.0,
        "min_rep_duration": 1.2,
        "rules": {
            "front_knee_depth": {
                "desc": "Góc gối chân trước",
                "check": lambda angles: 80.0 <= angles.get("front_knee_angle", 180) <= 100.0,
                "error_msg": "Gối chân trước hạ chưa đủ góc 90 độ."
            },
            "torso_vertical": {
                "desc": "Thân người thẳng đứng",
                "check": lambda angles: angles.get("torso_angle", 80) >= 70.0,
                "error_msg": "Thân người bị chúi về phía trước."
            }
        }
    },
    "glute_bridge": {
        "name": "Glute Bridge",
        "type": "REPETITION",
        "primary_angle_key": "hip_angle",
        "start_angle": 110.0,
        "target_depth_angle": 175.0,  # Hip fully extended
        "min_rep_duration": 1.5,
        "rules": {
            "hip_extension": {
                "desc": "Nâng hông thẳng hàng",
                "check": lambda angles: angles.get("hip_angle", 120) >= 165.0,
                "error_msg": "Hông chưa nâng đủ cao! Xiết mông và đẩy hông lên."
            },
            "hyperextension": {
                "desc": "Võng thắt lưng",
                "check": lambda angles: angles.get("hip_angle", 120) <= 190.0,
                "error_msg": "Nâng hông quá cao gây võng thắt lưng!"
            }
        }
    },
    "forearm_plank": {
        "name": "Forearm Plank",
        "type": "ISOMETRIC",
        "rules": {
            "spine_alignment": {
                "desc": "Đường thẳng cột sống",
                "check": lambda angles: 165.0 <= angles.get("body_line_angle", 180) <= 190.0,
                "error_msg": "Cơ thể chưa thẳng hàng! Điều chỉnh hông ngang vai và gót chân."
            },
            "sagging_hip": {
                "desc": "Xệ bụng / võng lưng",
                "check": lambda angles: angles.get("body_line_angle", 180) < 195.0,
                "error_msg": "Bụng đang bị xệ xuống sàn! Xiết chặt cơ bụng."
            }
        }
    },
    "pilates_hundred": {
        "name": "Pilates Hundred",
        "type": "ISOMETRIC",
        "rules": {
            "leg_angle": {
                "desc": "Góc nâng chân",
                "check": lambda angles: 35.0 <= angles.get("leg_elevation_angle", 45) <= 60.0,
                "error_msg": "Góc nâng chân chưa chuẩn (giữ khoảng 45 độ)."
            },
            "chest_lift": {
                "desc": "Nâng ngực khỏi sàn",
                "check": lambda angles: angles.get("chest_lift_angle", 20) >= 15.0,
                "error_msg": "Nâng đầu và bả vai lên khỏi thảm."
            }
        }
    },
    "bird_dog": {
        "name": "Bird Dog",
        "type": "REPETITION",
        "primary_angle_key": "limb_extension",
        "start_angle": 90.0,
        "target_depth_angle": 170.0,
        "min_rep_duration": 1.5,
        "rules": {
            "straight_limbs": {
                "desc": "Duỗi thẳng tay và chân",
                "check": lambda angles: angles.get("limb_extension", 90) >= 160.0,
                "error_msg": "Duỗi thẳng hoàn toàn tay và chân đối bên."
            }
        }
    },
    "side_leg_raise": {
        "name": "Side Leg Raise",
        "type": "REPETITION",
        "primary_angle_key": "hip_abduction",
        "start_angle": 10.0,
        "target_depth_angle": 45.0,
        "min_rep_duration": 1.2,
        "rules": {
            "abduction_range": {
                "desc": "Biên độ nâng chân",
                "check": lambda angles: angles.get("hip_abduction", 0) >= 35.0,
                "error_msg": "Nâng chân cao hơn nữa (khoảng 40-45 độ)."
            }
        }
    }
}

def evaluate_form_rules(exercise_key: str, angles: Dict[str, float], phase: str = "START") -> Tuple[bool, List[str], int]:
    """
    Evaluate posture for the specified exercise based on kinematic angles and movement phase.
    Returns: (is_valid, error_messages, score_0_to_100)
    """
    cfg = EXERCISE_CONFIGS.get(exercise_key.lower())
    if not cfg:
        return True, [], 100

    # If user is just standing neutral or preparing, don't check depth errors
    if phase in ["START", "STANDBY", "WAITING_BODY"]:
        return True, [], 100

    rules = cfg.get("rules", {})
    errors = []
    total_rules = 0
    passed_rules = 0

    for r_key, r_info in rules.items():
        # Only check knee_depth if user is in BOTTOM or ASCENDING phase
        if "depth" in r_key and phase not in ["BOTTOM", "ASCENDING"]:
            continue

        total_rules += 1
        try:
            if r_info["check"](angles):
                passed_rules += 1
            else:
                errors.append(r_info["error_msg"])
        except Exception:
            pass

    score = int((passed_rules / total_rules * 100)) if total_rules > 0 else 100
    is_valid = len(errors) == 0
    return is_valid, errors, score
