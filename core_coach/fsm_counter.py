"""
Finite State Machine (FSM) & Rep Counter / Hold Timer
Tracks repetition phases and hold durations for fitness and Pilates exercises.
"""

from enum import Enum
from typing import Dict, Any, Tuple

class ExerciseState(Enum):
    START = "START"               # Standing upright in ready position
    DESCENDING = "DESCENDING"     # Lowering phase (eccentric)
    BOTTOM = "BOTTOM"             # Lowest / inflection point (peak depth)
    ASCENDING = "ASCENDING"       # Pushing back up (concentric)
    COMPLETED = "COMPLETED"       # Returned to starting posture (1 rep finished)

class RepetitionFSM:
    """
    Finite State Machine for Repetition-based exercises (e.g. Squat, Lunge, Glute Bridge).
    Features:
      - Natural Standing Range (>= 135 deg): Instantly active without rigid calibration.
      - Shallow Rep Detection: Detects early reverse without depth and flags depth_error.
      - Seated Protection: Sitting for > 3s safely pauses rep cycle and quiets audio.
    """
    def __init__(
        self,
        start_threshold: float = 140.0,    # Natural standing knee angle (relaxed)
        bottom_threshold: float = 105.0,   # Realistic depth angle for standard fitness
        min_rep_duration: float = 0.45,    # Minimum duration to count clean rep
        max_bottom_hold: float = 3.0       # Hold time threshold to detect sitting down
    ):
        self.start_threshold = start_threshold
        self.bottom_threshold = bottom_threshold
        self.min_rep_duration = min_rep_duration
        self.max_bottom_hold = max_bottom_hold

        self.current_state = ExerciseState.START
        self.rep_count = 0
        self.current_rep_start_time = None
        self.bottom_start_time = None
        self.lowest_angle = 180.0
        self.reached_bottom = False
        self.is_seated = False

    def update(self, angle: float, timestamp: float) -> Tuple[ExerciseState, int, Dict[str, Any]]:
        """
        Process current frame angle and timestamp.
        Returns: (current_state, rep_count, rep_metadata)
        """
        meta = {
            "lowest_angle": self.lowest_angle,
            "rep_duration": 0.0,
            "is_valid_depth": self.reached_bottom,
            "depth_error": False,
            "rep_completed": False,
            "is_seated": self.is_seated
        }

        # 1. START: Standing or neutral
        if self.current_state == ExerciseState.START:
            self.reached_bottom = False
            self.bottom_start_time = None

            # If user is standing up
            if angle >= (self.start_threshold - 6.0):
                self.is_seated = False
                self.lowest_angle = angle

            # If user has been sitting at desk (< 115 deg) without moving
            if self.is_seated:
                meta["is_seated"] = True
                return self.current_state, self.rep_count, meta

            # Transition to DESCENDING when knees bend down below threshold
            if angle < (self.start_threshold - 8.0) and not self.is_seated:
                self.current_state = ExerciseState.DESCENDING
                self.current_rep_start_time = timestamp
                self.lowest_angle = angle

        # 2. DESCENDING: Moving downward
        elif self.current_state == ExerciseState.DESCENDING:
            if angle < self.lowest_angle:
                self.lowest_angle = angle

            # Case A: Good depth reached! Enter BOTTOM
            if angle <= self.bottom_threshold:
                self.current_state = ExerciseState.BOTTOM
                self.reached_bottom = True
                self.bottom_start_time = timestamp

            # Case B: SHALLOW SQUAT! User stopped descending early and started pushing up!
            # e.g. lowest angle stopped at 120 deg, and angle is now rising back up (+7 deg)
            elif angle > (self.lowest_angle + 7.0):
                self.reached_bottom = False
                self.current_state = ExerciseState.ASCENDING
                meta["depth_error"] = True

        # 3. BOTTOM: Peak depth phase
        elif self.current_state == ExerciseState.BOTTOM:
            if angle < self.lowest_angle:
                self.lowest_angle = angle

            if self.bottom_start_time is None:
                self.bottom_start_time = timestamp

            # Seated detection: If resting stationary at bottom for > max_bottom_hold
            if (timestamp - self.bottom_start_time) > self.max_bottom_hold:
                self.current_state = ExerciseState.START
                self.is_seated = True
                self.reached_bottom = False
                self.lowest_angle = 180.0
                meta["is_seated"] = True
                return self.current_state, self.rep_count, meta

            # When pushing back up from bottom
            if angle > (self.bottom_threshold + 6.0):
                self.current_state = ExerciseState.ASCENDING

        # 4. ASCENDING: Returning to standing
        elif self.current_state == ExerciseState.ASCENDING:
            # If user turned back up prematurely, keep signaling depth_error
            if not self.reached_bottom:
                meta["depth_error"] = True

            # When returned to upright posture
            if angle >= (self.start_threshold - 8.0):
                duration = timestamp - (self.current_rep_start_time or timestamp)

                # Only count rep if bottom was genuinely achieved
                if self.reached_bottom and duration >= self.min_rep_duration:
                    self.rep_count += 1
                    meta["rep_completed"] = True
                elif not self.reached_bottom:
                    # Explicit shallow rep warning flag at completion
                    meta["depth_error"] = True

                meta["rep_duration"] = duration
                meta["is_valid_depth"] = self.reached_bottom
                meta["lowest_angle"] = self.lowest_angle

                self.current_state = ExerciseState.START
                self.lowest_angle = 180.0
                self.reached_bottom = False
                self.bottom_start_time = None

        meta["lowest_angle"] = self.lowest_angle
        meta["is_valid_depth"] = self.reached_bottom
        meta["is_seated"] = self.is_seated
        return self.current_state, self.rep_count, meta

    def reset(self):
        self.current_state = ExerciseState.START
        self.rep_count = 0
        self.current_rep_start_time = None
        self.bottom_start_time = None
        self.lowest_angle = 180.0
        self.reached_bottom = False
        self.is_seated = False


class HoldTimer:
    """
    Timer for isometric exercises (e.g. Forearm Plank, Pilates Hundred).
    Accumulates hold duration only when user maintains proper form.
    """
    def __init__(self):
        self.total_hold_time = 0.0
        self.last_valid_time = None
        self.is_holding = False

    def update(self, is_form_valid: bool, timestamp: float) -> float:
        """
        Update timer: increments elapsed time if is_form_valid is True.
        """
        if is_form_valid:
            if self.last_valid_time is not None:
                dt = timestamp - self.last_valid_time
                if 0.0 < dt < 0.5:  # ensure continuity without large pauses
                    self.total_hold_time += dt
            self.last_valid_time = timestamp
            self.is_holding = True
        else:
            self.last_valid_time = None
            self.is_holding = False

        return self.total_hold_time

    def reset(self):
        self.total_hold_time = 0.0
        self.last_valid_time = None
        self.is_holding = False
