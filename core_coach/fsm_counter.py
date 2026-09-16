"""
Finite State Machine (FSM) & Rep Counter / Hold Timer
Tracks repetition phases and hold durations for fitness and Pilates exercises.
"""

from enum import Enum
from typing import Dict, Any, Tuple

class ExerciseState(Enum):
    START = "START"               # Initial neutral posture
    DESCENDING = "DESCENDING"     # Lowering phase (eccentric)
    BOTTOM = "BOTTOM"             # Lowest / inflection point (peak depth)
    ASCENDING = "ASCENDING"       # Pushing back up (concentric)
    COMPLETED = "COMPLETED"       # Returned to starting posture (1 rep finished)

class RepetitionFSM:
    """
    Finite State Machine for Repetition-based exercises (e.g. Squat, Lunge, Glute Bridge).
    Uses primary joint angle thresholds and temporal hysteresis to avoid false counts.
    """
    def __init__(
        self,
        start_threshold: float = 155.0,    # Angle for neutral / standing position
        bottom_threshold: float = 95.0,     # Angle to reach bottom phase
        min_rep_duration: float = 1.0       # Minimum seconds required for a valid rep
    ):
        self.start_threshold = start_threshold
        self.bottom_threshold = bottom_threshold
        self.min_rep_duration = min_rep_duration

        self.current_state = ExerciseState.START
        self.rep_count = 0
        self.current_rep_start_time = None
        self.lowest_angle = 180.0
        self.reached_bottom = False

    def update(self, angle: float, timestamp: float) -> Tuple[ExerciseState, int, Dict[str, Any]]:
        """
        Process current frame angle and timestamp.
        Returns: (current_state, rep_count, rep_metadata)
        """
        rep_completed = False
        meta = {
            "lowest_angle": self.lowest_angle,
            "rep_duration": 0.0,
            "is_valid_depth": False
        }

        if self.current_state == ExerciseState.START:
            self.lowest_angle = 180.0
            self.reached_bottom = False
            # Transition to DESCENDING when angle drops below start threshold
            if angle < (self.start_threshold - 10.0):
                self.current_state = ExerciseState.DESCENDING
                self.current_rep_start_time = timestamp

        elif self.current_state == ExerciseState.DESCENDING:
            if angle < self.lowest_angle:
                self.lowest_angle = angle

            # If user drops deep enough, enter BOTTOM
            if angle <= self.bottom_threshold:
                self.current_state = ExerciseState.BOTTOM
                self.reached_bottom = True
            # If user started descending but suddenly goes back up without reaching depth
            elif angle > (self.start_threshold - 5.0):
                self.current_state = ExerciseState.START

        elif self.current_state == ExerciseState.BOTTOM:
            if angle < self.lowest_angle:
                self.lowest_angle = angle

            # When angle starts increasing again
            if angle > (self.bottom_threshold + 10.0):
                self.current_state = ExerciseState.ASCENDING

        elif self.current_state == ExerciseState.ASCENDING:
            # Check if returned to starting neutral position
            if angle >= (self.start_threshold - 5.0):
                duration = timestamp - (self.current_rep_start_time or timestamp)
                # Validate rep duration to avoid twitch / bounce false positives
                if duration >= self.min_rep_duration and self.reached_bottom:
                    self.rep_count += 1
                    rep_completed = True

                meta["rep_duration"] = duration
                meta["is_valid_depth"] = self.reached_bottom
                self.current_state = ExerciseState.START
                self.lowest_angle = 180.0
                self.reached_bottom = False

        return self.current_state, self.rep_count, meta

    def reset(self):
        self.current_state = ExerciseState.START
        self.rep_count = 0
        self.current_rep_start_time = None
        self.lowest_angle = 180.0
        self.reached_bottom = False


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
