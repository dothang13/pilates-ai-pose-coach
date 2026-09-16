"""
Core Coach Package - Exercise Understanding, Assessment & Realtime Feedback
Pilates AI Pose Coach
"""

from .smoothing import MovingAverageFilter, OneEuroFilter
from .fsm_counter import ExerciseState, RepetitionFSM, HoldTimer
from .exercise_rules import EXERCISE_CONFIGS, evaluate_form_rules
from .feedback_engine import FeedbackEngine
from .error_classifier import PostureErrorClassifier

__all__ = [
    "MovingAverageFilter",
    "OneEuroFilter",
    "ExerciseState",
    "RepetitionFSM",
    "HoldTimer",
    "EXERCISE_CONFIGS",
    "evaluate_form_rules",
    "FeedbackEngine",
    "PostureErrorClassifier"
]
