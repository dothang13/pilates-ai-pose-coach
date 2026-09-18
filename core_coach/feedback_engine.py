"""
Realtime Multi-sensory Feedback Engine
Manages Visual HUD Color Highlights & Debounced Audio/TTS Feedback.
Guarantees zero voice overlap and zero delay.
"""

import time
from typing import List, Dict, Any, Tuple
from .audio_speaker import AsyncVoiceSpeaker

class FeedbackEngine:
    """
    Handles audio cue throttling, non-blocking voice speaking, and visual status indicators.
    """
    def __init__(self, debounce_cooldown: float = 3.0, enable_voice: bool = True):
        self.debounce_cooldown = debounce_cooldown
        self.enable_voice = enable_voice
        self.last_audio_time = 0.0
        self.last_spoken_message = ""
        self.speaker = AsyncVoiceSpeaker() if enable_voice else None

    def notify_rep_completed(self, reps: int, current_time: float = None):
        """
        Triggered when a repetition is cleanly completed.
        Plays congratulatory voice with high priority (interrupting any stale error cues)
        and adds a grace period so error cues don't immediately fire while standing up.
        """
        if current_time is None:
            current_time = time.time()

        # Add 1.8 second grace period after completing rep to prevent premature error warnings
        self.last_audio_time = current_time + 1.8

        if self.speaker is not None:
            self.speaker.play_cue("rep_good", interrupt=True)

    def process_feedback(
        self,
        is_form_valid: bool,
        error_messages: List[str],
        current_phase: str,
        current_time: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate if visual alerts or audio cues should be triggered.
        Only plays error audio during active workout phases, never when standing at rest.
        """
        if current_time is None:
            current_time = time.time()

        # 1. Visual HUD Status
        if is_form_valid:
            status_color = (0, 255, 127)  # Spring Green (BGR)
            status_text = "FORM CHUAN!"
        else:
            status_color = (0, 0, 255)    # Red (BGR)
            status_text = "SAI TU THE!"

        # 2. Audio Cue with Debounce Timer & Phase Gate
        audio_trigger = None
        # Only issue vocal corrections during active exercise phases
        active_phases = ["DESCENDING", "BOTTOM", "ASCENDING", "HOLDING"]
        if not is_form_valid and len(error_messages) > 0 and current_phase in active_phases:
            primary_error = error_messages[0]

            # Check debounce timer and speaker channel status
            if (current_time - self.last_audio_time) >= self.debounce_cooldown:
                if self.speaker is not None and not self.speaker.is_busy():
                    audio_trigger = primary_error
                    self.last_audio_time = current_time
                    self.last_spoken_message = primary_error
                    self.speaker.speak(audio_trigger, interrupt=False)

        return {
            "status_color_bgr": status_color,
            "status_text": status_text,
            "audio_cue_to_speak": audio_trigger,
            "all_errors": error_messages
        }

