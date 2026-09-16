"""
Realtime Multi-sensory Feedback Engine
Manages Visual HUD Color Highlights & Debounced Audio/TTS Feedback.
"""

import time
from typing import List, Dict, Any, Tuple
from .audio_speaker import AsyncVoiceSpeaker

class FeedbackEngine:
    """
    Handles audio cue throttling, non-blocking voice speaking, and visual status indicators.
    """
    def __init__(self, debounce_cooldown: float = 2.5, enable_voice: bool = True):
        self.debounce_cooldown = debounce_cooldown
        self.enable_voice = enable_voice
        self.last_audio_time = 0.0
        self.last_spoken_message = ""
        self.speaker = AsyncVoiceSpeaker() if enable_voice else None

    def process_feedback(
        self,
        is_form_valid: bool,
        error_messages: List[str],
        current_phase: str,
        current_time: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate if visual alerts or audio cues should be triggered.
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

        # 2. Audio Cue with Debounce Timer
        audio_trigger = None
        if not is_form_valid and len(error_messages) > 0:
            primary_error = error_messages[0]
            # Check debounce timer
            if (current_time - self.last_audio_time) >= self.debounce_cooldown:
                audio_trigger = primary_error
                self.last_audio_time = current_time
                self.last_spoken_message = primary_error
                if self.speaker is not None:
                    self.speaker.speak(audio_trigger)

        return {
            "status_color_bgr": status_color,
            "status_text": status_text,
            "audio_cue_to_speak": audio_trigger,
            "all_errors": error_messages
        }
