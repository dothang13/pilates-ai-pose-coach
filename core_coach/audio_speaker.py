"""
Natural Studio Voice Speaker for Pilates AI Coach
Uses Pygame Mixer to play crystal-clear natural human Vietnamese voice cues (Neural TTS).
Features:
- 0ms Latency: All audio pre-loaded directly into RAM memory.
- Anti-Overlap: Dedicated single channel (Channel 0) guarantees voices never collide.
- Smart Priority: High priority cues (rep completed, exercise switch) instantly interrupt errors.
- Zero Lag: Stale/redundant cues are dropped instead of clogging a queue.
"""

import os
import pygame
from typing import Optional

AUDIO_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio_cache"))

AUDIO_MAP = {
    "welcome": os.path.join(AUDIO_CACHE_DIR, "welcome.mp3"),
    "rep_good": os.path.join(AUDIO_CACHE_DIR, "rep_good.mp3"),
    "squat_depth_error": os.path.join(AUDIO_CACHE_DIR, "squat_depth_error.mp3"),
    "squat_valgus_error": os.path.join(AUDIO_CACHE_DIR, "squat_valgus_error.mp3"),
    "squat_back_error": os.path.join(AUDIO_CACHE_DIR, "squat_back_error.mp3"),
    "plank_hip_error": os.path.join(AUDIO_CACHE_DIR, "plank_hip_error.mp3"),
    "plank_piking_error": os.path.join(AUDIO_CACHE_DIR, "plank_piking_error.mp3"),
    "bridge_hip_error": os.path.join(AUDIO_CACHE_DIR, "bridge_hip_error.mp3"),
    "switch_squat": os.path.join(AUDIO_CACHE_DIR, "switch_squat.mp3"),
    "switch_forward_lunge": os.path.join(AUDIO_CACHE_DIR, "switch_forward_lunge.mp3"),
    "switch_glute_bridge": os.path.join(AUDIO_CACHE_DIR, "switch_glute_bridge.mp3"),
    "switch_forearm_plank": os.path.join(AUDIO_CACHE_DIR, "switch_forearm_plank.mp3"),
    "switch_pilates_hundred": os.path.join(AUDIO_CACHE_DIR, "switch_pilates_hundred.mp3"),
    "switch_bird_dog": os.path.join(AUDIO_CACHE_DIR, "switch_bird_dog.mp3"),
    "switch_side_leg_raise": os.path.join(AUDIO_CACHE_DIR, "switch_side_leg_raise.mp3")
}

class AsyncVoiceSpeaker:
    """
    High-fidelity audio speaker using pre-cached Vietnamese Neural Voices.
    Completely non-blocking, zero-latency, realistic human tone, zero overlap.
    """
    def __init__(self):
        self.mixer_ready = False
        self.sounds = {}
        self.channel = None

        # Initialize pygame mixer with low latency buffer
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.channel = pygame.mixer.Channel(0)
            self.mixer_ready = True
        except Exception as e:
            print(f"[!] Warning: Audio mixer could not be initialized: {e}")
            self.mixer_ready = False

        # Pre-load all sound files into RAM memory for instant 0ms playback
        if self.mixer_ready:
            self._preload_sounds()

    def _preload_sounds(self):
        loaded = 0
        for key, path in AUDIO_MAP.items():
            if os.path.exists(path):
                try:
                    self.sounds[key] = pygame.mixer.Sound(path)
                    loaded += 1
                except Exception as e:
                    print(f"[!] Warning: Could not load sound '{key}': {e}")
        print(f"[+] AsyncVoiceSpeaker: Pre-loaded {loaded}/{len(AUDIO_MAP)} audio cues into RAM.")

    def match_cue_key(self, text: str) -> Optional[str]:
        """Maps an incoming message text to the corresponding pre-loaded audio cue key."""
        if not text:
            return None
        text_lower = text.lower()

        # Priority 1: Exercise Switches
        if "bài squat" in text_lower or "chuyển sang squat" in text_lower:
            return "switch_squat"
        elif "lunge" in text_lower:
            return "switch_forward_lunge"
        elif "cầu mông" in text_lower or "bridge" in text_lower:
            return "switch_glute_bridge"
        elif "plank" in text_lower:
            return "switch_forearm_plank"
        elif "hundred" in text_lower:
            return "switch_pilates_hundred"
        elif "bird dog" in text_lower:
            return "switch_bird_dog"
        elif "nâng chân" in text_lower:
            return "switch_side_leg_raise"

        # Priority 2: Rep achievements & Greetings
        if "rất tốt" in text_lower or "chuẩn" in text_lower or "tốt" in text_lower:
            return "rep_good"
        elif "sẵn sàng" in text_lower or "chào" in text_lower:
            return "welcome"

        # Priority 3: Form Error Corrections
        # Back/Spine (Check first to avoid 'gập người quá sâu' matching knee depth)
        if "lưng" in text_lower or "gập" in text_lower or "ngực" in text_lower or "cong" in text_lower:
            return "squat_back_error"
        # Knee Depth
        elif "sâu" in text_lower or "hạ thấp" in text_lower or "khớp gối chưa đủ" in text_lower or "depth" in text_lower:
            return "squat_depth_error"
        # Knee Valgus
        elif "chụm" in text_lower or "mở rộng gối" in text_lower or "valgus" in text_lower:
            return "squat_valgus_error"
        # Plank Sagging
        elif "xệ" in text_lower or "bụng" in text_lower:
            return "plank_hip_error"
        # Plank Piking
        elif "chổng" in text_lower or ("mông" in text_lower and "cầu mông" not in text_lower):
            return "plank_piking_error"
        # Bridge Hip
        elif "hông chưa nâng" in text_lower or "đẩy hông" in text_lower:
            return "bridge_hip_error"

        return None

    def play_cue(self, cue_key: str, interrupt: bool = False):
        """
        Plays a pre-loaded audio cue on the dedicated channel.
        - interrupt=True: Immediately cuts off whatever was playing and plays this cue (e.g. rep finish, exercise switch).
        - interrupt=False: If coach is already speaking, drops the cue (avoids audio overlap and delay).
        """
        if not self.mixer_ready or not self.channel:
            return

        sound = self.sounds.get(cue_key)
        if not sound:
            return

        if interrupt:
            # Stop any currently playing audio immediately
            self.channel.stop()
            self.channel.play(sound)
        else:
            # Only play if channel is not currently busy speaking
            if not self.channel.get_busy():
                self.channel.play(sound)

    def speak(self, text: str, interrupt: Optional[bool] = None):
        """
        Unified speak API: matches text to audio cue and plays without delay or overlap.
        """
        if not text:
            return

        cue_key = self.match_cue_key(text)
        if not cue_key:
            return

        # By default, rep accomplishments, greetings, and switches interrupt any prior voice
        if interrupt is None:
            interrupt = cue_key in [
                "rep_good", "welcome",
                "switch_squat", "switch_forward_lunge", "switch_glute_bridge",
                "switch_forearm_plank", "switch_pilates_hundred", "switch_bird_dog", "switch_side_leg_raise"
            ]

        self.play_cue(cue_key, interrupt=interrupt)

    def is_busy(self) -> bool:
        """Returns True if the speaker is currently outputting voice."""
        return bool(self.channel and self.channel.get_busy())

    def stop(self):
        """Stops all audio playback."""
        if self.mixer_ready and self.channel:
            try:
                self.channel.stop()
            except Exception:
                pass

