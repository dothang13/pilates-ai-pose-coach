"""
Natural Studio Voice Speaker for Pilates AI Coach
Uses Pygame Mixer to play crystal-clear natural human Vietnamese voice cues (Neural TTS).
Fallback to pyttsx3 if custom audio is unavailable.
"""

import os
import queue
import threading
import pygame

AUDIO_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio_cache"))

AUDIO_MAP = {
    "welcome": os.path.join(AUDIO_CACHE_DIR, "welcome.mp3"),
    "rep_good": os.path.join(AUDIO_CACHE_DIR, "rep_good.mp3"),
    "squat_depth_error": os.path.join(AUDIO_CACHE_DIR, "squat_depth_error.mp3"),
    "squat_valgus_error": os.path.join(AUDIO_CACHE_DIR, "squat_valgus_error.mp3"),
    "squat_back_error": os.path.join(AUDIO_CACHE_DIR, "squat_back_error.mp3"),
    "plank_hip_error": os.path.join(AUDIO_CACHE_DIR, "plank_hip_error.mp3"),
    "plank_piking_error": os.path.join(AUDIO_CACHE_DIR, "plank_piking_error.mp3")
}

class AsyncVoiceSpeaker:
    """
    High-fidelity audio speaker using pre-cached Vietnamese Neural Voices.
    Completely non-blocking, zero-latency, realistic human tone.
    """
    def __init__(self):
        self.queue = queue.Queue(maxsize=3)
        self.is_running = True
        
        # Initialize pygame mixer
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.mixer_ready = True
        except Exception as e:
            self.mixer_ready = False

        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _find_audio_file(self, text: str) -> str:
        text_lower = text.lower()
        if "sâu" in text_lower or "hạ thấp" in text_lower or "depth" in text_lower:
            return AUDIO_MAP.get("squat_depth_error")
        elif "chụm" in text_lower or "mở rộng" in text_lower or "valgus" in text_lower:
            return AUDIO_MAP.get("squat_valgus_error")
        elif "lưng" in text_lower or "gập" in text_lower or "ngực" in text_lower:
            return AUDIO_MAP.get("squat_back_error")
        elif "xệ" in text_lower or "bụng" in text_lower:
            return AUDIO_MAP.get("plank_hip_error")
        elif "chổng" in text_lower or "mông" in text_lower:
            return AUDIO_MAP.get("plank_piking_error")
        elif "tốt" in text_lower or "chuẩn" in text_lower:
            return AUDIO_MAP.get("rep_good")
        elif "sẵn sàng" in text_lower or "chào" in text_lower:
            return AUDIO_MAP.get("welcome")
        return None

    def _speech_worker(self):
        while self.is_running:
            try:
                text = self.queue.get(timeout=0.2)
                if text is None:
                    break

                audio_file = self._find_audio_file(text)
                if self.mixer_ready and audio_file and os.path.exists(audio_file):
                    try:
                        sound = pygame.mixer.Sound(audio_file)
                        sound.play()
                        # Allow sound to play without blocking queue indefinitely
                    except Exception:
                        pass
                else:
                    # Fallback to pyttsx3 if no audio file matched
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                continue

    def speak(self, text: str):
        if not text:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self):
        self.is_running = False
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
