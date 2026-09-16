import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core_coach.audio_speaker import AsyncVoiceSpeaker

print("=" * 60)
print("TESTING ASYNC VOICE SPEAKER (SPEAKING THROUGH LAPTOP SPEAKER)")
print("=" * 60)

speaker = AsyncVoiceSpeaker()

print("[*] Speaking: 'Chào bạn, hệ thống AI Huấn luyện viên đã sẵn sàng!'...")
speaker.speak("Chào bạn, hệ thống AI Huấn luyện viên đã sẵn sàng!")
time.sleep(3.5)

print("[*] Speaking error cue: 'Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa.'...")
speaker.speak("Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa.")
time.sleep(3.5)

print("[+] Studio neural voice speaker test finished successfully!")
speaker.stop()
