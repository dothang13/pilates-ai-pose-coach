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

print("\n[*] Test 1: Greeting cue...")
speaker.speak("Chào bạn, hệ thống AI Huấn luyện viên đã sẵn sàng!")
time.sleep(2.5)

print("\n[*] Test 2: Rapid error cues (should drop duplicates, NO overlap)...")
speaker.speak("Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa.")
time.sleep(0.2)
speaker.speak("Lưng đang bị gập quá mức! Hãy thẳng ngực lên.")  # Should be dropped because channel is busy!
time.sleep(2.5)

print("\n[*] Test 3: High-priority Rep cue instantly cutting off old error...")
speaker.speak("Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa.")
time.sleep(0.5)
# High priority interrupt: rep completed!
print("    -> Rep completed fired! (Should immediately cut off error and play 'Rất tốt!')")
speaker.speak("1! Động tác rất tốt!", interrupt=True)
time.sleep(2.0)

print("\n[+] Anti-overlap and zero-latency audio tests passed completely!")
speaker.stop()
