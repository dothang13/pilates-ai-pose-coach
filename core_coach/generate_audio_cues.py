"""
Vietnamese Neural Audio Generator using Microsoft Edge Neural Voice
Generates realistic, natural human-like voice cues for Pilates AI Coach.
"""

import os
import sys
import asyncio
import edge_tts

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio_cache"))

# Voice Options:
# "vi-VN-HoaiMyNeural" (Giọng nữ MC truyền cảm, tự nhiên)
# "vi-VN-NamMinhNeural" (Giọng nam dứt khoát, chuẩn HLV thể hình)
VOICE = "vi-VN-HoaiMyNeural"

AUDIO_CUES = {
    "welcome": "Chào bạn, hệ thống AI Huấn luyện viên đã sẵn sàng!",
    "rep_good": "Rất tốt!",
    "squat_depth_error": "Chưa đủ độ sâu! Hãy hạ thấp mông hơn nữa.",
    "squat_valgus_error": "Đầu gối đang bị chụm vào trong! Hãy mở rộng gối ra.",
    "squat_back_error": "Lưng đang bị gập quá mức! Hãy thẳng ngực lên.",
    "plank_hip_error": "Bụng đang bị xệ! Hãy xiết chặt cơ bụng.",
    "plank_piking_error": "Mông đang chổng quá cao! Hãy hạ người thẳng hàng.",
    "bridge_hip_error": "Hông chưa nâng đủ cao! Hãy đẩy hông lên thẳng hàng.",
    "switch_squat": "Chuyển sang bài Squat.",
    "switch_forward_lunge": "Chuyển sang bài Lunge.",
    "switch_glute_bridge": "Chuyển sang bài Cầu mông.",
    "switch_forearm_plank": "Chuyển sang bài Plank.",
    "switch_pilates_hundred": "Chuyển sang bài Hundred.",
    "switch_bird_dog": "Chuyển sang bài Bird Dog.",
    "switch_side_leg_raise": "Chuyển sang bài Nâng chân ngang."
}

async def generate_all_cues():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    print(f"[*] Generating natural Vietnamese Neural voice cues using '{VOICE}'...")
    
    for key, text in AUDIO_CUES.items():
        file_path = os.path.join(AUDIO_DIR, f"{key}.mp3")
        communicate = edge_tts.Communicate(text, VOICE, rate="+10%")
        await communicate.save(file_path)
        print(f"    [+] Saved: {key}.mp3 -> '{text}'")

    print("\n[+] All voice files generated successfully in audio_cache/!")

if __name__ == "__main__":
    asyncio.run(generate_all_cues())
