import os
import edge_tts
import tempfile
import logging
import httpx

logger = logging.getLogger("omnivoice.tts")
logger.setLevel(logging.WARNING)


def _cloud_tts_allowed() -> bool:
    """Cổng opt-in cho TTS ĐÁM MÂY (edge-tts).

    QUYỀN RIÊNG TƯ (Zero-Logging): edge-tts đọc văn bản qua dịch vụ đám mây của Microsoft,
    nghĩa là VĂN BẢN ĐÃ DỊCH của người dùng bị gửi RA NGOÀI (exfiltration nội dung). Vì vậy
    đường cloud này MẶC ĐỊNH TẮT (fail-closed); vận hành viên phải đặt tường minh
    OMNIVOICE_ALLOW_CLOUD_TTS (chấp nhận đánh đổi riêng tư, ví dụ dev/test) mới bật.
    Production giữ mọi thứ CỤC BỘ bằng GPT-SoVITS (voice='gpt-sovits')."""
    return os.environ.get("OMNIVOICE_ALLOW_CLOUD_TTS", "").strip().lower() in ("1", "true", "yes", "on")


class TTSService:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        # Địa chỉ server voice-cloning cục bộ (GPT-SoVITS). Cấu hình qua ENV để
        # triển khai thật có thể trỏ tới host/port khác; mặc định loopback an toàn.
        self.gpt_sovits_url = os.environ.get("GPT_SOVITS_URL", "http://127.0.0.1:9880")
        
    async def synthesize(
        self,
        text: str,
        voice: str = "vi-VN-HoaiMyNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """
        Sinh giọng nói từ văn bản.
        Nếu voice = 'gpt-sovits', gọi API cục bộ của GPT-SoVITS (yêu cầu bật server).
        Nếu lỗi hoặc không phải gpt-sovits, Fallback về edge-tts.

        rate/volume/pitch: điều biến ngữ điệu theo cảm xúc (emotion) mà LLM gán,
        để tag EMOTION không chỉ để trưng bày mà thực sự tác động lên âm thanh.
        """
        if not text:
            return ""
            
        fd, path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        os.close(fd)
        
        if voice.startswith("gpt-sovits"):
            logger.info("Sử dụng GPT-SoVITS để sinh giọng nói...")
            try:
                # GPT-SoVITS API v2 parameters. Tùy biến ngôn ngữ nếu cần
                params = {
                    "text": text,
                    "text_lang": "vi" if "vi" in text.lower() else "en"
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(self.gpt_sovits_url, params=params)
                    response.raise_for_status()
                    
                    with open(path, "wb") as f:
                        f.write(response.content)
                    return path
            except httpx.ConnectError:
                # Zero-Logging: KHÔNG log URL nội bộ (lộ topo hạ tầng). Chỉ báo trạng thái.
                logger.error("Không thể kết nối đến GPT-SoVITS (kiểm tra server api_v2.py).")
                logger.info("Chuyển sang Edge-TTS fallback.")
            except Exception as e:
                # Zero-Logging: KHÔNG interpolate exception thô — httpx HTTPStatusError
                # nhúng cả URL request (?text=<kịch bản plaintext>) vào message. Chỉ log
                # tên loại lỗi, đồng nhất với audio_service/audio_engine.
                logger.error(f"Lỗi gọi GPT-SoVITS: {type(e).__name__}. Chuyển sang Edge-TTS fallback.")
        
        # Fallback hoặc mặc định Edge-TTS.
        # CỔNG RIÊNG TƯ: nếu chưa được phép dùng cloud TTS thì FAIL-CLOSED NGAY — xoá file
        # tạm .wav vừa tạo và trả "" để câu này coi như KHÔNG lồng được (process_job/mix sẽ
        # báo trung thực số câu mất tiếng). TUYỆT ĐỐI không gọi edge-tts ở đây: không một ký
        # tự bản dịch nào được gửi ra Microsoft cloud khi vận hành viên chưa opt-in.
        if not _cloud_tts_allowed():
            logger.warning(
                "TTS đám mây (edge-tts) bị KHÓA mặc định vì riêng tư — không gửi văn bản ra "
                "ngoài. Đặt OMNIVOICE_ALLOW_CLOUD_TTS=1 để bật (dev/test), hoặc dùng GPT-SoVITS cục bộ."
            )
            if os.path.exists(path):
                os.remove(path)
            return ""

        # Edge-tts chỉ hỗ trợ mp3
        if path.endswith(".wav"):
            os.remove(path)
            fd, path = tempfile.mkstemp(suffix=".mp3", dir=self.temp_dir)
            os.close(fd)
            
        fallback_voice = voice if not voice.startswith("gpt-") else "vi-VN-HoaiMyNeural"
        try:
            # Zero-Logging: KHÔNG log nội dung kịch bản (text). Chỉ log độ dài + voice.
            logger.info(f"Đang sinh giọng nói edge-tts ({len(text)} ký tự) với voice: {fallback_voice}")
            communicate = edge_tts.Communicate(
                text, fallback_voice, rate=rate, volume=volume, pitch=pitch
            )
            await communicate.save(path)
            return path
        except Exception as e:
            # Zero-Logging: chỉ log tên loại lỗi (exception có thể mang theo text/URL).
            logger.error(f"Lỗi khi sinh giọng nói edge-tts: {type(e).__name__}")
            if os.path.exists(path):
                os.remove(path)
            return ""
            
tts_service = TTSService()
