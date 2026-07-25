import httpx
import tempfile
import os
import logging

logger = logging.getLogger("omnivoice.audio")
logger.setLevel(logging.WARNING)

class AudioService:
    def __init__(self):
        # Tạo thư mục tạm để lưu audio nếu cần
        self.temp_dir = tempfile.gettempdir()

    def download_audio(self, audio_url: str) -> str:
        """
        Tải audio (đã tách sẵn phía client) từ URL đã ký về file tạm, trả path.
        Fail-closed: lỗi mạng/HTTP thì nổ RuntimeError — không trả file giả.
        """
        try:
            with httpx.Client(follow_redirects=True) as client:
                response = client.get(audio_url)
                response.raise_for_status()

                fd, path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
                with os.fdopen(fd, 'wb') as f:
                    f.write(response.content)
                logger.info("Đã tải thành công file audio.")
                return path
        except Exception as e:
            # Zero-Logging: không log URL (có thể chứa token đã ký). Chỉ log loại lỗi.
            logger.error(f"Lỗi tải audio: {type(e).__name__}")
            raise RuntimeError("Không thể tải audio từ URL đã cung cấp.") from e

audio_service = AudioService()
