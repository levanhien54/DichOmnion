import logging
import torch
from typing import List, Dict, Any

logger = logging.getLogger("omnivoice.asr")
logger.setLevel(logging.WARNING)

class ASRService:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self, model_size: str = "base"):
        """
        Khởi tạo mô hình Faster Whisper và ghim vào VRAM (nếu có GPU).
        """
        if self.is_loaded:
            return
            
        logger.info(f"Đang nạp mô hình Whisper ({model_size}) vào VRAM...")
        try:
            from faster_whisper import WhisperModel
            # Sử dụng CTranslate2 để ép model lên GPU. Nếu ko có GPU fallback về CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.is_loaded = True
            logger.info(f"Nạp Whisper ({model_size}) thành công trên {device}.")
        except ImportError as e:
            # Fail-closed: thiếu thư viện thì KHÔNG được coi là "đã nạp".
            # Trả về mock/giả vờ chạy là vi phạm nguyên tắc No-Fake-Success.
            raise RuntimeError(
                "faster-whisper chưa được cài đặt — không thể nạp mô hình ASR."
            ) from e
        except Exception as e:
            # Fail-closed: trọng số/model không nạp được thì để lỗi nổ ra,
            # tuyệt đối không im lặng rồi chạy pipeline rỗng. Zero-Logging: chỉ log
            # TÊN loại lỗi và raise thông điệp TĨNH (không nhúng {e}) để str() của
            # RuntimeError an toàn-theo-cấu-trúc với mọi caller bắt sau này — khớp
            # quy ước nhánh ImportError phía trên, translation_service.load_model và
            # transcribe(). `from e` vẫn giữ cause cho traceback boot của operator
            # (fail-closed, không chứa nội dung user); đây là vệ-sinh-nhất-quán,
            # KHÔNG phải khử rò rỉ traceback lúc khởi động.
            logger.error(f"Nạp Whisper thất bại: {type(e).__name__}")
            raise RuntimeError("Không thể nạp mô hình Whisper.") from e
            
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Bóc băng âm thanh, trả về danh sách các segments với cấu trúc chuẩn.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Faster Whisper model is not loaded yet!")
            
        logger.info(f"Bắt đầu bóc băng file: {audio_path}")
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            results = []
            for idx, segment in enumerate(segments):
                results.append({
                    "id": f"sub-{idx+1}",
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "duration": round(segment.end - segment.start, 2),
                    "text": segment.text.strip(),
                    "speaker": "SPEAKER_UNKNOWN" # Diarization thật sẽ cần WhisperX hoặc pyannote
                })
            
            return results
        except Exception as e:
            # Zero-Logging: exception của faster-whisper/av/ffmpeg thường nhúng đường
            # dẫn temp nội bộ (audio_path). Chỉ log tên loại; giữ nguyên object lỗi cho
            # caller tin cậy qua bare `raise` (bảo toàn traceback gốc).
            logger.error(f"Lỗi bóc băng: {type(e).__name__}")
            raise

asr_service = ASRService()
