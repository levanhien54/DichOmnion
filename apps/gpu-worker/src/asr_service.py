import logging
import torch
from typing import List, Dict, Any

logger = logging.getLogger("omnivoice.asr")
logger.setLevel(logging.WARNING)

# faster-whisper accepts ISO-639-1 language codes.  The client contract uses
# human-readable names; a bounded map lets Analyze use an explicit source hint
# without allowing arbitrary model/runtime options through the request.
_WHISPER_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "english": "en",
    "en": "en",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "vietnamese": "vi",
    "vi": "vi",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "spanish": "es",
    "es": "es",
    "portuguese": "pt",
    "pt": "pt",
    "italian": "it",
    "it": "it",
    "thai": "th",
    "th": "th",
    "indonesian": "id",
    "id": "id",
}


def whisper_language_code(value: Any) -> str | None:
    """Return a supported Whisper code, or ``None`` to retain auto-detection."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("_", "-")
    return _WHISPER_LANGUAGE_ALIASES.get(normalized)

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

    def transcribe_analyze(
        self, audio_path: str, *, language: str | None = None
    ) -> Dict[str, Any]:
        """Bóc băng cho bước ANALYZE (Human-in-the-Loop): trả về ngôn ngữ GỐC phát hiện
        được + segments KÈM tín hiệu ĐỘ TIN CẬY thô của mô hình (avg_logprob, no_speech_prob).

        Khác `transcribe` (đường render một-lượt cũ, chỉ cần text + mốc thời gian): người
        biên tập cần thấy câu nào mô hình KHÔNG chắc để soát tay. Ta SURFACE trực tiếp tín
        hiệu của faster-whisper thay vì bịa hằng số 'confidence' (No-Fake-Success). Diarization
        thật (WhisperX) tới ở S7 — tại đây mọi câu tạm gán SPEAKER_UNKNOWN (degraded)."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Faster Whisper model is not loaded yet!")

        logger.info(f"Bắt đầu bóc băng (analyze) file: {audio_path}")
        try:
            transcribe_kwargs = {"beam_size": 5}
            language_code = whisper_language_code(language)
            if language_code is not None:
                transcribe_kwargs["language"] = language_code
            segments, info = self.model.transcribe(audio_path, **transcribe_kwargs)

            results = []
            for idx, segment in enumerate(segments):
                results.append({
                    "id": f"sub-{idx+1}",
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    # Tín hiệu ĐỘ TIN CẬY thô của mô hình — surfaced verbatim, không bịa.
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "speaker": "SPEAKER_UNKNOWN",  # diarization thật ở S7
                })

            return {"language": info.language, "segments": results}
        except Exception as e:
            # Zero-Logging: chỉ log TÊN loại lỗi (đường dẫn temp có thể nằm trong message).
            logger.error(f"Lỗi bóc băng (analyze): {type(e).__name__}")
            raise

asr_service = ASRService()
