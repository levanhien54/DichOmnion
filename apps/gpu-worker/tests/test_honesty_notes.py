"""No-Fake-Success: process_job phải BÁO TRUNG THỰC các câu bị mất tiếng.

Chạy được trên CPU: mock mọi seam nặng (tải audio, TTS, tách nền, mix, watermark)
để chỉ còn LOGIC báo cáo (notes) của process_job được thực thi thật. Không GPU,
không mạng — nên khác với acceptance E2E (GPU-gated).
"""
import os
import tempfile

from src.model_manager import ModelManager


def _mk_wav() -> str:
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return p


class _FakeTranslation:
    """translation_service giả: trả đúng danh sách segment đã dịch mà test kiểm soát."""

    def __init__(self, segments):
        self._segments = segments

    def translate_segments(self, segments, target_language, style, source_language="en"):
        return self._segments


def _patch_pipeline(monkeypatch, translated):
    import src.audio_service as audio_service_mod
    import src.audio_engine as audio_engine_mod
    import src.tts_service as tts_mod

    # Seam MẠNG: tải audio -> bản WAV rỗng cục bộ (bị bước dọn temp xoá, không sao).
    # Chữ ký nhận thêm expected_md5 (ràng buộc toàn vẹn); ở đây bỏ qua vì kiểm md5
    # thật đã được mock đi — chỉ cần process_job vượt qua cổng "thiếu md5" (config có md5).
    monkeypatch.setattr(
        audio_service_mod.audio_service,
        "download_audio",
        lambda url, expected_md5=None: _mk_wav(),
    )

    # Seam TTS: sinh giọng -> luôn trả 1 file (coi như render thành công cho câu CÓ dịch).
    async def _fake_synth(text, voice=None, rate=None, volume=None, pitch=None):
        return _mk_wav()

    monkeypatch.setattr(tts_mod.tts_service, "synthesize", _fake_synth)

    # Seam AUDIO: tách nền/mix/watermark -> no-op trả file thật, separated/watermarked=False.
    monkeypatch.setattr(audio_engine_mod.audio_engine, "extract_instrumental", lambda p: (p, False))
    monkeypatch.setattr(
        audio_engine_mod.audio_engine, "mix_audio", lambda instr, clips, ducking_db=-5.0: _mk_wav()
    )
    monkeypatch.setattr(audio_engine_mod.audio_engine, "add_watermark", lambda p: (p, False))
    monkeypatch.setattr(
        audio_engine_mod.audio_engine,
        "last_mix_stats",
        {"clips": 1, "stretched": 0, "truncated": 0},
        raising=False,
    )

    mgr = ModelManager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    monkeypatch.setattr(mgr, "device", "cpu")
    monkeypatch.setattr(mgr, "models", {"translation": _FakeTranslation(translated)})
    return mgr


def _config():
    # segments không rỗng -> process_job BỎ QUA ASR (không cần mock asr_service).
    return {
        "target_language": "Vietnamese",
        "style": "Formal",
        "segments": [{"text": "x"}],
        "voice_map": {},
        "source_language": "en",
        # process_job fail-closed nếu thiếu md5; giá trị chỉ cần khác rỗng vì
        # download_audio đã bị mock (không hash thật) trong bộ test này.
        "audio_md5": "deadbeef",
    }


async def test_missing_translation_reported_honestly(monkeypatch):
    """Câu CÓ thoại gốc nhưng bản dịch RỖNG -> phải có note trung thực (No-Fake-Success)."""
    translated = [
        {"original_text": "Hello there", "translated_text": "", "start": 0.0, "end": 2.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
        {"original_text": "Hi", "translated_text": "Xin chào", "start": 2.0, "end": 4.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
    ]
    mgr = _patch_pipeline(monkeypatch, translated)

    result = await mgr.process_job("http://local/a.wav", _config())

    assert result["status"] == "success"
    notes = result["notes"]
    # Câu mất bản dịch được báo rõ, đúng số lượng (1 câu).
    assert any("KHÔNG có bản dịch" in n for n in notes), notes
    assert any("Có 1 câu" in n for n in notes), notes
    # KHÔNG được nhầm sang note "TTS lỗi": câu rỗng-dịch không tính vào tts_attempted.
    assert not any("TTS lỗi" in n for n in notes), notes


async def test_out_of_range_dropped_clip_reported_honestly(monkeypatch):
    """NFS-MIX-OOR: clip bị bỏ vì mốc bắt đầu NẰM NGOÀI độ dài nhạc nền (pydub overlay lặng
    lẽ nuốt) -> model_manager phải phát note trung thực từ last_mix_stats['dropped_oor'].
    Không có note = No-Fake-Success (báo thành công dù người dùng mất tiếng lồng đoạn đó)."""
    import src.audio_engine as audio_engine_mod

    translated = [
        {"original_text": "Hello", "translated_text": "Xin chào", "start": 0.0, "end": 2.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
    ]
    mgr = _patch_pipeline(monkeypatch, translated)
    # mix bỏ 1/1 clip vì mốc bắt đầu ngoài phạm vi nhạc nền.
    monkeypatch.setattr(
        audio_engine_mod.audio_engine,
        "last_mix_stats",
        {"clips": 1, "stretched": 0, "truncated": 0, "dropped_oor": 1},
        raising=False,
    )

    result = await mgr.process_job("http://local/a.wav", _config())

    assert result["status"] == "success"
    notes = result["notes"]
    assert any("ngoài độ dài nhạc nền" in n for n in notes), notes


async def test_true_silence_not_counted_as_missing(monkeypatch):
    """Câu vốn RỖNG cả gốc lẫn dịch (khoảng lặng thật) KHÔNG bị tính là thiếu bản dịch."""
    translated = [
        {"original_text": "", "translated_text": "", "start": 0.0, "end": 1.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
        {"original_text": "Hi", "translated_text": "Xin chào", "start": 1.0, "end": 3.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
    ]
    mgr = _patch_pipeline(monkeypatch, translated)

    result = await mgr.process_job("http://local/a.wav", _config())

    assert result["status"] == "success"
    notes = result["notes"]
    assert not any("KHÔNG có bản dịch" in n for n in notes), notes
