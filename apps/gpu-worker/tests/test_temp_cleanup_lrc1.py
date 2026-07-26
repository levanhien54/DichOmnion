"""Đợt 15/LRC1 — process_job phải DỌN mọi temp trung gian trên MỌI đường thoát.

Bản cũ đặt bước dọn temp ở CUỐI đường thành công (không try/finally). Khi một bước SAU
khi Demucs đã tách nền nổ (vd mix_audio fail-closed / watermark lỗi / TTS server rớt giữa
chừng), các stem trung gian rớt lại trong thư mục temp — trong đó có:
  * vocals.wav — GIỌNG GỐC đã bóc tách (nhạy cảm nhất),
  * audio nguồn đã tải về,
  * các clip TTS đã render.
Caller ở main.py chỉ bắt lỗi rồi trả HTTP 500, KHÔNG dọn temp. Vậy rác nhạy cảm nằm lại
trên đĩa worker -> vi phạm tiêu chí #2 (Zero-Logging Privacy) + phình đĩa.

Khắc phục: bọc thân pipeline trong try/finally, theo dõi mọi temp ngay khi tạo, dọn TẤT CẢ
trong finally; chỉ giữ lại đầu ra cuối (final_audio_path) trên nhánh thành công.

Chạy trên CPU: mock mọi seam nặng (tải/ TTS / tách nền / mix / watermark). Không GPU/mạng.
"""
import os

import pytest

from src.model_manager import ModelManager


class _FakeTranslation:
    """translation_service giả: trả đúng danh sách segment đã dịch mà test kiểm soát."""

    def __init__(self, segments):
        self._segments = segments

    def translate_segments(self, segments, target_language, style, source_language="en"):
        return self._segments


def _base_mgr(monkeypatch):
    mgr = ModelManager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    monkeypatch.setattr(mgr, "device", "cpu")
    monkeypatch.setattr(mgr, "models", {"translation": _FakeTranslation([
        {"original_text": "Hi", "translated_text": "Xin chào", "start": 0.0, "end": 2.0,
         "speaker_id": "SPEAKER_00", "emotion": "NEUTRAL"},
    ])})
    return mgr


def _config():
    # segments không rỗng -> BỎ QUA ASR; audio_md5 khác rỗng -> vượt cổng fail-closed md5.
    return {
        "target_language": "Vietnamese", "style": "Formal",
        "segments": [{"text": "x"}], "voice_map": {}, "source_language": "en",
        "audio_md5": "deadbeef",
    }


async def test_temp_cleaned_when_pipeline_raises_after_separation(monkeypatch, tmp_path):
    """mix_audio NỔ sau khi Demucs đã tách -> vocals.wav (giọng gốc), audio nguồn và clip
    TTS phải bị dọn sạch; thư mục job Demucs bị rmtree. Lỗi vẫn nổi lên caller."""
    import src.audio_service as audio_service_mod
    import src.audio_engine as audio_engine_mod
    import src.tts_service as tts_mod

    # --- Dựng file THẬT trong tmp_path để kiểm chính xác việc xoá ---
    downloaded = tmp_path / "source_audio.wav"
    downloaded.write_bytes(b"SOURCE AUDIO (nhay cam)")

    demucs_dir = tmp_path / "demucs_out" / "htdemucs" / "source_audio"
    demucs_dir.mkdir(parents=True)
    no_vocals = demucs_dir / "no_vocals.wav"   # instrumental_path
    vocals = demucs_dir / "vocals.wav"         # GIỌNG GỐC — nhạy cảm nhất
    no_vocals.write_bytes(b"instrumental")
    vocals.write_bytes(b"ORIGINAL VOICE (nhay cam)")

    tts_clip = tmp_path / "tts_clip.wav"
    tts_clip.write_bytes(b"tts clip")

    monkeypatch.setattr(
        audio_service_mod.audio_service, "download_audio",
        lambda url, expected_md5=None: str(downloaded),
    )

    async def _fake_synth(text, voice=None, rate=None, volume=None, pitch=None):
        return str(tts_clip)

    monkeypatch.setattr(tts_mod.tts_service, "synthesize", _fake_synth)

    # Tách THÀNH CÔNG -> trả no_vocals.wav + separated=True (mở nhánh demucs_job_dir/vocals).
    monkeypatch.setattr(
        audio_engine_mod.audio_engine, "extract_instrumental",
        lambda p: (str(no_vocals), True),
    )

    # mix_audio NỔ giữa pipeline SAU khi đã tách nền (mô phỏng fail-closed thật).
    def _boom_mix(instr, clips, ducking_db=-5.0):
        raise RuntimeError("Mix am thanh that bai (mo phong)")

    monkeypatch.setattr(audio_engine_mod.audio_engine, "mix_audio", _boom_mix)

    mgr = _base_mgr(monkeypatch)

    # Lỗi mix phải NỔI lên caller (main.py trả 500) — KHÔNG bị nuốt.
    with pytest.raises(RuntimeError):
        await mgr.process_job("http://local/a.wav", _config())

    # Bất biến LRC1: mọi temp trung gian bị dọn dù pipeline nổ.
    assert not downloaded.exists(), "audio nguồn tải-về phải bị xoá"
    assert not tts_clip.exists(), "clip TTS phải bị xoá"
    assert not vocals.exists(), "stem GIỌNG GỐC vocals.wav phải bị xoá (nhạy cảm nhất)"
    assert not no_vocals.exists(), "stem no_vocals phải bị xoá"
    assert not demucs_dir.exists(), "thư mục job Demucs phải bị rmtree"


async def test_final_output_preserved_but_intermediates_cleaned_on_success(monkeypatch, tmp_path):
    """Nhánh THÀNH CÔNG: đầu ra cuối (dubbed_audio) được GIỮ để client tải; mọi temp trung
    gian (audio nguồn, vocals.wav, no_vocals.wav, clip TTS, bản mix trước watermark) bị dọn."""
    import src.audio_service as audio_service_mod
    import src.audio_engine as audio_engine_mod
    import src.tts_service as tts_mod

    downloaded = tmp_path / "source_audio.wav"
    downloaded.write_bytes(b"SOURCE AUDIO")

    demucs_dir = tmp_path / "demucs_out" / "htdemucs" / "source_audio"
    demucs_dir.mkdir(parents=True)
    no_vocals = demucs_dir / "no_vocals.wav"
    vocals = demucs_dir / "vocals.wav"
    no_vocals.write_bytes(b"instrumental")
    vocals.write_bytes(b"ORIGINAL VOICE")

    tts_clip = tmp_path / "tts_clip.wav"
    tts_clip.write_bytes(b"tts clip")

    mixed = tmp_path / "mixed.wav"        # bản mix trước watermark (phải bị dọn)
    final = tmp_path / "final_dubbed.wav" # đầu ra cuối (phải được GIỮ)

    monkeypatch.setattr(
        audio_service_mod.audio_service, "download_audio",
        lambda url, expected_md5=None: str(downloaded),
    )

    async def _fake_synth(text, voice=None, rate=None, volume=None, pitch=None):
        return str(tts_clip)

    monkeypatch.setattr(tts_mod.tts_service, "synthesize", _fake_synth)
    monkeypatch.setattr(
        audio_engine_mod.audio_engine, "extract_instrumental",
        lambda p: (str(no_vocals), True),
    )

    def _fake_mix(instr, clips, ducking_db=-5.0):
        mixed.write_bytes(b"mixed")
        return str(mixed)

    monkeypatch.setattr(audio_engine_mod.audio_engine, "mix_audio", _fake_mix)

    def _fake_watermark(p):
        final.write_bytes(b"final dubbed")   # watermark tạo file MỚI khác bản mix
        return (str(final), True)

    monkeypatch.setattr(audio_engine_mod.audio_engine, "add_watermark", _fake_watermark)
    monkeypatch.setattr(
        audio_engine_mod.audio_engine, "last_mix_stats",
        {"clips": 1, "stretched": 0, "truncated": 0}, raising=False,
    )

    mgr = _base_mgr(monkeypatch)

    result = await mgr.process_job("http://local/a.wav", _config())

    assert result["status"] == "success"
    assert result["dubbed_audio"] == str(final)
    # Đầu ra cuối được GIỮ để client tải.
    assert final.exists(), "đầu ra cuối (dubbed_audio) TUYỆT ĐỐI không được xoá"
    # Mọi temp trung gian bị dọn.
    assert not downloaded.exists(), "audio nguồn phải bị dọn"
    assert not tts_clip.exists(), "clip TTS phải bị dọn"
    assert not vocals.exists(), "stem giọng gốc vocals.wav phải bị dọn"
    assert not no_vocals.exists(), "stem no_vocals phải bị dọn"
    assert not mixed.exists(), "bản mix trước watermark phải bị dọn (đã bị thay bằng final)"
    assert not demucs_dir.exists(), "thư mục job Demucs phải bị rmtree"


async def test_no_rmtree_when_separation_failed(monkeypatch, tmp_path):
    """Fail-closed Demucs (separated=False) -> instrumental == local_audio; demucs_job_dir
    vẫn None nên KHÔNG rmtree gốc temp (chỉ os.remove các file lẻ). Chống xoá nhầm cả thư mục."""
    import src.audio_service as audio_service_mod
    import src.audio_engine as audio_engine_mod
    import src.tts_service as tts_mod

    # local_audio nằm TRỰC TIẾP trong tmp_path (giả lập thư mục temp dùng chung).
    downloaded = tmp_path / "source_audio.wav"
    downloaded.write_bytes(b"SOURCE AUDIO")
    sibling = tmp_path / "unrelated_other_job.wav"   # file "của job khác" cùng thư mục temp
    sibling.write_bytes(b"OTHER JOB DATA")

    tts_clip = tmp_path / "tts_clip.wav"
    tts_clip.write_bytes(b"tts clip")

    monkeypatch.setattr(
        audio_service_mod.audio_service, "download_audio",
        lambda url, expected_md5=None: str(downloaded),
    )

    async def _fake_synth(text, voice=None, rate=None, volume=None, pitch=None):
        return str(tts_clip)

    monkeypatch.setattr(tts_mod.tts_service, "synthesize", _fake_synth)
    # separated=False -> trả CHÍNH local_audio_path (fail-closed, không tách được).
    monkeypatch.setattr(
        audio_engine_mod.audio_engine, "extract_instrumental",
        lambda p: (p, False),
    )

    def _boom_mix(instr, clips, ducking_db=-5.0):
        raise RuntimeError("Mix that bai")

    monkeypatch.setattr(audio_engine_mod.audio_engine, "mix_audio", _boom_mix)

    mgr = _base_mgr(monkeypatch)

    with pytest.raises(RuntimeError):
        await mgr.process_job("http://local/a.wav", _config())

    # File của job này bị dọn...
    assert not downloaded.exists(), "audio nguồn phải bị dọn"
    assert not tts_clip.exists(), "clip TTS phải bị dọn"
    # ...nhưng thư mục temp KHÔNG bị rmtree: file không liên quan phải còn nguyên.
    assert sibling.exists(), "KHÔNG được rmtree gốc temp khi không tách nền (xoá nhầm job khác)"
