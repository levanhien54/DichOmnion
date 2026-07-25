import pytest
import os
import shutil
from unittest.mock import patch, AsyncMock
from src.tts_service import tts_service
from src.audio_engine import audio_engine, AudioEngine, HAS_PYDUB

@pytest.mark.asyncio
async def test_tts_service_synthesize():
    """Test xem tts_service có tạo ra file audio không (dùng mock để khỏi tốn thời gian gọi mạng)"""
    with patch("src.tts_service.edge_tts.Communicate") as mock_comm:
        # Giả lập hàm save của Communicate
        mock_instance = mock_comm.return_value
        mock_instance.save = AsyncMock()
        
        path = await tts_service.synthesize("Xin chào thế giới")
        
        # Verify
        assert path.endswith(".mp3")
        mock_comm.assert_called_once()
        mock_instance.save.assert_called_once_with(path)
        
def test_audio_engine_mix_refuses_without_pydub():
    """No-Fake-Success: thiếu pydub thì mix_audio phải TỪ CHỐI (raise), không
    được trả về file WAV câm 44 byte rồi báo đã lồng tiếng thành công."""
    original = "dummy.wav"
    clips = [
        {"start": 1.0, "end": 2.0, "audio_path": "clip1.mp3"}
    ]

    with patch("os.path.exists", return_value=True):
        with patch("src.audio_engine.HAS_PYDUB", False):
            with pytest.raises(RuntimeError):
                audio_engine.mix_audio(original, clips)


def test_atempo_chain_single_and_chained():
    """Hệ số trong [0.5, 2.0] -> một filter; ngoài khoảng -> ghép chuỗi (nhân dồn)."""
    assert AudioEngine._atempo_chain(1.5) == "atempo=1.5"
    assert AudioEngine._atempo_chain(2.0) == "atempo=2.0"
    # 3.0 = 2.0 * 1.5 (mỗi mắt xích nằm trong [0.5, 2.0])
    assert AudioEngine._atempo_chain(3.0) == "atempo=2.0,atempo=1.5"
    # 0.4 = 0.5 * 0.8
    assert AudioEngine._atempo_chain(0.4) == "atempo=0.5,atempo=0.8"


@pytest.mark.skipif(not HAS_PYDUB, reason="pydub cần thiết để dựng đoạn thử")
def test_fit_to_duration_noop_when_close_or_no_target():
    """Đã đủ khớp (trong 5%) hoặc thiếu target -> KHÔNG đụng vào (không căn giả)."""
    from pydub import AudioSegment

    seg = AudioSegment.silent(duration=1000)
    # Lệch 2% (< tolerance 5%) -> giữ nguyên.
    out, stretched = audio_engine._fit_to_duration(seg, 1020)
    assert stretched is False
    assert out is seg
    # Không có target hợp lệ -> giữ nguyên.
    out2, stretched2 = audio_engine._fit_to_duration(seg, 0)
    assert stretched2 is False
    assert out2 is seg


@pytest.mark.skipif(
    not HAS_PYDUB or shutil.which("ffmpeg") is None,
    reason="cần pydub + ffmpeg trên PATH để co giãn thật",
)
def test_fit_to_duration_actually_stretches_to_target():
    """TTS dài 2000ms, đoạn hình chỉ 1000ms -> co lại còn ~1000ms (căn lip-sync thật)."""
    from pydub import AudioSegment

    seg = AudioSegment.silent(duration=2000)
    out, stretched = audio_engine._fit_to_duration(seg, 1000)
    assert stretched is True
    # Sau khi căn, độ dài phải tiệm cận target (sai số nhỏ do đóng khung atempo).
    assert abs(len(out) - 1000) <= 100


def test_to_seconds_normalizes_numbers_and_timecodes():
    """CC-1 phòng thủ: mốc thời gian có thể tới ở dạng số, chuỗi số, hoặc timecode.
    Không parse được -> 0.0 (không nổ ValueError làm hỏng cả bản mix)."""
    ts = AudioEngine._to_seconds
    assert ts(83) == 83.0
    assert ts(83.5) == 83.5
    assert ts("83.5") == 83.5
    assert ts("00:00:00") == 0.0
    assert ts("00:01:23") == 83.0            # 1 phút 23 giây
    assert ts("01:02:03") == 3723.0          # 1h 02m 03s
    assert ts("00:00:01,500") == 1.5         # dấu phẩy mili-giây (kiểu SRT)
    assert ts("01:30") == 90.0               # MM:SS
    assert ts("") == 0.0
    assert ts(None) == 0.0
    assert ts("rác không phải số") == 0.0
    assert ts(True) == 0.0                    # bool KHÔNG bị coi là 1 giây


@pytest.mark.skipif(
    not HAS_PYDUB or shutil.which("ffmpeg") is None,
    reason="cần pydub + ffmpeg trên PATH để mix + cắt clip",
)
def test_mix_truncates_overlong_clip_to_target(tmp_path):
    """WPC-2/NFS-03: clip TTS dài hơn nhiều so với đoạn hình (cần >2x tốc độ, đã bị
    kẹp ở 2.0) phải bị CẮT về đúng độ dài đoạn hình để không tràn đè đoạn kế; và
    việc cắt được báo cáo trung thực qua last_mix_stats['truncated']."""
    from pydub import AudioSegment

    bg = AudioSegment.silent(duration=10_000)
    bg_path = str(tmp_path / "bg.wav")
    bg.export(bg_path, format="wav")

    # Clip 5000ms nhưng đoạn hình chỉ 1000ms (0.0 -> 1.0s): 5x > kẹp 2x -> vẫn dư -> cắt.
    clip = AudioSegment.silent(duration=5000)
    clip_path = str(tmp_path / "clip.wav")
    clip.export(clip_path, format="wav")

    out = audio_engine.mix_audio(
        bg_path, [{"start": 0.0, "end": 1.0, "audio_path": clip_path}]
    )
    assert os.path.exists(out)
    assert audio_engine.last_mix_stats["truncated"] == 1
    os.remove(out)
