import pytest
import os
import shutil
from unittest.mock import patch, AsyncMock
from src.tts_service import tts_service
from src.audio_engine import audio_engine, AudioEngine, HAS_PYDUB

@pytest.mark.asyncio
async def test_tts_service_synthesize(monkeypatch):
    """edge-tts tạo file audio (mock để khỏi gọi mạng). CHỈ chạy đường cloud khi ĐÃ opt-in
    OMNIVOICE_ALLOW_CLOUD_TTS — test này tường minh chấp nhận đường cloud."""
    monkeypatch.setenv("OMNIVOICE_ALLOW_CLOUD_TTS", "1")
    with patch("src.tts_service.edge_tts.Communicate") as mock_comm:
        # Giả lập hàm save của Communicate
        mock_instance = mock_comm.return_value
        mock_instance.save = AsyncMock()

        path = await tts_service.synthesize("Xin chào thế giới")

        # Verify
        assert path.endswith(".mp3")
        mock_comm.assert_called_once()
        mock_instance.save.assert_called_once_with(path)


@pytest.mark.asyncio
async def test_cloud_tts_blocked_by_default_no_exfiltration(monkeypatch):
    """QUYỀN RIÊNG TƯ (Zero-Logging): MẶC ĐỊNH (không opt-in) đường edge-tts KHÔNG được gọi
    và synthesize trả "" — KHÔNG một ký tự bản dịch nào rời khỏi worker ra Microsoft cloud.
    Câu này coi như không lồng được (worker sẽ báo trung thực số câu mất tiếng)."""
    monkeypatch.delenv("OMNIVOICE_ALLOW_CLOUD_TTS", raising=False)  # mặc định: chưa opt-in
    with patch("src.tts_service.edge_tts.Communicate") as mock_comm:
        path = await tts_service.synthesize("Kịch bản tuyệt mật không được rời máy")
        assert path == ""
        mock_comm.assert_not_called()  # TUYỆT ĐỐI không gửi text ra cloud


@pytest.mark.asyncio
async def test_cloud_tts_gate_accepts_truthy_values(monkeypatch):
    """Cổng chấp nhận các giá trị bật thông dụng (1/true/yes/on), không phân biệt hoa/thường."""
    from src.tts_service import _cloud_tts_allowed

    for on in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("OMNIVOICE_ALLOW_CLOUD_TTS", on)
        assert _cloud_tts_allowed() is True
    for off in ("", "0", "false", "no", "off", "  "):
        monkeypatch.setenv("OMNIVOICE_ALLOW_CLOUD_TTS", off)
        assert _cloud_tts_allowed() is False
        
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


@pytest.mark.skipif(
    not HAS_PYDUB or shutil.which("ffmpeg") is None,
    reason="cần pydub + ffmpeg trên PATH để mix",
)
def test_mix_reports_out_of_range_dropped_clip(tmp_path):
    """NFS-MIX-OOR: clip có mốc bắt đầu NẰM NGOÀI độ dài nhạc nền bị pydub overlay lặng
    lẽ nuốt — mix_audio phải PHÁT HIỆN, bỏ sạch, và báo trung thực qua
    last_mix_stats['dropped_oor'] (nếu không => No-Fake-Success: người dùng mất tiếng
    lồng của đoạn đó mà pipeline vẫn báo thành công). Clip TRONG phạm vi vẫn được đặt."""
    from pydub import AudioSegment

    bg = AudioSegment.silent(duration=2000)  # nhạc nền chỉ dài 2s
    bg_path = str(tmp_path / "bg.wav")
    bg.export(bg_path, format="wav")

    clip = AudioSegment.silent(duration=500)
    clip_path = str(tmp_path / "clip.wav")
    clip.export(clip_path, format="wav")

    out = audio_engine.mix_audio(
        bg_path,
        [
            {"start": 0.0, "end": 0.5, "audio_path": clip_path},   # trong phạm vi -> đặt
            {"start": 5.0, "end": 5.5, "audio_path": clip_path},   # start 5000ms >= 2000ms -> bỏ
        ],
    )
    assert os.path.exists(out)
    assert audio_engine.last_mix_stats["dropped_oor"] == 1
    assert audio_engine.last_mix_stats["clips"] == 2
    os.remove(out)


def test_mix_no_dropped_when_all_in_range(tmp_path):
    """Ranh giới đặt ĐÚNG chỗ: mọi clip nằm trong phạm vi -> dropped_oor == 0 (không
    báo nhầm)."""
    if not HAS_PYDUB or shutil.which("ffmpeg") is None:
        pytest.skip("cần pydub + ffmpeg trên PATH để mix")
    from pydub import AudioSegment

    bg = AudioSegment.silent(duration=10_000)
    bg_path = str(tmp_path / "bg.wav")
    bg.export(bg_path, format="wav")

    clip = AudioSegment.silent(duration=500)
    clip_path = str(tmp_path / "clip.wav")
    clip.export(clip_path, format="wav")

    out = audio_engine.mix_audio(
        bg_path, [{"start": 1.0, "end": 1.5, "audio_path": clip_path}]
    )
    assert os.path.exists(out)
    assert audio_engine.last_mix_stats["dropped_oor"] == 0
    os.remove(out)


@pytest.mark.skipif(
    not HAS_PYDUB or shutil.which("ffmpeg") is None,
    reason="cần pydub + ffmpeg trên PATH để mix + đo ducking",
)
def test_mix_ducks_background_under_tts_by_ducking_db(tmp_path):
    """G5/Auto-Ducking: nhạc nền TRÙNG thời điểm có TTS phải bị HẠ đúng `ducking_db`.

    Thiết kế để cô lập ĐÚNG bước ducking (không lẫn tín hiệu TTS):
      - Nền là TÔNG thật (Sine 440Hz) -> dBFS đo được và ĐỀU trên toàn đoạn.
      - Clip TTS IM LẶNG -> overlay(position) cộng 0 tín hiệu, nên vùng 'during'
        chỉ phản ánh phần nền ĐÃ ducking, không bị TTS làm sai số đo.
    Vùng giữa (1.0-2.0s) phải thấp hơn hai vùng nền nguyên (trước/sau) xấp xỉ |ducking_db|.
    """
    from pydub import AudioSegment
    from pydub.generators import Sine

    ducking_db = -10.0
    # Nền tông 3s, chừa headroom (-3 dBFS đỉnh) để không đụng trần khi đo.
    bg = Sine(440).to_audio_segment(duration=3000, volume=-3.0)
    bg_path = str(tmp_path / "bg_tone.wav")
    bg.export(bg_path, format="wav")

    # TTS im lặng đúng 1s (khớp end-start=1.0s -> không co giãn, không cắt).
    clip = AudioSegment.silent(duration=1000)
    clip_path = str(tmp_path / "clip_silent.wav")
    clip.export(clip_path, format="wav")

    out = audio_engine.mix_audio(
        bg_path,
        [{"start": 1.0, "end": 2.0, "audio_path": clip_path}],
        ducking_db=ducking_db,
    )
    assert os.path.exists(out)

    final = AudioSegment.from_wav(out)
    before = final[0:1000].dBFS      # nền nguyên
    during = final[1000:2000].dBFS   # nền đã ducking + TTS im lặng
    after = final[2000:3000].dBFS    # nền nguyên

    # Vùng ducking phải thấp hơn RÕ so với hai vùng nền nguyên (không phải nhiễu).
    assert during < before - 5.0
    assert during < after - 5.0
    # Và bám sát giá trị danh nghĩa (nới ±3 dB cho sai số khung/lấy mẫu ở biên).
    assert abs((during - before) - ducking_db) <= 3.0
    os.remove(out)
