"""Đợt 17 F1 — vòng đời đầu ra CUỐI của worker phải BỊ CHẶN (chống rò đĩa + rò riêng tư).

process_job CỐ Ý giữ file kết quả cuối (*_final.wav / *_wm.wav do mkstemp sinh) để client
tải về, nhưng KHÔNG có đường nào xóa nó sau đó -> mỗi job để lại một file VĨNH VIỄN: phình
đĩa tới khi worker chết (khả dụng) VÀ audio LỒNG TIẾNG nhạy cảm nằm lại vô thời hạn
(Zero-Logging #2). sweep_stale_finals dọn CƠ HỘI ở đầu mỗi job: xóa đúng hai hậu tố worker
tự sinh khi cũ hơn TTL, chừa file còn trong TTL (client có thể đang tải) và MỌI file khác.

Kiểm thuần đĩa (os.utime tiêm mtime, `now` tất định) — không cần GPU/mạng. Thêm một test
wiring khẳng định process_job THỰC SỰ gọi sweep ở đầu job (không để cơ chế thành mã chết).
"""
import os

import pytest

from src.audio_engine import AudioEngine


def _mk(dir_path, name, *, age_s, now):
    """Tạo file `name` trong dir_path với mtime = now - age_s (tuổi tất định)."""
    p = os.path.join(str(dir_path), name)
    with open(p, "wb") as f:
        f.write(b"x")
    os.utime(p, (now - age_s, now - age_s))
    return p


def _fresh_engine(tmp_path):
    """AudioEngine trỏ temp_dir vào tmp_path — cô lập, không đụng temp thật của máy."""
    eng = AudioEngine()
    eng.temp_dir = str(tmp_path)
    return eng


def test_sweeps_stale_finals_keeps_fresh(tmp_path):
    """final QUÁ hạn (cả _final.wav lẫn _wm.wav) bị xóa; final CÒN trong TTL được giữ."""
    eng = _fresh_engine(tmp_path)
    now = 1_000_000.0
    ttl = 3600.0
    stale_final = _mk(tmp_path, "abc_final.wav", age_s=ttl + 10, now=now)
    stale_wm = _mk(tmp_path, "def_wm.wav", age_s=ttl + 10, now=now)
    fresh_final = _mk(tmp_path, "ghi_final.wav", age_s=ttl - 10, now=now)
    fresh_wm = _mk(tmp_path, "jkl_wm.wav", age_s=ttl - 10, now=now)

    removed = eng.sweep_stale_finals(ttl, now=now)

    assert removed == 2
    assert not os.path.exists(stale_final)
    assert not os.path.exists(stale_wm)
    # Còn trong TTL -> client có thể đang/sắp tải -> TUYỆT ĐỐI không xóa.
    assert os.path.exists(fresh_final)
    assert os.path.exists(fresh_wm)


def test_boundary_exactly_ttl_is_kept(tmp_path):
    """Đúng bằng TTL vẫn GIỮ — điều kiện xóa là '>' (chặt) chứ không phải '>='."""
    eng = _fresh_engine(tmp_path)
    now = 1_000_000.0
    ttl = 3600.0
    at_ttl = _mk(tmp_path, "edge_final.wav", age_s=ttl, now=now)

    removed = eng.sweep_stale_finals(ttl, now=now)

    assert removed == 0
    assert os.path.exists(at_ttl)


def test_never_touches_non_final_files(tmp_path):
    """File KHÔNG phải final, dù CỰC cũ, TUYỆT ĐỐI không bị đụng (chỉ khớp 2 hậu tố)."""
    eng = _fresh_engine(tmp_path)
    now = 1_000_000.0
    ttl = 3600.0
    src = _mk(tmp_path, "source_audio.wav", age_s=ttl * 100, now=now)   # audio nguồn/khác
    pre = _mk(tmp_path, "xyz_pre.wav", age_s=ttl * 100, now=now)        # temp time-stretch
    txt = _mk(tmp_path, "notes.txt", age_s=ttl * 100, now=now)          # phi-wav

    removed = eng.sweep_stale_finals(ttl, now=now)

    assert removed == 0
    assert os.path.exists(src)
    assert os.path.exists(pre)
    assert os.path.exists(txt)


def test_missing_dir_is_noop(tmp_path):
    """temp_dir không tồn tại -> trả 0, KHÔNG raise (best-effort, không hỏng job)."""
    eng = AudioEngine()
    eng.temp_dir = os.path.join(str(tmp_path), "does_not_exist")
    assert eng.sweep_stale_finals(3600.0, now=1_000_000.0) == 0


def test_default_now_uses_walltime(tmp_path):
    """Không truyền `now` -> dùng time.time(): file epoch-1970 chắc chắn quá mọi TTL dương."""
    eng = _fresh_engine(tmp_path)
    p = os.path.join(str(tmp_path), "old_final.wav")
    with open(p, "wb") as f:
        f.write(b"x")
    os.utime(p, (0, 0))  # 1970 -> cũ hơn bất kỳ ttl dương nào so với walltime hiện tại

    removed = eng.sweep_stale_finals(1.0)  # now mặc định = time.time()

    assert removed == 1
    assert not os.path.exists(p)


@pytest.mark.asyncio
async def test_process_job_invokes_sweep_at_start(monkeypatch):
    """WIRING: process_job phải GỌI sweep_stale_finals ở ĐẦU job (cơ chế không phải mã chết).

    Cho download_audio nổ NGAY sau bước sweep -> job fail sớm, nhưng sweep (đứng TRƯỚC
    download trong try) đã chạy TRƯỚC đó, nên spy phải ghi nhận được lời gọi + TTL đã parse.
    """
    import src.audio_engine as audio_engine_mod
    import src.audio_service as audio_service_mod
    from src.model_manager import ModelManager

    seen = {}

    def _spy(ttl_s, now=None):
        seen["ttl"] = ttl_s
        return 0

    monkeypatch.setattr(audio_engine_mod.audio_engine, "sweep_stale_finals", _spy)

    def _boom(url, expected_md5=None):
        raise RuntimeError("dừng ngay sau sweep")

    monkeypatch.setattr(audio_service_mod.audio_service, "download_audio", _boom)

    mgr = ModelManager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    monkeypatch.setenv("WORKER_FINAL_TTL_S", "1234")

    with pytest.raises(RuntimeError):
        await mgr.process_job(
            "http://local/a.wav",
            {"audio_md5": "deadbeef", "segments": [{"id": 1, "text": "x"}]},
        )

    assert seen.get("ttl") == 1234.0, "process_job phải gọi sweep với TTL parse từ env"
