"""Ràng buộc TOÀN VẸN NỘI DUNG cho audio tải-về (Đợt 14; ghi chú lại Đợt 30).

BỐI CẢNH Đợt 14 (vì sao có lớp md5 này): thuở đó object key R2 dùng CHUNG cho mọi job/bản
build (presigned PUT trỏ tới ĐÚNG MỘT key scalar build-time). Hai job chồng lấn -> job sau
ghi đè object -> worker của job A có thể tải nhầm audio RIÊNG TƯ của tenant B. Chữ ký ECDSA
chỉ ràng buộc TÁC GIẢ payload, không ràng buộc BYTES tại URL.

Layer 1 (đóng rò rỉ chéo tenant theo tiêu chí #2 Zero-Logging Privacy): client KÝ kèm md5
của audio; worker so md5 bytes tải-về với md5 đã ký và FAIL-CLOSED khi lệch -> biến "tải
nhầm audio tenant khác" thành TỪ CHỐI an toàn, không xử lý audio không rõ nguồn.

Layer 2 (ĐÃ TRIỂN KHAI ở Đợt 30 — Option A "Gateway ký URL mỗi job"): Gateway ký presigned
PUT/GET cho key DUY NHẤT theo (device, job) = audio/<deviceId>/<jobId>.wav, nên hai job
KHÔNG còn dùng chung object -> vector ghi-đè-chéo-tenant đã đóng ngay ở tầng key. Lớp md5
dưới đây GIỮ NGUYÊN như phòng-thủ-nhiều-tầng (defense-in-depth): dù key đã duy nhất, worker
vẫn từ chối bytes không khớp md5 đã ký. Các test này khóa chính bất biến md5 đó.

Các test khóa 3 bất biến:
  1) md5 khớp  -> trả path, ghi ĐÚNG bytes.
  2) md5 lệch  -> RuntimeError, KHÔNG ghi file tạm (không lưu bytes lạ).
  3) process_job THIẾU md5 -> từ chối job (fail-closed) TRƯỚC khi tải.
"""
import hashlib
import socket

import pytest

import src.audio_service as audio_mod
from src.audio_service import audio_service


def _fake_getaddrinfo_public(host, port, *args, **kwargs):
    # Host "công khai" giả để vượt cổng SSRF mà không chạm mạng thật.
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("104.18.0.1", port))]


def _install_fake_http(monkeypatch, content: bytes, *, chunk_size: int | None = None):
    """Giả httpx.Client.stream() trả 200 + `content`, DNS trỏ host ra IP công khai (qua cổng
    SSRF). KHÔNG mở kết nối thật. `chunk_size` != None -> phát body theo nhiều khối để mô phỏng
    streaming thật (dùng cho test bộ đếm size F7); None -> một khối (đủ cho kiểm md5)."""
    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _fake_getaddrinfo_public)

    class _Resp:
        status_code = 200
        headers: dict = {}

        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            if chunk_size is None:
                yield self._body
                return
            for i in range(0, len(self._body), chunk_size):
                yield self._body[i:i + chunk_size]

    class _Stream:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return _Resp(self._body)

        def __exit__(self, *a):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            # follow_redirects phải False (bất biến chống pivot SSRF) — giữ nguyên kỳ vọng.
            assert kwargs.get("follow_redirects") is False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            # `content` là biến tự do -> đóng gói qua closure của hàm bao (khác class body).
            return _Stream(content)

    monkeypatch.setattr(audio_mod.httpx, "Client", _Client)


def test_download_accepts_matching_md5(monkeypatch, tmp_path):
    """Bytes tải-về khớp md5 đã ký -> trả path, nội dung file ĐÚNG bytes gốc."""
    content = b"RIFF....WAVEfmt real audio bytes"
    md5 = hashlib.md5(content).hexdigest()
    _install_fake_http(monkeypatch, content)
    # Ghi vào tmp_path để test tự dọn, không rác thư mục temp hệ thống.
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))

    path = audio_service.download_audio("https://pub-abc.r2.dev/a.wav", md5)

    with open(path, "rb") as f:
        assert f.read() == content


def test_download_accepts_md5_case_and_whitespace_insensitive(monkeypatch, tmp_path):
    """md5 đã ký có hoa/thường + khoảng trắng thừa vẫn khớp (source .strip().lower())."""
    content = b"some audio payload"
    md5 = hashlib.md5(content).hexdigest()
    _install_fake_http(monkeypatch, content)
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))

    path = audio_service.download_audio("https://pub-abc.r2.dev/a.wav", f"  {md5.upper()}  ")

    with open(path, "rb") as f:
        assert f.read() == content


def test_download_rejects_mismatched_md5_and_writes_nothing(monkeypatch, tmp_path):
    """Bytes tải-về KHÔNG khớp md5 đã ký (object bị tenant khác ghi đè) -> RuntimeError,
    và TUYỆT ĐỐI không ghi file tạm (không lưu bytes lạ ra đĩa)."""
    victim_bytes = b"PRIVATE audio of another tenant"
    signed_md5 = hashlib.md5(b"what THIS job actually uploaded").hexdigest()
    _install_fake_http(monkeypatch, victim_bytes)
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))

    # mkstemp KHÔNG được gọi: kiểm md5 nằm TRƯỚC khi tạo file tạm (fail-closed sạch).
    def _boom_mkstemp(*args, **kwargs):
        raise AssertionError("mkstemp KHÔNG được gọi khi md5 lệch — không được ghi bytes lạ")

    monkeypatch.setattr(audio_mod.tempfile, "mkstemp", _boom_mkstemp)

    with pytest.raises(RuntimeError):
        audio_service.download_audio("https://pub-abc.r2.dev/a.wav", signed_md5)

    # Không có file .wav nào rơi vào thư mục tạm.
    assert list(tmp_path.glob("*.wav")) == []


# --- Đợt 18 F7: trần kích thước audio tải-về (chống thổi RAM -> OOM-kill) -----------

def test_download_rejects_oversized_body_mid_stream(monkeypatch, tmp_path):
    """Body vượt _MAX_DOWNLOAD_BYTES -> RuntimeError, hủy GIỮA chừng, KHÔNG ghi file.

    audio_url do client kiểm soát; bản cũ đọc TRỌN body vào RAM -> nhiều GB = OOM-kill worker
    (giết cả model thường trú VRAM). Bộ đếm cứng phải hủy khi tổng byte vượt trần."""
    monkeypatch.setattr(audio_mod, "_MAX_DOWNLOAD_BYTES", 16)
    big = b"a" * 64  # > 16; phát theo khối 8 -> vượt trần sau ~3 khối (24B), chưa đọc hết 64B
    _install_fake_http(monkeypatch, big, chunk_size=8)
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))

    # md5=None: đi đường test cổng SSRF/size; cổng size nằm TRƯỚC kiểm md5.
    with pytest.raises(RuntimeError):
        audio_service.download_audio("https://pub-abc.r2.dev/a.wav")

    # Vượt trần TRƯỚC mkstemp -> không có bytes lạ nào rơi ra đĩa.
    assert list(tmp_path.glob("*.wav")) == []


def test_download_rejects_oversized_content_length_early(monkeypatch, tmp_path):
    """Server KHAI Content-Length vượt trần -> từ chối SỚM, KHÔNG đọc body (iter_bytes không gọi)."""
    monkeypatch.setattr(audio_mod, "_MAX_DOWNLOAD_BYTES", 16)
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))
    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _fake_getaddrinfo_public)

    class _Resp:
        status_code = 200
        headers = {"content-length": "1000000"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            raise AssertionError("iter_bytes KHÔNG được gọi khi Content-Length đã vượt trần")

    class _Stream:
        def __enter__(self):
            return _Resp()

        def __exit__(self, *a):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("follow_redirects") is False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _Stream()

    monkeypatch.setattr(audio_mod.httpx, "Client", _Client)

    with pytest.raises(RuntimeError):
        audio_service.download_audio("https://pub-abc.r2.dev/a.wav")
    assert list(tmp_path.glob("*.wav")) == []


def test_download_accepts_body_at_cap(monkeypatch, tmp_path):
    """Body ĐÚNG bằng trần (biên) vẫn nhận — điều kiện hủy là '>' chứ không '>=' (không chặn nhầm)."""
    monkeypatch.setattr(audio_mod, "_MAX_DOWNLOAD_BYTES", 32)
    content = b"c" * 32
    md5 = hashlib.md5(content).hexdigest()
    _install_fake_http(monkeypatch, content, chunk_size=8)
    monkeypatch.setattr(audio_service, "temp_dir", str(tmp_path))

    path = audio_service.download_audio("https://pub-abc.r2.dev/a.wav", md5)

    with open(path, "rb") as f:
        assert f.read() == content


async def test_process_job_fails_closed_without_md5(monkeypatch):
    """process_job THIẾU audio_md5 (client cũ/bị tước trường) -> từ chối job TRƯỚC khi tải.
    Không có md5 = không có 'vé' ràng buộc nội dung -> không xử lý audio không rõ nguồn."""
    from src.model_manager import ModelManager

    mgr = ModelManager()
    monkeypatch.setattr(mgr, "is_loaded", True)

    # download_audio KHÔNG được chạm tới: từ chối phải xảy ra ở cổng md5 trước bước tải.
    import src.audio_service as audio_service_mod

    def _boom_download(*args, **kwargs):
        raise AssertionError("download_audio KHÔNG được gọi khi thiếu md5 (phải từ chối trước)")

    monkeypatch.setattr(audio_service_mod.audio_service, "download_audio", _boom_download)

    config = {
        "target_language": "Vietnamese",
        "style": "Formal",
        "segments": [{"text": "x"}],
        "voice_map": {},
        "source_language": "en",
        # audio_md5 CỐ TÌNH thiếu.
    }
    with pytest.raises(RuntimeError, match="md5"):
        await mgr.process_job("https://pub-abc.r2.dev/a.wav", config)
