"""SSRF regression gate cho audio_service.download_audio (Đợt 13).

Worker tải audio_url do CLIENT kiểm soát; chữ ký ECDSA chỉ ràng buộc tác giả chứ không
ràng buộc nội dung URL, nên một thiết bị đã đăng ký có thể trỏ audio_url vào hạ tầng nội
bộ (metadata 169.254.169.254, loopback 127.0.0.1, RFC1918). Các test dưới khóa bất biến:
CHỈ https tới IP công khai được chấp nhận; mọi thứ khác bị TỪ CHỐI TRƯỚC khi mở kết nối
(fail-closed) và KHÔNG theo redirect.
"""
import socket

import pytest

import src.audio_service as audio_mod
from src.audio_service import _assert_safe_audio_url, _ip_is_public, audio_service


# --- Cổng thuần (không mạng): literal IP nội bộ bị chặn -----------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://127.0.0.1:9880/",                     # GPT-SoVITS nội bộ (loopback)
        "https://10.0.0.5/audio.wav",                  # RFC1918
        "https://192.168.1.10/audio.wav",              # RFC1918
        "https://172.16.0.9/audio.wav",                # RFC1918
        "https://[::1]/audio.wav",                     # loopback IPv6
        "https://0.0.0.0/audio.wav",                   # unspecified
    ],
)
def test_guard_rejects_internal_literal_ips(url):
    with pytest.raises(ValueError):
        _assert_safe_audio_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://pub-abc.r2.dev/audio.wav",   # http (không phải https)
        "file:///etc/passwd",                # scheme file
        "gopher://127.0.0.1:70/",            # scheme lạ
        "ftp://example.com/audio.wav",       # ftp
    ],
)
def test_guard_rejects_non_https_scheme(url):
    with pytest.raises(ValueError):
        _assert_safe_audio_url(url)


def test_guard_rejects_missing_host():
    with pytest.raises(ValueError):
        _assert_safe_audio_url("https:///audio.wav")


def test_guard_rejects_bad_port():
    with pytest.raises(ValueError):
        _assert_safe_audio_url("https://pub-abc.r2.dev:99999/audio.wav")


# --- Cổng với DNS giả lập: hostname phân giải ra IP nội bộ bị chặn ------------------
def _fake_getaddrinfo(addr):
    def _inner(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, port))]

    return _inner


def test_guard_rejects_hostname_resolving_to_private(monkeypatch):
    # Attacker DNS trỏ một host "đẹp" về RFC1918.
    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _fake_getaddrinfo("10.1.2.3"))
    with pytest.raises(ValueError):
        _assert_safe_audio_url("https://audio.evil.example/a.wav")


def test_guard_rejects_split_horizon_public_plus_private(monkeypatch):
    # Một bản ghi công khai + một bản ghi nội bộ -> vẫn phải từ chối.
    def _mixed(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("104.18.0.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("169.254.169.254", port)),
        ]

    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _mixed)
    with pytest.raises(ValueError):
        _assert_safe_audio_url("https://audio.split.example/a.wav")


def test_guard_accepts_public_https(monkeypatch):
    # Host R2 công khai phân giải ra IP công khai -> KHÔNG ném (đường hợp lệ vẫn qua).
    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _fake_getaddrinfo("104.18.0.1"))
    _assert_safe_audio_url("https://pub-abc.r2.dev/audio.wav")  # không raise


def test_ip_is_public_helper():
    assert _ip_is_public("104.18.0.1") is True
    assert _ip_is_public("8.8.8.8") is True
    assert _ip_is_public("169.254.169.254") is False
    assert _ip_is_public("127.0.0.1") is False
    assert _ip_is_public("10.0.0.1") is False
    assert _ip_is_public("::1") is False
    assert _ip_is_public("not-an-ip") is False  # fail-closed


def test_ip_is_public_rejects_ipv4_mapped_internal():
    """IPv4-mapped IPv6 KHÔNG lách được: dải nội bộ nhúng vẫn bị chặn (is_private/
    is_reserved bắt trên 3.11; is_loopback/is_link_local bắt trên 3.13+)."""
    assert _ip_is_public("::ffff:169.254.169.254") is False
    assert _ip_is_public("::ffff:127.0.0.1") is False
    assert _ip_is_public("::ffff:10.0.0.1") is False


@pytest.mark.parametrize(
    "url",
    [
        "https://2130706433/a.wav",   # 127.0.0.1 thập phân
        "https://0x7f000001/a.wav",    # 127.0.0.1 hex
        "https://0177.0.0.1/a.wav",    # 127.0.0.1 bát phân
        "https://127.1/a.wav",         # 127.0.0.1 rút gọn
    ],
)
def test_guard_rejects_numeric_encoded_loopback(url):
    """Phân giải-RỒI-phân-loại: dù getaddrinfo có giải mã kiểu số hay không, kết quả
    đều bị chặn (giải mã -> IP nội bộ bị phân loại chặn; không giải mã -> gaierror).
    KHÔNG dựa vào so khớp chuỗi hostname (vốn dễ bị lách bằng mã hoá số)."""
    with pytest.raises(ValueError):
        _assert_safe_audio_url(url)


# --- download_audio: từ chối đích nội bộ mà KHÔNG mở kết nối nào --------------------
def test_download_audio_blocks_internal_without_opening_connection(monkeypatch):
    """audio_url nội bộ phải bị chặn ở cổng TRƯỚC khi bất kỳ httpx.Client nào được tạo."""

    def _boom(*args, **kwargs):
        raise AssertionError("httpx.Client KHÔNG được khởi tạo cho audio_url nội bộ")

    monkeypatch.setattr(audio_mod.httpx, "Client", _boom)
    with pytest.raises(RuntimeError):
        audio_service.download_audio("https://169.254.169.254/latest/meta-data/")


def test_download_audio_rejects_redirect(monkeypatch):
    """Đích công khai nhưng trả 3xx -> từ chối, KHÔNG đi theo Location (chặn pivot)."""
    monkeypatch.setattr(audio_mod.socket, "getaddrinfo", _fake_getaddrinfo("104.18.0.1"))

    class _Resp:
        status_code = 302
        headers = {"Location": "https://169.254.169.254/"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            # Kiểm 3xx đứng TRƯỚC vòng đọc body -> iter_bytes tuyệt đối không được chạm tới.
            raise AssertionError("iter_bytes KHÔNG được gọi khi status là 3xx (không đọc body redirect)")

    class _Stream:
        def __enter__(self):
            return _Resp()

        def __exit__(self, *a):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            # follow_redirects phải False (không tự đi theo redirect).
            assert kwargs.get("follow_redirects") is False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _Stream()

    monkeypatch.setattr(audio_mod.httpx, "Client", _Client)
    with pytest.raises(RuntimeError):
        audio_service.download_audio("https://pub-abc.r2.dev/audio.wav")
