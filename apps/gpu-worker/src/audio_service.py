import hashlib
import hmac
import httpx
import ipaddress
import socket
import tempfile
import os
import logging
from urllib.parse import urlparse

logger = logging.getLogger("omnivoice.audio")
logger.setLevel(logging.WARNING)

# --- Cổng chống SSRF cho audio_url (do CLIENT kiểm soát) ---------------------------
# audio nguồn LÀ presigned URL CÔNG KHAI tới R2 (xem apps/client/src/lib/transport.ts
# và CSP client: https://*.r2.cloudflarestorage.com / *.r2.dev). URL này client tự KÝ
# bằng khóa CỦA CHÍNH NÓ, nên chữ ký chỉ ràng buộc TÁC GIẢ chứ KHÔNG ràng buộc nội dung
# URL: một thiết bị đã đăng ký (Zero-Trust coi client là không tin cậy) có thể trỏ
# audio_url vào hạ tầng nội bộ — http://169.254.169.254/... (cloud metadata),
# http://127.0.0.1:9880 (GPT-SoVITS nội bộ), RFC1918... Vì worker chạy trong mạng GPU
# riêng, một GET mù tới đó là Server-Side Request Forgery (CWE-918).
#
# Fail-closed: CHỈ cho https tới hostname phân giải ra TOÀN BỘ là IP công khai, và
# KHÔNG theo redirect (chặn pivot 3xx tới đích nội bộ). Kiến trúc hiện tại chỉ dùng R2
# công khai; object store nội bộ (nếu sau này cần) sẽ thêm allowlist tường minh — YAGNI
# lúc này. Residual đã biết: DNS rebinding (host đổi bản ghi giữa lúc kiểm và lúc httpx
# nối) — kiểm TẤT CẢ bản ghi thu hẹp khe này nhưng không đóng tuyệt đối; không "giả vờ"
# đã khử hết (No-Fake-Success).
_ALLOWED_SCHEMES = frozenset(("https",))

# --- Trần kích thước audio tải-về (Đợt 18 F7 — chống thổi RAM -> OOM-kill) ----------
# audio_url do CLIENT kiểm soát (Zero-Trust). Bản trước đọc TRỌN body vào RAM
# (response.content) mà KHÔNG chặn kích thước: một thiết bị đã đăng ký có thể trỏ audio_url
# tới một tài nguyên https CÔNG KHAI nhiều GB -> worker nuốt cả body vào RAM -> MemoryError /
# bị OOM-killer hạ NGUYÊN tiến trình (kéo theo cả model thường trú VRAM) = DoS. SSRF gate chỉ
# ràng buộc ĐÍCH chứ không ràng buộc KÍCH THƯỚC. Đọc theo chunk với bộ đếm CỨNG, hủy ngay khi
# vượt trần (fail-closed). Client tách audio 16kHz mono (~32 KB/s, xem client/src-tauri/src/
# main.rs) nên mặc định 1 GiB ~ 9 giờ thoại: quá đủ cho mọi video thực nhưng chặn payload phá
# hoại. Env-tunable để box GPU lớn (Track A) nới ra mà không sửa mã.
_MAX_DOWNLOAD_BYTES = max(
    1, int(os.environ.get("WORKER_MAX_DOWNLOAD_BYTES", str(1024 * 1024 * 1024)))
)


def _ip_is_public(addr: str) -> bool:
    """True chỉ khi addr là IP định tuyến công khai. Mọi dải nội bộ/đặc biệt -> False
    (fail-closed: parse lỗi cũng coi là KHÔNG công khai)."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (
        ip.is_private        # RFC1918 (10/8, 172.16/12, 192.168/16), fc00::/7...
        or ip.is_loopback    # 127/8, ::1
        or ip.is_link_local  # 169.254/16 (gồm 169.254.169.254 metadata), fe80::/10
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified  # 0.0.0.0, ::
    )


def _assert_safe_audio_url(audio_url: str) -> None:
    """Cổng SSRF fail-closed: ném ValueError nếu audio_url không phải https tới host
    phân giải ra IP công khai. Zero-Logging: KHÔNG nhúng URL vào thông điệp lỗi (URL
    chứa token đã ký)."""
    parsed = urlparse(audio_url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError("audio_url: scheme không được phép")
    host = parsed.hostname
    if not host:
        raise ValueError("audio_url: thiếu host")
    try:
        port = parsed.port or 443
    except ValueError as e:
        # urlparse.port ném ValueError nếu port ngoài 0..65535.
        raise ValueError("audio_url: port không hợp lệ") from e
    # Phân giải TẤT CẢ bản ghi; nếu BẤT KỲ địa chỉ nào không công khai thì từ chối (chặn
    # host "split-horizon" trả 1 IP công khai + 1 IP nội bộ). getaddrinfo với IP thô trả
    # về chính nó (không tra DNS) nên literal 169.254.169.254 / 127.0.0.1 cũng bị bắt.
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError("audio_url: không phân giải được host") from e
    addrs = {info[4][0] for info in infos}
    if not addrs:
        raise ValueError("audio_url: không phân giải được host")
    for addr in addrs:
        if not _ip_is_public(addr):
            raise ValueError("audio_url: trỏ tới địa chỉ nội bộ")


class AudioService:
    def __init__(self):
        # Tạo thư mục tạm để lưu audio nếu cần
        self.temp_dir = tempfile.gettempdir()

    def download_audio(self, audio_url: str, expected_md5: str | None = None) -> str:
        """
        Tải audio (đã tách sẵn phía client) từ URL đã ký về file tạm, trả path.
        Fail-closed: lỗi mạng/HTTP thì nổ RuntimeError — không trả file giả.
        Chống SSRF: chỉ https tới IP công khai, không theo redirect (xem cổng ở trên).

        Ràng buộc TOÀN VẸN: nếu `expected_md5` được cung cấp, bytes tải-về PHẢI khớp md5
        client đã KÝ; lệch -> từ chối (fail-closed), KHÔNG ghi file. Object key R2 dùng chung
        cho mọi job (presigned PUT chỉ trỏ 1 key) nên một job khác có thể ghi đè object; kiểm
        này biến "tải nhầm audio của tenant khác" thành từ chối an toàn. Tham số để mặc định
        None CHỈ phục vụ test cổng SSRF gọi trực tiếp; đường sản xuất (process_job) LUÔN truyền
        md5 và đã fail-closed nếu thiếu, nên không có khe bỏ qua kiểm ở runtime thật.
        """
        # Cổng SSRF TRƯỚC khi mở bất kỳ kết nối nào (fail-closed). Vì kiểm host nội bộ
        # xảy ra trước cả khi connect, không có khác biệt thời gian "host up vs down"
        # cho đích nội bộ -> đóng luôn cả timing-oracle cho mạng nội bộ.
        try:
            _assert_safe_audio_url(audio_url)
        except ValueError as e:
            # Zero-Logging: KHÔNG log URL. Chỉ log loại lỗi.
            logger.error(f"Từ chối audio_url không an toàn: {type(e).__name__}")
            raise RuntimeError("audio_url không hợp lệ hoặc không được phép.") from e

        try:
            # follow_redirects=False: một 3xx do server đích trả về có thể trỏ tới đích
            # nội bộ (pivot SSRF). Presigned R2/S3 trả 200 trực tiếp nên redirect là bất
            # thường -> từ chối, không đi theo.
            with httpx.Client(follow_redirects=False) as client:
                # F7: stream + bộ đếm CỨNG thay cho response.content (vốn đọc trọn body vô hạn
                # vào RAM). follow_redirects=False: 3xx do server đích trả có thể trỏ đích nội
                # bộ (pivot SSRF) -> từ chối, không đi theo; presigned R2/S3 trả 200 trực tiếp.
                with client.stream("GET", audio_url) as response:
                    if 300 <= response.status_code < 400:
                        raise RuntimeError("audio_url trả redirect — từ chối (fail-closed).")
                    response.raise_for_status()
                    # Từ chối SỚM nếu server KHAI Content-Length vượt trần (tiết kiệm băng thông).
                    # Header có thể thiếu/dối nên KHÔNG tin tuyệt đối — bộ đếm streaming bên dưới
                    # mới là chốt thực thi cứng.
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            over = int(declared) > _MAX_DOWNLOAD_BYTES
                        except ValueError:
                            over = False  # header rác -> để bộ đếm streaming quyết
                        if over:
                            raise RuntimeError("audio vượt trần kích thước — từ chối (fail-closed).")
                    buf = bytearray()
                    for chunk in response.iter_bytes():
                        buf += chunk
                        if len(buf) > _MAX_DOWNLOAD_BYTES:
                            # Hủy TẢI ngay giữa chừng — không giữ quá trần trong RAM.
                            raise RuntimeError("audio vượt trần kích thước — từ chối (fail-closed).")
                    content = bytes(buf)

                # Ràng buộc toàn vẹn TRƯỚC khi ghi ra đĩa: nếu bytes tải-về không khớp md5 đã
                # ký thì đây là object bị (job/tenant khác) ghi đè — từ chối, KHÔNG lưu bytes
                # lạ. So sánh hằng-thời-gian (hygiene) và không phân biệt hoa/thường (Rust
                # md5::compute cho hex thường; hexdigest() cũng thường).
                if expected_md5 is not None:
                    actual_md5 = hashlib.md5(content).hexdigest()
                    if not hmac.compare_digest(actual_md5, expected_md5.strip().lower()):
                        # Zero-Logging: KHÔNG log URL/md5/nội dung. Chỉ nêu bản chất sự cố để
                        # operator phân biệt "object bị ghi đè" với lỗi mạng thường.
                        logger.warning("Audio md5 không khớp — nghi object bị ghi đè (fail-closed).")
                        raise RuntimeError("audio tải về không khớp md5 đã ký — từ chối (fail-closed).")

                fd, path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)
                logger.info("Đã tải thành công file audio.")
                return path
        except Exception as e:
            # Zero-Logging: không log URL (có thể chứa token đã ký). Chỉ log loại lỗi.
            logger.error(f"Lỗi tải audio: {type(e).__name__}")
            raise RuntimeError("Không thể tải audio từ URL đã cung cấp.") from e


audio_service = AudioService()
