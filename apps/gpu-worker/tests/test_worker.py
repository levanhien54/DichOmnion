import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

# Tiêu chuẩn Công nghiệp: Không dùng GPU thật khi chạy Unit Test
@pytest.fixture
def mock_model_manager():
    with patch('src.main.model_manager') as mock_manager:
        # Mock trạng thái đã load
        mock_manager.is_loaded = True
        mock_manager.device = "cpu_mocked"
        
        from unittest.mock import AsyncMock
        mock_manager.process_job = AsyncMock(return_value={
            "status": "success",
            "message": "Đã mix xong nhạc nền và giọng TTS mới",
            "device_used": "cpu_mocked",
            "pipeline": ["UVR", "Whisper", "Translation", "TTS", "Mix"]
        })
        yield mock_manager

client = TestClient(app)

def test_health_no_auth_and_zero_logging_shape():
    """/health phải trả 200 mà KHÔNG cần JWT — Docker HEALTHCHECK và orchestrator
    (RunPod/Modal) probe được TRƯỚC khi worker có khoá Gateway. Và chỉ lộ cờ boolean +
    enum thiết bị: KHÔNG đường dẫn / model_id / token (Zero-Logging). Trên hộp dev
    (không CUDA, lifespan không nạp model) -> models_loaded=False, device='cpu'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Bất biến hình dạng: đúng 3 khoá vô hại, không hơn (chặn rò rỉ nội dung).
    assert set(data.keys()) == {"status", "models_loaded", "device"}
    assert data["status"] == "ok"
    assert isinstance(data["models_loaded"], bool)
    assert data["device"] in ("cuda", "cpu")


def test_gpu_worker_process_endpoint(mock_model_manager):
    """
    Kiểm tra xem Endpoint nhận Job có hoạt động chuẩn không.
    (Sử dụng Mock để tiết kiệm chi phí Test GPU).
    """
    from src.main import verify_gateway_jwt
    # Token binding: Gateway ký jobId vào token; override phải khớp job_id của payload.
    app.dependency_overrides[verify_gateway_jwt] = lambda: {
        "role": "gateway",
        "jobId": "TEST_WORKER_01",
    }

    payload = {
        "job_id": "TEST_WORKER_01",
        "audio_url": "http://fake-audio.wav",
        "target_language": "Vietnamese",
        "translation_style": "Formal",
        "segments": []
    }
    
    response = client.post("/api/worker/process", json=payload)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "TEST_WORKER_01"
    assert data["result"]["device_used"] == "cpu_mocked"

    # Kiểm tra xem Model Manager có được gọi đúng tham số không. Payload không kèm
    # voice_map/source_language nên config nhận giá trị mặc định ({} và None).
    mock_model_manager.process_job.assert_called_once_with(
        "http://fake-audio.wav",
        {"target_language": "Vietnamese", "style": "Formal", "segments": [],
         "voice_map": {}, "source_language": None, "audio_md5": ""}
    )


def test_worker_forwards_voice_map_and_source_language(mock_model_manager):
    """Ánh xạ đa giọng (voice_map) và ngôn ngữ gốc phải chảy từ payload -> config
    xuống process_job. Bản cũ không có 2 trường này nên đa giọng chết ở gateway."""
    from src.main import verify_gateway_jwt
    app.dependency_overrides[verify_gateway_jwt] = lambda: {
        "role": "gateway",
        "jobId": "TEST_WORKER_02",
    }

    payload = {
        "job_id": "TEST_WORKER_02",
        "audio_url": "http://fake-audio.wav",
        "target_language": "Vietnamese",
        "translation_style": "Casual",
        "segments": [],
        "voice_map": {"SPEAKER_01": "vi-VN-NamMinhNeural", "SPEAKER_02": "vi-VN-HoaiMyNeural"},
        "source_language": "en",
    }

    response = client.post("/api/worker/process", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    mock_model_manager.process_job.assert_called_once_with(
        "http://fake-audio.wav",
        {"target_language": "Vietnamese", "style": "Casual", "segments": [],
         "voice_map": {"SPEAKER_01": "vi-VN-NamMinhNeural", "SPEAKER_02": "vi-VN-HoaiMyNeural"},
         "source_language": "en", "audio_md5": ""},
    )


# ── Đợt 33 CC33-01: mã trạng thái phân loại retry của Gateway ──────────────────────
# Gateway (index.ts:848) coi 5xx là lỗi worker THOÁNG QUA -> RE-DISPATCH tới
# MAX_DISPATCH_ATTEMPTS lần; 4xx là lỗi XÁC ĐỊNH -> terminal, KHÔNG retry. Một lỗi toàn
# vẹn audio (md5 lệch/thiếu) là XÁC ĐỊNH: tải lại cùng object hỏng y hệt. Nếu worker trả
# 500 cho nó, Gateway retry 3 lần, mỗi lần tải lại tối đa 1 GiB — khuếch đại băng thông/độ
# trễ mà không bao giờ thành công. Worker phải trả 422 để biến nó thành terminal.

def test_process_returns_422_on_audio_integrity_failure(mock_model_manager):
    """AudioIntegrityError (md5 lệch/thiếu — lỗi XÁC ĐỊNH) -> 422, KHÔNG 500.
    422 là 4xx -> Gateway đánh FAILED terminal, không khuếch đại retry tải lại."""
    from unittest.mock import AsyncMock
    from src.audio_service import AudioIntegrityError
    from src.main import verify_gateway_jwt

    mock_model_manager.process_job = AsyncMock(
        side_effect=AudioIntegrityError("audio tải về không khớp md5 đã ký — từ chối (fail-closed).")
    )
    app.dependency_overrides[verify_gateway_jwt] = lambda: {
        "role": "gateway", "jobId": "JOB-INTEGRITY",
    }
    try:
        resp = client.post("/api/worker/process", json={
            "job_id": "JOB-INTEGRITY",
            "audio_url": "http://a.wav",
            "target_language": "Vietnamese",
            "translation_style": "Formal",
            "segments": [],
        })
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_process_returns_500_on_transient_failure(mock_model_manager):
    """Ranh giới đối xứng của CC33-01: lỗi KHÔNG xác định (rớt mạng thoáng qua -> RuntimeError
    chung) PHẢI vẫn là 500 để Gateway RETRY. Gộp nhầm lỗi thoáng qua vào 422 sẽ biến một job
    lẽ ra hồi phục được thành FAILED vĩnh viễn."""
    from unittest.mock import AsyncMock
    from src.main import verify_gateway_jwt

    mock_model_manager.process_job = AsyncMock(
        side_effect=RuntimeError("Không thể tải audio từ URL đã cung cấp.")
    )
    app.dependency_overrides[verify_gateway_jwt] = lambda: {
        "role": "gateway", "jobId": "JOB-TRANSIENT",
    }
    try:
        resp = client.post("/api/worker/process", json={
            "job_id": "JOB-TRANSIENT",
            "audio_url": "http://a.wav",
            "target_language": "Vietnamese",
            "translation_style": "Formal",
            "segments": [],
        })
        assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


def test_process_without_jwt_rejected(mock_model_manager):
    """TQG-4 (Trạm 2): Gửi thẳng job vào worker mà KHÔNG có JWT của Gateway.

    Đây là bản ASSERT-hoá của kịch bản trong tests/hacker_test.ts cũ (script demo
    chỉ in console, cần server sống ở cổng 8000, KHÔNG có assertion nào). Ở đây ta
    KHÔNG override verify_gateway_jwt nên chuỗi phòng thủ thật chạy: thiếu header
    Authorization -> HTTPBearer chặn 403 NGAY, GPU không hề bị chạm tới."""
    payload = {
        "job_id": "HACKER-JOB-9999",
        "audio_url": "http://malicious-audio.wav",
        "target_language": "English",
        "translation_style": "Casual",
    }
    # Cố tình KHÔNG gửi Authorization header.
    response = client.post("/api/worker/process", json=payload)

    # Thiếu bearer -> HTTPBearer từ chối. Tuỳ phiên bản FastAPI/Starlette mã có thể
    # là 401 (thiếu credential) hoặc 403 (Not authenticated); cả hai đều là "chặn
    # xác thực". Điều BẤT BIẾN — và là điều ta bảo vệ — là job KHÔNG bao giờ chạy.
    assert response.status_code in (401, 403)
    # Cổng chặn nằm TRƯỚC pipeline: process_job không bao giờ được gọi.
    mock_model_manager.process_job.assert_not_called()


def test_terminate_then_process_returns_423(mock_model_manager):
    """TQG-5 (Financial Kill Switch phía worker): Trạm 3 ra lệnh terminate ->
    worker bật cờ cách ly (quarantine) -> mọi job mới bị từ chối 423 TRƯỚC khi
    chạm GPU. Chứng minh trọn vẹn chuỗi enforcement: lệnh -> cờ -> từ chối."""
    from src.main import verify_gateway_jwt, WORKER_STATE
    app.dependency_overrides[verify_gateway_jwt] = lambda: {"role": "gateway"}
    try:
        term = client.post("/api/worker/terminate", json={"reason": "anomaly_too_fast"})
        assert term.status_code == 200
        assert term.json()["status"] == "quarantined"

        # Sau cách ly: job mới bị chặn ở cổng quarantine, KHÔNG tốn GPU.
        resp = client.post("/api/worker/process", json={
            "job_id": "AFTER-KILL",
            "audio_url": "http://a.wav",
            "target_language": "English",
            "translation_style": "Casual",
        })
        assert resp.status_code == 423
        mock_model_manager.process_job.assert_not_called()
    finally:
        # Trả trạng thái global về mặc định để không rò rỉ sang test khác.
        WORKER_STATE["quarantined"] = False
        app.dependency_overrides.clear()


def test_process_token_jobid_mismatch_rejected(mock_model_manager):
    """Token binding (Trạm 2): JWT hợp lệ (role gateway) NHƯNG jobId trong token khác
    job_id trong payload => 403. Gateway ký RIÊNG token cho từng jobId; nếu token bị rò
    rỉ, kẻ tấn công chỉ tái dùng được cho ĐÚNG job đã ký — không thể "mượn" token của
    job A để đẩy job/nội dung B. Cổng chặn nằm TRƯỚC pipeline: GPU không hề bị chạm tới."""
    from src.main import verify_gateway_jwt
    # Token được ký cho JOB-A, nhưng payload khai JOB-B.
    app.dependency_overrides[verify_gateway_jwt] = lambda: {
        "role": "gateway",
        "jobId": "JOB-A",
    }
    try:
        response = client.post("/api/worker/process", json={
            "job_id": "JOB-B",
            "audio_url": "http://a.wav",
            "target_language": "English",
            "translation_style": "Casual",
        })
        assert response.status_code == 403
        mock_model_manager.process_job.assert_not_called()
    finally:
        app.dependency_overrides.clear()


# ── /api/worker/download: chống directory traversal + ràng buộc loại file ─────────
# Endpoint chỉ cho Gateway (JWT) tải file KẾT QUẢ nằm TRONG thư mục temp. Đây là bề mặt
# tấn công đọc-file-tuỳ-ý nếu containment sai, nên bốn nhánh dưới phải được khoá bằng test.

def _auth_gateway():
    """Override JWT: coi như request đã có JWT Gateway hợp lệ (ta test logic đường dẫn,
    KHÔNG test lại lớp xác thực — lớp đó đã có test riêng)."""
    from src.main import verify_gateway_jwt
    app.dependency_overrides[verify_gateway_jwt] = lambda: {"role": "gateway"}


def test_download_traversal_outside_temp_rejected():
    """403: đường dẫn thoát RA NGOÀI thư mục temp (../) bị chặn TRƯỚC khi chạm đĩa.
    realpath phân giải '..' về thư mục cha của temp -> ngoài ranh giới -> Access Denied."""
    _auth_gateway()
    try:
        evil = os.path.join(tempfile.gettempdir(), "..", "outside_secret.wav")
        resp = client.get("/api/worker/download", params={"path": evil})
        assert resp.status_code == 403
        assert "temp" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_download_bad_extension_rejected():
    """403: file NẰM TRONG temp nhưng đuôi không thuộc {wav,mp3,mp4} bị từ chối — chặn
    dùng endpoint để hút file tạm nhạy cảm khác (vd .env, .txt kịch bản) khỏi temp."""
    _auth_gateway()
    try:
        p = os.path.join(tempfile.gettempdir(), "note.txt")
        resp = client.get("/api/worker/download", params={"path": p})
        assert resp.status_code == 403
        assert "Invalid file type" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_download_missing_file_returns_404():
    """404: đường dẫn hợp lệ (trong temp, đuôi .wav) nhưng file không tồn tại. Xác nhận
    thứ tự kiểm: containment + đuôi PASS rồi mới tới tồn-tại (không lộ nhầm 403/404)."""
    _auth_gateway()
    try:
        p = os.path.join(tempfile.gettempdir(), "khong_ton_tai_abc123.wav")
        if os.path.exists(p):
            os.remove(p)
        resp = client.get("/api/worker/download", params={"path": p})
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_download_valid_temp_wav_succeeds():
    """200: file .wav THẬT trong temp được trả về nguyên vẹn với media_type audio/wav.
    Đây là đường hạnh phúc Gateway dùng để proxy kết quả cho client."""
    _auth_gateway()
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        payload = b"RIFF....WAVEfmt fake-wav-bytes"
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        resp = client.get("/api/worker/download", params={"path": path})
        assert resp.status_code == 200
        assert resp.content == payload
        assert resp.headers["content-type"].startswith("audio/wav")
    finally:
        app.dependency_overrides.clear()
        if os.path.exists(path):
            os.remove(path)


def test_download_without_jwt_rejected():
    """Không có JWT Gateway -> HTTPBearer chặn (401/403) TRƯỚC mọi logic đường dẫn.
    Không override verify_gateway_jwt để chuỗi phòng thủ thật chạy."""
    resp = client.get("/api/worker/download", params={"path": "whatever.wav"})
    assert resp.status_code in (401, 403)


# ── /api/worker/upload: staging AUDIO nội bộ (chỉ Gateway) ────────────────────────
# Chỉ nhận .wav/.mp3. Đây là điểm ghi-đĩa duy nhất từ mạng vào; ràng buộc đuôi phải test.

def test_upload_bad_extension_rejected():
    """400: filename không phải .wav/.mp3 (vd .txt) bị từ chối TRƯỚC khi ghi đĩa —
    chặn đẩy payload tuỳ ý (vd script, thực thi) vào temp của worker."""
    _auth_gateway()
    try:
        resp = client.post(
            "/api/worker/upload",
            files={"file": ("evil.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Invalid audio format" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_upload_valid_wav_succeeds():
    """200: .wav hợp lệ được lưu vào temp; trả audio_path nằm TRONG temp và file có
    đúng nội dung đã upload. Dọn file tạm sau khi assert để không rác temp."""
    _auth_gateway()
    saved_path = None
    try:
        payload = b"RIFF....WAVEfmt staged-bytes"
        resp = client.post(
            "/api/worker/upload",
            files={"file": ("clip.wav", payload, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        saved_path = body["audio_path"]
        # Đường dẫn phải nằm trong temp và trỏ tới file .wav vừa ghi đúng nội dung.
        temp_real = os.path.realpath(tempfile.gettempdir())
        assert os.path.realpath(saved_path).startswith(temp_real + os.sep)
        assert saved_path.endswith(".wav")
        with open(saved_path, "rb") as f:
            assert f.read() == payload
    finally:
        app.dependency_overrides.clear()
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


def test_upload_without_jwt_rejected():
    """Không có JWT Gateway -> chặn (401/403) TRƯỚC khi ghi bất cứ byte nào vào đĩa."""
    resp = client.post(
        "/api/worker/upload",
        files={"file": ("clip.wav", b"x", "audio/wav")},
    )
    assert resp.status_code in (401, 403)
