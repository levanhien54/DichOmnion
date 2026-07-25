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
         "voice_map": {}, "source_language": None}
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
         "source_language": "en"},
    )


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
