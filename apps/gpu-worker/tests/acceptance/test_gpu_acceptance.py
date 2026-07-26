"""Bài kiểm THẬT — chỉ chạy trên hộp GPU (RUN_GPU_ACCEPT=1 + torch.cuda.is_available()).

Đây là những đảm bảo mà CHỈ phần cứng thật chứng minh được, và tuyệt đối KHÔNG được
giả xanh trên hộp dev CPU (No-Fake-Success + tiêu chí #6 GPU Model Residence):

  1) test_models_resident_on_cuda   — Whisper + Qwen nạp THẬT và cư trú VRAM (device=cuda).
  2) test_qwen_translate_smoke      — Qwen cục bộ sinh ĐƯỢC bản dịch (không cloud, không mock).
  3) test_process_endpoint_e2e      — POST /api/worker/process với JWT gateway THẬT (ES256)
                                       chạy trọn pipeline -> trả file audio lồng tiếng có thật.
  4) test_health_reports_residence  — /health báo models_loaded=True, device=cuda SAU khi nạp
                                       (hợp đồng readiness mà orchestrator/Docker probe dựa vào).

Toàn bộ import ở cấp module đều AN TOÀN để COLLECT trên CPU (không nạp model, không chạm
CUDA lúc import). torch chỉ được chạm BÊN TRONG test. conftest.py sẽ skip cả module khi
không đủ điều kiện phần cứng, nên các fixture nặng (nạp VRAM) không bao giờ chạy trên CPU.

E2E (#3) dùng edge-tts mặc định -> cần egress mạng trên box (hoặc GPT_SOVITS_URL trỏ tới
GPT-SoVITS cục bộ). Đây đúng là pipeline sản xuất; nếu box chặn mạng, đặt GPT_SOVITS_URL.
Vì đường edge-tts MẶC ĐỊNH bị khóa (OMNIVOICE_ALLOW_CLOUD_TTS off = fail-closed để không
rò văn bản ra Microsoft), test #3 tường minh opt-in cờ đó cho ĐÚNG lần chạy này; production
thật giữ cờ TRỐNG và dùng GPT-SoVITS cục bộ.
"""
import hashlib
import math
import os
import shutil
import struct
import tempfile
import time
import wave

import jwt
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu_acceptance


# --------------------------------------------------------------------------- #
# Tiện ích: WAV thật tối giản (stdlib `wave`, KHÔNG cần ffmpeg / không commit nhị phân)
# --------------------------------------------------------------------------- #
def _write_tiny_wav(path: str, seconds: float = 0.4, framerate: int = 16000,
                    freq: float = 440.0) -> str:
    n = int(seconds * framerate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(framerate)
        frames = bytearray()
        for i in range(n):
            val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / framerate))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))
    return path


def _gateway_keypair_and_token(job_id: str, role: str = "gateway", ttl: int = 120):
    """Sinh cặp khoá EC P-256 và ký token Y HỆT gateway (ES256, role, jobId, exp).

    Trả (public_spki_pem, token). Đặt public vào GATEWAY_JWT_PUBLIC_KEY để worker xác minh
    bằng ĐÚNG đường phòng thủ thật (không dùng dependency_overrides) — chứng minh hợp đồng
    Trạm-2 trên hình dạng đã triển khai thật.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    priv = ec.generate_private_key(ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    token = jwt.encode(
        {"role": role, "jobId": job_id, "exp": int(time.time()) + ttl},
        priv_pem,
        algorithm="ES256",
    )
    return pub_pem, token


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loaded_manager():
    """Nạp model THẬT lên VRAM MỘT LẦN cho cả module. Chỉ chạy trên hộp GPU (conftest
    đã skip nếu không đủ điều kiện) — nên load_all_models() fail-closed sẽ không bị kích
    hoạt trên CPU."""
    from src.model_manager import ModelManager

    mgr = ModelManager()
    mgr.load_all_models()
    return mgr


@pytest.fixture
def tiny_wav(tmp_path):
    return _write_tiny_wav(str(tmp_path / "tone.wav"))


# --------------------------------------------------------------------------- #
# 1) Cư trú VRAM
# --------------------------------------------------------------------------- #
def test_models_resident_on_cuda(loaded_manager):
    import torch

    assert torch.cuda.is_available()
    assert loaded_manager.is_loaded is True
    assert loaded_manager.device == "cuda"

    # Qwen thực sự nằm trên GPU (không phải nạp lén lên CPU rồi "quảng cáo" thường trú).
    translation = loaded_manager.models["translation"]
    assert translation.is_loaded is True
    param_device = next(translation.model.parameters()).device
    assert param_device.type == "cuda"

    # Whisper (ASR) cũng đã đăng ký và sẵn sàng.
    assert loaded_manager.models["whisper"].is_loaded is True


# --------------------------------------------------------------------------- #
# 2) Qwen cục bộ sinh được bản dịch (không cloud, không mock)
# --------------------------------------------------------------------------- #
def test_qwen_translate_smoke(loaded_manager):
    translation = loaded_manager.models["translation"]
    out = translation.translate_segments(
        segments=[{"text": "Hello, how are you today?", "start": 0.0, "end": 2.0,
                   "speaker": "SPEAKER_00"}],
        target_language="Vietnamese",
        style="Formal",
        source_language="en",
    )
    assert isinstance(out, list) and len(out) == 1
    translated = out[0].get("translated_text", "")
    # Chỉ khẳng định CÓ đầu ra thật (chất lượng dịch là đánh giá riêng, không assert ở đây).
    assert isinstance(translated, str) and translated.strip() != ""


# --------------------------------------------------------------------------- #
# 3) E2E qua HTTP endpoint với JWT gateway THẬT -> file audio lồng tiếng có thật
# --------------------------------------------------------------------------- #
def test_process_endpoint_e2e(loaded_manager, tiny_wav, monkeypatch):
    job_id = "ACCEPT-E2E-01"
    pub_pem, token = _gateway_keypair_and_token(job_id)
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub_pem)
    # voice_map={} -> giọng edge-tts mặc định. Cổng cloud TTS fail-closed nên phải opt-in
    # tường minh ở E2E này (chạy trên box GPU đã chấp nhận đánh đổi); nếu có GPT-SoVITS cục
    # bộ thì đặt voice='gpt-sovits' thay vì bật cờ này.
    monkeypatch.setenv("OMNIVOICE_ALLOW_CLOUD_TTS", "1")

    # md5 THẬT của fixture — client sẽ ký giá trị này; đưa vào payload để process_job
    # vượt cổng "thiếu md5" (fail-closed) và mô phỏng ràng buộc toàn vẹn đúng như production.
    with open(tiny_wav, "rb") as fh:
        fixture_md5 = hashlib.md5(fh.read()).hexdigest()

    # Chỉ thay đúng SEAM MẠNG (tải audio_url) bằng bản sao WAV cục bộ — phần GPU (dịch/
    # TTS/tách nền/mix/watermark) vẫn chạy THẬT. Sao ra file mới để bước dọn temp của
    # job xoá bản sao, không xoá fixture gốc. Chữ ký nhận thêm expected_md5 (ràng buộc
    # toàn vẹn); bản sao chính là fixture nên md5 khớp — ở đây ta trả thẳng file, không
    # hash lại vì đây là seam thay cho tải mạng.
    def _fake_download(url, expected_md5=None):
        fd, p = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        shutil.copy(tiny_wav, p)
        return p

    monkeypatch.setattr("src.audio_service.audio_service.download_audio", _fake_download)

    from src.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/worker/process",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "job_id": job_id,
            "audio_url": "http://local/fixture.wav",  # bị _fake_download chặn, không ra mạng
            "audio_md5": fixture_md5,  # ràng buộc toàn vẹn (client ký md5 của audio)
            "target_language": "Vietnamese",
            "translation_style": "Formal",
            # Segments do người dùng duyệt (Human-in-the-loop) -> bỏ qua ASR, e2e xác định.
            "segments": [
                {"text": "Hello, how are you today?", "start": 0.0, "end": 2.0,
                 "speaker": "SPEAKER_00", "emotion": "NEUTRAL"},
            ],
            "voice_map": {},
            "source_language": "en",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["job_id"] == job_id
    result = data["result"]
    assert result["status"] == "success"
    # Đầu ra là AUDIO thật tồn tại trên đĩa (Zero-Logging: KHÔNG có transcript trong response).
    out_path = result["dubbed_audio"]
    assert out_path and os.path.exists(out_path)
    assert "translated_segments" not in result
    # Dọn đầu ra cuối sau khi đã kiểm chứng.
    try:
        os.remove(out_path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 4) /health phản ánh cư trú VRAM thật sau khi nạp
# --------------------------------------------------------------------------- #
def test_health_reports_residence(loaded_manager):
    from src.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] is True
    assert data["device"] == "cuda"
