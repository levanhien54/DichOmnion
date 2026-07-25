import os
import tempfile

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import jwt

from src.model_manager import ModelManager

# Khởi tạo Quản lý Mô hình AI (singleton, thường trú VRAM)
model_manager = ModelManager()
security = HTTPBearer()

# Trạng thái vận hành. Trạm 3 (Gateway) có thể ra lệnh cách ly (quarantine) worker
# này khi phát hiện gian lận/timeout; khi đó worker từ chối mọi job mới.
WORKER_STATE = {"quarantined": False}


def _gateway_public_key() -> str | None:
    """SPKI PEM của khóa CÔNG KHAI (public) của Gateway.

    Trạm 2 (Zero-Trust bất đối xứng): Worker CHỈ giữ public key nên dù image bị lộ
    hacker cũng không thể giả mạo quyền Gateway (không có private key để ký).
    Đọc lười (lazy) từ biến môi trường để test có thể nạp khóa runtime.
    """
    return os.environ.get("GATEWAY_JWT_PUBLIC_KEY")


# Middleware kiểm tra JWT bất đối xứng (ES256) từ Gateway — TRẠM 2
async def verify_gateway_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    public_key = _gateway_public_key()
    # Fail-closed: chưa được cấp public key thì không tin bất kỳ ai.
    if not public_key:
        raise HTTPException(
            status_code=503,
            detail="Worker not provisioned with Gateway public key",
        )
    try:
        # algorithms=["ES256"] cố định để chặn tấn công alg-confusion
        # (không cho phép HS256 dùng public key làm secret).
        # require exp: token KHÔNG có hạn dùng bị từ chối — Gateway luôn ký kèm exp 2m,
        # nên đây chặn token "vĩnh viễn" (nếu private key rò rỉ, không thể tái dùng mãi).
        payload = jwt.decode(
            credentials.credentials,
            public_key,
            algorithms=["ES256"],
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Gateway Token Expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid Gateway Signature")

    if payload.get("role") != "gateway":
        raise HTTPException(status_code=403, detail="Invalid Role")
    return payload


# Lifecycle: Nạp mô hình thường trú VRAM ngay khi Server Start
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting GPU Worker...")
    model_manager.load_all_models()
    yield
    print("Stopping GPU Worker, clearing VRAM...")


app = FastAPI(title="OmniVoice GPU Worker", lifespan=lifespan)

# CORS: giới hạn origin nội bộ (worker chỉ bind 127.0.0.1 và chỉ Gateway gọi tới).
# KHÔNG dùng "*" — đó là lỗ hổng cho phép mọi web origin đọc kết quả.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "WORKER_ALLOWED_ORIGINS",
        "http://localhost:1420,http://localhost:5173,tauri://localhost,https://tauri.localhost",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class JobPayload(BaseModel):
    job_id: str
    audio_url: str  # Client chỉ gửi AUDIO (đã tách cục bộ), không gửi video thô
    target_language: str
    translation_style: str
    segments: list = []  # Segments do người dùng duyệt (Human-in-the-loop)
    # Ánh xạ ĐA GIỌNG do người dùng gán ở client: speaker_id -> voice. Rỗng = mọi
    # người nói dùng giọng mặc định theo ngôn ngữ đích.
    voice_map: dict = {}
    # Ngôn ngữ GỐC của thoại (đếm âm tiết đúng để căn lip-sync). None = mặc định "en".
    source_language: str | None = None


class TerminatePayload(BaseModel):
    reason: str = "unspecified"


@app.post("/api/worker/process")
async def process_audio(
    payload: JobPayload, token: dict = Depends(verify_gateway_jwt)
):
    """Endpoint nội bộ nhận job từ Gateway. Chỉ chạy khi có JWT (Trạm 2)."""
    if WORKER_STATE["quarantined"]:
        raise HTTPException(status_code=423, detail="Worker quarantined by Gateway")

    # Token binding (Trạm 2): Gateway ký JWT RIÊNG cho jobId này (claim `jobId`). Nếu
    # token bị rò rỉ, nó chỉ dùng lại được cho ĐÚNG job đó — không thể tái dùng để xử
    # lý job/nội dung khác. Token có role hợp lệ nhưng jobId khác payload => từ chối.
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")

    try:
        config = {
            "target_language": payload.target_language,
            "style": payload.translation_style,
            "segments": payload.segments,
            "voice_map": payload.voice_map,
            "source_language": payload.source_language,
        }
        # Đưa việc cho GPU pipeline. KHÔNG có sleep/giả lập — thời gian thực tế
        # chính là căn cứ để Trạm 3 (Gateway) phát hiện gian lận.
        result = await model_manager.process_job(payload.audio_url, config)

        return {"job_id": payload.job_id, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        # Zero-Logging: str(e) có thể nhúng đường dẫn nội bộ / URL kèm ?text=<kịch bản
        # plaintext> / dấu vết hệ thống. Client chỉ nhận TÊN loại lỗi, không nhận nội dung thô.
        raise HTTPException(
            status_code=500,
            detail=f"Xử lý job thất bại ({type(e).__name__})",
        )


@app.post("/api/worker/terminate", dependencies=[Depends(verify_gateway_jwt)])
async def terminate(payload: TerminatePayload):
    """Trạm 3 enforcement: Gateway ra lệnh cách ly worker khi phát hiện bất thường.

    Ở môi trường serverless thật, việc "rút phích" cụm GPU do API của nhà cung cấp
    (RunPod/Modal) thực hiện; tại đây ta bật cờ để worker từ chối job mới ngay lập tức.
    """
    WORKER_STATE["quarantined"] = True
    return {"status": "quarantined", "reason": payload.reason}


@app.get("/api/worker/download", dependencies=[Depends(verify_gateway_jwt)])
async def download_audio(path: str):
    """Tải file kết quả. Chỉ Gateway (có JWT hợp lệ) mới truy cập được."""
    temp_dir = tempfile.gettempdir()
    abs_path = os.path.abspath(path)

    # Chống directory traversal: chỉ cho phép file trong thư mục temp
    if not abs_path.startswith(temp_dir):
        raise HTTPException(status_code=403, detail="Access Denied: Can only access temp files")
    if not (abs_path.endswith(".wav") or abs_path.endswith(".mp3") or abs_path.endswith(".mp4")):
        raise HTTPException(status_code=403, detail="Access Denied: Invalid file type")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "video/mp4" if abs_path.endswith(".mp4") else "audio/wav"
    return FileResponse(abs_path, media_type=media_type)


@app.post("/api/worker/upload", dependencies=[Depends(verify_gateway_jwt)])
async def upload_audio(file: UploadFile = File(...)):
    """Staging AUDIO nội bộ (chỉ Gateway gọi). Client KHÔNG upload trực tiếp tới worker."""
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid audio format (expect .wav/.mp3)")

    temp_dir = tempfile.gettempdir()
    suffix = ".wav" if file.filename.lower().endswith(".wav") else ".mp3"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())

    return {"status": "success", "audio_path": path}


if __name__ == "__main__":
    # Worker KHÔNG mở IP public — chỉ bind loopback. Gateway (trong cùng mạng riêng
    # / qua tunnel nội bộ) là đầu vào hợp lệ duy nhất.
    import uvicorn

    # Mặc định loopback (không mở IP public). ENV cho phép orchestrator (RunPod/Modal)
    # gán host/port trong mạng riêng mà không sửa mã.
    host = os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("WORKER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
