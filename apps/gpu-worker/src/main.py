import asyncio
import hashlib
import hmac
import json
import math
import os
import traceback
import weakref
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from contextlib import asynccontextmanager
import jwt

from src.model_manager import ModelManager, _run_blocking
from src.audio_service import AudioIntegrityError
from src.audio_engine import audio_engine, run_periodic_final_sweep
from src.timecode import to_seconds
from src.artifact_crypto import ARTIFACT_ALG, load_recipient_public_key, seal_artifact
from src.tts_service import (
    TTSError,
    TTSConfigurationError,
    TTSProfileError,
    local_tts_required,
    tts_service,
)
from src.gpu_memory_monitor import (
    GPU_MEMORY_METRICS_HEADER,
    encode_gpu_memory_metrics,
    gpu_memory_metrics_enabled,
)
from src.qwen_prompt_profiles import MAX_CUSTOM_INSTRUCTION_CHARS

# Khởi tạo Quản lý Mô hình AI (singleton, thường trú VRAM)
model_manager = ModelManager()
security = HTTPBearer()

# Trạng thái vận hành. Trạm 3 (Gateway) có thể ra lệnh cách ly (quarantine) worker
# này khi phát hiện gian lận/timeout; khi đó worker từ chối mọi job mới.
WORKER_STATE = {"quarantined": False}

# Admitted request tasks are scoped to the serving event loop for the same reason as the
# GPU semaphore below. The terminate route cancels the whole request, including its
# serialize/encrypt/hash/R2 tail, without waiting for blocking thread work to return.
_ACTIVE_GPU_TASKS = weakref.WeakKeyDictionary()

# Async ANALYZE jobs are intentionally scoped to the serving event loop.  The Gateway
# submits once and polls short requests, avoiding RunPod proxy's 90-second response cap.
# The key includes the signed attempt so a retry/revision can never receive another
# revision's ciphertext.  Completed entries are retained for a bounded period so a
# duplicate submit is idempotent while the Gateway queue message is redelivered.
_ASYNC_ANALYZE_JOBS = weakref.WeakKeyDictionary()
_ASYNC_ANALYZE_RESULT_TTL_S = 3600.0


def _async_analyze_jobs() -> dict:
    loop = asyncio.get_running_loop()
    jobs = _ASYNC_ANALYZE_JOBS.get(loop)
    if jobs is None:
        jobs = {}
        _ASYNC_ANALYZE_JOBS[loop] = jobs
    now = asyncio.get_running_loop().time()
    for key, entry in list(jobs.items()):
        if entry.get("task") is None and now - entry.get("finished_at", now) > _ASYNC_ANALYZE_RESULT_TTL_S:
            jobs.pop(key, None)
    return jobs


def _shutdown_drain_timeout_s() -> float:
    """Return a bounded grace period for cooperative request shutdown."""
    raw = os.environ.get("WORKER_SHUTDOWN_DRAIN_SECONDS", "15").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 15.0
    if not math.isfinite(value):
        value = 15.0
    return min(120.0, max(0.0, value))


def _active_gpu_tasks() -> set[asyncio.Task]:
    loop = asyncio.get_running_loop()
    tasks = _ACTIVE_GPU_TASKS.get(loop)
    if tasks is None:
        tasks = set()
        _ACTIVE_GPU_TASKS[loop] = tasks
    return tasks


def _cancel_active_gpu_jobs() -> int:
    """Request cancellation for admitted requests and return the affected task count.

    Synchronous CUDA calls cannot be preempted safely from Python. Their offload helper
    drains the current worker-thread call before propagating cancellation, which stops the
    pipeline at the next await boundary while preserving GPU/semaphore and temp-file ownership.
    """
    active = [task for task in _active_gpu_tasks() if not task.done()]
    for task in active:
        task.cancel()
    return len(active)


async def _drain_active_gpu_jobs() -> tuple[int, int]:
    """Cancel admitted requests and wait briefly for their cleanup to complete.

    A synchronous CUDA/FFmpeg call cannot be hard-preempted safely. The request
    cancellation handlers therefore drain the child task before releasing its GPU
    slot. A bounded wait keeps shutdown from hanging forever if a native call is
    stuck; the caller can then let the container supervisor escalate to SIGKILL.
    Returns ``(cancelled_count, still_running_count)`` for sanitized diagnostics.
    """
    active = [task for task in _active_gpu_tasks() if not task.done()]
    for task in active:
        task.cancel()
    if not active:
        return 0, 0

    done, pending = await asyncio.wait(
        active,
        timeout=_shutdown_drain_timeout_s(),
    )
    for task in done:
        # Retrieve late exceptions so shutdown does not emit noisy unhandled-task
        # warnings; the externally visible result is still cancellation.
        try:
            task.exception()
        except BaseException:
            pass
    if pending:
        print(
            "GPU request shutdown drain exceeded its bounded grace period; "
            f"remaining={len(pending)}"
        )
    return len(active), len(pending)


def _raise_if_quarantined() -> None:
    if WORKER_STATE["quarantined"]:
        raise HTTPException(status_code=423, detail="Worker quarantined by Gateway")


async def _run_gpu_job(job_coro):
    """Await one model pipeline without letting cancellation orphan GPU work.

    The model pipeline offloads synchronous inference to worker threads. Shielding the
    child task and draining it before propagating cancellation keeps the surrounding
    GPU semaphore held until the current blocking call and temp-file cleanup have really
    finished. Cancelling the child requests a cooperative stop at the next stage boundary;
    it cannot preempt a CUDA kernel or Python thread that is already executing.
    """
    task = asyncio.create_task(job_coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and not task.cancelled():
            try:
                task.exception()
            except BaseException:
                pass
        raise

# Hàng đợi 1-job (mặc định) cho MỘT GPU 24GB: Whisper + Qwen 4B đã thường trú VRAM;
# chạy song song nhiều job sẽ nạp thêm stem Demucs/AudioSeal và bộ nhớ trung gian ->
# nguy cơ OOM (tràn VRAM) và chết cả worker. Semaphore ép tuần tự hoá pipeline nặng.
# WORKER_MAX_CONCURRENT_JOBS cho phép nới ra nếu về sau chạy trên GPU lớn hơn — nhưng
# cổng xác thực/quarantine ở TRƯỚC semaphore nên job trái phép không bao giờ chiếm slot.
# M3-S9 (plan dòng 345 "concurrency … tại Worker"): ĐÂY là cổng concurrency phía Worker —
# một bound TOÀN CỤC, không-rò-rỉ (semaphore tự giải phóng khi request kết thúc, dù lỗi), KHÔNG
# phải bộ đếm in-flight theo-thiết-bị dễ rò. Phía Gateway, per-device được chặn bằng rate-limit
# nguyên tử (JOBS_RATE_LIMIT, M3-S5); xem ghi chú tại apps/gateway/src/limits.ts vì sao KHÔNG
# thêm bộ đếm concurrency theo-thiết-bị riêng (rủi ro rò/khoá thiết bị, không phải tiêu chí M3).
_MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("WORKER_MAX_CONCURRENT_JOBS", "1")))
_JOB_SEMAPHORES = weakref.WeakKeyDictionary()


def _job_semaphore() -> asyncio.Semaphore:
    """Return the one-GPU semaphore owned by the current serving event loop.

    Uvicorn serves this app on one loop. Keeping construction lazy avoids retaining a
    closed loop across TestClient/ASGI lifespans while preserving capacity=1 for every
    real serving loop.
    """
    loop = asyncio.get_running_loop()
    semaphore = _JOB_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JOBS)
        _JOB_SEMAPHORES[loop] = semaphore
    return semaphore


@asynccontextmanager
async def _admitted_gpu_request():
    """Admit and track one request until its durable-output tail has finished.

    The yielded callback releases the scarce GPU slot once model work and its cleanup are
    complete, while the outer request remains registered through serialization and R2 I/O.
    Calling it more than once is harmless, which keeps exception cleanup straightforward.
    """
    semaphore = _job_semaphore()
    await semaphore.acquire()
    try:
        _raise_if_quarantined()
    except BaseException:
        semaphore.release()
        raise

    request_task = asyncio.current_task()
    if request_task is None:
        semaphore.release()
        raise RuntimeError("GPU admission requires an asyncio request task")

    active_tasks = _active_gpu_tasks()
    active_tasks.add(request_task)
    gpu_slot_held = True

    def release_gpu_slot() -> None:
        nonlocal gpu_slot_held
        if gpu_slot_held:
            gpu_slot_held = False
            semaphore.release()

    try:
        yield release_gpu_slot
    finally:
        release_gpu_slot()
        active_tasks.discard(request_task)

# Giới hạn ĐẦU VÀO fail-closed (chống DoS OOM/treo — Đợt 17 F3/F4). Một thiết bị ĐÃ đăng ký
# nhưng KHÔNG đáng tin (Zero-Trust) vẫn KÝ hợp lệ được một payload với `segments` khổng lồ
# hoặc text mỗi segment cực dài. translate_segments GỘP TẤT CẢ segment vào MỘT prompt Qwen
# rồi tokenize + generate MỘT LẦN (translation_service._build_prompt/_generate) -> input
# khổng lồ làm TRÀN VRAM (OOM) hoặc TREO tới khi Gateway timeout 15' -> Trạm 3 ghi
# quarantine:<workerUrl> 24h -> TỪ CHỐI job của MỌI tenant (DoS chéo tenant). Chặn ở TẦNG
# API (pydantic -> 422) TRƯỚC cả semaphore/model: 422 là 4xx -> Gateway đánh FAILED (không
# retry, KHÔNG quarantine), biến "treo cả cụm" thành "một job hỏng sạch". Phòng thủ theo
# lớp: Gateway cũng chặn sớm (400) ở biên (xem apps/gateway/src/index.ts). Env-tunable để
# GPU lớn hơn nới ra mà không sửa mã. KHÔNG chunk prompt ở đây (đổi thiết kế lớn + rủi ro
# ID-parity xuyên chunk) — bound là bản vá DoS tối thiểu, trung thực; chunk là việc TƯƠNG LAI.
_MAX_SEGMENTS = max(1, int(os.environ.get("WORKER_MAX_SEGMENTS", "2000")))
_MAX_SEGMENT_TEXT_CHARS = max(1, int(os.environ.get("WORKER_MAX_SEGMENT_TEXT_CHARS", "2000")))
_MAX_TOTAL_TEXT_CHARS = max(1, int(os.environ.get("WORKER_MAX_TOTAL_TEXT_CHARS", "200000")))
# Đợt 18 F6 — NGOÀI `text`, translation_service còn nhúng hai trường client-controlled khác
# vào prompt Qwen: `id` (định danh segment, translate_segments dòng ~278) và `speaker`/
# `speaker_id` (nhãn người nói, ~284). F3/F4 CHỈ bound `text` nên kẻ tấn công gửi `text` tí
# hon kèm `id`/`speaker` KHỔNG LỒ vẫn lọt cả hai cổng biên -> phình prompt -> OOM/treo ->
# quarantine chéo tenant (đúng vector F3/F4 tưởng đã đóng). Bound độ dài MỖI trường (đủ rộng
# cho id/nhãn thực: "seg-00042", "SPEAKER_03", tên người) VÀ cộng luôn vào TỔNG ngân sách
# prompt để chặn cả trục gộp count × field.
_MAX_SEGMENT_META_CHARS = max(1, int(os.environ.get("WORKER_MAX_SEGMENT_META_CHARS", "256")))

# M3-S9 (plan dòng 345: "Bổ sung giới hạn … request body tại CẢ Gateway LẪN Worker") — cổng
# kích thước body THÔ, đối xứng với biên Gateway (auth.ts MAX_REQUEST_BODY_BYTES). Các cổng
# theo-TRƯỜNG ở trên (pydantic) đo trường ĐÃ PARSE nhưng KHÔNG đo body thô: một payload `text`
# tí hon kèm `original_text` khổng lồ lọt hết mọi field-cap mà vẫn phình body vô hạn (buffer +
# JSON-parse hàng trăm KB trước khi field-validator kịp chạy). Chặn ở middleware theo
# Content-Length TRƯỚC cả auth/parse là phòng thủ DoS rẻ nhất. Mặc định 256 KiB (khớp Gateway).
_MAX_REQUEST_BODY_BYTES = max(1, int(os.environ.get("WORKER_MAX_REQUEST_BODY_BYTES", str(256 * 1024))))

# CỐ Ý KHÔNG coalescing in-flight theo job_id. Một bản trước từng gộp các dispatch trùng
# job_id vào cùng một asyncio.Task để né render GPU trùng khi Gateway auto-retry — NHƯNG
# job_id chỉ DUY NHẤT theo thiết bị (Gateway tự namespace state của nó bằng
# job:<deviceId>:<jobId>), còn JWT worker chỉ mang claim `jobId` (không có deviceId). Gộp
# theo job_id TRẦN nên phá vỡ ranh giới tenant: hai thiết bị khác nhau gửi cùng chuỗi job_id
# sẽ va vào nhau và thiết bị thứ hai NHẬN kết quả (đường dẫn dubbed_audio) của thiết bị đầu
# -> rò rỉ audio xuyên tenant; đồng thời một retry gộp trả về "quá nhanh" làm Trạm 3 gán
# nhầm gian lận và tự cách ly worker 24h. Mỗi request giờ chạy pipeline độc lập, tuần tự hoá
# qua semaphore. Phần dư (một lần render trùng hiếm gặp khi retry mạng thoáng qua) là ranh
# giới G-03 đã ghi nhận & chấp nhận ("không phải lỗ hổng bảo mật"), KHÔNG đánh đổi bằng lỗ
# rò tenant. Đóng triệt để double-dispatch cần compare-and-set nguyên tử ở tầng điều phối
# (Durable Object) — ngoài phạm vi mã worker.


def _gateway_public_key() -> str | None:
    """SPKI PEM của khóa CÔNG KHAI (public) của Gateway.

    Trạm 2 (Zero-Trust bất đối xứng): Worker CHỈ giữ public key nên dù image bị lộ
    hacker cũng không thể giả mạo quyền Gateway (không có private key để ký).
    Đọc lười (lazy) từ biến môi trường để test có thể nạp khóa runtime.
    """
    return os.environ.get("GATEWAY_JWT_PUBLIC_KEY")


# ── Hợp đồng JWT Trạm 2 (Gateway→Worker) ─────────────────────────────────────
# MIRROR của packages/shared-types/src/index.ts (WORKER_JWT_AUDIENCE /
# WorkerJwtActSchema / WorkerJwtClaimsSchema). Bản TS là NGUỒN SỰ THẬT: jose ký,
# PyJWT (dưới đây) xác thực. Giữ hai giá trị này ĐỒNG BỘ tuyệt đối.
WORKER_JWT_AUDIENCE = "dichomnion-gpu-worker"
# Gateway and GPU hosts are provisioned independently; keep the tolerance far below
# the two-minute token lifetime while avoiding false rejects at a one-second boundary.
_WORKER_JWT_CLOCK_SKEW_SECONDS = 5

# M3-S8: MIRROR của packages/shared-types/src/index.ts WORKER_RESPONSE_SCHEMA_VERSION.
# Bản TS là NGUỒN SỰ THẬT — Gateway REQUIRE đúng giá trị này khi validate response
# worker TRƯỚC khi DONE, nên một body do hợp đồng khác (cũ/mới) shape sẽ bị từ chối
# thay vì hiểu nhầm. Bump ĐỒNG BỘ hai phía mỗi khi đổi shape response.
WORKER_RESPONSE_SCHEMA_VERSION = 1

# M4-S5: MIRROR của packages/shared-types/src/index.ts. Đường ANALYZE có HAI hợp đồng
# phiên bản RIÊNG (dù giá trị hiện đều = 1): (1) WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION —
# version của BODY worker trả từ /api/worker/analyze mà Gateway REQUIRE khớp (literal)
# TRƯỚC khi flip AWAITING_REVIEW; (2) ANALYZE_RESULT_SCHEMA_VERSION — version của
# AnalyzeResult PLAINTEXT nằm BÊN TRONG ciphertext (client giải mã + kiểm). Giữ RIÊNG
# thay vì tái dùng WORKER_RESPONSE_SCHEMA_VERSION để bump từng hợp đồng độc lập khi shape
# đổi (một số coincidence, không phải cùng một contract). Bump ĐỒNG BỘ với bản TS.
WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION = 1
ANALYZE_RESULT_SCHEMA_VERSION = 1
WORKER_VOICE_CAPABILITIES_SCHEMA_VERSION = 1
WORKER_READINESS_SCHEMA_VERSION = 1

# M3-S10: MIRROR của packages/shared-types/src/index.ts FAILURE_REASONS.INTERNAL_ERROR.
# Mã lỗi ĐÃ SANITIZE, cố định — là token DUY NHẤT được phép đi vào HTTP response khi
# xử lý job ném lỗi không lường trước (500). KHÔNG BAO GIỜ nhúng tên class, str(e),
# đường dẫn nội bộ, URL kèm ?text=<plaintext>, md5 hay nội dung user vào response
# (Zero-Logging). Traceback đầy đủ chỉ đi ra STDERR khi WORKER_DEBUG bật.
_INTERNAL_ERROR_REASON = "internal_error"
_TTS_INVALID_REASON = "tts_profile_invalid"
_TTS_UNAVAILABLE_REASON = "tts_unavailable"


def _tts_http_error(error: TTSError) -> HTTPException:
    if isinstance(error, (TTSConfigurationError, TTSProfileError)):
        return HTTPException(status_code=422, detail=_TTS_INVALID_REASON)
    return HTTPException(status_code=503, detail=_TTS_UNAVAILABLE_REASON)


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
        # audience: token PHẢI mang aud == WORKER_JWT_AUDIENCE. require aud ép cả token
        # KHÔNG có aud lẫn token mang aud của dịch vụ KHÁC đều bị từ chối ngay tại biên
        # crypto — một token cấp cho dịch vụ khác không thể chuyển hướng vào worker (M3-S7).
        payload = jwt.decode(
            credentials.credentials,
            public_key,
            algorithms=["ES256"],
            audience=WORKER_JWT_AUDIENCE,
            options={"require": ["exp", "aud"]},
            leeway=_WORKER_JWT_CLOCK_SKEW_SECONDS,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Gateway Token Expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid Gateway Signature")

    if payload.get("role") != "gateway":
        raise HTTPException(status_code=403, detail="Invalid Role")
    # Internal provenance marker. Tests that replace this dependency intentionally exercise
    # route logic without pretending to cross the cryptographic Station-2 boundary; every
    # token accepted by the real verifier is marked and must pass body-digest binding below.
    payload["_station2_verified"] = True
    return payload


def require_action(expected: str):
    """Factory dependency: xác thực JWT Trạm 2 (chữ ký/exp/aud/role qua
    verify_gateway_jwt) VÀ ràng buộc claim ``act`` == ``expected``. Ép PER ROUTE nên
    token ``dispatch`` không thể lái /terminate và ngược lại (M3-S7/A3). Cố ý là lớp
    bọc MỎNG trên verify_gateway_jwt để app.dependency_overrides[verify_gateway_jwt]
    trong test vẫn chảy qua (override cấp payload, act-check vẫn áp dụng)."""

    async def _require(
        request: Request,
        token: dict = Depends(verify_gateway_jwt),
    ) -> dict:
        if token.get("act") != expected:
            raise HTTPException(status_code=403, detail="Token action mismatch")

        # A readiness token authorizes one read-only GET. It must never acquire the
        # compute scope carried by attempt/bodyDigest claims, even when signed by the
        # trusted Gateway key.
        if expected == "probe" and (
            "attempt" in token or "bodyDigest" in token
        ):
            raise HTTPException(status_code=403, detail="Token scope mismatch")

        # Bind every compute token to the EXACT JSON bytes sent by Gateway. Gateway must:
        #   1. serialize the request body once;
        #   2. set bodyDigest = sha256(UTF-8 body bytes).hexdigest() in the signed JWT;
        #   3. send that same serialized string as the HTTP body.
        # Exact-byte binding avoids cross-language JSON canonicalization ambiguity and covers
        # every field, including URLs, encryption keys, approved text, and upload pointers.
        # compare_digest keeps mismatch timing independent of the matching prefix length.
        # The provenance guard only preserves FastAPI dependency overrides used by CPU route
        # tests; it cannot be reached by a network token without passing verify_gateway_jwt.
        if expected in {"dispatch", "analyze", "render"} and token.get(
            "_station2_verified"
        ) is True:
            claimed_digest = token.get("bodyDigest")
            if not (
                isinstance(claimed_digest, str)
                and len(claimed_digest) == 64
                and all(ch in "0123456789abcdef" for ch in claimed_digest)
            ):
                raise HTTPException(status_code=403, detail="Token body digest missing or invalid")
            actual_digest = hashlib.sha256(await request.body()).hexdigest()
            if not hmac.compare_digest(actual_digest, claimed_digest):
                raise HTTPException(status_code=403, detail="Token/Body Mismatch")
        return token

    return _require


# M2-S5e — task nền dọn đầu ra CUỐI quá hạn theo CHU KỲ (backstop khi worker RẢNH).
# sweep_stale_finals chỉ chạy CƠ HỘI ở đầu mỗi job; một worker rảnh sẽ giữ audio lồng
# tiếng nhạy cảm + đĩa vô thời hạn tới job kế. Giữ handle ở module để lifespan hủy sạch
# lúc shutdown (và để test wiring quan sát được). None = chưa/không lên lịch (đã tắt).
_final_reaper_task: "asyncio.Task | None" = None


def _final_reaper_settings() -> "tuple[float, float]":
    """(interval_s, ttl_s) cho reaper nền, parse env PHÒNG THỦ (giá trị rác -> mặc định).

    interval mặc định 3600s (khớp nhịp cron sweeper của Gateway); TTL DÙNG CHUNG
    WORKER_FINAL_TTL_S với sweep per-job nên hai cơ chế đồng nhất ngưỡng. interval <= 0
    -> TẮT reaper (van an toàn: môi trường hạn chế / test có thể tắt hẳn task nền)."""
    try:
        interval_s = float(os.environ.get("WORKER_FINAL_REAP_INTERVAL_S", "3600"))
    except (TypeError, ValueError):
        interval_s = 3600.0
    try:
        ttl_s = float(os.environ.get("WORKER_FINAL_TTL_S", "3600"))
    except (TypeError, ValueError):
        ttl_s = 3600.0
    return interval_s, ttl_s


# Lifecycle: Nạp mô hình thường trú VRAM ngay khi Server Start
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _final_reaper_task
    print("Starting GPU Worker...")
    model_manager.load_all_models()
    # Khởi động reaper nền (nếu bật). Task fire-and-forget: tự nuốt mọi lỗi + CancelledError.
    interval_s, ttl_s = _final_reaper_settings()
    if interval_s > 0:
        _final_reaper_task = asyncio.create_task(run_periodic_final_sweep(interval_s, ttl_s))
    else:
        _final_reaper_task = None
    try:
        yield
    finally:
        print("Stopping GPU Worker, clearing VRAM...")
        # Stop admitted work before closing the local TTS client. This gives normal
        # cancellation paths a chance to drain native calls and remove job temp files.
        await _drain_active_gpu_jobs()
        # Shutdown SẠCH: hủy reaper & chờ nó dừng (reaper nuốt CancelledError -> await êm).
        if _final_reaper_task is not None:
            _final_reaper_task.cancel()
            try:
                await _final_reaper_task
            except asyncio.CancelledError:
                pass
            _final_reaper_task = None
        await tts_service.aclose()


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


@app.middleware("http")
async def _enforce_request_body_cap(request: Request, call_next):
    """M3-S9 — cổng kích thước body THÔ, chạy TRƯỚC auth/parse (đối xứng Gateway auth.ts).

    FastAPI middleware bọc TOÀN BỘ xử lý request nên chạy trước cả Depends(verify_gateway_jwt)
    và pydantic parse — đây là điểm phòng thủ DoS sớm nhất trong worker: một body khổng lồ bị
    413 mà KHÔNG bao giờ được buffer/parse hay xác thực JWT. Kiểm Content-Length (Gateway —
    nguồn gọi hợp lệ DUY NHẤT — luôn gửi header này). Body chunked/thiếu Content-Length không
    bị chặn ở đây (chấp nhận: cổng theo-TRƯỜNG pydantic vẫn bound nội dung ở tầng sau). 413 là
    4xx → Gateway đánh FAILED (không retry, KHÔNG quarantine), khớp ngữ nghĩa các cổng biên khác."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > _MAX_REQUEST_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            pass  # Content-Length không phải số → để tầng sau xử lý bình thường
    return await call_next(request)


def _reject_lone_surrogates(s: str, label: str) -> None:
    """Đợt 24 F13 — chuỗi chứa LONE SURROGATE (nửa cặp UTF-16, vd "\\ud800") là UTF
    KHÔNG well-formed: nó lọt mọi cổng KIỂU/ĐỘ-DÀI (vẫn là `str`, len hợp lệ) nhưng khi
    Qwen fast-tokenizer chuyển str -> Rust String (PyO3) sẽ ném UnicodeEncodeError
    "surrogates not allowed". Lỗi đó nổ trong translation_service._generate — NGOÀI khối
    try/except retry (translation_service.py:304 gọi _generate TRƯỚC `try` dòng 305) ->
    không bắt -> HTTP 500 -> Gateway coi 5xx là retryable -> chạy lại TOÀN BỘ pipeline
    tối đa 3× (mỗi lần re-run Whisper ASR khi client gửi payload không-segments = khuếch
    đại ~3× chi phí GPU/job rác). `str.encode("utf-8")` ném CÙNG lỗi trên CÙNG tập lone
    surrogate -> dùng nó làm cổng fail-closed: vi phạm -> ValueError -> FastAPI 422 terminal
    (4xx = Gateway FAILED, KHÔNG quarantine, KHÔNG retry). Biên đối xứng ở Gateway
    (validateJobSize, !isWellFormed) chặn sớm 400 — phòng thủ hai lớp như F3–F12."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} chứa ký tự UTF không hợp lệ (lone surrogate)")


class PromptProfilePayload(BaseModel):
    """Bounded style guidance; immutable translation policy stays server-owned."""

    preset_id: Literal[
        "faithful_dubbing", "natural_commentary", "concise_voiceover"
    ] = "faithful_dubbing"
    preset_revision: Literal[1] = 1
    custom_instructions: str | None = Field(
        default=None, min_length=1, max_length=MAX_CUSTOM_INSTRUCTION_CHARS
    )

    @field_validator("custom_instructions")
    @classmethod
    def _custom_instructions_well_formed(cls, value: str | None):
        if value is None:
            return value
        _reject_lone_surrogates(value, "custom_instructions")
        if any(
            (ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127
            for char in value
        ):
            raise ValueError("custom_instructions chứa ký tự điều khiển không hợp lệ")
        return value


class JobPayload(BaseModel):
    job_id: str
    audio_url: str  # Client chỉ gửi AUDIO (đã tách cục bộ), không gửi video thô
    # md5 (hex) của bytes audio, do client tính cục bộ và KÝ kèm (JobRequest.videoAudioMd5).
    # Object key R2 dùng chung cho mọi job (presigned PUT chỉ trỏ 1 key) nên worker PHẢI so
    # bytes tải-về với md5 này và fail-closed khi lệch — chống tải nhầm/rò rỉ audio chéo
    # tenant. Default "" để KHÔNG phá thứ tự kiểm (auth/quarantine chạy TRƯỚC); enforce
    # non-empty ở process_job (fail-closed), không ở tầng Pydantic.
    audio_md5: str = ""
    # Bound các trường free-text CŨNG được nhúng vào prompt Qwen (chống bloat prompt qua
    # target/style/source, không chỉ segments). Độ dài này quá đủ cho TÊN ngôn ngữ /
    # phong cách ("Brazilian Portuguese", "Formal", ...).
    target_language: str = Field(max_length=200)
    translation_style: str = Field(max_length=200)
    segments: list = []  # Segments do người dùng duyệt (Human-in-the-loop)
    # Ánh xạ ĐA GIỌNG do người dùng gán ở client: speaker_id -> voice. Rỗng = mọi
    # người nói dùng giọng mặc định theo ngôn ngữ đích.
    voice_map: dict = {}
    # Ngôn ngữ GỐC của thoại (đếm âm tiết đúng để căn lip-sync). None = mặc định "en".
    source_language: str | None = Field(default=None, max_length=200)
    prompt_profile: PromptProfilePayload | None = None
    # M2-S3 (result → R2). Gateway cấp presigned PUT ngắn hạn cho ĐÚNG object kết quả của
    # attempt này (results/<device>/<job>/<attempt>.wav) tại dispatch; worker tải bản render
    # cuối lên đó rồi báo md5/size để Gateway HEAD-verify TRƯỚC khi DONE. Cả hai do GATEWAY
    # cấp (đã qua JWT Trạm 2) nên là trường TIN CẬY — KHÔNG cổng SSRF như audio_url (vốn do
    # CLIENT kiểm soát): host luôn là <account>.r2.cloudflarestorage.com của chính ta. Default
    # "" để giữ tương thích đường dispatch chưa bật R2 (worker bỏ qua upload khi rỗng).
    result_upload_url: str = ""
    result_key: str = ""
    # Số render attempt (Gateway ký kèm vào token dispatch VÀ gửi trong body). process_audio
    # ràng buộc token.attempt == payload.attempt: một token cấp cho attempt 1 KHÔNG thể bị
    # phát lại vào slot attempt 2 (M3-S7). Default 1 để không phá thứ tự kiểm — token dispatch
    # mới là thẩm quyền (nó luôn mang attempt), body chỉ để đối chiếu.
    attempt: int = 1

    @field_validator("segments")
    @classmethod
    def _bound_segments(cls, v: list) -> list:
        """Đợt 17 F3/F4 — cổng fail-closed chống DoS OOM/treo tại BIÊN không tin cậy.

        Chặn 3 trục thổi phồng prompt Qwen: (1) SỐ LƯỢNG segment, (2) độ dài text MỖI
        segment, (3) TỔNG độ dài text. Vi phạm -> ValueError -> FastAPI trả 422 (4xx) NGAY
        khi parse, TRƯỚC semaphore/model — Gateway coi 4xx là FAILED terminal (không
        quarantine). Chỉ đọc `text`/`original_text` (đúng field translate_segments dùng)."""
        if len(v) > _MAX_SEGMENTS:
            raise ValueError(f"too many segments (>{_MAX_SEGMENTS})")
        total = 0
        for seg in v:
            # Đợt 20 F9 — phần tử segment PHẢI là object. Bản cũ `continue` (bỏ qua) với niềm
            # tin "translate_segments xử lý bằng .get an toàn" là SAI: str/int/None/list KHÔNG có
            # .get -> translation_service.py `seg.get("text", ...)` ném AttributeError. Lỗi nổ MUỘN
            # (sau khi tải audio tới trần F7) -> HTTP 500 -> Gateway retry TOÀN BỘ pipeline 3× trên
            # rác. Hợp đồng shared-types quy định mỗi segment là object; fail-closed tại parse (422
            # terminal, 4xx = Gateway FAILED không retry) thay vì đẩy crash xuống hạ nguồn.
            if not isinstance(seg, dict):
                raise ValueError("segment must be an object")
            # Đợt 22 F12 — KHÔNG dùng `or ""`: phép này biến MỌI falsy (None/0/False/[]/{}) thành ""
            # TRƯỚC isinstance nên value phi-chuỗi falsy — đặc biệt JSON `null` — LỌT cổng. Consumer
            # translation_service.py:265 đọc `seg.get("text", seg.get("original_text",""))` KHÔNG
            # chuẩn hóa nên khi key "text" hiện diện value None -> orig_text=None -> _merge (gọi NGOÀI
            # try/except retry) dựng TranslatedSegment(original_text=None) [original_text: str BẮT BUỘC]
            # -> pydantic ValidationError KHÔNG được bắt -> HTTP 500 -> Gateway retry TOÀN BỘ pipeline
            # 3× (khuếch đại ~3× Qwen). Đọc THẲNG như consumer (CÙNG biểu thức) rồi ép str: key vắng ->
            # default "" (hợp lệ, consumer cũng vậy); None/0/False/... -> 422 terminal TRƯỚC model.
            text = seg.get("text", seg.get("original_text", ""))
            if not isinstance(text, str):
                raise ValueError("segment text must be a string")
            if len(text) > _MAX_SEGMENT_TEXT_CHARS:
                raise ValueError(f"segment text too long (>{_MAX_SEGMENT_TEXT_CHARS} chars)")
            _reject_lone_surrogates(text, "segment text")  # Đợt 24 F13 (cùng tokenizer sink)
            total += len(text)
            # F6: `id` và `speaker`/`speaker_id` cũng đi thẳng vào prompt Qwen (nhúng qua
            # _build_prompt) — bound ĐỘ DÀI TỪNG trường + cộng vào TỔNG.
            # Đợt 21 F11 — nhưng độ-dài CHƯA đủ: hai trường này còn đi VERBATIM (không chuẩn hóa)
            # vào translation_service `_merge` (dòng ~284 -> ~442/450) dựng `TranslatedSegment`,
            # nơi `id: int|str` và `speaker_id: str`. `_merge` được gọi NGOÀI khối try/except retry
            # (translation_service.py:317), nên một giá trị SAI KIỂU (dict/list/None/float...) —
            # vốn lọt cả cổng cũ vì cổng cũ chỉ đo độ dài KHI đã là chuỗi (F6 lo trục ĐỘ-DÀI, giả
            # định kiểu đúng) — sẽ nổ MUỘN ở `TranslatedSegment(...)` -> pydantic ValidationError
            # KHÔNG được bắt -> HTTP 500 -> Gateway retry TOÀN BỘ pipeline 3× (khuếch đại ~3× chi
            # phí GPU). Ép KIỂU tại parse (422 terminal, 4xx = Gateway FAILED không retry) — cùng
            # trục KIỂU với F9 (phần tử)/F10 (voice_map value), theo đúng kiểu sink chấp nhận.
            seg_id = seg.get("id", "")
            # id: hợp đồng shared-types là `string`; sink TranslatedSegment.id là `int|str`. Chấp
            # nhận str HOẶC int; loại bool (True/False là int con -> str(True)="True" bẩn parity) và
            # mọi kiểu khác (dict/list/None/float -> ValidationError ở _merge).
            if isinstance(seg_id, bool) or not isinstance(seg_id, (str, int)):
                raise ValueError("segment id phải là chuỗi hoặc số nguyên")
            if isinstance(seg_id, str):
                if len(seg_id) > _MAX_SEGMENT_META_CHARS:
                    raise ValueError(f"segment id too long (>{_MAX_SEGMENT_META_CHARS} chars)")
                _reject_lone_surrogates(seg_id, "segment id")  # Đợt 24 F13 (cùng tokenizer sink)
                total += len(seg_id)
            speaker = seg.get("speaker", seg.get("speaker_id", ""))
            # speaker -> TranslatedSegment.speaker_id (str-ONLY) verbatim; phi-chuỗi (kể cả None khi
            # key hiện diện) nổ ở _merge. Chỉ chấp nhận chuỗi.
            if not isinstance(speaker, str):
                raise ValueError("segment speaker phải là chuỗi")
            if len(speaker) > _MAX_SEGMENT_META_CHARS:
                raise ValueError(f"segment speaker too long (>{_MAX_SEGMENT_META_CHARS} chars)")
            _reject_lone_surrogates(speaker, "segment speaker")  # Đợt 24 F13 (cùng tokenizer sink)
            total += len(speaker)
            # Đợt 19 F8 — start/end/duration đi thẳng vào SỐ HỌC hạ nguồn mà KHÔNG cổng nào
            # kiểm: audio_engine.mix_audio làm int(to_seconds(start)*1000), translation_service
            # làm round(duration, 2). Chuỗi "1e999"/"inf" -> float() Python thành inf -> int(inf)
            # ném OverflowError; "nan" -> int(nan) ném ValueError; duration phi-số -> round() ném
            # TypeError. Các lỗi này nổ MUỘN (sau tải audio + Qwen + tối đa 2000 TTS + Demucs) ->
            # HTTP 500 -> Gateway RETRY toàn bộ pipeline 3 lần (khuếch đại ~3× chi phí GPU, xem
            # index.ts MAX_DISPATCH_ATTEMPTS). Hợp đồng shared-types quy định start/end/duration là
            # SỐ giây HỮU HẠN; chặn tại cổng (422 terminal, 4xx = Gateway FAILED không retry, và
            # xảy ra TRƯỚC mọi compute) mọi giá trị chuẩn hóa ra vô cực/NaN. Dùng CHÍNH to_seconds
            # pipeline dùng để cùng hợp đồng: số / chuỗi số / "HH:MM:SS" hợp lệ đều hữu hạn; rác
            # không parse -> 0.0 (worker coi như 0, không nổ) nên KHÔNG chặn nhầm.
            for _tc in ("start", "end"):
                if _tc in seg and not math.isfinite(to_seconds(seg.get(_tc))):
                    raise ValueError(f"segment {_tc} không phải số hữu hạn")
            # Đợt 24 CC23-01 — HOÀN THIỆN F8: vòng trên chỉ kiểm hữu-hạn Ở GIÂY, nhưng sink THẬT
            # là int(to_seconds(start)*1000) (audio_engine.mix_audio:177) và int((end-start)*1000)
            # (:191). Một giá trị HỮU HẠN ở giây nhưng ×1000 -> vô cực (vd 1e306*1000=1e309=inf; hoặc
            # HIỆU end-start rất lớn: start=-1e305, end=1e305 -> mỗi cái ×1000 hữu hạn nhưng HIỆU
            # ×1000 = 2e308 = inf) khiến int(inf) ném OverflowError — nổ MUỘN trong mix_audio (chạy
            # ở prod: GPT-SoVITS CỤC BỘ sinh clip -> tts_clips không rỗng -> KHÔNG fail-closed sớm)
            # -> HTTP 500 -> Gateway retry TOÀN BỘ pipeline 3×. Kiểm ĐÚNG phép nhân/hiệu mà sink làm
            # (dùng CHÍNH to_seconds + default 0 như consumer nên field vắng/không-parse -> 0, không
            # chặn nhầm). Đây là hoàn thiện F8 (cùng sink/cổng), KHÔNG phải trục mới.
            _s = to_seconds(seg.get("start", 0))
            _e = to_seconds(seg.get("end", 0))
            if not (
                math.isfinite(_s * 1000)
                and math.isfinite(_e * 1000)
                and math.isfinite((_e - _s) * 1000)
            ):
                raise ValueError("segment start/end quá lớn (×1000 tràn vô cực)")
            if "start" in seg and "end" in seg and _e <= _s:
                raise ValueError("segment start phải nhỏ hơn end")
            duration = seg.get("duration")
            if duration is not None and (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
            ):
                # duration KHÔNG đi qua to_seconds — round(duration, 2) tiêu thụ trực tiếp, nên
                # phải là số thực hữu hạn. Loại bool (round(True,2)=1 sẽ nuốt lỗi client âm thầm).
                raise ValueError("segment duration phải là số hữu hạn")
        if total > _MAX_TOTAL_TEXT_CHARS:
            raise ValueError(f"total segment text too long (>{_MAX_TOTAL_TEXT_CHARS} chars)")
        return v

    @field_validator("voice_map")
    @classmethod
    def _bound_voice_map(cls, v: dict) -> dict:
        """Đợt 20 F10 — voice_map (speaker_id -> voice) đi vào model_manager._resolve_voice:
        `VOICE_ID_GENDER.get(chosen)` ném TypeError nếu `chosen` KHÔNG hashable (list/dict), hoặc
        trả `chosen` phi-chuỗi (vd int) -> tts_service `voice.startswith(...)` ném AttributeError.
        Cả hai nổ MUỘN trong vòng TTS (sau tải audio + Qwen) -> HTTP 500 -> Gateway retry toàn bộ
        pipeline 3×. Hợp đồng shared-types là Record<string,string> (CHỈ compile-time; Gateway
        forward verbatim, worker nhận `voice_map: dict` không kiểm). Ép RUNTIME tại cổng: khóa &
        GIÁ TRỊ phải là chuỗi -> 422 terminal (4xx = Gateway FAILED, không retry). CHỈ kiểm KIỂU,
        KHÔNG chặn SỐ LƯỢNG mục: trục "voice_map nhiều mục" đã được Đợt 19 xét và BÁC (map thừa
        chỉ tra-cứu theo speaker_id vốn đã bị _MAX_SEGMENTS chặn -> không phải trục thổi phồng)."""
        if not isinstance(v, dict):
            raise ValueError("voice_map must be an object")
        for k, val in v.items():
            if not isinstance(k, str) or not isinstance(val, str):
                raise ValueError("voice_map keys and values must be strings")
        return v

    @field_validator("target_language", "translation_style", "source_language")
    @classmethod
    def _freetext_well_formed(cls, v):
        """Đợt 24 F13 — target_language/translation_style/source_language đi THẲNG vào
        prompt Qwen (_build_prompt) rồi tokenizer. `Field(max_length=200)` chặn ĐỘ DÀI và
        loại kiểu sai, nhưng KHÔNG kiểm charset: một lone surrogate vẫn là `str` độ dài hợp
        lệ -> lọt cổng -> nổ tokenizer NGOÀI retry -> 500 -> retry 3×. Ép well-formed UTF tại
        cổng (422 terminal). None (source_language mặc định) bỏ qua."""
        if v is not None:
            _reject_lone_surrogates(v, "trường free-text")
        return v

    @field_validator("job_id")
    @classmethod
    def _jobid_well_formed(cls, v: str) -> str:
        """Đợt 25 AMP-JOBID-SURROGATE-01 — job_id là chuỗi client kiểm soát DUY NHẤT mà F13
        (Đợt 24) BỎ SÓT ở CẢ HAI tầng: `str` trần, không validator; Gateway chỉ kiểm
        !jobId/typeof. Một lone surrogate ("job-\\ud800") lọt typeof/length, qua ECDSA verify
        + token-binding (cả claim `jobId` của JWT lẫn body job_id đều do JSON.stringify escape
        rồi json.loads/PyJWT dựng lại CÙNG lone surrogate -> so khớp ở dòng 340), chạy HẾT
        pipeline, rồi `return {"job_id": payload.job_id, ...}` đi vào Starlette JSONResponse
        .render = json.dumps(..., ensure_ascii=False).encode("utf-8"); .encode ném
        UnicodeEncodeError NGAY LÚC render response — SAU khi handler đã return, tức NGOÀI khối
        try/except của endpoint -> 500 KHÔNG bắt được -> Gateway coi 5xx là retryable -> chạy
        lại TOÀN BỘ pipeline GPU tối đa 3× (crash SAU render = khuếch đại Denial-of-Wallet tối
        đa; job ĐÃ render lại bị báo FAILED). Sink KHÁC F13 (tokenizer): đây là serializer
        response, job_id KHÔNG bao giờ vào prompt Qwen (config dựng KHÔNG kèm job_id, dòng
        344-352). Ép well-formed UTF tại cổng (422 terminal) — đối xứng biên Gateway (400)."""
        _reject_lone_surrogates(v, "job_id")
        return v


class TerminatePayload(BaseModel):
    # Gateway ký terminate token kèm jobId VÀ gửi job_id trong body; route terminate ràng
    # buộc token.jobId == payload.job_id (M3-S7) — một terminate token của job A không dùng
    # để cách ly dưới body khai job B. Bắt buộc (không default) để body thiếu job_id bị 422.
    job_id: str
    reason: str = "unspecified"


class AnalyzePayload(BaseModel):
    """M4-S5 — body của /api/worker/analyze (Human-in-the-Loop, bước ANALYZE).

    KHÁC JobPayload (đường render một-lượt): analyze KHÔNG nhận `segments` (nó SINH ra
    chúng) và KHÔNG có kênh plaintext — kết quả (transcript + bản dịch) chỉ đi ra dưới dạng
    CIPHERTEXT ECIES niêm phong tới khoá mã hoá per-device của client (`encryption_public_key`,
    tách biệt khoá ký ECDSA — ADR 0002). Vì thế kênh artifact (upload_url + key + enc-key) là
    BẮT BUỘC; thiếu bất kỳ mảnh nào -> fail-closed (endpoint trả 422) chứ KHÔNG bao giờ
    fallback trả plaintext. target/style/source đi vào prompt Qwen nên bound độ dài như
    JobPayload (chống bloat prompt)."""

    job_id: str
    audio_url: str  # Client chỉ gửi AUDIO (đã tách cục bộ)
    audio_md5: str = ""  # md5 đã ký; analyze_job fail-closed nếu lệch/thiếu (CC33-01)
    target_language: str = Field(max_length=200)
    translation_style: str = Field(max_length=200)
    source_language: str | None = Field(default=None, max_length=200)
    prompt_profile: PromptProfilePayload | None = None
    # Kênh artifact (do GATEWAY cấp sau JWT Trạm 2 — TIN CẬY như result_upload_url, KHÔNG
    # phải bề mặt SSRF do client kiểm như audio_url). Worker niêm phong AnalyzeResult rồi PUT
    # ciphertext lên artifact_upload_url; Gateway HEAD-verify object tại KEY nó tự DERIVE
    # (bỏ qua artifact_key worker gửi) với md5 worker báo TRƯỚC AWAITING_REVIEW. Default ""
    # để giữ thứ tự kiểm auth/quarantine; enforce non-empty tại endpoint (fail-closed 422).
    artifact_upload_url: str = ""
    artifact_key: str = ""
    # Khoá công khai ECDH per-device của client (base64url raw P-256, X9.62 uncompressed).
    # Worker niêm phong kết quả tới đây; KHÔNG BAO GIỜ giải mã. Point sai/không-trên-đường-cong
    # là lỗi client XÁC ĐỊNH -> endpoint map thành 422 terminal.
    encryption_public_key: str = ""
    # Số revision analyze (Gateway ký kèm token + gửi trong body). Endpoint ràng buộc
    # token.attempt == payload.attempt: token cấp cho revision 1 không phát lại vào revision 2.
    attempt: int = 1

    @field_validator("target_language", "translation_style", "source_language")
    @classmethod
    def _freetext_well_formed(cls, v):
        """Đối xứng JobPayload — target/style/source đi vào prompt Qwen (analyze_job ->
        translate_segments -> tokenizer); một lone surrogate lọt Field(max_length) nhưng nổ
        tokenizer NGOÀI retry -> 500 -> Gateway retry 3× (khuếch đại chi phí). Ép well-formed
        UTF tại cổng (422 terminal). None (source mặc định) bỏ qua.

        Ghi chú (đã kiểm nghiệm RED): pydantic-core cũng TỪ CHỐI lone surrogate cho 3 trường
        này vì chúng có Field(max_length=200) — ràng buộc str khiến core round-trip qua Rust
        String (không chứa được surrogate lẻ). Validator này là PHÒNG THỦ HAI LỚP tường minh:
        vẫn chặn nếu ai đó gỡ max_length sau này, và nêu rõ Ý ĐỊNH (đối xứng job_id + JobPayload
        F13). job_id KHÔNG có ràng buộc nên core KHÔNG chặn — validator _jobid_well_formed là
        cổng DUY NHẤT (test job_id RED khi gỡ, freetext vẫn GREEN — bất đối xứng đã xác nhận)."""
        if v is not None:
            _reject_lone_surrogates(v, "trường free-text")
        return v

    @field_validator("job_id")
    @classmethod
    def _jobid_well_formed(cls, v: str) -> str:
        """AMP-JOBID-SURROGATE-01 mở rộng cho analyze — job_id đi vào analyzeJobId của
        AnalyzeResult rồi json.dumps(...).encode('utf-8') lúc niêm phong; một lone surrogate
        ném UnicodeEncodeError trong try -> 500 -> Gateway retry 3× (khuếch đại Denial-of-Wallet).
        Ép well-formed UTF tại parse (422 terminal) TRƯỚC mọi compute — đối xứng biên Gateway.
        job_id là `str` TRẦN (không max_length) nên pydantic-core KHÔNG chặn — validator này là
        cổng DUY NHẤT (đã kiểm nghiệm: test job_id RED khi comment-out dòng dưới)."""
        _reject_lone_surrogates(v, "job_id")
        return v


class RenderPayload(BaseModel):
    """M4-S6 — body của /api/worker/render (Human-in-the-Loop, bước RENDER).

    Sinh đôi của JobPayload cho đường ĐÃ DUYỆT: segments ở đây là manifest ApprovedSegment
    (camelCase: sourceText/translatedText/speaker/emotion/…), KHÁC shape snake-case của
    JobPayload, nhưng chảy vào CÙNG đuôi lồng tiếng (_run_dub_pipeline) qua render_job — nên
    ĐỤNG CÙNG các sink thật. Vì thế cần cổng input-bound RIÊNG cho shape camelCase, ánh xạ
    các phòng thủ đã kiểm nghiệm của JobPayload (F3/F8/F9/F10/F11/F13/CC23-01/AMP-JOBID-…)
    sang đúng sink render THỰC SỰ chạm:

      • translatedText -> tts_service.synthesize(text)          (type/len/lone-surrogate)
      • sourceText     -> (original_text or "").strip()         (AttributeError nếu truthy phi-str)
      • speaker        -> _resolve_voice -> voice_map.get(spk)   (TypeError nếu unhashable)
      • emotion        -> EMOTION_PROSODY.get(emotion)           (TypeError nếu unhashable)
      • start/end      -> int(to_seconds(x)*1000) khi mix        (OverflowError nếu ×1000 -> inf)
      • phần tử seg    -> seg.get(...)                           (AttributeError nếu không phải object)
      • voice_map val  -> tts voice.startswith(...)              (AttributeError nếu phi-str)
      • job_id         -> {"job_id": …} render JSON response     (UnicodeEncodeError nếu lone surrogate)

    KHÔNG gate (không có sink render — TDD không thêm cổng không chứng minh được lỗi):
    target/style/source_language KHÔNG có rủi ro lone-surrogate ở đây vì render KHÔNG gọi
    Qwen (chúng chỉ chọn voice qua dict .get — lone surrogate không làm nổ); `id` không được
    đuôi render đọc. Vệ sinh độ dài/kiểu cho free-text vẫn giữ qua Field(max_length=200)."""

    job_id: str
    audio_url: str  # AUDIO đã tách phía client (worker không chạm video thô)
    audio_md5: str = ""  # md5 đã ký; render_job fail-closed nếu lệch/thiếu (CC33-01)
    target_language: str = Field(max_length=200)  # -> _resolve_voice (chọn giọng theo ngôn ngữ)
    translation_style: str = Field(max_length=200)  # chuyển tiếp từ manifest; render KHÔNG dịch lại
    source_language: str | None = Field(default=None, max_length=200)
    # Dòng dõi Gateway ràng buộc (analyze gốc + hash manifest ĐÃ DUYỆT). Worker TIN thẩm quyền
    # kiểm-manifest của Gateway (dispatchRenderToWorker đã fetch + RE-HASH trước khi lái) nên
    # đây KHÔNG phải sink — chấp nhận cho đúng hợp đồng, không dùng để tính toán. Default ""
    # để không phá thứ tự kiểm auth/binding.
    analyze_job_id: str = ""
    approved_manifest_hash: str = ""
    # Manifest ĐÃ DUYỆT, chuyển tiếp NGUYÊN VĂN — translatedText là nguồn duy nhất (KHÔNG
    # Qwen); TTS chỉ tạo bản đọc TN riêng. voice_map = speakerMapping đã giải quyết.
    segments: list = []
    voice_map: dict = {}
    # R2 result PUT (Gateway cấp sau JWT Trạm 2 — TIN CẬY như JobPayload; host luôn là
    # <account>.r2.cloudflarestorage.com). Default "" giữ tương thích đường chưa bật R2.
    result_upload_url: str = ""
    result_key: str = ""
    # Số render attempt (Gateway ký kèm token dispatch VÀ gửi trong body). Endpoint ràng buộc
    # token.attempt == payload.attempt: token cấp cho attempt 1 KHÔNG phát lại vào attempt 2.
    attempt: int = 1

    @field_validator("segments")
    @classmethod
    def _bound_approved_segments(cls, v: list) -> list:
        """Cổng fail-closed cho segments manifest ĐÃ DUYỆT (camelCase). Đối xứng
        JobPayload._bound_segments nhưng theo đúng field & sink của đường RENDER (đọc
        translatedText/sourceText/speaker/emotion/start/end trong _run_dub_pipeline; KHÔNG
        Qwen, KHÔNG đọc `id`). Vi phạm -> ValueError -> FastAPI 422 (4xx = Gateway FAILED
        terminal, KHÔNG retry render 3× GPU)."""
        if len(v) > _MAX_SEGMENTS:
            raise ValueError(f"too many segments (>{_MAX_SEGMENTS})")
        total = 0
        for seg in v:
            # F9: phần tử segment PHẢI là object — đuôi render đọc seg.get(...); phi-dict ->
            # AttributeError nổ MUỘN (sau tải audio) -> 500 -> retry 3×.
            if not isinstance(seg, dict):
                raise ValueError("segment must be an object")
            # translatedText -> tts_service.synthesize(spoken_text) rồi TN riêng. Phi-chuỗi truthy
            # (số/list) lọt `if not spoken_text` rồi synthesize(<phi-str>) -> nổ giữa vòng TTS.
            # F13: lone surrogate không well-formed UTF -> phá synthesizer. F3/F4: bound độ dài.
            translated_text = seg.get("translatedText", "")
            if not isinstance(translated_text, str):
                raise ValueError("segment translatedText must be a string")
            if len(translated_text) > _MAX_SEGMENT_TEXT_CHARS:
                raise ValueError(f"segment translatedText too long (>{_MAX_SEGMENT_TEXT_CHARS} chars)")
            _reject_lone_surrogates(translated_text, "segment translatedText")
            total += len(translated_text)
            # sourceText -> original_text -> `(original_text or "").strip()` (đếm câu mất bản
            # dịch). Truthy phi-chuỗi (list/dict/số) -> `.strip()` ném AttributeError giữa vòng
            # -> 500 -> retry 3×. Chỉ cần ép KIỂU chuỗi (lone surrogate sống sót .strip, KHÔNG
            # serialize -> không phải sink surrogate); cộng độ dài vào TỔNG (bộ nhớ).
            source_text = seg.get("sourceText", "")
            if not isinstance(source_text, str):
                raise ValueError("segment sourceText must be a string")
            total += len(source_text)
            # speaker -> speaker_id -> voice_map.get(speaker_id) trong _resolve_voice. speaker
            # unhashable (list/dict) -> TypeError giữa vòng TTS -> 500 -> retry 3×. Ép chuỗi
            # (F11). Vắng -> "" (falsy) -> _resolve_voice bỏ qua -> giọng mặc định (an toàn).
            speaker = seg.get("speaker", "")
            if not isinstance(speaker, str):
                raise ValueError("segment speaker phải là chuỗi")
            if len(speaker) > _MAX_SEGMENT_META_CHARS:
                raise ValueError(f"segment speaker too long (>{_MAX_SEGMENT_META_CHARS} chars)")
            total += len(speaker)
            # emotion -> EMOTION_PROSODY.get(emotion). Unhashable (list/dict) -> TypeError giữa
            # vòng TTS -> 500 -> retry 3×. Ép chuỗi (enum ngắn). Vắng -> "NEUTRAL" (an toàn).
            emotion = seg.get("emotion", "NEUTRAL")
            if not isinstance(emotion, str):
                raise ValueError("segment emotion phải là chuỗi")
            if len(emotion) > _MAX_SEGMENT_META_CHARS:
                raise ValueError(f"segment emotion too long (>{_MAX_SEGMENT_META_CHARS} chars)")
            voice_id = seg.get("voiceId")
            if not isinstance(voice_id, str) or not voice_id:
                raise ValueError("segment voiceId must be a non-empty string")
            if len(voice_id) > 128:
                raise ValueError("segment voiceId too long (>128 chars)")
            _reject_lone_surrogates(voice_id, "segment voiceId")
            # start/end -> int(to_seconds(x)*1000) (audio_engine.mix_audio). F8: hữu-hạn ở giây;
            # CC23-01: hữu-hạn Ở CẢ phép ×1000 và HIỆU ×1000 (giá trị hữu hạn ở giây nhưng ×1000
            # -> inf khiến int(inf) ném OverflowError giữa mix -> 500 -> retry 3×). Dùng CHÍNH
            # to_seconds pipeline dùng nên rác không-parse -> 0.0, không chặn nhầm.
            for _tc in ("start", "end"):
                if _tc in seg and not math.isfinite(to_seconds(seg.get(_tc))):
                    raise ValueError(f"segment {_tc} không phải số hữu hạn")
            _s = to_seconds(seg.get("start", 0))
            _e = to_seconds(seg.get("end", 0))
            if not (
                math.isfinite(_s * 1000)
                and math.isfinite(_e * 1000)
                and math.isfinite((_e - _s) * 1000)
            ):
                raise ValueError("segment start/end quá lớn (×1000 tràn vô cực)")
        if total > _MAX_TOTAL_TEXT_CHARS:
            raise ValueError(f"total segment text too long (>{_MAX_TOTAL_TEXT_CHARS} chars)")
        return v

    @field_validator("voice_map")
    @classmethod
    def _bound_voice_map(cls, v: dict) -> dict:
        """F10 — voice_map (speaker_id -> voice) đi vào _resolve_voice: value phi-chuỗi ->
        tts `voice.startswith(...)` ném AttributeError. Ép khóa & GIÁ TRỊ là chuỗi (422
        terminal). Giống JobPayload._bound_voice_map."""
        if not isinstance(v, dict):
            raise ValueError("voice_map must be an object")
        for k, val in v.items():
            if not isinstance(k, str) or not isinstance(val, str):
                raise ValueError("voice_map keys and values must be strings")
        return v

    @model_validator(mode="after")
    def _voice_contract_matches_manifest(self):
        for segment in self.segments:
            speaker = segment.get("speaker")
            if not speaker or self.voice_map.get(speaker) != segment.get("voiceId"):
                raise ValueError("segment voiceId must match speakerMapping")
        return self

    @field_validator("job_id")
    @classmethod
    def _jobid_well_formed(cls, v: str) -> str:
        """AMP-JOBID-SURROGATE-01 cho render — job_id dội lại vào JSON response
        ({"job_id": …}); lone surrogate ném UnicodeEncodeError lúc render response, SAU khi
        đã render GPU (khuếch đại Denial-of-Wallet tối đa) -> 500 KHÔNG bắt được -> retry 3×.
        job_id là `str` TRẦN (không max_length) nên pydantic-core KHÔNG chặn — cổng DUY NHẤT."""
        _reject_lone_surrogates(v, "job_id")
        return v


async def _r2_put_bytes(url: str, content: bytes) -> int:
    """PUT `content` lên presigned URL (R2 Option A) và trả HTTP status.

    Tách RANH GIỚI MẠNG ra một hàm nhỏ để test có thể stub đúng chỗ mạng mà vẫn chạy
    THẬT phần đọc-file/md5/size ở caller (_upload_result_to_r2). URL do Gateway ký với
    X-Amz-SignedHeaders=host (chỉ host được ký), nên PUT KHÔNG cần header đặc biệt và
    KHÔNG gửi Content-MD5 — ETag đơn-phần của R2 CHÍNH là md5 hex của body, khớp đúng giá
    trị Gateway HEAD-verify. follow_redirects=False: presigned R2 trả 200 trực tiếp, một
    3xx là bất thường (chống pivot)."""
    async with httpx.AsyncClient(follow_redirects=False) as client:
        resp = await client.put(url, content=content)
    return resp.status_code


def _read_and_hash_result(file_path: str) -> tuple[bytes, str, str, int]:
    """Read immutable result bytes and calculate checksums in one blocking pass."""
    with open(file_path, "rb") as file:
        content = file.read()
    return (
        content,
        hashlib.md5(content).hexdigest(),
        hashlib.sha256(content).hexdigest(),
        len(content),
    )


async def _upload_result_to_r2(upload_url: str, result_key: str, file_path: str) -> dict:
    """Tải bản render CUỐI lên R2 rồi trả checksum/size của chính bytes đã PUT.

    Bất biến No-Fake-Success: chỉ TRẢ md5 SAU khi bytes THẬT SỰ landed (PUT 2xx). PUT
    hỏng (non-2xx) là lỗi THOÁNG QUA (mạng/R2 nấc) -> ném RuntimeError -> process_audio
    map thành 500 -> Gateway retry; KHÔNG báo checksum cho object chưa tồn tại. Gateway
    còn HEAD-verify độc lập TRƯỚC khi DONE nên đây là phòng thủ hai lớp."""
    # Result tracks can approach the input size cap. Keep disk I/O and both whole-file
    # hashes off FastAPI's loop so health/terminate remain responsive. _run_blocking
    # drains the thread on cancellation before request ownership is released.
    content, result_md5, result_sha256, result_size = await _run_blocking(
        _read_and_hash_result, file_path
    )
    status = await _r2_put_bytes(upload_url, content)
    if status not in (200, 201):
        # Zero-Logging: chỉ nêu status, không nhúng URL (chứa token đã ký).
        raise RuntimeError(f"result upload to R2 failed (status {status})")
    return {
        "result_key": result_key,
        "result_md5": result_md5,
        "result_sha256": result_sha256,
        "result_size": result_size,
    }


def _seal_analyze_result(
    full_result: dict,
    recipient_public_key,
    job_id: str,
    attempt: int,
) -> tuple[bytes, str]:
    """Serialize, encrypt and hash an Analyze artifact outside the request loop."""
    plaintext = json.dumps(full_result, ensure_ascii=False).encode("utf-8")
    envelope = seal_artifact(
        plaintext,
        recipient_public_key,
        analyze_job_id=job_id,
        analyze_revision=attempt,
        payload_schema_version=ANALYZE_RESULT_SCHEMA_VERSION,
    )
    artifact_bytes = json.dumps(envelope).encode("utf-8")
    return artifact_bytes, hashlib.md5(artifact_bytes).hexdigest()


@app.post("/api/worker/process")
async def process_audio(
    payload: JobPayload, token: dict = Depends(require_action("dispatch"))
):
    """Endpoint nội bộ nhận job từ Gateway. Chỉ chạy khi có JWT DISPATCH (Trạm 2):
    require_action("dispatch") đã ép act == 'dispatch' nên token terminate không lái
    được đường render."""
    _raise_if_quarantined()

    # Token binding (Trạm 2): Gateway ký JWT RIÊNG cho jobId này (claim `jobId`). Nếu
    # token bị rò rỉ, nó chỉ dùng lại được cho ĐÚNG job đó — không thể tái dùng để xử
    # lý job/nội dung khác. Token có role hợp lệ nhưng jobId khác payload => từ chối.
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")

    # Attempt binding (M3-S7): token dispatch mang claim `attempt` của ĐÚNG lần render
    # nó được cấp. Token của attempt 1 KHÔNG thể bị phát lại vào slot attempt 2 (body
    # khai attempt khác claim => từ chối). Gateway luôn ký kèm attempt cho dispatch.
    if token.get("attempt") != payload.attempt:
        raise HTTPException(status_code=403, detail="Token/Attempt Mismatch")

    try:
        prompt_profile = getattr(payload, "prompt_profile", None)
        config = {
            "target_language": payload.target_language,
            "style": payload.translation_style,
            "segments": payload.segments,
            "voice_map": payload.voice_map,
            "source_language": payload.source_language,
            # md5 ĐÃ KÝ để process_job kiểm toàn vẹn audio tải-về (fail-closed nếu lệch/thiếu).
            "audio_md5": payload.audio_md5,
        }
        if prompt_profile is not None:
            config["prompt_profile"] = prompt_profile.model_dump(exclude_none=True)

        # Tuần tự hoá qua semaphore: một GPU chỉ chạy MỘT pipeline nặng tại một thời điểm
        # (Whisper + Qwen đã thường trú VRAM; thêm stem Demucs/AudioSeal song song -> OOM).
        # KHÔNG có sleep/giả lập — thời gian thực tế chính là căn cứ để Trạm 3 (Gateway)
        # phát hiện gian lận. Mỗi request chạy pipeline ĐỘC LẬP (không gộp theo job_id) để
        # kết quả không bao giờ chảy xuyên tenant; xem ghi chú tại khai báo semaphore.
        async with _admitted_gpu_request() as release_gpu_slot:
            result = await _run_gpu_job(
                model_manager.process_job(payload.audio_url, config)
            )
            release_gpu_slot()

            # M2-S3: nếu Gateway đã cấp URL PUT kết quả cho attempt này, tải bản render cuối
            # lên R2 tại object key Gateway DERIVE (không phải chuỗi worker tự đặt) rồi báo
            # md5/size để Gateway HEAD-verify TRƯỚC khi DONE. Chỉ chạy khi CÓ url (giữ tương
            # thích đường dispatch chưa bật R2) VÀ result mang dubbed_audio (đường dẫn file
            # cuối). PUT hỏng -> RuntimeError -> 500 (Gateway retry), không fake success.
            upload_url = payload.result_upload_url
            dubbed = result.get("dubbed_audio") if isinstance(result, dict) else None
            if upload_url and dubbed:
                result.update(
                    await _upload_result_to_r2(upload_url, payload.result_key, dubbed)
                )

            # M3-S8: đóng dấu schema_version + DỘI LẠI job_id/attempt để Gateway VALIDATE
            # ĐẦY ĐỦ response TRƯỚC khi DONE (đúng job, đúng attempt, status='success',
            # đúng phiên bản hợp đồng). result.status='success' do process_job đặt.
            return {
                "schema_version": WORKER_RESPONSE_SCHEMA_VERSION,
                "job_id": payload.job_id,
                "attempt": payload.attempt,
                "result": result,
            }
    except HTTPException:
        raise
    except AudioIntegrityError:
        # Đợt 33 CC33-01: bytes audio lệch/thiếu md5 đã ký là lỗi XÁC ĐỊNH — tải lại CÙNG
        # object sẽ hỏng y hệt. Trả 422 (4xx) để Gateway (index.ts) đánh FAILED TERMINAL,
        # KHÔNG coi là 5xx thoáng qua rồi retry tải lại tối đa 1 GiB × MAX_DISPATCH_ATTEMPTS
        # (khuếch đại băng thông/độ trễ mà không bao giờ thành công). Zero-Logging: chỉ nêu
        # bản chất, không lộ url/md5/nội dung.
        raise HTTPException(
            status_code=422,
            detail="Audio tải về không toàn vẹn (md5 lệch hoặc thiếu) — từ chối (fail-closed).",
        )
    except TTSError as error:
        raise _tts_http_error(error)
    except Exception as e:
        # WORKER_DEBUG (chỉ bật trên hộp GPU của dev): in traceback ĐẦY ĐỦ ra STDERR của
        # tiến trình worker để chẩn đoán. KHÔNG BAO GIỜ đi vào HTTP response / KV / client —
        # Zero-Logging vẫn nguyên vẹn (response bên dưới chỉ có TÊN loại lỗi). Mặc định tắt.
        if os.environ.get("WORKER_DEBUG"):
            traceback.print_exc()
        # Zero-Logging: str(e) VÀ type(e).__name__ đều có thể là dữ liệu nội bộ (tên class
        # có thể lộ chi tiết triển khai; message nhúng đường dẫn / URL kèm ?text=<kịch bản
        # plaintext> / dấu vết hệ thống). Response chỉ mang MÃ CANONICAL đã sanitize —
        # khớp FAILURE_REASONS.INTERNAL_ERROR bên Gateway/shared-types. Tên class & traceback
        # chỉ ra STDERR ở nhánh WORKER_DEBUG bên trên, không đi vào response.
        raise HTTPException(
            status_code=500,
            detail=_INTERNAL_ERROR_REASON,
        )


@app.post("/api/worker/analyze")
async def analyze_audio(
    payload: AnalyzePayload, token: dict = Depends(require_action("analyze"))
):
    """M4-S5 — bước ANALYZE (Human-in-the-Loop): bóc băng + dịch + độ tin cậy + diarization,
    rồi NIÊM PHONG kết quả (ECIES) tới khoá client và tải CIPHERTEXT lên R2. Gateway chỉ nhận
    CON TRỎ (key + md5 + metadata không nhạy cảm) — KHÔNG BAO GIỜ thấy plaintext (zero-knowledge).

    require_action("analyze") đã ép act == 'analyze' nên token dispatch/terminate không lái
    được đường này. Mọi cổng auth/binding/quarantine chạy TRƯỚC khi chạm GPU."""
    _raise_if_quarantined()

    # Token binding (Trạm 2) — token ký RIÊNG cho jobId + attempt này; rò rỉ chỉ tái dùng
    # được cho ĐÚNG (job, revision) đó. Kiểm TRƯỚC pipeline: GPU không hề bị chạm nếu lệch.
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")
    if token.get("attempt") != payload.attempt:
        raise HTTPException(status_code=403, detail="Token/Attempt Mismatch")

    # Fail-closed: analyze KHÔNG có kênh plaintext. Thiếu nơi tải ciphertext (upload_url/key)
    # HOẶC khoá người nhận -> không có gì để niêm phong/gửi -> từ chối 422 TRƯỚC GPU. KHÔNG
    # BAO GIỜ fallback trả plaintext. 4xx = Gateway đánh FAILED terminal (không retry/quarantine).
    if not (
        payload.artifact_upload_url
        and payload.artifact_key
        and payload.encryption_public_key
    ):
        raise HTTPException(
            status_code=422, detail="Analyze requires the encrypted artifact channel"
        )

    # Point sai/không-trên-đường-cong P-256 là lỗi client XÁC ĐỊNH -> 422 terminal, TRƯỚC GPU.
    try:
        recipient_pub = load_recipient_public_key(payload.encryption_public_key)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid encryption public key")

    try:
        prompt_profile = getattr(payload, "prompt_profile", None)
        config = {
            "target_language": payload.target_language,
            "style": payload.translation_style,
            "source_language": payload.source_language,
            # md5 ĐÃ KÝ để analyze_job kiểm toàn vẹn audio tải-về (fail-closed nếu lệch/thiếu).
            "audio_md5": payload.audio_md5,
        }
        if prompt_profile is not None:
            config["prompt_profile"] = prompt_profile.model_dump(exclude_none=True)

        # Tuần tự hoá qua CÙNG semaphore với render: một GPU chạy MỘT pipeline nặng tại một
        # thời điểm (Whisper + Qwen thường trú VRAM). Thời gian thực = căn cứ chống gian lận
        # của Trạm 3 (không sleep/giả lập).
        async with _admitted_gpu_request() as release_gpu_slot:
            analyze_content = await _run_gpu_job(
                model_manager.analyze_job(payload.audio_url, config)
            )
            release_gpu_slot()

            # Bọc nội dung ML thuần thành AnalyzeResult CÓ PHIÊN BẢN + ràng buộc dòng dõi
            # (analyzeJobId/analyzeRevision) — đây là plaintext client sẽ giải mã + kiểm.
            full_result = {
                "schema_version": ANALYZE_RESULT_SCHEMA_VERSION,
                "analyzeJobId": payload.job_id,
                "analyzeRevision": payload.attempt,
                "sourceLanguage": analyze_content["sourceLanguage"],
                "targetLanguage": analyze_content["targetLanguage"],
                "segments": analyze_content["segments"],
                "diarization": analyze_content["diarization"],
            }

            # Serialization, encryption and hashing are CPU work. Keep them off the
            # request loop while retaining admitted-request ownership until R2 is durable.
            artifact_bytes, artifact_md5 = await _run_blocking(
                _seal_analyze_result,
                full_result,
                recipient_pub,
                payload.job_id,
                payload.attempt,
            )

            status = await _r2_put_bytes(payload.artifact_upload_url, artifact_bytes)
            if status not in (200, 201):
                # PUT hỏng = lỗi THOÁNG QUA (mạng/R2 nấc) -> RuntimeError -> 500 (Gateway retry).
                # KHÔNG báo md5 cho object chưa landed (No-Fake-Success). Zero-Logging: chỉ status.
                raise RuntimeError(f"artifact upload to R2 failed (status {status})")

            # M3-S8 (analyze twin): đóng dấu schema_version + dội lại job_id/attempt để Gateway
            # VALIDATE ĐẦY ĐỦ response TRƯỚC AWAITING_REVIEW. CHỈ con trỏ ciphertext + metadata
            # không nhạy cảm — KHÔNG transcript/bản dịch nào rời ciphertext (zero-knowledge M4 #4).
            return {
                "schema_version": WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION,
                "job_id": payload.job_id,
                "attempt": payload.attempt,
                "result": {
                    "status": "success",
                    "artifact_key": payload.artifact_key,
                    "artifact_md5": artifact_md5,
                    "alg": ARTIFACT_ALG,
                    "diarization": analyze_content["diarization"],
                    "segment_count": len(analyze_content["segments"]),
                },
            }
    except HTTPException:
        raise
    except AudioIntegrityError:
        # CC33-01 (đường analyze): md5 lệch/thiếu là lỗi XÁC ĐỊNH -> 422 terminal, KHÔNG 500
        # bị retry tải lại. Zero-Logging: chỉ nêu bản chất, không lộ url/md5/nội dung.
        raise HTTPException(
            status_code=422,
            detail="Audio tải về không toàn vẹn (md5 lệch hoặc thiếu) — từ chối (fail-closed).",
        )
    except Exception:
        # WORKER_DEBUG: traceback ĐẦY ĐỦ ra STDERR (không vào response/KV/client). Response
        # chỉ mang MÃ CANONICAL đã sanitize (Zero-Logging) — khớp FAILURE_REASONS.INTERNAL_ERROR.
        if os.environ.get("WORKER_DEBUG"):
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_REASON)


@app.post("/api/worker/analyze/submit", status_code=202)
async def submit_analyze(
    payload: AnalyzePayload, token: dict = Depends(require_action("analyze"))
):
    """Start ANALYZE and return before model inference completes.

    RunPod's public proxy limits one HTTP response to roughly 90 seconds.  This route
    keeps the authenticated, body-bound analyze contract but moves the existing
    synchronous handler into a background task.  A deterministic ``job_id/attempt``
    key makes queue redelivery safe: the same task/result is returned instead of
    admitting a second GPU pipeline.
    """
    _raise_if_quarantined()
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")
    if token.get("attempt") != payload.attempt:
        raise HTTPException(status_code=403, detail="Token/Attempt Mismatch")
    if not (payload.artifact_upload_url and payload.artifact_key and payload.encryption_public_key):
        raise HTTPException(status_code=422, detail="Analyze requires the encrypted artifact channel")
    try:
        load_recipient_public_key(payload.encryption_public_key)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid encryption public key")

    jobs = _async_analyze_jobs()
    key = f"{payload.job_id}:{payload.attempt}"
    entry = jobs.get(key)
    if entry is None:
        task = asyncio.create_task(analyze_audio(payload, token))
        entry = {"task": task, "submitted_at": asyncio.get_running_loop().time()}
        jobs[key] = entry

        def _remember_done(done_task: asyncio.Task) -> None:
            entry["task"] = None
            entry["finished_at"] = asyncio.get_running_loop().time()
            try:
                entry["result"] = done_task.result()
            except HTTPException as error:
                entry["error"] = "worker_rejected" if error.status_code < 500 else _INTERNAL_ERROR_REASON
            except asyncio.CancelledError:
                entry["error"] = "worker_cancelled"
            except Exception:
                entry["error"] = _INTERNAL_ERROR_REASON

        task.add_done_callback(_remember_done)

    return {
        "schema_version": WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION,
        "job_id": payload.job_id,
        "attempt": payload.attempt,
        "status": "completed" if entry.get("task") is None and "result" in entry else "queued",
    }


@app.get("/api/worker/analyze/status/{job_id}/{attempt}")
async def analyze_status(
    job_id: str,
    attempt: int,
    token: dict = Depends(require_action("analyze")),
):
    """Bounded read-only poll for an async ANALYZE task.

    The status poll reuses the analyze JWT action but binds it to the exact
    job/attempt.  Its GET body is empty and therefore carries no compute payload;
    result payloads are the same strict WorkerAnalyzeResponse shape as the
    synchronous endpoint, while failures expose only a canonical class.
    """
    if token.get("jobId") != job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")
    if attempt <= 0:
        raise HTTPException(status_code=422, detail="Invalid attempt")
    if token.get("attempt") != attempt:
        raise HTTPException(status_code=403, detail="Token/Attempt Mismatch")
    entry = _async_analyze_jobs().get(f"{job_id}:{attempt}")
    if entry is None:
        raise HTTPException(status_code=404, detail="Async analyze job not found")
    task = entry.get("task")
    if task is not None and not task.done():
        return {"job_id": job_id, "attempt": attempt, "status": "running"}
    if "result" in entry:
        return {
            "job_id": job_id,
            "attempt": attempt,
            "status": "completed",
            "response": entry["result"],
        }
    return {
        "job_id": job_id,
        "attempt": attempt,
        "status": "failed",
        "error": entry.get("error", _INTERNAL_ERROR_REASON),
    }


@app.post("/api/worker/render")
async def render_audio(
    payload: RenderPayload,
    response: Response,
    token: dict = Depends(require_action("render")),
):
    """M4-S6 — bước RENDER (Human-in-the-Loop): lồng tiếng từ manifest ĐÃ DUYỆT. Sinh đôi
    của /process cho đường ĐÃ DUYỆT — cùng kênh kết quả PLAINTEXT (bản .wav lồng tiếng lên
    R2, tái dùng WorkerResponseSchema + seam _upload_result_to_r2), KHÁC ở lõi ML: render_job
    lấy `translatedText` làm nguồn duy nhất và KHÔNG BAO GIỜ gọi Qwen/ASR (đã chứng minh ở
    test_render_job.py); TTS tạo bản đọc TN riêng. Ở BIÊN này ta chốt: endpoint chuyển tiếp segments đã duyệt vào config
    KHÔNG ĐỔI và tự giới hạn act='render' (token dispatch/analyze KHÔNG lái được đường này).

    require_action("render") đã ép act == 'render'. Mọi cổng auth/binding/quarantine chạy
    TRƯỚC khi chạm GPU."""
    _raise_if_quarantined()

    # Token binding (Trạm 2): Gateway ký JWT RIÊNG cho (jobId, attempt) render này. Token rò rỉ
    # chỉ tái dùng được cho ĐÚNG job + ĐÚNG attempt đó — không lái được job/nội dung khác, và
    # token attempt 1 không phát lại vào attempt 2. Kiểm TRƯỚC GPU: lệch => 403, GPU không chạm.
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")
    if token.get("attempt") != payload.attempt:
        raise HTTPException(status_code=403, detail="Token/Attempt Mismatch")

    # Render chỉ được coi là hoàn tất sau khi artifact đã nằm bền vững trong R2. Thiếu URL
    # PUT hoặc object key là lỗi dispatch xác định: từ chối TRƯỚC GPU, không quay về hành vi
    # legacy trả đường dẫn file tạm chỉ sống cùng worker.
    if not payload.result_upload_url or not payload.result_key:
        raise HTTPException(
            status_code=422, detail="Render requires the durable result artifact channel"
        )

    try:
        # Manifest ĐÃ DUYỆT chuyển tiếp NGUYÊN VĂN: segments = ApprovedSegment (camelCase), đọc
        # translatedText làm nguồn cho bản đọc TN; voice_map đã giải quyết; target_language
        # để chọn giọng; audio_md5 ĐÃ KÝ để render_job kiểm toàn vẹn (fail-closed nếu lệch/thiếu).
        # KHÔNG có style/source_language ở đây — render KHÔNG dịch lại nên render_job không đọc chúng.
        config = {
            "target_language": payload.target_language,
            "voice_map": payload.voice_map,
            "segments": payload.segments,
            "audio_md5": payload.audio_md5,
            # Internal only: never accepted from the client/Gateway body. The worker
            # reports the aggregate through a response header so the JSON contract and
            # persisted user result remain unchanged.
            "_collect_gpu_memory_metrics": gpu_memory_metrics_enabled(),
        }

        # Tuần tự hoá qua CÙNG semaphore với process/analyze: một GPU chạy MỘT pipeline nặng tại
        # một thời điểm. Thời gian thực = căn cứ chống gian lận của Trạm 3 (không sleep/giả lập).
        async with _admitted_gpu_request() as release_gpu_slot:
            result = await _run_gpu_job(
                model_manager.render_job(payload.audio_url, config)
            )
            release_gpu_slot()

            gpu_memory_metrics = (
                result.pop("gpu_memory_metrics", None)
                if isinstance(result, dict)
                else None
            )
            gpu_memory_header = (
                encode_gpu_memory_metrics(gpu_memory_metrics)
                if gpu_memory_metrics is not None
                else None
            )

            # M2-S3 (đường render): tải bản lồng tiếng cuối lên R2 tại key Gateway DERIVE rồi báo
            # md5/size để Gateway HEAD-verify TRƯỚC khi DONE. Kênh R2 đã được ép bắt buộc trước GPU;
            # thiếu output hoặc PUT hỏng đều là 500 để Gateway retry, tuyệt đối không trả success với
            # đường dẫn temp chỉ sống cùng worker.
            dubbed = result.get("dubbed_audio") if isinstance(result, dict) else None
            if not isinstance(dubbed, str) or not dubbed:
                raise RuntimeError("render completed without a durable output candidate")
            result.update(
                await _upload_result_to_r2(
                    payload.result_upload_url, payload.result_key, dubbed
                )
            )
            if gpu_memory_header is not None:
                response.headers[GPU_MEMORY_METRICS_HEADER] = gpu_memory_header

            # M3-S8: đóng dấu schema_version + DỘI LẠI job_id/attempt để Gateway VALIDATE ĐẦY ĐỦ
            # response TRƯỚC khi DONE. Render tái dùng WorkerResponseSchema (giống /process).
            return {
                "schema_version": WORKER_RESPONSE_SCHEMA_VERSION,
                "job_id": payload.job_id,
                "attempt": payload.attempt,
                "result": result,
            }
    except HTTPException:
        raise
    except AudioIntegrityError:
        # CC33-01 (đường render): bytes audio lệch/thiếu md5 ĐÃ KÝ là lỗi XÁC ĐỊNH — tải lại CÙNG
        # object hỏng y hệt. 422 (4xx) để Gateway đánh FAILED TERMINAL, KHÔNG retry tải lại +
        # render lại GPU 3× (khuếch đại Denial-of-Wallet). Zero-Logging: chỉ nêu bản chất.
        raise HTTPException(
            status_code=422,
            detail="Audio tải về không toàn vẹn (md5 lệch hoặc thiếu) — từ chối (fail-closed).",
        )
    except TTSError as error:
        raise _tts_http_error(error)
    except Exception:
        # WORKER_DEBUG: traceback ĐẦY ĐỦ ra STDERR (không vào response/KV/client). Response chỉ
        # mang MÃ CANONICAL đã sanitize (Zero-Logging) — khớp FAILURE_REASONS.INTERNAL_ERROR.
        if os.environ.get("WORKER_DEBUG"):
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_REASON)


@app.get("/api/worker/voices")
async def worker_voice_capabilities():
    """Expose only the versioned, sanitized local voice catalog to the Gateway."""
    return {
        "schema_version": WORKER_VOICE_CAPABILITIES_SCHEMA_VERSION,
        "catalog": await tts_service.capability_catalog(probe=True),
    }


def _model_health_snapshot() -> dict:
    """Reduce live residency telemetry to bounded readiness facts.

    The full residency map is useful inside the process but contains implementation
    topology that does not belong in the public health response. Any malformed or
    unavailable evidence fails closed without reflecting raw values or exceptions.
    """
    fallback = {
        "core_ready": False,
        "device": "unknown",
        "diarization_resident": False,
        "audioseal_resident": False,
    }
    try:
        status = model_manager.residency_status()
    except Exception:
        return fallback
    if not isinstance(status, dict):
        return fallback

    raw_device = status.get("device")
    device = (
        raw_device
        if isinstance(raw_device, str) and raw_device in {"cuda", "cpu"}
        else "unknown"
    )
    components = status.get("components")
    if not isinstance(components, dict):
        components = {}

    def resident(component: str) -> bool:
        value = components.get(component)
        return bool(
            isinstance(value, dict)
            and value.get("process_resident") is True
            and value.get("device") == "cuda"
        )

    return {
        "core_ready": bool(
            status.get("core_models_resident") is True and device == "cuda"
        ),
        "device": device,
        "diarization_resident": resident("diarization"),
        "audioseal_resident": resident("audioseal"),
    }


async def _worker_readiness_snapshot() -> tuple[dict, dict]:
    """Collect one live snapshot for both private and orchestrator health APIs.

    Keeping the TTS catalog in this snapshot prevents the authenticated readiness
    route from probing the sidecar once for worker health and again for voice
    capability. Both outward responses are sanitized fixed-shape dictionaries.
    """
    # ADR 0003: readiness must expose honest diarization degradation without leaking the
    # gated model id, HF token, cache path, or raw load exception. The adapter never loads a
    # model from this probe; it only reports capability and whether the lazy singleton is resident.
    from src.diarization_service import diarization_required, diarization_service

    quarantined = bool(WORKER_STATE["quarantined"])
    # A quarantined worker is already unavailable. Avoid delaying the kill-switch
    # readiness response on an unnecessary sidecar network probe.
    catalog, model_snapshot, diarization, audio_enhancements = await asyncio.gather(
        tts_service.capability_catalog(probe=not quarantined),
        asyncio.to_thread(_model_health_snapshot),
        asyncio.to_thread(diarization_service.health_status),
        asyncio.to_thread(audio_engine.enhancement_health_status),
    )
    required = local_tts_required()
    tts_ready = catalog.get("ready") is True
    core_ready = model_snapshot["core_ready"]
    diarization_is_required = diarization_required()
    diarization_ready = (
        not diarization_is_required
        or (
            diarization.get("available") is True
            and diarization.get("pipeline_loaded") is True
            and model_snapshot["diarization_resident"]
        )
    )
    enhancements_ready = all(
        (
            capability.get("available") is True
            and (
                name != "audioseal"
                or model_snapshot["audioseal_resident"]
            )
        )
        or capability.get("required") is not True
        for name, capability in audio_enhancements.items()
    )
    ready = (
        not quarantined
        and core_ready
        and diarization_ready
        and (tts_ready or not required)
        and enhancements_ready
    )
    private_response = {
        "schema_version": WORKER_READINESS_SCHEMA_VERSION,
        "ready": ready,
        "catalog": catalog,
    }
    public_response = {
        "status": "ok" if ready else "not_ready",
        "models_loaded": core_ready,
        "device": model_snapshot["device"],
        "diarization": diarization,
        "audio_enhancements": audio_enhancements,
        "tts": {
            "available": tts_ready,
            "required": required,
            "mode": "local",
            "profile_count": len(catalog.get("profiles", [])),
            "reason": "" if tts_ready else (catalog.get("reason") or "unavailable"),
        },
    }
    return private_response, public_response


@app.get("/api/worker/readiness")
async def worker_readiness(
    token: dict = Depends(require_action("probe")),
):
    """Return strict worker readiness to the authenticated Gateway.

    A reachable but not-ready worker still returns HTTP 200 with ``ready=false`` so
    the Gateway can distinguish configuration/model warm-up from transport failure.
    The response deliberately contains no URL, key, model path, or raw exception.
    """
    readiness, _ = await _worker_readiness_snapshot()
    return readiness


@app.get("/health")
async def health():
    """Kiểm tra sống/sẵn-sàng — KHÔNG cần JWT (Docker HEALTHCHECK, load balancer, và
    orchestrator RunPod/Modal cần probe được TRƯỚC khi có khoá Gateway).

    Zero-Logging: CHỈ trả cờ boolean + enum thiết bị ('cuda'/'cpu'/'unknown'). Tuyệt đối
    không lộ đường dẫn, token, model_id, hay bất kỳ nội dung nào. `models_loaded=false`
    nghĩa là residency của core model chưa được chứng minh -> chưa nên nhận job."""
    readiness, response = await _worker_readiness_snapshot()
    return JSONResponse(
        content=response,
        status_code=200 if readiness["ready"] else 503,
    )


@app.post("/api/worker/terminate")
async def terminate(
    payload: TerminatePayload, token: dict = Depends(require_action("terminate"))
):
    """Trạm 3 enforcement: Gateway ra lệnh cách ly worker khi phát hiện bất thường.

    require_action("terminate") ép act == 'terminate' nên token dispatch không lái được
    đường cách ly. Ở môi trường serverless thật, việc "rút phích" cụm GPU do API của nhà
    cung cấp (RunPod/Modal) thực hiện; tại đây ta bật cờ để worker từ chối job mới ngay.
    """
    # Job binding (M3-S7): terminate token của job A KHÔNG dùng để cách ly dưới body
    # khai job B (một token rò rỉ chỉ tác động đúng job nó được cấp). Gateway ký kèm
    # jobId cho terminate VÀ gửi job_id trong body để đối chiếu.
    if token.get("jobId") != payload.job_id:
        raise HTTPException(status_code=403, detail="Token/Job Mismatch")
    WORKER_STATE["quarantined"] = True
    _cancel_active_gpu_jobs()
    return {"status": "quarantined", "reason": payload.reason}


# M3-S7: /api/worker/download (đọc file temp tuỳ ý) và /api/worker/upload (ghi
# mạng→đĩa) đã bị GỠ. Sau khi chuyển sang R2 (M2), KHÔNG còn caller nào gọi hai
# endpoint này (Gateway đọc/ghi kết quả trực tiếp qua binding R2 gốc). Chúng là
# primitive đọc/ghi filesystem tuỳ ý còn sót; least-privilege = xoá hẳn (404), không
# chỉ khoá bằng auth. Xem test_worker_jwt_scope.py::test_{download,upload}_endpoint_removed.


if __name__ == "__main__":
    # Worker KHÔNG mở IP public — chỉ bind loopback. Gateway (trong cùng mạng riêng
    # / qua tunnel nội bộ) là đầu vào hợp lệ duy nhất.
    import uvicorn

    # Mặc định loopback (không mở IP public). ENV cho phép orchestrator (RunPod/Modal)
    # gán host/port trong mạng riêng mà không sửa mã.
    host = os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("WORKER_PORT", "8000"))
    # Zero-Logging: TẮT access log. Dòng access log của uvicorn ghi nguyên request line
    # (path + query) ra stdout/log tập trung — có thể rò rỉ job_id / tham số nhạy cảm.
    # Giữ tắt như một backstop Zero-Logging bất kể route nào đang mở.
    uvicorn.run(app, host=host, port=port, access_log=False)
