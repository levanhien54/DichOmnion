import asyncio
import math
import os
import tempfile
import traceback

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import jwt

from src.model_manager import ModelManager
from src.audio_service import AudioIntegrityError
from src.timecode import to_seconds

# Khởi tạo Quản lý Mô hình AI (singleton, thường trú VRAM)
model_manager = ModelManager()
security = HTTPBearer()

# Trạng thái vận hành. Trạm 3 (Gateway) có thể ra lệnh cách ly (quarantine) worker
# này khi phát hiện gian lận/timeout; khi đó worker từ chối mọi job mới.
WORKER_STATE = {"quarantined": False}

# Hàng đợi 1-job (mặc định) cho MỘT GPU 24GB: Whisper + Qwen 4B đã thường trú VRAM;
# chạy song song nhiều job sẽ nạp thêm stem Demucs/AudioSeal và bộ nhớ trung gian ->
# nguy cơ OOM (tràn VRAM) và chết cả worker. Semaphore ép tuần tự hoá pipeline nặng.
# WORKER_MAX_CONCURRENT_JOBS cho phép nới ra nếu về sau chạy trên GPU lớn hơn — nhưng
# cổng xác thực/quarantine ở TRƯỚC semaphore nên job trái phép không bao giờ chiếm slot.
_MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("WORKER_MAX_CONCURRENT_JOBS", "1")))
_JOB_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_JOBS)

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
            # md5 ĐÃ KÝ để process_job kiểm toàn vẹn audio tải-về (fail-closed nếu lệch/thiếu).
            "audio_md5": payload.audio_md5,
        }

        # Tuần tự hoá qua semaphore: một GPU chỉ chạy MỘT pipeline nặng tại một thời điểm
        # (Whisper + Qwen đã thường trú VRAM; thêm stem Demucs/AudioSeal song song -> OOM).
        # KHÔNG có sleep/giả lập — thời gian thực tế chính là căn cứ để Trạm 3 (Gateway)
        # phát hiện gian lận. Mỗi request chạy pipeline ĐỘC LẬP (không gộp theo job_id) để
        # kết quả không bao giờ chảy xuyên tenant; xem ghi chú tại khai báo semaphore.
        async with _JOB_SEMAPHORE:
            result = await model_manager.process_job(payload.audio_url, config)

        return {"job_id": payload.job_id, "result": result}
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
    except Exception as e:
        # WORKER_DEBUG (chỉ bật trên hộp GPU của dev): in traceback ĐẦY ĐỦ ra STDERR của
        # tiến trình worker để chẩn đoán. KHÔNG BAO GIỜ đi vào HTTP response / KV / client —
        # Zero-Logging vẫn nguyên vẹn (response bên dưới chỉ có TÊN loại lỗi). Mặc định tắt.
        if os.environ.get("WORKER_DEBUG"):
            traceback.print_exc()
        # Zero-Logging: str(e) có thể nhúng đường dẫn nội bộ / URL kèm ?text=<kịch bản
        # plaintext> / dấu vết hệ thống. Client chỉ nhận TÊN loại lỗi, không nhận nội dung thô.
        raise HTTPException(
            status_code=500,
            detail=f"Xử lý job thất bại ({type(e).__name__})",
        )


@app.get("/health")
async def health():
    """Kiểm tra sống/sẵn-sàng — KHÔNG cần JWT (Docker HEALTHCHECK, load balancer, và
    orchestrator RunPod/Modal cần probe được TRƯỚC khi có khoá Gateway).

    Zero-Logging: CHỈ trả cờ boolean + enum thiết bị ('cuda'/'cpu'). Tuyệt đối không
    lộ đường dẫn, token, model_id, hay bất kỳ nội dung nào. `models_loaded=false` nghĩa
    là worker còn đang nạp trọng số lên VRAM (cold start) -> chưa nên nhận job."""
    return {
        "status": "ok",
        "models_loaded": bool(model_manager.is_loaded),
        "device": model_manager.device,
    }


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
    # realpath (KHÔNG chỉ abspath) để phân giải symlink -> chặn lối thoát qua liên kết mềm.
    temp_dir = os.path.realpath(tempfile.gettempdir())
    real_path = os.path.realpath(path)

    # Chống directory traversal: file PHẢI nằm TRONG thư mục temp. Kiểm bằng ranh giới
    # separator (temp_dir + os.sep), KHÔNG phải startswith prefix trần — nếu không thì
    # thư mục cạnh nhau chia sẻ prefix (vd '<temp>_evil' cạnh '<temp>') sẽ lọt qua.
    if not (real_path == temp_dir or real_path.startswith(temp_dir + os.sep)):
        raise HTTPException(status_code=403, detail="Access Denied: Can only access temp files")
    if not (real_path.endswith(".wav") or real_path.endswith(".mp3") or real_path.endswith(".mp4")):
        raise HTTPException(status_code=403, detail="Access Denied: Invalid file type")
    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "video/mp4" if real_path.endswith(".mp4") else "audio/wav"
    return FileResponse(real_path, media_type=media_type)


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
    # Zero-Logging: TẮT access log. Dòng access log của uvicorn ghi nguyên request line,
    # mà /api/worker/download nhận đường dẫn temp NỘI BỘ qua query (?path=/tmp/...); bật
    # access log sẽ rò rỉ đường dẫn nội bộ (và mọi query khác) ra stdout/log tập trung.
    uvicorn.run(app, host=host, port=port, access_log=False)
