# GPU Worker (FastAPI)

Máy chủ suy luận thường trú của OmniVoice V2. Nhận **URL audio** (không nhận video), chạy
pipeline lồng tiếng trên VRAM GPU: **Whisper** (ASR) → **Qwen** (dịch cục bộ, thay cloud để
đóng lỗ rò transcript) → **TTS** (GPT-SoVITS / edge-tts) → **trộn âm** (pydub, auto-ducking).
Whisper + Qwen **thường trú VRAM**; xử lý tuần tự qua semaphore size-1. Worker **không bao
giờ** lộ IP công khai — chỉ nhận traffic qua `cloudflared` tunnel (xem
[docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md)).

## Chạy cục bộ

```bash
uv sync                                        # dependency mặc định (CPU-safe)
uv run uvicorn src.main:app --port 8000
```

Extra tuỳ chọn (cài khi triển khai GPU thật — fail-soft nếu thiếu, báo trung thực trong `notes`):

```bash
uv sync --extra enhance      # torchaudio + demucs + audioseal (tách nền / watermark)
uv sync --extra local-llm    # transformers + accelerate (Qwen dịch cục bộ in-process)
uv sync --extra dev          # pytest + pytest-asyncio
```

## Cấu hình

Xem **`.env.example`** (nguồn chuẩn của các biến môi trường). Hai điểm cốt lõi:

- **`GATEWAY_JWT_PUBLIC_KEY`** (SPKI PEM) — Worker CHỈ giữ khoá **công khai** của Gateway
  (Trạm 2, Zero-Trust bất đối xứng). **Fail-closed:** thiếu key → mọi endpoint có bảo vệ
  trả `503 "Worker not provisioned with Gateway public key"`. Sinh cặp khoá bằng
  `scripts/gen-gateway-keys.sh`.
- **`HF_HOME` + `TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1`** — ép chạy offline; boot sẽ
  **fail-closed** nếu thiếu trọng số. Bake lúc build (`--build-arg BAKE_WEIGHTS=1`) hoặc mount
  Network Volume chứa `HF_HOME` (xem DEPLOYMENT.md §2.3).

## Health probe

`GET /health` (không cần JWT — cho Docker HEALTHCHECK / orchestrator probe trước khi có khoá):

```json
{ "status": "ok", "models_loaded": true, "device": "cuda" }
```

Zero-Logging: chỉ trả cờ boolean + enum thiết bị (`cuda`/`cpu`), không lộ đường dẫn/token/nội
dung. `models_loaded:false` = còn nạp trọng số lên VRAM (cold start) → chưa nên nhận job.

## Test

```bash
python -m pytest -q          # bộ CPU mặc định (an toàn máy dev; nhóm gpu_acceptance bị loại)
```

> **`gpu_acceptance`** là bài kiểm THẬT (cư trú VRAM, dịch Qwen, `/process` e2e) — chỉ chạy
> trên hộp GPU với `RUN_GPU_ACCEPT=1 python -m pytest -m gpu_acceptance` (conftest gác thêm
> bằng `torch.cuda.is_available()` và cc ≥ 7). Thuộc nhóm **residual_hardware**: máy dev
> GTX 1070 8GB không chứng minh được, **không tô xanh giả**. Xem DEPLOYMENT.md §4.

## Docker

Xem `Dockerfile` (image `uvicorn src.main:app`, offline HF_HOME, HEALTHCHECK) và quy trình
build/run trong [docs/DEPLOYMENT_AND_TESTING.md](../../docs/DEPLOYMENT_AND_TESTING.md) §1.2.
