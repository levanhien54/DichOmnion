# syntax=docker/dockerfile:1
#
# OmniVoice GPU Worker — ảnh chạy THẬT trên GPU 24GB.
#
# RESIDUAL_HARDWARE: file này được viết theo chuẩn nhưng CHƯA build/chạy được ở máy dev
# (không Docker; GPU dev GTX 1070 8GB cc6.1 dưới chuẩn — không fp16, không đủ 24GB). Việc
# build + boot + fp16 + fit-VRAM chỉ kiểm chứng trên hộp GPU đủ chuẩn (Volta+ cc>=7) — KHÔNG
# giả xanh ở đây. Xem docs/DEPLOYMENT.md để build & probe.
#
# Base mặc định là CUDA 12.4 runtime. CUDA-minor của base PHẢI tương thích driver host và
# bản CUDA mà wheel torch mang theo (xem requirements.txt). Ghi đè khi cần:
#   docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 ...
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/models/hf

# --- Phụ thuộc hệ thống ---
# ffmpeg: BẮT BUỘC — audio_engine (atempo căn lip-sync, tách audio khỏi video) + faster-whisper
#         gọi ffmpeg qua system binary. Thiếu ffmpeg là pipeline chết ở runtime, không manifest
#         Python nào bù được. git: demucs/huggingface_hub cần khi tải. curl: HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip ffmpeg git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Phụ thuộc Python ---
# Cài từ requirements.txt (đã khớp import THẬT: transformers/accelerate thay openai; torch==
# torchaudio cùng version). Copy MỘT MÌNH requirements trước để tận dụng layer cache khi chỉ
# đổi src. Wheel torch mặc định trên Linux/PyPI ĐÃ kèm CUDA; nếu cần build cuXXX cụ thể, cài
# tay torch/torchaudio từ https://download.pytorch.org/whl/cuXXX TRƯỚC dòng này rồi để pip
# thấy đã thoả (requirements.txt cùng version).
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# --- Mã nguồn ứng dụng ---
COPY src ./src

# --- (TÙY CHỌN) Bake trọng số vào ảnh để chạy offline hoàn toàn ---
# Mặc định TẮT: ảnh gọn, mount volume HF_HOME đã tải sẵn (hợp pod RunPod bền vững — tải một
# lần, tái dùng qua nhiều lần restart). Bật để có ảnh tự chứa (hợp serverless/Modal):
#   docker build --build-arg BAKE_WEIGHTS=1 --build-arg HF_TOKEN=hf_xxx -t omnivoice-worker .
# Bake chạy ONLINE (chưa ép offline); ENV runtime bên dưới mới bật offline. Whisper 'base'
# tải trên CPU lúc build (không cần GPU) — chỉ để kéo file về HF_HOME.
ARG BAKE_WEIGHTS=0
ARG QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
ARG HF_TOKEN=
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
      python -m pip install --upgrade "huggingface_hub[cli]" && \
      huggingface-cli download "$QWEN_MODEL_ID" ${HF_TOKEN:+--token "$HF_TOKEN"} && \
      python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" ; \
    else \
      echo "BAKE_WEIGHTS=0 -> trọng số PHẢI được mount vào $HF_HOME lúc chạy (xem DEPLOYMENT.md)"; \
    fi

# --- ENV runtime ---
# 0.0.0.0: để cloudflared/orchestrator trong mạng riêng tới được (worker vẫn NẰM SAU tunnel/
#   mạng riêng — KHÔNG map cổng ra IP public). Offline: cấm nạp lười từ HF Hub lúc chạy (phi
#   xác định + rò rỉ metadata). Trọng số PHẢI có sẵn (baked hoặc mount) — thiếu thì
#   load_all_models fail-closed, worker không boot (ĐÚNG ý đồ, không chạy giả).
ENV WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8000 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    DEMUCS_DEVICE=cuda \
    DEMUCS_SEGMENT=7 \
    WORKER_MAX_CONCURRENT_JOBS=1

EXPOSE 8000

# HEALTHCHECK: /health KHÔNG cần JWT. FastAPI lifespan nạp Whisper+Qwen lên VRAM TRƯỚC khi
# phục vụ, nên /health chỉ 200 sau khi nạp xong (cold start có thể vài phút) — dùng
# --start-period rộng để không đánh dấu unhealthy oan trong lúc nạp. Chỉnh nếu model to hơn.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# --workers 1: MỘT tiến trình = MỘT bản model thường trú VRAM. Nhiều uvicorn worker sẽ nhân
# bản model -> nhân đôi/ba VRAM -> OOM trên 24GB. Đồng bộ với WORKER_MAX_CONCURRENT_JOBS=1.
# --no-access-log: Zero-Logging. Access log ghi nguyên request line, mà /api/worker/download
# nhận đường dẫn temp NỘI BỘ qua query (?path=/tmp/...) -> bật sẽ rò rỉ path nội bộ. Đường
# chạy production (CMD này) KHÔNG đi qua __main__ nên phải tắt access log ngay tại đây.
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
