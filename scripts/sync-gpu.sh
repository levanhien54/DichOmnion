#!/usr/bin/env bash
#
# TRACK B — vòng lặp sửa-trực-tiếp: đẩy MÃ đã sửa từ máy dev (CPU) lên hộp GPU để test THẬT,
# rồi restart worker. Máy dev vẫn giữ suite CPU (mock) xanh; hộp GPU chạy bài THẬT.
#
# Dùng:
#   GPU_SSH=user@host [GPU_DIR=/opt/DichOmnion] [GPU_RESTART=1] scripts/sync-gpu.sh
#
# Sau khi sync, chạy acceptance THẬT trên hộp GPU:
#   ssh $GPU_SSH 'cd /opt/DichOmnion && RUN_GPU_ACCEPT=1 pnpm run test:py:gpu'
set -euo pipefail

: "${GPU_SSH:?Đặt GPU_SSH=user@host (đích SSH của hộp GPU)}"
GPU_DIR="${GPU_DIR:-/opt/DichOmnion}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v rsync >/dev/null 2>&1 || { echo "Cần rsync trong PATH." >&2; exit 1; }

# Chỉ đẩy MÃ NGUỒN. Loại: git, node_modules, venv, build artifacts, media tạm, và .env
# (bí mật hộp GPU đặt tại chỗ, KHÔNG đồng bộ từ máy dev). --delete để hộp GPU khớp local.
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'node_modules/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.pytest_cache/' \
  --exclude '**/.venv/' \
  --exclude '**/venv/' \
  --exclude '**/target/' \
  --exclude '**/dist/' \
  --exclude '*.wav' --exclude '*.mp3' --exclude '*.mp4' \
  --exclude '.env' \
  "$REPO_ROOT/" "$GPU_SSH:$GPU_DIR/"

echo "Đã đồng bộ mã -> $GPU_SSH:$GPU_DIR"

# Restart worker trên hộp GPU. Lệnh phụ thuộc cách bạn chạy worker (docker compose / systemd /
# uvicorn trực tiếp). Mặc định thử docker compose; nếu khác, sửa nhánh dưới cho phù hợp.
if [ "${GPU_RESTART:-1}" = "1" ]; then
  ssh "$GPU_SSH" "cd '$GPU_DIR/apps/gpu-worker' && \
    (docker compose restart worker 2>/dev/null \
     || sudo systemctl restart omnivoice-worker 2>/dev/null \
     || echo 'Chưa cấu hình cách restart worker — sửa scripts/sync-gpu.sh cho môi trường của bạn.')"
fi
