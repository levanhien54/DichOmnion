# Triển khai OmniVoice — Runbook hai nhánh (Track A / Track B)

Tài liệu này hướng dẫn đưa worker lên **một GPU 24GB để test THẬT** (Track A) trong khi vẫn
**sửa trực tiếp trên máy dev hiện tại** (Track B). Nguyên tắc xuyên suốt: **không giả xanh** —
những gì chỉ GPU thật chứng minh được thì phải chạy trên GPU thật, không mock cho "qua bài".

> Máy dev (GPU **dưới chuẩn**: GTX 1070 8GB, Pascal cc6.1 — không fp16, không đủ 24GB) chạy
> suite **mock** (không đụng GPU) giữ xanh: `pnpm run test:py` + `pnpm run test:ts`.
> Hộp GPU **đủ chuẩn** (24GB, Volta+ cc>=7) chạy bài THẬT: `pnpm run test:py:gpu` (`gpu_acceptance`).
> Cổng acceptance yêu cầu `RUN_GPU_ACCEPT=1` **và** cuda **và** cc>=7 → hộp yếu tự **skip sạch**.

---

## 0. Kiến trúc rút gọn (đường đi một job)

```
Client (Tauri, tách audio cục bộ, KÝ ECDSA)
   │  chỉ gửi AUDIO + chữ ký
   ▼
Cloudflare Worker  = GATEWAY  (Trạm 2/3: Turnstile, JWT ES256, anomaly, kill switch, KV)
   │  ký JWT ES256 RIÊNG cho jobId  →  gọi qua Cloudflare Tunnel
   ▼
FastAPI GPU Worker (riêng tư, bind loopback; Whisper + Qwen 4B THƯỜNG TRÚ VRAM)
```

- **Gateway** giữ **private key** (ký) → Cloudflare secret `GATEWAY_JWT_PRIVATE_KEY`.
- **Worker** giữ **public key** (xác minh) → env `GATEWAY_JWT_PUBLIC_KEY`.
- Worker **không mở IP public**; Cloudflare Tunnel (`cloudflared`) là đường vào riêng duy nhất.

---

## 1. Sinh khoá Trạm 2 (một lần)

```bash
scripts/gen-gateway-keys.sh
```

In ra **PRIVATE** (nạp vào Gateway) và **PUBLIC** (nạp vào Worker). Khoá ghi ra thư mục tạm
ngoài repo — **không commit**. Xoá sau khi đã nạp cả hai nơi.

---

## 2. Track A — Deploy lên GPU 24GB + acceptance thật

### 2.1 Build ảnh worker

```bash
cd apps/gpu-worker
# Mặc định KHÔNG bake trọng số (ảnh gọn, mount volume ở bước 2.3):
docker build -t omnivoice-worker .
# HOẶC ảnh tự chứa (bake luôn trọng số — hợp serverless):
# docker build --build-arg BAKE_WEIGHTS=1 --build-arg HF_TOKEN=hf_xxx -t omnivoice-worker .
```

> **RESIDUAL_HARDWARE.** `Dockerfile` viết theo chuẩn nhưng chưa build được ở máy dev (không
> Docker; GPU dev dưới chuẩn). Cặp `torch==2.13.0` + `torchaudio==2.11.0` + `transformers`
> **đã kiểm import** trên cu126 ở máy dev; nhưng build image, **fp16 + fit-VRAM** Qwen+Whisper
> chỉ kiểm chứng trên hộp GPU đủ chuẩn. Lưu ý torchaudio **không cùng minor** với torch (dòng
> torchaudio đã tách) — pin đúng cặp đã kiểm, đừng ép khớp minor. Nếu index không có cặp này,
> hạ về cặp torch/torchaudio **cùng kênh phát hành**. CUDA-minor base image phải khớp driver host.

### 2.2 Cấp GPU (khuyến nghị: RunPod pod bền vững)

- **RunPod persistent pod**, GPU ≥ 24GB (RTX 4090 / A5000 / L4…), gắn **persistent volume**
  mount vào `HF_HOME` (`/models/hf`). Pod bền vững hợp với vòng lặp Track B (rsync + restart).
- Thay thế: **Modal** (serverless) — hợp khi bật `BAKE_WEIGHTS=1` (ảnh tự chứa), nhưng vòng
  lặp sửa-trực-tiếp kém tiện hơn pod bền vững.

### 2.3 Nạp trọng số vào volume (nếu không bake)

Trên hộp GPU, tải MỘT LẦN vào `HF_HOME`:

```bash
export HF_HOME=/models/hf
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

> Runtime ép `TRANSFORMERS_OFFLINE=1` — trọng số **phải** có sẵn trước khi worker boot, nếu
> không `load_all_models()` fail-closed (đúng ý đồ: không chạy giả).

### 2.4 Chạy worker

```bash
docker run --gpus all \
  -v /models/hf:/models/hf \
  -e GATEWAY_JWT_PUBLIC_KEY="$(cat gateway_public_spki.pem)" \
  -p 127.0.0.1:8000:8000 \
  omnivoice-worker
```

(Các env khác xem `apps/gpu-worker/.env.example`.) Bind `127.0.0.1` — chỉ tunnel tới được.

### 2.5 Mở Cloudflare Tunnel

```bash
# Nhanh (URL trycloudflare tạm):
cloudflared tunnel --url http://127.0.0.1:8000
# HOẶC named tunnel để có hostname ổn định (khuyến nghị cho production).
```

Ghi lại URL tunnel → dùng làm `WORKER_URL` của Gateway (bước 2.6).

### 2.6 Provision + deploy Gateway (Cloudflare Worker)

```bash
cd apps/gateway
wrangler kv namespace create KV_CACHE          # dán id trả về vào wrangler.toml
wrangler secret put GATEWAY_JWT_PRIVATE_KEY     # dán PRIVATE PEM từ bước 1
wrangler secret put TURNSTILE_SECRET
wrangler secret put ADMIN_TOKEN
# Đặt WORKER_URL = URL tunnel ở bước 2.5 (sửa [vars] trong wrangler.toml hoặc Dashboard).
pnpm run deploy                                 # wrangler deploy --minify src/index.ts
```

### 2.7 Probe (bắt buộc trước khi tin là "đã deploy")

```bash
# 1) Worker sống + đã nạp model lên VRAM:
curl -fsS <WORKER_URL>/health          # kỳ vọng: {"status":"ok","models_loaded":true,"device":"cuda"}
# 2) Round-trip JWT + pipeline THẬT trên hộp GPU:
ssh <GPU_SSH> 'cd /opt/DichOmnion && pnpm run test:py:gpu'
```

> **Không tuyên bố "đã deploy" nếu `/health` chưa `models_loaded:true device:cuda` và
> `test:py:gpu` chưa xanh.** Đây là ranh giới chống giả xanh.

---

## 3. Track B — Vòng lặp sửa trực tiếp trên máy dev (CPU)

```bash
# 1) Sửa mã trên máy dev, giữ suite mock xanh:
pnpm run test:py        # worker CPU (mock seam) — acceptance tự động deselect
pnpm run test:ts        # gateway + crypto + client tsc

# 2) Đẩy mã lên hộp GPU và restart worker:
GPU_SSH=user@host GPU_DIR=/opt/DichOmnion scripts/sync-gpu.sh

# 3) Chạy bài THẬT trên hộp GPU:
ssh user@host 'cd /opt/DichOmnion && pnpm run test:py:gpu'
```

`sync-gpu.sh` chỉ đẩy **mã nguồn** (loại git/venv/node_modules/media/.env) — bí mật và trọng
số nằm sẵn trên hộp GPU, không đồng bộ từ máy dev.

---

## 4. Checklist RESIDUAL_HARDWARE (chỉ hộp GPU chứng minh — KHÔNG giả xanh)

| Mục | Kiểm chứng trên hộp GPU |
|-----|--------------------------|
| torch/torchaudio cài được (cu12x) | `python -c "import torch,torchaudio; print(torch.version.cuda)"` *(import đã de-risk trên cu126 dev; wheel cho CUDA đích vẫn cần kiểm)* |
| GPU đủ chuẩn fp16 (Volta+ cc>=7) | `python -c "import torch;print(torch.cuda.get_device_capability(0))"` → `[0]>=7` (dev GTX 1070 = (6,1) → skip) |
| Qwen 4B + Whisper fit 24GB VRAM   | `nvidia-smi` lúc `/health` báo `models_loaded:true` |
| Cư trú VRAM (không rơi CPU)       | `test:py:gpu::test_models_resident_on_cuda` |
| Qwen sinh bản dịch thật           | `test:py:gpu::test_qwen_translate_smoke` |
| /process e2e ra audio thật        | `test:py:gpu::test_process_endpoint_e2e` |
| Độ trễ cold-start < `--start-period` (300s) | thời gian `/health` chuyển 200 |
| Demucs `-d cuda --segment 7` không OOM | `nvidia-smi` lúc tách nền |
| Tunnel + KV + secret + `wrangler deploy` | thao tác live, phản hồi thật của Cloudflare |

---

## 5. Tham chiếu biến môi trường

- **Worker:** `apps/gpu-worker/.env.example` (đã liệt kê đầy đủ, kèm giải thích từng biến).
- **Gateway:** `apps/gateway/wrangler.toml` (`[vars]` + danh sách secret cần `wrangler secret put`).
- **Client (Tauri):** `apps/client/.env.example` — 4 biến `VITE_*` **nhúng lúc build**
  (đổi giá trị phải build lại): `VITE_GATEWAY_URL` (URL Worker Gateway đã deploy; bỏ trống →
  fallback `http://localhost:8787`), `VITE_TURNSTILE_SITE_KEY` (site key công khai, đi cặp
  với secret `TURNSTILE_SECRET` của Gateway), và `VITE_AUDIO_UPLOAD_URL` +
  `VITE_AUDIO_PUBLIC_URL` (presigned PUT + public GET của bucket R2/S3 — **R2 nằm hoàn
  toàn phía Client, Gateway không bind R2**; thiếu 2 biến này → Client fail-closed, không
  job nào chạy).

### 5.1. Deploy Client (chân thứ 3, ngoài Worker + Gateway)
1. `cp apps/client/.env.example apps/client/.env` rồi điền 4 biến ở trên (ít nhất
   `VITE_GATEWAY_URL` + cặp AUDIO_UPLOAD/PUBLIC cho production).
2. Build: `cd apps/client && pnpm tauri build` (hoặc `pnpm tauri dev` để test — vẫn cần `.env`).
3. Bucket R2/S3 phải bật CORS cho origin Tauri (`tauri://localhost`,
   `https://tauri.localhost`) thì PUT presigned mới qua được.

> **Tùy chọn — Financial Kill Switch:** Gateway expose endpoint quản trị
> `POST /api/admin/kill-switch`, chỉ nhận request kèm header `X-Admin-Token` khớp secret
> `ADMIN_TOKEN` (§2.6); khi bật, mọi call job/register trả `503` (fail-closed). Monitor
> billing standalone **đã có sẵn** tại `apps/gateway/scripts/kill-switch-monitor.mjs`
> (watchdog thật: đọc chi phí → `POST` bật cờ khi vượt ngưỡng; No-Fake-Success: **TỪ CHỐI
> khởi động** nếu chưa cấu hình nguồn billing). Chạy NGOÀI Worker (cron/host riêng), env:
> `GATEWAY_URL`, `ADMIN_TOKEN` (khớp secret Worker), `BILLING_API_URL`, `SPEND_THRESHOLD_USD`
> (bắt buộc) + `BILLING_API_TOKEN`/`BILLING_JSON_FIELD`/`POLL_INTERVAL_S`/`ONESHOT` (tùy chọn).
> Chạy một lần cho cron: `ONESHOT=1 node apps/gateway/scripts/kill-switch-monitor.mjs`.
> **Phần dư thật (residual, xem RH7):** nối `BILLING_API_URL` của nhà cung cấp GPU + host
> chạy cron — KHÔNG phải "thiếu script". Không bắt buộc cho luồng E2E cơ bản.
