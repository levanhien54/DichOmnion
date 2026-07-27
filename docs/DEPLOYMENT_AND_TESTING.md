# HƯỚNG DẪN DEPLOY, XỬ LÝ LOG VÀ KIỂM THỬ THỰC TẾ (PRODUCTION E2E TEST)

Tài liệu này chuẩn hóa quy trình đưa **OmniVoice V2 (DichOmnion)** lên môi trường Server
thực (Cloudflare Workers + một GPU Pod 24GB), thiết lập đường thu Log, và chạy kiểm thử
thực tế (E2E) trên GPU thật.

> **Runbook chuẩn (canonical) là [DEPLOYMENT.md](DEPLOYMENT.md).** Tài liệu này là bản
> hướng dẫn thao tác + kịch bản test đi kèm; khi có mâu thuẫn, DEPLOYMENT.md là nguồn đúng.
>
> **Ranh giới No-Fake-Success:** mọi thứ trong nhóm `residual_hardware` (build/boot image,
> fp16 vừa VRAM, cold-start, `gpu_acceptance`/`test:py:gpu` thật, độ trễ ~40s, tạo KV
> namespace/secret/tunnel thật, presign R2 mỗi job) **chỉ chứng minh được trên phần cứng
> ≥24GB + tài khoản Cloudflare thật** — KHÔNG mô phỏng, KHÔNG tự tô xanh. Xem
> DEPLOYMENT.md §4 (bảng RESIDUAL_HARDWARE). Máy dev GTX 1070 8GB không chứng minh được
> nhóm này.

---

## 1. QUY TRÌNH ĐƯA LÊN SERVER THỰC TẾ (DEPLOYMENT)

Hệ thống gồm **3 phần deploy độc lập**: Gateway (Cloudflare Worker), GPU Worker (một Pod
FastAPI thường trú sau cloudflared tunnel), và Client (Tauri build lại trỏ về Gateway).
Worker **không** phải Serverless: nó là HTTP server `uvicorn src.main:app` (xem
`apps/gpu-worker/Dockerfile`), chạy thường trú và chỉ nhận traffic qua tunnel.

### 1.1. Khối Gateway (Cloudflare Worker)
Cửa ngõ: xác minh chữ ký ECDSA của Client (Trạm 1), cấp JWT ES256 mỗi job (Trạm 2),
chống bot Turnstile + anomaly/timing + Financial Kill Switch (Trạm 3). Chỉ bind **KV**
(không có R2 ở Gateway — R2 nằm hoàn toàn phía Client, xem §1.3).

1. Cài đặt tại máy Local:
   ```bash
   cd apps/gateway
   pnpm install
   ```
2. Tạo KV namespace thật và dán `id` trả về vào `wrangler.toml` (`[[kv_namespaces]]
   binding = "KV_CACHE"` — hiện đang để placeholder `local-kv-id-for-dev`):
   ```bash
   npx wrangler kv namespace create KV_CACHE
   ```
3. Nạp **3 secret** (KHÔNG có "R2 API" — Gateway không dùng R2):
   ```bash
   npx wrangler secret put GATEWAY_JWT_PRIVATE_KEY   # khóa ký JWT ES256 worker↔gateway (PKCS8/JWK)
   npx wrangler secret put TURNSTILE_SECRET          # cặp với VITE_TURNSTILE_SITE_KEY của Client
   npx wrangler secret put ADMIN_TOKEN               # bảo vệ endpoint quản trị (kill-switch)
   ```
4. Đặt biến `WORKER_URL` = URL tunnel của GPU Worker (lấy ở §1.2) trong `[vars]` của
   `wrangler.toml` (mặc định dev là `http://127.0.0.1:8000`).
5. Đẩy code lên Cloudflare Workers (Production):
   ```bash
   npx wrangler deploy
   ```
6. *Kết quả kỳ vọng:* trả về URL Production (VD `https://gateway.<account>.workers.dev`).
   URL này chính là **VITE_GATEWAY_URL** cần nạp cho Client ở §1.3.

### 1.2. Khối GPU Worker (Pod FastAPI thường trú + cloudflared tunnel)
Nơi nhận URL audio, chạy mô hình AI (Whisper, Qwen3-4B, Demucs, TTS) trên VRAM GPU thật.
Worker **không bao giờ** lộ IP công khai — cloudflared tunnel là ingress DUY NHẤT.

1. Build Docker Image (image name tự đặt theo registry của bạn):
   ```bash
   cd apps/gpu-worker
   docker build -t <registry>/omnivoice-worker:<tag> .
   docker push <registry>/omnivoice-worker:<tag>
   ```
   Mặc định image **không bake weights** (`BAKE_WEIGHTS=0`) và ép offline
   (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) → boot sẽ **fail-closed** nếu thiếu
   weights. Chọn 1 trong 2 cách nạp weights:
   - Bake lúc build: `docker build --build-arg BAKE_WEIGHTS=1 -t <registry>/omnivoice-worker:<tag> .`
   - Hoặc mount Network Volume chứa `HF_HOME` (tải trước bằng `huggingface-cli` — xem
     DEPLOYMENT.md §2.3), rồi trỏ `HF_HOME` vào volume đó.
2. Chạy Pod trên GPU RTX 4090 / A10G / L4 (≥24GB VRAM), bind về loopback + dựng tunnel:
   ```bash
   docker run --gpus all -p 127.0.0.1:8000:8000 \
     -v /data/hf:/models/hf \
     -e GATEWAY_JWT_PUBLIC_KEY="$(cat gateway_public_spki.pem)" \
     <registry>/omnivoice-worker:<tag>
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. *Kết quả kỳ vọng:* cloudflared in ra URL tunnel công khai. Nạp URL này vào biến
   `WORKER_URL` của Gateway (§1.1 bước 4) rồi `wrangler deploy` lại.

### 1.3. Khối Client (Tauri — build lại trỏ về Production)
Client bóc tách audio cục bộ (ffmpeg sidecar), ký ECDSA, xin Gateway ký presigned URL cho
MỖI job (`POST /api/uploads/presign`, Option A — Đợt 30), PUT audio thẳng lên R2 rồi gửi
presigned GET cho Gateway. **Biến `VITE_*` được nhúng lúc build** — đổi giá trị PHẢI build
lại (xem `apps/client/.env.example`).

1. Sao chép `apps/client/.env.example` → `apps/client/.env` và điền (**2 biến**, không còn
   biến R2 nào ở Client — R2 nằm ở Gateway):
   - `VITE_GATEWAY_URL` = URL Gateway Production (§1.1). Bỏ trống → fallback
     `http://localhost:8787` (chỉ hợp cho dev).
   - `VITE_TURNSTILE_SITE_KEY` = site key Turnstile (cặp với `TURNSTILE_SECRET` Gateway).
   - *Cấu hình R2* (`R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`…) nằm ở **Gateway** (§1.1 +
     `apps/gateway/.dev.vars.example`); thiếu → `POST /api/uploads/presign` FAIL-CLOSED 503,
     không job nào upload audio được (No-Fake-Success: không bịa URL giả).
2. Build:
   ```bash
   cd apps/client
   pnpm tauri build     # hoặc `pnpm tauri dev` để test nhanh (vẫn phải có .env)
   ```

---

## 2. HỆ THỐNG NHẬN LOG VÀ GIÁM SÁT (LOGGING & MONITORING)

Tuân thủ **Zero-Logging**: KHÔNG ghi thông tin nhạy cảm của người dùng (kịch bản dịch, URL
âm thanh, IP gốc, token) ra plaintext. Log chỉ chứa sự kiện/định danh chung, **không** job
id hay nội dung.

### 2.1. Cloudflare Gateway Logs (Trạm 1 & 3)
Bám đuôi log real-time để giám sát luồng chặn request gian lận / kill-switch:
```bash
npx wrangler tail --format=pretty
```
*Log mang tính sự kiện, không lộ job id / nội dung, ví dụ:*
> `[INFO] ECDSA signature verified (Station-1).`
> `[WARN] Worker render timeout (Station-3): job terminated + device quarantined.`
> `[ALERT] Financial Kill Switch engaged (fail-closed): rejecting new jobs.`

> **Lưu ý hành vi thật:** khi Worker vượt ngưỡng render timeout, Gateway **kết thúc job
> (terminate) và cách ly thiết bị (quarantine, trạng thái `TERMINATED_TIMEOUT`)** —
> đây là hành vi fail-closed chống gian lận, **KHÔNG** có "Re-queue Idempotent". Cơ chế
> re-queue nhiều-worker không tồn tại (chỉ một `WORKER_URL`) — đó là hạng mục
> residual_hardware, không phải tính năng đang chạy.

### 2.2. GPU Worker Logs (Pod)
Giám sát cấp phát VRAM và tải của Pod qua log container:
```bash
docker logs -f <container>
```
*Mục tiêu giám sát (đo trên GPU box — residual, không mô phỏng được ở dev):*
- Thời gian `load_all_models()` lúc cold start (mục tiêu < 300s — xem DEPLOYMENT.md §4).
- Có cảnh báo CUDA OOM hay không (fp16 Qwen3-4B + Whisper phải vừa 24GB).
- Độ trễ inference/video (**mục tiêu ~40s/video là chỉ tiêu residual cần ĐO trên GPU
  thật**, chưa được xác nhận; xử lý tuần tự qua semaphore size-1, **không** dùng CUDA
  Streams).

---

## 3. QUY TRÌNH KIỂM THỬ THỰC TẾ (REAL-WORLD E2E TEST)

Bài test E2E đi trọn luồng **2 Bước (Human-in-the-Loop)** từ lúc đưa Video vào đến lúc
nhận Video lồng tiếng hoàn chỉnh, có GPU thật.

**Chuẩn bị Môi trường Test:**
- Client đã build/chạy với `VITE_GATEWAY_URL` trỏ về Gateway Production (§1.3).
- Gateway đã `deploy` với `WORKER_URL` = tunnel của Pod đang chạy (§1.1–§1.2).
- Trước khi test E2E, xác nhận Worker sống: `GET <tunnel>/health` trả
  `models_loaded:true device:cuda`, và `pnpm run test:py:gpu` xanh trên GPU box
  (đây là điều kiện tiên quyết để được phép nói "đã deploy").

### Kịch Bản Test (Stress Test & Happy Path)
1. **Bóc tách Audio cục bộ (Client-Centric):**
   - Kéo-thả 1 Video MP4 ~10 phút vào giao diện Tauri.
   - *Pass:* Client chạy ffmpeg sidecar bóc `.wav`, dung lượng upload giảm mạnh; **chỉ
     audio** rời máy (Video MP4 gốc vẫn nằm nguyên ở ổ cứng Local).
2. **Bảo mật ECDSA & Cross-Validation:**
   - Client tự sinh key và ký request.
   - Dùng Postman ném request rác/giả mạo.
   - *Pass (phân biệt 401 vs 403):*
     - Request **thiếu chữ ký / device chưa đăng ký** → `401` (Missing/Unauthorized
       Device).
     - Request **bị chỉnh sửa hoặc replay** (sao chép rồi đổi payload/timestamp cũ) →
       `403` (Tampering Detected / Request Expired).
     - Turnstile fail → `403`. Client hợp lệ → `202 QUEUED`.
3. **Khớp Khẩu Hình (Lip-Sync `atempo`):**
   - Đợi job xong, nhấn "Xem trước đoạn Audio".
   - *Pass:* Giọng lồng bị nén/giãn đúng `(end-start)`, không tràn khỏi sóng, không méo
     cao độ (không "chipmunk").
4. **Pha Trộn Âm (Auto-Ducking & Mix):**
   - *Pass:* Nghe rõ tiếng lồng thoại; nhạc nền (instrumental) tự **hụp xuống `-5dB`**
     (giá trị production ở `model_manager.py`) ngay trước khi nhân vật cất giọng và khôi
     phục lúc dứt câu.
5. **Hậu Kỳ Xuất File (Local Muxing):**
   - Bấm **Export Lồng Tiếng**.
   - *Pass:* Tauri chạy `-c:v copy`, trả MP4 hoàn thiện; hình ảnh giữ nguyên bản 100%
     (không re-encode, không vỡ hạt).

---

*Tài liệu này thuộc kho quy trình DevOps của dự án. Runbook chuẩn: [DEPLOYMENT.md](DEPLOYMENT.md).
Các tham số residual (~40s, cold-start, VRAM) sẽ được điền số thật sau khi đo trên GPU box.*
