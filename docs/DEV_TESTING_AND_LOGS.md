# KIỂM THỬ & THU LOG TRÊN MÁY DEV — VÒNG LẶP SỬA LỖI

Tài liệu này chuẩn hoá **những gì kiểm chứng được NGAY trên máy dev** (Windows 10,
GTX 1070 8GB — dưới chuẩn: không fp16, < 24GB VRAM), **cách thu log** để chẩn đoán, và
**vòng lặp sửa lỗi** (reproduce → log → fix → chạy lại cổng xanh). Mục tiêu: vắt cạn mọi thứ
sửa được ở local TRƯỚC khi tốn tài nguyên GPU box thật.

> **Quan hệ tài liệu:**
> - [DEPLOYMENT.md](DEPLOYMENT.md) — runbook triển khai **chuẩn (canonical)**.
> - [DEPLOYMENT_AND_TESTING.md](DEPLOYMENT_AND_TESTING.md) — deploy + **E2E test trên GPU
>   server thật** (Track A).
> - **Tài liệu này** — **vòng test + thu log ở máy DEV (Track B)**, bổ trợ, KHÔNG thay thế.
>
> **Ranh giới No-Fake-Success:** máy dev **KHÔNG** chứng minh được nhóm `residual_hardware`
> (build/boot image, fp16 vừa 24GB, cold-start, `gpu_acceptance` thật, độ trễ inference, KV/
> secret/tunnel Cloudflare thật, presign R2). Không tô xanh giả những gì chỉ GPU box + tài
> khoản Cloudflare thật kiểm được. Bảng residual đầy đủ: DEPLOYMENT.md §4.

---

## 1. BẢN ĐỒ: GÌ CHỨNG MINH ĐƯỢC Ở DEV, GÌ KHÔNG

| Khối / tiêu chí | Kiểm được ở DEV (Track B) | Chỉ ở GPU box thật (residual, Track A) |
|---|---|---|
| Gateway (Trạm 1/2/3) | ✅ Toàn bộ logic: verify ECDSA, ký JWT ES256, anomaly/timing, kill-switch, fail-closed, throttle, bound-input (Vitest 120 test + wrangler dry-run) | Tạo KV namespace/secret/tunnel THẬT; `wrangler deploy` production |
| Client (Tauri/Rust) | ✅ Validate đường dẫn, whitelist, cleanup temp, base64, mux (cargo 18 test); tsc client | Build cài đặt Tauri hoàn chỉnh + E2E kéo-thả video thật |
| Worker — hợp đồng vào/ra | ✅ Auth JWT (403/503), bound-input 422, Zero-Logging response, mix/ducking/atempo bằng pydub+ffmpeg (Pytest 228 test) | Whisper/Qwen **thường trú VRAM**, dịch thật, `/process` e2e, fp16 vừa 24GB |
| Crypto (ký/verify, deterministicStringify) | ✅ Đầy đủ (chạy trong test:ts) | — |
| Độ trễ ~40s/video, cold-start < 300s | ❌ | ✅ ĐO trên GPU box (chưa có số thật) |

**Nguyên tắc:** mọi *logic* + *hợp đồng* + *fail-closed* kiểm được ở dev. Mọi *chất lượng do
phần cứng chứng minh* (mô hình cư trú VRAM, tốc độ, fp16) là residual.

---

## 2. BỘ CỔNG XANH TRÊN MÁY DEV (lệnh + kỳ vọng)

Chạy từ thư mục gốc `D:\DichOmnion`. Số liệu kỳ vọng là baseline hiện tại (Đợt 29) — lệch
xuống = hồi quy cần điều tra.

| # | Lệnh | Kỳ vọng (baseline) |
|---|---|---|
| 1 | `pnpm run test:ts` | Gateway **120 passed**, Client **7 passed** (+ `tsc --noEmit` sạch), packages build-check |
| 2 | `pnpm run test:py` | **228 passed, 1 skipped, 4 deselected** (4 deselected = `gpu_acceptance`, cố ý loại) |
| 3 | `cd apps/gateway && npx wrangler deploy --dry-run` | Bundle hợp lệ (~128 KiB), in đúng binding KV_CACHE + vars; **không** deploy |
| 4 | `pnpm run build` | `packages/shared-types` + `packages/crypto-utils` → `Done` |
| 5 | `cd apps/client/src-tauri && cargo test` | **18 passed; 0 failed** |

Gộp nhanh hai cổng chính: `pnpm run test` (= `test:ts` + `test:py`).

> **Windows/PowerShell:** `pnpm run test:py:gpu` dùng cú pháp env inline kiểu POSIX
> (`RUN_GPU_ACCEPT=1 …`) nên **KHÔNG chạy trực tiếp** trên PowerShell/CMD — đó là bài GPU box
> (POSIX). Ở dev chỉ chạy tới cổng #1–#5.

---

## 3. THU LOG TRÊN MÁY DEV (theo từng khối)

### 3.1. Gateway (wrangler dev)
```bash
cd apps/gateway
cp .dev.vars.example .dev.vars     # điền secret dev (xem file mẫu); thiếu key -> đường auth fail-closed
pnpm run dev                       # wrangler dev src/index.ts -> http://localhost:8787
```
Log Gateway **chỉ ghi sự-kiện lỗi** (toàn bộ là `console.error`; Zero-Logging: không job id /
không nội dung — đường thành công KHÔNG log gì), in ra console wrangler. Các dòng thật hiện có:
`[station2] cannot dispatch job: <reason>` (vd `<reason>` = `gateway_key_missing`),
`[station3] refused dispatch to quarantined worker`,
`[station3] rejected worker result: impossibly fast`,
`[station3] worker timed out; termination signalled`,
`[dispatch] background job failed`. Với Gateway **đã deploy** (GPU box), bám log real-time
bằng: `npx wrangler tail --format=pretty`.

### 3.2. GPU Worker (uvicorn)
```bash
cd apps/gpu-worker
$env:WORKER_DEBUG = "1"             # (PowerShell) in traceback ĐẦY ĐỦ ra STDERR để chẩn đoán
uv run uvicorn src.main:app --port 8000
```
- **`WORKER_DEBUG=1`**: khi job ném lỗi, worker `traceback.print_exc()` ra **STDERR của tiến
  trình** — KHÔNG bao giờ đi vào HTTP response / KV / client (Zero-Logging còn nguyên: client
  chỉ nhận **tên loại lỗi**). Mặc định TẮT. Bật khi cần bắt lỗi, tắt lại khi xong.
- **`GET http://localhost:8000/health`** → `{status, models_loaded, device}`. Trên dev
  `device` có thể là `cuda` (GTX 1070) nhưng **fp16 + Qwen3-4B cư trú 24GB là residual** —
  `models_loaded:true` đầy đủ chỉ đạt trên GPU box. `WORKER_HOST`/`WORKER_PORT` đổi chỗ bind.
- Logger mặc định mức `WARNING` (`omnivoice.*`). Ở dev, log bắt được: boot, `/health`, auth
  (403/503), **bound-input 422**, và lỗi mix/ffmpeg — KHÔNG bắt được inference thật (residual).

### 3.3. Client (Vite / Tauri / Rust)
```bash
cd apps/client
pnpm dev            # Vite -> http://localhost:1420 (strictPort); log ở console trình duyệt (F12)
pnpm tauri dev      # cửa sổ Tauri thật + devtools; log Rust ra terminal chạy lệnh
```
Lỗi Rust/side-effect (path validation, cleanup, mux) hiện ở **terminal** chạy `tauri dev`.
Unit-test Rust in chi tiết: `cd src-tauri && cargo test -- --nocapture`.

### 3.4. Ghi log ra FILE để chia sẻ / soi lại
- **PowerShell:** `pnpm run test:py *>&1 | Tee-Object -FilePath dev-testpy.log`
  (hoặc `... > dev-testpy.log 2>&1` nếu không cần xem song song).
- **Bash/POSIX:** `pnpm run test:py 2>&1 | tee dev-testpy.log`.
- Test verbose khi cần dấu vết sâu: `cd apps/gpu-worker && python -m pytest -v --tb=short`.

> **⚠ Zero-Logging khi CHIA SẺ log:** pipeline đã lược nội dung nhạy cảm khỏi log/response,
> nhưng `WORKER_DEBUG` traceback (STDERR local) hoặc log tự thêm khi debug CÓ THỂ lộ đường
> dẫn/URL/kịch bản. **Trước khi dán log ra ngoài**, rà lại và bỏ mọi URL audio, token, và
> văn bản kịch bản. Log dev là để sửa lỗi tại chỗ, không phải để xuất khẩu nội dung người dùng.

---

## 4. VÒNG LẶP SỬA LỖI (dev)

1. **Tái hiện (reproduce):** dựng đúng khối lỗi (§3). Lỗi hợp đồng/auth/validation → tái hiện
   bằng test hoặc `curl`/Postman vào Gateway `:8787` / Worker `:8000`.
2. **Bắt log:** bật `WORKER_DEBUG=1` (worker) hoặc đọc console wrangler/Vite/terminal Rust;
   ghi ra file (§3.4).
3. **Khoanh vùng:** ánh xạ tên loại lỗi / dòng log → file nguồn. Viết **1 test tái hiện đỏ**
   trước khi sửa (Vitest / Pytest / cargo tuỳ khối).
4. **Sửa 1 hạng mục / lần**, chạy lại đúng cổng của khối đó (§2) cho tới khi xanh.
5. **Chống hồi quy:** chạy trọn `pnpm run test` + cổng còn lại của khối vừa đụng.
6. **Kiểm No-Fake-Success:** hỏi "chỗ này có đang báo 'thành công' cho việc chưa thực sự chạy
   không?". Nếu lỗi thuộc residual (cần GPU/Cloudflare thật) → **KHÔNG vá giả** ở dev; ghi vào
   nhóm residual và chuyển Track A.
7. **Ghi nhận:** cập nhật số baseline (§2) nếu số test đổi; ghi thay đổi vào
   [CODE_REVIEW.md](CODE_REVIEW.md).

---

## 5. KHI ĐÃ CẠN Ở DEV → TRACK A (GPU box)

Hết những gì §1 đánh dấu ✅ mà vẫn còn hạng mục ❌ (inference thật, fp16, tốc độ, deploy
Cloudflare thật) → chuyển sang GPU box theo [DEPLOYMENT_AND_TESTING.md](DEPLOYMENT_AND_TESTING.md):
build/boot image, `GET <tunnel>/health` = `models_loaded:true device:cuda`, và
`pnpm run test:py:gpu` xanh **là điều kiện tiên quyết** để được phép nói "đã deploy". Trước mốc
đó, mọi tuyên bố "chạy được trên GPU" là chưa kiểm chứng.
