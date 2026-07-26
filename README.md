# DichOmnion — OmniVoice V2

Hệ thống lồng tiếng (dubbing) video **Human-in-the-Loop**, kiến trúc **Zero-Trust /
Zero-Logging**: Client bóc tách audio cục bộ và **chỉ upload AUDIO** (không bao giờ gửi
video gốc), ký **ECDSA P-256**; Gateway (Cloudflare Worker) xác minh chữ ký + cấp **JWT
ES256** mỗi job + chống gian lận; GPU Worker (FastAPI) chạy Whisper + Qwen + TTS + trộn âm
trên VRAM. Đây là monorepo `pnpm` gồm **3 ứng dụng** + **2 package dùng chung**.

> **Runbook triển khai chuẩn (canonical): [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).**
> Kiểm thử thực tế + thu log: [docs/DEPLOYMENT_AND_TESTING.md](docs/DEPLOYMENT_AND_TESTING.md).
> Lịch sử review + trạng thái từng tiêu chí: [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md).

---

## Cấu trúc kho

| Đường dẫn | Vai trò |
|---|---|
| `apps/client` | **Client Desktop** — Tauri (Rust) + React + Vite. Bóc audio bằng ffmpeg sidecar, ký ECDSA, PUT audio lên R2/S3 (presigned phía Client), gửi URL công khai cho Gateway, tải kết quả về và mux lại video cục bộ. |
| `apps/gateway` | **API Gateway** — Cloudflare Worker (Hono). Trạm 1 xác minh ECDSA, Trạm 2 cấp JWT ES256 mỗi job, Trạm 3 Turnstile + anomaly/timing + Financial Kill Switch. Chỉ bind **KV** (không Queue, không R2). |
| `apps/gpu-worker` | **GPU Worker** — FastAPI (`uvicorn src.main:app`), thường trú sau `cloudflared` tunnel. Whisper (ASR) + Qwen (dịch cục bộ) thường trú VRAM, Demucs (tách nền), TTS (GPT-SoVITS/edge-tts), trộn âm (pydub), semaphore size-1. |
| `packages/crypto-utils` | `@dichomnion/crypto-utils` — ký/verify ECDSA, chuẩn hoá payload tất định (dùng chung Client + Gateway). |
| `packages/shared-types` | `@dichomnion/shared-types` — kiểu TypeScript hợp đồng job (dùng chung Client + Gateway). |

---

## Yêu cầu môi trường

- **Node ≥ 18** và **pnpm ≥ 9** (đã khai báo trong `package.json` gốc; `packageManager: pnpm@9`).
- **Python 3.11+** và **[uv](https://docs.astral.sh/uv/)** — cho `apps/gpu-worker`.
- **Rust toolchain** (cargo) + [điều kiện tiên quyết Tauri v1](https://tauri.app/v1/guides/getting-started/prerequisites) — chỉ khi build/chạy Client Desktop.
- **ffmpeg** khả dụng cho Client (đóng gói dạng sidecar khi build Tauri).

---

## Cài đặt

```bash
pnpm install        # cài toàn bộ workspace (apps/* + packages/*) từ thư mục gốc
```

GPU Worker có môi trường Python riêng (không nằm trong pnpm):

```bash
cd apps/gpu-worker
uv sync                        # cài dependency chạy mặc định (CPU-safe)
# tuỳ chọn khi lên GPU thật:
uv sync --extra enhance        # torchaudio + demucs + audioseal (tách nền / watermark)
uv sync --extra local-llm      # transformers + accelerate (Qwen dịch cục bộ)
```

---

## Chạy cục bộ (dev)

Mỗi app chạy độc lập. Cách nhanh nhất trên Windows là dùng launcher ở gốc:

- **`start-all.bat`** — mở 3 cửa sổ: Gateway (`:8787`), Client Vite (`:1420`), GPU Worker (`:8000`, `--reload`).
- **`start-omnivoice-desktop.bat`** — chạy GPU Worker + Gateway rồi mở app Tauri đã build
  (`apps/client/src-tauri/target/release/OmniVoice.exe`; cần `pnpm tauri build` trước).

Hoặc chạy tay từng app:

```bash
# Client (giao diện web dev, Vite → http://localhost:1420 — cổng ghim strictPort trong vite.config.ts)
cd apps/client && pnpm dev
# Client Desktop (Tauri): pnpm tauri dev   |   đóng gói: pnpm tauri build

# Gateway (wrangler dev → http://localhost:8787)
cd apps/gateway && pnpm run dev
#   Sao chép apps/gateway/.dev.vars.example → .dev.vars để cấp secret cục bộ
#   (thiếu key → đường có xác-thực fail-closed, xem file mẫu).

# GPU Worker (FastAPI → http://localhost:8000)
cd apps/gpu-worker && uv run uvicorn src.main:app --port 8000
```

Cấu hình môi trường: xem `apps/client/.env.example`, `apps/gpu-worker/.env.example`,
`apps/gateway/.dev.vars.example` (đều là placeholder, KHÔNG chứa khoá thật).

---

## Build & Test

```bash
pnpm run build      # CHỈ build packages/* (crypto-utils, shared-types) — KHÔNG build apps/*.
                    # Build app: Client -> pnpm tauri build; Gateway -> deploy; Worker -> Docker.

pnpm run test       # = test:ts (tsc/vitest toàn workspace) + test:py (pytest CPU của gpu-worker)
pnpm run test:ts    # pnpm -r run test  (client: tsc --noEmit; gateway: vitest run; packages)
pnpm run test:py    # cd apps/gpu-worker && python -m pytest -q  (bộ CPU, an toàn máy dev)
```

Kiểm tra bổ sung:

```bash
cd apps/gateway && npx wrangler deploy --dry-run   # validate bundle offline (không deploy)
cd apps/client/src-tauri && cargo test             # unit test Rust của Client
```

> **`pnpm run test:py:gpu`** (`RUN_GPU_ACCEPT=1 … -m gpu_acceptance`) là bài kiểm THẬT chỉ
> chạy trên **GPU box (shell POSIX)** — dùng cú pháp env inline kiểu POSIX nên **không chạy
> trực tiếp trên PowerShell/CMD của Windows**. Đây là hạng mục residual (xem bên dưới).

---

## Ranh giới No-Fake-Success (residual_hardware)

Dự án **không tô xanh giả**. Mọi thứ chỉ phần cứng thật chứng minh được — build/boot image,
fp16 vừa 24GB VRAM, cold-start, `gpu_acceptance`/`test:py:gpu` thật, độ trễ inference, tạo KV
namespace/secret/tunnel Cloudflare thật, presign R2 mỗi job — thuộc nhóm **residual_hardware**
và **chỉ nghiệm thu trên GPU ≥24GB + tài khoản Cloudflare thật**. Bộ test mặc định (`pnpm run
test`) cố ý **loại** nhóm này để suite CPU của máy dev vẫn xanh trung thực. Bảng residual đầy
đủ: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) §4. Không được nói "đã deploy" cho tới khi
`GET <tunnel>/health` trả `models_loaded:true device:cuda` **và** `test:py:gpu` xanh trên GPU box.
