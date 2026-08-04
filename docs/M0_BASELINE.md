# M0 — Baseline & Reproducibility (status)

Ngắn gọn: M0 biến "test xanh vì máy dev tình cờ đã cài đủ gói" thành **môi trường khóa
được, tái lập từ một checkout sạch**, cộng một lệnh `verify` và CI tối thiểu. Đây là mốc
P0 nền cho M1→M8 (xem [CLAUDE_CODE_CONTINUATION_PLAN.md](CLAUDE_CODE_CONTINUATION_PLAN.md)).
KHÔNG dùng để tô xanh phần chỉ GPU/Cloudflare thật chứng minh được (xem §Residual).

## Lệnh tái lập (chạy từ gốc `D:\DichOmnion`)

| Lệnh | Ý nghĩa |
|---|---|
| `pnpm install --frozen-lockfile` | Cài JS đúng theo `pnpm-lock.yaml` (không đổi lock). |
| `pnpm run verify` | Cổng xanh dev đầy đủ, theo thứ tự: packages → client → gateway → worker → Rust. |
| `pnpm run test` | `test:ts` + `test:py` (không build/dry-run/Rust). |
| `cd apps/gpu-worker && uv lock --check` | Khẳng định `uv.lock` khớp `pyproject.toml`. |
| `cd apps/gpu-worker && uv sync --frozen` | Dựng `.venv` khóa được (cài **nhóm dev** mặc định — PEP 735). |
| `node scripts/provision-ffmpeg.mjs --check` | Xác minh sidecar FFmpeg đúng SHA-256 (fail-closed). |

`pnpm run verify` gồm các script con: `verify:packages`, `verify:client`
(provision-ffmpeg `--check` + `tsc && vite build`), `verify:gateway` (vitest + `wrangler
deploy --dry-run`), `verify:worker` (`uv run --frozen pytest`), `verify:rust`
(`cargo fmt --check` + `cargo clippy --all-targets -D warnings` + `cargo test`).

## Baseline đo được (mốc M0)

| Khối | Kết quả thật |
|---|---|
| `test:ts` | **163 passed** — crypto-utils 8, gateway 145 (5 file), client 10 (2 file) + `tsc --noEmit` sạch |
| `test:py` (locked `.venv`) | **228 passed, 5 skipped, 4 deselected** |
| Rust (`cargo test`) | **18 passed; 0 failed** |
| Rust fmt/clippy | `cargo fmt --check` sạch; `cargo clippy --all-targets -D warnings` sạch |
| Gateway `wrangler deploy --dry-run` | Bundle hợp lệ ~134 KiB, in binding KV_CACHE + vars; KHÔNG deploy |
| `uv lock --check` | Resolved 107 packages — lock khớp pyproject |

**5 skipped** trong `test:py` là các test mix/atempo trong `tests/test_audio.py` cần **ffmpeg
trên PATH** (fail-soft có chủ đích — bỏ qua trung thực, KHÔNG giả pass). Chúng chạy khi PATH
có ffmpeg. **4 deselected** = nhóm `gpu_acceptance` (chỉ chạy trên GPU box, `RUN_GPU_ACCEPT=1`).

> Chênh lệch với baseline cũ (232 passed dưới **system Python**): đúng 4 test mix chuyển
> passed→skipped vì `.venv` khóa không có ffmpeg trên PATH. Không phải mất coverage ngầm —
> là khác biệt PATH, giải quyết bằng cơ chế provisioning FFmpeg dưới đây.

## Thay đổi chính của M0

1. **`uv.lock` tái khóa** khớp `pyproject.toml`; gỡ dep cloud chết (openai/google/grpcio…)
   vốn đã rời code từ P1 nhưng lock chưa cập nhật. `uv lock --check` xanh (107 gói).
2. **Nhóm dev PEP 735** — chuyển `pytest`/`pytest-asyncio` từ `[project.optional-dependencies].dev`
   (extra) sang `[dependency-groups].dev`. Nhờ đó `uv sync --frozen` / `uv run pytest` tái lập
   **mặc định**, không cần nhớ `--extra dev`. `enhance`/`local-llm` vẫn là extra tính năng thật.
3. **`test:py` → môi trường khóa**: `python -m pytest` (system) → `uv run --frozen pytest -q`.
4. **Provisioning FFmpeg checksum-pinned** — `scripts/provision-ffmpeg.mjs` +
   `scripts/ffmpeg-manifest.json`. Xác minh sidecar theo SHA-256 **fail-closed**; hỗ trợ tải
   + verify khi có `url` cố định; **fail rõ ràng** với target chưa được pin (macOS/Linux).
5. **Lệnh `verify`** gộp toàn bộ cổng xanh dev theo thứ tự.
6. **CI tối thiểu** `.github/workflows/ci.yml` — chạy lại đúng các cổng trên từ checkout sạch,
   windows-latest, KHÔNG secret / KHÔNG GPU / KHÔNG deploy.
7. **Sửa hồi quy Rust** phát hiện khi dựng cổng: `cargo fmt` (định dạng) + 2 lỗi clippy
   `cloned_ref_to_slice_refs` (`&[root.clone()]` → `std::slice::from_ref(&root)`) trong test.

## FFmpeg sidecar — cơ chế đã có vs. quyết định còn lại

- **Đã có (committed):** cơ chế provisioning + xác minh checksum, fail-closed, có map
  target-triple, fail rõ ràng cho nền chưa pin. Hash Windows là **thật**, đo từ build đã cam kết:
  `ffmpeg-x86_64-pc-windows-msvc.exe` — `N-124606-gdfd11985e8-20260523` (BtbN win64-gpl),
  204,028,416 bytes, SHA-256 `b12e3dbaa20bed82221b07d27b4468bd155d1999a70b23f84be6749004be83c0`.
- **Còn lại (quyết định của chủ repo):** cách **giao bytes cho checkout sạch**. Hai lựa chọn:
  (a) **Git LFS** commit chính binary (git-lfs 3.7.1 đã có; ~195 MB nằm trong hạn LFS free của
  GitHub), hoặc (b) đặt **`url` immutable** trong manifest mà SHA-256 khớp hash đã pin. Tới khi
  chọn, checkout sạch thiếu binary sẽ **fail-closed trung thực** (không giả provisioning). Đây
  là quyết định repo-shape/hạ tầng — chờ chủ repo. macOS/Linux: thêm build thật + hash khi cần
  bản cài các nền đó (No-Fake-Success: không bịa hash).

## Residual (KHÔNG chứng minh được ở M0/máy dev)

- Tauri **bundle** đầy đủ + E2E kéo-thả video thật (cần binary sidecar + môi trường Tauri).
- Bất kỳ hạng mục `residual_hardware`: inference cư trú VRAM, fp16 24GB, `gpu_acceptance`,
  độ trễ, cold-start, KV/secret/tunnel Cloudflare thật, R2 round-trip sống.
- **CI**: đã viết + YAML hợp lệ, nhưng **lần chạy thật đầu tiên xảy ra khi push lên GitHub**
  (repo hiện chưa commit/push các thay đổi này — chờ lệnh của người dùng).

## Tiếp theo

M1 — điều phối job bền (Durable Objects + Cloudflare Queues) thay `202 + waitUntil`. Cần ADR
ngắn + **quyết định tài nguyên trả phí** (DO/Queues cần gói Workers trả phí) trước khi tạo tài
nguyên; sẽ dựng/kiểm trên Workerd/Miniflare cục bộ trước, KHÔNG deploy.
