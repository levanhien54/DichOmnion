# API Gateway (Cloudflare Workers)

Serverless API bảo mật chuẩn Zero-Trust (Cloudflare Workers). Nhận lệnh từ Client, xác minh chữ ký ECDSA P-256 (Trạm 1), cấp JWT ES256 mỗi job (Trạm 2), chống bot Turnstile + anomaly/timing + Financial Kill Switch (Trạm 3), rồi điều phối xuống GPU Worker qua **cloudflared tunnel** (HTTP nội-bộ). Gateway chỉ bind **KV** — KHÔNG có Queue, KHÔNG bind R2 (R2 nằm hoàn-toàn phía Client).

## Lệnh khởi chạy
- `pnpm run dev` — chạy local (`wrangler dev src/index.ts`, cổng 8787). Boot được KHÔNG cần secret; các đường có xác-thực sẽ **fail-closed** (`gateway_key_missing`) cho tới khi `.dev.vars` cấp `GATEWAY_JWT_PRIVATE_KEY` — xem `.dev.vars.example`.
- `pnpm run test` — bộ test vitest (`vitest run`). KHÔNG cần `.dev.vars` (test tự inject key).
- `pnpm run deploy` — deploy production (`wrangler deploy --minify src/index.ts`). Quy-trình đầy-đủ (KV/secret/tunnel): `docs/DEPLOYMENT.md` §2.6.
