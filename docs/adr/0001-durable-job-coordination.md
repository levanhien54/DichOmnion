# ADR 0001 — Điều phối job bền vững (Durable Objects + Cloudflare Queues)

- **Trạng thái:** Được chấp nhận (triển khai + kiểm thử CỤC BỘ trên Workerd/Miniflare).
  Chưa deploy, chưa tạo tài nguyên trả phí — xem §"Ranh giới".
- **Mốc:** M1 (P0) của [CLAUDE_CODE_CONTINUATION_PLAN.md](../CLAUDE_CODE_CONTINUATION_PLAN.md).
- **Ngày:** 2026-08-04.

## Bối cảnh — vì sao `202 + waitUntil` không đủ bền

Đường điều phối hiện tại (`apps/gateway/src/index.ts`):

1. `POST /api/jobs/create` xác minh chữ ký/replay/kích thước, ghi `job:<dev>:<id> = {status:QUEUED}`
   vào **KV**, rồi gọi `background(c, dispatchToWorker(...))` — tức `c.executionCtx.waitUntil(...)`.
2. `dispatchToWorker` mở **một fetch dài** tới GPU worker `/api/worker/process` với timeout tới
   `MAX_PLAUSIBLE_MS = 15 phút`, và **chờ trọn** lần render bên trong `waitUntil`.

Bốn điểm yếu (mỗi điểm là một tiêu chí M1):

- **W1 — `waitUntil` không bền.** Một lần gọi Worker có trần thời gian sống; nếu instance bị thu
  hồi giữa chừng, tác vụ trong `waitUntil` **biến mất im lặng** — job kẹt `QUEUED` vĩnh viễn, không
  có gì tái điều phối. Vòng lặp retry 3× chỉ sống trong đúng một `waitUntil` đó.
- **W2 — không có state machine nguyên tử.** Chuyển trạng thái là `setJob` = KV `put` last-writer-wins.
  KV **nhất quán cuối** (eventual), không giao dịch: hai request đồng thời (hoặc một retry) có thể
  ghi đè trạng thái terminal; không luật nào chặn `DONE → QUEUED`.
- **W3 — idempotency có TOCTOU.** Kiểm tra job trùng là KV `get` rồi `put`. Hai request cùng `jobId`
  chạy song song có thể **cùng miss** `existing` rồi **cùng dispatch** → chạy GPU hai lần.
- **W4 — không có hàng đợi bền.** Nếu dispatch hỏng SAU khi đã trả 202, không có gì tái chạy nó.
  Hack `FINALIZING` (đọc lệch propagation giữa PoP) tồn tại chính vì KV không nhất quán mạnh.

## Quyết định

Chèn **một tầng điều phối bền** trước GPU worker, đúng kiến trúc đích §5 của kế hoạch:

```
Client → R2(input) → Gateway ─create→ [Durable Object: JobCoordinator] ─enqueue→ [Queue: job-dispatch]
                                              ▲                                          │
                                    (nguồn sự thật, nguyên tử)                    queue() consumer
                                              │                                          ▼
                          poll ◀── KV projection (ghi-xuyên) ◀────────────── dispatchToWorker → GPU → R2(output)
```

### 1. Durable Object `JobCoordinator` — nguồn sự thật + state machine

- **Địa chỉ:** một DO cho mỗi `(deviceId, jobId)` qua `idFromName("<deviceId>:<jobId>")`.
  DO **đơn luồng cho mỗi object** ⇒ mọi thao tác lên nó **tuần tự và nguyên tử** (diệt W2/W3).
- **Lưu trạng thái** trong DO storage (nhất quán MẠNH), KHÔNG phải KV.
- API nội bộ (gọi qua stub trong cùng Worker):
  - `create()` — **check-and-set nguyên tử**: đã tồn tại → trả bản ghi cũ (`idempotent:true`);
    chưa có → khởi tạo `QUEUED`. Thay thế TOCTOU của W3.
  - `transition(to, meta)` — **ép luật** state machine; từ chối chuyển bất hợp lệ; **terminal là dính**
    (không rời được). Thay thế `setJob` last-writer-wins của W2.
  - `get()` — đọc trạng thái quyền uy (poll dùng, diệt hack `FINALIZING` của W4).
- **State machine hợp lệ:**

  ```
  QUEUED ─► DISPATCHING ─► PROCESSING ─► DONE            (terminal)
     ▲            │             │      └─► FAILED          (terminal)
     │            ▼             ▼      └─► REJECTED_FRAUD  (terminal)
     └── RETRYING ◄────────────┘      └─► TERMINATED_TIMEOUT (terminal)
                                      └─► ERROR            (terminal)
  ```
  `RETRYING → DISPATCHING` cho lần thử lại; mọi terminal đều dính (chuyển tiếp bị bỏ qua, không lỗi).

- **Ghi-xuyên KV projection.** Mỗi `transition` cũng ghi `job:<dev>:<id>` / `result:<dev>:<id>` như
  cũ, nên **hợp đồng poll/download và toàn bộ test hiện có giữ nguyên**. DO là nguồn; KV là bản chiếu
  đọc. (Không đổi source-of-truth trong một bước lớn — giảm rủi ro, giữ TDD xanh.)

### 2. Cloudflare Queue `job-dispatch` — tách chấp nhận khỏi xử lý + retry BỀN

- `/api/jobs/create`: sau `DO.create()`, **enqueue** `{deviceId, jobId, payload}` rồi trả 202 NGAY.
  Không còn `background(dispatchToWorker)` trong `waitUntil` (diệt W1).
- **`queue()` consumer** kéo message và chạy `dispatchToWorker`, có **DO glate** trạng thái:
  - transient (5xx / lỗi mạng) → `message.retry()` — retry **của nền tảng**, bền, có backoff +
    dead-letter, **sống sót qua thu hồi Worker** (đây là bản vá cốt lõi cho W1).
  - terminal (4xx / gian lận / timeout) → `message.ack()` + `DO.transition(terminal)`.
- **Idempotent tiêu thụ:** Queue giao *ít-nhất-một-lần*; consumer đọc DO trước — nếu đã terminal
  hoặc đang bay thì no-op. DO là trọng tài (không double-dispatch dù message lặp).

### 3. Điểm gắn (entrypoint)

Worker phải xuất `default { fetch, queue }` + class DO. Để **không đụng** cách test import Hono app,
ta **gắn** consumer lên chính app: `app.queue = handleJobQueue` rồi `export default app`. App Hono vẫn
có `.fetch`/`.request` (test dùng), giờ có thêm `.queue` (runtime dùng). Class `JobCoordinator` được
`export` để khai báo binding.

## Chiến lược kiểm thử cục bộ (No-Fake-Success)

- **Logic** (state machine, idempotency, phân loại retry, định tuyến consumer): unit test trong
  node-vitest hiện có, dùng **stub `DurableObjectState`/storage trung thực** (Map-backed, phản chiếu
  ngữ nghĩa tuần tự) + gọi handler trực tiếp. **Không thêm dependency**, khớp harness sẵn có
  (`app.request` + `MemoryKV` + `vi.stubGlobal('fetch')`).
- **Chứng minh runtime thật:** `wrangler deploy --dry-run` biên dịch cấu hình DO binding + migration +
  queue producer/consumer qua **đúng toolchain workerd/esbuild** (đã nằm trong `pnpm run verify` + CI).
  Tùy chọn: smoke bằng `wrangler dev` (Miniflare hiện thực DO + Queues cục bộ) — như Đợt 31 đã làm.
- Không có "queue giả": test kiểm phân nhánh `retry()` vs `ack()` bằng một đối tượng message giả có
  spy, và kiểm hiệu ứng DO/KV thật sau đó.

## Ranh giới — dừng ở đâu (chờ quyết định người dùng)

- **Durable Objects và Queues cần gói Workers TRẢ PHÍ để DEPLOY.** Dựng + test trên Miniflare/Workerd
  cục bộ là **miễn phí**. ADR này (và code kèm theo) **DỪNG trước**:
  - `wrangler deploy` (deploy thật),
  - `wrangler queues create job-dispatch` (tạo tài nguyên trả phí),
  - tạo/áp migration DO trên tài khoản thật.
- Đó là quyết định at-the-moment của chủ tài khoản. Cho tới lúc đó, đường bền được chứng minh **chỉ
  cục bộ**; không tuyên bố "đã deploy". Round-trip R2 sống + KV/secret/tunnel Cloudflare thật vẫn là
  residual_hardware như trước.

## Hệ quả

- **Được:** dispatch bền qua thu hồi Worker (W1); chuyển trạng thái nguyên tử (W2); idempotency không
  còn TOCTOU (W3); retry bền có dead-letter thay vòng lặp trong-một-request (W4); poll đọc nguồn nhất
  quán mạnh.
- **Mất/chi phí:** thêm một DO class + một binding queue + cấu hình migration; độ trễ dispatch tăng
  một nhịp enqueue (không đáng kể so với render); **phụ thuộc gói trả phí để chạy production**.
- **Không đổi:** hợp đồng HTTP client (`202`, hình dạng poll/download), Trạm 2 JWT, Trạm 3 anti-fraud,
  các cổng validate input Đợt 17–25 — tất cả giữ nguyên; `dispatchToWorker` giữ nguyên hành vi bên
  trong, chỉ đổi NGUỒN gọi (queue thay `waitUntil`) và có DO gate.
