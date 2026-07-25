#!/usr/bin/env node
/**
 * Financial Kill Switch — standalone billing watchdog.
 *
 * Chạy NGOÀI Cloudflare Worker (cron/host riêng). Nhiệm vụ: định kỳ đọc mức chi
 * tiêu thực tế từ API billing của nhà cung cấp GPU (RunPod/Modal/Cloudflare...),
 * và khi vượt ngưỡng thì gọi endpoint admin để BẬT kill switch — sau đó mọi
 * /api/auth/register và /api/jobs/create của Gateway trả 503, chặn cháy túi.
 *
 * Nguyên tắc No-Fake-Success: nếu chưa cấu hình nguồn billing thật, script TỪ CHỐI
 * khởi động (không giả vờ "đang giám sát"). Kill switch một khi đã bật sẽ KHÔNG tự
 * tắt — phải có người xác nhận rồi POST {active:false} để mở lại.
 *
 * Biến môi trường:
 *   GATEWAY_URL        (bắt buộc) vd https://gateway.example.workers.dev
 *   ADMIN_TOKEN        (bắt buộc) trùng với secret ADMIN_TOKEN của Worker
 *   BILLING_API_URL    (bắt buộc) endpoint trả JSON chứa mức chi tiêu hiện tại
 *   BILLING_API_TOKEN  (tuỳ chọn) Bearer token cho BILLING_API_URL
 *   BILLING_JSON_FIELD (tuỳ chọn) tên trường USD trong JSON (mặc định "spendUSD")
 *   SPEND_THRESHOLD_USD(bắt buộc) ngưỡng USD để kích hoạt kill switch
 *   POLL_INTERVAL_S    (tuỳ chọn) chu kỳ kiểm tra, mặc định 60s
 *   ONESHOT            (tuỳ chọn) "1" => kiểm tra một lần rồi thoát (dùng cho cron)
 */

function requireEnv(name) {
  const v = process.env[name];
  if (!v) {
    console.error(`[kill-switch] Thiếu biến môi trường bắt buộc: ${name}`);
    console.error('[kill-switch] Từ chối khởi động — không giám sát giả.');
    process.exit(1);
  }
  return v;
}

const GATEWAY_URL = requireEnv('GATEWAY_URL').replace(/\/$/, '');
const ADMIN_TOKEN = requireEnv('ADMIN_TOKEN');
const BILLING_API_URL = requireEnv('BILLING_API_URL');
const BILLING_API_TOKEN = process.env.BILLING_API_TOKEN;
const BILLING_JSON_FIELD = process.env.BILLING_JSON_FIELD || 'spendUSD';
const SPEND_THRESHOLD_USD = Number(requireEnv('SPEND_THRESHOLD_USD'));
const POLL_INTERVAL_MS = (Number(process.env.POLL_INTERVAL_S) || 60) * 1000;
const ONESHOT = process.env.ONESHOT === '1';

if (!Number.isFinite(SPEND_THRESHOLD_USD) || SPEND_THRESHOLD_USD <= 0) {
  console.error('[kill-switch] SPEND_THRESHOLD_USD phải là số dương.');
  process.exit(1);
}

/** Đọc mức chi tiêu USD hiện tại từ API billing thật. Ném lỗi nếu không đọc được. */
async function readSpendUSD() {
  const headers = BILLING_API_TOKEN ? { Authorization: `Bearer ${BILLING_API_TOKEN}` } : {};
  const res = await fetch(BILLING_API_URL, { headers });
  if (!res.ok) {
    throw new Error(`Billing API HTTP ${res.status}`);
  }
  const data = await res.json();
  const spend = Number(data?.[BILLING_JSON_FIELD]);
  if (!Number.isFinite(spend)) {
    throw new Error(`Billing API không trả trường số "${BILLING_JSON_FIELD}"`);
  }
  return spend;
}

/** Bật kill switch qua endpoint admin của Gateway. */
async function activateKillSwitch() {
  const res = await fetch(`${GATEWAY_URL}/api/admin/kill-switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Token': ADMIN_TOKEN },
    body: JSON.stringify({ active: true }),
  });
  if (!res.ok) {
    throw new Error(`Không bật được kill switch: HTTP ${res.status}`);
  }
  return res.json();
}

async function checkOnce() {
  // Zero-Logging: chỉ log con số tổng hợp, không log token/URL/nội dung nhạy cảm.
  const spend = await readSpendUSD();
  const ts = new Date().toISOString();
  if (spend >= SPEND_THRESHOLD_USD) {
    console.warn(`[kill-switch] ${ts} CHI TIÊU $${spend} >= ngưỡng $${SPEND_THRESHOLD_USD}. Kích hoạt!`);
    const out = await activateKillSwitch();
    console.warn(`[kill-switch] Trạng thái: ${JSON.stringify(out)}`);
    return true;
  }
  console.log(`[kill-switch] ${ts} OK — chi tiêu $${spend} / ngưỡng $${SPEND_THRESHOLD_USD}.`);
  return false;
}

async function main() {
  if (ONESHOT) {
    try {
      const tripped = await checkOnce();
      process.exit(tripped ? 2 : 0); // exit 2 = đã kích hoạt (để cron/alert nhận biết)
    } catch (e) {
      console.error(`[kill-switch] Lỗi kiểm tra: ${e.message}`);
      process.exit(1);
    }
  }

  console.log(`[kill-switch] Bắt đầu giám sát mỗi ${POLL_INTERVAL_MS / 1000}s. Ngưỡng $${SPEND_THRESHOLD_USD}.`);
  // Vòng lặp giám sát liên tục. Một khi đã kích hoạt thì dừng (cần người mở lại).
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const tripped = await checkOnce();
      if (tripped) {
        console.warn('[kill-switch] Đã kích hoạt — dừng giám sát. Cần xác nhận thủ công để mở lại.');
        process.exit(2);
      }
    } catch (e) {
      // Lỗi tạm thời của billing API KHÔNG tự động bật kill switch (tránh false-positive
      // gây sập dịch vụ oan). Chỉ ghi log và thử lại chu kỳ sau.
      console.error(`[kill-switch] Lỗi tạm thời khi đọc billing: ${e.message}`);
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

main();
