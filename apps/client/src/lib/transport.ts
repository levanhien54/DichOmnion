/**
 * Vận chuyển AUDIO lên lưu trữ đối tượng (R2) để GPU worker tải về xử lý.
 *
 * Nguyên tắc No-Fake-Success: client CHỈ tải AUDIO đã tách cục bộ (~30MB), KHÔNG
 * bao giờ tải video gốc (~600MB). URL tải lên KHÔNG còn là hằng số môi trường —
 * thay vào đó, với MỖI job, client xin Gateway ký một cặp URL presigned R2 riêng
 * (Option A: "Gateway ký URL mỗi job"). Nhờ vậy:
 *   • Gateway chỉ KÝ url, không bao giờ chạm vào bytes audio (Zero-Logging).
 *   • Mỗi job có object key riêng `audio/<deviceId>/<jobId>.wav` → không đụng nhau,
 *     url của job này không thể trỏ tới object của job khác.
 *   • URL sống ngắn (mặc định 2h) rồi hết hạn.
 *
 * Nếu Gateway từ chối ký (chưa cấu hình R2, bị throttle, kill-switch…), hàm NÉM
 * LỖI rõ ràng — tuyệt đối không bịa "URL giả", vì worker phải tải được audio THẬT.
 */

import { deterministicStringify } from '@dichomnion/shared-types';

export interface UploadAudioParams {
  /** Đường dẫn audio cục bộ (đã tách), đọc ở lớp Rust. */
  audioPath: string;
  /** Gốc Gateway, ví dụ 'http://localhost:8787'. */
  gatewayUrl: string;
  /** ID thiết bị đã đăng ký (khớp public key trong KV của Gateway). */
  deviceId: string;
  /** ID job — quyết định object key duy nhất cho audio của job này. */
  jobId: string;
  /** Ký ECDSA chuỗi payload bằng private key của thiết bị (giữ ngoài transport). */
  signPayload: (payload: string) => Promise<string>;
}

interface MintedUrls {
  key: string;
  uploadUrl: string;
  getUrl: string;
  expiresSeconds: number;
}

/**
 * Xin Gateway ký cặp URL presigned R2 cho MỘT job, tải audio cục bộ lên bằng URL
 * PUT vừa ký, rồi trả về URL GET để worker tải audio về.
 *
 * Auth trùng khớp /api/jobs/create: header X-Device-Id + X-ECDSA-Signature ký trên
 * deterministicStringify({ jobId, timestamp }). timestamp là thời điểm thực (chống
 * replay ở Gateway theo cửa sổ ±30s).
 */
export async function uploadAudioForWorker(
  params: UploadAudioParams,
): Promise<string> {
  const { audioPath, gatewayUrl, deviceId, jobId, signPayload } = params;

  // 1) Xin Gateway ký URL riêng cho job này.
  const mintBody = deterministicStringify({ jobId, timestamp: Date.now() });
  const signature = await signPayload(mintBody);

  const base = gatewayUrl.replace(/\/+$/, '');
  const mintRes = await fetch(`${base}/api/uploads/presign`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Device-Id': deviceId,
      'X-ECDSA-Signature': signature,
    },
    body: mintBody,
  });

  if (!mintRes.ok) {
    let detail = '';
    try {
      const err = (await mintRes.json()) as { error?: string };
      if (err?.error) detail = ` — ${err.error}`;
    } catch {
      /* body không phải JSON: bỏ qua, vẫn báo mã trạng thái */
    }
    throw new Error(
      `Gateway từ chối ký URL tải audio: HTTP ${mintRes.status}${detail}. ` +
        'Không dùng URL giả — worker phải tải được audio thật.',
    );
  }

  const minted = (await mintRes.json()) as MintedUrls;
  if (!minted?.uploadUrl || !minted?.getUrl) {
    throw new Error('Gateway trả về cặp URL ký không hợp lệ (thiếu uploadUrl/getUrl).');
  }

  // 2) Đọc bytes audio ở lớp Rust (toàn quyền đọc thư mục tạm), đưa qua IPC base64.
  const { invoke } = await import('@tauri-apps/api/tauri');
  const b64 = await invoke<string>('read_audio_b64', { audioPath });
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));

  // 3) Tải audio thẳng lên R2 bằng presigned PUT — Gateway không hề chạm vào bytes.
  const putRes = await fetch(minted.uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': 'audio/wav' },
    body: bytes,
  });
  if (!putRes.ok) {
    throw new Error(`Tải audio lên R2 thất bại: HTTP ${putRes.status}`);
  }

  return minted.getUrl;
}
