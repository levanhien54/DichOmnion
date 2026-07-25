/**
 * Vận chuyển AUDIO lên lưu trữ đối tượng (R2/S3) để GPU worker tải về xử lý.
 *
 * Nguyên tắc No-Fake-Success: client CHỈ tải AUDIO đã tách cục bộ (~30MB), KHÔNG
 * bao giờ tải video gốc (~600MB). Nếu chưa cấu hình lưu trữ đối tượng thật, hàm
 * NÉM LỖI rõ ràng — tuyệt đối không bịa "URL giả" (dummy) như bản cũ, vì worker
 * phải tải được audio THẬT thì mới xử lý được.
 */

/**
 * Tải audio cục bộ lên và trả về URL mà worker sẽ dùng để tải audio về.
 *
 * Cấu hình (biến môi trường Vite của deployer):
 *   VITE_AUDIO_UPLOAD_URL  — presigned PUT URL để đẩy audio lên bucket.
 *   VITE_AUDIO_PUBLIC_URL  — URL worker dùng để GET audio (đọc được từ worker).
 */
export async function uploadAudioForWorker(audioPath: string): Promise<string> {
  const putUrl = import.meta.env.VITE_AUDIO_UPLOAD_URL as string | undefined;
  const getUrl = import.meta.env.VITE_AUDIO_PUBLIC_URL as string | undefined;

  if (!putUrl || !getUrl) {
    throw new Error(
      'Chưa cấu hình lưu trữ đối tượng (R2/S3) cho audio. Cần đặt ' +
        'VITE_AUDIO_UPLOAD_URL (presigned PUT) và VITE_AUDIO_PUBLIC_URL (URL worker ' +
        'tải audio về). Không dùng URL giả — worker phải tải được audio thật.',
    );
  }

  // Đọc bytes audio ở lớp Rust (toàn quyền đọc thư mục tạm), đưa qua IPC dạng base64.
  const { invoke } = await import('@tauri-apps/api/tauri');
  const b64 = await invoke<string>('read_audio_b64', { audioPath });
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));

  const res = await fetch(putUrl, {
    method: 'PUT',
    headers: { 'Content-Type': 'audio/wav' },
    body: bytes,
  });
  if (!res.ok) {
    throw new Error(`Tải audio lên lưu trữ đối tượng thất bại: HTTP ${res.status}`);
  }

  return getUrl;
}
