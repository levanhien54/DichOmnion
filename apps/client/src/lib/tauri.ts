/**
 * Cầu nối tới lớp Rust của Tauri (FFmpeg cục bộ).
 *
 * Nguyên tắc No-Fake-Success: nếu KHÔNG chạy trong Tauri (ví dụ mở bằng trình duyệt
 * lúc dev), các lệnh dưới đây NÉM LỖI rõ ràng thay vì giả vờ tách/mux thành công.
 * Việc tách audio thật chỉ diễn ra trên desktop app có ffmpeg sidecar đi kèm.
 */

export interface AudioInfo {
  audio_path: string;
  md5: string;
  size_bytes: number;
}

/** True nếu đang chạy bên trong runtime Tauri (có cầu invoke). */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_IPC__' in window;
}

function assertTauri(): void {
  if (!isTauri()) {
    throw new Error(
      'Chức năng xử lý video cục bộ chỉ khả dụng trong ứng dụng desktop OmniVoice ' +
        '(cần ffmpeg đi kèm). Trình duyệt web không thể truy cập file gốc của bạn.',
    );
  }
}

/** Mở hộp thoại chọn video, trả về đường dẫn tuyệt đối trên máy (hoặc null nếu hủy). */
export async function pickVideoFile(): Promise<string | null> {
  assertTauri();
  const { open } = await import('@tauri-apps/api/dialog');
  const selected = await open({
    multiple: false,
    filters: [{ name: 'Video', extensions: ['mp4', 'mkv', 'mov', 'webm', 'avi'] }],
  });
  if (typeof selected === 'string') return selected;
  return null;
}

/** Tách audio 16kHz mono từ video CỤC BỘ bằng ffmpeg (Rust). Chỉ audio rời máy. */
export async function extractAudio(videoPath: string): Promise<AudioInfo> {
  assertTauri();
  const { invoke } = await import('@tauri-apps/api/tauri');
  return invoke<AudioInfo>('extract_audio', { videoPath });
}

/** Ghép track audio đã lồng tiếng vào lại video gốc CỤC BỘ (giữ nguyên hình ảnh). */
export async function muxAudioToVideo(videoPath: string, audioPath: string): Promise<string> {
  assertTauri();
  const { invoke } = await import('@tauri-apps/api/tauri');
  return invoke<string>('mux_audio_to_video', { videoPath, audioPath });
}

/**
 * CLIENT-01: ghi AUDIO lồng tiếng tải về (base64) ra file tạm, trả đường dẫn để mux.
 * Chỉ khả dụng trong Tauri (cần ffmpeg + ghi file cục bộ).
 */
export async function writeTempAudio(audioB64: string): Promise<string> {
  assertTauri();
  const { invoke } = await import('@tauri-apps/api/tauri');
  return invoke<string>('write_temp_audio', { audioB64 });
}

/**
 * CLIENT-01: chép video kết quả (temp) ra vị trí người dùng chọn. `destPath` lấy từ
 * hộp thoại Save của Tauri (đường dẫn tuyệt đối người dùng tự chọn).
 */
export async function saveOutputVideo(tempPath: string, destPath: string): Promise<void> {
  assertTauri();
  const { invoke } = await import('@tauri-apps/api/tauri');
  await invoke('save_output_video', { tempPath, destPath });
}

/**
 * TR-2: dọn file TẠM do app tạo (audio đã tách...) sau khi dùng xong. Best-effort —
 * KHÔNG ném lỗi và no-op ngoài Tauri, để việc dọn rác không bao giờ làm hỏng luồng
 * chính. Rust vẫn chặn xóa file ngoài thư mục temp/tiền tố "omnivoice_".
 */
export async function cleanupTempFile(path: string | null | undefined): Promise<void> {
  if (!path || !isTauri()) return;
  try {
    const { invoke } = await import('@tauri-apps/api/tauri');
    await invoke('cleanup_temp_file', { path });
  } catch {
    // Dọn rác là best-effort: nuốt lỗi, không ảnh hưởng kết quả người dùng.
  }
}
