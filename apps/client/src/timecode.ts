/**
 * Quy mốc thời gian người dùng nhập ("HH:MM:SS", "MM:SS", "SS", hoặc số/thập phân
 * kiểu "1,5") về GIÂY (số thực) TRƯỚC KHI KÝ. Hợp đồng `ClientSegment.start/end`
 * là số giây; worker (translation_service tính duration, audio_engine căn mix) đọc
 * SỐ — nếu gửi thẳng chuỗi "HH:MM:SS", `float()` phía worker sẽ hỏng rồi rơi về
 * pacing mặc định sai (CC-1). Không parse được -> 0 (không đoán bừa). Logic này
 * phản chiếu 1-1 src/timecode.py phía worker để hai đầu nhất quán tuyệt đối.
 *
 * Tách khỏi App.tsx (Đợt 29) để unit-test được BẰNG hàm thuần, không kéo theo
 * React/Tauri vào module graph của test.
 */
export function timecodeToSeconds(tc: string): number {
  const s = (tc ?? '').trim().replace(',', '.');
  if (!s) return 0;
  const direct = Number(s);
  if (!Number.isNaN(direct)) return direct; // đã là số ("83", "1.5")
  const parts = s.split(':').map((p) => Number(p));
  if (parts.some((n) => Number.isNaN(n))) return 0; // rác -> 0
  return parts.reduce((acc, n) => acc * 60 + n, 0);
}
