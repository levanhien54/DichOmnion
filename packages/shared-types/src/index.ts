/**
 * Định dạng thẻ cảm xúc chuẩn cho AI TTS.
 */
export type EmotionTag = 
  | 'NEUTRAL'
  | 'HAPPY'
  | 'ANGRY'
  | 'SAD'
  | 'WHISPERING'
  | 'SHOUTING';

/**
 * Định nghĩa cấu trúc của một câu dịch (Segment).
 */
export interface TranslationSegment {
  id: number;
  start: number;
  end: number;
  duration: number;
  original_text: string;
  original_syllables: number;
  translated_text: string;
  // translated_syllables/pacing_status đã bỏ khỏi hợp đồng: worker KHÔNG còn phát 2
  // trường này (xem translation_service.py) và không có consumer nào đọc chúng.
  emotion: EmotionTag;
  speaker_id: string;
}

/**
 * Cấu hình tham số dịch thuật do người dùng chọn
 */
export interface TranslationConfig {
  targetLanguage: string;
  // 'Original' đã bỏ: worker KHÔNG có nhánh giữ-nguyên-gốc (style chỉ là chuỗi bơm vào
  // prompt LLM, luôn dịch sang targetLanguage). Giữ nó = quảng cáo tính năng không tồn
  // tại (chọn 'Original' vẫn bị dịch) -> gỡ để hợp đồng khớp hành vi thật.
  translationStyle: 'Formal' | 'Casual' | 'Slang';
  // Ngôn ngữ GỐC của thoại (tùy chọn) — để worker đếm âm tiết đúng khi căn lip-sync.
  // Vắng mặt = worker mặc định "en". Optional nên không phá vỡ payload/chữ ký cũ.
  sourceLanguage?: string;
}

/**
 * Một câu thoại do NGƯỜI DÙNG duyệt/nhập ở client (Human-in-the-loop), gửi kèm
 * JobRequest để định hướng/ghi đè ASR. Đây là HỢP ĐỒNG client -> worker.
 *
 * CC-1: `start`/`end` là GIÂY (số thực), KHÔNG phải timecode "HH:MM:SS". Worker
 * dùng chúng để tính `duration = end - start` (căn lip-sync) và định vị clip khi
 * mix. UI hiển thị "HH:MM:SS" cho người xem, nhưng client PHẢI quy về giây trước
 * khi ký (xem timecodeToSeconds trong App.tsx) — nếu gửi chuỗi timecode, `float()`
 * phía worker sẽ hỏng và pacing rơi về mặc định sai.
 */
export interface ClientSegment {
  id: string;
  speaker: string; // worker đọc làm speaker_id (chọn giọng đa nhân vật)
  text: string;    // worker đọc làm original_text (thoại người dùng duyệt)
  start: number;   // giây
  end: number;     // giây
}

/**
 * Yêu cầu gửi từ Client lên Gateway (Job Request)
 */
export interface JobRequest {
  jobId: string;
  videoAudioUrl: string; // Presigned URL tới R2
  config: TranslationConfig;
  speakerMapping: Record<string, string>;
  timestamp: number;
  segments?: ClientSegment[];
}

/**
 * [OPTIMIZED] Deterministic Stringify
 * Hàm sắp xếp key theo thứ tự Alphabet (A-Z) để đảm bảo chuỗi JSON 
 * sinh ra ở Client và Gateway giống nhau 100%, phục vụ cho việc ký ECDSA.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function deterministicStringify(obj: any): string {
  if (obj === null) return 'null';

  const t = typeof obj;
  if (t !== 'object') {
    // Phản chiếu ĐÚNG ngữ nghĩa JSON.stringify: undefined/function/symbol KHÔNG
    // phải giá trị JSON hợp lệ. Ở vị trí phần tử mảng / giá trị đơn, JSON.stringify
    // quy chúng về "null" (còn ở vị trí khóa object thì bỏ hẳn khóa — xử lý ở dưới).
    // Bản cũ trả thẳng JSON.stringify(undefined) === undefined (giá trị JS, không
    // phải chuỗi) khiến template literal chèn "key":undefined -> JSON hỏng và chuỗi
    // ký ở Client ≠ chuỗi dựng lại ở Gateway sau round-trip JSON.
    if (t === 'undefined' || t === 'function' || t === 'symbol') return 'null';
    return JSON.stringify(obj);
  }

  if (Array.isArray(obj)) {
    return '[' + obj.map(deterministicStringify).join(',') + ']';
  }

  // Bỏ các khóa có giá trị không-serialize-được, đúng như JSON.stringify bỏ chúng:
  // {a: undefined} -> "{}" chứ không phải {"a":undefined}. Nếu không, một payload
  // mang khóa undefined tường minh sẽ ký ra chuỗi mà Gateway (nhận qua JSON, đã mất
  // khóa đó) không tài nào dựng lại -> chữ ký lệch dù dữ liệu hợp lệ.
  const keys = Object.keys(obj)
    .filter((key) => {
      const vt = typeof obj[key];
      return vt !== 'undefined' && vt !== 'function' && vt !== 'symbol';
    })
    .sort();
  const props = keys.map((key) => `${JSON.stringify(key)}:${deterministicStringify(obj[key])}`);
  return '{' + props.join(',') + '}';
}
