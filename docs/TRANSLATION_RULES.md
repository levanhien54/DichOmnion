# Tiêu Chuẩn Chất Lượng Dịch Thuật LLM (LLM Translation Guidelines)

Tài liệu này quy định các tiêu chuẩn khắt khe và hệ thống Prompt (câu lệnh chỉ thị) BẮT BUỘC phải sử dụng khi gọi Mô hình Ngôn ngữ Lớn để thực hiện tác vụ dịch thuật trong dự án OmniVoice.

Dịch thuật cho lồng tiếng (Dubbing) hoàn toàn khác với dịch thuật văn bản thông thường. Mục tiêu tối thượng là **Đồng bộ Khẩu hình (Lip-sync)** và **Giữ nguyên Cảm xúc (Emotion Preservation)**.

> **Cập nhật kiến trúc (Đợt 7 — Local Qwen):** dự án đã CHUYỂN từ LLM cloud (OpenAI `gpt-4o-mini`) sang **Qwen mã nguồn mở CHẠY CỤC BỘ** (mặc định `Qwen/Qwen3-4B-Instruct-2507`, chế độ non-thinking), thường trú VRAM cạnh Whisper (tiêu chí #6 GPU Model Residence). Lý do cốt lõi là **quyền riêng tư**: gửi transcript (từng câu thoại gốc) tới API cloud chính là hành vi rò rỉ plaintext ra bên thứ ba (tiêu chí #2), đúng lớp rò rỉ đã gắn cờ cho edge-tts. Chạy Qwen cục bộ khiến transcript KHÔNG bao giờ rời máy worker. Suy luận Qwen thật (chất lượng dịch, độ trễ, cư trú VRAM) chỉ kiểm chứng được trên GPU và được khai báo là `residual_hardware` — máy CPU-only fail-closed thay vì chạy giả.
>
> **Cập nhật (Đợt 8 — Self-review):** thêm cơ chế **Qwen tự chấm điểm + tự chỉnh sửa** bản dịch (§2.4). Vòng tự-soi này dùng **cổng KHÁCH QUAN đếm âm tiết** làm luật chấp nhận *never-worse* (không để mô hình tự khen là "tốt hơn"), chạy **FAIL-OPEN + bounded**, và **bỏ qua ngôn ngữ CJK/non-Latin** — chỉ tự-sửa trục **pacing** (lip-sync); phần fidelity/emotion-only là `residual_hardware`.

---

## 1. Cấu Hình Động Của Người Dùng (Dynamic User Configurations)

Hệ thống LLM không được phép dùng một Prompt cứng nhắc. API Gateway phải nhận các tham số (Parameters) từ người dùng (trên giao diện Client) và đưa vào System Prompt:

1. **Ngôn ngữ đích (Target Language):** Người dùng có quyền chọn dịch ra bất cứ ngôn ngữ nào (Việt, Anh, Nhật, Hàn, v.v.). LLM phải tuân thủ chuẩn ngữ pháp và văn hóa của ngôn ngữ đích đó.
2. **Phong cách dịch (Translation Style/Tone):** 
   - LLM phải linh hoạt thay đổi xưng hô và văn phong dựa trên tùy chọn của người dùng.
   - *Ví dụ:* `Trang trọng (Formal)` (Tôi - Bạn), `Tự nhiên (Casual)` (Mình - Cậu), `Đường phố/GenZ (Slang)` (Mày - Tao / Từ lóng). *(Tùy chọn "Giữ nguyên phong cách gốc" đã BỎ ở P3 — nó chưa từng được cài đặt trong đường dịch nên dropdown Client không còn liệt kê; giữ trong doc sẽ là quảng cáo tính năng ma.)*

---

## 2. Yêu Cầu Chất Lượng Dịch Thuật Lồng Tiếng (Dubbing Quality Standards)

### 2.1 Đo Tốc Độ & Đồng Bộ Khẩu Hình (Speed & Pacing Matching)
- **Vấn đề:** Văn bản tiếng Anh khi dịch sang tiếng Việt thường dài hơn 20-30%. Nếu cứ dịch sát nghĩa rồi để máy nói tốc độ bình thường, âm thanh sẽ bị tràn ra ngoài cảnh video.
- **Giải pháp (Đo tốc độ động):** 
  - Hệ thống ASR (WhisperX) sẽ cung cấp `start` và `end` time của câu gốc $\rightarrow$ Tính ra **Thời lượng nói (Duration)**.
  - Kết hợp với số lượng âm tiết gốc $\rightarrow$ Hệ thống tự tính ra **Tốc độ nói (Syllables per Second)** của người gốc trong cảnh đó.
  - **Yêu cầu LLM:** LLM sẽ nhận được thông số Tốc độ nói này. Nếu người gốc nói chớp nhoáng (nhanh), LLM BẮT BUỘC phải "rút gọn nghĩa" (Summarize/Paraphrase) câu dịch để số âm tiết dịch ra bằng đúng (hoặc nhỏ hơn) âm tiết gốc. Nếu người gốc nói chậm rãi, thong thả, LLM có thể dịch chi tiết hơn.

### 2.2 Nhận Diện Cảm Xúc (Emotion & Prosody Tagging)
- **Yêu cầu LLM:** Dựa vào ngữ cảnh câu nói, LLM phải gán thẻ cảm xúc vào kết quả JSON.
- Các thẻ cảm xúc tiêu chuẩn: `[NEUTRAL]`, `[HAPPY]`, `[ANGRY]`, `[SAD]`, `[WHISPERING]`, `[SHOUTING]`.
- Thẻ này sẽ được API Truyền qua (Pass-through) cho mô hình TTS ở Giai đoạn 2 để mô phỏng lại giọng điệu thực tế.

### 2.3 Giữ Nguyên Cấu Trúc JSON (Strict JSON Formatting)
- **Yêu cầu LLM:** Trả về ĐÚNG một object JSON `{"segments": [...]}`, mỗi phần tử CHỈ gồm schema TỐI THIỂU `{id, translated_text, emotion}`. Tuyệt đối không thêm văn bản bình luận thừa (Ví dụ: "Dưới đây là kết quả của bạn..."), không bọc ```` ```json ````.
- **Ghép mốc thời gian phía Worker (Zero-Trust với LLM cục bộ):** bản OpenAI cũ lấy hình dạng output từ `response_format` phía server; Qwen prompt-only KHÔNG có cơ chế đó nên Worker phải nhúng schema rõ ràng vào prompt VÀ **không tin** mô hình 4B echo lại `start`/`end`/`duration`/`speaker_id` — một mô hình nhỏ dễ renumber `id` hoặc làm hỏng số thực, đặt sai vị trí clip lồng tiếng (start/end là trọng yếu cho bước mix). Do đó mô hình chỉ điền `id` (để ghép) + `translated_text` + `emotion`; Worker GHÉP LẠI mọi mốc thời gian từ dữ liệu đầu vào ĐÃ TIN CẬY. Bất biến bắt buộc: tập `id` đầu ra phải TRÙNG KHỚP tập `id` đầu vào (không thêm/bớt/renumber), sai sẽ kích hoạt Auto-Retry.

### 2.4 Tự Chấm Điểm & Tự Chỉnh Sửa (Self-Review — Đợt 8)

Sau khi vòng dịch ban đầu (retry-max-3) cho ra bản HỢP LỆ, Qwen chạy một **vòng tự-soi (self-review)** để tự chấm điểm + viết gọn lại những câu **lệch nhịp lip-sync** của CHÍNH nó. Đây là *self-refine/reflection loop* — KHÁC với Auto-Retry (Retry chỉ kích hoạt khi output HỎNG CẤU TRÚC; self-review nhắm CHẤT LƯỢNG pacing của bản đã hợp lệ). Cơ chế tuân **No-Fake-Success** bằng cách tách **HAI tín hiệu**:

- **(A) Cổng KHÁCH QUAN — đếm âm tiết (`_pacing_penalty`).** Đây là tín hiệu **DUY NHẤT** quyết định: (i) câu nào cần sửa (`penalty > 2`) và (ii) có chấp nhận bản viết lại không (**chỉ khi penalty giảm NGẶT** — luật *"never-worse"*). TRÀN âm tiết bị phạt **gấp đôi** HỤT (§2.1: audio tràn khỏi cảnh là lỗi nặng nhất). "Cải thiện" luôn là một phép **giảm số nguyên ĐO ĐƯỢC**, không phải lời tự khen của mô hình.
- **(B) Điểm tự-chấm 1-5 của mô hình.** Qwen tự chấm fidelity/pacing/emotion và đề xuất `revised_text`. Điểm này **CHỈ ADVISORY** — định hướng bản viết lại, **KHÔNG BAO GIỜ** là cổng chấp nhận, và **KHÔNG được log** (Zero-Logging §3.1).

**Bất biến bắt buộc:**
- **FAIL-OPEN:** khác vòng dịch ban đầu (fail-CLOSED/raise), self-review là enhancement tùy chọn — mọi lỗi (JSON review hỏng, v.v.) đều **giữ bản dịch baseline đã hợp lệ**, KHÔNG đánh sập cả lô.
- **Bounded:** trần cứng `QWEN_MAX_REVIEW_ROUNDS` (mặc định 2) + thoát sớm khi một vòng không còn cải thiện NGẶT nào.
- **Zero-Trust:** vòng review chỉ được đổi `translated_text` + `emotion`; mọi mốc thời gian/`id`/`speaker_id` GHÉP LẠI từ đầu vào tin cậy (echo bậy bị vứt). Hợp đồng 9 trường + ID-parity giữ nguyên như §2.3.
- **Chống cắt cụt:** từ chối bản quá ngắn (sàn tỷ lệ `0.5×` **+** sàn tuyệt đối 2 âm tiết) để cổng "giảm penalty" không biến thành động cơ rút gọn mất nghĩa.
- **Chỉ ngôn ngữ ĐO ĐƯỢC:** self-review **bỏ qua sạch** khi source HOẶC target là CJK/non-Latin (`count_syllables` suy biến về 1 → không có tín hiệu pacing thật). `original_syllables` đếm ở phía **SOURCE** nên phải kiểm **CẢ HAI** đầu (lỗ hổng cắt-cụt-CJK-source do kiểm toán đối kháng phát hiện).

**Phạm vi trung thực:** self-review **chỉ tự-sửa trục PACING** (lip-sync). Bản viết lại cải thiện **nghĩa/emotion mà KHÔNG đổi số âm tiết** thì KHÔNG được tự-áp trên CPU (không có cổng khách quan → tránh fake-success) — đó là `residual_hardware` (dùng self-score làm cổng phụ trên GPU + LLM-as-judge). Kill-switch: `QWEN_SELF_REVIEW=0` tắt hoàn toàn.

---

## 3. Tiêu Chuẩn Viết Bình Luận & Log Khi Code Prompt LLM

Trong quá trình lập trình viên sử dụng các công cụ LLM Code (Cursor, Copilot) để viết hàm `translate_subtitles()`, bắt buộc tuân thủ các quy tắc sau:

### 3.1 Cấm Rò Rỉ Kịch Bản (Zero-Logging & Zero-Egress Policy)
- **Yêu cầu (Log):** Không được viết `console.log(translatedText)` hoặc `print(response.json())` trong mã nguồn. Cũng KHÔNG được log `str(exception)` của bước parse — message của `json.JSONDecodeError`/`pydantic.ValidationError` có thể nhúng NGUYÊN đoạn output mô hình sinh ra (rò rỉ plaintext qua log). Chỉ log METADATA (đếm segment, số lần thử) + TÊN loại lỗi (`type(e).__name__`).
- **Yêu cầu (Egress) — MỞ RỘNG QUAN TRỌNG:** Zero-Logging không chỉ là chuyện `console.log`. **Gửi transcript/bản dịch tới bất kỳ API bên thứ ba (cloud) nào TỰ NÓ đã là hành vi rò rỉ (exfiltration)** — dù không có một dòng log nào. Đây chính là lý do BỎ OpenAI và chạy Qwen cục bộ (offline cưỡng chế: `TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1`, không phone-home).
- **Cách Log đúng:**
  ```python
  # [CHUẨN] Chỉ log metadata + TÊN loại lỗi — không plaintext.
  logger.info(f"Đã dịch 45 segment (lần 1).")
  logger.warning(f"Qwen output không hợp lệ (lần 2/3): JSONDecodeError")
  ```

### 3.2 Viết Bình Luận Code (Code Commenting) Giải Thích Prompt
Khi viết System Prompt gửi cho LLM, lập trình viên phải viết comment rõ TẠI SAO lại dùng tham số đó.
  ```javascript
  // SYSTEM PROMPT CONFIGURATION:
  // - Inject User Params: targetLanguage và translationStyle để cá nhân hóa kết quả.
  // - Cung cấp Duration (Thời lượng) và Original_Syllables để ép LLM phải căn chỉnh (Lip-sync).
  // - Nếu tốc độ nói cao, LLM phải tự động paraphrase (rút gọn từ).
  const systemPrompt = `
    You are a professional dubbing translator. 
    Target language: ${params.targetLanguage}.
    Style/Tone: ${params.translationStyle}.
    
    Rule 1 (LIP-SYNC): The original segment duration is ${segment.duration}s. 
    You MUST keep your translated syllable count strictly close to ${original_syllables} to match the pacing.
    If the text is too long, paraphrase it!
  `;
  ```

---

## 4. Bản Mẫu JSON (Expected Output Standard)

Phân biệt HAI hợp đồng khác nhau:

**(a) Cái mô hình Qwen PHẢI xuất ra** — schema tối thiểu, chỉ 3 trường/segment (Worker nhúng schema này vào prompt):

```json
{
  "segments": [
    { "id": 1, "translated_text": "Cút khỏi nhà tao!", "emotion": "SHOUTING" }
  ]
}
```

**(b) Cái `translate_segments()` TRẢ VỀ cho các bước hạ nguồn** (process_job/TTS) sau khi GHÉP bản dịch + emotion lên mốc thời gian tin cậy — hợp đồng 9 trường, KHÔNG đổi:

```json
{
  "segments": [
    {
      "id": 1,
      "start": 0.00,
      "end": 2.50,
      "duration": 2.50,
      "original_text": "Get out of my house!",
      "original_syllables": 5,
      "translated_text": "Cút khỏi nhà tao!",
      "emotion": "SHOUTING",
      "speaker_id": "SPEAKER_01"
    }
  ]
}
```

> Các trường `translated_syllables`/`pacing_status` đã BỎ ở P4 (hạ nguồn không đọc; enum cũ lệch shared-types). Mô hình vẫn phải TỰ căn số âm tiết theo chỉ thị LIP-SYNC trong prompt, chỉ không báo lại con số.

> **Ghi chú (Auto-Retry — điều kiện THỰC):** Cơ chế `Auto-Retry` (giới hạn 3 lần, sau đó **fail-closed** — KHÔNG bịa/không trả một phần, tiêu chí #1) kích hoạt khi output mô hình HỎNG VỀ CẤU TRÚC: JSON không parse được, không khớp schema tối thiểu (`pydantic.ValidationError`), hoặc SAI tập `id`/số lượng segment so với đầu vào. Nhãn `emotion` lệch enum KHÔNG kích hoạt retry: nó được quy an toàn về `NEUTRAL` (có đếm + log metadata) vì là gợi ý ngữ điệu thứ yếu mà TTS hạ nguồn đã map `unknown → NEUTRAL` — đánh sập cả lô vì một nhãn lệch sẽ hạ độ tin cậy với mô hình nhỏ mà gần như không thêm tính đúng đắn. Ràng buộc số âm tiết (pacing) là CHỈ THỊ trong prompt, KHÔNG phải điều kiện retry cứng ở bản này; guided-decoding cưỡng chế cấu trúc/số âm tiết (xgrammar/outlines) là lever chất lượng chỉ kiểm chứng trên GPU — `residual_hardware`.
