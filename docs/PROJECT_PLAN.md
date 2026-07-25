# Bản Kế Hoạch Phát Triển Ứng Dụng Lồng Tiếng Tự Động (Commercial-Grade OmniVoice)

> **Tổng quan Dự án:** Ứng dụng lồng tiếng (dubbing) video tự động đạt chuẩn thương mại. Kết hợp kiến trúc **Client-Centric**, **Pipeline AI V2 (Hardcore Optimization)** và hệ thống **Bảo mật Đa tầng & Kiểm tra chéo (Anti-Compute Theft & 3-Way Cross-Validation)**, dự án không chỉ mang lại chất lượng lồng tiếng khớp miệng hoàn hảo mà còn là một pháo đài bất khả xâm phạm về bảo mật tài chính.

---

## 1. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống phânĐán 3 khối (Serverless + Scale-to-zero) để đưa chi phí duy trì về 0:
1. **Client Side (Tauri/Web):** Giao diện người dùng. Xử lý video, tách/ghép audio cục bộ.
2. **Serverless Gateway & Storage (Cloudflare):** Điều phối API, kiểm tra chéo (Cross-Validation), chống DDoS.
3. **Serverless GPU Worker (RunPod/Modal):** Pipeline AI tối ưu tốc độ bằng CUDA cấp thấp.

---

## 2. Chiến Lược Bảo Mật Đa Tầng & Kiểm Tra Chéo (Chuẩn Ngân Hàng)

Hệ thống áp dụng nguyên lý **Zero-Trust (Không tin tưởng bất cứ ai)** và mã hóa bất đối xứng để chống ăn cắp GPU (Compute Theft).

### Cơ chế 3-Way Cross-Validation (Kiểm tra chéo 3 chiều)
- **Trạm 1 (Gateway kiểm tra Client):** 
  - **ECDSA Signature:** Client tự tạo Private/Public Key. Mọi request phải được ký bằng Private Key. Kể cả Hacker trộm được JWT Token cũng không thể gửi lệnh vì thiếu Private Key.
  - Gateway giải mã chữ ký, kết hợp gọi API Cloudflare Turnstile để chặn Bot.
- **Trạm 2 (GPU Worker kiểm tra Gateway):** 
  - Worker KHÔNG mở IP Public. 
  - Chỉ chấp nhận chạy lệnh khi có JWT nội bộ do chính Gateway ký. Worker tải Public Key của Gateway về để đối chiếu. Trùng khớp 100% mới chạy GPU.
- **Trạm 3 (Gateway kiểm tra Worker - Anti-Fraud):** 
  - **Anomaly Detection:** Nếu Worker trả kết quả video 10 phút chỉ trong 1 giây (hoặc kẹt quá lâu), Gateway lập tức phát hiện gian lận/lỗi. Từ chối kết quả và kích hoạt lệnh xóa sổ (Terminate) Worker đó ngay lập tức.

### Công tắc Hủy Diệt (Financial Kill Switch)
- Script chạy độc lập giám sát API Billing. Nếu chi phí Server tăng bất thường (VD: vượt $5/giờ), tự động "rút phích cắm" toàn bộ cụm GPU.

---

## 3. Bảng Yêu Cầu Kỹ Thuật Chi Tiết (Detailed Functional Requirements)

### Khối 1: Client Application (Tauri / Web App)
- **Giao diện (Human-in-the-Loop & Multi-Speaker):** Trình soạn thảo Phụ đề và **Gán giọng Nhân vật (Voice Profile Mapping)**. Cho phép User ánh xạ (VD: `Speaker_01 -> Giọng Nam 1`, `Speaker_02 -> Giọng Nữ 1`).
- **Tiền xử lý & Hậu xử lý (FFmpeg Local Bắt buộc):** Trích xuất `WAV 16kHz Mono` và ghép vào video bằng `-c:v copy`. Tự động băm (Hash MD5).
- **Mật mã học:** Sinh cặp khóa ECDSA cục bộ, ký mọi Payload trước khi gửi đi.

### Khối 2: Serverless Gateway & Storage
- **Gateway Server (Cloudflare Workers/Vercel):** Phí cố định 0đ/tháng. Đóng vai trò là Tòa án xác thực (Xác thực chữ ký ECDSA, JWT, Turnstile).
- **R2 Storage & Dọn rác:** Presigned URL giới hạn `Max 30MB`, `audio/wav`. Xóa file input tức thì sau khi hoàn thành.

### Khối 3: Serverless GPU Worker - AI Pipeline V2
**Giai đoạn 1: Bóc băng & Dịch**
1. **Vocal Isolation:** Dùng **UVR (MDX23)** phân tách làm 2 Track: `Vocal.wav` (Giọng nói) và `Instrumental.wav` (Nhạc nền & SFX).
2. **ASR & Diarization:** Dùng **faster-whisper** + **WhisperX**.
3. **Length-Constrained Translation:** Dịch bằng LLM sai số thời lượng < 10%.

**Giai đoạn 2: Render Audio (Audio Engineering Chuẩn Điện Ảnh)**
4. **Voice Cloning & TTS (Đa Nhân Vật):** Sinh giọng tuân thủ biểu đồ *Voice Profile Mapping*. Bắt buộc có **Emotion & Prosody Transfer**.
5. **Post-Processing (Kỹ Xảo Âm Thanh BẮT BUỘC):** 
   - **Time-Stretching:** Ép thời gian (**Time-Stretching**) và Đồng bộ âm học (**Acoustic Matching**).
   - **Mix & Auto-Ducking:** Phải trộn giọng lồng tiếng mới đè lên track `Instrumental.wav` gốc. Áp dụng Auto-Ducking tự động hụp âm lượng nhạc nền khi có tiếng nói để không làm hỏng trải nghiệm âm thanh của video nguyên bản.

**Hardcore GPU 24GB Optimization:**
- **Model Residence:** Lưu trú toàn bộ mô hình trên VRAM. Không load/unload.
- Kỹ thuật cấp thấp: **Pinned Memory**, **CUDA Streams (Luồng chéo)**, và **FlashAttention-2** để đẩy tốc độ render video 10 phút xuống còn ~40 giây.

---

## 4. Chiến Lược Ổn Định & Riêng Tư (SaaS Commercial Rules)

1. **Idempotency & Tự Phục Hồi (Auto-Retry):** Mọi Job gửi đi phải là Idempotent (Chạy lại không bị lỗi). Nếu Server sập ngang, Gateway phải tự Re-queue sang máy khác. KHÔNG trừ tiền/credit của User nếu chưa trả mã 200 OK.
2. **Graceful Degradation (Suy thoái nhẹ):** Khi hệ thống quá tải (nhiều người dùng), API không được sập. Bắt buộc trả về trạng thái Xếp hàng (Queued) kèm ETA. Client UI hiển thị thanh tiến trình thay vì báo lỗi.
3. **Zero-Logging Privacy:** Tuyệt đối KHÔNG print/log Kịch bản dịch, Audio URL, Token, IP khách hàng dưới dạng Plain text ra Console Server. Bảo vệ 100% quyền riêng tư (GDPR).

---

## 5. Chỉ Số Yêu Cầu Đạt Được (KPIs)

| Chỉ số Target | Mô hình Truyền thống | Mô hình Client-Centric V2 (Banking Security) |
| :--- | :--- | :--- |
| **Bảo mật Hacker** | Dễ bị hack API chạy chùa | **Chặn 100% nhờ ECDSA & Cross-Validation** |
| **Sự Cố Quá Tải** | Sập toàn bộ Server | **Tự động xếp hàng & Re-queue an toàn** |
| **Băng thông mạng** | ~600 MB (Upload Video) | **~30 MB** (Chỉ Audio) |
| **Thời gian GPU (10m)**| ~3 phút (PyTorch chuẩn) | **~ 40 giây** (Nhờ CUDA Streams) |
| **CHI PHÍ / VIDEO** | ~1,000 VNĐ | **< 200 VNĐ** (Chạy siêu tốc) |
