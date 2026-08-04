# Kế hoạch Nâng cấp Tiếp theo (Future Upgrade Plan)

Dựa trên nền tảng kiến trúc V2 vững chắc hiện tại của DichOmnion (Zero-Trust, Zero-Logging, Client-Centric Audio Processing, GPU Residence), dưới đây là lộ trình 5 bước chiến lược để đưa dự án đạt tiêu chuẩn thương mại cao cấp (Enterprise/Hollywood-grade).

## 1. Tích hợp Speaker Diarization (Nhận diện đa người nói)
*   **Thực trạng:** `asr_service.py` hiện tại bóc băng nhưng chưa tự động phân biệt được ai đang nói (đang gán cứng `"speaker": "SPEAKER_UNKNOWN"`).
*   **Mục tiêu:** Tự động hóa quá trình chia thoại, giảm thiểu sự can thiệp thủ công (Human-in-the-loop) ở Client khi phải tự phân đoạn người nói.
*   **Giải pháp Kỹ thuật:** 
    *   Tích hợp mô hình phân cụm người nói (Diarization) như **WhisperX** hoặc **pyannote.audio** chạy ngay sau bước lấy Text của Whisper.
    *   Gắn nhãn tự động (`SPEAKER_01`, `SPEAKER_02`...) và đồng bộ hóa ánh xạ nhãn này lên Client UI.

## 2. Triển khai Guided Decoding (Structured Generation) cho Qwen
*   **Thực trạng:** LLM cục bộ (Qwen 4B) dịch và được yêu cầu sinh JSON. Mặc dù đã có retry 3 lần, cách sinh chữ thuần (Greedy) đôi khi vẫn làm hỏng cấu trúc JSON, tốn thời gian parse/retry.
*   **Mục tiêu:** Đạt tỷ lệ lỗi JSON (Parse Error) bằng 0%, tối đa hóa tốc độ suy luận.
*   **Giải pháp Kỹ thuật:**
    *   Tích hợp thư viện **Outlines** hoặc **XGrammar** vào `translation_service.py`.
    *   Các thư viện này can thiệp vào Logits Processor của LLM, ép hệ thống chỉ sinh ra các Token khớp 100% với Pydantic Schema.
    *   Kết quả: Bỏ qua hoàn toàn việc bắt lỗi và retry, tăng độ trễ (latency) cho từng chu kỳ dịch.

## 3. Voice Cloning Cao Cấp (Nâng cấp GPT-SoVITS)
*   **Thực trạng:** Đang dùng Edge-TTS (với giới hạn ít giọng điệu) và map giọng thông qua hệ thống phân giải giả lập Nam/Nữ.
*   **Mục tiêu:** Lồng tiếng giữ đúng **chất giọng gốc** của người trong video, tăng tính chân thực và cá nhân hóa.
*   **Giải pháp Kỹ thuật:**
    *   Mở khóa hạng mục *residual_hardware* bằng cách chính thức đưa **GPT-SoVITS** (Zero-shot / Few-shot Voice Cloning) vào pipeline sinh âm thanh.
    *   Trích xuất tự động 3-5 giây âm thanh gốc từ chính đoạn thoại của Diarization làm *prompt audio*.
    *   TTS sẽ được sinh ra dựa trên chất giọng của chính người đang nói thay vì giọng chuẩn mặc định.

## 4. Kiến trúc Điều phối Queue Phân tán (Scale-out Hướng Sản Xuất)
*   **Thực trạng:** Bảo vệ VRAM 1 Node GPU bằng `asyncio.Semaphore(1)` là hoàn hảo cho 1 máy, nhưng sẽ gây nghẽn cổ chai (bottleneck/timeout) nếu có hàng chục request cùng lúc dồn tới.
*   **Mục tiêu:** Mở rộng linh hoạt lên N máy GPU Worker mà không làm treo Gateway.
*   **Giải pháp Kỹ thuật:**
    *   Bổ sung một lớp Message Queue (như **Redis**, **RabbitMQ** hoặc **Cloudflare Queues**) ở giữa Gateway và Worker.
    *   Đổi từ kiến trúc đồng bộ (Gateway chờ HTTP kết quả) sang kiến trúc **Bất đồng bộ (Async)**.
    *   Gateway phát Job vào Queue -> Client nhận mã Tracking và gọi Polling / WebSockets -> GPU Worker rảnh sẽ chủ động bốc Job từ Queue, xử lý xong ném kết quả lên R2 presigned.

## 5. Nâng cấp Thuật toán Căn khớp Khẩu hình (Lip-sync Alignment)
*   **Thực trạng:** Hệ thống đang đếm số âm tiết (syllable count) để căn độ dài và dùng `time-stretch` cơ bản toàn bộ câu audio để co/giãn.
*   **Mục tiêu:** Đảm bảo tiếng khớp hoàn hảo với khẩu hình, không bị hiệu ứng méo tiếng (robotic) khi tua thanh.
*   **Giải pháp Kỹ thuật:**
    *   Sử dụng mô hình Forced Alignment (như **Wav2Vec2** hoặc **Gentle**) để xác định chính xác timestamp của từng âm vị (phoneme) ở cả giọng gốc và giọng TTS.
    *   Thực hiện thuật toán `Dynamic Time Warping` (DTW) ở tầng ghép mix (Pydub/Librosa): Chỉ co giãn những nguyên âm dài hoặc khoảng lặng tĩnh, giữ nguyên tốc độ nói của các phụ âm bật.

## 6. Tái cấu trúc Frontend React & Quản lý State
*   **Thực trạng:** Logic gọi API, trạng thái ứng dụng và giao diện UI đang tập trung quá nhiều vào file gốc `App.tsx`, gây khó khăn cho việc bảo trì và mở rộng.
*   **Mục tiêu:** Tách biệt rõ ràng Business Logic và View, tăng độ ổn định của giao diện khi tương tác.
*   **Giải pháp Kỹ thuật:**
    *   Chia nhỏ `App.tsx` thành các Component độc lập (VD: `VideoUploader`, `SubtitleEditor`, `ProcessingTracker`).
    *   Áp dụng **Zustand** hoặc **Redux Toolkit** để quản lý trạng thái toàn cục.
    *   Thiết lập cơ chế Cache cục bộ (IndexedDB/Local Storage): Lưu nháp tiến trình chỉnh sửa phụ đề phòng trường hợp rớt mạng đột ngột.

## 7. Thiết lập Tự động hóa CI/CD (Continuous Integration / Deployment)
*   **Thực trạng:** Việc test và deploy hiện đang thực hiện thủ công qua các script `pnpm run test` hoặc lệnh `wrangler deploy`.
*   **Mục tiêu:** Tự động hóa quá trình kiểm định chất lượng code và triển khai lên môi trường Production.
*   **Giải pháp Kỹ thuật:**
    *   **CI (Kiểm thử):** Tích hợp **GitHub Actions**. Mỗi khi có Pull Request, tự động chạy Vitest (cho Frontend/Gateway) và Pytest (mock CPU cho GPU Worker) để bắt lỗi (Linting/Testing).
    *   **CD (Triển khai):** Tạo luồng tự động build image Docker cho GPU Worker, và push code của Gateway lên mạng lưới Cloudflare ngay khi nhánh `main` được cập nhật.

## 8. Kiểm thử Tích hợp Toàn trình (End-to-End / E2E Testing)
*   **Thực trạng:** Thiếu kịch bản test tự động bao phủ toàn bộ luồng sống còn của ứng dụng (Từ Client -> Gateway -> Worker). Nguy cơ lỗi Regression cao khi hệ thống phức tạp lên.
*   **Mục tiêu:** Đảm bảo luồng (Happy Path) "Nhập Video -> Tách Audio -> Dịch -> Lồng Tiếng -> Mux Video" luôn hoạt động hoàn hảo trước mỗi đợt phát hành.
*   **Giải pháp Kỹ thuật:**
    *   Triển khai **Playwright** hoặc **Tauri WebDriver**.
    *   Viết kịch bản tự động mô phỏng thao tác của người dùng trên giao diện Tauri, gửi API qua Gateway và sử dụng môi trường Mock để giả lập phản hồi của GPU Worker.

## 9. Khai thác Tối đa Sức mạnh VRAM 24GB (Extreme 24GB Throughput)
*   **Thực trạng:** Cấu hình hiện tại nạp Qwen 4B và Whisper chiếm khoảng 10-12GB VRAM, nhưng `Semaphore(1)` ép hệ thống chạy nối đuôi. Hơn 12GB VRAM đang bị bỏ không. Nếu chạy Batching mà không kiểm soát, các mô hình sẽ "giành giật" VRAM dẫn đến sập (OOM).
*   **Mục tiêu:** Ép xung và nhồi nhét xử lý song song kịch kim vào 24GB (RTX 3090/4090). Giữ nguyên chất lượng (bfloat16), đảm bảo các mô hình hoạt động hòa bình ở mức tải 100%.
*   **Giải pháp Kỹ thuật:**
    *   **Phân lô VRAM Cứng (Hard Partitioning):** Quy hoạch khắt khe 24GB VRAM: Cấp 4GB cho Whisper (chạy Batched), 4GB cho TTS, và 16GB còn lại khóa cứng cho vLLM (Qwen). Các mô hình không được vượt rào, đảm bảo hệ thống không bao giờ crash vì OOM dù tải cao đến đâu.
    *   **Bơm Batch liên tục (Continuous Batching với vLLM):** Sử dụng 16GB đã cấp cho Qwen để mở rộng vùng nhớ `KV Cache` (PagedAttention). Gỡ bỏ `Semaphore(1)`, cho phép GPU dịch **hàng chục đoạn hội thoại cùng lúc**.
    *   **Whisper Batched Inference:** Cắt audio thành nhiều chunk và bơm vào Whisper dưới dạng Batch cực lớn (batch_size=32). Ép Cuda Cores chạy hết công suất, tăng tốc độ bóc băng lên x3 - x4 lần.
    *   **Kiến trúc Streaming Pipeline (Gối đầu):** Trong lúc Whisper đang bóc băng phút thứ 3, Qwen tiến hành dịch phút thứ 2, và TTS đọc phút thứ 1. Đảm bảo toàn bộ pipeline vận hành song song không độ trễ.
    *   **FlashAttention-2 & CUDA Graphs:** Kích hoạt ở tầng core để tăng tốc nhân ma trận và loại bỏ độ trễ do CPU điều phối.

## 10. Tối ưu hóa Máy chủ & Chi phí (Kiến trúc Hybrid Edge-Cloud)
*   **Thực trạng:** Dù có chia Microservices (như ý tưởng cũ), ta vẫn tốn tiền thuê GPU Cloud để chạy ASR và TTS, đồng thời tốn băng thông + lưu trữ (R2) để đẩy file Audio 30MB lên xuống. Việc quản lý 3 cụm Server riêng biệt cũng tạo ra một cơn ác mộng về vận hành (DevOps) cho một hệ thống đang phát triển.
*   **Mục tiêu:** Kéo chi phí vận hành phần ASR/TTS và Băng thông về đúng **$0**. Đơn giản hóa kiến trúc Cloud, đẩy bảo mật lên mức tuyệt đối (Zero-Data-Transfer).
*   **Giải pháp Kỹ thuật:**
    *   **Tận dụng Sức mạnh Thiết bị Người dùng (Edge AI):** Ứng dụng client đang được build bằng Tauri (Desktop App). Ta hoàn toàn có thể tích hợp **Whisper.cpp** (hoặc ONNX Runtime) trực tiếp vào Client. Máy tính/Laptop của người dùng sẽ **tự bóc băng** video của chính họ bằng CPU hoặc GPU nội bộ.
    *   **Zero-Bandwidth Translation:** Thay vì thiết kế luồng Upload Audio lên R2 phức tạp, Client giờ đây chỉ việc gửi một cục Text (vài KB) lên Gateway. Gánh nặng băng thông và chi phí lưu trữ S3/R2 bị triệt tiêu hoàn toàn. Cực kỳ bảo mật vì file âm thanh không bao giờ rời khỏi máy người dùng.
    *   **Cloud GPU Độc tôn cho LLM:** Máy chủ 24GB Cloud giờ đây được giải phóng hoàn toàn khỏi Whisper và Audio. 100% VRAM (24GB) được cống hiến cho vLLM (Qwen) để bơm Batch dịch thuật. Chi phí Server giảm thê thảm nhưng tốc độ tổng thể lại tăng vọt vì Cloud chỉ xử lý Text.
    *   **Edge TTS (Tùy chọn):** Nếu không dùng Zero-shot Voice Cloning quá nặng, khâu tổng hợp giọng nói cũng có thể gọi API miễn phí (Edge-TTS) ngay từ Client, Cloud không cần can thiệp. Mọi tác vụ nặng về Audio đều được "phi tập trung hóa" (Decentralized) về máy người dùng.
