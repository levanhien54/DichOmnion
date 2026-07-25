# Global AI Rules cho Dự án OmniVoice (DichOmnion)

## KIẾN TRÚC CLIENT-CENTRIC & ZERO-TRUST SECURITY (BẮT BUỘC TUÂN THỦ)

1. **Bảo mật Chống Thất thoát GPU Chuẩn Ngân hàng:**
   - **BẮT BUỘC** áp dụng mã hóa phi đối xứng (ECDSA) cho Payload và **3-Way Cross-Validation** (Kiểm tra chéo 3 chiều).
   - Thiết lập Kill Switch tài chính.
2. **Tính Bất Biến & Tự Phục Hồi (Reliability):**
   - API Queue phải chuẩn Idempotency. Tự động Re-queue nếu lỗi. Chống sập hệ thống (Graceful Degradation).
   - **Zero-Logging:** Cấm ghi log trần (plain text) các thông tin nhạy cảm của người dùng.
3. **Không Upload Video:** Video gốc LUÔN nằm ở máy Client.
4. **Luồng dữ liệu 2 Bước (Human-in-the-Loop):**
   - Bóc băng & Dịch -> Trả Text cho Client sửa -> Render Audio -> Trả về Client ghép.
5. **Chuẩn Kỹ Xảo Âm Thanh Thương Mại (Audio Engineering):**
   - Áp dụng Emotion Transfer, Time-Stretching, và Acoustic Matching.
   - **BẮT BUỘC:** Giữ lại track Nhạc nền/Tiếng động gốc (Instrumental), áp dụng Auto-Ducking và trộn (Mix) cùng giọng lồng tiếng mới trước khi trả kết quả để tránh làm mất âm thanh nguyên bản của video.
   - Duy trì tính nhất quán giọng nói qua Multi-Speaker Voice Mapping.
6. **Ép xung Phần cứng GPU:**
   - Sử dụng *CUDA Streams*, *Pinned Memory*, và *FlashAttention-2*.
   - Giữ toàn bộ Model thường trú trên VRAM.
7. **Tiêu chuẩn Kiểm thử Công nghiệp (Industrial Testing):**
   - TypeScript: BẮT BUỘC dùng Vitest và comment JSDoc.
   - Python: BẮT BUỘC dùng pytest, Type Hinting, Docstring và phải Mocking GPU khi test API để tiết kiệm chi phí.
