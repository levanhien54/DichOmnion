"""Chuẩn hóa mốc thời gian về GIÂY (float) — dùng chung cho cả pipeline worker.

CC-1 (contract mismatch): client có thể gửi mốc thời gian ở dạng số (giây),
chuỗi số ("83.5"), hoặc timecode "HH:MM:SS" / "MM:SS" / "HH:MM:SS,mmm" (kiểu SRT).
Hai nơi trong worker tiêu thụ start/end:
  - translation_service: tính `duration = end - start` để căn lip-sync;
  - audio_engine.mix_audio: định vị clip bằng `int(start_s * 1000)`.
Cả hai trước đây dùng `float()` trần -> chuỗi "HH:MM:SS" ném ValueError rồi rơi về
fallback (pacing sai / bản mix hỏng). Gom về MỘT hàm để hành vi nhất quán, và để
client chỉ cần đối chiếu một hợp đồng duy nhất.

Không parse được -> 0.0 (KHÔNG đoán bừa). Client vẫn PHẢI gửi mốc thật; đây là
lớp phòng thủ, không phải chỗ để bịa dữ liệu.
"""


def to_seconds(value) -> float:
    """Đưa một mốc thời gian bất kỳ về giây (float). bool KHÔNG bị coi là 0/1 giây."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return 0.0
    s = str(value).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        pass
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return 0.0
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec
