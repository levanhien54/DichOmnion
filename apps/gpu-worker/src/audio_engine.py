import os
import tempfile
import logging
from typing import List, Dict, Any

from src.timecode import to_seconds as _tc_to_seconds

# Cố gắng import pydub, có thể yêu cầu ffmpeg trên máy
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

logger = logging.getLogger("omnivoice.audio_engine")
logger.setLevel(logging.WARNING)

class AudioEngine:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        # Thống kê lần mix gần nhất để báo cáo TRUNG THỰC (No-Fake-Success): đã căn
        # lip-sync (time-stretch) bao nhiêu clip, và cắt bớt bao nhiêu clip quá dài.
        self.last_mix_stats = {"clips": 0, "stretched": 0, "truncated": 0}

    @staticmethod
    def _to_seconds(value) -> float:
        """Chuẩn hóa mốc thời gian về giây (float) — ủy quyền cho src.timecode.

        Giữ nguyên API (mix_audio + test gọi qua đây) nhưng dùng CHUNG một hàm với
        translation_service để hành vi nhất quán tuyệt đối. Xem src/timecode.py (CC-1)."""
        return _tc_to_seconds(value)

    @staticmethod
    def _atempo_chain(speed: float) -> str:
        """Dựng chuỗi filter ffmpeg `atempo` cho hệ số tốc độ `speed`.

        atempo chỉ nhận mỗi bộ lọc trong [0.5, 2.0] (giữ NGUYÊN cao độ, không bị
        'giọng chuột'); hệ số ngoài khoảng được ghép chuỗi (nhân dồn). speed > 1 =>
        tăng tốc (rút ngắn); speed < 1 => chậm lại (kéo dài)."""
        factors = []
        s = speed
        while s > 2.0:
            factors.append(2.0)
            s /= 2.0
        while s < 0.5:
            factors.append(0.5)
            s /= 0.5
        factors.append(round(s, 4))
        return ",".join(f"atempo={f}" for f in factors)

    def _fit_to_duration(self, seg, target_ms: int, tolerance: float = 0.05):
        """Co/giãn một đoạn TTS về đúng độ dài `target_ms` của đoạn hình để căn
        lip-sync, GIỮ NGUYÊN cao độ bằng ffmpeg atempo.

        Trả về (đoạn_đã_căn, đã_căn_thật). Fail-safe: nếu thiếu ffmpeg/target không
        hợp lệ/đã đủ khớp thì trả (đoạn gốc, False) — KHÔNG giả vờ đã căn.
        Hệ số được kẹp trong [0.5, 2.0] để tránh phá chất tiếng; lệch quá lớn chỉ
        được căn một phần (caller báo trung thực phần dư)."""
        cur = len(seg)
        if target_ms <= 0 or cur <= 0:
            return seg, False
        if abs(cur - target_ms) <= tolerance * target_ms:
            return seg, False  # đã đủ khớp, không cần đụng vào

        speed = cur / target_ms
        speed = max(0.5, min(2.0, speed))  # kẹp để giữ độ dễ nghe

        import subprocess
        src_path = out_path = None
        try:
            src_fd, src_path = tempfile.mkstemp(suffix="_pre.wav", dir=self.temp_dir)
            os.close(src_fd)
            out_fd, out_path = tempfile.mkstemp(suffix="_fit.wav", dir=self.temp_dir)
            os.close(out_fd)
            seg.export(src_path, format="wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", src_path, "-filter:a", self._atempo_chain(speed), out_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            fitted = AudioSegment.from_wav(out_path)
            return fitted, True
        except Exception as e:
            logger.error(f"Căn lip-sync (atempo) lỗi: {type(e).__name__}. Giữ độ dài tự nhiên.")
            return seg, False
        finally:
            for p in (src_path, out_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def extract_instrumental(self, audio_path: str) -> tuple[str, bool]:
        """Sử dụng Demucs để tách vocal và lấy nhạc nền (instrumental).

        Trả về (đường_dẫn, đã_tách_thật). Nếu Demucs lỗi/thiếu, trả về
        (audio_path gốc, False) — KHÔNG giả vờ đã tách nền (No-Fake-Success):
        caller PHẢI báo trung thực rằng nhạc nền chưa được tách, vì khi đó bản
        mix vẫn còn giọng gốc nằm dưới bản lồng tiếng."""
        import subprocess
        logger.info(f"Đang bóc tách nhạc nền (Demucs) cho file: {audio_path}")
        out_dir = os.path.join(self.temp_dir, "demucs_out")
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Chạy demucs. Mặc định demucs dùng model htdemucs, tách ra 4 stems.
            # Lệnh: demucs -n htdemucs -o out_dir audio_path --two-stems vocals
            subprocess.run([
                "demucs", "-n", "htdemucs", "--two-stems", "vocals",
                "-o", out_dir, audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # File nhạc nền sẽ ở out_dir/htdemucs/{tên_file_gốc}/no_vocals.wav
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            instrumental_path = os.path.join(out_dir, "htdemucs", basename, "no_vocals.wav")

            if os.path.exists(instrumental_path):
                logger.info("Tách nhạc nền thành công bằng Demucs.")
                return instrumental_path, True
            logger.error("Demucs chạy xong nhưng không thấy file no_vocals — coi như CHƯA tách nền.")
            return audio_path, False
        except Exception as e:
            logger.error(f"Lỗi khi chạy Demucs (có thể chưa cài đặt?): {type(e).__name__}. CHƯA tách được nền.")
            return audio_path, False

    def mix_audio(self, original_audio_path: str, tts_clips: List[Dict[str, Any]], ducking_db: float = -10.0) -> str:
        """
        Trộn các file audio TTS vào audio gốc, thực hiện ducking (hạ âm lượng) 
        nền ở các thời điểm có TTS.
        
        tts_clips là danh sách các dict chứa:
        {
            "start": float (giây),
            "end": float (giây),
            "audio_path": str (đường dẫn tới file mp3/wav sinh ra bởi TTS)
        }
        """
        if not HAS_PYDUB:
            # Fail-closed: thiếu pydub thì KHÔNG thể trộn âm. Trả file im lặng rồi
            # báo "thành công" là fake-success — người dùng nhận bản lồng tiếng câm.
            raise RuntimeError(
                "pydub không khả dụng — không thể trộn âm thanh (mix_audio)."
            )

        try:
            # Load âm thanh gốc
            logger.info("Đang nạp file audio gốc để làm nhạc nền...")
            if original_audio_path.endswith(".wav"):
                bg_audio = AudioSegment.from_wav(original_audio_path)
            elif original_audio_path.endswith(".mp3"):
                bg_audio = AudioSegment.from_mp3(original_audio_path)
            else:
                bg_audio = AudioSegment.from_file(original_audio_path)
                
            # Duyệt qua từng TTS clip
            stretched_count = 0
            truncated_count = 0
            for clip in tts_clips:
                clip_path = clip.get("audio_path")
                # Phòng thủ CC-1: chấp nhận cả số lẫn timecode chuỗi ("HH:MM:SS").
                start_s = self._to_seconds(clip.get("start", 0))
                end_s = self._to_seconds(clip.get("end", 0))
                start_ms = int(start_s * 1000)

                if not clip_path or not os.path.exists(clip_path):
                    continue

                # Load TTS audio
                if clip_path.endswith(".mp3"):
                    tts_audio = AudioSegment.from_mp3(clip_path)
                else:
                    tts_audio = AudioSegment.from_file(clip_path)

                # Căn lip-sync: co/giãn TTS về đúng độ dài đoạn hình (end-start).
                # Bản cũ bỏ qua clip['end'] hoàn toàn -> tiếng lồng trôi khỏi khẩu hình
                # và tràn sang đoạn kế; giờ khớp mốc thời gian THẬT của video.
                target_ms = int((end_s - start_s) * 1000) if end_s and end_s > start_s else 0
                tts_audio, stretched = self._fit_to_duration(tts_audio, target_ms)
                if stretched:
                    stretched_count += 1

                # WPC-2/NFS-03: nếu clip VẪN dài hơn đoạn hình sau khi căn (bản dịch
                # quá dài, cần >2x tốc độ mới vừa nhưng đã bị kẹp ở 2.0 để giữ chất
                # tiếng), cắt về đúng target_ms + fade-out để KHÔNG tràn đè lên
                # segment kế tiếp (chồng hai giọng). Báo cáo trung thực số clip bị cắt.
                if target_ms > 0 and len(tts_audio) > target_ms:
                    tts_audio = tts_audio[:target_ms].fade_out(min(50, target_ms))
                    truncated_count += 1

                end_ms = start_ms + len(tts_audio)
                
                # Ducking: Cắt nhạc nền ra làm 3 phần (Trước TTS, Trong TTS, Sau TTS)
                before = bg_audio[:start_ms]
                during = bg_audio[start_ms:end_ms]
                after = bg_audio[end_ms:]
                
                # Giảm âm lượng phần nhạc nền trùng với TTS
                ducked_during = during + ducking_db
                
                # Nối nhạc nền lại
                bg_audio = before + ducked_during + after
                
                # Overlay TTS đè lên nhạc nền đã ducking
                # Vì độ dài bg_audio không đổi, ta có thể dùng overlay
                bg_audio = bg_audio.overlay(tts_audio, position=start_ms)
                
            # Ghi lại thống kê căn lip-sync để model_manager báo cáo trung thực.
            self.last_mix_stats = {
                "clips": len(tts_clips),
                "stretched": stretched_count,
                "truncated": truncated_count,
            }

            # Xuất file hoàn chỉnh
            fd, final_path = tempfile.mkstemp(suffix="_final.wav", dir=self.temp_dir)
            os.close(fd)
            
            logger.info(f"Đang xuất file mix cuối cùng ra: {final_path}")
            bg_audio.export(final_path, format="wav")

            return final_path

        except Exception as e:
            # Fail-closed: lỗi trộn âm phải nổ ra, không được che giấu bằng file rỗng.
            logger.error(f"Lỗi trong quá trình mixing: {type(e).__name__}")
            raise RuntimeError(f"Mix âm thanh thất bại: {e}") from e

    def extract_audio_from_video(self, video_path: str) -> str:
        """Sử dụng FFmpeg để bóc tách audio từ video (16kHz mono cho ASR)."""
        import subprocess
        fd, audio_path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        os.close(fd)

        logger.info("Đang bóc tách audio từ video...")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1", audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return audio_path
        except Exception as e:
            # Fail-closed: không tách được audio thì báo lỗi, không trả file câm.
            logger.error(f"Lỗi khi bóc audio bằng ffmpeg: {type(e).__name__}")
            raise RuntimeError(f"Bóc tách audio thất bại: {e}") from e

    # LƯU Ý: Việc ghép (mux) audio vào video KHÔNG chạy ở worker. Theo kiến trúc
    # bảo mật (client chỉ upload AUDIO, giữ video thô cục bộ), client tự mux bằng
    # ffmpeg sidecar trong Tauri (main.rs::mux_audio_to_video, fail-closed). Hàm
    # mux phía Python trước đây là mã CHẾT và còn trả về video gốc khi ffmpeg lỗi
    # (fake-success) nên đã được gỡ bỏ.

    def add_watermark(self, audio_path: str) -> tuple[str, bool]:
        """Gài watermark ẩn vào file âm thanh bằng Meta AudioSeal.

        Trả về (đường_dẫn, đã_gài_thật). Nếu AudioSeal lỗi/thiếu, trả về
        (audio_path gốc, False) — KHÔNG quảng cáo 'đã watermark' khi thực tế
        chưa gài (No-Fake-Success)."""
        try:
            import torch
            import torchaudio
            from audioseal import AudioSeal

            logger.info("Đang nạp mô hình AudioSeal để gài watermark...")
            model = AudioSeal.load_generator("audioseal_wm_16bits")
            
            # Đọc audio
            wav, sample_rate = torchaudio.load(audio_path)
            
            # Thêm chiều batch (batch_size, channels, length)
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)
                
            logger.info("Đang xử lý watermark (Watermarking)...")
            watermark = model.get_watermark(wav, sample_rate)
            
            # Trộn watermark vào audio (chúng ta có thể điều chỉnh alpha/độ lớn nếu cần, mặc định là +)
            watermarked_audio = wav + watermark
            
            # Xóa chiều batch
            watermarked_audio = watermarked_audio.squeeze(0)
            
            # Xuất ra file mới
            fd, wm_path = tempfile.mkstemp(suffix="_wm.wav", dir=self.temp_dir)
            os.close(fd)
            
            torchaudio.save(wm_path, watermarked_audio, sample_rate)
            logger.info(f"Đã gài watermark thành công: {wm_path}")
            return wm_path, True
        except Exception as e:
            logger.error(f"Lỗi khi gài watermark: {type(e).__name__}. CHƯA gài được watermark.")
            return audio_path, False

audio_engine = AudioEngine()
