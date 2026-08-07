import asyncio
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import logging
from dataclasses import dataclass
from threading import Lock
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

DEMUCS_MODEL_ID = "hf://htdemucs"
DEMUCS_OUTPUT_DIR = "htdemucs"
AUDIOSEAL_MODEL_ID = "audioseal_wm_16bits"
AUDIOSEAL_DETECTOR_ID = "audioseal_detector_16bits"
AUDIOSEAL_SAMPLE_RATE = 16_000
AUDIOSEAL_DETECTION_THRESHOLD = 0.5
AUDIOSEAL_MESSAGE_ACCURACY_THRESHOLD = 0.8
AUDIOSEAL_REPO_ID = "facebook/audioseal"
AUDIOSEAL_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
AUDIOSEAL_GENERATOR_FILENAME = "generator_base.pth"
AUDIOSEAL_GENERATOR_SHA256 = (
    "7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748"
)
AUDIOSEAL_DETECTOR_FILENAME = "detector_base.pth"
AUDIOSEAL_DETECTOR_SHA256 = (
    "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9"
)
AUDIOSEAL_NBITS = 16
DEMUCS_PROBE_TIMEOUT_SECONDS = 30
ALIGNMENT_TOLERANCE_MS = 40
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 80


def _required_enhancement(name: str) -> bool:
    """Parse an enhancement policy without accepting ambiguous production values."""
    value = os.environ.get(name, "0").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise RuntimeError(f"{name} must be a boolean value.")


def audioseal_required() -> bool:
    return _required_enhancement("AUDIOSEAL_REQUIRED")


def demucs_required() -> bool:
    return _required_enhancement("DEMUCS_REQUIRED")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audioseal_checkpoint_paths(
    *, local_files_only: bool
) -> tuple[str, str]:
    """Resolve and authenticate the two immutable public AudioSeal checkpoints."""
    from huggingface_hub import hf_hub_download

    resolved: list[str] = []
    for filename, expected_sha256 in (
        (AUDIOSEAL_GENERATOR_FILENAME, AUDIOSEAL_GENERATOR_SHA256),
        (AUDIOSEAL_DETECTOR_FILENAME, AUDIOSEAL_DETECTOR_SHA256),
    ):
        path = Path(
            hf_hub_download(
                repo_id=AUDIOSEAL_REPO_ID,
                filename=filename,
                revision=AUDIOSEAL_REVISION,
                token=False,
                local_files_only=local_files_only,
            )
        )
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise RuntimeError("AudioSeal checkpoint checksum verification failed.")
        resolved.append(str(path))
    return resolved[0], resolved[1]


@dataclass(frozen=True)
class SourceSeparationResult:
    """Paths produced by one Demucs two-stem run.

    ``separated`` is true only when both stems exist. Callers must treat false as a
    real capability/runtime failure, not as proof that the mixed source is a vocal stem.
    ``job_dir`` is retained even for an incomplete run so the caller can clean it.
    """

    vocals_path: str
    instrumental_path: str
    job_dir: str | None
    separated: bool


class AudioEngine:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self._audioseal_generator = None
        self._audioseal_detector = None
        self._audioseal_no_compile = nullcontext
        self._audioseal_load_lock = Lock()
        self._audioseal_checked = False
        self._audioseal_available = False
        self._demucs_checked = False
        self._demucs_available = False
        # Thống kê lần mix gần nhất để báo cáo TRUNG THỰC (No-Fake-Success): đã căn
        # lip-sync (time-stretch) bao nhiêu clip, và cắt bớt bao nhiêu clip quá dài.
        self.last_mix_stats = {
            "clips": 0,
            "aligned": 0,
            "stretched": 0,
            "truncated": 0,
            "dropped_oor": 0,
            "invalid_timeline": 0,
            "unresolved_underfill": 0,
            "unresolved_overfill": 0,
            "max_abs_residual_ms": 0,
            "alignment_tolerance_ms": ALIGNMENT_TOLERANCE_MS,
        }

    @staticmethod
    def _to_seconds(value) -> float:
        """Chuẩn hóa mốc thời gian về giây (float) — ủy quyền cho src.timecode.

        Giữ nguyên API (mix_audio + test gọi qua đây) nhưng dùng CHUNG một hàm với
        translation_service để hành vi nhất quán tuyệt đối. Xem src/timecode.py (CC-1)."""
        return _tc_to_seconds(value)

    @staticmethod
    def _duck_background(
        background: "AudioSegment",
        speech_intervals: List[tuple[int, int]],
        ducking_db: float,
    ) -> "AudioSegment":
        """Apply one smooth ducking envelope to the background only.

        Nearby speech intervals are grouped when their attack/release windows would
        overlap. Holding the background at the ducked level through those short gaps
        avoids both compounded gain and audible pumping between adjacent sentences.
        """
        if not speech_intervals or ducking_db == 0.0 or len(background) == 0:
            return background

        duration_ms = len(background)
        intervals = sorted(
            (max(0, start_ms), min(duration_ms, end_ms))
            for start_ms, end_ms in speech_intervals
            if start_ms < duration_ms and end_ms > 0 and end_ms > start_ms
        )
        if not intervals:
            return background

        groups: List[List[int]] = []
        envelope_span_ms = DUCK_ATTACK_MS + DUCK_RELEASE_MS
        for start_ms, end_ms in intervals:
            if groups and start_ms - groups[-1][1] < envelope_span_ms:
                groups[-1][1] = max(groups[-1][1], end_ms)
            else:
                groups.append([start_ms, end_ms])

        parts = []
        cursor_ms = 0
        for start_ms, end_ms in groups:
            attack_start_ms = max(cursor_ms, start_ms - DUCK_ATTACK_MS)
            release_end_ms = min(duration_ms, end_ms + DUCK_RELEASE_MS)

            if cursor_ms < attack_start_ms:
                parts.append(background[cursor_ms:attack_start_ms])

            if attack_start_ms < start_ms:
                attack = background[attack_start_ms:start_ms]
                parts.append(
                    attack.fade(
                        from_gain=0.0,
                        to_gain=ducking_db,
                        start=0,
                        duration=len(attack),
                    )
                )

            if start_ms < end_ms:
                parts.append(background[start_ms:end_ms] + ducking_db)

            if end_ms < release_end_ms:
                release = background[end_ms:release_end_ms]
                parts.append(
                    release.fade(
                        from_gain=ducking_db,
                        to_gain=0.0,
                        start=0,
                        duration=len(release),
                    )
                )

            cursor_ms = release_end_ms

        if cursor_ms < duration_ms:
            parts.append(background[cursor_ms:])

        ducked = sum(parts, AudioSegment.empty())
        # Millisecond slicing should preserve duration, but keep the public
        # contract exact if a backend rounds a boundary differently.
        if len(ducked) > duration_ms:
            ducked = ducked[:duration_ms]
        elif len(ducked) < duration_ms:
            ducked += background[len(ducked):duration_ms]
        return ducked

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

        Trả về (đoạn_đã_căn, đã_căn_đạt_tolerance). Fail-safe: nếu thiếu ffmpeg/target không
        hợp lệ/đã đủ khớp thì trả (đoạn gốc, False) — KHÔNG giả vờ đã căn.
        Hệ số được kẹp trong [0.5, 2.0] để tránh phá chất tiếng; lệch quá lớn chỉ
        được căn một phần. Khi đó ffmpeg có thể đã chạy nhưng cờ vẫn là False để caller
        ghi nhận phần lệch chưa giải quyết thay vì báo căn thành công."""
        cur = len(seg)
        if target_ms <= 0 or cur <= 0:
            return seg, False
        tolerance_ms = min(
            ALIGNMENT_TOLERANCE_MS,
            max(1, int(math.ceil(tolerance * target_ms))),
        )
        if abs(cur - target_ms) <= tolerance_ms:
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
            return fitted, abs(len(fitted) - target_ms) <= tolerance_ms
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
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        job_dir = os.path.join(out_dir, DEMUCS_OUTPUT_DIR, basename)

        try:
            # The hf:// prefix forces Demucs 4.1 through its Hugging Face loader. A bare
            # htdemucs name catches Hub failures and falls back to the legacy AWS repository,
            # which would bypass the serving process's HF_HUB_OFFLINE policy.
            #
            # -d cuda: ép Demucs tách nền TRÊN GPU. Không ép thì Demucs tự dò và có thể rơi
            #   về CPU -> tách một clip mất hàng phút, dễ đụng trần MAX_RENDER_MS của Trạm 3
            #   rồi bị quarantine oan (worker "chậm bất thường"). Worker vốn fail-closed nếu
            #   không có CUDA (Qwen không nạp) nên mặc định 'cuda' là nhất quán; DEMUCS_DEVICE
            #   cho phép ép 'cpu' khi cần.
            # --segment 7: chặn TRẦN VRAM của Demucs. htdemucs xử theo cửa sổ; cửa sổ càng dài,
            #   đỉnh VRAM càng cao. Trên một GPU 24GB đã cõng Whisper + Qwen 4B thường trú, một
            #   đỉnh Demucs không giới hạn có thể OOM. 7s nằm trong giới hạn model htdemucs (~7.8s)
            #   và giữ đỉnh bộ nhớ ổn định. DEMUCS_SEGMENT tinh chỉnh nếu cần.
            device = os.environ.get("DEMUCS_DEVICE", "cuda")
            segment = os.environ.get("DEMUCS_SEGMENT", "7")
            subprocess.run([
                "demucs", "-n", DEMUCS_MODEL_ID, "--two-stems", "vocals",
                "-d", device, "--segment", segment,
                "-o", out_dir, audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # File nhạc nền sẽ ở out_dir/htdemucs/{tên_file_gốc}/no_vocals.wav
            instrumental_path = os.path.join(
                job_dir, "no_vocals.wav"
            )

            if os.path.exists(instrumental_path):
                logger.info("Tách nhạc nền thành công bằng Demucs.")
                return instrumental_path, True
            logger.error("Demucs chạy xong nhưng không thấy file no_vocals — coi như CHƯA tách nền.")
            shutil.rmtree(job_dir, ignore_errors=True)
            return audio_path, False

        except Exception as e:
            logger.error(f"Lỗi khi chạy Demucs (có thể chưa cài đặt?): {type(e).__name__}. CHƯA tách được nền.")
            # Demucs may leave a source-vocal stem after an interrupted run.
            # Remove only this deterministic per-input directory before degrading.
            shutil.rmtree(job_dir, ignore_errors=True)
            return audio_path, False

    def extract_analysis_stems(self, audio_path: str) -> SourceSeparationResult:
        """Run Demucs once and return both stems required by Analyze.

        Analyze must transcribe and diarize ``vocals_path`` rather than the mixed source.
        A successful CLI exit is insufficient: both ``vocals.wav`` and
        ``no_vocals.wav`` must exist before ``separated`` can be true. On any failure the
        original path is returned as a non-authoritative placeholder and callers must fail
        the Analyze job; they must not silently continue on mixed audio.
        """
        instrumental_path, instrumental_ready = self.extract_instrumental(audio_path)
        if not instrumental_ready:
            # extract_instrumental already removes its deterministic partial-output
            # directory before returning false; no incomplete stem is authoritative here.
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            expected_job_dir = os.path.join(
                self.temp_dir, "demucs_out", DEMUCS_OUTPUT_DIR, basename
            )
            partial_vocals = os.path.join(expected_job_dir, "vocals.wav")
            partial_instrumental = os.path.join(expected_job_dir, "no_vocals.wav")
            return SourceSeparationResult(
                vocals_path=partial_vocals if os.path.isfile(partial_vocals) else audio_path,
                instrumental_path=(
                    partial_instrumental
                    if os.path.isfile(partial_instrumental)
                    else audio_path
                ),
                job_dir=expected_job_dir if os.path.isdir(expected_job_dir) else None,
                separated=False,
            )

        job_dir = os.path.dirname(instrumental_path)
        vocals_path = os.path.join(job_dir, "vocals.wav")
        both_stems_ready = os.path.isfile(instrumental_path) and os.path.isfile(vocals_path)
        if not both_stems_ready:
            logger.error("Demucs output is incomplete; Analyze source separation failed.")
        return SourceSeparationResult(
            vocals_path=vocals_path if both_stems_ready else audio_path,
            instrumental_path=instrumental_path,
            job_dir=job_dir,
            separated=both_stems_ready,
        )

    # Default -5.0 khớp call-site production (model_manager.py truyền ducking_db=-5.0);
    # giữ tường minh để default hàm không lệch với hành-vi thực-tế.
    def mix_audio(self, original_audio_path: str, tts_clips: List[Dict[str, Any]], ducking_db: float = -5.0) -> str:
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
        final_path = None
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
            dropped_oor_count = 0
            aligned_count = 0
            invalid_timeline_count = 0
            unresolved_underfill_count = 0
            unresolved_overfill_count = 0
            max_abs_residual_ms = 0
            prepared_clips = []
            for clip in tts_clips:
                clip_path = clip.get("audio_path")
                # Phòng thủ CC-1: chấp nhận cả số lẫn timecode chuỗi ("HH:MM:SS").
                start_s = self._to_seconds(clip.get("start", 0))
                end_s = self._to_seconds(clip.get("end", 0))

                if not clip_path or not os.path.exists(clip_path):
                    continue

                # A negative/non-finite start or a non-positive segment duration is not
                # a meaningful placement. In particular, pydub interprets a negative
                # overlay position relative to the end of the destination, which would
                # silently put speech at the wrong point in the program.
                if (
                    not math.isfinite(start_s)
                    or not math.isfinite(end_s)
                    or start_s < 0
                    or end_s <= start_s
                ):
                    invalid_timeline_count += 1
                    logger.warning(
                        "Clip TTS có timeline không hợp lệ — bỏ qua (đã ghi nhận trung thực)."
                    )
                    continue
                start_ms = int(start_s * 1000)
                target_ms = int((end_s - start_s) * 1000)

                # NFS-MIX-OOR: pydub overlay(position=start_ms) ÂM THẦM bỏ clip khi
                # start_ms >= len(bg_audio) (mốc bắt đầu nằm ngoài đuôi nhạc nền) — clip
                # biến mất KHÔNG dấu vết nhưng model_manager vẫn báo "thành công" => đây là
                # No-Fake-Success (người dùng mất tiếng lồng của đoạn đó mà không hề được báo).
                # Mọi nhánh drop anh em (stretched/truncated/tts-fail/missing-translation) đều
                # có ghi chú trung thực; nhánh này cũng phải có. Phát hiện SỚM (trước khi nạp/
                # co-giãn để không đếm nhầm clip này vào stretched/truncated), bỏ SẠCH, đếm lại.
                # bg_audio giữ nguyên độ dài suốt vòng lặp (before+during+after và overlay đều
                # bảo toàn độ dài) nên so với len(bg_audio) là nhất quán mọi vòng.
                if start_ms >= len(bg_audio):
                    dropped_oor_count += 1
                    logger.warning(
                        "Clip TTS bắt đầu ngoài phạm vi nhạc nền — bỏ qua (đã ghi nhận trung thực)."
                    )
                    continue


                # overlay() preserves the background length and silently discards any
                # tail beyond it. Permit only the same 40 ms frame/rounding tolerance as
                # the alignment gate; a larger declared overrun is a lossy timeline and
                # must be visible to Render's fail-closed structured metric.
                end_overrun_ms = round(end_s * 1000 - len(bg_audio), 6)
                if end_overrun_ms > ALIGNMENT_TOLERANCE_MS:
                    invalid_timeline_count += 1
                    logger.warning(
                        "Clip TTS vượt quá cuối nhạc nền — bỏ qua (đã ghi nhận trung thực)."
                    )
                    continue

                # Load TTS audio
                if clip_path.endswith(".mp3"):
                    tts_audio = AudioSegment.from_mp3(clip_path)
                else:
                    tts_audio = AudioSegment.from_file(clip_path)

                # Căn lip-sync: co/giãn TTS về đúng độ dài đoạn hình (end-start).
                # Bản cũ bỏ qua clip['end'] hoàn toàn -> tiếng lồng trôi khỏi khẩu hình
                # và tràn sang đoạn kế; giờ khớp mốc thời gian THẬT của video.
                if target_ms <= 0:
                    invalid_timeline_count += 1
                    logger.warning(
                        "Clip TTS có timeline không hợp lệ — bỏ qua (đã ghi nhận trung thực)."
                    )
                    continue
                tts_audio, stretched = self._fit_to_duration(tts_audio, target_ms)
                residual_ms = len(tts_audio) - target_ms
                max_abs_residual_ms = max(max_abs_residual_ms, abs(residual_ms))
                if abs(residual_ms) <= ALIGNMENT_TOLERANCE_MS:
                    aligned_count += 1
                    if stretched:
                        stretched_count += 1
                elif residual_ms < 0:
                    unresolved_underfill_count += 1
                else:
                    unresolved_overfill_count += 1

                # WPC-2/NFS-03: nếu clip VẪN dài hơn đoạn hình sau khi căn (bản dịch
                # quá dài, cần >2x tốc độ mới vừa nhưng đã bị kẹp ở 2.0 để giữ chất
                # tiếng), cắt về đúng target_ms + fade-out để KHÔNG tràn đè lên
                # segment kế tiếp (chồng hai giọng). Báo cáo trung thực số clip bị cắt.
                if len(tts_audio) - target_ms > ALIGNMENT_TOLERANCE_MS:
                    tts_audio = tts_audio[:target_ms].fade_out(min(50, target_ms))
                    truncated_count += 1

                prepared_clips.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": start_ms + len(tts_audio),
                        "audio": tts_audio,
                    }
                )

            events = []
            for prepared in prepared_clips:
                events.append((prepared["start_ms"], 1))
                events.append((prepared["end_ms"], -1))
            active = 0
            max_concurrent = 0
            # End events sort before starts at the same timestamp, so adjacent
            # clips are not counted as overlapping.
            for _position, delta in sorted(events, key=lambda item: (item[0], item[1])):
                active += delta
                max_concurrent = max(max_concurrent, active)

            component_count = 1 + max_concurrent
            mix_headroom_db = -20.0 * math.log10(component_count) - 0.5
            recoverable_gain_db = -mix_headroom_db

            # Mix attenuated PCM32 signals so integer overlay cannot saturate and
            # low-level PCM16 detail is retained while headroom is reserved.
            bg_audio = bg_audio.set_sample_width(4) + mix_headroom_db
            bg_audio = self._duck_background(
                bg_audio,
                [
                    (prepared["start_ms"], prepared["end_ms"])
                    for prepared in prepared_clips
                ],
                ducking_db,
            )
            for prepared in prepared_clips:
                start_ms = prepared["start_ms"]
                tts_audio = prepared["audio"].set_sample_width(4) + mix_headroom_db
                bg_audio = bg_audio.overlay(tts_audio, position=start_ms)

            # Recover at most the reserved gain, while keeping one decibel of
            # sample-peak headroom for the AudioSeal watermark applied next.
            if math.isfinite(bg_audio.max_dBFS):
                restore_db = min(
                    recoverable_gain_db,
                    max(0.0, -1.0 - bg_audio.max_dBFS),
                )
                bg_audio += restore_db
            bg_audio = bg_audio.set_sample_width(2)

            # Ghi lại thống kê căn lip-sync để model_manager báo cáo trung thực.
            self.last_mix_stats = {
                "clips": len(tts_clips),
                "aligned": aligned_count,
                "stretched": stretched_count,
                "truncated": truncated_count,
                "dropped_oor": dropped_oor_count,
                "invalid_timeline": invalid_timeline_count,
                "unresolved_underfill": unresolved_underfill_count,
                "unresolved_overfill": unresolved_overfill_count,
                "max_abs_residual_ms": max_abs_residual_ms,
                "alignment_tolerance_ms": ALIGNMENT_TOLERANCE_MS,
                "max_concurrent": max_concurrent,
                "mix_headroom_db": round(mix_headroom_db, 3),
                "peak_dbfs": round(bg_audio.max_dBFS, 3),
            }

            # Xuất file hoàn chỉnh
            fd, final_path = tempfile.mkstemp(suffix="_final.wav", dir=self.temp_dir)
            os.close(fd)
            
            logger.info(f"Đang xuất file mix cuối cùng ra: {final_path}")
            bg_audio.export(final_path, format="wav")

            return final_path

        except Exception as e:
            if final_path:
                try:
                    os.remove(final_path)
                except OSError:
                    pass
            # Fail-closed: lỗi trộn âm phải nổ ra, không được che giấu bằng file rỗng.
            logger.error(f"Lỗi trong quá trình mixing: {type(e).__name__}")
            raise RuntimeError(f"Mix âm thanh thất bại: {e}") from e

    def extract_audio_from_video(self, video_path: str) -> str:
        """Sử dụng FFmpeg để bóc tách audio từ video (16kHz mono cho ASR)."""
        import subprocess
        audio_path = None
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
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            # Fail-closed: không tách được audio thì báo lỗi, không trả file câm.
            logger.error(f"Lỗi khi bóc audio bằng ffmpeg: {type(e).__name__}")
            raise RuntimeError(f"Bóc tách audio thất bại: {e}") from e

    # LƯU Ý: Việc ghép (mux) audio vào video KHÔNG chạy ở worker. Theo kiến trúc
    # bảo mật (client chỉ upload AUDIO, giữ video thô cục bộ), client tự mux bằng
    # ffmpeg sidecar trong Tauri (main.rs::mux_audio_to_video, fail-closed). Hàm
    # mux phía Python trước đây là mã CHẾT và còn trả về video gốc khi ffmpeg lỗi
    # (fake-success) nên đã được gỡ bỏ.

    def _get_audioseal_models(self):
        """Load and cache the AudioSeal generator and verification detector."""
        if (
            self._audioseal_generator is not None
            and self._audioseal_detector is not None
        ):
            return self._audioseal_generator, self._audioseal_detector

        try:
            with self._audioseal_load_lock:
                if (
                    self._audioseal_generator is None
                    or self._audioseal_detector is None
                ):
                    import torch
                    from audioseal import AudioSeal
                    from audioseal.libs.moshi.utils.compile import no_compile

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info("Loading process-resident AudioSeal models...")
                    generator_path, detector_path = resolve_audioseal_checkpoint_paths(
                        local_files_only=True
                    )
                    generator = AudioSeal.load_generator(
                        generator_path, nbits=AUDIOSEAL_NBITS, device=device
                    )
                    detector = AudioSeal.load_detector(
                        detector_path, nbits=AUDIOSEAL_NBITS, device=device
                    )
                    generator.eval()
                    detector.eval()

                    # Publish the pair atomically so a detector load failure never leaves
                    # a partially initialized watermark capability behind.
                    self._audioseal_generator = generator
                    self._audioseal_detector = detector
                    self._audioseal_no_compile = no_compile
                self._audioseal_checked = True
                self._audioseal_available = True
        except Exception:
            self._audioseal_checked = True
            self._audioseal_available = False
            raise

        return self._audioseal_generator, self._audioseal_detector

    @staticmethod
    def _probe_demucs_cli() -> bool:
        """Verify both the CLI entrypoint and cached model in offline subprocesses."""
        executable = shutil.which("demucs")
        if not executable:
            return False
        probe_environment = os.environ.copy()
        probe_environment["HF_HUB_OFFLINE"] = "1"
        probe_environment["TRANSFORMERS_OFFLINE"] = "1"
        probe_environment.pop("HF_TOKEN", None)
        model_probe = (
            "from demucs.pretrained import get_model; "
            f"model = get_model({DEMUCS_MODEL_ID!r}); "
            "assert model is not None"
        )
        try:
            subprocess.run(
                [executable, "--help"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DEMUCS_PROBE_TIMEOUT_SECONDS,
            )
            subprocess.run(
                [sys.executable, "-c", model_probe],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DEMUCS_PROBE_TIMEOUT_SECONDS,
                env=probe_environment,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    def prewarm_required_enhancements(self) -> dict[str, dict[str, Any]]:
        """Prewarm AudioSeal and verify the Demucs CLI plus offline checkpoint.

        Failures are retained only as sanitized capability flags. The caller decides
        whether an unavailable capability is fatal from the corresponding required flag.
        """
        if audioseal_required():
            try:
                generator, detector = self._get_audioseal_models()
                self._audioseal_checked = True
                self._audioseal_available = generator is not None and detector is not None
            except Exception:
                self._audioseal_checked = True
                self._audioseal_available = False
                logger.error("Required AudioSeal startup prewarm failed.")

        if demucs_required():
            self._demucs_checked = True
            self._demucs_available = self._probe_demucs_cli()
            if not self._demucs_available:
                logger.error("Required Demucs CLI startup probe failed.")

        return self.enhancement_health_status()

    def enhancement_health_status(self) -> dict[str, dict[str, Any]]:
        """Return public readiness metadata with fixed reasons and no model/path detail."""
        def capability(available: bool, checked: bool, required: bool) -> dict[str, Any]:
            return {
                "available": bool(available),
                "required": required,
                "reason": "" if available else ("unavailable" if checked else "not_checked"),
            }

        return {
            "audioseal": capability(
                self._audioseal_available,
                self._audioseal_checked,
                audioseal_required(),
            ),
            "demucs": capability(
                self._demucs_available,
                self._demucs_checked,
                demucs_required(),
            ),
        }

    def _get_audioseal_generator(self):
        return self._get_audioseal_models()[0]

    def _get_audioseal_detector(self):
        return self._get_audioseal_models()[1]

    def add_watermark(self, audio_path: str) -> tuple[str, bool]:
        """Gài watermark ẩn vào file âm thanh bằng Meta AudioSeal.

        Trả về (đường_dẫn, đã_gài_thật). Nếu AudioSeal lỗi/thiếu, trả về
        (audio_path gốc, False) — KHÔNG quảng cáo 'đã watermark' khi thực tế
        chưa gài (No-Fake-Success)."""
        wm_path = None
        try:
            import torch
            import torchaudio

            generator, detector = self._get_audioseal_models()
            wav, sample_rate = torchaudio.load(audio_path)
            if wav.dim() != 2 or wav.shape[0] < 1 or wav.shape[-1] < 1:
                raise ValueError("AudioSeal requires non-empty [channels, frames] audio.")
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError("AudioSeal requires a positive integer sample rate.")
            if not torch.isfinite(wav).all():
                raise ValueError("AudioSeal input contains non-finite samples.")

            try:
                model_device = next(generator.parameters()).device
            except (AttributeError, StopIteration):
                model_device = torch.device("cpu")

            source = wav.to(device=model_device, dtype=torch.float32)
            channels, source_frames = source.shape
            if sample_rate == AUDIOSEAL_SAMPLE_RATE:
                source_16khz = source
            else:
                source_16khz = torchaudio.functional.resample(
                    source, sample_rate, AUDIOSEAL_SAMPLE_RATE
                )

            # AudioSeal is mono. Treat source channels as independent batch items so
            # stereo remains stereo without passing a two-channel model input.
            model_input = source_16khz.unsqueeze(1)
            message = generator.random_message(1).to(model_device)
            message = message.repeat(channels, 1)

            logger.info("Applying and verifying the AudioSeal watermark...")
            with torch.inference_mode(), self._audioseal_no_compile():
                watermark_16khz = generator.get_watermark(
                    model_input,
                    sample_rate=AUDIOSEAL_SAMPLE_RATE,
                    message=message,
                )

            if watermark_16khz.shape != model_input.shape:
                raise RuntimeError("AudioSeal returned an unexpected watermark shape.")
            if not torch.isfinite(watermark_16khz).all():
                raise RuntimeError("AudioSeal returned non-finite watermark samples.")

            watermark = watermark_16khz.squeeze(1)
            if sample_rate != AUDIOSEAL_SAMPLE_RATE:
                watermark = torchaudio.functional.resample(
                    watermark, AUDIOSEAL_SAMPLE_RATE, sample_rate
                )
            if watermark.shape[-1] > source_frames:
                watermark = watermark[..., :source_frames]
            elif watermark.shape[-1] < source_frames:
                watermark = torch.nn.functional.pad(
                    watermark, (0, source_frames - watermark.shape[-1])
                )

            watermarked_audio = torch.clamp(source + watermark, -1.0, 1.0)
            fd, wm_path = tempfile.mkstemp(suffix="_wm.wav", dir=self.temp_dir)
            os.close(fd)
            torchaudio.save(
                wm_path,
                watermarked_audio.detach().cpu(),
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            # Verify the exact PCM16 artifact returned to the caller. Checking the
            # float tensor before serialization can miss quantization damage.
            serialized_audio, serialized_rate = torchaudio.load(wm_path)
            if serialized_rate != sample_rate:
                raise RuntimeError("AudioSeal output sample rate changed after save.")
            if serialized_audio.shape != wav.shape:
                raise RuntimeError("AudioSeal output shape changed after save.")
            if not torch.isfinite(serialized_audio).all():
                raise RuntimeError("AudioSeal output contains non-finite samples.")

            detector_source = serialized_audio.to(
                device=model_device, dtype=torch.float32
            )
            if sample_rate != AUDIOSEAL_SAMPLE_RATE:
                detector_source = torchaudio.functional.resample(
                    detector_source, sample_rate, AUDIOSEAL_SAMPLE_RATE
                )
            with torch.inference_mode(), self._audioseal_no_compile():
                detect_probability, detected_message = detector.detect_watermark(
                    detector_source.unsqueeze(1),
                    sample_rate=AUDIOSEAL_SAMPLE_RATE,
                    detection_threshold=AUDIOSEAL_DETECTION_THRESHOLD,
                )
            if detect_probability.numel() != channels:
                raise RuntimeError("AudioSeal detector returned an unexpected batch size.")
            if not torch.isfinite(detect_probability).all():
                raise RuntimeError("AudioSeal detector returned non-finite probabilities.")
            if not torch.all(
                detect_probability >= AUDIOSEAL_DETECTION_THRESHOLD
            ).item():
                raise RuntimeError("AudioSeal detector did not verify every channel.")
            if detected_message.shape != message.shape:
                raise RuntimeError("AudioSeal detector returned an unexpected message shape.")
            if not torch.isfinite(detected_message).all():
                raise RuntimeError("AudioSeal detector returned a non-finite message.")

            expected_bits = message >= 0.5
            detected_bits = detected_message.to(model_device) >= 0.5
            message_accuracy = (detected_bits == expected_bits).float().mean(dim=1)
            if not torch.all(
                message_accuracy >= AUDIOSEAL_MESSAGE_ACCURACY_THRESHOLD
            ).item():
                raise RuntimeError(
                    "AudioSeal detector did not recover the embedded message "
                    "from every channel."
                )
            logger.info("AudioSeal watermark applied and verified successfully.")
            return wm_path, True
        except Exception as e:
            if wm_path:
                try:
                    os.remove(wm_path)
                except OSError:
                    pass
            logger.error(f"Lỗi khi gài watermark: {type(e).__name__}. CHƯA gài được watermark.")
            return audio_path, False

    def sweep_stale_finals(self, ttl_s: float, now: float | None = None) -> int:
        """Đợt 17 F1 — dọn CƠ HỘI các file kết quả cuối (dubbed_audio) đã quá hạn.

        process_job (LRC1) CỐ Ý giữ đầu ra cuối (*_final.wav / *_wm.wav) để client tải
        về, nhưng KHÔNG có bước nào xóa nó sau đó -> mỗi job để lại một file trên đĩa
        VĨNH VIỄN: (1) phình đĩa tới khi worker chết (khả dụng, tiêu chí #6/#4), (2) audio
        LỒNG TIẾNG (nhạy cảm) nằm lại vô thời hạn (Zero-Logging #2). Client tải ngay sau
        khi job DONE, nên một final cũ hơn TTL (mặc định 1h) coi như đã tải xong / bị bỏ
        -> thu hồi an toàn. KHÔNG xóa-sau-khi-serve (sẽ phá retry tải lại) và KHÔNG chạy
        reaper nền nặng nề — chỉ quét cơ hội ở đầu mỗi job.

        Quét CHỈ hai hậu tố worker TỰ SINH bằng mkstemp (*_final.wav, *_wm.wav) trong
        temp_dir, nên KHÔNG BAO GIỜ đụng audio nguồn / temp trung gian (_pre.wav/_fit.wav)
        / file tiến trình khác. Best-effort tuyệt đối: nuốt mọi OSError (stat/xóa/liệt kê)
        để một bước dọn rác KHÔNG BAO GIỜ làm hỏng job đang chạy. Trả số file đã xóa
        (phục vụ test & quan sát). `now` cho phép test tiêm mốc thời gian tất định."""
        if now is None:
            now = time.time()
        removed = 0
        try:
            entries = os.listdir(self.temp_dir)
        except OSError:
            return 0
        for name in entries:
            if not (name.endswith("_final.wav") or name.endswith("_wm.wav")):
                continue
            path = os.path.join(self.temp_dir, name)
            try:
                if not os.path.isfile(path):
                    continue
                if now - os.path.getmtime(path) > ttl_s:
                    os.remove(path)
                    removed += 1
            except OSError:
                # File biến mất giữa chừng / đang bị khóa / thiếu quyền — bỏ qua.
                continue
        return removed

audio_engine = AudioEngine()


async def run_periodic_final_sweep(
    interval_s: float,
    ttl_s: float,
    *,
    sleep=None,
    engine=None,
) -> None:
    """M2-S5e — BACKSTOP nền ĐỊNH KỲ dọn đầu ra CUỐI quá hạn KHI worker RẢNH.

    sweep_stale_finals (Đợt 17 F1) chỉ chạy CƠ HỘI ở đầu MỖI job (process_job). Một worker
    render một cụm job rồi NGỒI IM sẽ giữ mọi *_final.wav/_wm.wav quá hạn tới job kế — có thể
    KHÔNG BAO GIỜ tới: audio LỒNG TIẾNG nhạy cảm nằm lại vô thời hạn (Zero-Logging #2) + đĩa
    phình dần (khả dụng). Task nền này quét theo chu kỳ cố định nên một worker RẢNH VẪN thu
    hồi đĩa + purge đầu ra nhạy cảm đúng hạn. BỔ SUNG (không thay) cho sweep per-job.

    Best-effort tuyệt đối: một lượt quét nổ KHÔNG được giết vòng lặp (nuốt Exception -> tick
    kế vẫn chạy), và task NUỐT CancelledError để thoát SẠCH khi lifespan hủy nó lúc shutdown
    (đây là task nền fire-and-forget ở đỉnh — hủy chỉ xảy ra khi server dừng có chủ đích).
    `sleep`/`engine` tiêm được để test tất định (không cần đồng hồ/GPU thật)."""
    _sleep = sleep or asyncio.sleep
    _engine = engine if engine is not None else audio_engine
    try:
        while True:
            await _sleep(interval_s)
            try:
                _engine.sweep_stale_finals(ttl_s)
            except Exception:
                # Best-effort: một lượt quét hỏng (OSError/…) KHÔNG được chặn tick kế.
                logger.warning("periodic final sweep tick failed; next tick retries")
    except asyncio.CancelledError:
        # Shutdown sạch — lifespan hủy task này khi server dừng.
        return
