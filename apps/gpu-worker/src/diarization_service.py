"""M4-S7 — Diarization adapter (ADR 0003): WhisperX/pyannote khi phần cứng đủ, degraded
TRUNG THỰC khi không.

Tách adapter mỏng để phần LOGIC — gán nhãn speaker vào timestamp ASR theo overlap lớn nhất —
là hàm thuần Python, kiểm thử ĐẦY ĐỦ trên CPU dev box KHÔNG cần model/GPU. Suy luận thật
(pyannote 3.1 qua WhisperX) là residual gpu_acceptance: cần GPU 24 GB Volta+ (cc≥7), torch
fp16, và HF token đã chấp nhận điều khoản model — đó là bước tài khoản người dùng/deploy,
không phải giá trị secret ta tự nhập (ADR 0003 §"Ranh giới").

Nguyên tắc No-Fake-Success: khi engine/HF_TOKEN/CUDA thiếu, adapter NÓI RÕ degraded và
diarize() FAIL-CLOSED — không bao giờ giả vờ tách được nhiều giọng.
"""
import importlib.util
import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("omnivoice.diarization")
logger.setLevel(logging.WARNING)

SPEAKER_UNKNOWN = "SPEAKER_UNKNOWN"
MIN_GPU_MEMORY_BYTES = 24_000_000_000
PYANNOTE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"


def diarization_required() -> bool:
    value = os.environ.get("DIARIZATION_REQUIRED", "0").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise RuntimeError("DIARIZATION_REQUIRED must be a boolean value.")


def _whisperx_importable() -> bool:
    """True nếu whisperx (kéo theo pyannote) cài được. Tách thành hàm module-level để test
    monkeypatch cổng năng lực mà không cần cài thư viện nặng trên dev box."""
    return importlib.util.find_spec("whisperx") is not None


def _pyannote_checkpoint_context() -> Any:
    """Allow only the metadata classes used by the pinned official pyannote checkpoint.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``.  The official
    ``pyannote/segmentation-3.0`` checkpoint contains four non-tensor metadata classes, so
    pyannote-audio 3.4 cannot load it unless those exact globals are allowlisted.  Keep the
    exception local to model construction instead of disabling ``weights_only`` globally.

    A minimal test install may replace WhisperX with a fake without installing pyannote; in
    that case there is no real checkpoint to load and a no-op context preserves the seam.
    """
    try:
        import torch
        from pyannote.audio.core.task import Problem, Resolution, Specifications
    except (ImportError, AttributeError):
        return nullcontext()

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    torch_version = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    if safe_globals is None or torch_version is None:
        return nullcontext()
    return safe_globals([torch_version, Problem, Resolution, Specifications])


class DiarizationAdapter:
    """Gán nhãn speaker vào ASR segment. Ranh giới rõ:
      • assign_speakers — PURE, test được trên CPU (không model).
      • capability      — cổng degraded trung thực (engine/HF_TOKEN/CUDA).
      • diarize         — SEAM THẬT residual gpu_acceptance; fail-closed khi degraded.
    """

    def __init__(self) -> None:
        """Create a lazy, process-resident diarization adapter.

        The model is loaded on the first eligible Analyze call, then reused for every later
        call in this worker process. The gated-model token is removed from the worker
        environment as soon as that pipeline is resident.
        """
        self._pipeline: Any | None = None
        self._pipeline_load_failed = False

    @property
    def pipeline_loaded(self) -> bool:
        """Whether the real WhisperX pipeline is resident in this worker process."""
        return self._pipeline is not None

    def capability(self) -> Tuple[bool, str]:
        """(available, reason). Degraded TRUNG THỰC khi thiếu BẤT KỲ điều kiện nào (ADR 0003):
        engine (whisperx/pyannote) chưa cài, HF_TOKEN vắng (model pyannote gated), hoặc CUDA
        không sẵn. available=True chỉ mở cổng — suy luận thật vẫn residual gpu_acceptance."""
        if not _whisperx_importable():
            return (False, "WhisperX/pyannote chưa cài — diarization degraded (một giọng).")
        if self._pipeline is None and not os.environ.get("HF_TOKEN"):
            return (False, "Thiếu HF_TOKEN cho model pyannote gated — diarization degraded.")
        # torch cài sẵn (đường suy luận), nhưng chỉ .cuda mới phân biệt phần cứng thật.
        import torch

        if not torch.cuda.is_available():
            return (False, "CUDA không sẵn — diarization cần GPU; degraded (một giọng).")
        try:
            compute_major = int(torch.cuda.get_device_capability(0)[0])
            if compute_major < 7:
                return (
                    False,
                    "GPU cần compute capability >= 7.0 cho diarization production; degraded.",
                )
            total_memory = int(torch.cuda.get_device_properties(0).total_memory)
            if total_memory < MIN_GPU_MEMORY_BYTES:
                return (
                    False,
                    "GPU cần ít nhất 24 GB VRAM cho diarization production; degraded.",
                )
        except Exception:
            return (False, "Không đọc được năng lực GPU — diarization degraded.")
        if self._pipeline_load_failed:
            return (
                False,
                "Diarization pipeline nạp lỗi — degraded; cần khởi động lại worker.",
            )
        return (True, "")

    def health_status(self) -> Dict[str, Any]:
        """Return sanitized capability/residence metadata for the public health probe.

        No model id, filesystem path, exception text, or HF token is exposed. `available`
        describes whether a new Analyze can use diarization now; `pipeline_loaded` separately
        states whether the lazy model is already resident in VRAM.
        """
        available, reason = self.capability()
        return {
            "available": available,
            "mode": "full" if available else "degraded",
            "pipeline_loaded": self.pipeline_loaded,
            "reason": reason,
        }

    def load_model(self) -> None:
        """Prewarm the real pipeline so production readiness proves GPU residence."""

        available, reason = self.capability()
        if not available:
            raise RuntimeError(f"Diarization prewarm failed: {reason}")
        import whisperx

        token = os.environ.get("HF_TOKEN", "")
        self._get_pipeline(whisperx, "cuda", token)
        if not self.pipeline_loaded:
            raise RuntimeError("Diarization prewarm did not produce a resident pipeline.")

    def _get_pipeline(self, whisperx: Any, device: str, hf_token: str) -> Any:
        """Load WhisperX once, then reuse the process-resident pipeline."""
        if self._pipeline is not None:
            os.environ.pop("HF_TOKEN", None)
            return self._pipeline
        try:
            # WhisperX 3.6.x defines this class in `whisperx.diarize` and does not export
            # it from the package root. Retain the root lookup for older releases and the
            # lightweight CPU test seam, then use the version-pinned public submodule.
            pipeline_class = getattr(whisperx, "DiarizationPipeline", None)
            if pipeline_class is None:
                from whisperx.diarize import DiarizationPipeline

                pipeline_class = DiarizationPipeline
            with _pyannote_checkpoint_context():
                self._pipeline = pipeline_class(
                    model_name=PYANNOTE_PIPELINE_ID,
                    use_auth_token=hf_token,
                    device=device,
                )
        except Exception:
            # Do not retain exception text: it may contain a token, cache path, or model URL.
            # A failed heavyweight load remains degraded until process restart instead of
            # repeatedly spending network/GPU resources on every Analyze job.
            self._pipeline_load_failed = True
            raise RuntimeError("Diarization pipeline initialization failed.") from None
        if self._pipeline is not None:
            # Pyannote has resolved its gated checkpoints. Later inference is served by this
            # resident object and the pinned offline cache, so the token no longer belongs in
            # the long-lived worker environment.
            os.environ.pop("HF_TOKEN", None)
        return self._pipeline

    def assign_speakers(
        self, asr_segments: List[Dict[str, Any]], turns: List[Tuple[float, float, str]]
    ) -> List[Dict[str, Any]]:
        """PURE: gán mỗi ASR segment cho turn diarization CHỒNG LẤN NHIỀU NHẤT theo thời gian.

        `turns` = danh sách (start, end, speaker) từ pyannote. Trả về danh sách segment MỚI
        (bản sao) với trường `speaker` đã gán; segment KHÔNG chồng lấn turn nào -> SPEAKER_UNKNOWN
        (không gán bừa theo turn gần nhất — đó là bịa danh tính). KHÔNG sửa input tại chỗ."""
        out: List[Dict[str, Any]] = []
        for seg in asr_segments:
            s = float(seg.get("start", 0) or 0)
            e = float(seg.get("end", 0) or 0)
            overlap_by_speaker: Dict[str, float] = {}
            for (t_start, t_end, speaker) in turns:
                # Độ chồng lấn thời gian; âm/không có nghĩa là rời nhau -> bỏ qua.
                overlap = max(0.0, min(e, t_end) - max(s, t_start))
                if overlap > 0:
                    overlap_by_speaker[speaker] = (
                        overlap_by_speaker.get(speaker, 0.0) + overlap
                    )
            best_speaker = (
                max(overlap_by_speaker, key=overlap_by_speaker.get)
                if overlap_by_speaker
                else SPEAKER_UNKNOWN
            )
            new_seg = dict(seg)
            new_seg["speaker"] = best_speaker
            out.append(new_seg)
        return out

    def diarize(
        self, audio_path: str, asr_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """SEAM THẬT (residual gpu_acceptance): chạy pyannote qua WhisperX -> danh sách turn ->
        assign_speakers. CHỈ hợp lệ khi capability().available; degraded -> RuntimeError
        (fail-closed, KHÔNG chạy model rỗng rồi trả split bịa). Trên dev box (whisperx chưa
        cài) không bao giờ tới nhánh nạp model — cổng dưới đây chặn trước."""
        available, reason = self.capability()
        if not available:
            # Zero-Logging: reason chỉ mô tả điều kiện HW/engine, không nội dung người dùng.
            raise RuntimeError(f"diarization không khả dụng: {reason}")

        # Nạp LƯỜI — chỉ khi phần cứng đủ (không import ở cấp module để COLLECT test trên CPU
        # không kéo theo pyannote/torch nặng). Đây là đường production trên GPU box.
        import torch
        import whisperx  # noqa: F401

        hf_token = os.environ.get("HF_TOKEN")
        device = "cuda"
        diarize_pipeline = self._get_pipeline(whisperx, device, hf_token)
        diar_result = diarize_pipeline(audio_path)
        # pyannote trả DataFrame với cột start/end/speaker; đọc thành các turn thời gian.
        turns: List[Tuple[float, float, str]] = [
            (float(row["start"]), float(row["end"]), str(row["speaker"]))
            for _, row in diar_result.iterrows()
        ]
        return self.assign_speakers(asr_segments, turns)


diarization_service = DiarizationAdapter()
