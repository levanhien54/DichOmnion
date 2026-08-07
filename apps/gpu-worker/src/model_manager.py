import asyncio
import inspect
import math
import os
import re
import torch
import logging


# Tuân thủ nguyên tắc Zero-Logging: Tắt log các thông tin nhạy cảm
logging.getLogger("omnivoice.models").setLevel(logging.WARNING)
logger = logging.getLogger("omnivoice.models")


async def _run_blocking(function, /, *args, **kwargs):
    """Run synchronous I/O/GPU work without blocking FastAPI's event loop.

    ``asyncio.to_thread`` alone leaves its thread running when the request task is
    cancelled. Waiting for that thread before propagating cancellation is deliberate:
    the endpoint keeps holding the one-GPU semaphore and the pipeline keeps ownership of
    its temp files until the blocking operation has actually stopped.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Repeated cancellation must not release the GPU semaphore while the worker
        # thread is still using resident models or job-scoped files.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and not task.cancelled():
            # Retrieve a late exception so asyncio does not emit "never retrieved";
            # cancellation remains the externally visible result.
            try:
                task.exception()
            except BaseException:
                pass
        raise


async def _translate_with_quality_policy(
    translation_service,
    *,
    segments,
    target_language,
    style,
    source_language,
    prompt_profile=None,
    quality_mode=None,
    semantic_judges=None,
    semantic_judge_passed=False,
):
    """Call old/new translation seams without weakening the real worker contract.

    Acceptance fakes from pre-quality builds expose only the original five
    arguments. We inspect the bound method and pass quality controls only when
    supported; the real ``TranslationService`` always receives them. No
    exception text or segment content is logged here.
    """

    function = translation_service.translate_segments
    kwargs = {
        "segments": segments,
        "target_language": target_language,
        "style": style,
        "source_language": source_language,
        "prompt_profile": prompt_profile,
        "quality_mode": quality_mode,
        "semantic_judges": semantic_judges,
        "semantic_judge_passed": semantic_judge_passed,
    }
    try:
        parameters = inspect.signature(function).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {
                key: value for key, value in kwargs.items() if key in parameters
            }
    except (TypeError, ValueError):
        # Unknown callable shape: retain the legacy argument set. This path is
        # for test seams only; the concrete worker method is introspectable.
        kwargs.pop("quality_mode", None)
        kwargs.pop("semantic_judges", None)
        kwargs.pop("semantic_judge_passed", None)
        kwargs.pop("prompt_profile", None)
    return await _run_blocking(function, **kwargs)


def _process_quality_mode() -> str:
    """Return the automatic-path policy without inheriting Analyze's observe mode."""

    return os.environ.get("TRANSLATION_PROCESS_QUALITY_MODE", "strict")


def _analyze_quality_metadata_enabled() -> bool:
    """Return the rollout flag for per-segment Analyze warnings.

    The current desktop schema and Review UI consume this additive field, so release
    builds enable it by default. Operators may still set ``0`` for a coordinated
    rollback to an older client.
    """

    return os.environ.get("ANALYZE_TRANSLATION_QUALITY_METADATA", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_QUALITY_DECISIONS = frozenset({"accept", "review", "reject"})
_QUALITY_SEMANTIC_STATES = frozenset({"passed", "failed", "not_run"})
_QUALITY_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _quality_rows_for_analyze(
    translation_service,
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate the in-memory quality projection before sealing it in Analyze.

    This boundary is intentionally stricter than a best-effort UI hint: if the
    rollout flag is enabled, a malformed/missing projection fails closed rather
    than emitting a partial artifact that could hide a bad translation.
    """

    raw = getattr(translation_service, "last_quality_reports", None)
    if not isinstance(raw, list):
        raise RuntimeError("translation_quality_metadata_missing")
    rows: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("translation_quality_metadata_invalid")
        identifier = item.get("id")
        score = item.get("score")
        decision = item.get("decision")
        semantic_state = item.get("semanticState")
        issue_codes = item.get("issueCodes")
        if (
            not isinstance(identifier, str)
            or not identifier
            or len(identifier) > 128
            or type(score) is not int
            or score < 0
            or score > 100
            or decision not in _QUALITY_DECISIONS
            or semantic_state not in _QUALITY_SEMANTIC_STATES
            or not isinstance(issue_codes, list)
            or len(issue_codes) > 16
            or any(
                not isinstance(code, str)
                or not _QUALITY_CODE_RE.fullmatch(code)
                for code in issue_codes
            )
            or len(set(issue_codes)) != len(issue_codes)
            or identifier in rows
        ):
            raise RuntimeError("translation_quality_metadata_invalid")
        rows[identifier] = {
            "score": score,
            "decision": decision,
            "semanticState": semantic_state,
            "issueCodes": list(issue_codes),
        }
    if expected_ids is not None and set(rows) != expected_ids:
        raise RuntimeError("translation_quality_metadata_segment_mismatch")
    return rows


def _confidence_from_logprob(avg_logprob) -> float:
    """avg_logprob (log-xác suất trung bình token của Whisper) -> điểm tin cậy [0,1].

    exp(avg_logprob) xấp xỉ xác suất trung bình; kẹp [0,1] vì hiếm khi avg_logprob > 0.
    Đây là tín hiệu THẬT của mô hình để người biên tập biết câu nào ASR không chắc —
    KHÔNG phải hằng số trang trí (No-Fake-Success). Giá trị rác -> 0.0 (an toàn, không nổ)."""
    try:
        conf = math.exp(avg_logprob)
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, conf)), 4)

# Bản đồ ngôn ngữ đích -> voice edge-tts mặc định (giọng nữ). Mặc định về giọng Việt.
VOICE_BY_LANG = {
    "vietnamese": "vi-VN-HoaiMyNeural",
    "english": "en-US-AriaNeural",
    "japanese": "ja-JP-NanamiNeural",
    "korean": "ko-KR-SunHiNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "french": "fr-FR-DeniseNeural",
    "spanish": "es-ES-ElviraNeural",
    "german": "de-DE-KatjaNeural",
}

# Giọng THẬT edge-tts hỗ trợ, tách theo giới tính cho từng ngôn ngữ đích. Dùng để
# hiện thực hóa "đa giọng": client gửi voice_id TRỪU TƯỢNG (nam_tram, nu_cao...) — nếu
# đưa thẳng chuỗi đó cho edge-tts sẽ lỗi -> segment câm. Ta ánh xạ theo giới tính sang
# giọng edge-tts CÓ THẬT của đúng ngôn ngữ đích.
VOICES_BY_LANG_GENDER = {
    "vietnamese": {"male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"},
    "english": {"male": "en-US-GuyNeural", "female": "en-US-AriaNeural"},
    "japanese": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "korean": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "chinese": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "french": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "spanish": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "german": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
}

# voice_id trừu tượng (từ UI VoiceMapper của client) -> giới tính. LƯU Ý TRUNG THỰC:
# edge-tts chỉ có ~1 nam + 1 nữ mỗi ngôn ngữ, nên 5 "nhân vật" này rút về 2 giọng
# (nam/nữ) mỗi ngôn ngữ. Phân biệt 5 giọng nhân vật riêng biệt (và nhân bản giọng thật)
# cần engine cục bộ GPT-SoVITS — xem residual_hardware.
VOICE_ID_GENDER = {
    "nam_tram": "male",
    "nam_tre": "male",
    "nu_cao": "female",
    "nu_truyen_cam": "female",
    "tre_em": "female",  # edge-tts không có giọng trẻ em -> dùng giọng nữ (gần nhất)
}

# Bản đồ cảm xúc (do LLM gán) -> ngữ điệu edge-tts. Giúp tag EMOTION thực sự
# tác động lên âm thanh thay vì chỉ là metadata trưng bày.
EMOTION_PROSODY = {
    "SHOUTING": {"rate": "+8%", "volume": "+30%", "pitch": "+15Hz"},
    "WHISPERING": {"rate": "-8%", "volume": "-40%", "pitch": "-10Hz"},
    "ANGRY": {"rate": "+10%", "volume": "+15%", "pitch": "+8Hz"},
    "SAD": {"rate": "-12%", "volume": "+0%", "pitch": "-12Hz"},
    "HAPPY": {"rate": "+5%", "volume": "+5%", "pitch": "+12Hz"},
    "NEUTRAL": {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
}


def _normalized_device(value) -> str:
    """Collapse arbitrary framework device objects into a public, bounded value."""
    try:
        raw = getattr(value, "type", value)
        name = str(raw).strip().lower()
    except Exception:
        return "unknown"
    if name == "cuda" or name.startswith("cuda:"):
        return "cuda"
    if name == "cpu" or name.startswith("cpu:"):
        return "cpu"
    if name == "meta" or name.startswith("meta:"):
        return "meta"
    return "unknown"


def _model_device(model, *, _seen=None, _depth: int = 0) -> str:
    """Inspect a model handle without returning class names, paths, or raw values."""
    if model is None or _depth > 2:
        return "unknown"
    seen = set() if _seen is None else _seen
    identity = id(model)
    if identity in seen:
        return "unknown"
    seen.add(identity)

    try:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            # torch.nn.Module.parameters() yields tensors, while pyannote.Pipeline uses
            # the same method name for a dict of tuning hyperparameters. Ignore values
            # that are not tensor-like instead of turning dictionary keys into a false
            # ``unknown`` device; Pipeline.to() exposes its actual runtime device below.
            devices = {
                _normalized_device(parameter.device)
                for parameter in parameters()
                if hasattr(parameter, "device")
            }
            if len(devices) == 1:
                return next(iter(devices))
            if len(devices) > 1:
                return "mixed"
    except Exception:
        return "unknown"

    for attribute in ("device", "_device"):
        try:
            device = _normalized_device(getattr(model, attribute, None))
        except Exception:
            continue
        if device != "unknown":
            return device

    # faster-whisper and WhisperX wrap the actual compute module one or two levels deep.
    for attribute in ("model", "_model", "pipeline", "audio_tokenizer"):
        try:
            nested = getattr(model, attribute, None)
        except Exception:
            continue
        device = _model_device(nested, _seen=seen, _depth=_depth + 1)
        if device != "unknown":
            return device
    return "unknown"


def _process_model_status(handle, *, loaded: bool, lifecycle: str = "process") -> dict:
    device = _model_device(handle) if loaded and handle is not None else "unknown"
    return {
        "loaded": bool(loaded and handle is not None),
        "process_resident": bool(loaded and handle is not None and device == "cuda"),
        "device": device,
        "lifecycle": lifecycle,
    }


class ModelManager:
    """
    Quản lý toàn bộ vòng đời của các mô hình AI trên VRAM.
    Tuân thủ nguyên tắc "Giữ toàn bộ Model thường trú trên VRAM".

    Whisper và Qwen cư trú trong worker này; pyannote và AudioSeal cũng được
    prewarm/cư trú khi policy yêu cầu. Demucs hiện chạy subprocess theo job, còn
    MOSS-TTS cư trú trong sidecar riêng. Telemetry phải phân biệt các ranh giới này
    thay vì dùng tensor giả để báo sẵn sàng.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.is_loaded = False

    def residency_status(self) -> dict:
        """Return sanitized process-residency evidence from the live model handles.

        ``is_loaded`` is only a startup latch. This method also proves that both core
        handles still exist and that their underlying runtime parameters/backend remain on
        CUDA. Optional components are reported independently; external/per-job engines are
        labelled explicitly instead of being advertised as process-resident.
        """
        whisper_service = self.models.get("whisper")
        translation_service = self.models.get("translation")
        diarization_service = self.models.get("diarization")
        audio_engine = self.models.get("audio_enhancements")

        whisper = _process_model_status(
            getattr(whisper_service, "model", None),
            loaded=getattr(whisper_service, "is_loaded", False) is True,
        )
        translation = _process_model_status(
            getattr(translation_service, "model", None),
            loaded=getattr(translation_service, "is_loaded", False) is True,
        )
        diarization = _process_model_status(
            getattr(diarization_service, "_pipeline", None),
            loaded=getattr(diarization_service, "pipeline_loaded", False) is True,
        )

        generator = getattr(audio_engine, "_audioseal_generator", None)
        detector = getattr(audio_engine, "_audioseal_detector", None)
        generator_status = _process_model_status(
            generator, loaded=generator is not None
        )
        detector_status = _process_model_status(detector, loaded=detector is not None)
        audioseal_devices = {
            generator_status["device"],
            detector_status["device"],
        }
        audioseal_loaded = generator_status["loaded"] and detector_status["loaded"]
        audioseal_device = (
            next(iter(audioseal_devices)) if len(audioseal_devices) == 1 else "mixed"
        )
        audioseal = {
            "loaded": audioseal_loaded,
            "process_resident": bool(audioseal_loaded and audioseal_device == "cuda"),
            "device": audioseal_device if audioseal_loaded else "unknown",
            "lifecycle": "process",
        }

        manager_device = _normalized_device(self.device)
        return {
            "manager_loaded": self.is_loaded is True,
            "device": manager_device,
            "core_models_resident": bool(
                self.is_loaded is True
                and manager_device == "cuda"
                and whisper["process_resident"]
                and translation["process_resident"]
            ),
            "components": {
                "whisper": whisper,
                "translation": translation,
                "diarization": diarization,
                "audioseal": audioseal,
                "demucs": {
                    "loaded": False,
                    "process_resident": False,
                    "device": "unknown",
                    "lifecycle": "per_job_subprocess",
                },
                "tts": {
                    "loaded": False,
                    "process_resident": False,
                    "device": "unknown",
                    "lifecycle": "external_sidecar",
                },
            },
        }

    def load_all_models(self):
        """Nạp các mô hình thường trú. Fail-closed nếu trọng số không nạp được."""
        if self.is_loaded:
            return

        # Fail before downloads/VRAM allocation on an undersized paid GPU Pod. The same
        # boundary is enforced by gpu_acceptance, but startup must protect itself too.
        from src.gpu_runtime import validate_production_gpu

        gpu_info = validate_production_gpu()
        self.device = gpu_info.device

        logger.info(f"Loading resident models to {self.device} VRAM...")

        # Build/load every advertised NeMo WFST grammar before accepting paid jobs.
        # This uses CPU and the persistent FAR cache, so a missing Pynini wheel or a
        # broken language grammar fails at startup instead of during the first Render.
        from src.text_preprocessing import (
            DEFAULT_PRELOAD_LANGUAGES,
            text_preprocessing_service,
        )
        from src.tts_service import local_tts_required

        text_preprocessing_service.validate_runtime_policy(
            local_tts_is_required=local_tts_required()
        )
        loaded_text_grammars = text_preprocessing_service.prewarm_configured()
        if text_preprocessing_service.required and set(loaded_text_grammars) != set(
            DEFAULT_PRELOAD_LANGUAGES
        ):
            raise RuntimeError("Required NeMo grammar coverage is incomplete.")
        self.models["text_preprocessing"] = text_preprocessing_service

        # Nạp Whisper (Bóc băng) THẬT và thường trú VRAM.
        # asr_service.load_model đã fail-closed (raise) nếu thiếu thư viện/trọng số.
        from src.asr_service import asr_service
        asr_service.load_model(model_size="base")

        if not asr_service.is_loaded:
            # Không bao giờ đánh dấu "đã nạp" khi mô hình lõi chưa thực sự sẵn sàng.
            raise RuntimeError("Whisper ASR không nạp được — worker không thể phục vụ job.")

        self.models["whisper"] = asr_service

        # Nạp mô hình DỊCH Qwen cục bộ và thường trú VRAM cạnh Whisper (tiêu chí #6 GPU
        # Model Residence). translation_service.load_model() đã fail-closed (raise) khi
        # thiếu transformers/CUDA/trọng số — GIỐNG Whisper, không bao giờ mock/giả. Trên
        # máy KHÔNG có GPU, load_all_models sẽ raise ở đây: đó là hành vi ĐÚNG cho prod
        # (worker không boot nếu không thể dịch cục bộ) và trung thực hơn bản OpenAI cũ
        # (boot "khỏe" nhưng chỉ rò rỉ transcript ra cloud lúc chạy). Chất lượng suy luận,
        # cư trú VRAM, độ trễ là residual_hardware — kiểm chứng trên GPU thật.
        from src.translation_service import translation_service
        translation_service.load_model()
        if not translation_service.is_loaded:
            raise RuntimeError("Qwen dịch không nạp được — worker không thể phục vụ job.")
        self.models["translation"] = translation_service

        # Production readiness must prove that gated diarization can load now, not discover
        # a missing checkpoint or OOM on the first paid Analyze request.
        from src.diarization_service import diarization_required, diarization_service

        if diarization_required():
            diarization_service.load_model()
            self.models["diarization"] = diarization_service

        # Production must discover missing AudioSeal weights or a broken Demucs CLI
        # before the first paid Render. Optional development environments retain the
        # same truthful degraded status without blocking the core worker.
        from src.audio_engine import (
            audio_engine,
            audioseal_required,
            demucs_required,
        )

        enhancement_status = audio_engine.prewarm_required_enhancements()
        if (
            audioseal_required()
            and enhancement_status["audioseal"]["available"] is not True
        ):
            raise RuntimeError("Required AudioSeal capability is unavailable.")
        if (
            demucs_required()
            and enhancement_status["demucs"]["available"] is not True
        ):
            raise RuntimeError("Required Demucs capability is unavailable.")
        self.models["audio_enhancements"] = audio_engine

        self.is_loaded = True
        logger.info("Resident models loaded to VRAM. Worker ready.")

    def _resolve_voice(self, target_language: str, speaker_id: str | None = None,
                       voice_map: dict | None = None) -> str:
        """Chọn giọng edge-tts THẬT cho một segment theo người nói.

        Ưu tiên ánh xạ ĐA GIỌNG người dùng gán ở client (speaker_id -> voice). Giá trị
        ánh xạ có thể là:
          (a) voice_id TRỪU TƯỢNG từ UI (nam_tram, nu_cao...) -> quy theo GIỚI TÍNH sang
              giọng edge-tts có thật của đúng ngôn ngữ đích;
          (b) một voice edge-tts CỤ THỂ (vi-VN-NamMinhNeural) -> dùng thẳng.
        Không có ánh xạ cho speaker này -> giọng mặc định (nữ) theo ngôn ngữ đích.

        Trước đây hàm bỏ qua speaker hoàn toàn -> mọi nhân vật dùng CHUNG một giọng
        (đa giọng chỉ là mô tả, không chạy). Và nếu đưa thẳng voice_id trừu tượng cho
        edge-tts thì segment sẽ câm -> đa giọng giả. Cả hai đều được sửa ở đây."""
        key = (target_language or "").strip().lower()
        lang_voices = VOICES_BY_LANG_GENDER.get(key, VOICES_BY_LANG_GENDER["vietnamese"])

        if voice_map and speaker_id:
            chosen = voice_map.get(speaker_id)
            # F10 defense-in-depth: chỉ nhận chuỗi. Cổng JobPayload._bound_voice_map đã loại
            # value phi-chuỗi (422), nhưng chốt cả ở đây để value bất thường (nếu tới từ đường
            # khác) DEGRADE về giọng mặc định thay vì ném TypeError/AttributeError giữa vòng TTS.
            if isinstance(chosen, str) and chosen:
                gender = VOICE_ID_GENDER.get(chosen)
                if gender:
                    # (a) voice_id trừu tượng -> giọng thật theo giới tính + ngôn ngữ.
                    return lang_voices.get(gender, lang_voices["female"])
                # (b) đã là voice edge-tts cụ thể -> tôn trọng lựa chọn.
                return chosen

        return VOICE_BY_LANG.get(key, VOICE_BY_LANG["vietnamese"])

    @staticmethod
    def _local_voice_context(
        catalog: dict, target_language: str, voice_map: dict
    ) -> tuple[set[str], str]:
        """Validate the live local catalog and all selected profiles before a job starts."""
        from src.tts_service import TTSProfileError, TTSUnavailableError

        if not isinstance(catalog, dict) or catalog.get("ready") is not True:
            raise TTSUnavailableError("local TTS is not ready")

        target = (target_language or "").strip().casefold()
        profiles = catalog.get("profiles")
        if not target or not isinstance(profiles, list):
            raise TTSProfileError("local TTS catalog is invalid")

        compatible: set[str] = set()
        for profile in profiles:
            if not isinstance(profile, dict) or not isinstance(profile.get("id"), str):
                raise TTSProfileError("local TTS catalog is invalid")
            languages = profile.get("languages")
            if not isinstance(languages, list) or not all(
                isinstance(language, str) for language in languages
            ):
                raise TTSProfileError("local TTS catalog is invalid")
            if any(language.strip().casefold() == target for language in languages):
                compatible.add(profile["id"])

        if not compatible:
            raise TTSProfileError("no local TTS profile supports the target language")
        if not isinstance(voice_map, dict):
            raise TTSProfileError("local TTS voice mapping is invalid")
        if any(
            not isinstance(profile_id, str) or profile_id not in compatible
            for profile_id in voice_map.values()
        ):
            raise TTSProfileError("local TTS voice mapping is invalid")

        default_profile_id = catalog.get("defaultProfileId")
        if default_profile_id not in compatible:
            default_profile_id = sorted(compatible)[0]
        return compatible, default_profile_id

    @staticmethod
    def _cleanup_temp(paths):
        """Dọn file tạm trung gian của MỘT job (best-effort).

        Chỉ xóa file tạm nội bộ do worker sinh ra (audio tải về, clip TTS, stem
        Demucs, bản mix trước watermark). KHÔNG bao giờ xóa đầu ra cuối
        (dubbed_audio) — caller đã discard nó khỏi tập này. Lỗi xóa được nuốt để
        không ảnh hưởng kết quả job."""
        import os
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    @staticmethod
    def _aggregate_tts_metrics(samples: list[dict]) -> dict | None:
        """Aggregate sanitized, job-local MOSS samples without retaining request data."""
        if not samples:
            return None
        latency_values = [sample["latency_ms"] for sample in samples]
        rtf_values = [sample["rtf"] for sample in samples]
        return {
            "engine": "moss-tts",
            "clip_count": len(samples),
            "latency_ms_total": sum(latency_values),
            "latency_ms_max": max(latency_values),
            "rtf_mean": round(math.fsum(rtf_values) / len(samples), 6),
            "rtf_max": max(rtf_values),
            "peak_allocated_vram_bytes_max": max(
                sample["peak_vram_bytes"] for sample in samples
            ),
        }

    @staticmethod
    def _alignment_metrics(mix_stats: dict, expected_clip_count: int) -> dict:
        """Build and validate the safe, structured M5 alignment contract.

        Older unit-test seams only expose the four original counters. They are mapped
        conservatively: a legacy truncation is unresolved overfill, while an otherwise
        clean legacy result is aligned. Production ``AudioEngine`` emits every field.
        """
        tolerance_ms = 40
        if not isinstance(mix_stats, dict):
            raise RuntimeError("Mix alignment metrics are invalid.")

        clips = mix_stats.get("clips")
        stretched = mix_stats.get("stretched", 0)
        truncated = mix_stats.get("truncated", 0)
        dropped = mix_stats.get("dropped_oor", 0)
        invalid_timeline = mix_stats.get("invalid_timeline", 0)
        underfill = mix_stats.get("unresolved_underfill", 0)
        overfill = mix_stats.get("unresolved_overfill", truncated)
        aligned = mix_stats.get(
            "aligned",
            clips - dropped - invalid_timeline - underfill - overfill
            if isinstance(clips, int) and not isinstance(clips, bool)
            else -1,
        )
        max_residual = mix_stats.get(
            "max_abs_residual_ms",
            tolerance_ms + 1 if underfill or overfill else 0,
        )
        reported_tolerance = mix_stats.get("alignment_tolerance_ms", tolerance_ms)

        integer_values = (
            clips,
            aligned,
            stretched,
            truncated,
            dropped,
            invalid_timeline,
            underfill,
            overfill,
            max_residual,
            reported_tolerance,
            expected_clip_count,
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values)
            or any(value < 0 for value in integer_values)
            or expected_clip_count <= 0
            or clips != expected_clip_count
            or reported_tolerance != tolerance_ms
            or aligned + dropped + invalid_timeline + underfill + overfill != clips
            or stretched > aligned
            or truncated > overfill
            or (underfill == 0 and overfill == 0 and max_residual > tolerance_ms)
            or ((underfill > 0 or overfill > 0) and max_residual <= tolerance_ms)
        ):
            raise RuntimeError("Mix alignment metrics are invalid.")

        return {
            "clip_count": clips,
            "aligned_clip_count": aligned,
            "stretched_clip_count": stretched,
            "truncated_clip_count": truncated,
            "dropped_out_of_range_clip_count": dropped,
            "invalid_timeline_clip_count": invalid_timeline,
            "unresolved_underfill_clip_count": underfill,
            "unresolved_overfill_clip_count": overfill,
            "max_abs_residual_ms": max_residual,
            "tolerance_ms": tolerance_ms,
        }

    async def _run_dub_pipeline(
        self,
        audio_url: str,
        config: dict,
        prepare_segments,
        *,
        strict_alignment: bool = False,
    ):
        """Xương sống DÙNG CHUNG của bước lồng tiếng (dub) cho cả process_job (một-lần) và
        render_job (bản người dùng đã DUYỆT).

        Pipeline THẬT (không sleep/giả lập):
        (tải audio) -> [chuẩn bị segment: CALLER quyết định] -> TTS -> Tách nền -> Mix -> Watermark

        Khác biệt chính giữa hai lối gọi là cách lấy `translated_segments`: process_job chạy
        ASR (nếu thiếu) + Qwen; render_job đọc NGUYÊN VĂN bản dịch trong manifest đã duyệt
        (KHÔNG Qwen, KHÔNG ASR). Mọi bước còn lại — tải audio fail-closed md5, TTS đa giọng theo
        cảm xúc, tách nền, mix, watermark, báo cáo TRUNG THỰC và dọn temp trên MỌI đường thoát —
        là MỘT.

        prepare_segments(local_audio_path) -> (translated_segments, prep_stages):
          • translated_segments: list dict snake_case (translated_text/original_text/speaker_id/
            emotion/start/end) cho vòng TTS;
          • prep_stages: các nhãn pipeline mà bước chuẩn bị THỰC SỰ chạy (vd ["Whisper",
            "Translation"] cho process, [] cho render) — để pipeline báo cáo TRUNG THỰC, không
            quảng cáo bước không chạy (No-Fake-Success).

        Đầu vào là AUDIO đã tách phía client (quyền riêng tư: worker không chạm video thô).
        Đầu ra là track lồng tiếng đã mix + watermark; client tự ghép (mux) vào video cục bộ.
        """
        if not self.is_loaded:
            raise RuntimeError("Models chưa được nạp lên VRAM!")

        from src.audio_service import audio_service, AudioIntegrityError
        from src.audio_engine import audio_engine, audioseal_required, demucs_required
        from src.tts_service import local_tts_required, tts_service

        # Ràng buộc TOÀN VẸN NỘI DUNG (fail-closed): object key R2 dùng chung cho mọi job/bản
        # build (presigned PUT chỉ trỏ 1 key), nên một job khác có thể ghi đè object trước khi
        # ta tải -> ta có thể nhận nhầm audio của tenant khác. md5 client KÝ kèm là "vé" ràng
        # buộc: thiếu vé -> từ chối (không xử lý audio không rõ nguồn); download_audio so md5
        # tải-về với vé và fail-closed khi lệch. Biến rò rỉ chéo tenant thành từ chối an toàn.
        expected_md5 = (config.get("audio_md5") or "").strip()
        if not expected_md5:
            # Đợt 33 CC33-01: thiếu md5 là lỗi XÁC ĐỊNH của payload (không tự khỏi khi retry)
            # -> AudioIntegrityError -> main.py trả 422 (terminal), không để Gateway retry.
            raise AudioIntegrityError("Thiếu md5 toàn vẹn audio — từ chối job (fail-closed).")

        # M5: production dubbing is local-only. Probe and validate the complete selection
        # before downloading tenant audio or starting any ASR/TTS/audio computation.
        target_language = config.get("target_language", "Vietnamese")
        voice_map = config.get("voice_map") or {}
        local_default_profile: str | None = None
        if local_tts_required():
            catalog = await tts_service.capability_catalog(probe=True)
            _, local_default_profile = self._local_voice_context(
                catalog, target_language, voice_map
            )

        import os

        # Đợt 15/LRC1 — dọn temp TRÊN MỌI ĐƯỜNG THOÁT (fail-closed cả khi lỗi): pipeline
        # có nhiều bước có thể NỔ giữa chừng (TTS synthesize, mix_audio fail-closed,
        # watermark...). Bản cũ đặt bước dọn ở CUỐI đường thành công nên khi một bước nổ,
        # cleanup KHÔNG chạy -> stem GIỌNG GỐC (vocals.wav), audio nguồn tải-về và clip TTS
        # rớt lại trong temp: vừa rò rỉ nội dung nhạy cảm (Zero-Logging #2) vừa phình đĩa
        # (caller ở main.py chỉ trả 500, KHÔNG dọn). Ta THEO DÕI mọi temp trung gian ngay
        # khi tạo ra, rồi dọn TẤT CẢ trong finally; chỉ giữ lại đầu ra cuối (dubbed_audio).
        cleanup_paths = set()
        demucs_job_dir = None
        final_audio_path = None
        gpu_memory_monitor = None
        gpu_memory_metrics = None
        mix_stats = None
        alignment_metrics = None
        try:
            # Đợt 17 F1 — dọn CƠ HỘI các file kết quả cuối (*_final.wav/*_wm.wav) đã quá hạn.
            # process_job GIỮ đầu ra cuối cho client tải nhưng KHÔNG có bước nào xóa nó sau đó
            # -> tích tụ VĨNH VIỄN (phình đĩa + audio lồng tiếng nhạy cảm nằm lại). Mỗi job mới
            # thu hồi đĩa từ các final cũ hơn TTL (client đã tải xong). Xem sweep_stale_finals;
            # bản thân sweep đã nuốt hết OSError nên không thể làm hỏng job — chỉ cần chắn lỗi
            # PHÂN TÍCH env (giá trị rác) để bước dọn rác không bao giờ chặn một job thật.
            try:
                _final_ttl_s = float(os.environ.get("WORKER_FINAL_TTL_S", "3600"))
            except (TypeError, ValueError):
                _final_ttl_s = 3600.0
            await _run_blocking(audio_engine.sweep_stale_finals, _final_ttl_s)

            # 1) Luôn tải audio nguồn: cần cho cả ASR (nếu thiếu segments) LẪN việc
            #    tách nhạc nền khi mix. Đây chính là chỗ bug cũ bỏ sót ở nhánh có segments.
            downloaded_path = await _run_blocking(
                audio_service.download_audio, audio_url, expected_md5
            )
            cleanup_paths.add(downloaded_path)
            local_audio_path = downloaded_path

            # Phòng thủ: nếu vô tình nhận container video, tách audio 16kHz mono.
            if local_audio_path.endswith((".mp4", ".mkv", ".mov")):
                local_audio_path = await _run_blocking(
                    audio_engine.extract_audio_from_video, local_audio_path
                )
                cleanup_paths.add(local_audio_path)

            # 2+3) Chuẩn bị segment ĐÃ DỊCH — chiến lược do CALLER quyết định (xem docstring):
            #   • process_job: ASR (nếu client chưa gửi segments) + Qwen dịch cục bộ;
            #   • render_job: đọc NGUYÊN VĂN bản dịch đã DUYỆT (KHÔNG ASR, KHÔNG Qwen).
            # prep_stages = các nhãn pipeline bước chuẩn bị THỰC SỰ chạy -> gắn nhãn TRUNG THỰC.
            translated_segments, prep_stages = await prepare_segments(local_audio_path)
            if not translated_segments:
                raise RuntimeError("Không có segment nào để xử lý (rỗng sau khi chuẩn bị).")

            # Acceptance deployments explicitly opt in to measuring the whole visible
            # GPU. PyTorch allocator metrics from the MOSS sidecar omit resident worker
            # models and the Demucs/AudioSeal processes.
            if config.get("_collect_gpu_memory_metrics") is True:
                from src.gpu_memory_monitor import GpuMemoryMonitor

                gpu_memory_monitor = GpuMemoryMonitor.from_env()
                gpu_memory_monitor.start()

            # 4) TTS: đọc ĐÚNG trường translated_text (bug cũ đọc "text" -> rỗng -> câm),
            #    chọn voice THEO TỪNG NGƯỜI NÓI (đa giọng thật) và điều biến ngữ điệu theo
            #    cảm xúc. voice_map = ánh xạ speaker_id -> voice do người dùng gán ở client.
            tts_clips = []
            tts_metric_samples = []
            used_voices = set()
            tts_attempted = 0  # số câu CÓ thoại đã thử sinh giọng (để báo trung thực nếu rớt)
            missing_translation = 0  # câu CÓ thoại gốc nhưng bản dịch RỖNG -> không thể lồng tiếng
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.begin_stage("TTS")
            for seg in translated_segments:
                spoken_text = seg.get("translated_text", "")
                if not spoken_text:
                    # I6/NFS: phân biệt "khoảng lặng thật" với "translator bỏ sót". Nếu câu có
                    # thoại GỐC (original_text) mà bản dịch rỗng thì đây là câu MẤT tiếng do
                    # thiếu bản dịch — ĐẾM để báo trung thực thay vì lặng lẽ bỏ. Bug cũ chỉ
                    # continue, và tts_attempted không tính các câu này nên note TTS bên dưới
                    # không bao giờ phản ánh chúng (mất tiếng âm thầm = fake success).
                    if (seg.get("original_text") or "").strip():
                        missing_translation += 1
                    continue
                tts_attempted += 1
                if local_default_profile is not None:
                    voice = voice_map.get(seg.get("speaker_id"), local_default_profile)
                else:
                    voice = self._resolve_voice(
                        target_language, seg.get("speaker_id"), voice_map
                    )
                prosody = EMOTION_PROSODY.get(seg.get("emotion", "NEUTRAL"), EMOTION_PROSODY["NEUTRAL"])
                emotion = seg.get("emotion", "NEUTRAL")
                duration_seconds = max(
                    0.001,
                    float(seg.get("end", 0)) - float(seg.get("start", 0)),
                )
                synth_kwargs = {
                    "voice": voice,
                    "rate": prosody["rate"],
                    "volume": prosody["volume"],
                    "pitch": prosody["pitch"],
                    "emotion": emotion,
                    "duration_seconds": duration_seconds,
                    "metrics_sink": tts_metric_samples,
                }
                if local_default_profile is not None:
                    synth_kwargs.update(
                        {"profile_id": voice, "language": target_language}
                    )
                audio_path = await tts_service.synthesize(spoken_text, **synth_kwargs)
                if audio_path:
                    used_voices.add(voice)
                    # Theo dõi để dọn ở finally kể cả khi bước sau nổ giữa chừng.
                    cleanup_paths.add(audio_path)
                    tts_clips.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "audio_path": audio_path,
                    })
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.end_stage()

            # 5) Mix KHÔNG điều kiện khi có TTS clips (bug cũ dùng 'local_audio_path' in locals()
            #    khiến nhánh có-sẵn-segments bị bỏ mix hoàn toàn -> mất tiếng).
            if not tts_clips:
                raise RuntimeError("TTS không tạo được clip nào — không thể render bản lồng tiếng.")

            if gpu_memory_monitor is not None:
                gpu_memory_monitor.begin_stage("Demucs")
            instrumental_path, separated = await _run_blocking(
                audio_engine.extract_instrumental, local_audio_path
            )
            cleanup_paths.add(instrumental_path)   # stem no_vocals của Demucs (== local nếu fail-closed)
            if separated:
                # Demucs xuất <temp>/demucs_out/htdemucs/<basename>/{no_vocals,vocals}.wav.
                # instrumental_path là no_vocals.wav; stem vocals.wav (GIỌNG GỐC đã bóc tách —
                # NHẠY CẢM) VÀ cả thư mục job phải bị dọn ở finally, nếu không giọng gốc rò rỉ
                # trong temp. Ghim demucs_job_dir NGAY tại đây để finally rmtree được kể cả khi
                # mix_audio/watermark phía dưới nổ.
                demucs_job_dir = os.path.dirname(instrumental_path)
                cleanup_paths.add(os.path.join(demucs_job_dir, "vocals.wav"))  # stem giọng gốc
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.end_stage()
            if demucs_required() and not separated:
                raise RuntimeError("Required Demucs stage did not produce an instrumental stem.")
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.begin_stage("Mix")
            mixed_audio_path = await _run_blocking(
                audio_engine.mix_audio,
                instrumental_path,
                tts_clips,
                ducking_db=-5.0,
            )
            cleanup_paths.add(mixed_audio_path)  # bản mix trước watermark (== final nếu không watermark)
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.end_stage()
            mix_stats = getattr(audio_engine, "last_mix_stats", None)
            alignment_metrics = self._alignment_metrics(mix_stats, len(tts_clips))
            if strict_alignment and (
                alignment_metrics["aligned_clip_count"]
                != alignment_metrics["clip_count"]
                or alignment_metrics["truncated_clip_count"] != 0
                or alignment_metrics["dropped_out_of_range_clip_count"] != 0
                or alignment_metrics["invalid_timeline_clip_count"] != 0
                or alignment_metrics["unresolved_underfill_clip_count"] != 0
                or alignment_metrics["unresolved_overfill_clip_count"] != 0
                or alignment_metrics["max_abs_residual_ms"]
                > alignment_metrics["tolerance_ms"]
            ):
                raise RuntimeError("Render alignment gate rejected the output.")
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.begin_stage("Watermark")
            final_audio_path, watermarked = await _run_blocking(
                audio_engine.add_watermark, mixed_audio_path
            )
            cleanup_paths.add(final_audio_path)
            if gpu_memory_monitor is not None:
                try:
                    gpu_memory_monitor.end_stage()
                except Exception:
                    # The output exists but the acceptance measurement failed. Mark it
                    # non-final so the shared finally block deletes it.
                    final_audio_path = None
                    raise
            if audioseal_required() and not watermarked:
                # The false contract normally returns mixed_audio_path. Clear the final
                # marker before raising so finally removes every sensitive intermediate.
                final_audio_path = None
                raise RuntimeError("Required AudioSeal stage did not verify the output.")

            if gpu_memory_monitor is not None:
                try:
                    gpu_memory_metrics = gpu_memory_monitor.finish()
                except Exception:
                    # A required measurement failure invalidates the otherwise successful
                    # output; let the existing cleanup path remove that output as well.
                    final_audio_path = None
                    raise

            # Xây danh sách pipeline TRUNG THỰC theo bước THỰC SỰ chạy — không quảng cáo
            # bước chưa chạy (No-Fake-Success). prep_stages đến từ bước chuẩn bị của CALLER:
            # process_job trả ["Whisper"?, "Translation"], render_job trả [] (không transcribe,
            # không dịch lại) — nên Render KHÔNG bao giờ liệt kê "Whisper"/"Translation".
            # TTS + Mix luôn chạy (mix_audio đã fail-closed).
            pipeline = list(prep_stages)
            pipeline.append("TTS")
            if separated:
                pipeline.append("Demucs")
            pipeline.append("Mix")
            if watermarked:
                pipeline.append("Watermark")

            # Cảnh báo trung thực khi bước tăng cường không khả dụng.
            notes = []
            if not separated:
                notes.append(
                    "Chưa tách được nhạc nền (Demucs không khả dụng) — giọng gốc có thể "
                    "còn nằm dưới bản lồng tiếng."
                )
            if not watermarked:
                notes.append("Chưa gài được watermark (AudioSeal không khả dụng).")

            # Báo cáo TRUNG THỰC việc căn lip-sync (time-stretch) — chỉ khi thực sự có clip
            # được co/giãn về đúng độ dài đoạn hình.
            if mix_stats.get("stretched"):
                notes.append(
                    f"Đã căn lip-sync (time-stretch) {mix_stats['stretched']}/{mix_stats['clips']} "
                    "clip về đúng độ dài đoạn hình."
                )
            # WPC-2/NFS-03: báo trung thực nếu phải cắt bớt clip quá dài (dịch dài hơn thời lượng gốc).
            if mix_stats.get("truncated"):
                notes.append(
                    f"Đã cắt bớt {mix_stats['truncated']}/{mix_stats['clips']} clip lồng tiếng quá dài "
                    "(bản dịch dài hơn thời lượng gốc) để không tràn sang đoạn kế."
                )
            # NFS-MIX-OOR: báo trung thực nếu có clip bị bỏ vì mốc bắt đầu nằm NGOÀI phạm vi
            # nhạc nền (pydub overlay lặng lẽ nuốt) — nếu không, người dùng mất tiếng lồng của
            # các đoạn đó mà pipeline vẫn báo "thành công" (No-Fake-Success).
            if mix_stats.get("dropped_oor"):
                notes.append(
                    f"Có {mix_stats['dropped_oor']}/{mix_stats['clips']} clip lồng tiếng bị bỏ vì "
                    "mốc bắt đầu nằm ngoài độ dài nhạc nền — bản lồng tiếng thiếu các đoạn này."
                )
            if mix_stats.get("invalid_timeline"):
                notes.append(
                    f"Có {mix_stats['invalid_timeline']}/{mix_stats['clips']} clip lồng tiếng bị bỏ "
                    "vì timeline không hợp lệ."
                )
            if mix_stats.get("unresolved_underfill"):
                notes.append(
                    f"Có {mix_stats['unresolved_underfill']}/{mix_stats['clips']} clip ngắn hơn "
                    "timeline quá tolerance sau khi căn lip-sync."
                )
            if mix_stats.get("unresolved_overfill"):
                notes.append(
                    f"Có {mix_stats['unresolved_overfill']}/{mix_stats['clips']} clip dài hơn "
                    "timeline quá tolerance sau khi căn lip-sync."
                )

            # WPC-1/NFS-02: nếu một số câu có thoại nhưng TTS không sinh được (voice server
            # lỗi/rớt), báo trung thực số câu thiếu tiếng thay vì lặng lẽ bỏ qua.
            tts_rendered = len(tts_clips)
            if tts_rendered < tts_attempted:
                notes.append(
                    f"Có {tts_attempted - tts_rendered}/{tts_attempted} câu không sinh được giọng "
                    "(TTS lỗi) — bản lồng tiếng thiếu các câu này."
                )

            # I6/NFS: câu CÓ thoại gốc nhưng bản dịch RỖNG (translator bỏ sót) — báo trung thực
            # số câu mất tiếng vì THIẾU BẢN DỊCH, tách bạch với lỗi TTS ở trên.
            if missing_translation:
                notes.append(
                    f"Có {missing_translation} câu có thoại gốc nhưng KHÔNG có bản dịch "
                    "(bị bỏ sót khi dịch) — bản lồng tiếng thiếu các câu này."
                )

            message = (
                "Đã tách nền, lồng tiếng và mix xong track âm thanh lồng tiếng."
                if separated
                else "Đã lồng tiếng và mix xong track âm thanh (chưa tách nhạc nền)."
            )

            return {
                "status": "success",
                "message": message,
                "device_used": self.device,
                "pipeline": pipeline,
                "separated": separated,
                "watermarked": watermarked,
                # Số giọng THỰC SỰ đã dùng — báo trung thực năng lực đa giọng thay vì
                # quảng cáo "multi-speaker" khi thực tế chỉ render một giọng.
                "distinct_voices": len(used_voices),
                "tts_metrics": self._aggregate_tts_metrics(tts_metric_samples),
                "alignment_metrics": alignment_metrics,
                **(
                    {"gpu_memory_metrics": gpu_memory_metrics}
                    if gpu_memory_metrics is not None
                    else {}
                ),
                "notes": notes,
                # Zero-Logging: KHÔNG trả translated_segments (chứa kịch bản GỐC + BẢN DỊCH
                # dạng plaintext). Client không dùng tới; nếu trả thì gateway sẽ ghi nó vào KV
                # 24h và lộ qua poll — rò rỉ nội dung nhạy cảm. Worker chỉ trả AUDIO đầu ra.
                "dubbed_audio": final_audio_path,
            }
        finally:
            if gpu_memory_monitor is not None:
                gpu_memory_monitor.close()
            # WPC-3 + Đợt 15/LRC1: dọn MỌI temp trung gian dù job THÀNH CÔNG hay NỔ giữa
            # chừng. Trên nhánh thành công, đầu ra cuối (final_audio_path) được loại khỏi tập
            # để client tải; trên nhánh lỗi final_audio_path=None nên KHÔNG loại gì -> dọn
            # sạch, không để stem giọng gốc (vocals.wav), audio nguồn hay clip TTS rớt lại
            # (rò rỉ Zero-Logging + phình đĩa). Best-effort: _cleanup_temp nuốt lỗi xóa.
            cleanup_paths.discard(final_audio_path)  # nếu không watermark thì final == mixed -> vẫn giữ đúng đầu ra
            cleanup_paths.discard(None)
            await _run_blocking(self._cleanup_temp, cleanup_paths)
            # Xóa nốt thư mục job htdemucs (CHỈ khi ĐÃ tách thật -> demucs_job_dir được gán).
            # Nếu fail-closed thì instrumental_path == local_audio_path, demucs_job_dir vẫn
            # None nên TUYỆT ĐỐI không rmtree gốc temp.
            if demucs_job_dir:
                import shutil
                await _run_blocking(shutil.rmtree, demucs_job_dir, ignore_errors=True)

    async def process_job(self, audio_url: str, config: dict):
        """Bước MỘT-LẦN (không Human-in-the-Loop): tải audio -> [ASR nếu client chưa gửi
        segments] -> Qwen dịch -> TTS -> tách nền -> mix -> watermark.

        Ủy thác toàn bộ phần lồng tiếng cho _run_dub_pipeline; chỉ KHÁC ở cách lấy
        translated_segments = ASR (nếu thiếu) + Qwen dịch cục bộ THƯỜNG TRÚ VRAM."""

        async def _prepare_via_asr_and_qwen(local_audio_path):
            # ASR chỉ khi client chưa gửi segments đã được người dùng duyệt.
            segments = config.get("segments", [])
            asr_ran = False
            if not segments:
                from src.asr_service import asr_service
                segments = await _run_blocking(
                    asr_service.transcribe, local_audio_path
                )
                asr_ran = True  # để báo pipeline TRUNG THỰC: chỉ liệt kê Whisper khi thật sự chạy

            if not segments:
                raise RuntimeError("Không có segment nào để xử lý (audio rỗng hoặc ASR trả về trống).")

            # Dịch bằng Qwen cục bộ THƯỜNG TRÚ VRAM (fail-closed nếu chưa nạp). Gọi ĐÚNG
            # instance đã đăng ký ở load_all_models (self.models["translation"]) — không
            # re-import singleton để tránh mọi khả năng nạp bản thứ hai / nạp lười lúc chạy.
            # Truyền ngôn ngữ GỐC để đếm âm tiết đúng (căn lip-sync), không ghim cứng "en".
            translation_service = self.models["translation"]
            translated_segments = await _translate_with_quality_policy(
                translation_service,
                segments=segments,
                target_language=config.get("target_language", "Vietnamese"),
                style=config.get("style", "Formal"),
                source_language=config.get("source_language") or "en",
                prompt_profile=config.get("prompt_profile"),
                # Quality policy is deployment configuration, not a client-controlled
                # field. A request must never downgrade a strict process to observe.
                # Process is the one-shot automatic path. Keep its default
                # fail-closed even when the global mode is ``observe`` for Analyze.
                quality_mode=_process_quality_mode(),
                # A semantic pass must come from the trusted judge/HITL boundary;
                # never accept a client/config boolean as proof of correctness.
                semantic_judges=None,
            )
            # prep_stages TRUNG THỰC: Whisper CHỈ khi tự ASR; Translation luôn (Qwen đã chạy).
            prep_stages = (["Whisper"] if asr_ran else []) + ["Translation"]
            return translated_segments, prep_stages

        return await self._run_dub_pipeline(
            audio_url,
            config,
            _prepare_via_asr_and_qwen,
        )

    async def render_job(self, audio_url: str, config: dict):
        """Bước RENDER (Human-in-the-Loop, ADR 0002): lồng tiếng từ manifest ĐÃ ĐƯỢC DUYỆT.

        Ràng buộc CỐT LÕI (CLAUDE_CODE_CONTINUATION_PLAN.md dòng 392-393): Render đọc NGUYÊN
        VĂN `translatedText` trong bản duyệt và KHÔNG gọi Qwen lại, KHÔNG chạy ASR lại. Đây là
        nguồn nội dung duy nhất; TTS chỉ tạo bản đọc TN riêng và không ghi ngược manifest.
        Nếu Render dịch lại thì
        mô hình có thể ÂM THẦM ghi đè quyết định của người biên tập (mất zero-trust HITL);
        nếu ASR lại thì bỏ qua các chỉnh sửa thủ công. Ủy thác phần lồng tiếng chung cho
        _run_dub_pipeline; chỉ khác ở cách lấy translated_segments = ánh xạ NGUYÊN VĂN bản duyệt."""

        from src.tts_service import TTSProfileError

        approved = config.get("segments") or []
        voice_map = config.get("voice_map") or {}
        if not isinstance(approved, list) or not isinstance(voice_map, dict):
            raise TTSProfileError("approved voice mapping is invalid")
        for segment in approved:
            if not isinstance(segment, dict):
                raise TTSProfileError("approved voice mapping is invalid")
            speaker = segment.get("speaker")
            voice_id = segment.get("voiceId")
            if (
                not isinstance(speaker, str)
                or not speaker
                or not isinstance(voice_id, str)
                or not voice_id
                or voice_map.get(speaker) != voice_id
            ):
                raise TTSProfileError("approved voice mapping is invalid")

        async def _prepare_from_approved(local_audio_path):
            # KHÔNG dùng local_audio_path để lấy NỘI DUNG — segment đã có sẵn trong bản duyệt
            # (audio vẫn được tải để tách nhạc nền + mix). Ánh xạ camelCase (ApprovedSegment
            # trên dây từ Gateway) -> snake_case mà vòng TTS đọc, GIỮ NGUYÊN translatedText đã
            # duyệt. prep_stages=[] -> pipeline KHÔNG liệt kê "Whisper"/"Translation" (TRUNG
            # THỰC: Render không transcribe, không dịch).
            translated_segments = [
                {
                    "id": seg.get("id"),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "translated_text": seg.get("translatedText", ""),  # nguồn duyệt; TN tạo bản đọc riêng
                    "original_text": seg.get("sourceText", ""),
                    "speaker_id": seg.get("speaker"),
                    "voice_id": seg.get("voiceId"),
                    "emotion": seg.get("emotion", "NEUTRAL"),
                }
                for seg in approved
            ]
            return translated_segments, []

        return await self._run_dub_pipeline(
            audio_url,
            config,
            _prepare_from_approved,
            strict_alignment=True,
        )

    async def analyze_job(self, audio_url: str, config: dict) -> dict:
        """Bước ANALYZE (Human-in-the-Loop, ADR 0002): tải audio -> ASR KÈM độ tin cậy ->
        Dịch -> lắp AnalyzeSegment với sourceText & translatedText TÁCH BẠCH + metadata
        diarization TRUNG THỰC.

        Trả về NỘI DUNG phân tích thô ({sourceLanguage, targetLanguage, segments, diarization})
        — CHƯA gắn danh tính job / version / seal. Endpoint bọc thành AnalyzeResult, mã hóa
        ECIES tới khóa mã hóa của thiết bị rồi upload ciphertext lên R2. Zero-knowledge:
        transcript + bản dịch CHỈ tồn tại trong artifact đã mã hóa — không bao giờ ở response
        HTTP, log, hay KV của Gateway.

        Suy luận ML thật (Whisper/Qwen) thường trú VRAM (residual_hardware); phần LẮP RÁP ở
        đây (map confidence, tách text, diarization degraded) kiểm thử được đầy đủ trên CPU."""
        if not self.is_loaded:
            raise RuntimeError("Models chưa được nạp lên VRAM!")

        from src.audio_service import audio_service, AudioIntegrityError

        # Ràng buộc toàn vẹn (fail-closed) y như process_job: object key R2 dùng chung nên
        # một job khác có thể ghi đè trước khi ta tải. md5 client ký kèm là "vé" ràng buộc;
        # thiếu vé -> AudioIntegrityError -> endpoint 422 (terminal), KHÔNG để Gateway retry
        # (CC33-01: lỗi payload xác định không tự khỏi khi retry).
        expected_md5 = (config.get("audio_md5") or "").strip()
        if not expected_md5:
            raise AudioIntegrityError("Thiếu md5 toàn vẹn audio — từ chối analyze (fail-closed).")

        cleanup_paths = set()
        demucs_job_dir = None
        try:
            from src.asr_service import asr_service
            from src.audio_engine import audio_engine

            downloaded_path = await _run_blocking(
                audio_service.download_audio, audio_url, expected_md5
            )
            cleanup_paths.add(downloaded_path)

            # ADR 0002 makes source separation a required Analyze stage. Demucs must
            # produce BOTH stems; ASR and diarization consume vocals.wav only. Continuing
            # on the mixed source would silently lower transcript/speaker quality while
            # still returning success, so an unavailable/incomplete separation is a hard
            # job failure (mapped to the canonical 500 by the endpoint for retry/failover).
            stems = await _run_blocking(
                audio_engine.extract_analysis_stems, downloaded_path
            )
            demucs_job_dir = stems.job_dir
            cleanup_paths.add(stems.instrumental_path)
            cleanup_paths.add(stems.vocals_path)
            if not stems.separated:
                raise RuntimeError("Analyze source separation is unavailable or incomplete.")

            analysis_audio_path = stems.vocals_path
            asr = await _run_blocking(
                asr_service.transcribe_analyze, analysis_audio_path
            )
            asr_segments = asr.get("segments", [])
            detected_language = asr.get("language") or ""

            # Diarization TRUNG THỰC (ADR 0003): hỏi adapter về NĂNG LỰC trước. CHỈ khi khả dụng
            # (engine + HF_TOKEN + CUDA) mới chạy seam thật (pyannote qua WhisperX) để gán nhãn
            # speaker vào timestamp ASR — nhãn đó chảy qua `seg.get("speaker")` xuống segments_out.
            # Degraded HOẶC lỗi runtime giữa chừng -> KHÔNG bịa multi-speaker: asr_segments giữ
            # nguyên SPEAKER_UNKNOWN và metadata nói rõ degraded (No-Fake-Success, plan 403-404).
            from src.diarization_service import SPEAKER_UNKNOWN, diarization_service

            diar_available, diar_reason = diarization_service.capability()
            if diar_available and asr_segments:
                try:
                    asr_segments = await _run_blocking(
                        diarization_service.diarize,
                        analysis_audio_path,
                        asr_segments,
                    )
                except Exception:
                    # Seam GPU nổ giữa chừng KHÔNG được làm hỏng cả Analyze, cũng KHÔNG được
                    # báo cáo là tách-giọng thành công — rơi về degraded trung thực.
                    diar_available = False
                    diar_reason = "Diarization runtime lỗi — degraded (một giọng)."

            # sourceLanguage: ưu tiên client chỉ định (nếu có) — nếu không, dùng ngôn ngữ
            # ASR phát hiện. Cũng truyền vào translate để đếm âm tiết đúng ngôn ngữ gốc.
            source_language = config.get("source_language") or detected_language or "en"
            target_language = config.get("target_language", "Vietnamese")

            segments_out = []
            if asr_segments:
                translation_service = self.models["translation"]
                translated = await _translate_with_quality_policy(
                    translation_service,
                    segments=asr_segments,
                    target_language=target_language,
                    style=config.get("style", "Formal"),
                    source_language=source_language,
                    prompt_profile=config.get("prompt_profile"),
                    quality_mode=os.environ.get(
                        "TRANSLATION_ANALYZE_QUALITY_MODE", "observe"
                    ),
                    semantic_judges=None,
                )
                # translate_segments giữ ID-parity, nên map theo id để ghép đúng câu
                # (không dựa vào thứ tự list phòng khi backend đổi thứ tự).
                by_id = {str(t.get("id")): t for t in translated}
                quality_by_id = (
                    _quality_rows_for_analyze(
                        translation_service,
                        {str(segment.get("id")) for segment in asr_segments},
                    )
                    if _analyze_quality_metadata_enabled()
                    else {}
                )

                for seg in asr_segments:
                    sid = str(seg.get("id"))
                    t = by_id.get(sid, {})
                    segment_out = {
                        "id": sid,
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
                        # TÁCH BẠCH (zero-knowledge): bóc băng gốc và bản dịch là HAI trường
                        # riêng — người biên tập sửa translatedText, KHÔNG chạm sourceText.
                        "sourceText": seg.get("text", ""),
                        "translatedText": t.get("translated_text", ""),
                        "confidence": _confidence_from_logprob(seg.get("avg_logprob", 0.0)),
                        "emotion": t.get("emotion", "NEUTRAL"),
                        "noSpeechProb": seg.get("no_speech_prob", 0.0),
                    }
                    quality_summary = quality_by_id.get(sid)
                    if quality_summary is not None:
                        segment_out["translationQuality"] = quality_summary
                    segments_out.append(segment_out)

            # Diarization TRUNG THỰC: capability chỉ nói engine CÓ THỂ chạy; kết quả chỉ được
            # coi là "full" khi ít nhất một segment thật sự nhận nhãn speaker. UNKNOWN là nhãn
            # thiếu dữ liệu, không phải một người nói và vì vậy không được cộng vào speakerCount.
            # Điều này chặn trường hợp pyannote trả rỗng/toàn UNKNOWN nhưng worker vẫn quảng cáo
            # diarization thành công.
            real_speakers = {
                s.get("speaker")
                for s in segments_out
                if s.get("speaker") and s.get("speaker") != SPEAKER_UNKNOWN
            }
            diarized_ok = bool(diar_available and segments_out and real_speakers)
            speaker_count = len(real_speakers)
            if diar_available and segments_out and not real_speakers:
                diar_reason = "Diarization không gán được nhãn speaker — degraded (một giọng)."
            diarization = {
                "available": diarized_ok,
                # MIRROR DiarizationInfoSchema trong packages/shared-types.
                "mode": "full" if diarized_ok else "degraded",
                "speakerCount": speaker_count,
                "reason": "" if diarized_ok else diar_reason,
            }

            return {
                "sourceLanguage": source_language,
                "targetLanguage": target_language,
                "segments": segments_out,
                "diarization": diarization,
            }
        finally:
            # Dọn audio nguồn tải-về dù analyze THÀNH CÔNG hay NỔ (Zero-Logging + phình đĩa):
            # analyze KHÔNG sinh đầu ra cuối cần giữ (artifact mã hóa upload thẳng từ endpoint).
            cleanup_paths.discard(None)
            await _run_blocking(self._cleanup_temp, cleanup_paths)
            if demucs_job_dir:
                import shutil
                await _run_blocking(shutil.rmtree, demucs_job_dir, ignore_errors=True)
