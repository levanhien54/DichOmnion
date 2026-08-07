"""Loopback-only MOSS-TTS Local v1.5 compatibility service.

The public worker sends only an opaque profile id.  This process owns model loading and
the optional consent registry, so reference paths and approved text never cross the
worker/Gateway capability boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import struct
import tempfile
import time
import unicodedata
import wave
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.tts_service import LocalTTSProfile, TTSConfigurationError, _parse_local_profiles


MOSS_TTS_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MOSS_TTS_MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
MOSS_TTS_CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MOSS_TTS_CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
MOSS_TTS_SAMPLE_RATE = 48_000
MOSS_TTS_CHANNELS = 2
MOSS_TTS_FRAMES_PER_SECOND = 12.5

logger = logging.getLogger("omnivoice.moss_tts")
logger.setLevel(logging.WARNING)

_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RATE_RE = re.compile(r"^[+-](?:0|[1-9][0-9]?)%$")
_VOLUME_RE = re.compile(r"^[+-](?:0|[1-9][0-9]?)%$")
_PITCH_RE = re.compile(r"^[+-](?:0|[1-9][0-9]{0,2})Hz$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_SUFFIX = ".wav"
_MIN_REFERENCE_SECONDS = 0.5
_MAX_REFERENCE_SECONDS = 30.0
_MIN_REFERENCE_SAMPLE_RATE = 8_000
_MAX_REFERENCE_SAMPLE_RATE = 96_000
_MIN_REFERENCE_RMS = 1e-3
_MAX_DEBUG_DIAGNOSTIC_CHARS = 512
_LOAD_STAGES = frozenset(
    {
        "torch_import",
        "transformers_import",
        "cuda_preflight",
        "profile_registry",
        "model_snapshot",
        "codec_snapshot",
        "attention_config",
        "cuda_backend_config",
        "dtype_selection",
        "processor_codec_load",
        "codec_to_cuda",
        "model_load",
        "model_to_cuda",
        "model_eval",
    }
)
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]{1,64}-----.*?(?:-----END [A-Z0-9 ]{1,64}-----|$)",
    re.IGNORECASE | re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_TOKEN_RE = re.compile(
    r"(?i)\b(?:hf_[A-Za-z0-9]{8,}|rpa_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{8,})"
)
_LABELED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|private[_-]?key|client[_-]?secret|"
    r"secret(?:[_-]?key)?|credential|signature|password|authorization|token)"
    r"\b(\s*[:=]\s*)([^\s,;]+)"
)
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@")
_JWT_RE = re.compile(
    r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"
)
_PRIVATE_DETAIL_RE = re.compile(
    r"(?is)\b(?:profile(?:[_ -]?(?:json|configuration))?|reference(?:[_ -]?(?:"
    r"transcript|text|audio|path))?|transcript)\b\s*[:=]\s*.*$"
)
_LANGUAGES = Literal[
    "English",
    "Japanese",
    "Korean",
    "Cantonese",
    "Chinese",
    "Vietnamese",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Italian",
]
_EMOTIONS = Literal["NEUTRAL", "HAPPY", "ANGRY", "SAD", "WHISPERING", "SHOUTING"]

# These exact triples are emitted by ModelManager.  Requiring the matching emotion prevents
# numeric Edge-style controls from being accepted and then silently ignored by MOSS.
EMOTION_PROSODY: dict[str, dict[str, str]] = {
    "SHOUTING": {"rate": "+8%", "volume": "+30%", "pitch": "+15Hz"},
    "WHISPERING": {"rate": "-8%", "volume": "-40%", "pitch": "-10Hz"},
    "ANGRY": {"rate": "+10%", "volume": "+15%", "pitch": "+8Hz"},
    "SAD": {"rate": "-12%", "volume": "+0%", "pitch": "-12Hz"},
    "HAPPY": {"rate": "+5%", "volume": "+5%", "pitch": "+12Hz"},
    "NEUTRAL": {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
}

EMOTION_INSTRUCTIONS: dict[str, str] = {
    "SHOUTING": "Speak loudly with strong projection and urgent energy.",
    "WHISPERING": "Whisper softly with restrained breath and low intensity.",
    "ANGRY": "Speak with controlled anger, forceful emphasis, and a faster pace.",
    "SAD": "Speak sadly with subdued energy, a lower tone, and a slower pace.",
    "HAPPY": "Speak warmly and cheerfully with an upbeat, slightly faster pace.",
    "NEUTRAL": "Speak naturally with a neutral, conversational delivery.",
}


class MossRequestError(RuntimeError):
    pass


class MossUnavailableError(RuntimeError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProsodyPayload(_StrictModel):
    rate: str
    volume: str
    pitch: str

    @field_validator("rate")
    @classmethod
    def _valid_rate(cls, value: str) -> str:
        if not _RATE_RE.fullmatch(value):
            raise ValueError("invalid rate")
        return value

    @field_validator("volume")
    @classmethod
    def _valid_volume(cls, value: str) -> str:
        if not _VOLUME_RE.fullmatch(value):
            raise ValueError("invalid volume")
        return value

    @field_validator("pitch")
    @classmethod
    def _valid_pitch(cls, value: str) -> str:
        if not _PITCH_RE.fullmatch(value):
            raise ValueError("invalid pitch")
        return value


class OutputPayload(_StrictModel):
    format: Literal["wav"]


class SynthesisPayload(_StrictModel):
    schema_version: Literal[1]
    text: str = Field(min_length=1, max_length=2_000)
    language: _LANGUAGES
    profile_id: str = Field(min_length=1, max_length=128)
    emotion: _EMOTIONS = "NEUTRAL"
    duration_seconds: float | None = Field(default=None, gt=0.0, le=300.0)
    prosody: ProsodyPayload
    output: OutputPayload

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("invalid schema version")
        return value

    @field_validator("text")
    @classmethod
    def _valid_text(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 and char not in "\t\n\r" for char in value):
            raise ValueError("invalid text")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("invalid text") from exc
        return value

    @field_validator("profile_id")
    @classmethod
    def _valid_profile_id(cls, value: str) -> str:
        if not _PROFILE_ID_RE.fullmatch(value):
            raise ValueError("invalid profile id")
        return value


@dataclass(frozen=True)
class RuntimeProfile:
    public: LocalTTSProfile
    reference_path: Path | None
    reference_transcript: str | None
    reference_id: str | None = None
    reference_sha256: str | None = None
    consent_expires_at: datetime | None = None
    retention_delete_after: datetime | None = None
    # Populated only for the request-local revalidation immediately before synthesis.
    # Startup state never retains a second in-memory copy of private reference audio.
    reference_audio: bytes | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _ValidatedReference:
    path: Path
    transcript: str
    reference_id: str
    sha256: str
    consent_expires_at: datetime | None
    retention_delete_after: datetime | None
    audio: bytes | None = field(default=None, repr=False, compare=False)


def _diagnostic_sensitive_values(
    profiles: dict[str, RuntimeProfile] | None = None,
) -> tuple[str, ...]:
    values: set[str] = set()
    credential_markers = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")
    private_config_names = {
        "LOCAL_TTS_PROFILES_JSON",
        "MOSS_TTS_PROFILE_REGISTRY_PATH",
        "MOSS_TTS_REFERENCE_ROOT",
    }
    for name, value in os.environ.items():
        upper_name = name.upper()
        if value and (
            name in private_config_names
            or any(marker in upper_name for marker in credential_markers)
        ):
            values.add(value)
    for profile in (profiles or {}).values():
        if profile.reference_transcript:
            values.add(profile.reference_transcript)
        if profile.reference_path is not None:
            values.add(str(profile.reference_path))
    return tuple(sorted(values, key=len, reverse=True))


def _sanitize_debug_message(
    error: Exception,
    sensitive_values: tuple[str, ...] = (),
) -> str:
    try:
        message = str(error)
    except Exception:
        message = "unprintable exception"

    for sensitive in sensitive_values:
        if sensitive:
            message = message.replace(sensitive, "[REDACTED]")
    message = _PEM_RE.sub("[REDACTED_PEM]", message)
    message = _BEARER_RE.sub("Bearer [REDACTED]", message)
    message = _TOKEN_RE.sub("[REDACTED_TOKEN]", message)
    message = _JWT_RE.sub("[REDACTED_JWT]", message)
    message = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", message)
    message = _LABELED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        message,
    )
    message = _PRIVATE_DETAIL_RE.sub("private_detail=[REDACTED]", message)

    # If a library embeds even part of the private profile document, discarding the
    # complete message is safer than trying to parse an arbitrary exception string.
    profile_json_marker = re.search(
        r"(?i)[\"']?(?:profiles|displayName|referenceAudio|referenceId|transcript|"
        r"provenance|consent)[\"']?\s*:",
        message,
    )
    if profile_json_marker and ("{" in message or "[" in message):
        message = "[REDACTED_PROFILE_CONFIGURATION]"

    message = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in message
    )
    message = " ".join(message.split()) or "empty exception message"
    if len(message) > _MAX_DEBUG_DIAGNOSTIC_CHARS:
        message = message[: _MAX_DEBUG_DIAGNOSTIC_CHARS - 3].rstrip() + "..."
    return message


def _log_load_failure(
    stage: str,
    error: Exception,
    sensitive_values: tuple[str, ...] = (),
) -> None:
    safe_stage = stage if stage in _LOAD_STAGES else "unknown"
    error_class = re.sub(r"[^A-Za-z0-9_.-]", "_", type(error).__name__)[:128]
    error_class = error_class or "Exception"
    if os.environ.get("MOSS_TTS_DEBUG") == "1":
        logger.error(
            "MOSS load failed stage=%s error_class=%s debug_message=%s",
            safe_stage,
            error_class,
            _sanitize_debug_message(error, sensitive_values),
        )
    else:
        logger.error(
            "MOSS load failed stage=%s error_class=%s",
            safe_stage,
            error_class,
        )


@contextmanager
def _load_stage(stage: str, sensitive_values: tuple[str, ...] = ()):
    try:
        yield
    except Exception as error:
        _log_load_failure(stage, error, sensitive_values)
        raise MossUnavailableError("MOSS model could not be loaded") from None


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs_hook,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite")),
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 64:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise TTSConfigurationError("MOSS profile registry is invalid") from None
    if parsed.tzinfo is None:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    return parsed.astimezone(timezone.utc)


def _validate_pcm_reference(payload: bytes) -> None:
    """Require a bounded, audible PCM16 WAV before a profile can be ready."""
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            raw = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        raise TTSConfigurationError("MOSS profile registry is invalid") from None

    expected_bytes = frame_count * channels * sample_width
    duration = frame_count / sample_rate if sample_rate > 0 else math.inf
    if (
        compression != "NONE"
        or channels not in (1, 2)
        or sample_width != 2
        or not _MIN_REFERENCE_SAMPLE_RATE <= sample_rate <= _MAX_REFERENCE_SAMPLE_RATE
        or frame_count <= 0
        or len(raw) != expected_bytes
        or not math.isfinite(duration)
        or not _MIN_REFERENCE_SECONDS <= duration <= _MAX_REFERENCE_SECONDS
    ):
        raise TTSConfigurationError("MOSS profile registry is invalid")

    squared = 0.0
    sample_count = 0
    for (sample,) in struct.iter_unpack("<h", raw):
        normalized = sample / 32768.0
        squared += normalized * normalized
        sample_count += 1
    rms = math.sqrt(squared / sample_count) if sample_count else 0.0
    if not math.isfinite(rms) or rms <= _MIN_REFERENCE_RMS:
        raise TTSConfigurationError("MOSS profile registry is invalid")


def _registry_rows(path: Path) -> dict[str, dict[str, Any]]:
    try:
        if path.stat().st_size > 256 * 1024:
            raise TTSConfigurationError("MOSS profile registry is invalid")
        document = _strict_json(path.read_text(encoding="utf-8"))
    except TTSConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError):
        raise TTSConfigurationError("MOSS profile registry is invalid") from None
    if not isinstance(document, dict) or set(document) != {"schema_version", "profiles"}:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    if document["schema_version"] != 1 or isinstance(document["schema_version"], bool):
        raise TTSConfigurationError("MOSS profile registry is invalid")
    rows = document["profiles"]
    if not isinstance(rows, list) or len(rows) > 128:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TTSConfigurationError("MOSS profile registry is invalid")
        expected = {
            "profileId",
            "referenceId",
            "audioFile",
            "sha256",
            "transcript",
            "provenance",
            "consent",
            "retention",
        }
        if set(row) != expected:
            raise TTSConfigurationError("MOSS profile registry is invalid")
        profile_id = row.get("profileId")
        if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
            raise TTSConfigurationError("MOSS profile registry is invalid")
        if profile_id in result:
            raise TTSConfigurationError("MOSS profile registry is invalid")
        result[profile_id] = row
    return result


def _validate_registry_profile(
    row: dict[str, Any],
    reference_root: Path,
    now: datetime,
    *,
    include_reference_audio: bool,
) -> _ValidatedReference:
    for key in ("referenceId", "audioFile", "transcript", "provenance"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
            raise TTSConfigurationError("MOSS profile registry is invalid")

    # Authorize before opening private audio. A revoked/expired row must not even cause
    # its reference bytes to be read during health or request revalidation.
    consent = row.get("consent")
    if not isinstance(consent, dict) or set(consent) != {
        "granted",
        "revoked",
        "scope",
        "expiresAt",
    }:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    if consent.get("granted") is not True or consent.get("revoked") is not False:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    if consent.get("scope") not in (
        "voice-cloning",
        "synthetic-no-natural-person",
    ):
        raise TTSConfigurationError("MOSS profile registry is invalid")
    expires_at = _parse_timestamp(consent.get("expiresAt"))
    if expires_at is not None and expires_at <= now:
        raise TTSConfigurationError("MOSS profile registry is invalid")

    retention = row.get("retention")
    if not isinstance(retention, dict) or set(retention) != {"deleteAfter"}:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    delete_after = _parse_timestamp(retention.get("deleteAfter"))
    if delete_after is not None and delete_after <= now:
        raise TTSConfigurationError("MOSS profile registry is invalid")

    audio_file = row["audioFile"]
    if Path(audio_file).is_absolute() or "\x00" in audio_file:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    try:
        root = reference_root.resolve(strict=True)
        unresolved = root / audio_file
        if unresolved.is_symlink():
            raise TTSConfigurationError("MOSS profile registry is invalid")
        candidate = unresolved.resolve(strict=True)
    except TTSConfigurationError:
        raise
    except (OSError, RuntimeError):
        raise TTSConfigurationError("MOSS profile registry is invalid") from None
    try:
        candidate.relative_to(root)
    except ValueError:
        raise TTSConfigurationError("MOSS profile registry is invalid") from None
    if not candidate.is_file():
        raise TTSConfigurationError("MOSS profile registry is invalid")
    if candidate.suffix.casefold() != _REFERENCE_SUFFIX:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    max_bytes = _bounded_int(
        "MOSS_TTS_MAX_REFERENCE_BYTES",
        32 * 1024 * 1024,
        1_024,
        128 * 1024 * 1024,
    )
    try:
        # Read once from the resolved file handle. Hashing and decoding these same bytes
        # prevents a path replacement between separate stat/hash/decode operations.
        with candidate.open("rb") as handle:
            reference_audio = handle.read(max_bytes + 1)
    except OSError:
        raise TTSConfigurationError("MOSS profile registry is invalid") from None
    if len(reference_audio) < 44 or len(reference_audio) > max_bytes:
        raise TTSConfigurationError("MOSS profile registry is invalid")
    expected_sha = row.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise TTSConfigurationError("MOSS profile registry is invalid")
    actual_sha = hashlib.sha256(reference_audio).hexdigest()
    if not hmac.compare_digest(actual_sha, expected_sha):
        raise TTSConfigurationError("MOSS profile registry is invalid")
    _validate_pcm_reference(reference_audio)

    return _ValidatedReference(
        path=candidate,
        transcript=row["transcript"].strip(),
        reference_id=row["referenceId"].strip(),
        sha256=expected_sha,
        consent_expires_at=expires_at,
        retention_delete_after=delete_after,
        audio=reference_audio if include_reference_audio else None,
    )


def load_runtime_profiles(
    now: datetime | None = None,
    *,
    reference_audio_profile_id: str | None = None,
) -> dict[str, RuntimeProfile]:
    public_profiles = _parse_local_profiles()
    if not public_profiles:
        raise TTSConfigurationError("MOSS profiles are not configured")
    if any(profile.engine != "moss-tts" for profile in public_profiles.values()):
        raise TTSConfigurationError("MOSS profile catalog contains another engine")
    # Reference material belongs in the registry file, never in the public-catalog env.
    if any(profile.internal_reference is not None for profile in public_profiles.values()):
        raise TTSConfigurationError("MOSS references must use the private registry")

    required_ids = {
        profile_id
        for profile_id, profile in public_profiles.items()
        if profile.reference_required
    }
    rows: dict[str, dict[str, Any]] = {}
    if required_ids:
        raw_registry = os.environ.get("MOSS_TTS_PROFILE_REGISTRY_PATH", "").strip()
        raw_root = os.environ.get("MOSS_TTS_REFERENCE_ROOT", "").strip()
        if not raw_registry or not raw_root:
            raise TTSConfigurationError("MOSS private profile registry is not configured")
        registry_path = Path(raw_registry)
        reference_root = Path(raw_root)
        if not registry_path.is_absolute() or not reference_root.is_absolute():
            raise TTSConfigurationError("MOSS private profile registry is invalid")
        rows = _registry_rows(registry_path)
        if not required_ids.issubset(rows):
            raise TTSConfigurationError("MOSS private profile registry is incomplete")

    checked_at = now or datetime.now(timezone.utc)
    runtime: dict[str, RuntimeProfile] = {}
    for profile_id, profile in public_profiles.items():
        reference_path: Path | None = None
        reference_transcript: str | None = None
        private_profile: _ValidatedReference | None = None
        if profile.reference_required:
            private_profile = _validate_registry_profile(
                rows[profile_id],
                Path(os.environ["MOSS_TTS_REFERENCE_ROOT"]),
                checked_at,
                include_reference_audio=(reference_audio_profile_id == profile_id),
            )
            reference_path = private_profile.path
            reference_transcript = private_profile.transcript
        runtime[profile_id] = RuntimeProfile(
            public=profile,
            reference_path=reference_path,
            reference_transcript=reference_transcript,
            reference_id=(private_profile.reference_id if private_profile else None),
            reference_sha256=(
                private_profile.sha256 if private_profile else None
            ),
            consent_expires_at=(
                private_profile.consent_expires_at if private_profile else None
            ),
            retention_delete_after=(
                private_profile.retention_delete_after if private_profile else None
            ),
            reference_audio=(private_profile.audio if private_profile else None),
        )
    return runtime


@contextmanager
def _private_reference_snapshot(profile: RuntimeProfile):
    """Expose only the already-authorized bytes to MOSS, then remove them.

    MOSS accepts a filesystem path. Passing the registry path directly would reopen a
    mutable file after its hash/consent check, allowing a swap between validation and
    inference. A mode-0600 ``mkstemp`` snapshot closes that race and is always removed.
    """
    if not profile.public.reference_required:
        yield None
        return
    payload = profile.reference_audio
    expected_sha = profile.reference_sha256
    if (
        not isinstance(payload, bytes)
        or not payload
        or not isinstance(expected_sha, str)
        or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha)
    ):
        raise MossUnavailableError("MOSS reference authorization is unavailable")

    fd, path = tempfile.mkstemp(prefix=".moss-reference-", suffix=_REFERENCE_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
        yield path
    except MossUnavailableError:
        raise
    except Exception:
        raise MossUnavailableError(
            "MOSS reference authorization is unavailable"
        ) from None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(path)
        except OSError:
            pass


def _snapshot_path(repo_id: str, revision: str, override_name: str) -> str:
    override = os.environ.get(override_name, "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute() or not path.is_dir():
            raise MossUnavailableError("MOSS model cache is unavailable")
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        raise MossUnavailableError("MOSS model cache is unavailable") from None


def _target_frames(payload: SynthesisPayload, maximum: int) -> int:
    if payload.duration_seconds is not None:
        seconds = payload.duration_seconds
    else:
        characters_per_second = 4.0 if payload.language in {"Chinese", "Cantonese", "Japanese"} else 12.0
        seconds = max(1.2, len(payload.text.strip()) / characters_per_second)
    return max(8, min(maximum, int(round(seconds * MOSS_TTS_FRAMES_PER_SECOND))))


def _wav_bytes(audio: Any, sample_rate: int) -> tuple[bytes, float]:
    import torch

    tensor = audio.detach().float().cpu()
    if tensor.ndim != 2 or tensor.shape[0] != MOSS_TTS_CHANNELS or tensor.shape[1] <= 0:
        raise MossUnavailableError("MOSS returned invalid audio")
    if sample_rate != MOSS_TTS_SAMPLE_RATE or not bool(torch.isfinite(tensor).all()):
        raise MossUnavailableError("MOSS returned invalid audio")
    duration = tensor.shape[1] / float(sample_rate)
    maximum = _bounded_float("MOSS_TTS_MAX_OUTPUT_SECONDS", 90.0, 1.0, 300.0)
    if not 0.05 <= duration <= maximum:
        raise MossUnavailableError("MOSS returned invalid audio")
    peak = float(tensor.abs().max().item())
    rms = float(torch.sqrt(torch.mean(tensor.square())).item())
    if not math.isfinite(peak) or not math.isfinite(rms) or rms < 1e-5:
        raise MossUnavailableError("MOSS returned invalid audio")
    if peak > 0.98:
        tensor = tensor * (0.98 / peak)
    pcm = (
        tensor.clamp(-1.0, 1.0)
        .transpose(0, 1)
        .contiguous()
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .numpy()
        .tobytes()
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(MOSS_TTS_CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue(), duration


def _runtime_device(value: Any) -> str:
    """Return only a bounded device class from a live model/codec handle."""
    if value is None:
        return "unknown"
    try:
        parameters = getattr(value, "parameters", None)
        if callable(parameters):
            devices = set()
            for parameter in parameters():
                if not hasattr(parameter, "device"):
                    continue
                raw = getattr(getattr(parameter, "device", None), "type", None)
                if raw is None:
                    raw = getattr(parameter, "device", None)
                name = str(raw).strip().lower()
                if name == "cuda" or name.startswith("cuda:"):
                    devices.add("cuda")
                elif name == "cpu" or name.startswith("cpu:"):
                    devices.add("cpu")
                elif name == "meta" or name.startswith("meta:"):
                    devices.add("meta")
                else:
                    devices.add("unknown")
            if len(devices) == 1:
                return next(iter(devices))
            if len(devices) > 1:
                return "mixed"
        raw_device = getattr(value, "device", getattr(value, "_device", None))
        raw = getattr(raw_device, "type", raw_device)
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


class MossTTSEngine:
    def __init__(self) -> None:
        self.ready = False
        self.profile_ids: tuple[str, ...] = ()
        self._profiles: dict[str, RuntimeProfile] = {}
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cuda"
        self.last_metrics: dict[str, float | int] = {}

    def residency_status(self) -> dict[str, Any]:
        """Prove model and codec residence from their current CUDA handles."""
        model_device = _runtime_device(self._model)
        try:
            codec = getattr(self._processor, "audio_tokenizer", None)
        except Exception:
            codec = None
        codec_device = _runtime_device(codec)
        model_resident = self._model is not None and model_device == "cuda"
        codec_resident = codec is not None and codec_device == "cuda"
        devices = {model_device, codec_device}
        device = next(iter(devices)) if len(devices) == 1 else "mixed"
        return {
            "model_resident": model_resident,
            "codec_resident": codec_resident,
            "all_resident": model_resident and codec_resident,
            "device": device,
        }

    def authorized_profile_ids(self) -> tuple[str, ...]:
        """Revalidate reference grants for health without retaining reference bytes."""
        if not self.ready:
            return ()
        if not any(
            profile.public.reference_required for profile in self._profiles.values()
        ):
            return self.profile_ids
        try:
            current = load_runtime_profiles()
        except Exception:
            return ()
        current_ids = tuple(sorted(current))
        return current_ids if current_ids == self.profile_ids else ()

    def _profile_for_synthesis(self, profile_id: str) -> RuntimeProfile:
        cached = self._profiles.get(profile_id)
        if cached is None:
            raise MossRequestError("MOSS profile is unavailable")
        if not cached.public.reference_required:
            return cached
        try:
            current = load_runtime_profiles(reference_audio_profile_id=profile_id)
        except Exception:
            # Registry parser/path/hash errors are deliberately collapsed. They can
            # contain private filesystem or consent details and are operational, not a
            # malformed tenant request.
            raise MossUnavailableError(
                "MOSS reference authorization is unavailable"
            ) from None
        refreshed = current.get(profile_id)
        if (
            refreshed is None
            or not refreshed.public.reference_required
            or refreshed.reference_audio is None
        ):
            raise MossUnavailableError(
                "MOSS reference authorization is unavailable"
            )
        return refreshed

    def load(self) -> None:
        import importlib.util

        if self.ready:
            if self.residency_status()["all_resident"]:
                return
            raise MossUnavailableError("MOSS resident state is inconsistent")

        sensitive_values = _diagnostic_sensitive_values()
        with _load_stage("torch_import", sensitive_values):
            import torch
        with _load_stage("transformers_import", sensitive_values):
            from transformers import AutoModel, AutoProcessor

        with _load_stage("cuda_preflight", sensitive_values):
            if not torch.cuda.is_available():
                raise MossUnavailableError("MOSS requires CUDA")
        with _load_stage("profile_registry", sensitive_values):
            profiles = load_runtime_profiles()
        sensitive_values = _diagnostic_sensitive_values(profiles)
        with _load_stage("model_snapshot", sensitive_values):
            model_path = _snapshot_path(
                MOSS_TTS_MODEL_ID,
                MOSS_TTS_MODEL_REVISION,
                "MOSS_TTS_MODEL_PATH",
            )
        with _load_stage("codec_snapshot", sensitive_values):
            codec_path = _snapshot_path(
                MOSS_TTS_CODEC_ID,
                MOSS_TTS_CODEC_REVISION,
                "MOSS_TTS_CODEC_PATH",
            )
        with _load_stage("attention_config", sensitive_values):
            attention = os.environ.get("MOSS_TTS_ATTENTION", "sdpa").strip().lower()
            if attention not in {"sdpa", "eager", "flash_attention_2"}:
                raise MossUnavailableError("MOSS attention configuration is invalid")
            if attention == "flash_attention_2":
                major, _minor = torch.cuda.get_device_capability(0)
                if major < 8 or importlib.util.find_spec("flash_attn") is None:
                    raise MossUnavailableError("MOSS FlashAttention is unavailable")

        with _load_stage("cuda_backend_config", sensitive_values):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        with _load_stage("dtype_selection", sensitive_values):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            codec_weight_dtype = "bf16" if dtype is torch.bfloat16 else "fp32"
        with _load_stage("processor_codec_load", sensitive_values):
            processor = AutoProcessor.from_pretrained(
                model_path,
                codec_path=codec_path,
                codec_weight_dtype=codec_weight_dtype,
                codec_compute_dtype="bf16" if dtype is torch.bfloat16 else None,
                codec_attention_implementation=attention,
                trust_remote_code=True,
            )
        with _load_stage("codec_to_cuda", sensitive_values):
            processor.audio_tokenizer = processor.audio_tokenizer.to(self._device)
        with _load_stage("model_load", sensitive_values):
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation=attention,
                torch_dtype=dtype,
            )
        with _load_stage("model_to_cuda", sensitive_values):
            model = model.to(self._device)
        with _load_stage("model_eval", sensitive_values):
            model.eval()

        self._torch = torch
        self._processor = processor
        self._model = model
        self._profiles = profiles
        self.profile_ids = tuple(sorted(profiles))
        self.ready = True

    def synthesize(self, payload: SynthesisPayload) -> tuple[bytes, dict[str, float | int]]:
        if not self.ready or self._processor is None or self._model is None or self._torch is None:
            raise MossUnavailableError("MOSS is not ready")
        cached_profile = self._profiles.get(payload.profile_id)
        if (
            cached_profile is None
            or payload.language not in cached_profile.public.languages
        ):
            raise MossRequestError("MOSS profile is unavailable")
        if payload.prosody.model_dump() != EMOTION_PROSODY[payload.emotion]:
            raise MossRequestError("MOSS prosody does not match emotion")
        profile = self._profile_for_synthesis(payload.profile_id)
        if payload.language not in profile.public.languages:
            raise MossRequestError("MOSS profile is unavailable")

        maximum_frames = _bounded_int("MOSS_TTS_MAX_NEW_TOKENS", 750, 16, 4_096)
        frames = _target_frames(payload, maximum_frames)
        torch = self._torch
        try:
            with _private_reference_snapshot(profile) as reference_path:
                reference = [reference_path] if reference_path is not None else None
                message = self._processor.build_user_message(
                    text=payload.text,
                    reference=reference,
                    instruction=EMOTION_INSTRUCTIONS[payload.emotion],
                    tokens=frames,
                    language=payload.language,
                )
                # CUDA kernels are asynchronous. Synchronize the measurement boundaries so
                # the reported synthesis latency/RTF cannot understate paid GPU time.
                torch.cuda.synchronize()
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                seed = _bounded_int("MOSS_TTS_SEED", 20260805, 0, 2_147_483_647)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                batch = self._processor([[message]], mode="generation")
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                with torch.inference_mode():
                    outputs = self._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(maximum_frames, frames + 16),
                        do_sample=True,
                        audio_temperature=_bounded_float(
                            "MOSS_TTS_AUDIO_TEMPERATURE", 0.6, 0.1, 2.0
                        ),
                        audio_top_p=_bounded_float("MOSS_TTS_AUDIO_TOP_P", 0.95, 0.1, 1.0),
                        audio_top_k=_bounded_int("MOSS_TTS_AUDIO_TOP_K", 25, 1, 1_024),
                        audio_repetition_penalty=_bounded_float(
                            "MOSS_TTS_AUDIO_REPETITION_PENALTY", 1.2, 0.5, 2.0
                        ),
                    )
                decoded = self._processor.decode(outputs)
                messages = [item for item in decoded if item is not None and item.audio_codes_list]
                if len(messages) != 1:
                    raise MossUnavailableError("MOSS returned invalid audio")
                wav, duration = _wav_bytes(
                    messages[0].audio_codes_list[0],
                    int(self._processor.model_config.sampling_rate),
                )
                torch.cuda.synchronize()
        except MossUnavailableError:
            raise
        except Exception:
            raise MossUnavailableError("MOSS synthesis failed") from None
        elapsed = time.perf_counter() - started
        metrics: dict[str, float | int] = {
            "latency_seconds": round(elapsed, 6),
            "duration_seconds": round(duration, 6),
            "rtf": round(elapsed / duration, 6),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
        self.last_metrics = metrics
        return wav, metrics


engine = MossTTSEngine()
_synthesis_lock = asyncio.Lock()


async def _run_blocking(function, /, *args):
    """Drain a synchronous engine call before propagating task cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and not task.cancelled():
            try:
                task.exception()
            except BaseException:
                pass
        raise


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _run_blocking(engine.load)
    try:
        yield
    finally:
        engine.ready = False
        engine.profile_ids = ()
        engine._profiles = {}
        engine._processor = None
        engine._model = None
        if engine._torch is not None and engine._torch.cuda.is_available():
            engine._torch.cuda.empty_cache()
        engine._torch = None


app = FastAPI(title="OmniVoice MOSS-TTS Sidecar", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, _error: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "invalid_request"})


@app.get("/health")
async def health():
    profile_ids, residency = await asyncio.gather(
        asyncio.to_thread(engine.authorized_profile_ids),
        asyncio.to_thread(engine.residency_status),
    )
    ready = (
        engine.ready is True
        and bool(profile_ids)
        and residency["all_resident"] is True
    )
    body = {
        "ready": ready,
        "engine": "moss-tts",
        "modelRevision": MOSS_TTS_MODEL_REVISION,
        "codecRevision": MOSS_TTS_CODEC_REVISION,
        "residency": {
            "modelResident": residency["model_resident"],
            "codecResident": residency["codec_resident"],
            "device": residency["device"],
        },
        "profileIds": list(profile_ids) if ready else [],
    }
    return JSONResponse(content=body, status_code=200 if ready else 503)


@app.post("/v1/synthesize")
async def synthesize(payload: SynthesisPayload):
    try:
        async with _synthesis_lock:
            wav, metrics = await _run_blocking(engine.synthesize, payload)
    except MossRequestError:
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})
    except MossUnavailableError:
        return JSONResponse(status_code=503, content={"detail": "tts_unavailable"})
    except Exception:
        return JSONResponse(status_code=500, content={"detail": "internal_error"})
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-OmniVoice-TTS-Engine": "moss-tts",
            "X-OmniVoice-TTS-Latency-Ms": str(int(float(metrics["latency_seconds"]) * 1_000)),
            "X-OmniVoice-TTS-RTF": str(metrics["rtf"]),
            "X-OmniVoice-TTS-Peak-VRAM-Bytes": str(metrics["peak_vram_bytes"]),
        },
    )
