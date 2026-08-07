import asyncio
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import edge_tts
import httpx

from src.text_preprocessing import (
    TextPreprocessingError,
    TextPreprocessingUnavailableError,
    prepare_tts_text,
)


logger = logging.getLogger("omnivoice.tts")
logger.setLevel(logging.WARNING)

LOCAL_TTS_SCHEMA_VERSION = 1
VOICE_CATALOG_SCHEMA_VERSION = 1
DEFAULT_LOCAL_TTS_URL = "http://127.0.0.1:9880/v1/synthesize"
DEFAULT_MOSS_TTS_MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
DEFAULT_MOSS_TTS_CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"

_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MOSS_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_EDGE_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-[A-Za-z0-9]+Neural$")
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_VOICE_ENGINES = frozenset({"gpt-sovits", "omnivoice", "moss-tts"})
_VOICE_GENDERS = frozenset({"male", "female", "neutral", "unknown"})
_TTS_METRIC_DECIMAL_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
)
_TTS_METRIC_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_TTS_METRIC_MAX_LATENCY_MS = 3_600_000
_TTS_METRIC_MAX_RTF = 10_000.0
_TTS_METRIC_MAX_PEAK_VRAM_BYTES = 256 * 1024**3
_VOICE_LANGUAGES = frozenset(
    {
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
    }
)
_PROFILE_CONFIG_KEYS = frozenset(
    {
        "id",
        "displayName",
        "engine",
        "languages",
        "gender",
        "styles",
        "previewSupported",
        "referenceRequired",
        # Private registry metadata accepted for the sidecar, never sent to clients.
        "voice",
        "reference_id",
        "reference_audio",
        "referenceId",
        "referenceAudio",
        "reference",
    }
)


class TTSError(RuntimeError):
    """Base class for sanitized TTS failures."""


class TTSConfigurationError(TTSError):
    """The local engine or its profile registry is not configured safely."""


class TTSProfileError(TTSError):
    """The requested local voice profile is absent or incompatible."""


class TTSRequestError(TTSProfileError):
    """The sidecar rejected a deterministic synthesis request."""


class TTSUnavailableError(TTSError):
    """The local engine cannot currently serve synthesis requests."""


class TTSCircuitOpenError(TTSUnavailableError):
    """Calls are being rejected while the local engine circuit is open."""


class TTSTransientError(TTSUnavailableError):
    """A bounded set of retries could not reach a healthy local engine."""


class TTSResponseError(TTSError):
    """The local engine returned an invalid or unsafe response."""


class _RetryableLocalFailure(Exception):
    pass


async def _prepare_private_tts_text(
    text: str, language: str | None
) -> str:
    """Create the private spoken copy and preserve the public TTS error contract."""

    try:
        return await asyncio.to_thread(
            prepare_tts_text,
            text,
            language,
            ensure_terminal=True,
        )
    except TextPreprocessingUnavailableError:
        raise TTSUnavailableError(
            "required text normalization is unavailable"
        ) from None
    except (TextPreprocessingError, TypeError):
        raise TTSRequestError("TTS text normalization rejected the input") from None


@dataclass(frozen=True)
class LocalTTSProfile:
    profile_id: str
    display_name: str
    engine: str
    languages: tuple[str, ...]
    gender: str
    styles: tuple[str, ...]
    preview_supported: bool
    reference_required: bool
    internal_voice: str | None
    internal_reference: str | None

    def catalog_entry(self) -> dict[str, Any]:
        """Return only client-safe metadata; never expose a reference path or endpoint."""
        return {
            "id": self.profile_id,
            "displayName": self.display_name,
            "engine": self.engine,
            "languages": list(self.languages),
            "gender": self.gender,
            "styles": list(self.styles),
            "previewSupported": self.preview_supported,
            "referenceRequired": self.reference_required,
        }


def _cloud_tts_allowed() -> bool:
    """Edge-TTS is a development-only, explicit cloud egress opt-in."""
    return os.environ.get("OMNIVOICE_ALLOW_CLOUD_TTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def local_tts_required() -> bool:
    """Return the production TTS policy, failing closed on unknown values."""
    value = os.environ.get("LOCAL_TTS_REQUIRED", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _clean_profile_string(value: Any, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 for ch in cleaned):
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    return cleaned


def _string_list(
    value: Any,
    *,
    maximum_items: int,
    maximum_length: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    if (not values and not allow_empty) or len(values) > maximum_items:
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        normalized = _clean_profile_string(item, maximum=maximum_length)
        key = normalized.casefold()
        if key not in seen:
            cleaned.append(normalized)
            seen.add(key)
    return tuple(cleaned)


def _profile_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and "profiles" in value:
        rows = value.get("profiles")
    elif isinstance(value, dict):
        rows = []
        for profile_id, metadata in value.items():
            if not isinstance(metadata, dict):
                raise TTSConfigurationError("local TTS profile configuration is invalid")
            row = dict(metadata)
            if "id" in row and row["id"] != profile_id:
                raise TTSConfigurationError("local TTS profile configuration is invalid")
            row["id"] = profile_id
            rows.append(row)
    else:
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    if not isinstance(rows, list) or len(rows) > 128:
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    if not all(isinstance(row, dict) for row in rows):
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    return rows


def _parse_local_profiles(raw: str | None = None) -> dict[str, LocalTTSProfile]:
    raw = os.environ.get("LOCAL_TTS_PROFILES_JSON", "") if raw is None else raw
    if not raw.strip():
        return {}
    if len(raw) > 64 * 1024:
        raise TTSConfigurationError("local TTS profile configuration is invalid")
    try:
        document = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        # JSON parser errors can echo the original environment value. Do not retain a
        # cause that an outer debug traceback could use to disclose profile metadata.
        raise TTSConfigurationError(
            "local TTS profile configuration is invalid"
        ) from None

    profiles: dict[str, LocalTTSProfile] = {}
    for row in _profile_rows(document):
        if set(row) - _PROFILE_CONFIG_KEYS:
            raise TTSConfigurationError("local TTS profile configuration is invalid")
        profile_id = _clean_profile_string(row.get("id"), maximum=128)
        if not _PROFILE_ID_RE.fullmatch(profile_id) or profile_id in profiles:
            raise TTSConfigurationError("local TTS profile configuration is invalid")
        display_name = _clean_profile_string(row.get("displayName"), maximum=128)
        engine = _clean_profile_string(row.get("engine"), maximum=64).casefold()
        if engine not in _VOICE_ENGINES:
            raise TTSConfigurationError("local TTS profile configuration is invalid")

        languages_raw = row.get("languages")
        if not isinstance(languages_raw, list):
            raise TTSConfigurationError("local TTS profile configuration is invalid")
        languages = _string_list(
            languages_raw, maximum_items=32, maximum_length=48
        )
        if any(language not in _VOICE_LANGUAGES for language in languages):
            raise TTSConfigurationError("local TTS profile configuration is invalid")

        gender = _clean_profile_string(row.get("gender"), maximum=32).casefold()
        if gender not in _VOICE_GENDERS:
            raise TTSConfigurationError("local TTS profile configuration is invalid")

        styles_raw = row.get("styles")
        if not isinstance(styles_raw, list):
            raise TTSConfigurationError("local TTS profile configuration is invalid")
        styles = _string_list(
            styles_raw,
            maximum_items=32,
            maximum_length=48,
            allow_empty=True,
        )

        preview_supported = row.get("previewSupported")
        reference_required = row.get("referenceRequired")
        if not isinstance(preview_supported, bool) or not isinstance(
            reference_required, bool
        ):
            raise TTSConfigurationError("local TTS profile configuration is invalid")

        internal_voice_raw = row.get("voice")
        internal_voice = (
            _clean_profile_string(internal_voice_raw, maximum=128)
            if internal_voice_raw is not None
            else None
        )
        reference_values = [
            row[key]
            for key in (
                "reference_id",
                "reference_audio",
                "referenceId",
                "referenceAudio",
                "reference",
            )
            if key in row
        ]
        if len(reference_values) > 1:
            raise TTSConfigurationError("local TTS profile configuration is invalid")
        internal_reference = (
            _clean_profile_string(reference_values[0], maximum=2048)
            if reference_values
            else None
        )
        profiles[profile_id] = LocalTTSProfile(
            profile_id=profile_id,
            display_name=display_name,
            engine=engine,
            languages=languages,
            gender=gender,
            styles=styles,
            preview_supported=preview_supported,
            reference_required=reference_required,
            internal_voice=internal_voice,
            internal_reference=internal_reference,
        )
    return profiles


def _local_host_allowed(hostname: str) -> bool:
    host = hostname.rstrip(".").casefold()
    configured = {
        item.strip().rstrip(".").casefold()
        for item in os.environ.get("LOCAL_TTS_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    }
    if host in configured:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return (
            "." not in host
            or host == "localhost"
            or host.endswith((".localhost", ".local", ".internal", ".svc"))
            or host.endswith(".svc.cluster.local")
        )
    return address.is_loopback or address.is_private or address.is_link_local


def _validate_local_url(endpoint: str, *, error_message: str) -> str:
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        # urlsplit errors can contain the endpoint; keep the typed error opaque.
        raise TTSConfigurationError(error_message) from None
    if (
        len(endpoint) > 2048
        or parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not _local_host_allowed(parsed.hostname)
    ):
        raise TTSConfigurationError(error_message)
    return endpoint


def _local_endpoint() -> str:
    endpoint = os.environ.get("LOCAL_TTS_URL", DEFAULT_LOCAL_TTS_URL).strip()
    return _validate_local_url(
        endpoint, error_message="local TTS endpoint configuration is invalid"
    )


def _local_health_endpoint() -> str:
    configured = os.environ.get("LOCAL_TTS_HEALTH_URL", "").strip()
    if configured:
        endpoint = configured
    else:
        synthesis = urlsplit(_local_endpoint())
        endpoint = urlunsplit(
            (synthesis.scheme, synthesis.netloc, "/health", "", "")
        )
    return _validate_local_url(
        endpoint, error_message="local TTS health endpoint configuration is invalid"
    )


def _default_profile_id(profiles: dict[str, LocalTTSProfile]) -> str | None:
    configured = os.environ.get("LOCAL_TTS_DEFAULT_PROFILE_ID", "").strip()
    if not profiles:
        return None
    if not configured:
        return sorted(profiles)[0]
    profile_id = _clean_profile_string(configured, maximum=128)
    if not _PROFILE_ID_RE.fullmatch(profile_id) or profile_id not in profiles:
        raise TTSConfigurationError("local TTS default profile configuration is invalid")
    return profile_id


def _catalog_revision(
    profiles: list[dict[str, Any]], default_profile_id: str | None
) -> str:
    canonical = json.dumps(
        {
            "schema_version": VOICE_CATALOG_SCHEMA_VERSION,
            "localOnly": True,
            "defaultProfileId": default_profile_id,
            "profiles": profiles,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _expected_moss_revisions() -> tuple[str, str]:
    revisions = (
        os.environ.get(
            "MOSS_TTS_MODEL_REVISION", DEFAULT_MOSS_TTS_MODEL_REVISION
        ).strip(),
        os.environ.get(
            "MOSS_TTS_CODEC_REVISION", DEFAULT_MOSS_TTS_CODEC_REVISION
        ).strip(),
    )
    if not all(_MOSS_REVISION_RE.fullmatch(revision) for revision in revisions):
        raise TTSConfigurationError("local MOSS revision configuration is invalid")
    return revisions


def _moss_health_contract_reason(
    document: Any,
    configured_profile_ids: set[str],
    expected_model_revision: str,
    expected_codec_revision: str,
) -> str | None:
    if not isinstance(document, dict) or document.get("ready") is not True:
        return "health_probe_failed"

    advertised = document.get("profileIds")
    if (
        not isinstance(advertised, list)
        or not all(
            isinstance(item, str) and _PROFILE_ID_RE.fullmatch(item)
            for item in advertised
        )
        or len(advertised) != len(set(advertised))
        or set(advertised) != configured_profile_ids
    ):
        return "health_profile_mismatch"

    residency = document.get("residency")
    if (
        document.get("engine") != "moss-tts"
        or document.get("modelRevision") != expected_model_revision
        or document.get("codecRevision") != expected_codec_revision
        or not isinstance(residency, dict)
        or residency.get("modelResident") is not True
        or residency.get("codecResident") is not True
        or residency.get("device") != "cuda"
    ):
        return "health_contract_mismatch"
    return None


def _validate_nonempty_wav(payload: bytes) -> None:
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise TTSResponseError("local TTS returned invalid WAV audio")
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            if (
                wav_file.getnchannels() <= 0
                or wav_file.getsampwidth() <= 0
                or wav_file.getframerate() <= 0
                or wav_file.getnframes() <= 0
                or not wav_file.readframes(1)
            ):
                raise TTSResponseError("local TTS returned empty WAV audio")
    except (EOFError, wave.Error):
        raise TTSResponseError("local TTS returned invalid WAV audio") from None


def _parse_local_tts_metrics(headers: httpx.Headers) -> dict[str, Any] | None:
    """Return a complete, bounded MOSS metric sample or reject a malformed one.

    Response headers are still untrusted input even though the sidecar is local. A
    partial sample would produce misleading aggregates, so every field is required.
    A response with no metric headers remains valid for other local TTS engines.
    """
    names = (
        "x-omnivoice-tts-engine",
        "x-omnivoice-tts-latency-ms",
        "x-omnivoice-tts-rtf",
        "x-omnivoice-tts-peak-vram-bytes",
    )
    present = [headers.get(name) is not None for name in names]
    if not any(present):
        return None
    if not all(present) or headers.get("x-omnivoice-tts-engine") != "moss-tts":
        raise TTSResponseError("local TTS returned invalid metrics")

    latency_raw = headers.get("x-omnivoice-tts-latency-ms")
    rtf_raw = headers.get("x-omnivoice-tts-rtf")
    peak_vram_raw = headers.get("x-omnivoice-tts-peak-vram-bytes")
    if not all(isinstance(value, str) and 0 < len(value) <= 32 for value in (
        latency_raw,
        rtf_raw,
        peak_vram_raw,
    )):
        raise TTSResponseError("local TTS returned invalid metrics")
    if not _TTS_METRIC_INTEGER_RE.fullmatch(latency_raw):
        raise TTSResponseError("local TTS returned invalid metrics")
    if not _TTS_METRIC_DECIMAL_RE.fullmatch(rtf_raw):
        raise TTSResponseError("local TTS returned invalid metrics")
    if not _TTS_METRIC_INTEGER_RE.fullmatch(peak_vram_raw):
        raise TTSResponseError("local TTS returned invalid metrics")

    try:
        latency_ms = int(latency_raw)
        rtf = float(rtf_raw)
        peak_vram_bytes = int(peak_vram_raw)
    except (TypeError, ValueError, OverflowError):
        raise TTSResponseError("local TTS returned invalid metrics") from None
    if (
        not 0 < latency_ms <= _TTS_METRIC_MAX_LATENCY_MS
        or not math.isfinite(rtf)
        or not 0.0 < rtf <= _TTS_METRIC_MAX_RTF
        or not 0 < peak_vram_bytes <= _TTS_METRIC_MAX_PEAK_VRAM_BYTES
    ):
        raise TTSResponseError("local TTS returned invalid metrics")
    return {
        "engine": "moss-tts",
        "latency_ms": latency_ms,
        "rtf": rtf,
        "peak_vram_bytes": peak_vram_bytes,
    }


class TTSService:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None):
        self.temp_dir = tempfile.gettempdir()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._probe_cache_key: str | None = None
        self._probe_cache_until = 0.0
        self._probe_cache_ready = False
        self._probe_cache_reason: str | None = None

    def catalog(self) -> list[dict[str, Any]]:
        """Return a deterministic, sanitized catalog for Gateway/client capability APIs."""
        try:
            profiles = _parse_local_profiles()
        except TTSConfigurationError:
            return []
        return [profiles[key].catalog_entry() for key in sorted(profiles)]

    def voice_catalog(self) -> list[dict[str, Any]]:
        """Explicit alias used by capability integration code."""
        return self.catalog()

    @staticmethod
    def _catalog_payload(
        *,
        revision: str,
        ready: bool,
        reason: str | None,
        default_profile_id: str | None,
        profiles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": VOICE_CATALOG_SCHEMA_VERSION,
            "revision": revision,
            "ready": ready,
            "localOnly": True,
            "reason": None if ready else reason,
            "defaultProfileId": default_profile_id,
            "profiles": profiles,
        }

    async def _probe_local_health(
        self,
        endpoint: str,
        configured_profile_ids: set[str],
        expected_model_revision: str,
        expected_codec_revision: str,
        cache_key: str,
    ) -> tuple[bool, str | None]:
        now = time.monotonic()
        if self._probe_cache_key == cache_key and now < self._probe_cache_until:
            return self._probe_cache_ready, self._probe_cache_reason

        ready = False
        reason: str | None = "health_probe_failed"
        response: httpx.Response | None = None
        try:
            timeout_seconds = _bounded_float(
                "LOCAL_TTS_HEALTH_TIMEOUT_SECONDS", 2.0, 0.1, 5.0
            )
            client = self._client_instance()
            async with asyncio.timeout(timeout_seconds):
                request = client.build_request(
                    "GET", endpoint, headers={"Accept": "application/json"}
                )
                response = await client.send(request, stream=True)
                if 200 <= response.status_code < 300:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > 64 * 1024:
                            raise ValueError("health response too large")
                        body.extend(chunk)
                    document = json.loads(bytes(body))
                    reason = _moss_health_contract_reason(
                        document,
                        configured_profile_ids,
                        expected_model_revision,
                        expected_codec_revision,
                    )
                    ready = reason is None
        except Exception:
            # Network/parser errors can contain endpoint or response details. Collapse every
            # failure into one canonical capability reason and never log the exception.
            ready = False
            reason = "health_probe_failed"
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except Exception:
                    ready = False
                    reason = "health_probe_failed"

        ttl_seconds = _bounded_float(
            "LOCAL_TTS_HEALTH_CACHE_SECONDS", 5.0, 0.1, 30.0
        )
        self._probe_cache_key = cache_key
        self._probe_cache_until = time.monotonic() + ttl_seconds
        self._probe_cache_ready = ready
        self._probe_cache_reason = reason
        return ready, reason

    async def capability_catalog(self, probe: bool = True) -> dict[str, Any]:
        """Return the exact shared VoiceCatalog wire shape with optional live probing."""
        try:
            configured = _parse_local_profiles()
        except TTSConfigurationError:
            return self._catalog_payload(
                revision="unavailable",
                ready=False,
                reason="invalid_profile_configuration",
                default_profile_id=None,
                profiles=[],
            )
        if not configured:
            return self._catalog_payload(
                revision="unavailable",
                ready=False,
                reason="profiles_not_configured",
                default_profile_id=None,
                profiles=[],
            )

        profiles = [configured[key].catalog_entry() for key in sorted(configured)]
        try:
            default_profile_id = _default_profile_id(configured)
        except TTSConfigurationError:
            return self._catalog_payload(
                revision=_catalog_revision(profiles, None),
                ready=False,
                reason="invalid_default_profile_configuration",
                default_profile_id=None,
                profiles=profiles,
            )
        revision = _catalog_revision(profiles, default_profile_id)
        try:
            _local_endpoint()
        except TTSConfigurationError:
            return self._catalog_payload(
                revision=revision,
                ready=False,
                reason="invalid_synthesis_endpoint_configuration",
                default_profile_id=default_profile_id,
                profiles=profiles,
            )
        if self._circuit_is_open():
            return self._catalog_payload(
                revision=revision,
                ready=False,
                reason="circuit_open",
                default_profile_id=default_profile_id,
                profiles=profiles,
            )
        if not probe:
            return self._catalog_payload(
                revision=revision,
                ready=True,
                reason=None,
                default_profile_id=default_profile_id,
                profiles=profiles,
            )

        try:
            health_endpoint = _local_health_endpoint()
        except TTSConfigurationError:
            return self._catalog_payload(
                revision=revision,
                ready=False,
                reason="invalid_health_endpoint_configuration",
                default_profile_id=default_profile_id,
                profiles=profiles,
            )
        try:
            expected_model_revision, expected_codec_revision = (
                _expected_moss_revisions()
            )
        except TTSConfigurationError:
            return self._catalog_payload(
                revision=revision,
                ready=False,
                reason="invalid_moss_revision_configuration",
                default_profile_id=default_profile_id,
                profiles=profiles,
            )
        ready, reason = await self._probe_local_health(
            health_endpoint,
            set(configured),
            expected_model_revision,
            expected_codec_revision,
            (
                f"{revision}:{health_endpoint}:"
                f"{expected_model_revision}:{expected_codec_revision}"
            ),
        )
        return self._catalog_payload(
            revision=revision,
            ready=ready,
            reason=reason,
            default_profile_id=default_profile_id,
            profiles=profiles,
        )

    def _circuit_is_open(self) -> bool:
        if self._circuit_open_until <= 0:
            return False
        if time.monotonic() >= self._circuit_open_until:
            self._circuit_open_until = 0.0
            self._consecutive_failures = 0
            return False
        return True

    def health_status(self) -> dict[str, Any]:
        """Report configuration/circuit state without endpoint, profile secrets, or errors."""
        try:
            profiles = _parse_local_profiles()
        except TTSConfigurationError:
            return {
                "available": False,
                "mode": "local",
                "profile_count": 0,
                "circuit_open": self._circuit_is_open(),
                "reason": "invalid_profile_configuration",
            }
        try:
            _default_profile_id(profiles)
        except TTSConfigurationError:
            return {
                "available": False,
                "mode": "local",
                "profile_count": len(profiles),
                "circuit_open": self._circuit_is_open(),
                "reason": "invalid_default_profile_configuration",
            }
        circuit_open = self._circuit_is_open()
        if not profiles:
            reason = "profiles_not_configured"
        elif circuit_open:
            reason = "circuit_open"
        else:
            reason = ""
        return {
            "available": bool(profiles) and not circuit_open,
            "mode": "local",
            "profile_count": len(profiles),
            "circuit_open": circuit_open,
            "reason": reason,
        }

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        threshold = _bounded_int("LOCAL_TTS_CIRCUIT_FAILURES", 3, 1, 10)
        if self._consecutive_failures >= threshold:
            open_seconds = _bounded_float(
                "LOCAL_TTS_CIRCUIT_OPEN_SECONDS", 30.0, 0.1, 300.0
            )
            self._circuit_open_until = time.monotonic() + open_seconds

    def _client_instance(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout_seconds = _bounded_float(
                "LOCAL_TTS_TIMEOUT_SECONDS", 30.0, 1.0, 60.0
            )
            timeout = httpx.Timeout(
                timeout_seconds,
                connect=min(5.0, timeout_seconds),
                pool=min(5.0, timeout_seconds),
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def _select_language(profile: LocalTTSProfile, language: str | None) -> str:
        requested = profile.languages[0] if language is None else language.strip()
        if not requested:
            raise TTSProfileError("local TTS profile language is invalid")
        lookup = {item.casefold(): item for item in profile.languages}
        selected = lookup.get(requested.casefold())
        if selected is None:
            raise TTSProfileError("local TTS profile does not support the requested language")
        return selected

    @staticmethod
    def _is_local_intent(
        voice: str, profile_id: str | None, profiles: dict[str, LocalTTSProfile]
    ) -> tuple[bool, str]:
        if profile_id is not None:
            return True, profile_id
        if voice.startswith("local:"):
            return True, voice[len("local:") :]
        if voice in profiles:
            return True, voice
        lowered = voice.casefold()
        if lowered.startswith(("gpt-sovits", "omnivoice", "moss-tts")):
            return True, voice
        # Once a local catalog exists, an unknown non-Edge ID is treated as a bad local
        # profile instead of being silently sent to Microsoft's service.
        if profiles and not _EDGE_VOICE_RE.fullmatch(voice):
            return True, voice
        return False, voice

    async def _read_local_response(
        self, endpoint: str, request_body: dict[str, Any]
    ) -> tuple[bytes, dict[str, Any] | None]:
        client = self._client_instance()
        max_bytes = _bounded_int(
            "LOCAL_TTS_MAX_RESPONSE_BYTES", 32 * 1024 * 1024, 1024, 64 * 1024 * 1024
        )
        timeout_seconds = _bounded_float(
            "LOCAL_TTS_TIMEOUT_SECONDS", 30.0, 1.0, 60.0
        )
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                request = client.build_request("POST", endpoint, json=request_body)
                response = await client.send(request, stream=True)
                if response.status_code in _TRANSIENT_HTTP_STATUSES:
                    raise _RetryableLocalFailure()
                if 400 <= response.status_code < 500:
                    raise TTSRequestError("local TTS rejected the synthesis request")
                if response.status_code < 200 or response.status_code >= 300:
                    raise TTSResponseError("local TTS rejected the synthesis request")

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise TTSResponseError("local TTS response exceeds the size limit")
                    except ValueError:
                        pass

                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > max_bytes:
                        raise TTSResponseError("local TTS response exceeds the size limit")
                    result.extend(chunk)
                return bytes(result), _parse_local_tts_metrics(response.headers)
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            raise _RetryableLocalFailure() from exc
        finally:
            if response is not None:
                await response.aclose()

    async def _synthesize_local(
        self,
        text: str,
        profile_id: str,
        language: str | None,
        rate: str,
        volume: str,
        pitch: str,
        emotion: str,
        duration_seconds: float | None,
        metrics_sink: list[dict[str, Any]] | None,
    ) -> str:
        profiles = _parse_local_profiles()
        if not profiles:
            raise TTSUnavailableError("local TTS profiles are not configured")
        profile = profiles.get(profile_id)
        if profile is None:
            raise TTSProfileError("local TTS profile is not available")
        selected_language = self._select_language(profile, language)
        if self._circuit_is_open():
            raise TTSCircuitOpenError("local TTS circuit is open")

        endpoint = _local_endpoint()
        # Normalize exactly once before constructing the immutable request body.
        # Retries below reuse this body and therefore cannot re-run or compound TN.
        text = await _prepare_private_tts_text(text, selected_language)
        request_body = {
            "schema_version": LOCAL_TTS_SCHEMA_VERSION,
            "text": text,
            "language": selected_language,
            "profile_id": profile.profile_id,
            "emotion": emotion,
            "prosody": {"rate": rate, "volume": volume, "pitch": pitch},
            "output": {"format": "wav"},
        }
        if duration_seconds is not None:
            if (
                isinstance(duration_seconds, bool)
                or not isinstance(duration_seconds, (int, float))
                or not math.isfinite(float(duration_seconds))
                or not 0 < float(duration_seconds) <= 300
            ):
                raise TTSRequestError("local TTS duration is invalid")
            request_body["duration_seconds"] = float(duration_seconds)
        retries = _bounded_int("LOCAL_TTS_MAX_RETRIES", 2, 0, 3)
        retry_base = _bounded_float("LOCAL_TTS_RETRY_BASE_SECONDS", 0.1, 0.0, 2.0)

        wav_payload: bytes | None = None
        metric_sample: dict[str, Any] | None = None
        for attempt in range(retries + 1):
            try:
                wav_payload, metric_sample = await self._read_local_response(
                    endpoint, request_body
                )
                break
            except _RetryableLocalFailure:
                if attempt >= retries:
                    self._record_failure()
                    raise TTSTransientError(
                        "local TTS is unavailable after bounded retries"
                    ) from None
                await asyncio.sleep(retry_base * (2**attempt))

        if wav_payload is None:
            self._record_failure()
            raise TTSTransientError("local TTS is unavailable after bounded retries")
        if profile.engine == "moss-tts" and metric_sample is None:
            self._record_failure()
            raise TTSResponseError("local MOSS TTS did not report metrics")
        try:
            _validate_nonempty_wav(wav_payload)
        except TTSResponseError:
            self._record_failure()
            raise

        fd, path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        try:
            with os.fdopen(fd, "wb") as output_file:
                output_file.write(wav_payload)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(path)
            except OSError:
                pass
            self._record_failure()
            raise TTSResponseError("local TTS audio could not be stored") from None
        self._record_success()
        if metrics_sink is not None and metric_sample is not None:
            metrics_sink.append(metric_sample)
        return path

    async def _synthesize_edge(
        self, text: str, voice: str, rate: str, volume: str, pitch: str
    ) -> str:
        if not _cloud_tts_allowed():
            logger.warning("cloud TTS is disabled; no text was sent outside the worker")
            return ""

        fd, path = tempfile.mkstemp(suffix=".mp3", dir=self.temp_dir)
        os.close(fd)
        try:
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, volume=volume, pitch=pitch
            )
            await communicate.save(path)
            return path
        except Exception:
            logger.error("edge TTS synthesis failed")
            try:
                os.remove(path)
            except OSError:
                pass
            return ""

    async def synthesize(
        self,
        text: str,
        voice: str = "vi-VN-HoaiMyNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        *,
        language: str | None = None,
        profile_id: str | None = None,
        emotion: str = "NEUTRAL",
        duration_seconds: float | None = None,
        metrics_sink: list[dict[str, Any]] | None = None,
    ) -> str:
        """Synthesize with a configured local profile or the explicit dev Edge path.

        Existing callers may continue passing ``text, voice, rate, volume, pitch``. A
        configured profile ID (or explicit ``profile_id``/``local:<id>``) selects the local
        JSON/POST adapter. Local errors are typed and never fall back to Edge.
        """
        if not isinstance(text, str) or not isinstance(voice, str):
            raise TTSConfigurationError("TTS text and voice must be strings")
        if profile_id is not None and not isinstance(profile_id, str):
            raise TTSProfileError("local TTS profile id must be a string")
        if metrics_sink is not None and not isinstance(metrics_sink, list):
            raise TTSConfigurationError("local TTS metrics sink must be a list")
        if not text:
            return ""

        try:
            profiles = _parse_local_profiles()
        except TTSConfigurationError:
            profiles = {}
        local_intent, selected_profile = self._is_local_intent(
            voice, profile_id, profiles
        )
        if local_intent:
            return await self._synthesize_local(
                text,
                selected_profile,
                language,
                rate,
                volume,
                pitch,
                emotion,
                duration_seconds,
                metrics_sink,
            )
        if local_tts_required():
            raise TTSUnavailableError("local TTS is required")
        text = await _prepare_private_tts_text(text, language or voice)
        return await self._synthesize_edge(text, voice, rate, volume, pitch)


tts_service = TTSService()
