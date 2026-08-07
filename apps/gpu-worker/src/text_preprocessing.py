"""Multilingual text normalization at the private TTS boundary.

The approved subtitle remains the source of truth.  This module creates a
separate spoken copy with NVIDIA NeMo Text Processing before synthesis.  NeMo
is loaded lazily and cached per language because constructing its WFST grammar
is CPU-heavy.  Production can require and prewarm every advertised language;
development platforms unsupported by Pynini retain an explicit deterministic
fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Callable, Protocol

from src.translation_quality import normalize_translation_text
from src.vi_tts_adapter import prepare_vietnamese_tts


logger = logging.getLogger("omnivoice.text_preprocessing")
logger.setLevel(logging.WARNING)

NEMO_TEXT_PROCESSING_VERSION = "1.2.0"
DEFAULT_PRELOAD_LANGUAGES = (
    "vi",
    "en",
    "ja",
    "ko",
    "zh",
    "fr",
    "es",
    "de",
    "pt",
    "it",
)
DEFAULT_MAX_INPUT_CHARS = 2_000
DEFAULT_MAX_INPUT_WORDS = 500
DEFAULT_MAX_OUTPUT_CHARS = 2_000

_LANGUAGE_ALIASES = {
    "vi": "vi",
    "vie": "vi",
    "vietnamese": "vi",
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    # Supported by the local voice contract even though the current desktop
    # language picker does not expose them yet.
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "it": "it",
    "ita": "it",
    "italian": "it",
}

# NeMo 1.2.0's Portuguese grammar is Brazilian, and its Chinese grammar must
# never be selected for a Hong Kong/Cantonese locale through a base-tag guess.
_REGIONAL_LANGUAGE_ALIASES = {
    "pt-br": "pt",
    "pt-latn-br": "pt",
    "por-br": "pt",
    "por-latn-br": "pt",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hans-cn": "zh",
    "zho-cn": "zh",
    "zho-hans": "zh",
    "zho-hans-cn": "zh",
    "chi-cn": "zh",
    "chi-hans": "zh",
    "chi-hans-cn": "zh",
}
_REGION_SENSITIVE_BASES = frozenset({"pt", "zh"})


class TextPreprocessingError(RuntimeError):
    """Base class for sanitized text-normalization failures."""


class TextPreprocessingConfigurationError(TextPreprocessingError):
    """The deployment policy for text normalization is invalid."""


class TextPreprocessingUnavailableError(TextPreprocessingError):
    """A required NeMo grammar could not be loaded or applied."""


class _NeMoNormalizer(Protocol):
    def normalize(
        self,
        text: str,
        verbose: bool = False,
        punct_pre_process: bool = False,
        punct_post_process: bool = False,
    ) -> str: ...


_NormalizerFactory = Callable[[str, str], _NeMoNormalizer]

_GRAMMAR_GOLDENS = {
    "vi": "mười hai",
    "en": "twelve",
    "ja": "十二",
    "ko": "십이",
    "zh": "十二",
    "fr": "douze",
    "es": "doce",
    "de": "zwölf",
    "pt": "doze",
    "it": "dodici",
}


@dataclass(frozen=True)
class TTSNormalizationResult:
    """A spoken copy plus count-only audit metadata."""

    subtitle_text: str
    tts_text: str
    language_code: str | None
    backend: str
    changed: bool


def language_code(language: str | None) -> str | None:
    """Resolve app language names and common ISO aliases to a NeMo TN code."""

    key = (language or "").strip().casefold().replace("_", "-")
    if not key:
        return None
    exact = _LANGUAGE_ALIASES.get(key) or _REGIONAL_LANGUAGE_ALIASES.get(key)
    if exact is not None:
        return exact
    base = key.split("-", 1)[0]
    canonical_base = _LANGUAGE_ALIASES.get(base)
    if canonical_base in _REGION_SENSITIVE_BASES:
        return None
    return canonical_base


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise TextPreprocessingConfigurationError(
        f"{name} must be an explicit boolean value"
    )


def _silence_nemo_logging() -> None:
    for logger_name in ("nemo_text_processing", "NeMo-text-processing"):
        upstream_logger = logging.getLogger(logger_name)
        upstream_logger.disabled = True
        upstream_logger.propagate = False


def _contains_control_character(text: str) -> bool:
    """Reject C0/C1 bytes that have no valid role in a TTS request."""

    return any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in text)


def _looks_like_pynini_failure(source: str, normalized: str) -> bool:
    """Detect NeMo's documented fail-open return of ``pynini.escape(text)``."""

    try:
        import pynini
        from nemo_text_processing.text_normalization.normalize import pre_process

        escaped_source = pynini.escape(pre_process(source).strip())
        if normalized == escaped_source:
            return normalized != source or any(
                char.isdecimal() for char in source
            )
    except (ImportError, ModuleNotFoundError):
        # Unit tests and unsupported development platforms may inject a fake
        # normalizer without installing Pynini. Production takes the exact path.
        pass

    if normalized == source:
        return any(char.isdecimal() for char in source)
    unescaped = normalized.replace("\\\\", "\\")
    unescaped = unescaped.replace("\\[", "[").replace("\\]", "]")
    return unescaped == source


@contextmanager
def _exclusive_cache_lock(cache_root: Path, code: str):
    """Serialize FAR creation across worker processes sharing one volume."""

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f".{code}.lock"
    with lock_path.open("a+b") as lock_file:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - NeMo pip is unsupported on Windows.
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _safe_remove_cache_dir(path: Path, cache_root: Path) -> None:
    resolved_root = cache_root.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_root or path.is_symlink():
        raise TextPreprocessingConfigurationError(
            "NeMo cache layout is unsafe"
        )
    if path.exists():
        shutil.rmtree(path)


def _validate_grammar(normalizer: _NeMoNormalizer, code: str) -> None:
    expected = _GRAMMAR_GOLDENS.get(code)
    if expected is None:
        raise TextPreprocessingConfigurationError(
            "NeMo grammar has no product validation sentinel"
        )
    actual = normalizer.normalize(
        "12",
        verbose=False,
        punct_pre_process=False,
        punct_post_process=False,
    )
    if actual != expected:
        raise TextPreprocessingUnavailableError(
            f"NVIDIA NeMo grammar validation failed for language '{code}'"
        )


def _default_factory(code: str, cache_dir: str) -> _NeMoNormalizer:
    # NeMo may log the complete failed input at WARNING.  The worker's
    # zero-logging contract must suppress that upstream message before any
    # tenant text is processed.
    _silence_nemo_logging()
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "nemo_import_failed language=%s error_type=%s",
            code,
            type(exc).__name__,
        )
        raise TextPreprocessingUnavailableError(
            "NVIDIA NeMo Text Processing is not installed"
        ) from None

    # Importing NeMo creates its mixed-case logger; enforce suppression again
    # before grammar construction or any tenant-text rewrite.
    _silence_nemo_logging()

    root = Path(cache_dir)
    marker_value = (
        f"nemo-text-processing={NEMO_TEXT_PROCESSING_VERSION}\n"
        f"language={code}\ninput_case=cased\ndeterministic=true\n"
    )

    def construct(path: Path) -> _NeMoNormalizer:
        normalizer = Normalizer(
            input_case="cased",
            lang=code,
            deterministic=True,
            cache_dir=str(path),
            overwrite_cache=False,
            post_process=True,
        )
        _validate_grammar(normalizer, code)
        return normalizer

    try:
        with _exclusive_cache_lock(root, code):
            final_dir = root / code
            marker = final_dir / ".ready"
            marker_matches = (
                marker.is_file()
                and marker.read_text(encoding="utf-8") == marker_value
            )
            if marker_matches:
                try:
                    return construct(final_dir)
                except Exception:
                    # A killed process or an unsafe shared-volume writer may
                    # leave a syntactically readable but semantically corrupt FAR.
                    _safe_remove_cache_dir(final_dir, root)
            elif final_dir.exists():
                _safe_remove_cache_dir(final_dir, root)

            for stale_dir in root.glob(f".{code}-building-*"):
                if stale_dir.is_dir():
                    _safe_remove_cache_dir(stale_dir, root)

            build_dir = Path(
                tempfile.mkdtemp(prefix=f".{code}-building-", dir=str(root))
            )
            try:
                normalizer = construct(build_dir)
                (build_dir / ".ready").write_text(
                    marker_value, encoding="utf-8", newline="\n"
                )
                os.replace(build_dir, final_dir)
                return normalizer
            except Exception:
                _safe_remove_cache_dir(build_dir, root)
                raise
    except TextPreprocessingConfigurationError:
        raise
    except Exception as exc:
        # Pynini exceptions can contain cache paths, grammar internals, or the
        # validation input. Keep the public failure stable and sanitized. The
        # exception class is bounded diagnostic metadata and never includes the
        # input, cache path, model URL, or secret-bearing exception message.
        logger.warning(
            "nemo_grammar_build_failed language=%s error_type=%s",
            code,
            type(exc).__name__,
        )
        raise TextPreprocessingUnavailableError(
            f"NVIDIA NeMo grammar is unavailable for language '{code}'"
        ) from None


class TextPreprocessingService:
    """Thread-safe NeMo TN registry used immediately before local TTS."""

    def __init__(
        self,
        *,
        normalizer_factory: _NormalizerFactory | None = None,
        enabled: bool | None = None,
        required: bool | None = None,
        cache_dir: str | None = None,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_input_words: int = DEFAULT_MAX_INPUT_WORDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.enabled = (
            _env_flag("NEMO_TEXT_PROCESSING_ENABLED", True)
            if enabled is None
            else bool(enabled)
        )
        self.required = (
            _env_flag("NEMO_TEXT_PROCESSING_REQUIRED", False)
            if required is None
            else bool(required)
        )
        if self.required and not self.enabled:
            raise TextPreprocessingConfigurationError(
                "required NeMo text processing cannot be disabled"
            )
        limits = (max_input_chars, max_input_words, max_output_chars)
        if any(isinstance(value, bool) or value < 1 for value in limits):
            raise TextPreprocessingConfigurationError(
                "text-normalization limits must be positive"
            )
        cache_root = cache_dir or os.environ.get(
            "NEMO_TEXT_PROCESSING_CACHE_DIR", "/tmp/omnivoice-nemo-tn"
        )
        self.cache_dir = str(
            Path(cache_root)
            / f"v{NEMO_TEXT_PROCESSING_VERSION}-cased-deterministic"
        )
        self.max_input_chars = int(max_input_chars)
        self.max_input_words = int(max_input_words)
        self.max_output_chars = int(max_output_chars)
        self._factory = normalizer_factory or _default_factory
        self._normalizers: dict[str, _NeMoNormalizer] = {}
        self._unavailable: set[str] = set()
        self._warned: set[str] = set()
        self._lock = threading.RLock()

    def _warn_fallback_once(self, code: str, reason: str) -> None:
        marker = f"{code}:{reason}"
        with self._lock:
            if marker in self._warned:
                return
            self._warned.add(marker)
        logger.warning(
            "text_normalization_fallback language=%s reason=%s", code, reason
        )

    def validate_runtime_policy(self, *, local_tts_is_required: bool) -> None:
        """Keep production local synthesis coupled to required NeMo TN."""

        if local_tts_is_required and not self.required:
            raise TextPreprocessingConfigurationError(
                "required local TTS requires required NeMo text processing"
            )

    def _get_normalizer(self, code: str) -> _NeMoNormalizer | None:
        if not self.enabled:
            return None
        with self._lock:
            cached = self._normalizers.get(code)
            if cached is not None:
                return cached
            if code in self._unavailable:
                if self.required:
                    raise TextPreprocessingUnavailableError(
                        f"required NeMo grammar is unavailable for language '{code}'"
                    )
                return None
            try:
                normalizer = self._factory(code, self.cache_dir)
                if not callable(getattr(normalizer, "normalize", None)):
                    raise TypeError("invalid NeMo normalizer")
            except Exception as exc:
                logger.warning(
                    "nemo_normalizer_unavailable language=%s error_type=%s",
                    code,
                    type(exc).__name__,
                )
                self._unavailable.add(code)
                if self.required:
                    raise TextPreprocessingUnavailableError(
                        f"required NeMo grammar is unavailable for language '{code}'"
                    ) from None
                self._warn_fallback_once(code, "backend_unavailable")
                return None
            self._normalizers[code] = normalizer
            return normalizer

    @staticmethod
    def _fallback(text: str, code: str | None, ensure_terminal: bool) -> str:
        if code == "vi":
            return prepare_vietnamese_tts(
                text, ensure_terminal=ensure_terminal
            ).tts_text
        return normalize_translation_text(text, ensure_terminal=ensure_terminal)

    def prepare(
        self,
        text: str,
        target_language: str | None,
        *,
        ensure_terminal: bool = True,
    ) -> TTSNormalizationResult:
        if not isinstance(text, str):
            raise TypeError("TTS text must be a string")
        if len(text) > self.max_input_chars:
            raise TextPreprocessingError(
                "TTS text exceeds the normalization input limit"
            )

        subtitle = normalize_translation_text(text, ensure_terminal=False)
        if len(subtitle.split()) > self.max_input_words:
            raise TextPreprocessingError(
                "TTS text exceeds the normalization word limit"
            )
        code = language_code(target_language)
        if not subtitle:
            return TTSNormalizationResult("", "", code, "empty", False)

        normalizer = self._get_normalizer(code) if code is not None else None
        backend = "nemo" if normalizer is not None else "deterministic_fallback"
        if normalizer is None:
            if code is None and target_language and target_language.strip():
                self._warn_fallback_once("unmapped", "language_unsupported")
            if code is not None and self.required:
                raise TextPreprocessingUnavailableError(
                    f"required NeMo grammar is unavailable for language '{code}'"
                )
            spoken = self._fallback(subtitle, code, ensure_terminal)
        else:
            try:
                # One NeMo instance is shared by all requests for a language.
                # Pynini rewrite/parser objects are not documented as re-entrant.
                with self._lock:
                    spoken = normalizer.normalize(
                        subtitle,
                        verbose=False,
                        punct_pre_process=True,
                        punct_post_process=True,
                    )
                if not isinstance(spoken, str) or not spoken.strip():
                    raise ValueError("empty text-normalization output")
                if _contains_control_character(spoken):
                    raise TextPreprocessingError(
                        "normalized TTS text contains unsupported control characters"
                    )
                if _looks_like_pynini_failure(subtitle, spoken):
                    if self.required:
                        raise TextPreprocessingError(
                            "NVIDIA NeMo could not normalize the TTS input"
                        )
                    raise ValueError("NeMo returned an escaped fallback")
                # Preserve the existing project-specific Vietnamese acronym and
                # identifier vocabulary after NeMo expands general semiotic classes.
                if code == "vi":
                    spoken = prepare_vietnamese_tts(
                        spoken, ensure_terminal=False
                    ).tts_text
                spoken = normalize_translation_text(
                    spoken, ensure_terminal=ensure_terminal
                )
                if any(char.isdecimal() for char in spoken):
                    if self.required:
                        raise TextPreprocessingError(
                            "NVIDIA NeMo left unresolved numeric TTS input"
                        )
                    raise ValueError("unresolved NeMo normalization output")
            except TextPreprocessingError:
                raise
            except Exception:
                if self.required:
                    raise TextPreprocessingError(
                        f"NVIDIA NeMo rejected TTS input for language '{code}'"
                    ) from None
                self._warn_fallback_once(code, "normalization_failed")
                backend = "deterministic_fallback"
                spoken = self._fallback(subtitle, code, ensure_terminal)

        if _contains_control_character(spoken):
            raise TextPreprocessingError(
                "normalized TTS text contains unsupported control characters"
            )
        if len(spoken) > self.max_output_chars:
            raise TextPreprocessingError(
                "normalized TTS text exceeds the synthesis input limit"
            )
        return TTSNormalizationResult(
            subtitle_text=subtitle,
            tts_text=spoken,
            language_code=code,
            backend=backend,
            changed=spoken != subtitle,
        )

    def prewarm(self, languages: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Build/load each configured grammar without processing tenant text."""

        loaded: list[str] = []
        for language in languages:
            code = language_code(language)
            if code is None or code in loaded:
                continue
            if self._get_normalizer(code) is not None:
                loaded.append(code)
        return tuple(loaded)

    def prewarm_configured(self) -> tuple[str, ...]:
        raw = os.environ.get(
            "NEMO_TEXT_PROCESSING_PRELOAD_LANGUAGES",
            ",".join(DEFAULT_PRELOAD_LANGUAGES),
        )
        requested = [item.strip() for item in raw.split(",") if item.strip()]
        unknown = [item for item in requested if language_code(item) is None]
        if unknown:
            raise TextPreprocessingConfigurationError(
                "NeMo preload language list contains unsupported entries"
            )
        requested_codes = {
            code for item in requested if (code := language_code(item)) is not None
        }
        if self.required and not set(DEFAULT_PRELOAD_LANGUAGES).issubset(
            requested_codes
        ):
            raise TextPreprocessingConfigurationError(
                "required NeMo preload list must cover every supported NeMo language"
            )
        return self.prewarm(requested)


text_preprocessing_service = TextPreprocessingService()


def prepare_tts_text(
    text: str,
    target_language: str | None,
    *,
    ensure_terminal: bool = True,
) -> str:
    """Return only the private spoken copy expected by ModelManager."""

    return text_preprocessing_service.prepare(
        text, target_language, ensure_terminal=ensure_terminal
    ).tts_text


__all__ = [
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_MAX_INPUT_WORDS",
    "DEFAULT_PRELOAD_LANGUAGES",
    "NEMO_TEXT_PROCESSING_VERSION",
    "TTSNormalizationResult",
    "TextPreprocessingConfigurationError",
    "TextPreprocessingError",
    "TextPreprocessingService",
    "TextPreprocessingUnavailableError",
    "language_code",
    "prepare_tts_text",
    "text_preprocessing_service",
]
