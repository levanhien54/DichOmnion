"""Vietnamese text preparation for the local TTS boundary.

The translation shown to an editor and the text sent to a voice engine have
different requirements.  ``translation_quality`` owns the canonical subtitle
surface and its pronunciation hints; this adapter renders a *spoken* copy
without changing the approved subtitle.  It is deliberately deterministic and
model-free so an engine cannot silently invent pronunciations.

The multilingual TTS boundary may use this adapter as a Vietnamese fallback or
domain-specific post-processor. The approved ``translatedText`` remains
byte-for-byte intact because only a private spoken copy is transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.translation_quality import (
    PronunciationPlan,
    build_pronunciation_plan,
    normalize_translation_text,
)


_VI_LANGUAGE_KEYS = frozenset({"vi", "vie", "vietnamese"})

# Longer units must be listed first (``km/h`` before ``m``).  These replacements
# are intentionally conservative: a unit is expanded only when it immediately
# follows a numeric value, so ordinary words such as ``m`` in a name are safe.
_UNIT_RE = re.compile(
    r"(?<![\w])(?P<number>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>km/h|km²|m²|cm|mm|km|kg|mg|ml|gb|mb|kb|hz|khz|°c|°f|g|m|l|s|h)"
    r"(?![\w])",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?<![\w])(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{2,4})(?![\w])"
)
_TIME_RE = re.compile(
    r"(?<![\w])(?P<hour>\d{1,2}):(?P<minute>\d{2})(?![\w])"
)
_IP_RE = re.compile(r"(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])")
_VERSION_RE = re.compile(r"(?<![\w])v?\d+(?:\.\d+){2,}(?![\w])", re.IGNORECASE)
_LONG_DIGITS_RE = re.compile(r"(?<![\w])\d{7,}(?![\w])")
_ALNUM_ID_RE = re.compile(
    r"(?<![\w])(?=[A-Za-z]*\d)(?:[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*)(?![\w])"
)

_UNIT_WORDS = {
    "km/h": "ki-lô-mét một giờ",
    "km²": "ki-lô-mét vuông",
    "m²": "mét vuông",
    "khz": "ki-lô-héc",
    "hz": "héc",
    "kg": "ki-lô-gam",
    "mg": "mi-li-gam",
    "ml": "mi-li-lít",
    "gb": "gi-ga-bai",
    "mb": "mê-ga-bai",
    "kb": "ki-lô-bai",
    "cm": "xen-ti-mét",
    "mm": "mi-li-mét",
    "km": "ki-lô-mét",
    "°c": "độ xê",
    "°f": "độ ép",
    "g": "gam",
    "m": "mét",
    "l": "lít",
    "s": "giây",
    "h": "giờ",
}

_CURRENCY_REPLACEMENTS = {
    "₫": " đồng ",
    "$": " đô la ",
    "€": " euro ",
    "¥": " yên ",
}

# Common abbreviations in Vietnamese scripts.  Unknown all-capital tokens are
# handled by ``PronunciationHint`` (letter-by-letter) instead of this semantic
# dictionary, which avoids guessing a meaning that is not in context.
_ABBREVIATIONS = {
    "AI": "ây ai",
    "API": "ây pi ai",
    "CPU": "xi pi iu",
    "GPU": "gi pi iu",
    "USB": "iu ét bi",
    "TV": "ti vi",
    "HD": "hát đê",
    "4K": "bốn ka",
    "URL": "u a e lờ",
    "CEO": "xi i ô",
    "VN": "Việt Nam",
    "VND": "đồng Việt Nam",
    "USD": "đô la Mỹ",
    "HCM": "Hồ Chí Minh",
    "TP.HCM": "thành phố Hồ Chí Minh",
    "COVID-19": "cô vít mười chín",
}

# ASR/LLM output occasionally capitalizes a whole sentence for emphasis.  The
# quality module quite correctly records every all-cap token as a possible
# acronym, but spelling ordinary Vietnamese words letter-by-letter would make
# TTS much worse.  Skip this bounded stop-list; genuinely technical acronyms
# still use the explicit map or generic letter hints.
_COMMON_VI_UPPERCASE_WORDS = frozenset(
    {
        "ANH",
        "BẠN",
        "CÁC",
        "CHÀO",
        "CHÚNG",
        "CÔ",
        "CỦA",
        "ĐÃ",
        "ĐANG",
        "ĐÂY",
        "ĐI",
        "ĐƯỢC",
        "EM",
        "GÌ",
        "HỌ",
        "KHÔNG",
        "LÀ",
        "MỘT",
        "NÀY",
        "NGƯỜI",
        "NHỮNG",
        "RA",
        "RẤT",
        "SẼ",
        "TA",
        "TÔI",
        "TRONG",
        "VÀ",
        "VỚI",
        "XIN",
    }
)

_DIGIT_WORDS = (
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
)
_LETTER_WORDS = {
    "A": "a",
    "B": "bê",
    "C": "xê",
    "D": "đê",
    "Đ": "đê",
    "E": "e",
    "F": "ép",
    "G": "giê",
    "H": "hát",
    "I": "i",
    "J": "giây",
    "K": "ca",
    "L": "e lờ",
    "M": "em",
    "N": "en",
    "O": "ô",
    "P": "pê",
    "Q": "quy",
    "R": "a",
    "S": "ét",
    "T": "tê",
    "U": "u",
    "V": "vê",
    "W": "đáp bê liu",
    "X": "ích",
    "Y": "i",
    "Z": "dét",
}


@dataclass(frozen=True)
class VietnameseTTSPreparation:
    """Audit-friendly result of preparing one segment for TTS.

    ``subtitle_text`` is the canonical display form and ``tts_text`` is the
    spoken form.  The source text itself is never logged by this module.
    ``changed`` is metadata only; callers can count it without retaining text.
    """

    subtitle_text: str
    tts_text: str
    pronunciation: PronunciationPlan
    changed: bool


def _language_key(language: str | None) -> str:
    return (language or "").strip().casefold().replace("_", "-").split("-", 1)[0]


def _replace_currency(text: str) -> str:
    for symbol, spoken in _CURRENCY_REPLACEMENTS.items():
        text = text.replace(symbol, spoken)
    return text


def _replace_time_and_date(text: str) -> str:
    # ``12:30`` is much less reliable for Vietnamese TTS than an explicit
    # phrase.  Keep the numbers so the pronunciation plan can spell them.
    text = _TIME_RE.sub(
        lambda m: f"{m.group('hour')} giờ {m.group('minute')}", text
    )
    def date_repl(match: re.Match[str]) -> str:
        # Avoid producing ``ngày ngày`` when the source already labels the date.
        prefix = text[max(0, match.start() - 8) : match.start()]
        day_word = "" if re.search(r"ngày\s*$", prefix, re.IGNORECASE) else "ngày "
        return (
            f"{day_word}{match.group('day')} tháng {match.group('month')} "
            f"năm {match.group('year')}"
        )

    return _DATE_RE.sub(date_repl, text)


def _replace_units(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        unit = match.group("unit").casefold()
        return f"{match.group('number')} {_UNIT_WORDS.get(unit, unit)}"

    return _UNIT_RE.sub(repl, text)


def _spell_digits(value: str) -> str:
    return " ".join(_DIGIT_WORDS[int(char)] for char in value if char.isdigit())


def _replace_identifiers(text: str) -> str:
    """Read IDs/IPs/versions digit-by-digit instead of as cardinal numbers."""

    def ip_repl(match: re.Match[str]) -> str:
        octets = match.group(0).split(".")
        if any(int(octet) > 255 for octet in octets):
            return match.group(0)
        return " chấm ".join(_spell_digits(octet) for octet in octets)

    text = _IP_RE.sub(ip_repl, text)

    def version_repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        prefix = ""
        if raw[:1].casefold() == "v":
            prefix = "vê "
            raw = raw[1:]
        return prefix + " chấm ".join(_spell_digits(part) for part in raw.split("."))

    text = _VERSION_RE.sub(version_repl, text)
    text = _LONG_DIGITS_RE.sub(lambda match: _spell_digits(match.group(0)), text)

    def id_repl(match: re.Match[str]) -> str:
        spoken: list[str] = []
        for char in match.group(0):
            if char.isdigit():
                spoken.append(_DIGIT_WORDS[int(char)])
            elif char.upper() in _LETTER_WORDS:
                spoken.append(_LETTER_WORDS[char.upper()])
            else:
                spoken.append(char)
        return " ".join(spoken)

    return _ALNUM_ID_RE.sub(id_repl, text)


def _replace_abbreviations(text: str) -> str:
    # Sort by length so ``TP.HCM`` wins before the shorter ``HCM`` alternative.
    for token in sorted(_ABBREVIATIONS, key=len, reverse=True):
        pattern = re.compile(
            # Acronyms are intentionally case-sensitive: Vietnamese ``ai`` /
            # ``Ai`` is an ordinary pronoun and must not become ``ây ai``.
            rf"(?<![\w]){re.escape(token)}(?![\w])"
        )
        text = pattern.sub(_ABBREVIATIONS[token], text)
    return text


def _replace_hints(text: str, plan: PronunciationPlan) -> str:
    """Apply non-destructive numeric/acronym hints to a spoken copy.

    Hints are matched as tokens rather than with ``str.replace`` so a number in
    ``S1`` or a word containing ``AI`` is not accidentally rewritten.  The
    quality module intentionally keeps hints metadata-only; this is the sole
    point where a TTS engine opts into their spoken forms.
    """

    rendered = plan.text
    # Semantic abbreviations were expanded before the plan was built.  Generic
    # acronym hints are still useful for e.g. ``NATO`` or a model name.
    for hint in plan.hints:
        if (
            hint.kind == "acronym"
            and hint.token.casefold().upper() in _COMMON_VI_UPPERCASE_WORDS
        ):
            continue
        pattern = re.compile(
            rf"(?<![\w]){re.escape(hint.token)}(?![\w])"
        )
        rendered = pattern.sub(hint.spoken, rendered)
    return rendered


def _make_plan(text: str, *, ensure_terminal: bool) -> PronunciationPlan:
    """Use the quality module's canonicalizer while keeping its public API stable."""

    normalized = normalize_translation_text(text, ensure_terminal=ensure_terminal)
    plan = build_pronunciation_plan(normalized)
    # Older quality-module revisions always add a terminal period while building
    # the plan.  Keep this adapter backward-compatible until that optional
    # argument is available there; no mark/hint is generated for the synthetic
    # period, so dropping it is lossless.
    if (
        not ensure_terminal
        and normalized
        and normalized[-1].isalnum()
        and plan.text == normalized + "."
    ):
        plan = PronunciationPlan(
            text=normalized,
            marks=tuple(mark for mark in plan.marks if mark.offset < len(normalized)),
            hints=plan.hints,
        )
    return plan


def prepare_vietnamese_tts(
    text: str,
    *,
    ensure_terminal: bool = False,
) -> VietnameseTTSPreparation:
    """Return a canonical subtitle form and a natural spoken Vietnamese form.

    The operation is bounded by the caller's segment limits (the worker already
    rejects overlong text); non-string values are rejected instead of coerced,
    preventing an invalid manifest from becoming a misleading TTS success.
    """

    if not isinstance(text, str):
        raise TypeError("Vietnamese TTS text must be a string")
    if not text.strip():
        empty_plan = _make_plan("", ensure_terminal=False)
        return VietnameseTTSPreparation("", "", empty_plan, False)

    # Build the display form first.  Date/time, units, currency and known
    # abbreviations are only expanded in the private spoken copy below.
    spoken_source = _replace_currency(text)
    spoken_source = _replace_time_and_date(spoken_source)
    spoken_source = _replace_units(spoken_source)
    spoken_source = _replace_abbreviations(spoken_source)
    spoken_source = _replace_identifiers(spoken_source)
    plan = _make_plan(spoken_source, ensure_terminal=ensure_terminal)
    spoken = _replace_hints(plan.text, plan)
    # Re-run only whitespace/punctuation normalization after substitutions;
    # do not add a terminal mark unless the caller explicitly requests it.
    final_plan = _make_plan(spoken, ensure_terminal=ensure_terminal)
    subtitle = _make_plan(text, ensure_terminal=ensure_terminal).text
    return VietnameseTTSPreparation(
        subtitle_text=subtitle,
        tts_text=final_plan.text,
        pronunciation=final_plan,
        changed=final_plan.text != subtitle,
    )


def prepare_tts_text(
    text: str,
    target_language: str | None,
    *,
    ensure_terminal: bool = False,
    normalize_non_vietnamese: bool = False,
) -> str:
    """Prepare a spoken copy without applying Vietnamese rules to other languages.

    Vietnamese receives the full number/unit/acronym adapter.  Other languages
    may opt into the locale-neutral Unicode/punctuation normalizer, but are
    deliberately not passed through Vietnamese pronunciation dictionaries.
    """

    if _language_key(target_language) not in _VI_LANGUAGE_KEYS:
        if not isinstance(text, str):
            raise TypeError("TTS text must be a string")
        return normalize_translation_text(text, ensure_terminal=ensure_terminal) if normalize_non_vietnamese else text
    return prepare_vietnamese_tts(
        text, ensure_terminal=ensure_terminal
    ).tts_text


__all__ = [
    "VietnameseTTSPreparation",
    "prepare_tts_text",
    "prepare_vietnamese_tts",
]
