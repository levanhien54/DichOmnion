"""Deterministic post-translation quality and speech-mark normalization.

The translation model remains the source of meaning. This module is deliberately
model-free: it catches output that is empty, copied from the source, contaminated
by CJK/model artifacts, repetitive, or malformed before text reaches TTS. It also
turns punctuation into stable pause marks so a later TTS adapter can render a
natural prosody plan without guessing from raw text.

No source or target text is logged here. Callers may keep ``QualityReport`` in
memory or discard the normalized text according to their privacy policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from collections import Counter
from typing import Iterable, Literal, Mapping


QualityDecision = Literal["accept", "review", "reject"]

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w])[+-]?\d+(?:[.,]\d+)?%?(?![\w])")
_NUMBER_SCAN_RE = re.compile(r"(?<![\w])[+-]?\d+(?:[.,]\d+)*%?(?![\w])")
# Traditional/Simplified Chinese numerals are handled separately from the
# Arabic semiotic scanner.  The parser is intentionally conservative: a bare
# classifier such as ``一個`` is not treated as an exact entity, while a value
# attached to a measurement/unit (``三噸``, ``一米四``) is.
_CJK_NUMERAL_RE = re.compile(
    r"[0-9零〇○一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+"
)
_CJK_DECIMAL_RE = re.compile(
    r"(?P<whole>[0-9零〇○一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+)"
    r"(?P<marker>米|公尺|點|点)(?P<fraction>[0-9零〇○一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+)"
)
_CJK_RATIO_RE = re.compile(
    r"(?P<left>[0-9零〇○一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+)"
    r"比(?P<right>[0-9零〇○一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+)"
)
_CJK_UNKNOWN_QUANTITY_MARKERS = frozenset({"幾", "几", "數", "数", "多"})
_CJK_MEASURE_UNITS = frozenset(
    {
        "米", "公尺", "公分", "厘米", "公升", "升", "噸", "吨", "斤", "公斤",
        "克", "千克", "公斤", "條", "条", "項", "项", "年", "月", "日", "度",
        "%", "公里", "千米", "公里", "公顷", "公頃",
    }
)
_ACRONYM_RE = re.compile(r"(?<![\w])[A-ZĐ]{2,}(?![\w])")
_PUNCT_RE = re.compile(r"(?:\.{3}|[,.!?;:])\s*")
_ARTIFACT_RE = re.compile(
    r"(?:<unk>|<pad>|<s>|</s>|\[UNK\]|▁|�|\\u[0-9a-fA-F]{4})",
    re.IGNORECASE,
)
_MOJIBAKE_RE = re.compile(
    r"(?:Ã(?:[\u0080-\u00bf]|\s)|Â(?:[\u0080-\u00bf]|\s)|"
    r"â(?:€|™|œ|¦|‚)|ðŸ|ï¿½)"
)
_VI_MARKERS = frozenset(
    {
        "anh",
        "bạn",
        "chúng",
        "cô",
        "của",
        "đã",
        "đang",
        "đây",
        "được",
        "em",
        "hai",
        "không",
        "là",
        "một",
        "người",
        "này",
        "những",
        "ra",
        "rất",
        "sẽ",
        "tôi",
        "trong",
        "và",
        "với",
        "xin",
        "chào",
    }
)
_CJK_LANGUAGE_KEYS = frozenset({
    "zh", "zho", "chi", "chinese", "cmn", "mandarin",
    "ja", "jpn", "japanese", "yue", "cantonese",
})
_CJK_NEGATION_MARKERS = frozenset({"不", "没", "沒有", "没有", "不是", "别", "別", "未"})
_JA_NEGATION_MARKERS = frozenset({"ない", "ません", "ぬ", "ず", "無い", "なし"})
_KO_NEGATION_MARKERS = frozenset({"아니", "않", "못", "없", "말고", "아니다"})
_VI_NEGATION_MARKERS = frozenset({"không", "chưa", "chẳng", "chả", "đừng", "chớ", "chưa phải"})
_VI_NUMBER_WORDS = frozenset(
    {
        "một",
        "hai",
        "ba",
        "bốn",
        "năm",
        "sáu",
        "bảy",
        "tám",
        "chín",
        "mười",
        "mươi",
        "trăm",
        "nghìn",
        "ngàn",
        "triệu",
        "tỷ",
        "phẩy",
    }
)
_VI_NUMERIC_CONTEXT_WORDS = _VI_NUMBER_WORDS | frozenset(
    {"không", "lẻ", "linh", "phần"}
)
_SUSPICIOUS_BOILERPLATE = (
    "đây là bản dịch",
    "tôi không thể giúp",
    "tôi không biết",
    "không thể dịch",
    "vui lòng cung cấp thêm",
    "xin lỗi, tôi",
)
_PROTECTED_RE = re.compile(r"(?:https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|#[\w-]+)")

# Language profiles keep structural checks useful outside the original Chinese ->
# Vietnamese path.  Unknown languages deliberately receive conservative checks
# (empty marker sets) instead of a guessed language verdict.
_LATIN_LANGUAGE_KEYS = frozenset({
    "vi", "vie", "vietnamese", "en", "eng", "english", "fr", "fra", "fre",
    "french", "de", "deu", "ger", "german", "es", "spa", "spanish", "pt",
    "por", "portuguese", "it", "ita", "italian", "nl", "nld", "dutch", "id",
    "ind", "indonesian", "ms", "msa", "malay",
})
_CYRILLIC_LANGUAGE_KEYS = frozenset({"ru", "rus", "russian", "uk", "ukr", "ukrainian"})
_ARABIC_LANGUAGE_KEYS = frozenset({"ar", "ara", "arabic", "fa", "fas", "persian"})
_THAI_LANGUAGE_KEYS = frozenset({"th", "tha", "thai"})
_HANGUL_LANGUAGE_KEYS = frozenset({"ko", "kor", "korean"})

# Speaking-rate profiles are conservative, target-language-specific production
# estimates. Latin profiles use words/second, Vietnamese uses space-delimited
# syllables, and CJK/Korean profiles use readable native-script characters/second.
_SPEECH_RATE_BY_LANGUAGE = {
    "vi": 3.8, "vie": 3.8, "vietnamese": 3.8,
    "en": 2.7, "eng": 2.7, "english": 2.7,
    "fr": 3.0, "fra": 3.0, "fre": 3.0, "french": 3.0,
    "es": 3.0, "spa": 3.0, "spanish": 3.0,
    "de": 2.6, "deu": 2.6, "ger": 2.6, "german": 2.6,
    "pt": 3.0, "por": 3.0, "portuguese": 3.0,
    "it": 3.0, "ita": 3.0, "italian": 3.0,
    "zh": 4.2, "zho": 4.2, "chi": 4.2, "chinese": 4.2,
    "cmn": 4.2, "mandarin": 4.2, "yue": 4.2, "cantonese": 4.2,
    "ja": 4.3, "jpn": 4.3, "japanese": 4.3,
    "ko": 4.0, "kor": 4.0, "korean": 4.0,
}
_CANONICAL_LANGUAGE_KEYS = {
    "vi": "vi", "vie": "vi", "vietnamese": "vi",
    "en": "en", "eng": "en", "english": "en",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "es": "es", "spa": "es", "spanish": "es",
    "de": "de", "deu": "de", "ger": "de", "german": "de",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "it": "it", "ita": "it", "italian": "it",
    "zh": "zh", "zho": "zh", "chi": "zh", "chinese": "zh",
    "cmn": "zh", "mandarin": "zh",
    "yue": "yue", "cantonese": "yue",
    "ja": "ja", "jpn": "ja", "japanese": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko",
}
_NEGATION_MARKERS_BY_LANGUAGE = {
    "vi": _VI_NEGATION_MARKERS,
    "vie": _VI_NEGATION_MARKERS,
    "vietnamese": _VI_NEGATION_MARKERS,
    "en": frozenset({"not", "never", "no", "neither", "nor", "without"}),
    "eng": frozenset({"not", "never", "no", "neither", "nor", "without"}),
    "english": frozenset({"not", "never", "no", "neither", "nor", "without"}),
    "fr": frozenset({"ne", "n'", "pas", "jamais", "aucun", "sans"}),
    "fra": frozenset({"ne", "n'", "pas", "jamais", "aucun", "sans"}),
    "french": frozenset({"ne", "n'", "pas", "jamais", "aucun", "sans"}),
    "de": frozenset({"nicht", "kein", "keine", "nie", "ohne"}),
    "deu": frozenset({"nicht", "kein", "keine", "nie", "ohne"}),
    "german": frozenset({"nicht", "kein", "keine", "nie", "ohne"}),
    "es": frozenset({"no", "nunca", "nadie", "ningún", "ningun", "sin"}),
    "spa": frozenset({"no", "nunca", "nadie", "ningún", "ningun", "sin"}),
    "spanish": frozenset({"no", "nunca", "nadie", "ningún", "ningun", "sin"}),
    "pt": frozenset({"não", "nunca", "ninguém", "nenhum", "sem"}),
    "por": frozenset({"não", "nunca", "ninguém", "nenhum", "sem"}),
    "portuguese": frozenset({"não", "nunca", "ninguém", "nenhum", "sem"}),
    "it": frozenset({"non", "mai", "nessuno", "senza"}),
    "ita": frozenset({"non", "mai", "nessuno", "senza"}),
    "italian": frozenset({"non", "mai", "nessuno", "senza"}),
}
_LANGUAGE_MARKERS_BY_LANGUAGE = {
    "vi": _VI_MARKERS,
    "vie": _VI_MARKERS,
    "vietnamese": _VI_MARKERS,
    "en": frozenset({"the", "and", "you", "your", "to", "of", "is", "are", "in", "not"}),
    "eng": frozenset({"the", "and", "you", "your", "to", "of", "is", "are", "in", "not"}),
    "english": frozenset({"the", "and", "you", "your", "to", "of", "is", "are", "in", "not"}),
    "fr": frozenset({"le", "la", "les", "et", "vous", "de", "des", "est", "pas"}),
    "fra": frozenset({"le", "la", "les", "et", "vous", "de", "des", "est", "pas"}),
    "french": frozenset({"le", "la", "les", "et", "vous", "de", "des", "est", "pas"}),
    "de": frozenset({"der", "die", "das", "und", "sie", "du", "ist", "nicht", "zu"}),
    "deu": frozenset({"der", "die", "das", "und", "sie", "du", "ist", "nicht", "zu"}),
    "german": frozenset({"der", "die", "das", "und", "sie", "du", "ist", "nicht", "zu"}),
    "es": frozenset({"el", "la", "los", "las", "y", "de", "que", "es", "no"}),
    "spa": frozenset({"el", "la", "los", "las", "y", "de", "que", "es", "no"}),
    "spanish": frozenset({"el", "la", "los", "las", "y", "de", "que", "es", "no"}),
}
_NUMBER_WORDS_BY_LANGUAGE = {
    "vi": _VI_NUMBER_WORDS,
    "vie": _VI_NUMBER_WORDS,
    "vietnamese": _VI_NUMBER_WORDS,
    "en": frozenset({"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "point", "percent"}),
    "eng": frozenset({"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "point", "percent"}),
    "english": frozenset({"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "point", "percent"}),
    "fr": frozenset({"zéro", "zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "cent", "mille", "million", "virgule", "pourcent"}),
    "fra": frozenset({"zéro", "zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "cent", "mille", "million", "virgule", "pourcent"}),
    "french": frozenset({"zéro", "zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "cent", "mille", "million", "virgule", "pourcent"}),
    "de": frozenset({"null", "eins", "zwei", "drei", "vier", "fünf", "funf", "sechs", "sieben", "acht", "neun", "zehn", "hundert", "tausend", "million", "komma", "prozent"}),
    "deu": frozenset({"null", "eins", "zwei", "drei", "vier", "fünf", "funf", "sechs", "sieben", "acht", "neun", "zehn", "hundert", "tausend", "million", "komma", "prozent"}),
    "german": frozenset({"null", "eins", "zwei", "drei", "vier", "fünf", "funf", "sechs", "sieben", "acht", "neun", "zehn", "hundert", "tausend", "million", "komma", "prozent"}),
}


@dataclass(frozen=True)
class PunctuationMark:
    """A punctuation boundary and the pause a TTS engine should apply."""

    char: str
    offset: int
    pause_ms: int


@dataclass(frozen=True)
class PronunciationHint:
    """A token for which a voice adapter may provide an explicit spoken form."""

    token: str
    kind: Literal["number", "acronym"]
    spoken: str


@dataclass(frozen=True)
class PronunciationPlan:
    """Normalized text plus deterministic punctuation and pronunciation metadata."""

    text: str
    marks: tuple[PunctuationMark, ...]
    hints: tuple[PronunciationHint, ...]


@dataclass(frozen=True)
class QualityReport:
    """Post-translation quality result.

    ``score`` is a conservative screening score, not a claim of semantic
    correctness. ``review`` is the intended outcome when a human/stronger model
    must inspect a sentence; only ``accept`` should flow automatically to TTS.
    """

    normalized_text: str
    score: int
    decision: QualityDecision
    issues: tuple[str, ...]
    metrics: dict[str, float | int]
    pronunciation: PronunciationPlan


@dataclass(frozen=True)
class QualityBatchReport:
    """Aggregate gate result for a translation batch.

    The deterministic checks can only establish structural risk. Unless a
    separate semantic judge has explicitly returned ``passed`` for every
    segment, a clean batch remains ``review`` and must not auto-render.
    """

    reports: tuple[QualityReport, ...]
    decision: QualityDecision
    accepted: int
    review: int
    rejected: int
    semantic_judge_pending: int
    semantic_judge_failed: int
    duplicate_outputs: int = 0

    @property
    def auto_render_allowed(self) -> bool:
        return (
            self.decision == "accept"
            and self.semantic_judge_pending == 0
            and self.semantic_judge_failed == 0
        )


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_latin(char: str) -> bool:
    return _is_letter(char) and "LATIN" in unicodedata.name(char, "")


def _replace_fullwidth_punctuation(text: str) -> str:
    replacements = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
        "…": "...",
        "–": "-",
        "—": "-",
        "　": " ",
    }
    return "".join(replacements.get(char, char) for char in text)


def normalize_translation_text(text: str, *, ensure_terminal: bool = True) -> str:
    """Normalize Unicode, whitespace, and punctuation without changing wording.

    The function intentionally does not translate, delete words, or infer
    pronunciation. It is therefore safe to run before a human review pass.
    """

    value = unicodedata.normalize("NFC", str(text or ""))
    value = _replace_fullwidth_punctuation(value)
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = re.sub(r"(?:^|\s)```(?:\w+)?\s*|\s*```(?:$|\s)", " ", value)
    value = re.sub(
        r"^\s*(?:(?:translation|translated\s*text|bản\s*dịch|dịch)\s*:\s*)",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""

    # Protect URLs/emails while applying punctuation spacing. A naive colon rule
    # would turn ``https://`` into ``https: //`` and corrupt a protected token.
    protected_values: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected_values.append(match.group(0))
        return f"OMNIURLTOKEN{len(protected_values) - 1}"

    value = re.sub(
        r"(?:https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",
        _protect,
        value,
        flags=re.IGNORECASE,
    )

    # Punctuation spacing is deterministic and locale-neutral. Keep decimal
    # separators and intra-word hyphens intact.
    value = re.sub(r"!{2,}", "!", value)
    value = re.sub(r"\?{2,}", "?", value)
    value = re.sub(r"\.{4,}", "...", value)
    value = re.sub(r"\s+([,.;:!?%])", r"\1", value)
    value = re.sub(r"([,;:!?])(?=[^\s\d,;:!?])", r"\1 ", value)
    value = re.sub(r"([\(\[\{])\s+", r"\1", value)
    value = re.sub(r"\s+([\)\]\}])", r"\1", value)
    value = re.sub(r"(?:,\s*){2,}", ", ", value)
    for index, original in enumerate(protected_values):
        value = value.replace(f"OMNIURLTOKEN{index}", original)
    value = value.strip()

    if ensure_terminal and value and value[-1].isalnum():
        value += "."
    return value


def _number_to_vietnamese(value: str) -> str:
    """Spell common integer/decimal/percent tokens for optional TTS hints."""

    raw = value.strip()
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1]
    sign = ""
    if raw.startswith(("-", "+")):
        sign = "âm " if raw[0] == "-" else "dương "
        raw = raw[1:]
    if not raw or not raw.isdigit() and not any(char in raw for char in ".,"):
        return value

    # A separator followed by groups of exactly three digits is the common
    # thousands notation (1.000 / 1,000). Otherwise it is a decimal separator
    # (3,5 / 3.5); this deterministic rule avoids pronouncing a thousands mark
    # as "phẩy" while retaining decimal digits individually.
    grouped = re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw)
    if grouped:
        try:
            spoken = _integer_to_vietnamese(int(raw.replace(".", "").replace(",", "")))
        except (TypeError, ValueError):
            return value
    elif "." in raw or "," in raw:
        separator_match = re.search(r"[.,]", raw)
        if separator_match is None:
            return value
        left, right = re.split(r"[.,]", raw, maxsplit=1)
        if not left.isdigit() or not right.isdigit():
            return value
        spoken = f"{_integer_to_vietnamese(int(left))} phẩy " + " ".join(
            _DIGITS[int(char)] for char in right
        )
    else:
        try:
            spoken = _integer_to_vietnamese(int(raw))
        except (TypeError, ValueError):
            return value
    spoken = sign + spoken
    return f"{spoken} phần trăm" if percent else spoken


def number_to_vietnamese(value: str) -> str:
    """Public wrapper for TTS adapters; returns the input for unsupported forms."""

    return _number_to_vietnamese(value)


_DIGITS = (
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
_TEENS = {
    10: "mười",
    11: "mười một",
    12: "mười hai",
    13: "mười ba",
    14: "mười bốn",
    15: "mười lăm",
    16: "mười sáu",
    17: "mười bảy",
    18: "mười tám",
    19: "mười chín",
}


def _integer_to_vietnamese(number: int) -> str:
    if number < 0 or number > 999_999_999_999:
        return str(number)
    if number == 0:
        return _DIGITS[0]

    def under_thousand(value: int) -> str:
        if value < 10:
            return _DIGITS[value]
        if value < 20:
            return _TEENS[value]
        if value < 100:
            tens, ones = divmod(value, 10)
            result = f"{_DIGITS[tens]} mươi"
            if ones:
                if ones == 1 and tens > 1:
                    result += " mốt"
                elif ones == 5:
                    result += " lăm"
                else:
                    result += f" {_DIGITS[ones]}"
            return result
        hundreds, rest = divmod(value, 100)
        result = f"{_DIGITS[hundreds]} trăm"
        if rest:
            if rest < 10:
                result += f" lẻ {_DIGITS[rest]}"
            else:
                result += f" {under_thousand(rest)}"
        return result

    # Read groups from high to low. Omitting zero groups keeps speech natural
    # (2,026 -> "hai nghìn không trăm hai mươi sáu" is not required when the
    # omitted group is unambiguous), while preserving every non-zero group.
    scales = ((1_000_000_000, "tỷ"), (1_000_000, "triệu"), (1_000, "nghìn"))
    parts: list[str] = []
    remaining = number
    for scale, name in scales:
        group, remaining = divmod(remaining, scale)
        if group:
            parts.append(f"{under_thousand(group)} {name}")
    if remaining:
        parts.append(under_thousand(remaining))
    return " ".join(parts)


_LETTER_NAMES = {
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


def build_pronunciation_plan(
    text: str,
    *,
    language: str = "vi",
    ensure_terminal: bool = True,
) -> PronunciationPlan:
    """Create punctuation marks and non-destructive pronunciation hints.

    Hints are metadata: the original normalized text remains untouched. A TTS
    adapter can opt into spelling numbers/acronyms, while a subtitle renderer
    can continue using ``plan.text`` exactly.
    """

    normalized = normalize_translation_text(text, ensure_terminal=ensure_terminal)
    marks: list[PunctuationMark] = []
    pause_by_char = {
        ",": 100,
        ";": 160,
        ":": 140,
        ".": 240,
        "?": 300,
        "!": 280,
        "...": 420,
    }
    protected_spans = [
        (match.start(), match.end())
        for match in _PROTECTED_RE.finditer(normalized)
    ]
    for match in _PUNCT_RE.finditer(normalized):
        token = match.group(0).strip()
        char = "..." if token.startswith("...") else token[:1]
        if any(start <= match.start() < end for start, end in protected_spans):
            continue
        if char in {",", "."}:
            # Decimal/thousands separators are not speech boundaries.
            before = normalized[match.start() - 1 : match.start()]
            after = normalized[match.end() : match.end() + 1]
            if before.isdigit() and after.isdigit():
                continue
        if char == ":":
            prefix = normalized[: match.start()]
            suffix = normalized[match.end() :]
            # Colons in URLs and HH:MM times are structural, not speech pauses.
            if (
                re.search(r"https?$", prefix, re.IGNORECASE)
                and suffix.startswith("//")
            ) or re.search(r"https?://[^\s]*$", prefix, re.IGNORECASE):
                continue
            if re.search(r"\b\d{1,2}$", prefix) and re.match(r"\d{2}\b", suffix):
                continue
        url_match = re.search(r"https?://[^\s]+", normalized, re.IGNORECASE)
        if url_match and url_match.start() <= match.start() < url_match.end():
            continue
        marks.append(PunctuationMark(char, match.start(), pause_by_char[char]))

    hints: list[PronunciationHint] = []
    # The Vietnamese number/acronym dictionary is intentionally language-gated.
    # Applying it to English, Japanese, or another target would silently change
    # pronunciation semantics; those adapters can provide their own profile.
    if _language_key(language) in {"vi", "vie", "vietnamese"}:
        for match in _NUMBER_RE.finditer(normalized):
            spoken = _number_to_vietnamese(match.group(0))
            if spoken != match.group(0):
                hints.append(PronunciationHint(match.group(0), "number", spoken))
        for match in _ACRONYM_RE.finditer(normalized):
            spoken = " ".join(_LETTER_NAMES.get(char, char) for char in match.group(0))
            hints.append(PronunciationHint(match.group(0), "acronym", spoken))
    return PronunciationPlan(normalized, tuple(marks), tuple(hints))


def _normalized_compact(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def _tokenize(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(value)]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _language_key(language: str) -> str:
    key = (language or "").casefold().split("-", 1)[0].split("_", 1)[0]
    return _CANONICAL_LANGUAGE_KEYS.get(key, key)


def _language_script_family(language: str) -> str:
    key = _language_key(language)
    if key in _HANGUL_LANGUAGE_KEYS:
        return "hangul"
    if key in _CJK_LANGUAGE_KEYS:
        return "cjk"
    if key in _CYRILLIC_LANGUAGE_KEYS:
        return "cyrillic"
    if key in _ARABIC_LANGUAGE_KEYS:
        return "arabic"
    if key in _THAI_LANGUAGE_KEYS:
        return "thai"
    if key in _LATIN_LANGUAGE_KEYS:
        return "latin"
    return "unknown"


def _script_ratios(value: str) -> dict[str, float]:
    counts = {"latin": 0, "cjk": 0, "hangul": 0, "cyrillic": 0, "arabic": 0, "thai": 0}
    visible = max(1, sum(1 for char in value if not char.isspace()))
    for char in value:
        if _is_cjk(char):
            counts["cjk"] += 1
        elif "HANGUL" in unicodedata.name(char, ""):
            counts["hangul"] += 1
        elif any(mark in unicodedata.name(char, "") for mark in ("HIRAGANA", "KATAKANA")):
            counts["cjk"] += 1
        elif "CYRILLIC" in unicodedata.name(char, ""):
            counts["cyrillic"] += 1
        elif "ARABIC" in unicodedata.name(char, ""):
            counts["arabic"] += 1
        elif "THAI" in unicodedata.name(char, ""):
            counts["thai"] += 1
        elif _is_latin(char):
            counts["latin"] += 1
    return {family: _ratio(count, visible) for family, count in counts.items()}


def _semantic_units(value: str, language: str) -> int:
    """Return a coarse unit count for length-risk screening only."""

    key = _language_key(language)
    if key in _CJK_LANGUAGE_KEYS:
        native_count = sum(
            1
            for char in value
            if _is_cjk(char)
            or any(mark in unicodedata.name(char, "") for mark in ("HIRAGANA", "KATAKANA"))
        )
        latin_runs = len(re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?", value))
        return native_count + latin_runs
    if key in _HANGUL_LANGUAGE_KEYS:
        hangul_count = sum("HANGUL" in unicodedata.name(char, "") for char in value)
        latin_runs = len(re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?", value))
        return hangul_count + latin_runs
    # Numeric semiotic classes are valid spoken content even though ``_TOKEN_RE``
    # intentionally returns letter tokens only. Count each raw numeric run as one
    # semantic unit; timing uses the private TN-expanded spoken copy when available.
    return len(_tokenize(value)) + len(_NUMBER_SCAN_RE.findall(value))


def speech_unit_count(value: str, language: str) -> int:
    """Count target-side units suitable for estimating spoken duration."""

    units = _semantic_units(value, language)
    # A private NeMo/locale TN copy normally contains no digits. If a development
    # fallback or unsupported locale still leaves a semiotic class unresolved,
    # count each digit rather than treating a long number as one silent token.
    numeric_runs = len(_NUMBER_SCAN_RE.findall(value))
    digit_count = sum(char.isdecimal() for char in value)
    return units + max(0, digit_count - numeric_runs)


def estimate_spoken_duration_seconds(value: str, language: str) -> float:
    """Estimate natural speech duration without comparing two languages' units."""

    units = speech_unit_count(value, language)
    if units <= 0:
        return 0.0
    rate = _SPEECH_RATE_BY_LANGUAGE.get(_language_key(language), 2.7)
    punctuation_pause = 0.0
    for match in _PUNCT_RE.finditer(value):
        mark = match.group(0).strip()
        punctuation_pause += 0.08 if mark[:1] in {",", ";", ":"} else 0.12
    return round((units / rate) + min(0.45, punctuation_pause), 4)


def timing_overflow_penalty(value: str, language: str, duration_seconds: float) -> float:
    """Return a deterministic overflow penalty; zero means within 8% tolerance."""

    try:
        available = float(duration_seconds)
    except (TypeError, ValueError):
        return 0.0
    if available <= 0:
        return 0.0
    ratio = estimate_spoken_duration_seconds(value, language) / available
    return round(max(0.0, ratio - 1.08) * 200.0, 4)


def _canonical_number(value: str) -> str:
    value = value.strip().replace(",", ".")
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = value[0], value[1:]
    # A single separator followed by exactly three digits is treated as a
    # thousands grouping (1.000 == 1,000), not as a decimal point. When both
    # separators occur, the final one is the decimal separator.
    if "." in value or "," in value:
        separators = [index for index, char in enumerate(value) if char in ".,"]
        last = separators[-1]
        fractional = value[last + 1 :]
        grouped = len(separators) == 1 and len(fractional) == 3
        if len(separators) > 1:
            decimal_separator = value[last]
            grouped_value = value[:last].replace(".", "").replace(",", "")
            value = f"{grouped_value}.{fractional}" if decimal_separator else grouped_value
        elif grouped:
            value = value.replace(".", "").replace(",", "")
        else:
            left, right = value.replace(",", ".").split(".", 1)
            value = f"{int(left or 0)}.{right.rstrip('0')}".rstrip(".")
    else:
        try:
            value = str(int(value))
        except ValueError:
            pass
    return f"{sign}{value}%" if percent else f"{sign}{value}"


_CJK_DIGITS = {
    "零": 0,
    "〇": 0,
    "○": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "两": 2,
    "兩": 2,
    "贰": 2,
    "貳": 2,
    "三": 3,
    "叁": 3,
    "參": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "陸": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
_CJK_SMALL_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CJK_LARGE_UNITS = {"万": 10_000, "萬": 10_000, "亿": 100_000_000, "億": 100_000_000}
_MAX_CJK_NUMERAL_CHARS = 64
_MAX_CJK_NUMERAL_VALUE = 10**15


def _cjk_integer(value: str) -> int | None:
    """Parse a bounded Chinese integer phrase, returning ``None`` if uncertain."""

    if (
        not value
        or len(value) > _MAX_CJK_NUMERAL_CHARS
        or any(
            char not in _CJK_DIGITS
            and char not in _CJK_SMALL_UNITS
            and char not in _CJK_LARGE_UNITS
            and not char.isdecimal()
            for char in value
        )
    ):
        return None
    # A mixed Arabic/CJK phrase is accepted only when every Arabic run is an
    # integer.  Decimal punctuation is handled by the outer Arabic scanner.
    tokens: list[tuple[str, int]] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char.isdecimal():
            end = index + 1
            while end < len(value) and value[end].isdecimal():
                end += 1
            try:
                tokens.append(("digit", int(value[index:end])))
            except (TypeError, ValueError):
                return None
            index = end
            continue
        if char in _CJK_DIGITS:
            tokens.append(("digit", _CJK_DIGITS[char]))
        elif char in _CJK_SMALL_UNITS:
            tokens.append(("small", _CJK_SMALL_UNITS[char]))
        else:
            tokens.append(("large", _CJK_LARGE_UNITS[char]))
        index += 1

    # No unit means a sequence such as 二〇二六; read it digit-by-digit.  A
    # single digit is also valid when the caller proved a measurement unit.
    if all(kind == "digit" for kind, _ in tokens):
        digits = "".join(str(number) for _, number in tokens)
        try:
            result = int(digits) if digits else None
        except (TypeError, ValueError):
            return None
        return result if result is not None and result <= _MAX_CJK_NUMERAL_VALUE else None

    total = 0
    section = 0
    pending = 0
    for kind, number in tokens:
        if kind == "digit":
            pending = number
            continue
        if kind == "small":
            # 十/百/千 without an explicit leading digit means one ten/hundred.
            section += (pending or 1) * number
            pending = 0
            continue
        section += pending
        total += section * number
        section = 0
        pending = 0
    result = total + section + pending
    return result if result <= _MAX_CJK_NUMERAL_VALUE else None


def _cjk_number_values(value: str) -> list[str]:
    """Extract exact CJK quantities for the fidelity check.

    This is deliberately a structural helper, not a Chinese tokenizer.  It
    ignores approximate quantities (``幾百``/``数百``), ordinals, and bare
    classifiers so a translation is not penalized for choosing a natural word
    instead of an Arabic digit.  Measurement-attached values and ratios remain
    checked because dropping them changes meaning.
    """

    covered: list[tuple[int, int]] = []
    values: list[str] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in covered)

    for match in _CJK_DECIMAL_RE.finditer(value):
        whole = match.group("whole")
        fraction = match.group("fraction")
        if any(char in _CJK_UNKNOWN_QUANTITY_MARKERS for char in whole + fraction):
            continue
        whole_value = _cjk_integer(whole)
        fraction_value = _cjk_integer(fraction)
        if whole_value is None or fraction_value is None:
            continue
        # ``一米四`` means 1.4 in the source domain; preserve each fractional
        # digit instead of treating ``四`` as forty.
        fraction_digits = "".join(str(_CJK_DIGITS.get(char, char)) for char in fraction)
        if not fraction_digits.isdigit():
            fraction_digits = str(fraction_value)
        values.append(_canonical_number(f"{whole_value}.{fraction_digits}"))
        covered.append(match.span())

    for match in _CJK_RATIO_RE.finditer(value):
        left = match.group("left")
        right = match.group("right")
        if any(char in _CJK_UNKNOWN_QUANTITY_MARKERS for char in left + right):
            continue
        left_value = _cjk_integer(left)
        right_value = _cjk_integer(right)
        if left_value is None or right_value is None:
            continue
        values.extend((str(left_value), str(right_value)))
        covered.append(match.span())

    for match in _CJK_NUMERAL_RE.finditer(value):
        start, end = match.span()
        if overlaps(start, end):
            continue
        phrase = match.group(0)
        if any(char in _CJK_UNKNOWN_QUANTITY_MARKERS for char in phrase):
            continue
        if start > 0 and value[start - 1] in {"幾", "几", "數", "数", "多"}:
            continue
        if start > 0 and value[start - 1] == "第":
            continue
        has_magnitude = any(char in _CJK_SMALL_UNITS or char in _CJK_LARGE_UNITS for char in phrase)
        following = value[end:]
        if following and following[0] in _CJK_UNKNOWN_QUANTITY_MARKERS:
            continue
        has_measure = any(following.startswith(unit) for unit in _CJK_MEASURE_UNITS)
        if not has_magnitude and not has_measure:
            continue
        parsed = _cjk_integer(phrase)
        if parsed is not None:
            values.append(str(parsed))

    return values


def _numbers(value: str, language: str | None = None) -> Counter[str]:
    numbers = [_canonical_number(item) for item in _NUMBER_SCAN_RE.findall(value)]
    if language is None or _language_key(language) in _CJK_LANGUAGE_KEYS:
        numbers.extend(_cjk_number_values(value))
    return Counter(numbers)


def _contains_negation(value: str, language: str) -> bool:
    key = _language_key(language)
    if key in _HANGUL_LANGUAGE_KEYS:
        return any(marker in value for marker in _KO_NEGATION_MARKERS)
    if key in {"ja", "jpn", "japanese"}:
        return any(marker in value for marker in _JA_NEGATION_MARKERS)
    if key in _CJK_LANGUAGE_KEYS:
        return any(marker in value for marker in _CJK_NEGATION_MARKERS)
    markers = _NEGATION_MARKERS_BY_LANGUAGE.get(key, frozenset())
    tokens = _tokenize(value)
    return any(
        marker in value.casefold() if " " in marker else marker in tokens
        for marker in markers
    )


def _number_word_runs(value: str, language: str = "vi") -> int:
    """Count coarse number-word phrases for spelled-out entity checks.

    This is intentionally a count signal, not a parser: ``mười hai`` and
    ``hai mươi phần trăm`` each count as one phrase, while ``12:30`` translated
    as ``mười hai giờ ba mươi`` counts as two. The independent semantic judge
    remains responsible for verifying the actual numeric value.
    """

    number_words = set(_NUMBER_WORDS_BY_LANGUAGE.get(_language_key(language), ()))
    if not number_words:
        return 0
    if _language_key(language) in {"vi", "vie", "vietnamese"}:
        number_words.update({"lẻ", "linh", "phần"})
    runs = 0
    in_run = False
    for token in _tokenize(value):
        is_number_word = token in number_words
        if is_number_word and not in_run:
            runs += 1
        in_run = is_number_word
    return runs


def _protected_tokens(value: str) -> Counter[str]:
    return Counter(item.casefold().rstrip(".,;:!?)]}") for item in _PROTECTED_RE.findall(value))


def _has_source_negation(value: str, language: str) -> bool:
    return _contains_negation(value, language)


def _has_target_negation(value: str, language: str) -> bool:
    key = _language_key(language)
    if key in {"vi", "vie", "vietnamese"}:
        tokens = _tokenize(value)
        # Sentence-final "không?" is a Vietnamese yes/no question particle,
        # not necessarily a negation of the source proposition.
        if tokens and tokens[-1] == "không":
            tokens = tokens[:-1]
        for index, token in enumerate(tokens):
            if token not in _VI_NEGATION_MARKERS:
                continue
            # In a spoken number, "không" means zero (for example,
            # "hai nghìn không trăm hai mươi sáu"), not negation.
            if token == "không":
                neighbors = tokens[max(0, index - 1) : index + 2]
                if any(
                    item in _VI_NUMERIC_CONTEXT_WORDS
                    for item in neighbors
                    if item != token
                ):
                    continue
            return True
        return False
    return _contains_negation(value, language)


def score_translation(
    source_text: str,
    translated_text: str,
    *,
    source_language: str = "zh",
    target_language: str = "vi",
    duration_seconds: float | None = None,
    spoken_text: str | None = None,
) -> QualityReport:
    """Score a candidate translation with conservative, explainable signals.

    This is a *screening* gate. It cannot prove semantic fidelity; low scores
    force review instead of silently passing malformed text to TTS.
    """

    raw_target = str(translated_text or "")
    raw_trimmed = raw_target.strip()
    missing_terminal = bool(raw_trimmed and raw_trimmed[-1].isalnum())
    normalized = normalize_translation_text(raw_target)
    issues: list[str] = []
    score = 100
    source = str(source_text or "")
    target_key = _language_key(target_language)
    source_compact = _normalized_compact(source)
    target_compact = _normalized_compact(normalized)
    tokens = _tokenize(normalized)
    letters = sum(1 for char in normalized if _is_letter(char))
    cjk = sum(1 for char in normalized if _is_cjk(char))
    latin = sum(1 for char in normalized if _is_latin(char))
    visible = sum(1 for char in normalized if not char.isspace())
    cjk_ratio = _ratio(cjk, max(1, visible))
    script_ratios = _script_ratios(normalized)
    latin_ratio = _ratio(latin, max(1, letters))
    source_units = _semantic_units(source, source_language)
    target_units = _semantic_units(normalized, target_language)
    length_ratio = _ratio(target_units, source_units)
    source_numbers = _numbers(source, source_language)
    target_numbers = _numbers(normalized, target_language)
    source_protected = _protected_tokens(source)
    target_protected = _protected_tokens(normalized)

    if not normalized:
        issues.append("empty_translation")
        score = 0
    if not source.strip() and normalized:
        issues.append("source_empty_target_nonempty")
        score -= 40
    if source.strip() and not tokens and not any(char.isdecimal() for char in normalized):
        issues.append("target_has_no_words")
        score -= 40
    if _ARTIFACT_RE.search(normalized):
        issues.append("model_artifact")
        score -= 45
    if _MOJIBAKE_RE.search(normalized):
        issues.append("encoding_artifact")
        score -= 40
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        issues.append("control_character")
        score -= 25
    target_script = _language_script_family(target_language)
    source_script = _language_script_family(source_language)
    if target_script != "unknown" and source_script != "unknown" and target_script != source_script:
        residue_ratio = script_ratios.get(source_script, 0.0)
        if residue_ratio >= 0.08:
            issues.append("source_script_residue")
            score -= min(45, round(residue_ratio * 120))
    source_language_key = _language_key(source_language)
    same_language = source_language_key == target_key and bool(source_language_key)
    if (
        source_compact
        and target_compact
        and source_compact == target_compact
        and not same_language
        and (
            any(_is_letter(char) for char in source)
            or any(_is_letter(char) for char in normalized)
        )
    ):
        issues.append("copied_source")
        score -= 35
    elif source_compact and target_compact and len(target_compact) >= 8:
        # A high compact overlap catches an untranslated Latin source while
        # allowing names and short loanwords to survive.
        overlap = len(set(_tokenize(source)) & set(tokens)) / max(1, len(set(tokens)))
        if overlap >= 0.85 and source_language.casefold() not in {"vi", "vie", "vietnamese"}:
            issues.append("likely_untranslated")
            score -= 25

    # These checks are semantic-risk signals, not semantic proof. A missing
    # number/entity can change the meaning, so require review or a second judge.
    # Spelled-out Vietnamese numbers are counted by phrase, rather than accepting
    # any single token such as ``một`` as proof that every source number survived.
    number_words_cover = (
        target_key in _NUMBER_WORDS_BY_LANGUAGE
        and not target_numbers
        and _number_word_runs(normalized, target_language) >= sum(source_numbers.values())
    )
    if source_numbers != target_numbers and not number_words_cover:
        issues.append("number_mismatch")
        missing_numbers = sum((source_numbers - target_numbers).values())
        score -= min(24, 12 * max(1, missing_numbers))
    if source_protected != target_protected:
        issues.append("protected_token_mismatch")
        score -= min(18, 9 * max(1, sum((source_protected - target_protected).values())))
    if source_units >= 5 and (length_ratio < 0.2 or length_ratio > 4.0):
        issues.append("extreme_length_ratio")
        score -= 12
    if source_units >= 4 and _has_source_negation(source, source_language) != _has_target_negation(normalized, target_language):
        issues.append("negation_mismatch")
        score -= 12
    target_folded = normalized.casefold()
    if any(phrase in target_folded for phrase in _SUSPICIOUS_BOILERPLATE):
        issues.append("boilerplate_hallucination")
        score -= 20

    if len(tokens) == 1 and len(_tokenize(source)) >= 4:
        issues.append("too_short")
        score -= 20
    if len(tokens) >= 3:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        max_repeat = max(counts.values())
        repeat_ratio = max_repeat / len(tokens)
        if repeat_ratio >= 0.5:
            issues.append("repeated_token_loop")
            score -= min(30, round(repeat_ratio * 40))
    else:
        repeat_ratio = 0.0

    malformed_spacing = bool(re.search(r"\s+[,.!?;:]|[,!?;:](?=\S)", str(translated_text or "")))
    if malformed_spacing:
        issues.append("punctuation_spacing")
        score -= 6
    if re.search(r"([!?])\1|\.{4,}", str(translated_text or "")):
        issues.append("punctuation_run")
        score -= 5
    if missing_terminal:
        issues.append("missing_terminal_punctuation")
        score -= 2
    target_markers = _LANGUAGE_MARKERS_BY_LANGUAGE.get(target_key, frozenset())
    if target_markers and tokens:
        marker_count = sum(1 for token in tokens if token in target_markers)
        marker_ratio = _ratio(marker_count, len(tokens))
        # This is only a weak signal: names and technical sentences may have
        # no stopwords, so it never rejects on its own.
        if len(tokens) >= 5 and marker_count == 0:
            issues.append("weak_target_language_signal")
            score -= 8
    else:
        marker_ratio = 0.0

    # The subtitle is the semantic authority, while timing must follow the private
    # text-normalized copy that TTS actually reads. Callers that own the TN boundary
    # pass ``spoken_text``; direct/legacy callers retain the conservative raw fallback.
    timing_text = (
        normalize_translation_text(str(spoken_text), ensure_terminal=True)
        if spoken_text is not None
        else normalized
    )
    spoken_units = speech_unit_count(timing_text, target_language)
    estimated_spoken_seconds = estimate_spoken_duration_seconds(timing_text, target_language)
    timing_ratio = 0.0
    timing_penalty = 0.0
    if duration_seconds is not None:
        try:
            available_seconds = max(0.0, float(duration_seconds))
        except (TypeError, ValueError):
            available_seconds = 0.0
        if available_seconds > 0:
            timing_ratio = estimated_spoken_seconds / available_seconds
            timing_penalty = timing_overflow_penalty(
                timing_text, target_language, available_seconds
            )
            if timing_ratio > 1.15:
                issues.append("timing_overflow")
                score -= min(30, 10 + round((timing_ratio - 1.15) * 24))
            elif timing_ratio < 0.45 and spoken_units >= 2:
                # Underfill is review-worthy but intentionally mild: silence can be padded,
                # whereas inventing words to fill a shot would damage fidelity.
                issues.append("timing_underfill")
                score -= 5
    else:
        available_seconds = 0.0

    score = max(0, min(100, int(score)))
    severe = {
        "empty_translation",
        "source_empty_target_nonempty",
        "target_has_no_words",
        "model_artifact",
        "encoding_artifact",
        "source_script_residue",
        "copied_source",
        "repeated_token_loop",
    }
    if score < 45 or severe.intersection(issues):
        decision: QualityDecision = "reject"
    elif score < 78 or issues:
        decision = "review"
    else:
        decision = "accept"

    metrics: dict[str, float | int] = {
        "source_chars": len(source),
        "target_chars": len(normalized),
        "source_tokens": len(_tokenize(source)),
        "target_tokens": len(tokens),
        "source_units": source_units,
        "target_units": target_units,
        "length_ratio": round(length_ratio, 4),
        "source_numbers": sum(source_numbers.values()),
        "target_numbers": sum(target_numbers.values()),
        "protected_tokens": sum(source_protected.values()),
        "cjk_ratio": round(cjk_ratio, 4),
        "latin_ratio": round(latin_ratio, 4),
        "repeat_ratio": round(repeat_ratio, 4),
        # Keep the historical key for log/report compatibility; the generic
        # name is the authoritative field for non-Vietnamese profiles.
        "vietnamese_marker_ratio": round(marker_ratio, 4),
        "target_language_marker_ratio": round(marker_ratio, 4),
        "speech_units": spoken_units,
        "estimated_spoken_seconds": round(estimated_spoken_seconds, 4),
        "available_seconds": round(available_seconds, 4),
        "timing_ratio": round(timing_ratio, 4),
        "timing_overflow_penalty": round(timing_penalty, 4),
    }
    return QualityReport(
        normalized_text=normalized,
        score=score,
        decision=decision,
        issues=tuple(dict.fromkeys(issues)),
        metrics=metrics,
        pronunciation=build_pronunciation_plan(normalized, language=target_language),
    )


def quality_gate(
    pairs: Iterable[tuple[str, str]],
    *,
    source_language: str = "zh",
    target_language: str = "vi",
    durations: Iterable[float | None] | None = None,
    spoken_texts: Iterable[str | None] | None = None,
    semantic_judges: Iterable[str] | None = None,
    require_semantic_judge: bool = True,
) -> QualityBatchReport:
    """Evaluate a batch and fail closed when semantic review is absent.

    ``pairs`` contains ``(source_text, translated_text)`` values. The optional
    ``semantic_judges`` iterable must contain one ``passed``/``failed``/``not_run``
    value per pair. It is intentionally separate from the deterministic score:
    a model judging its own output is not accepted as proof unless the caller
    explicitly marks the result ``passed`` after an independent judge.
    """

    pair_list = list(pairs)
    duration_list = list(durations) if durations is not None else [None] * len(pair_list)
    if len(duration_list) != len(pair_list):
        raise ValueError("translation_durations_length_mismatch")
    spoken_list = list(spoken_texts) if spoken_texts is not None else [None] * len(pair_list)
    if len(spoken_list) != len(pair_list):
        raise ValueError("translation_spoken_texts_length_mismatch")
    judge_list = list(semantic_judges) if semantic_judges is not None else []
    if semantic_judges is not None and len(judge_list) != len(pair_list):
        raise ValueError("semantic_judges_length_mismatch")
    valid_judges = {"passed", "failed", "not_run"}
    if any(not isinstance(state, str) or state.casefold() not in valid_judges for state in judge_list):
        raise ValueError("semantic_judge_state_invalid")

    reports = tuple(
        score_translation(
            source,
            translated,
            source_language=source_language,
            target_language=target_language,
            duration_seconds=duration_list[index],
            spoken_text=spoken_list[index],
        )
        for index, (source, translated) in enumerate(pair_list)
    )
    accepted = sum(report.decision == "accept" for report in reports)
    review = sum(report.decision == "review" for report in reports)
    rejected = sum(report.decision == "reject" for report in reports)
    target_sources: dict[str, set[str]] = {}
    for (source, _), report in zip(pair_list, reports, strict=True):
        target_key = _normalized_compact(report.normalized_text)
        if len(_tokenize(report.normalized_text)) < 3 or not target_key:
            continue
        source_key = _normalized_compact(source)
        if source_key:
            target_sources.setdefault(target_key, set()).add(source_key)
    # Repeated identical source lines are legitimate; only flag one target used
    # for two or more *different* source lines.
    duplicate_outputs = sum(
        len(source_keys) - 1
        for source_keys in target_sources.values()
        if len(source_keys) > 1
    )
    pending = 0
    failed = 0
    if require_semantic_judge:
        if semantic_judges is None:
            pending = len(reports)
        else:
            pending = sum(state.casefold() == "not_run" for state in judge_list)
            failed = sum(state.casefold() == "failed" for state in judge_list)

    if rejected or failed:
        decision: QualityDecision = "reject"
    elif review or pending or duplicate_outputs:
        decision = "review"
    else:
        decision = "accept"
    return QualityBatchReport(
        reports=reports,
        decision=decision,
        accepted=accepted,
        review=review,
        rejected=rejected,
        semantic_judge_pending=pending,
        semantic_judge_failed=failed,
        duplicate_outputs=duplicate_outputs,
    )


def quality_gate_segments(
    segments: Iterable[Mapping[str, object]],
    *,
    source_key: str = "original_text",
    target_key: str = "translated_text",
    source_language: str = "zh",
    target_language: str = "vi",
    spoken_texts: Iterable[str | None] | None = None,
    semantic_judges: Iterable[str] | None = None,
    require_semantic_judge: bool = True,
) -> QualityBatchReport:
    """Adapter for the worker's segment dictionaries.

    Text plus read-only duration are scored. IDs/start/end remain outside the
    scorer, and no timing field can be rewritten by this adapter.
    """

    pairs = []
    durations = []
    for segment in segments:
        pairs.append((str(segment.get(source_key, "")), str(segment.get(target_key, ""))))
        raw_duration = segment.get("duration")
        durations.append(
            float(raw_duration)
            if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool)
            else None
        )
    return quality_gate(
        pairs,
        source_language=source_language,
        target_language=target_language,
        durations=durations,
        spoken_texts=spoken_texts,
        semantic_judges=semantic_judges,
        require_semantic_judge=require_semantic_judge,
    )


def quality_gate_metadata(report: QualityBatchReport) -> dict[str, object]:
    """Return count-only metadata that is safe for logs/manifests.

    Source text, translated text, pronunciation hints, segment IDs, and model
    output are intentionally absent. This function is the only representation
    callers should serialize under the worker's zero-logging policy.
    """

    scores = [item.score for item in report.reports]
    issue_counts = Counter(
        issue for item in report.reports for issue in item.issues
    )
    if report.duplicate_outputs:
        issue_counts["duplicate_output"] += report.duplicate_outputs
    return {
        "decision": report.decision,
        "segments": len(report.reports),
        "accepted": report.accepted,
        "review": report.review,
        "rejected": report.rejected,
        "semantic_judge_pending": report.semantic_judge_pending,
        "semantic_judge_failed": report.semantic_judge_failed,
        "duplicate_outputs": report.duplicate_outputs,
        "score_min": min(scores) if scores else 0,
        "score_mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


__all__ = [
    "PunctuationMark",
    "PronunciationHint",
    "PronunciationPlan",
    "QualityBatchReport",
    "QualityDecision",
    "QualityReport",
    "build_pronunciation_plan",
    "estimate_spoken_duration_seconds",
    "normalize_translation_text",
    "number_to_vietnamese",
    "quality_gate",
    "quality_gate_metadata",
    "quality_gate_segments",
    "score_translation",
    "speech_unit_count",
    "timing_overflow_penalty",
]
