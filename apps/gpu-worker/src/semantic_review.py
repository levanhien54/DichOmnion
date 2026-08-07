"""Fail-closed, auditable semantic-review input.

The deterministic quality gate can identify risk, but it cannot prove that a fluent
translation preserves meaning. This module validates the *separate* verdict file used by
an editor or an independent judge. It deliberately accepts verdicts and bounded reason
codes only; free-form model commentary is rejected so meaningless prose cannot become a
release signal.

The file is a local QA/operations boundary. Production Analyze -> Review -> Approve remains
the preferred human path, while a trusted judge service may emit the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Literal


SemanticState = Literal["passed", "failed", "not_run"]
ReviewerKind = Literal["human", "independent_model"]

SCHEMA_VERSION = 1
MAX_REVIEW_FILE_BYTES = 512 * 1024
MAX_REVIEWER_ID_LENGTH = 128
MAX_JUDGE_MODEL_LENGTH = 200
MAX_SEGMENT_ID_LENGTH = 128
MAX_ISSUE_CODES = 16
MAX_ISSUE_CODE_LENGTH = 64

_STATE_VALUES = frozenset({"passed", "failed", "not_run"})
_REVIEWER_KINDS = frozenset({"human", "independent_model"})
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")


class SemanticReviewError(ValueError):
    """Stable, sanitized error raised for an invalid review document."""

    def __init__(self, code: str):
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", code):
            raise ValueError("invalid semantic review error code")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SemanticReview:
    segment_id: str
    state: SemanticState
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class SemanticReviewBatch:
    schema_version: int
    reviewer_kind: ReviewerKind
    reviewer_id: str
    judge_model: str | None
    judge_revision: str | None
    reviews: tuple[SemanticReview, ...]

    def states_for(self, segment_ids: Iterable[str]) -> tuple[SemanticState, ...]:
        """Return verdicts in the caller's segment order after exact-ID validation."""

        by_id = {item.segment_id: item.state for item in self.reviews}
        requested = tuple(str(item) for item in segment_ids)
        if len(set(requested)) != len(requested) or set(requested) != set(by_id):
            raise SemanticReviewError("semantic_review_segment_ids_mismatch")
        return tuple(by_id[item] for item in requested)


def _reject_constants(_: str) -> None:
    raise ValueError("non-finite JSON constant")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constants,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        raise SemanticReviewError("semantic_review_json_invalid") from None


def _bounded_text(value: Any, *, code: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SemanticReviewError(code)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise SemanticReviewError(f"{code}_encoding_invalid") from None
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise SemanticReviewError(f"{code}_control_character")
    return value.strip()


def _parse_document(document: Any, segment_ids: tuple[str, ...], *, translation_model_id: str | None) -> SemanticReviewBatch:
    if not isinstance(document, dict):
        raise SemanticReviewError("semantic_review_document_invalid")
    allowed = {"schema_version", "reviewer_kind", "reviewer_id", "reviews", "judge_model", "judge_revision"}
    if set(document) - allowed:
        raise SemanticReviewError("semantic_review_unknown_field")
    # ``bool`` is an ``int`` subclass in Python; require the exact JSON number
    # type so ``true`` cannot masquerade as schema version 1.
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise SemanticReviewError("semantic_review_schema_unsupported")

    reviewer_kind = document.get("reviewer_kind")
    if reviewer_kind not in _REVIEWER_KINDS:
        raise SemanticReviewError("semantic_review_reviewer_kind_invalid")
    reviewer_id = _bounded_text(
        document.get("reviewer_id"),
        code="semantic_review_reviewer_id_invalid",
        maximum=MAX_REVIEWER_ID_LENGTH,
    )

    judge_model: str | None = None
    judge_revision: str | None = None
    if reviewer_kind == "independent_model":
        judge_model = _bounded_text(
            document.get("judge_model"),
            code="semantic_review_judge_model_invalid",
            maximum=MAX_JUDGE_MODEL_LENGTH,
        )
        if translation_model_id and judge_model.casefold() == translation_model_id.casefold():
            raise SemanticReviewError("semantic_review_judge_must_be_independent")
        judge_revision = _bounded_text(
            document.get("judge_revision"),
            code="semantic_review_judge_revision_invalid",
            maximum=MAX_JUDGE_MODEL_LENGTH,
        )
    elif "judge_model" in document or "judge_revision" in document:
        raise SemanticReviewError("semantic_review_human_has_model_fields")

    reviews = document.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(segment_ids):
        raise SemanticReviewError("semantic_review_count_mismatch")
    expected_ids = set(segment_ids)
    parsed: list[SemanticReview] = []
    seen: set[str] = set()
    for item in reviews:
        if not isinstance(item, dict):
            raise SemanticReviewError("semantic_review_item_invalid")
        if set(item) - {"segment_id", "state", "issue_codes"}:
            raise SemanticReviewError("semantic_review_item_unknown_field")
        segment_id = _bounded_text(
            item.get("segment_id"),
            code="semantic_review_segment_id_invalid",
            maximum=MAX_SEGMENT_ID_LENGTH,
        )
        if segment_id not in expected_ids or segment_id in seen:
            raise SemanticReviewError("semantic_review_segment_id_invalid")
        state = item.get("state")
        if not isinstance(state, str) or state.casefold() not in _STATE_VALUES:
            raise SemanticReviewError("semantic_review_state_invalid")
        raw_codes = item.get("issue_codes", [])
        if not isinstance(raw_codes, list) or len(raw_codes) > MAX_ISSUE_CODES:
            raise SemanticReviewError("semantic_review_issue_codes_invalid")
        issue_codes: list[str] = []
        for code in raw_codes:
            if not isinstance(code, str) or not _CODE_RE.fullmatch(code):
                raise SemanticReviewError("semantic_review_issue_code_invalid")
            issue_codes.append(code)
        if len(set(issue_codes)) != len(issue_codes):
            raise SemanticReviewError("semantic_review_issue_codes_duplicate")
        if state.casefold() == "failed" and not issue_codes:
            raise SemanticReviewError("semantic_review_failed_without_reason")
        seen.add(segment_id)
        parsed.append(
            SemanticReview(
                segment_id=segment_id,
                state=state.casefold(),
                issue_codes=tuple(issue_codes),
            )
        )
    if seen != expected_ids:
        raise SemanticReviewError("semantic_review_segment_ids_mismatch")
    return SemanticReviewBatch(
        schema_version=SCHEMA_VERSION,
        reviewer_kind=reviewer_kind,
        reviewer_id=reviewer_id,
        judge_model=judge_model,
        judge_revision=judge_revision,
        reviews=tuple(parsed),
    )


def load_semantic_review(
    path: str | Path,
    segment_ids: Iterable[str],
    *,
    translation_model_id: str | None = None,
) -> SemanticReviewBatch:
    """Load and validate a per-segment review file without logging its contents."""

    requested = tuple(str(item) for item in segment_ids)
    if len(set(requested)) != len(requested):
        raise SemanticReviewError("semantic_review_segment_ids_duplicate")
    review_path = Path(path)
    try:
        if not review_path.is_file() or review_path.stat().st_size > MAX_REVIEW_FILE_BYTES:
            raise SemanticReviewError("semantic_review_file_invalid")
        raw = review_path.read_bytes()
    except SemanticReviewError:
        raise
    except (OSError, ValueError):
        raise SemanticReviewError("semantic_review_file_unreadable") from None
    return _parse_document(
        _strict_json(raw), requested, translation_model_id=translation_model_id
    )


def semantic_review_metadata(batch: SemanticReviewBatch) -> dict[str, object]:
    """Return count-only metadata suitable for logs/manifests."""

    states = [item.state for item in batch.reviews]
    return {
        "provided": True,
        "schema_version": batch.schema_version,
        "reviewer_kind": batch.reviewer_kind,
        "segments": len(states),
        "passed": sum(state == "passed" for state in states),
        "failed": sum(state == "failed" for state in states),
        "not_run": sum(state == "not_run" for state in states),
    }


__all__ = [
    "SCHEMA_VERSION",
    "SemanticReview",
    "SemanticReviewBatch",
    "SemanticReviewError",
    "ReviewerKind",
    "SemanticState",
    "load_semantic_review",
    "semantic_review_metadata",
]
