"""Focused regression tests for source-context and ASR-confidence prompt wiring."""

import json
from unittest.mock import MagicMock

from src.translation_quality import score_translation
from src.translation_service import TranslationService


def _service() -> TranslationService:
    service = TranslationService()
    service.is_loaded = True
    return service


def _output(ids):
    return json.dumps(
        {
            "segments": [
                {
                    "id": item,
                    "translated_text": f"line {item}",
                    "emotion": "NEUTRAL",
                }
                for item in ids
            ]
        }
    )


def test_translation_prompt_carries_bounded_adjacent_source_context(monkeypatch):
    monkeypatch.setenv("QWEN_SELF_REVIEW", "0")
    service = _service()
    captured = []
    service._generate = MagicMock(
        side_effect=lambda prompt: (captured.append(prompt) or _output([1, 2, 3]))
    )

    service.translate_segments(
        [
            {"id": 1, "start": 0, "end": 1, "text": "first", "speaker": "S1", "confidence": 0.91},
            {"id": 2, "start": 1, "end": 2, "text": "current", "speaker": "S1", "confidence": 0.37},
            {"id": 3, "start": 2, "end": 3, "text": "last", "speaker": "S2", "confidence": 0.82},
        ],
        "Vietnamese",
        "Formal",
        source_language="zh",
    )

    assert len(captured) == 1
    prompt = str(captured[0])
    payload = json.loads(prompt.split("INPUT:\n", 1)[1])
    rows = payload["input_segments"]
    assert rows[0].get("context_before") is None
    assert rows[1]["context_before"] == {
        "id": 1,
        "speaker_id": "S1",
        "original_text": "first",
    }
    assert rows[1]["context_after"] == {
        "id": 3,
        "speaker_id": "S2",
        "original_text": "last",
    }
    assert rows[1]["asr_confidence"] == 0.37
    assert "CONTEXT WINDOW" in prompt
    assert "never merge, split, copy" in prompt


def test_translation_prompt_derives_and_bounds_avg_logprob_confidence(monkeypatch):
    monkeypatch.setenv("QWEN_SELF_REVIEW", "0")
    service = _service()
    captured = []
    service._generate = MagicMock(
        side_effect=lambda prompt: (captured.append(prompt) or _output([1]))
    )

    service.translate_segments(
        [
            {
                "id": 1,
                "start": 0,
                "end": 1,
                "text": "uncertain",
                "speaker": "S1",
                "avg_logprob": -2.0,
            }
        ],
        "Vietnamese",
        "Formal",
        source_language="zh",
    )

    row = json.loads(str(captured[0]).split("INPUT:\n", 1)[1])["input_segments"][0]
    assert row["asr_confidence"] == 0.1353

    captured.clear()
    service._generate.reset_mock()
    service.translate_segments(
        [
            {
                "id": 1,
                "start": 0,
                "end": 1,
                "text": "invalid",
                "speaker": "S1",
                "confidence": 99,
            }
        ],
        "Vietnamese",
        "Formal",
        source_language="zh",
    )
    row = json.loads(str(captured[0]).split("INPUT:\n", 1)[1])["input_segments"][0]
    assert row["asr_confidence"] == 1.0


def test_terminal_punctuation_defaults_on_for_production(monkeypatch):
    monkeypatch.setenv("QWEN_SELF_REVIEW", "0")
    monkeypatch.delenv("TRANSLATION_ENSURE_TERMINAL", raising=False)
    service = _service()
    service._generate = MagicMock(return_value=_output([1]))

    result = service.translate_segments(
        [{"id": 1, "start": 0, "end": 2, "text": "source", "speaker": "S1"}],
        "Vietnamese",
        "Formal",
        source_language="zh",
    )

    assert result[0]["translated_text"] == "line 1."
    assert "missing_terminal_punctuation" not in service.last_quality_reports[0]["issueCodes"]


def test_chinese_written_measurements_match_arabic_target_numbers():
    cases = [
        ("全都是一千兩百項", "Toàn bộ đều dùng 1.200 mục."),
        ("全都是一米四", "Tất cả đều dài 1,4 mét."),
        ("比如三噸魚池", "Ví dụ như bể cá 3 tấn."),
    ]
    for source, target in cases:
        report = score_translation(
            source,
            target,
            source_language="Chinese",
            target_language="Vietnamese",
        )
        assert "number_mismatch" not in report.issues


def test_chinese_written_number_change_still_requires_review():
    report = score_translation(
        "那過立倉也差不多是三噸",
        "Kho chứa cũng khoảng 2 tấn.",
        source_language="Chinese",
        target_language="Vietnamese",
    )
    assert "number_mismatch" in report.issues


def test_approximate_chinese_quantity_is_not_treated_as_exact_number():
    for source in ("幾百條魚", "数百条鱼", "三百多条鱼"):
        report = score_translation(
            source,
            "Vài trăm con cá.",
            source_language="Chinese",
            target_language="Vietnamese",
        )
        assert "number_mismatch" not in report.issues


def test_chinese_percent_and_lexical_compounds_do_not_create_number_mismatch():
    exact_percent = score_translation(
        "百分之三",
        "Chỉ 3%.",
        source_language="Chinese",
        target_language="Vietnamese",
    )
    assert "number_mismatch" not in exact_percent.issues

    for source, target in (
        ("千萬不要這樣做", "Đừng làm như vậy."),
        ("十全十美", "Hoàn hảo."),
    ):
        report = score_translation(
            source,
            target,
            source_language="Chinese",
            target_language="Vietnamese",
        )
        assert "number_mismatch" not in report.issues


def test_chinese_ratio_check_remains_exact_after_lexical_filtering():
    report = score_translation(
        "一比一",
        "Tỷ lệ 1:1.",
        source_language="Chinese",
        target_language="Vietnamese",
    )
    assert "number_mismatch" not in report.issues
