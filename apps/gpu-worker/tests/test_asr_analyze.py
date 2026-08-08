"""M4-S5 — ASR confidence surfacing for the Analyze step.

The legacy `transcribe` (used by the single-shot render pipeline) returns only text +
timings. The Analyze step needs MORE: a per-segment confidence signal so the human editor
can see which lines the model was unsure about, plus the detected source language. Rather
than perturb the render path, Analyze uses a dedicated `transcribe_analyze` that surfaces
faster-whisper's raw `avg_logprob` / `no_speech_prob` and `info.language`.

The real inference is GPU/model-blocked; what these tests pin is that the method faithfully
surfaces the model's own confidence signals instead of inventing them (No-Fake-Success —
confidence must be a real ASR output, not a decorative constant).
"""
from types import SimpleNamespace

import pytest

from src.asr_service import ASRService


class _FakeWhisper:
    """Stand-in for faster-whisper's WhisperModel — returns canned segments + info so the
    surfacing logic is exercised without weights or a GPU."""

    def __init__(self, segments, language):
        self._segments = segments
        self._language = language
        self.last_kwargs = None

    def transcribe(self, audio_path, beam_size=5, **kwargs):
        self.last_kwargs = {"beam_size": beam_size, **kwargs}
        info = SimpleNamespace(language=self._language)
        return iter(self._segments), info


def _seg(start, end, text, avg_logprob, no_speech_prob):
    return SimpleNamespace(
        start=start, end=end, text=text,
        avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
    )


def test_transcribe_analyze_requires_loaded_model():
    svc = ASRService()
    with pytest.raises(RuntimeError):
        svc.transcribe_analyze("/tmp/whatever.wav")


def test_transcribe_analyze_surfaces_language_and_confidence_signals():
    svc = ASRService()
    svc.model = _FakeWhisper(
        segments=[
            _seg(0.0, 1.5, "  Hello there ", avg_logprob=-0.12, no_speech_prob=0.01),
            _seg(1.5, 3.2, "second line", avg_logprob=-1.4, no_speech_prob=0.6),
        ],
        language="en",
    )
    svc.is_loaded = True

    out = svc.transcribe_analyze("/tmp/audio.wav")

    assert out["language"] == "en"
    assert len(out["segments"]) == 2

    s0 = out["segments"][0]
    assert s0["id"] == "sub-1"
    assert s0["start"] == 0.0 and s0["end"] == 1.5
    assert s0["text"] == "Hello there"          # stripped
    assert s0["avg_logprob"] == -0.12           # surfaced verbatim, not invented
    assert s0["no_speech_prob"] == 0.01
    assert s0["speaker"] == "SPEAKER_UNKNOWN"    # real diarization arrives in S7

    s1 = out["segments"][1]
    assert s1["id"] == "sub-2"
    assert s1["avg_logprob"] == -1.4
    assert s1["no_speech_prob"] == 0.6


def test_transcribe_analyze_maps_explicit_source_language_hint():
    svc = ASRService()
    fake = _FakeWhisper(segments=[], language="zh")
    svc.model = fake
    svc.is_loaded = True

    svc.transcribe_analyze("/tmp/audio.wav", language="Chinese")

    assert fake.last_kwargs == {"beam_size": 5, "language": "zh"}


def test_transcribe_analyze_ignores_unsupported_language_hint():
    svc = ASRService()
    fake = _FakeWhisper(segments=[], language="en")
    svc.model = fake
    svc.is_loaded = True

    svc.transcribe_analyze("/tmp/audio.wav", language="not-a-supported-language")

    assert fake.last_kwargs == {"beam_size": 5}
