"""Tests for the short-request ANALYZE submit/status bridge."""

import asyncio
import base64
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from src.main import (
    AnalyzePayload,
    _async_analyze_jobs,
    analyze_status,
    submit_analyze,
)


def _recipient_public_key() -> str:
    private_key = ec.generate_private_key(ec.SECP256R1())
    raw = private_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _payload(pub: str, job_id: str = "ASYNC-JOB", attempt: int = 1) -> AnalyzePayload:
    return AnalyzePayload.model_validate(
        {
            "job_id": job_id,
            "attempt": attempt,
            "audio_url": "https://r2/input?sig=x",
            "audio_md5": "abc123",
            "artifact_upload_url": "https://r2/artifact?sig=x",
            "artifact_key": f"artifacts/device/{job_id}/{attempt}.json",
            "encryption_public_key": pub,
            "target_language": "Vietnamese",
            "translation_style": "Natural",
            "source_language": "Chinese",
        }
    )


def test_submit_is_idempotent_and_status_is_bound_to_job_attempt():
    pub = _recipient_public_key()
    started = asyncio.Event()
    release = asyncio.Event()
    result = {
        "schema_version": 1,
        "job_id": "ASYNC-JOB",
        "attempt": 1,
        "result": {"status": "success"},
    }

    async def run():
        async def fake_analyze(payload, token):
            started.set()
            await release.wait()
            return result

        token = {
            "role": "gateway",
            "act": "analyze",
            "jobId": "ASYNC-JOB",
            "attempt": 1,
        }
        with patch("src.main.analyze_audio", new=AsyncMock(side_effect=fake_analyze)) as analyze:
            first = await submit_analyze(_payload(pub), token)
            second = await submit_analyze(_payload(pub), token)
            assert first["status"] == "queued"
            assert second["status"] == "queued"
            await started.wait()
            running = await analyze_status("ASYNC-JOB", 1, token)
            assert running["status"] == "running"
            release.set()
            task = _async_analyze_jobs()["ASYNC-JOB:1"]["task"]
            await task
            completed = await analyze_status("ASYNC-JOB", 1, token)
            assert completed["status"] == "completed"
            assert completed["response"] == result
            analyze.assert_awaited_once()

    asyncio.run(run())


def test_status_rejects_a_token_from_another_attempt():
    pub = _recipient_public_key()

    async def run():
        token = {
            "role": "gateway",
            "act": "analyze",
            "jobId": "ASYNC-JOB",
            "attempt": 1,
        }
        with pytest.raises(HTTPException) as error:
            await analyze_status("ASYNC-JOB", 2, token)
        assert error.value.status_code == 403

    asyncio.run(run())
