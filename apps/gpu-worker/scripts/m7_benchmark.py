"""Deterministic M7.0 benchmark runner and sanitized report builder.

This module is measurement infrastructure, not a GPU acceptance gate.  It deliberately
separates a successfully generated report from verified GPU evidence and never evaluates
the product KPI.  A CPU-only machine can therefore exercise the runner without producing
a false GPU pass.

Child commands may write a bounded observation document to the path in
``OMNIVOICE_BENCHMARK_OBSERVATION_PATH``.  Only stage latency and aggregate, device-wide
VRAM fields are accepted; paths, media content, transcripts, credentials, and GPU UUIDs
cannot enter the report through that protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


REPORT_SCHEMA = "omnivoice.m7-benchmark.v1"
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_PATH_ENV = "OMNIVOICE_BENCHMARK_OBSERVATION_PATH"
RUN_INDEX_ENV = "OMNIVOICE_BENCHMARK_RUN_INDEX"
RUN_KIND_ENV = "OMNIVOICE_BENCHMARK_RUN_KIND"
MAX_OBSERVATION_BYTES = 64 * 1024
MAX_STAGE_COUNT = 64
MAX_RUN_COUNT = 100
MAX_GPU_COUNT = 8
MAX_GPU_CAPACITY_BYTES = 256 * 1024**3
MAX_LATENCY_MS = 7 * 24 * 60 * 60 * 1000
MIB = 1024**2

_STAGE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_SCENARIO_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_LANGUAGE_RE = re.compile(r"[A-Za-z][A-Za-z -]{0,31}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}")
_SHA256_RE = re.compile(r"[A-Fa-f0-9]{64}")
_MODEL_ID_RE = re.compile(
    r"(?:hf://)?[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})?"
)
_MODEL_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")


class BenchmarkError(RuntimeError):
    """A sanitized benchmark configuration or measurement failure."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        raise BenchmarkError(f"{name} is outside the accepted bound")
    return value


def _bounded_number(
    value: Any, *, name: str, minimum: float = 0.0, maximum: float = MAX_LATENCY_MS
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise BenchmarkError(f"{name} is outside the accepted bound")
    return normalized


def _safe_token(value: Any, *, name: str, maximum: int = 128) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum or any(ord(ch) < 32 for ch in normalized):
        raise BenchmarkError(f"{name} is invalid")
    return normalized


def _model_component(value: Any) -> str:
    component = _safe_token(value, name="model component", maximum=64)
    if not _STAGE_RE.fullmatch(component):
        raise BenchmarkError("model component is invalid")
    return component


def _model_id(value: Any) -> str:
    model_id = _safe_token(value, name="model id", maximum=256)
    if not _MODEL_ID_RE.fullmatch(model_id):
        raise BenchmarkError("model id is invalid")
    return model_id


def _model_revision(value: Any) -> str:
    revision = _safe_token(value, name="model revision", maximum=128)
    if not _MODEL_REVISION_RE.fullmatch(revision):
        raise BenchmarkError("model revision is invalid")
    return revision


def normalize_sha256(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BenchmarkError("input SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def validate_input_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "media_kind",
        "duration_ms",
        "size_bytes",
        "sha256",
        "speaker_count",
        "languages",
        "sample_rate_hz",
        "channels",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise BenchmarkError("input metadata has an invalid schema")
    media_kind = value["media_kind"]
    if media_kind not in {"audio", "video"}:
        raise BenchmarkError("input media kind is invalid")
    languages_raw = value["languages"]
    if (
        not isinstance(languages_raw, list)
        or not languages_raw
        or len(languages_raw) > 64
        or any(
            not isinstance(language, str) or not _LANGUAGE_RE.fullmatch(language)
            for language in languages_raw
        )
    ):
        raise BenchmarkError("input language metadata is invalid")
    languages = sorted(set(languages_raw))
    if len(languages) != len(languages_raw):
        raise BenchmarkError("input language metadata contains duplicates")
    sample_rate = value["sample_rate_hz"]
    channels = value["channels"]
    if sample_rate is not None:
        sample_rate = _bounded_int(
            sample_rate, name="sample rate", minimum=8_000, maximum=384_000
        )
    if channels is not None:
        channels = _bounded_int(channels, name="channels", minimum=1, maximum=32)
    return {
        "media_kind": media_kind,
        "duration_ms": _bounded_int(
            value["duration_ms"], name="input duration", minimum=1, maximum=86_400_000
        ),
        "size_bytes": _bounded_int(
            value["size_bytes"],
            name="input size",
            minimum=1,
            maximum=20 * 1024**3,
        ),
        "sha256": normalize_sha256(value["sha256"]),
        "speaker_count": _bounded_int(
            value["speaker_count"], name="speaker count", minimum=1, maximum=64
        ),
        "languages": languages,
        "sample_rate_hz": sample_rate,
        "channels": channels,
    }


def validate_models(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not 1 <= len(value) <= 64
    ):
        raise BenchmarkError("model revision metadata has an invalid schema")
    expected_keys = {"component", "model_id", "revision", "revision_evidence"}
    allowed_evidence = {"source_pin", "declared_runtime_snapshot", "unresolved"}
    result = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise BenchmarkError("model revision metadata has an invalid schema")
        component = _model_component(row["component"])
        model_id = _model_id(row["model_id"])
        evidence = row["revision_evidence"]
        revision_raw = row["revision"]
        if evidence not in allowed_evidence:
            raise BenchmarkError("model revision evidence is invalid")
        if revision_raw is None:
            revision = None
        else:
            revision = _model_revision(revision_raw)
        if (revision is None) != (evidence == "unresolved"):
            raise BenchmarkError("model revision evidence is inconsistent")
        if component in seen:
            raise BenchmarkError("model revision metadata repeats a component")
        seen.add(component)
        result.append(
            {
                "component": component,
                "model_id": model_id,
                "revision": revision,
                "revision_evidence": evidence,
            }
        )
    return sorted(result, key=lambda row: row["component"])


def percentile(
    values: Sequence[float],
    quantile: float,
    *,
    maximum: float = MAX_LATENCY_MS,
) -> float:
    """Return a deterministic Hyndman-Fan type-7 percentile."""

    if not values:
        raise BenchmarkError("cannot summarize an empty measurement set")
    if not 0 <= quantile <= 100:
        raise BenchmarkError("percentile is outside the accepted bound")
    ordered = sorted(
        _bounded_number(item, name="measurement", maximum=maximum) for item in values
    )
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    values: Sequence[float],
    *,
    digits: int = 3,
    maximum: float = MAX_LATENCY_MS,
) -> dict[str, Any]:
    if not values:
        raise BenchmarkError("cannot summarize an empty measurement set")
    normalized = [
        _bounded_number(item, name="measurement", maximum=maximum) for item in values
    ]

    def rounded(value: float) -> float:
        return round(value, digits)

    return {
        "sample_count": len(normalized),
        "min": rounded(min(normalized)),
        "mean": rounded(math.fsum(normalized) / len(normalized)),
        "p50": rounded(percentile(normalized, 50, maximum=maximum)),
        "p90": rounded(percentile(normalized, 90, maximum=maximum)),
        "p95": rounded(percentile(normalized, 95, maximum=maximum)),
        "p99": rounded(percentile(normalized, 99, maximum=maximum)),
        "max": rounded(max(normalized)),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_model_revisions() -> list[dict[str, Any]]:
    """Record source-pinned revisions and make every unresolved revision explicit."""

    # Metadata collection must not inherit optional audio/FFmpeg import warnings. The runtime
    # itself remains untouched; only this side-effect-free fingerprint import is quieted.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preload = importlib.import_module("scripts.preload_models")
    fingerprint = preload.cache_fingerprint()
    rows = [
        (
            "audioseal",
            fingerprint["audioseal_repo_id"],
            fingerprint["audioseal_revision"],
            "source_pin",
        ),
        (
            "demucs",
            fingerprint["demucs_model_id"],
            None,
            "unresolved",
        ),
        (
            "moss_codec",
            fingerprint["moss_tts_codec_id"],
            fingerprint["moss_tts_codec_revision"],
            "source_pin",
        ),
        (
            "moss_tts",
            fingerprint["moss_tts_model_id"],
            fingerprint["moss_tts_model_revision"],
            "source_pin",
        ),
        (
            "pyannote_diarization",
            fingerprint["pyannote_pipeline_id"],
            None,
            "unresolved",
        ),
        (
            "pyannote_embedding",
            fingerprint["pyannote_embedding_id"],
            None,
            "unresolved",
        ),
        (
            "pyannote_segmentation",
            fingerprint["pyannote_segmentation_id"],
            None,
            "unresolved",
        ),
        (
            "qwen_translation",
            fingerprint["qwen_model_id"],
            None,
            "unresolved",
        ),
        (
            "whisper_asr",
            fingerprint["whisper_model_size"],
            None,
            "unresolved",
        ),
    ]
    return [
        {
            "component": component,
            "model_id": model_id,
            "revision": revision,
            "revision_evidence": evidence,
        }
        for component, model_id, revision, evidence in sorted(rows)
    ]


def load_model_manifest(path: Path) -> list[dict[str, Any]]:
    """Load an explicit runtime snapshot manifest without treating it as source proof."""

    try:
        if path.stat().st_size > MAX_OBSERVATION_BYTES:
            raise BenchmarkError("model manifest exceeds the accepted size")
        document = json.loads(path.read_text(encoding="utf-8"))
    except BenchmarkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("model manifest is unreadable") from exc
    if not isinstance(document, list) or not 1 <= len(document) <= 64:
        raise BenchmarkError("model manifest has an invalid schema")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_keys = {"component", "model_id", "revision"}
    for row in document:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise BenchmarkError("model manifest has an invalid schema")
        component = _model_component(row["component"])
        model_id = _model_id(row["model_id"])
        revision = _model_revision(row["revision"])
        if component in seen:
            raise BenchmarkError("model manifest repeats a component")
        seen.add(component)
        result.append(
            {
                "component": component,
                "model_id": model_id,
                "revision": revision,
                "revision_evidence": "declared_runtime_snapshot",
            }
        )
    return sorted(result, key=lambda row: row["component"])


def _run_probe(command: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkError("hardware probe failed") from exc


def _parse_gpu_inventory(raw: str) -> list[dict[str, Any]]:
    rows = list(csv.reader(line for line in raw.splitlines() if line.strip()))
    if not 1 <= len(rows) <= MAX_GPU_COUNT:
        raise BenchmarkError("GPU inventory has an invalid device count")
    devices: list[dict[str, Any]] = []
    driver_versions: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise BenchmarkError("GPU inventory has an invalid schema")
        name = _safe_token(row[0], name="GPU name", maximum=128)
        driver = row[1].strip()
        memory_raw = row[2].strip()
        if not _VERSION_RE.fullmatch(driver) or not memory_raw.isdecimal():
            raise BenchmarkError("GPU inventory contains an invalid value")
        memory_bytes = int(memory_raw) * MIB
        if not 0 < memory_bytes <= MAX_GPU_CAPACITY_BYTES:
            raise BenchmarkError("GPU inventory memory is outside the accepted bound")
        driver_versions.add(driver)
        devices.append({"name": name, "memory_total_bytes": memory_bytes})
    if len(driver_versions) != 1:
        raise BenchmarkError("GPU inventory reports inconsistent driver versions")
    for device in devices:
        device["driver_version"] = next(iter(driver_versions))
    return devices


def collect_hardware() -> tuple[dict[str, Any], dict[str, str]]:
    """Collect bounded, non-identifying hardware facts and CUDA evidence."""

    hardware: dict[str, Any] = {
        "os": platform.system().lower() or "unknown",
        "os_release": platform.release()[:64] or "unknown",
        "architecture": platform.machine().lower() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "gpus": [],
        "nvidia_driver_version": None,
        "cuda_runtime_version": None,
        "torch_version": _package_version("torch"),
    }
    executable = shutil.which("nvidia-smi")
    if not executable:
        return hardware, {"status": "unavailable", "reason": "nvidia_smi_unavailable"}
    try:
        completed = _run_probe(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if completed.returncode != 0:
            return hardware, {"status": "unavailable", "reason": "nvidia_smi_failed"}
        devices = _parse_gpu_inventory(completed.stdout)
    except BenchmarkError:
        return hardware, {"status": "unavailable", "reason": "gpu_inventory_invalid"}
    hardware["gpus"] = devices
    hardware["nvidia_driver_version"] = devices[0]["driver_version"]

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        runtime = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        return hardware, {"status": "unavailable", "reason": "torch_cuda_probe_failed"}
    if (
        not cuda_available
        or cuda_count != len(devices)
        or not isinstance(runtime, str)
        or not _VERSION_RE.fullmatch(runtime)
    ):
        return hardware, {"status": "unavailable", "reason": "cuda_runtime_unavailable"}
    hardware["cuda_runtime_version"] = runtime
    return hardware, {"status": "measured", "reason": "cuda_and_nvidia_smi_verified"}


def _read_device_memory(executable: str, expected_devices: int) -> dict[str, int]:
    completed = _run_probe(
        [
            executable,
            "--query-gpu=uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        raise BenchmarkError("GPU memory probe failed")
    rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
    if len(rows) != expected_devices:
        raise BenchmarkError("GPU topology changed during benchmark")
    identifiers: set[str] = set()
    used_bytes = 0
    capacity_bytes = 0
    for row in rows:
        if len(row) != 3:
            raise BenchmarkError("GPU memory probe has an invalid schema")
        identifier, used_raw, capacity_raw = (item.strip() for item in row)
        if (
            not identifier
            or identifier in identifiers
            or not used_raw.isdecimal()
            or not capacity_raw.isdecimal()
        ):
            raise BenchmarkError("GPU memory probe has an invalid value")
        identifiers.add(identifier)
        used = int(used_raw) * MIB
        capacity = int(capacity_raw) * MIB
        if capacity <= 0 or used < 0 or used > capacity:
            raise BenchmarkError("GPU memory probe is outside the accepted bound")
        used_bytes += used
        capacity_bytes += capacity
    if not 0 < capacity_bytes <= MAX_GPU_CAPACITY_BYTES:
        raise BenchmarkError("aggregate GPU capacity is outside the accepted bound")
    return {
        "used_bytes": used_bytes,
        "capacity_bytes": capacity_bytes,
        "device_count": expected_devices,
    }


def _normalize_memory_sample(sample: Any) -> dict[str, int]:
    required = {"used_bytes", "capacity_bytes", "device_count"}
    if not isinstance(sample, dict) or set(sample) != required:
        raise BenchmarkError("GPU memory reader returned an invalid schema")
    capacity = _bounded_int(
        sample["capacity_bytes"],
        name="GPU capacity",
        minimum=1,
        maximum=MAX_GPU_CAPACITY_BYTES,
    )
    used = _bounded_int(
        sample["used_bytes"], name="GPU used memory", minimum=0, maximum=capacity
    )
    devices = _bounded_int(
        sample["device_count"], name="GPU device count", minimum=1, maximum=MAX_GPU_COUNT
    )
    return {
        "used_bytes": used,
        "capacity_bytes": capacity,
        "device_count": devices,
    }


class _MemorySampler:
    def __init__(
        self,
        reader: Callable[[], dict[str, int]],
        *,
        interval_ms: int,
    ) -> None:
        self.reader = reader
        self.interval_ms = _bounded_int(
            interval_ms, name="sample interval", minimum=50, maximum=5_000
        )
        self._samples: list[dict[str, int]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BenchmarkError | None = None

    def _record(self) -> None:
        normalized = _normalize_memory_sample(self.reader())
        with self._lock:
            if self._samples and (
                normalized["capacity_bytes"] != self._samples[0]["capacity_bytes"]
                or normalized["device_count"] != self._samples[0]["device_count"]
            ):
                raise BenchmarkError("GPU topology changed during benchmark")
            self._samples.append(normalized)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            try:
                self._record()
            except Exception:
                self._failure = BenchmarkError("GPU memory sampling failed")
                self._stop.set()
                return

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def start(self) -> None:
        self._record()
        self._thread = threading.Thread(
            target=self._loop, name="m7-benchmark-vram", daemon=True
        )
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 1000.0 + 0.5))
            if self._thread.is_alive():
                raise BenchmarkError("GPU memory sampler did not stop")
        if self._failure is not None:
            raise self._failure
        self._record()
        with self._lock:
            samples = list(self._samples)
        baseline = samples[0]
        final = samples[-1]
        peak = max(sample["used_bytes"] for sample in samples)
        return {
            "schema_version": 1,
            "source": "nvidia-smi",
            "scope": "visible_devices_total",
            "device_count": baseline["device_count"],
            "sample_interval_ms": self.interval_ms,
            "capacity_bytes": baseline["capacity_bytes"],
            "baseline_used_bytes": baseline["used_bytes"],
            "final_used_bytes": final["used_bytes"],
            "peak_used_bytes": peak,
            "stages": [
                {
                    "stage": "command",
                    "start_used_bytes": baseline["used_bytes"],
                    "peak_used_bytes": peak,
                    "end_used_bytes": final["used_bytes"],
                    "sample_count": len(samples),
                }
            ],
        }


def _validate_gpu_memory(
    value: Any, *, allow_command_stage: bool = False
) -> dict[str, Any]:
    top_keys = {
        "schema_version",
        "source",
        "scope",
        "device_count",
        "sample_interval_ms",
        "capacity_bytes",
        "baseline_used_bytes",
        "final_used_bytes",
        "peak_used_bytes",
        "stages",
    }
    stage_keys = {
        "stage",
        "start_used_bytes",
        "peak_used_bytes",
        "end_used_bytes",
        "sample_count",
    }
    if not isinstance(value, dict) or set(value) != top_keys:
        raise BenchmarkError("GPU memory observation has an invalid schema")
    if (
        value["schema_version"] != 1
        or value["source"] != "nvidia-smi"
        or value["scope"] != "visible_devices_total"
    ):
        raise BenchmarkError("GPU memory observation has an invalid provenance")
    devices = _bounded_int(
        value["device_count"], name="GPU device count", minimum=1, maximum=MAX_GPU_COUNT
    )
    interval = _bounded_int(
        value["sample_interval_ms"],
        name="GPU sample interval",
        minimum=50,
        maximum=5_000,
    )
    capacity = _bounded_int(
        value["capacity_bytes"],
        name="GPU capacity",
        minimum=1,
        maximum=MAX_GPU_CAPACITY_BYTES,
    )
    baseline = _bounded_int(
        value["baseline_used_bytes"], name="GPU baseline", minimum=0, maximum=capacity
    )
    final = _bounded_int(
        value["final_used_bytes"], name="GPU final", minimum=0, maximum=capacity
    )
    peak = _bounded_int(
        value["peak_used_bytes"], name="GPU peak", minimum=0, maximum=capacity
    )
    if peak < baseline or peak < final:
        raise BenchmarkError("GPU memory observation has inconsistent peaks")
    raw_stages = value["stages"]
    if not isinstance(raw_stages, list) or not 1 <= len(raw_stages) <= MAX_STAGE_COUNT:
        raise BenchmarkError("GPU memory observation has invalid stage coverage")
    stages = []
    seen: set[str] = set()
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict) or set(raw_stage) != stage_keys:
            raise BenchmarkError("GPU memory stage has an invalid schema")
        stage_name = raw_stage["stage"]
        if not isinstance(stage_name, str) or not _STAGE_RE.fullmatch(stage_name):
            raise BenchmarkError("GPU memory stage name is invalid")
        canonical_name = stage_name.casefold()
        if canonical_name == "command" and not allow_command_stage:
            raise BenchmarkError("GPU memory stage name is reserved")
        if canonical_name in seen:
            raise BenchmarkError("GPU memory observation repeats a stage")
        seen.add(canonical_name)
        start = _bounded_int(
            raw_stage["start_used_bytes"], name="stage start", minimum=0, maximum=capacity
        )
        stage_peak = _bounded_int(
            raw_stage["peak_used_bytes"], name="stage peak", minimum=0, maximum=capacity
        )
        end = _bounded_int(
            raw_stage["end_used_bytes"], name="stage end", minimum=0, maximum=capacity
        )
        sample_count = _bounded_int(
            raw_stage["sample_count"], name="stage samples", minimum=2, maximum=10_000_000
        )
        if stage_peak < start or stage_peak < end or stage_peak > peak:
            raise BenchmarkError("GPU memory stage has inconsistent peaks")
        stages.append(
            {
                "stage": canonical_name,
                "start_used_bytes": start,
                "peak_used_bytes": stage_peak,
                "end_used_bytes": end,
                "sample_count": sample_count,
            }
        )
    return {
        "schema_version": 1,
        "source": "nvidia-smi",
        "scope": "visible_devices_total",
        "device_count": devices,
        "sample_interval_ms": interval,
        "capacity_bytes": capacity,
        "baseline_used_bytes": baseline,
        "final_used_bytes": final,
        "peak_used_bytes": peak,
        "stages": stages,
    }


def load_observation(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        if not path.is_file() or path.stat().st_size > MAX_OBSERVATION_BYTES:
            raise BenchmarkError("benchmark observation exceeds the accepted size")
        document = json.loads(path.read_text(encoding="utf-8"))
    except BenchmarkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("benchmark observation is unreadable") from exc
    expected_keys = {"schema_version", "stage_latencies_ms", "gpu_memory"}
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise BenchmarkError("benchmark observation has an invalid schema")
    if document["schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise BenchmarkError("benchmark observation schema is unsupported")
    raw_latencies = document["stage_latencies_ms"]
    if not isinstance(raw_latencies, dict) or len(raw_latencies) > MAX_STAGE_COUNT:
        raise BenchmarkError("stage latency observation has an invalid schema")
    latencies: dict[str, float] = {}
    for raw_name, raw_latency in raw_latencies.items():
        if not isinstance(raw_name, str) or not _STAGE_RE.fullmatch(raw_name):
            raise BenchmarkError("stage latency name is invalid")
        name = raw_name.casefold()
        if name == "command" or name in latencies:
            raise BenchmarkError("stage latency observation repeats a reserved stage")
        latencies[name] = round(
            _bounded_number(raw_latency, name="stage latency"), 3
        )
    memory = None if document["gpu_memory"] is None else _validate_gpu_memory(document["gpu_memory"])
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "stage_latencies_ms": dict(sorted(latencies.items())),
        "gpu_memory": memory,
    }


def _empty_memory_summary(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "scope": "visible_devices_total",
        "capacity_bytes": None,
        "baseline_used_bytes": None,
        "final_used_bytes": None,
        "peak_used_bytes": None,
        "stage_peak_used_bytes": {},
    }


def _measurement_stage_signature(
    run: Mapping[str, Any],
) -> tuple[tuple[str, ...], str | None, tuple[str, ...] | None]:
    latencies = run.get("stage_latencies_ms")
    if not isinstance(latencies, dict) or not latencies:
        raise BenchmarkError("run latency record has an invalid schema")
    latency_stages = tuple(sorted(latencies))
    memory = run.get("gpu_memory")
    if memory is None:
        return latency_stages, None, None
    if not isinstance(memory, Mapping):
        raise BenchmarkError("run GPU memory record has an invalid schema")
    source = memory.get("stage_detail_source")
    detail = memory.get("stage_detail") or memory.get("command_envelope")
    if source not in {"child_observation", "command_envelope"} or not isinstance(
        detail, Mapping
    ):
        raise BenchmarkError("run GPU memory record has an invalid schema")
    stages = detail.get("stages")
    if not isinstance(stages, list) or not stages:
        raise BenchmarkError("run GPU memory record has an invalid schema")
    names = []
    for stage in stages:
        if not isinstance(stage, Mapping) or not isinstance(stage.get("stage"), str):
            raise BenchmarkError("run GPU memory record has an invalid schema")
        names.append(stage["stage"])
    return latency_stages, source, tuple(sorted(names))


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    measured = [
        run
        for run in runs
        if run.get("kind") == "measurement" and run.get("outcome") == "success"
    ]
    if not measured:
        return {}, _empty_memory_summary("unavailable", "no_successful_measurements")
    signatures = {_measurement_stage_signature(run) for run in measured}
    if len(signatures) != 1:
        raise BenchmarkError("measurement stage coverage changed between runs")
    latency_values: dict[str, list[float]] = {}
    for run in measured:
        latencies = run.get("stage_latencies_ms")
        if not isinstance(latencies, dict):
            raise BenchmarkError("run latency record has an invalid schema")
        for stage, value in latencies.items():
            latency_values.setdefault(stage, []).append(float(value))
    latency_summary = {
        stage: summarize(values) for stage, values in sorted(latency_values.items())
    }

    if any(run.get("gpu_memory") is None for run in measured):
        return latency_summary, _empty_memory_summary(
            "unavailable", "one_or_more_runs_lack_verified_vram"
        )
    envelopes = [run["gpu_memory"]["command_envelope"] for run in measured]
    capacities = {item["capacity_bytes"] for item in envelopes}
    device_counts = {item["device_count"] for item in envelopes}
    if len(capacities) != 1 or len(device_counts) != 1:
        raise BenchmarkError("GPU topology changed between benchmark runs")
    stage_values: dict[str, list[float]] = {}
    for run in measured:
        memory = run["gpu_memory"]
        detail = memory["stage_detail"] or memory["command_envelope"]
        for stage in detail["stages"]:
            stage_values.setdefault(stage["stage"], []).append(stage["peak_used_bytes"])
    memory_summary = {
        "status": "measured",
        "reason": "all_measurements_have_device_wide_samples",
        "scope": "visible_devices_total",
        "device_count": next(iter(device_counts)),
        "capacity_bytes": next(iter(capacities)),
        "baseline_used_bytes": summarize(
            [item["baseline_used_bytes"] for item in envelopes],
            digits=0,
            maximum=MAX_GPU_CAPACITY_BYTES,
        ),
        "final_used_bytes": summarize(
            [item["final_used_bytes"] for item in envelopes],
            digits=0,
            maximum=MAX_GPU_CAPACITY_BYTES,
        ),
        "peak_used_bytes": summarize(
            [item["peak_used_bytes"] for item in envelopes],
            digits=0,
            maximum=MAX_GPU_CAPACITY_BYTES,
        ),
        "stage_peak_used_bytes": {
            stage: summarize(values, digits=0, maximum=MAX_GPU_CAPACITY_BYTES)
            for stage, values in sorted(stage_values.items())
        },
    }
    return latency_summary, memory_summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_coverage(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resolved = sum(1 for model in models if model.get("revision") is not None)
    return {"resolved": resolved, "total": len(models), "complete": resolved == len(models)}


def build_report(
    *,
    scenario: str,
    input_metadata: Mapping[str, Any],
    hardware: Mapping[str, Any],
    initial_gpu_evidence: Mapping[str, str],
    models: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    requested_warmups: int,
    requested_iterations: int,
    generated_at_utc: str | None = None,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    latency_summary, memory_summary = aggregate_runs(runs)
    validated_input = validate_input_metadata(input_metadata)
    validated_models = validate_models(models)
    successful_measurements = sum(
        run.get("kind") == "measurement" and run.get("outcome") == "success"
        for run in runs
    )
    all_successful = (
        blocked_reason is None
        and len(runs) == requested_warmups + requested_iterations
        and all(run.get("outcome") == "success" for run in runs)
        and successful_measurements == requested_iterations
    )
    measurement_status = "completed" if all_successful else "blocked" if blocked_reason else "failed"
    hardware_status = initial_gpu_evidence.get("status")
    if hardware_status != "measured":
        gpu_evidence = dict(initial_gpu_evidence)
    elif memory_summary["status"] != "measured":
        gpu_evidence = {"status": "incomplete", "reason": memory_summary["reason"]}
    else:
        gpu_evidence = {
            "status": "measured",
            "reason": "cuda_runtime_and_device_wide_vram_verified",
        }
    return {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "scenario": scenario,
        "measurement_status": measurement_status,
        "blocked_reason": blocked_reason,
        "evidence": {
            "gpu": gpu_evidence,
            "report_scope": "measurement_only",
        },
        "claims": {
            "gpu_acceptance": False,
            "kpi_evaluated": False,
            "reason": "separate_acceptance_policy_required",
        },
        "input": validated_input,
        "hardware": dict(hardware),
        "software": {
            "packages": {
                name: _package_version(name)
                for name in (
                    "audioseal",
                    "demucs",
                    "faster-whisper",
                    "torch",
                    "torchaudio",
                    "transformers",
                    "whisperx",
                )
            }
        },
        "models": validated_models,
        "model_revision_coverage": _model_coverage(validated_models),
        "execution": {
            "requested_warmups": requested_warmups,
            "requested_iterations": requested_iterations,
            "completed_warmups": sum(
                run.get("kind") == "warmup" and run.get("outcome") == "success"
                for run in runs
            ),
            "completed_iterations": successful_measurements,
            "percentile_method": "Hyndman-Fan type 7",
        },
        "stage_latency_ms": latency_summary,
        "gpu_memory": memory_summary,
        "runs": list(runs),
    }


class _WindowsJob:
    """Windows Job Object containment with kill-on-close and active-process accounting."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are unavailable")
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "could not create workload job")
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "could not configure workload job")

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._accounting_type = _BasicAccountingInformation
        self._handle = handle
        self._closed = False

    def assign_and_resume(self, process: subprocess.Popen[Any]) -> None:
        process_handle = self._wintypes.HANDLE(int(process._handle))
        if not self._kernel32.AssignProcessToJobObject(self._handle, process_handle):
            raise OSError(
                self._ctypes.get_last_error(), "could not contain workload process"
            )
        psutil.Process(process.pid).resume()

    def active_processes(self) -> int:
        information = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            1,
            self._ctypes.byref(information),
            self._ctypes.sizeof(information),
            None,
        ):
            raise OSError(
                self._ctypes.get_last_error(), "could not inspect workload job"
            )
        return int(information.ActiveProcesses)

    def terminate_and_verify(self, *, grace_seconds: float) -> bool:
        verified = bool(self._kernel32.TerminateJobObject(self._handle, 1))
        deadline = time.monotonic() + grace_seconds
        while verified:
            try:
                if self.active_processes() == 0:
                    return True
            except OSError:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if not self._closed:
            self._kernel32.CloseHandle(self._handle)
            self._closed = True


def _terminate_uncontained_process(
    process: subprocess.Popen[Any], *, grace_seconds: float = 1.0
) -> bool:
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/F"],
                    check=False,
                    capture_output=True,
                    timeout=grace_seconds,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None


def _initial_process_tree(
    process: subprocess.Popen[Any],
) -> tuple[dict[int, psutil.Process], bool]:
    try:
        root = psutil.Process(process.pid)
        return {root.pid: root}, True
    except psutil.Error:
        return {}, False


def _refresh_process_tree(targets: dict[int, psutil.Process]) -> bool:
    verified = True
    for target in list(targets.values()):
        try:
            targets.update(
                {child.pid: child for child in target.children(recursive=True)}
            )
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            verified = False
    return verified


def _wait_for_process_tree(
    process: subprocess.Popen[Any],
    targets: dict[int, psutil.Process],
    *,
    timeout_seconds: float,
    abort_requested: Callable[[], bool] | None = None,
) -> tuple[str, bool]:
    deadline = time.monotonic() + timeout_seconds
    tracking_verified = True
    while True:
        tracking_verified = _refresh_process_tree(targets) and tracking_verified
        if process.poll() is not None:
            tracking_verified = _refresh_process_tree(targets) and tracking_verified
            return "completed", tracking_verified
        if abort_requested is not None and abort_requested():
            return "aborted", tracking_verified
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout", tracking_verified
        try:
            process.wait(timeout=min(0.05, remaining))
        except subprocess.TimeoutExpired:
            continue


def _has_live_descendants(
    root_pid: int, targets: Mapping[int, psutil.Process]
) -> bool:
    for pid, target in targets.items():
        if pid == root_pid:
            continue
        try:
            if target.is_running() and target.status() != psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return True
    return False


def _posix_process_group_active(process: subprocess.Popen[Any]) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    known_targets: Mapping[int, psutil.Process] | None = None,
    tracking_verified: bool = True,
    windows_job: _WindowsJob | None = None,
    grace_seconds: float = 1.0,
) -> bool:
    """Terminate and verify the isolated workload tree without exposing process details."""

    if windows_job is not None:
        try:
            try:
                cleanup_verified = windows_job.terminate_and_verify(
                    grace_seconds=grace_seconds
                )
            except Exception:
                cleanup_verified = False
        finally:
            windows_job.close()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            cleanup_verified = False
        return cleanup_verified and process.poll() is not None

    cleanup_verified = tracking_verified
    targets = dict(known_targets or {})
    try:
        root = psutil.Process(process.pid)
        targets[root.pid] = root
        try:
            targets.update({child.pid: child for child in root.children(recursive=True)})
        except psutil.Error:
            cleanup_verified = False
    except psutil.NoSuchProcess:
        root = None
    except psutil.Error:
        root = None
        cleanup_verified = False

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_verified = False
    elif process.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=grace_seconds,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    for target in targets.values():
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            cleanup_verified = False
    if process.poll() is None and root is None:
        try:
            process.terminate()
        except OSError:
            cleanup_verified = False

    try:
        _, alive = psutil.wait_procs(list(targets.values()), timeout=grace_seconds)
    except psutil.Error:
        alive = list(targets.values())
        cleanup_verified = False

    # Refresh descendants while the root is still addressable, then force-kill survivors.
    for target in list(alive):
        try:
            targets.update(
                {child.pid: child for child in target.children(recursive=True)}
            )
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            cleanup_verified = False
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_verified = False
    for target in targets.values():
        try:
            if target.is_running() and target.status() != psutil.STATUS_ZOMBIE:
                target.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            cleanup_verified = False

    try:
        _, alive = psutil.wait_procs(list(targets.values()), timeout=grace_seconds)
    except psutil.Error:
        alive = list(targets.values())
        cleanup_verified = False
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        cleanup_verified = False
        try:
            process.kill()
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.SubprocessError):
            cleanup_verified = False

    if os.name != "nt":
        deadline = time.monotonic() + grace_seconds
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            except OSError:
                cleanup_verified = False
                break
            if time.monotonic() >= deadline:
                cleanup_verified = False
                break
            time.sleep(0.05)

    survivors = []
    for target in alive:
        try:
            if target.is_running() and target.status() != psutil.STATUS_ZOMBIE:
                survivors.append(target)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            cleanup_verified = False
    return cleanup_verified and not survivors and process.poll() is not None


def _run_once(
    command: Sequence[str],
    *,
    kind: str,
    index: int,
    timeout_seconds: float,
    memory_reader: Callable[[], dict[str, int]] | None,
    sample_interval_ms: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnivoice-m7-benchmark-") as directory:
        observation_path = Path(directory) / "observation.json"
        environment = os.environ.copy()
        environment[OBSERVATION_PATH_ENV] = str(observation_path)
        environment[RUN_INDEX_ENV] = str(index)
        environment[RUN_KIND_ENV] = kind
        sampler = (
            _MemorySampler(memory_reader, interval_ms=sample_interval_ms)
            if memory_reader is not None
            else None
        )
        memory_envelope = None
        memory_error = None
        if sampler is not None:
            try:
                sampler.start()
            except Exception:
                return {
                    "kind": kind,
                    "index": index,
                    "outcome": "failed",
                    "exit_code": None,
                    "error_code": "vram_sampling_failed",
                    "stage_latencies_ms": {"command": 0.0},
                    "gpu_memory": None,
                }
        started = time.perf_counter_ns()
        exit_code: int | None = None
        error_code: str | None = None
        windows_job: _WindowsJob | None = None
        try:
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                windows_job = _WindowsJob()
                popen_options["creationflags"] = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | 0x00000004
                )
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(
                list(command), env=environment, shell=False, **popen_options
            )
            if windows_job is not None:
                try:
                    windows_job.assign_and_resume(process)
                except Exception as exc:
                    windows_job.terminate_and_verify(grace_seconds=1.0)
                    windows_job.close()
                    if not _terminate_uncontained_process(process):
                        raise BenchmarkError(
                            "workload containment cleanup failed"
                        ) from exc
                    raise OSError("workload containment failed") from exc
            tracked_processes, initial_tracking_verified = _initial_process_tree(process)
            try:
                process_state, continuous_tracking_verified = _wait_for_process_tree(
                    process,
                    tracked_processes,
                    timeout_seconds=timeout_seconds,
                    abort_requested=(
                        (lambda: sampler.failed) if sampler is not None else None
                    ),
                )
            except BaseException:
                _terminate_process_tree(
                    process,
                    known_targets=tracked_processes,
                    tracking_verified=initial_tracking_verified,
                    windows_job=windows_job,
                )
                raise
            if process_state == "completed" and windows_job is not None:
                try:
                    background_process_detected = windows_job.active_processes() > 0
                except OSError:
                    background_process_detected = True
            else:
                background_process_detected = (
                    process_state == "completed"
                    and (
                        _has_live_descendants(process.pid, tracked_processes)
                        or _posix_process_group_active(process)
                    )
                )
            cleanup_verified = _terminate_process_tree(
                process,
                known_targets=tracked_processes,
                tracking_verified=(
                    initial_tracking_verified and continuous_tracking_verified
                ),
                windows_job=windows_job,
            )
            if process_state == "timeout":
                error_code = (
                    "command_timeout"
                    if cleanup_verified
                    else "command_timeout_cleanup_failed"
                )
            elif process_state == "aborted":
                error_code = (
                    "vram_sampling_failed"
                    if cleanup_verified
                    else "vram_sampling_cleanup_failed"
                )
            else:
                exit_code = process.returncode
                if not cleanup_verified:
                    error_code = "process_tree_cleanup_failed"
                elif background_process_detected:
                    error_code = "background_process_detected"
                elif exit_code != 0:
                    error_code = "command_failed"
        except (OSError, ValueError, subprocess.SubprocessError):
            if windows_job is not None:
                windows_job.close()
            error_code = "command_start_failed"
        elapsed_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
        if sampler is not None:
            try:
                memory_envelope = _validate_gpu_memory(
                    sampler.finish(), allow_command_stage=True
                )
            except BenchmarkError:
                memory_error = "vram_sampling_failed"
        observation = None
        if error_code is None:
            try:
                observation = load_observation(observation_path)
            except BenchmarkError:
                error_code = "observation_invalid"
        latencies = {"command": elapsed_ms}
        if observation is not None:
            latencies.update(observation["stage_latencies_ms"])
        stage_detail = observation["gpu_memory"] if observation is not None else None
        if stage_detail is not None and memory_envelope is None:
            error_code = "unverified_stage_vram"
        if stage_detail is not None and memory_envelope is not None and (
            stage_detail["capacity_bytes"] != memory_envelope["capacity_bytes"]
            or stage_detail["device_count"] != memory_envelope["device_count"]
        ):
            error_code = "gpu_topology_mismatch"
        if (
            stage_detail is not None
            and memory_envelope is not None
            and stage_detail["peak_used_bytes"] > memory_envelope["peak_used_bytes"]
        ):
            error_code = "stage_vram_exceeds_envelope"
        gpu_memory = (
            {
                "command_envelope": memory_envelope,
                "stage_detail": stage_detail,
                "stage_detail_source": (
                    "child_observation" if stage_detail is not None else "command_envelope"
                ),
            }
            if memory_envelope is not None
            else None
        )
        if memory_error is not None and error_code is None:
            error_code = memory_error
        return {
            "kind": kind,
            "index": index,
            "outcome": "success" if error_code is None else "failed",
            "exit_code": exit_code,
            "error_code": error_code,
            "stage_latencies_ms": dict(sorted(latencies.items())),
            "gpu_memory": gpu_memory,
        }


def run_benchmark(
    *,
    command: Sequence[str],
    scenario: str,
    input_metadata: Mapping[str, Any],
    warmups: int,
    iterations: int,
    timeout_seconds: float,
    sample_interval_ms: int,
    require_gpu: bool,
    hardware_collector: Callable[[], tuple[dict[str, Any], dict[str, str]]] = collect_hardware,
    models: Sequence[Mapping[str, Any]] | None = None,
    memory_reader: Callable[[], dict[str, int]] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise BenchmarkError("benchmark command is required")
    if not _SCENARIO_RE.fullmatch(scenario):
        raise BenchmarkError("scenario name is invalid")
    warmups = _bounded_int(warmups, name="warmups", minimum=0, maximum=MAX_RUN_COUNT)
    iterations = _bounded_int(
        iterations, name="iterations", minimum=1, maximum=MAX_RUN_COUNT
    )
    if warmups + iterations > MAX_RUN_COUNT:
        raise BenchmarkError("total benchmark run count is outside the accepted bound")
    timeout_seconds = _bounded_number(
        timeout_seconds, name="timeout", minimum=1.0, maximum=24 * 60 * 60
    )
    sample_interval_ms = _bounded_int(
        sample_interval_ms, name="sample interval", minimum=50, maximum=5_000
    )
    input_metadata = validate_input_metadata(input_metadata)
    selected_models = validate_models(
        list(models if models is not None else collect_model_revisions())
    )
    hardware, gpu_evidence = hardware_collector()
    if gpu_evidence.get("status") == "measured" and memory_reader is None:
        executable = shutil.which("nvidia-smi")
        if not executable:
            gpu_evidence = {"status": "unavailable", "reason": "nvidia_smi_unavailable"}
        else:
            expected_devices = len(hardware.get("gpus", []))
            memory_reader = lambda: _read_device_memory(executable, expected_devices)
    if gpu_evidence.get("status") == "measured" and memory_reader is not None:
        try:
            preflight = _normalize_memory_sample(memory_reader())
            if preflight["device_count"] != len(hardware.get("gpus", [])):
                raise BenchmarkError("GPU topology does not match hardware inventory")
        except Exception:
            gpu_evidence = {
                "status": "unavailable",
                "reason": "gpu_memory_preflight_failed",
            }
            memory_reader = None
    if require_gpu and gpu_evidence.get("status") != "measured":
        return build_report(
            scenario=scenario,
            input_metadata=input_metadata,
            hardware=hardware,
            initial_gpu_evidence=gpu_evidence,
            models=selected_models,
            runs=[],
            requested_warmups=warmups,
            requested_iterations=iterations,
            generated_at_utc=generated_at_utc,
            blocked_reason="verified_gpu_required",
        )
    runs: list[dict[str, Any]] = []
    measurement_signature = None
    for kind, count in (("warmup", warmups), ("measurement", iterations)):
        for index in range(1, count + 1):
            run = _run_once(
                command,
                kind=kind,
                index=index,
                timeout_seconds=timeout_seconds,
                memory_reader=memory_reader,
                sample_interval_ms=sample_interval_ms,
            )
            if kind == "measurement" and run["outcome"] == "success":
                current_signature = _measurement_stage_signature(run)
                if measurement_signature is None:
                    measurement_signature = current_signature
                elif current_signature != measurement_signature:
                    run = dict(run)
                    run["outcome"] = "failed"
                    run["error_code"] = "stage_coverage_mismatch"
            runs.append(run)
            if run["outcome"] != "success":
                return build_report(
                    scenario=scenario,
                    input_metadata=input_metadata,
                    hardware=hardware,
                    initial_gpu_evidence=gpu_evidence,
                    models=selected_models,
                    runs=runs,
                    requested_warmups=warmups,
                    requested_iterations=iterations,
                    generated_at_utc=generated_at_utc,
                )
    return build_report(
        scenario=scenario,
        input_metadata=input_metadata,
        hardware=hardware,
        initial_gpu_evidence=gpu_evidence,
        models=selected_models,
        runs=runs,
        requested_warmups=warmups,
        requested_iterations=iterations,
        generated_at_utc=generated_at_utc,
    )


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def preflight_report_destination(path: Path) -> Path:
    resolved = path.resolve()
    try:
        if resolved.exists() and not resolved.is_file():
            raise BenchmarkError("report output is not a file")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists() and not os.access(resolved, os.W_OK):
            raise BenchmarkError("report output is not writable")
        descriptor, probe = tempfile.mkstemp(
            prefix=f".{resolved.name}.preflight.", dir=resolved.parent
        )
        os.close(descriptor)
        Path(probe).unlink()
    except BenchmarkError:
        raise
    except OSError as exc:
        raise BenchmarkError("report output is not writable") from exc
    return resolved


def _input_metadata(args: argparse.Namespace) -> dict[str, Any]:
    languages = sorted(set(args.language))
    if not languages or any(not _LANGUAGE_RE.fullmatch(item) for item in languages):
        raise BenchmarkError("input language metadata is invalid")
    return {
        "media_kind": args.media_kind,
        "duration_ms": round(
            _bounded_number(
                args.input_duration_seconds,
                name="input duration",
                minimum=0.001,
                maximum=24 * 60 * 60,
            )
            * 1000
        ),
        "size_bytes": _bounded_int(
            args.input_size_bytes,
            name="input size",
            minimum=1,
            maximum=20 * 1024**3,
        ),
        "sha256": normalize_sha256(args.input_sha256),
        "speaker_count": _bounded_int(
            args.speaker_count, name="speaker count", minimum=1, maximum=64
        ),
        "languages": languages,
        "sample_rate_hz": args.sample_rate_hz,
        "channels": args.channels,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sanitized M7.0 benchmark and write a versioned JSON report."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-duration-seconds", required=True, type=float)
    parser.add_argument("--input-size-bytes", required=True, type=int)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--speaker-count", required=True, type=int)
    parser.add_argument("--language", action="append", required=True)
    parser.add_argument("--media-kind", choices=("audio", "video"), default="audio")
    parser.add_argument("--sample-rate-hz", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        if args.sample_rate_hz is not None:
            _bounded_int(
                args.sample_rate_hz,
                name="sample rate",
                minimum=8_000,
                maximum=384_000,
            )
        if args.channels is not None:
            _bounded_int(args.channels, name="channels", minimum=1, maximum=32)
        models = (
            load_model_manifest(args.model_manifest)
            if args.model_manifest is not None
            else None
        )
        output = preflight_report_destination(args.output)
        report = run_benchmark(
            command=command,
            scenario=args.scenario,
            input_metadata=_input_metadata(args),
            warmups=args.warmups,
            iterations=args.iterations,
            timeout_seconds=args.timeout_seconds,
            sample_interval_ms=args.sample_interval_ms,
            require_gpu=args.require_gpu,
            models=models,
        )
        write_report(output, report)
    except BenchmarkError as exc:
        parser.error(str(exc))
    summary = {
        "gpu_evidence": report["evidence"]["gpu"]["status"],
        "measurement_status": report["measurement_status"],
        "output_written": True,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["measurement_status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
