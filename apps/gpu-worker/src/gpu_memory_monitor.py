"""Device-wide GPU memory instrumentation for Render acceptance.

PyTorch allocator metrics only describe one process.  MOSS, Demucs, AudioSeal, and
the resident worker models can live in different processes, so right-sizing a Pod
requires a device-wide reading.  This module samples every GPU visible inside the
container through ``nvidia-smi`` and reports only aggregate byte counts.  It never
returns GPU UUIDs, PIDs, command lines, paths, or request data.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


GPU_MEMORY_METRICS_HEADER = "X-OmniVoice-GPU-Memory-Metrics"
GPU_MEMORY_METRICS_SCHEMA_VERSION = 1
GPU_MEMORY_STAGE_ORDER = ("TTS", "Demucs", "Mix", "Watermark")
MIB = 1024**2
MAX_GPU_MEMORY_BYTES = 256 * 1024**3
_HEADER_PREFIX = "v1."


class GpuMemoryMeasurementError(RuntimeError):
    """A sanitized, fail-closed device memory measurement failure."""


@dataclass(frozen=True)
class DeviceMemorySample:
    used_bytes: int
    capacity_bytes: int
    device_count: int


def _parse_nvidia_smi_output(raw: str, *, expected_devices: int) -> DeviceMemorySample:
    """Parse aggregate visible-device memory without retaining device identifiers."""
    if isinstance(expected_devices, bool) or not 1 <= expected_devices <= 8:
        raise GpuMemoryMeasurementError("invalid expected GPU device count")

    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(rows) != expected_devices:
        raise GpuMemoryMeasurementError("unexpected visible GPU device count")

    total_used = 0
    total_capacity = 0
    seen_identifiers: set[str] = set()
    for row in rows:
        columns = [column.strip() for column in row.split(",")]
        if len(columns) != 3:
            raise GpuMemoryMeasurementError("nvidia-smi returned an invalid schema")
        identifier, used_mib_raw, capacity_mib_raw = columns
        if not identifier or identifier in seen_identifiers:
            raise GpuMemoryMeasurementError("nvidia-smi returned invalid GPU identities")
        seen_identifiers.add(identifier)
        if not used_mib_raw.isdecimal() or not capacity_mib_raw.isdecimal():
            raise GpuMemoryMeasurementError("nvidia-smi returned non-integer memory values")
        used_mib = int(used_mib_raw)
        capacity_mib = int(capacity_mib_raw)
        if capacity_mib <= 0 or used_mib < 0 or used_mib > capacity_mib:
            raise GpuMemoryMeasurementError("nvidia-smi returned invalid memory bounds")
        total_used += used_mib * MIB
        total_capacity += capacity_mib * MIB

    if total_capacity <= 0 or total_capacity > MAX_GPU_MEMORY_BYTES:
        raise GpuMemoryMeasurementError("aggregate GPU capacity is outside the accepted bound")
    return DeviceMemorySample(total_used, total_capacity, len(rows))


class NvidiaSmiMemoryReader:
    """Read device-wide memory for every GPU visible in the Pod container."""

    def __init__(self, *, expected_devices: int = 1, executable: str | None = None):
        self.expected_devices = expected_devices
        self.executable = executable or shutil.which("nvidia-smi")

    def __call__(self) -> DeviceMemorySample:
        if not self.executable:
            raise GpuMemoryMeasurementError("nvidia-smi is unavailable")
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    "--query-gpu=uuid,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GpuMemoryMeasurementError("nvidia-smi measurement failed") from exc
        if completed.returncode != 0:
            raise GpuMemoryMeasurementError("nvidia-smi measurement failed")
        return _parse_nvidia_smi_output(
            completed.stdout, expected_devices=self.expected_devices
        )


def _strict_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise GpuMemoryMeasurementError(f"{name} must be a boolean")


def gpu_memory_metrics_enabled() -> bool:
    """Return the explicit per-deployment opt-in, rejecting contradictory config."""
    enabled = _strict_bool_env("GPU_MEMORY_METRICS_ENABLED", False)
    required = _strict_bool_env("GPU_MEMORY_METRICS_REQUIRED", False)
    if required and not enabled:
        raise GpuMemoryMeasurementError(
            "required GPU memory metrics must also be enabled"
        )
    return enabled


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuMemoryMeasurementError(f"{name} must be an integer") from exc
    if isinstance(value, bool) or not minimum <= value <= maximum:
        raise GpuMemoryMeasurementError(f"{name} is outside the accepted bound")
    return value


class GpuMemoryMonitor:
    """Sample device-wide VRAM and attribute peaks to explicit Render stages."""

    def __init__(
        self,
        *,
        required: bool,
        sample_interval_ms: int = 100,
        expected_devices: int = 1,
        reader: Callable[[], DeviceMemorySample] | None = None,
    ):
        if not 50 <= sample_interval_ms <= 5_000:
            raise GpuMemoryMeasurementError("GPU memory sample interval is invalid")
        if not 1 <= expected_devices <= 8:
            raise GpuMemoryMeasurementError("expected GPU device count is invalid")
        self.required = required
        self.sample_interval_ms = sample_interval_ms
        self.expected_devices = expected_devices
        self._reader = reader or NvidiaSmiMemoryReader(
            expected_devices=expected_devices
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_stage: dict[str, Any] | None = None
        self._stages: list[dict[str, Any]] = []
        self._baseline: DeviceMemorySample | None = None
        self._latest: DeviceMemorySample | None = None
        self._peak_used_bytes = 0
        self._failure: GpuMemoryMeasurementError | None = None
        self._enabled = False
        self._finished = False

    @classmethod
    def from_env(cls) -> "GpuMemoryMonitor":
        required = _strict_bool_env("GPU_MEMORY_METRICS_REQUIRED", False)
        interval = _bounded_int_env(
            "GPU_MEMORY_SAMPLE_INTERVAL_MS", 250, 50, 5_000
        )
        expected_devices = _bounded_int_env(
            "GPU_MEMORY_EXPECTED_DEVICES", 1, 1, 8
        )
        return cls(
            required=required,
            sample_interval_ms=interval,
            expected_devices=expected_devices,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _read(self) -> DeviceMemorySample:
        sample = self._reader()
        if (
            not isinstance(sample, DeviceMemorySample)
            or sample.device_count != self.expected_devices
            or sample.capacity_bytes <= 0
            or sample.capacity_bytes > MAX_GPU_MEMORY_BYTES
            or sample.used_bytes < 0
            or sample.used_bytes > sample.capacity_bytes
        ):
            raise GpuMemoryMeasurementError("GPU memory reader returned an invalid sample")
        if self._baseline is not None and (
            sample.capacity_bytes != self._baseline.capacity_bytes
            or sample.device_count != self._baseline.device_count
        ):
            raise GpuMemoryMeasurementError("visible GPU topology changed during Render")
        return sample

    def _record(self, sample: DeviceMemorySample) -> None:
        with self._lock:
            self._latest = sample
            self._peak_used_bytes = max(self._peak_used_bytes, sample.used_bytes)
            if self._active_stage is not None:
                stage = self._active_stage
                stage["peak_used_bytes"] = max(
                    stage["peak_used_bytes"], sample.used_bytes
                )
                stage["sample_count"] += 1

    def _take_sample(self) -> DeviceMemorySample:
        sample = self._read()
        self._record(sample)
        return sample

    def _remember_failure(self, error: GpuMemoryMeasurementError) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
        self._stop_event.set()

    def _sampling_loop(self) -> None:
        interval_seconds = self.sample_interval_ms / 1000.0
        while not self._stop_event.wait(interval_seconds):
            try:
                self._take_sample()
            except GpuMemoryMeasurementError as exc:
                self._remember_failure(exc)
                return
            except Exception as exc:  # sanitize arbitrary reader failures
                self._remember_failure(
                    GpuMemoryMeasurementError("GPU memory sampling failed")
                )
                return

    def start(self) -> bool:
        if self._thread is not None or self._baseline is not None:
            raise GpuMemoryMeasurementError("GPU memory monitor was already started")
        try:
            baseline = self._read()
        except GpuMemoryMeasurementError:
            if self.required:
                raise
            return False
        self._baseline = baseline
        self._record(baseline)
        self._enabled = True
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="gpu-memory-monitor",
            daemon=True,
        )
        self._thread.start()
        return True

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self._enabled:
            yield
            return
        self.begin_stage(name)
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            try:
                self.end_stage()
            except GpuMemoryMeasurementError:
                if not body_failed:
                    raise

    def begin_stage(self, name: str) -> None:
        """Open a stage around an async call without requiring an async context API."""
        if not self._enabled:
            return
        if name not in GPU_MEMORY_STAGE_ORDER:
            raise GpuMemoryMeasurementError("unknown GPU memory stage")
        with self._lock:
            if self._active_stage is not None or any(
                stage["stage"] == name for stage in self._stages
            ):
                raise GpuMemoryMeasurementError("GPU memory stages overlap or repeat")
            stage = {
                "stage": name,
                "start_used_bytes": 0,
                "peak_used_bytes": 0,
                "end_used_bytes": 0,
                "sample_count": 0,
            }
            self._active_stage = stage
        try:
            start = self._take_sample()
            stage["start_used_bytes"] = start.used_bytes
        except GpuMemoryMeasurementError:
            with self._lock:
                if self._active_stage is stage:
                    self._active_stage = None
            raise

    def end_stage(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            stage = self._active_stage
        if stage is None:
            raise GpuMemoryMeasurementError("GPU memory stage is not active")
        try:
            end = self._take_sample()
            stage["end_used_bytes"] = end.used_bytes
        except GpuMemoryMeasurementError as exc:
            self._remember_failure(exc)
            raise
        finally:
            with self._lock:
                if self._active_stage is stage:
                    self._active_stage = None
                self._stages.append(stage)

    def _stop_thread(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_ms / 1000.0 + 0.5))
            if self._thread.is_alive():
                self._remember_failure(
                    GpuMemoryMeasurementError("GPU memory monitor did not stop")
                )
            self._thread = None

    def finish(self) -> dict[str, Any] | None:
        if self._finished:
            raise GpuMemoryMeasurementError("GPU memory monitor was already finished")
        self._finished = True
        if not self._enabled:
            if self.required:
                raise GpuMemoryMeasurementError("required GPU memory monitor is unavailable")
            return None
        try:
            final = self._take_sample()
        finally:
            self._stop_thread()
        if self._failure is not None:
            raise self._failure
        if self._baseline is None:
            raise GpuMemoryMeasurementError("GPU memory baseline is unavailable")
        stage_names = [stage["stage"] for stage in self._stages]
        if stage_names != list(GPU_MEMORY_STAGE_ORDER):
            raise GpuMemoryMeasurementError("GPU memory stage coverage is incomplete")
        if any(
            stage["sample_count"] < 2
            or stage["peak_used_bytes"] < stage["start_used_bytes"]
            or stage["peak_used_bytes"] < stage["end_used_bytes"]
            for stage in self._stages
        ):
            raise GpuMemoryMeasurementError("GPU memory stage samples are invalid")
        return {
            "schema_version": GPU_MEMORY_METRICS_SCHEMA_VERSION,
            "source": "nvidia-smi",
            "scope": "visible_devices_total",
            "device_count": self._baseline.device_count,
            "sample_interval_ms": self.sample_interval_ms,
            "capacity_bytes": self._baseline.capacity_bytes,
            "baseline_used_bytes": self._baseline.used_bytes,
            "final_used_bytes": final.used_bytes,
            "peak_used_bytes": self._peak_used_bytes,
            "stages": [dict(stage) for stage in self._stages],
        }

    def close(self) -> None:
        self._stop_thread()


def _strict_metric_shape(metrics: Any) -> dict[str, Any]:
    """Validate the exact sanitized wire shape used by the response header."""
    keys = {
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
    if not isinstance(metrics, dict) or set(metrics) != keys:
        raise GpuMemoryMeasurementError("GPU memory metrics have an invalid schema")
    device_count = metrics.get("device_count")
    interval = metrics.get("sample_interval_ms")
    capacity = metrics.get("capacity_bytes")
    byte_fields = (
        metrics.get("baseline_used_bytes"),
        metrics.get("final_used_bytes"),
        metrics.get("peak_used_bytes"),
    )
    stages = metrics.get("stages")
    if (
        metrics.get("schema_version") != GPU_MEMORY_METRICS_SCHEMA_VERSION
        or isinstance(metrics.get("schema_version"), bool)
        or metrics.get("source") != "nvidia-smi"
        or metrics.get("scope") != "visible_devices_total"
        or isinstance(device_count, bool)
        or not isinstance(device_count, int)
        or not 1 <= device_count <= 8
        or isinstance(interval, bool)
        or not isinstance(interval, int)
        or not 50 <= interval <= 5_000
        or isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not 0 < capacity <= MAX_GPU_MEMORY_BYTES
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= capacity
            for value in byte_fields
        )
        or metrics["peak_used_bytes"] < metrics["baseline_used_bytes"]
        or metrics["peak_used_bytes"] < metrics["final_used_bytes"]
        or not isinstance(stages, list)
        or len(stages) != len(GPU_MEMORY_STAGE_ORDER)
    ):
        raise GpuMemoryMeasurementError("GPU memory metrics are outside accepted bounds")
    for expected_name, stage in zip(GPU_MEMORY_STAGE_ORDER, stages, strict=True):
        if not isinstance(stage, dict) or set(stage) != stage_keys:
            raise GpuMemoryMeasurementError("GPU memory stage metrics have an invalid schema")
        values = (
            stage.get("start_used_bytes"),
            stage.get("peak_used_bytes"),
            stage.get("end_used_bytes"),
        )
        sample_count = stage.get("sample_count")
        if (
            stage.get("stage") != expected_name
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= capacity
                for value in values
            )
            or stage["peak_used_bytes"] < stage["start_used_bytes"]
            or stage["peak_used_bytes"] < stage["end_used_bytes"]
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 2
        ):
            raise GpuMemoryMeasurementError("GPU memory stage metrics are outside accepted bounds")
    return metrics


def encode_gpu_memory_metrics(metrics: Any) -> str:
    validated = _strict_metric_shape(metrics)
    payload = json.dumps(
        validated, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    if len(encoded) > 6_000:
        raise GpuMemoryMeasurementError("GPU memory metrics header is too large")
    return _HEADER_PREFIX + encoded


def decode_gpu_memory_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.startswith(_HEADER_PREFIX):
        raise GpuMemoryMeasurementError("GPU memory metrics header is missing or invalid")
    encoded = value[len(_HEADER_PREFIX) :]
    if not encoded or len(encoded) > 6_000 or not all(
        character.isalnum() or character in "-_" for character in encoded
    ):
        raise GpuMemoryMeasurementError("GPU memory metrics header is invalid")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        metrics = json.loads(raw.decode("ascii"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuMemoryMeasurementError("GPU memory metrics header is invalid") from exc
    return _strict_metric_shape(metrics)
