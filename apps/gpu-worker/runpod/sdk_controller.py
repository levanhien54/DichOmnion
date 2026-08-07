"""RunPod Pod controller that publishes a short-lived worker target to Gateway.

The official RunPod Python SDK is used for control-plane reads, an explicitly
authorized resume, and fail-safe stopping of a Pod. The default controller never
creates or deletes a Pod. ``--auto-provision`` opts into the separate REST v2
provisioner, which creates only after a durable Gateway slot lease and hands the
exact resulting Pod ID to this controller. Gateway remains responsible for job
authentication and dispatch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx


TARGET_SCHEMA_VERSION = 1
ADMIN_TARGET_PATH = "/api/admin/worker-target"
TRANSPORTS = frozenset({"https_tunnel", "direct_https", "runpod_proxy"})
STOPPED_STATUSES = frozenset({"EXITED", "STOPPED"})
RUNNING_STATUS = "RUNNING"
PROXY_MAX_REQUEST_MS = 90_000
DEFAULT_LONG_REQUEST_MS = 15 * 60 * 1_000
# A busy liveness signal may extend the ordinary unhealthy grace while one async
# inference is finishing, but never indefinitely.  This matches the Gateway's
# maximum async ANALYZE deadline and bounds paid idle time after a hung task.
MAX_BUSY_LIVENESS_SECONDS = DEFAULT_LONG_REQUEST_MS / 1_000
MAX_TARGET_TTL_MS = 5 * 60 * 1_000
MAX_HEALTH_RESPONSE_BYTES = 64 * 1_024
MAX_LIVENESS_RESPONSE_BYTES = 8 * 1_024
MAX_CONTROL_RESPONSE_BYTES = 4 * 1_024
OFFICIAL_RUNPOD_API_BASE_URL = "https://api.runpod.io"
_FORBIDDEN_SDK_NETWORK_ENV = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
)
_REASON_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_RESOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}")
_GENERATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")


class ControllerError(RuntimeError):
    """Stable failure without provider responses, secrets, or target URLs."""

    def __init__(self, code: str):
        if not _REASON_RE.fullmatch(code):
            raise ValueError("invalid controller reason code")
        super().__init__(code)
        self.code = code


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ControllerError("arguments_invalid")


class PodProvider(Protocol):
    def list_pods(self) -> Sequence[Mapping[str, Any]]: ...

    def resume_pod(self, pod_id: str, gpu_count: int) -> None: ...

    def stop_pod(self, pod_id: str) -> None: ...


class ControllerTransport(Protocol):
    def acquire_control(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None: ...

    def probe_health(self, base_url: str) -> bool: ...

    def probe_liveness(self, base_url: str) -> "LivenessResult": ...

    def publish_target(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None: ...

    def clear_target(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None: ...


@dataclass(frozen=True)
class ControllerConfig:
    api_key: str
    pod_id: str | None
    pod_name: str | None
    gateway_endpoint: str | None
    gateway_admin_token: str | None
    worker_url: str | None
    transport: str | None
    worker_port: int
    max_request_ms: int | None
    target_ttl_ms: int
    heartbeat_seconds: float
    ready_timeout_seconds: float
    unhealthy_grace_seconds: float
    probe_interval_seconds: float
    allow_resume: bool
    max_hourly_cost_usd: Decimal | None
    dry_run: bool


@dataclass(frozen=True)
class SelectedPod:
    pod_id: str
    desired_status: str
    gpu_count: int | None
    hourly_cost_usd: Decimal | None


@dataclass(frozen=True)
class WorkerTarget:
    base_url: str
    transport: str
    max_request_ms: int


@dataclass(frozen=True)
class ReconcileResult:
    ready: bool
    published: bool
    resumed: bool
    dry_run: bool
    reason: str

    def safe_report(self) -> dict[str, object]:
        return {
            "schema": "omnivoice.runpod-sdk-controller.v1",
            "ready": self.ready,
            "published": self.published,
            "resumed": self.resumed,
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LivenessResult:
    """A minimal process signal used while a previously-ready worker is busy."""

    alive: bool
    busy: bool = False


def _clean_secret(value: str | None, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 1_024
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise ControllerError(code)
    return value


def _clean_pod_name(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if (
        value != value.strip()
        or not 1 <= len(value) <= 128
        or not value.isascii()
        or not value.isprintable()
    ):
        raise ControllerError("pod_selector_invalid")
    return value


def _clean_pod_id(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not _RESOURCE_ID_RE.fullmatch(value):
        raise ControllerError("pod_selector_invalid")
    return value


def _secure_sdk_network_environment(
    environ: MutableMapping[str, str],
) -> None:
    """Pin SDK traffic to RunPod and reject inherited proxy/CA interception."""

    base_value: str | None = None
    for key, value in environ.items():
        normalized = key.upper()
        if normalized == "RUNPOD_API_BASE_URL":
            base_value = value
        if normalized in _FORBIDDEN_SDK_NETWORK_ENV and value:
            raise ControllerError("runpod_sdk_network_environment_unsafe")
    if base_value not in (None, OFFICIAL_RUNPOD_API_BASE_URL):
        raise ControllerError("runpod_sdk_api_base_invalid")
    environ["RUNPOD_API_BASE_URL"] = OFFICIAL_RUNPOD_API_BASE_URL


def _positive_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    if isinstance(value, str) and len(value) > 64:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return parsed


def _positive_decimal_argument(value: str) -> Decimal:
    parsed = _positive_decimal(value)
    if parsed is None or parsed > Decimal("100"):
        raise argparse.ArgumentTypeError("invalid positive decimal")
    return parsed


def _bounded_int(
    value: str | int | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
    code: str,
) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ControllerError(code)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ControllerError(code) from None
    if str(parsed) != str(value).strip() or not minimum <= parsed <= maximum:
        raise ControllerError(code)
    return parsed


def _bounded_float(
    value: str | float | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
    code: str,
) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ControllerError(code)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ControllerError(code) from None
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ControllerError(code)
    return parsed


def _normalize_endpoint(value: str | None) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ControllerError("gateway_configuration_invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ControllerError("gateway_configuration_invalid") from None
    host = parsed.hostname
    is_loopback = host in {"localhost", "127.0.0.1", "::1"}
    if (
        host is None
        or not host.isascii()
        or not host.isprintable()
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.scheme not in ({"http", "https"} if is_loopback else {"https"})
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ControllerError("gateway_configuration_invalid")
    path = parsed.path.rstrip("/")
    if not path:
        path = ADMIN_TARGET_PATH
    elif path != ADMIN_TARGET_PATH:
        raise ControllerError("gateway_configuration_invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _normalize_worker_url(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ControllerError("worker_url_invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ControllerError("worker_url_invalid") from None
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or host is None
        or not host.isascii()
        or not host.isprintable()
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/")
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ControllerError("worker_url_invalid")
    normalized = urlunsplit(("https", parsed.netloc, "", "", "")).rstrip("/")
    return normalized, host.lower()


def _validate_config(config: ControllerConfig) -> ControllerConfig:
    if not isinstance(config, ControllerConfig):
        raise ControllerError("controller_configuration_invalid")
    pod_id = _clean_pod_id(config.pod_id)
    pod_name = _clean_pod_name(config.pod_name)
    if (pod_id is None) == (pod_name is None):
        raise ControllerError("pod_selector_invalid")
    api_key = _clean_secret(config.api_key, code="runpod_api_key_invalid")
    if config.transport is not None and config.transport not in TRANSPORTS:
        raise ControllerError("worker_transport_invalid")
    worker_url = config.worker_url
    if config.worker_url is not None:
        worker_url, _host = _normalize_worker_url(config.worker_url)
    worker_port = _bounded_int(
        config.worker_port,
        default=8_000,
        minimum=1,
        maximum=65_535,
        code="worker_port_invalid",
    )
    max_request_ms = None
    if config.max_request_ms is not None:
        max_request_ms = _bounded_int(
            config.max_request_ms,
            default=PROXY_MAX_REQUEST_MS,
            minimum=1_000,
            maximum=900_000,
            code="worker_request_limit_invalid",
        )
    target_ttl_ms = _bounded_int(
        config.target_ttl_ms,
        default=90_000,
        minimum=30_000,
        maximum=MAX_TARGET_TTL_MS,
        code="target_ttl_invalid",
    )
    heartbeat_seconds = _bounded_float(
        config.heartbeat_seconds,
        default=30.0,
        minimum=1.0,
        maximum=300.0,
        code="heartbeat_interval_invalid",
    )
    if heartbeat_seconds * 2_000 > target_ttl_ms:
        raise ControllerError("heartbeat_interval_invalid")
    ready_timeout_seconds = _bounded_float(
        config.ready_timeout_seconds,
        default=1_800.0,
        minimum=0.0,
        maximum=1_800.0,
        code="ready_timeout_invalid",
    )
    unhealthy_grace_seconds = _bounded_float(
        config.unhealthy_grace_seconds,
        default=120.0,
        minimum=0.0,
        maximum=600.0,
        code="unhealthy_grace_invalid",
    )
    probe_interval_seconds = _bounded_float(
        config.probe_interval_seconds,
        default=5.0,
        minimum=0.1,
        maximum=60.0,
        code="probe_interval_invalid",
    )
    if probe_interval_seconds > target_ttl_ms / 2_000:
        raise ControllerError("probe_interval_exceeds_lease_renewal_budget")
    if not isinstance(config.allow_resume, bool) or not isinstance(config.dry_run, bool):
        raise ControllerError("controller_configuration_invalid")
    max_hourly_cost = (
        None
        if config.max_hourly_cost_usd is None
        else _positive_decimal(config.max_hourly_cost_usd)
    )
    if config.max_hourly_cost_usd is not None and max_hourly_cost is None:
        raise ControllerError("resume_cost_guard_invalid")
    if config.allow_resume and max_hourly_cost is None:
        raise ControllerError("resume_cost_guard_required")
    if max_hourly_cost is not None and max_hourly_cost > Decimal("100"):
        raise ControllerError("resume_cost_guard_invalid")
    gateway_endpoint = config.gateway_endpoint
    if gateway_endpoint is not None:
        gateway_endpoint = _normalize_endpoint(gateway_endpoint)
    gateway_admin_token = config.gateway_admin_token
    if gateway_admin_token is not None:
        gateway_admin_token = _clean_secret(
            gateway_admin_token, code="gateway_admin_token_invalid"
        )
    if not config.dry_run:
        if gateway_endpoint is None or gateway_admin_token is None:
            raise ControllerError("gateway_configuration_invalid")
    return replace(
        config,
        api_key=api_key,
        pod_id=pod_id,
        pod_name=pod_name,
        gateway_endpoint=gateway_endpoint,
        gateway_admin_token=gateway_admin_token,
        worker_url=worker_url,
        worker_port=worker_port,
        max_request_ms=max_request_ms,
        target_ttl_ms=target_ttl_ms,
        heartbeat_seconds=heartbeat_seconds,
        ready_timeout_seconds=ready_timeout_seconds,
        unhealthy_grace_seconds=unhealthy_grace_seconds,
        probe_interval_seconds=probe_interval_seconds,
        max_hourly_cost_usd=max_hourly_cost,
    )


def select_exact_pod(
    rows: Sequence[Mapping[str, Any]],
    *,
    pod_id: str | None,
    pod_name: str | None,
) -> SelectedPod:
    """Select exactly one configured Pod and reject ambiguous provider data."""

    clean_id = _clean_pod_id(pod_id)
    clean_name = _clean_pod_name(pod_name)
    if (clean_id is None) == (clean_name is None):
        raise ControllerError("pod_selector_invalid")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ControllerError("pod_inventory_invalid")
    if len(rows) > 10_000 or any(not isinstance(row, Mapping) for row in rows):
        raise ControllerError("pod_inventory_invalid")
    matches = [
        row
        for row in rows
        if (row.get("id") == clean_id if clean_id is not None else row.get("name") == clean_name)
    ]
    if not matches:
        raise ControllerError("pod_not_found")
    if len(matches) != 1:
        raise ControllerError("pod_selector_ambiguous")
    row = matches[0]
    selected_id = row.get("id")
    status = row.get("desiredStatus")
    if (
        not isinstance(selected_id, str)
        or not _RESOURCE_ID_RE.fullmatch(selected_id)
        or not isinstance(status, str)
        or not 1 <= len(status) <= 32
        or status != status.strip()
        or not status.isascii()
        or not status.isprintable()
    ):
        raise ControllerError("pod_inventory_invalid")
    raw_gpu_count = row.get("gpuCount")
    gpu_count = (
        raw_gpu_count
        if isinstance(raw_gpu_count, int)
        and not isinstance(raw_gpu_count, bool)
        and 1 <= raw_gpu_count <= 8
        else None
    )
    return SelectedPod(
        pod_id=selected_id,
        desired_status=status.upper(),
        gpu_count=gpu_count,
        hourly_cost_usd=_positive_decimal(row.get("costPerHr")),
    )


def derive_worker_target(config: ControllerConfig, pod: SelectedPod) -> WorkerTarget:
    """Build a validated target without exposing credentials or URL details."""

    expected_proxy_host = f"{pod.pod_id}-{config.worker_port}.proxy.runpod.net".lower()
    if config.worker_url:
        base_url, host = _normalize_worker_url(config.worker_url)
        transport = config.transport or (
            "runpod_proxy" if host.endswith(".proxy.runpod.net") else "https_tunnel"
        )
        if transport == "runpod_proxy" and host != expected_proxy_host:
            raise ControllerError("worker_proxy_pod_mismatch")
        if transport != "runpod_proxy" and host.endswith(".proxy.runpod.net"):
            raise ControllerError("worker_transport_invalid")
    else:
        if config.transport not in (None, "runpod_proxy"):
            raise ControllerError("worker_url_required")
        base_url = f"https://{expected_proxy_host}"
        transport = "runpod_proxy"
    if transport not in TRANSPORTS:
        raise ControllerError("worker_transport_invalid")
    request_limit = config.max_request_ms
    if request_limit is None:
        request_limit = (
            PROXY_MAX_REQUEST_MS
            if transport == "runpod_proxy"
            else DEFAULT_LONG_REQUEST_MS
        )
    if transport == "runpod_proxy" and request_limit > PROXY_MAX_REQUEST_MS:
        raise ControllerError("worker_proxy_request_limit_invalid")
    return WorkerTarget(
        base_url=base_url,
        transport=transport,
        max_request_ms=request_limit,
    )


class SdkRunPodProvider:
    """Narrow official-SDK adapter; it cannot create or delete resources."""

    def __init__(
        self,
        sdk: Any,
        api_key: str,
        *,
        environ: MutableMapping[str, str] | None = None,
    ):
        if not callable(getattr(sdk, "get_pods", None)) or not callable(
            getattr(sdk, "resume_pod", None)
        ) or not callable(getattr(sdk, "stop_pod", None)):
            raise ControllerError("runpod_sdk_unavailable")
        self._sdk = sdk
        self._api_key = _clean_secret(api_key, code="runpod_api_key_invalid")
        self._environ = os.environ if environ is None else environ
        _secure_sdk_network_environment(self._environ)
        # resume_pod currently reads the module-level key, while get_pods accepts
        # an explicit key. Keep both paths bound to the same server-side secret.
        self._sdk.api_key = self._api_key

    @classmethod
    def load(
        cls,
        api_key: str,
        *,
        environ: MutableMapping[str, str] | None = None,
    ) -> "SdkRunPodProvider":
        active_environ = os.environ if environ is None else environ
        _secure_sdk_network_environment(active_environ)
        try:
            sdk = importlib.import_module("runpod")
        except (ImportError, ModuleNotFoundError):
            raise ControllerError("runpod_sdk_unavailable") from None
        return cls(sdk, api_key, environ=active_environ)

    def list_pods(self) -> Sequence[Mapping[str, Any]]:
        _secure_sdk_network_environment(self._environ)
        try:
            rows = self._sdk.get_pods(api_key=self._api_key)
        except Exception:
            raise ControllerError("runpod_inventory_query_failed") from None
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ControllerError("pod_inventory_invalid")
        return rows

    def resume_pod(self, pod_id: str, gpu_count: int) -> None:
        _secure_sdk_network_environment(self._environ)
        try:
            self._sdk.resume_pod(pod_id, gpu_count)
        except Exception:
            # The outcome can be ambiguous after a transport failure. The
            # controller records the attempt before calling and never blindly
            # repeats it in the same process.
            raise ControllerError("pod_resume_outcome_ambiguous") from None

    def stop_pod(self, pod_id: str) -> None:
        _secure_sdk_network_environment(self._environ)
        try:
            self._sdk.stop_pod(pod_id)
        except Exception:
            raise ControllerError("pod_stop_outcome_ambiguous") from None


class HttpxControllerTransport:
    def __init__(
        self,
        *,
        health_timeout_seconds: float = 5.0,
        gateway_timeout_seconds: float = 10.0,
    ):
        self._health_timeout_seconds = health_timeout_seconds
        self._gateway_timeout_seconds = gateway_timeout_seconds

    def acquire_control(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None:
        lease_endpoint = f"{endpoint.rstrip('/')}/lease"
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self._gateway_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    "POST",
                    lease_endpoint,
                    headers={"X-Admin-Token": admin_token},
                    json=dict(payload),
                ) as response:
                    status = response.status_code
                    content = self._read_bounded_response(
                        response,
                        MAX_CONTROL_RESPONSE_BYTES,
                        invalid_code="gateway_control_lease_response_invalid",
                    )
        except ControllerError:
            raise
        except (httpx.HTTPError, ValueError):
            raise ControllerError("gateway_control_lease_failed") from None

        document: Mapping[str, Any] | None = None
        if content:
            try:
                parsed = json.loads(content)
            except (UnicodeError, ValueError, TypeError):
                parsed = None
            if isinstance(parsed, Mapping):
                document = parsed

        if status == 200:
            if (
                document is not None
                and document.get("granted") is True
                and document.get("outcome") in {"accepted", "idempotent"}
            ):
                return
            raise ControllerError("gateway_control_lease_response_invalid")
        if status == 409:
            raise ControllerError("worker_control_lease_denied")
        if status == 423:
            if document is not None and document.get("outcome") == "kill_switch_active":
                raise ControllerError("worker_control_kill_switch_active")
            raise ControllerError("worker_control_blocked")
        raise ControllerError("gateway_control_lease_failed")

    @staticmethod
    def _read_bounded_response(
        response: httpx.Response,
        maximum_bytes: int,
        *,
        invalid_code: str,
    ) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                raise ControllerError(invalid_code) from None
            if declared_bytes < 0 or declared_bytes > maximum_bytes:
                raise ControllerError(invalid_code)
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ControllerError(invalid_code)
        return bytes(content)

    def probe_health(self, base_url: str) -> bool:
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self._health_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream("GET", f"{base_url}/health") as response:
                    if response.status_code != 200:
                        return False
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            if int(declared) > MAX_HEALTH_RESPONSE_BYTES:
                                return False
                        except ValueError:
                            return False
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_HEALTH_RESPONSE_BYTES:
                            return False
        except (httpx.HTTPError, ValueError):
            return False
        try:
            document = json.loads(bytes(content))
        except (UnicodeError, ValueError, TypeError):
            return False
        return bool(
            isinstance(document, dict)
            and document.get("status") == "ok"
            and document.get("models_loaded") is True
            and document.get("device") == "cuda"
        )

    def probe_liveness(self, base_url: str) -> LivenessResult:
        """Check the lightweight liveness contract used during an active job.

        This is intentionally not a readiness check: the endpoint must not probe
        sidecars or attempt model loads.  A missing endpoint (old image) fails closed
        and preserves the controller's previous unhealthy-grace behaviour.
        """
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self._health_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    "GET", f"{base_url}/api/worker/liveness"
                ) as response:
                    if response.status_code != 200:
                        return LivenessResult(False)
                    content = self._read_bounded_response(
                        response,
                        MAX_LIVENESS_RESPONSE_BYTES,
                        invalid_code="worker_liveness_response_invalid",
                    )
        except (ControllerError, httpx.HTTPError, ValueError):
            return LivenessResult(False)
        try:
            document = json.loads(content)
        except (UnicodeError, ValueError, TypeError):
            return LivenessResult(False)
        if not isinstance(document, Mapping):
            return LivenessResult(False)
        alive = bool(
            document.get("status") == "ok"
            and document.get("models_loaded") is True
            and document.get("device") == "cuda"
        )
        busy = bool(document.get("busy") is True)
        return LivenessResult(alive=alive, busy=busy)

    def publish_target(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None:
        self._gateway_request("POST", endpoint, admin_token, payload)

    def clear_target(
        self, endpoint: str, admin_token: str, payload: Mapping[str, Any]
    ) -> None:
        self._gateway_request("DELETE", endpoint, admin_token, payload)

    def _gateway_request(
        self,
        method: str,
        endpoint: str,
        admin_token: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self._gateway_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    method,
                    endpoint,
                    headers={"X-Admin-Token": admin_token},
                    json=dict(payload),
                ) as response:
                    status = response.status_code
                    self._read_bounded_response(
                        response,
                        MAX_CONTROL_RESPONSE_BYTES,
                        invalid_code="gateway_target_response_invalid",
                    )
        except ControllerError:
            raise
        except (httpx.HTTPError, ValueError):
            raise ControllerError("gateway_target_update_failed") from None
        if status not in {200, 204}:
            raise ControllerError("gateway_target_update_failed")


class RunPodController:
    def __init__(
        self,
        config: ControllerConfig,
        provider: PodProvider,
        transport: ControllerTransport,
        *,
        generation: str | None = None,
        generation_started_at_ms: int | None = None,
        preowned_pod_id: str | None = None,
        wall_time: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = _validate_config(config)
        self.provider = provider
        self.transport = transport
        self.generation = generation or uuid.uuid4().hex
        if not _GENERATION_RE.fullmatch(self.generation):
            raise ControllerError("generation_invalid")
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._sleep = sleep
        now_ms = int(self._wall_time() * 1_000)
        self.generation_started_at_ms = (
            generation_started_at_ms
            if generation_started_at_ms is not None
            else now_ms
        )
        if not 0 < self.generation_started_at_ms <= now_ms + 60_000:
            raise ControllerError("generation_time_invalid")
        self._last_issued_at_ms = self.generation_started_at_ms - 1
        self._owns_target = False
        self._resume_attempted_pod_id: str | None = None
        self._resume_deadline_monotonic: float | None = None
        self._resume_outcome_ambiguous = False
        if preowned_pod_id is not None:
            if (
                not _RESOURCE_ID_RE.fullmatch(preowned_pod_id)
                or self.config.pod_id != preowned_pod_id
            ):
                raise ControllerError("preowned_pod_invalid")
        self._owned_pod_id: str | None = preowned_pod_id
        self._owned_pod_was_ready = False
        self._stop_attempted_pod_id: str | None = None
        self._busy_since_monotonic: float | None = None

    def _next_issued_at_ms(self) -> int:
        issued_at = max(
            int(self._wall_time() * 1_000), self._last_issued_at_ms + 1
        )
        self._last_issued_at_ms = issued_at
        return issued_at

    def _list_and_select(self) -> SelectedPod:
        try:
            rows = self.provider.list_pods()
        except ControllerError:
            raise
        except Exception:
            raise ControllerError("runpod_inventory_query_failed") from None
        return select_exact_pod(
            rows, pod_id=self.config.pod_id, pod_name=self.config.pod_name
        )

    def _check_resume_guard(self, pod: SelectedPod) -> None:
        if not self.config.allow_resume:
            raise ControllerError("pod_resume_not_allowed")
        if self.config.max_hourly_cost_usd is None:
            raise ControllerError("resume_cost_guard_required")
        if pod.hourly_cost_usd is None:
            raise ControllerError("pod_hourly_cost_unavailable")
        if pod.hourly_cost_usd > self.config.max_hourly_cost_usd:
            raise ControllerError("pod_hourly_cost_guard_exceeded")
        if pod.gpu_count is None:
            raise ControllerError("pod_gpu_count_invalid")

    def _control_lease_payload(
        self, pod_id: str, issued_at_ms: int
    ) -> dict[str, object]:
        return {
            "schema_version": TARGET_SCHEMA_VERSION,
            "provider": "runpod",
            "generation": self.generation,
            "generationStartedAt": self.generation_started_at_ms,
            "issuedAt": issued_at_ms,
            "validUntil": issued_at_ms + self.config.target_ttl_ms,
            "podId": pod_id,
        }

    def _acquire_control(self, pod_id: str) -> None:
        if self.config.dry_run:
            return
        assert self.config.gateway_endpoint is not None
        assert self.config.gateway_admin_token is not None
        issued_at = self._next_issued_at_ms()
        self.transport.acquire_control(
            self.config.gateway_endpoint,
            self.config.gateway_admin_token,
            self._control_lease_payload(pod_id, issued_at),
        )

    def _clear_owned_target_best_effort(self) -> None:
        if not self._owns_target:
            return
        try:
            self.clear_owned_target()
        except ControllerError:
            # Clear is generation-scoped and the heartbeat expires shortly. A
            # control-plane clear failure must not prevent a cost-reducing stop.
            self._owns_target = False

    def _stop_owned_pod(self, pod_id: str, cause: ControllerError) -> None:
        if self._owned_pod_id != pod_id:
            raise cause
        if self._stop_attempted_pod_id == pod_id:
            raise ControllerError("pod_stop_reconciliation_pending")

        self._clear_owned_target_best_effort()
        # Always attempt a final lease renewal before stop. A denial cannot veto
        # an emergency cost-reducing stop after this generation resumed the Pod.
        try:
            self._acquire_control(pod_id)
        except ControllerError:
            pass
        self._stop_attempted_pod_id = pod_id
        try:
            self.provider.stop_pod(pod_id)
        except ControllerError:
            raise
        except Exception:
            raise ControllerError("pod_stop_outcome_ambiguous") from None
        raise cause

    def _abandon_owned_pod(self, pod_id: str) -> None:
        if self._owned_pod_id != pod_id:
            return
        self._clear_owned_target_best_effort()
        self._owned_pod_id = None
        self._owned_pod_was_ready = False
        self._resume_deadline_monotonic = None
        self._resume_outcome_ambiguous = False
        self._busy_since_monotonic = None

    def _renew_control_or_stop(self, pod_id: str) -> None:
        try:
            self._acquire_control(pod_id)
        except ControllerError as exc:
            if self._owned_pod_id == pod_id:
                if exc.code == "worker_control_lease_denied":
                    # A 409 means another generation owns a live lease. This
                    # stale controller must not stop the new owner's Pod.
                    self._abandon_owned_pod(pod_id)
                    raise
                self._stop_owned_pod(pod_id, exc)
            raise

    def _wait_for_health(self, pod: SelectedPod, base_url: str) -> bool:
        if self._owns_target:
            deadline = self._monotonic() + self.config.unhealthy_grace_seconds
        elif self._owned_pod_id == pod.pod_id and self._resume_deadline_monotonic is not None:
            deadline = self._resume_deadline_monotonic
        else:
            deadline = self._monotonic() + self.config.ready_timeout_seconds
        while True:
            self._renew_control_or_stop(pod.pod_id)
            if self.transport.probe_health(base_url):
                self._busy_since_monotonic = None
                return True

            # A strict readiness probe may be 503 while a long async job owns the
            # single GPU.  Once this generation has published a READY target, use the
            # separate liveness contract to keep that Pod alive until the job finishes.
            # Initial bootstrap remains strict: an unready Pod is never published.
            if self._owns_target:
                probe_liveness = getattr(self.transport, "probe_liveness", None)
                if callable(probe_liveness):
                    try:
                        liveness = probe_liveness(base_url)
                    except Exception:
                        liveness = LivenessResult(False)
                    if isinstance(liveness, LivenessResult) and liveness.alive:
                        if liveness.busy:
                            now = self._monotonic()
                            if self._busy_since_monotonic is None:
                                self._busy_since_monotonic = now
                            busy_deadline = (
                                self._busy_since_monotonic
                                + MAX_BUSY_LIVENESS_SECONDS
                            )
                            if now < busy_deadline:
                                # Reset the short unhealthy window while work is
                                # demonstrably active; the absolute busy deadline
                                # still guarantees a hung task is eventually stopped.
                                deadline = max(deadline, busy_deadline)
                                self._sleep(
                                    min(
                                        self.config.probe_interval_seconds,
                                        max(0.0, busy_deadline - now),
                                    )
                                )
                                continue
                        else:
                            self._busy_since_monotonic = None

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._busy_since_monotonic = None
                return False
            self._sleep(min(self.config.probe_interval_seconds, remaining))

    def _target_payload(
        self, pod: SelectedPod, target: WorkerTarget, issued_at_ms: int
    ) -> dict[str, object]:
        return {
            "schema_version": TARGET_SCHEMA_VERSION,
            "provider": "runpod",
            "generation": self.generation,
            "generationStartedAt": self.generation_started_at_ms,
            "issuedAt": issued_at_ms,
            "validUntil": issued_at_ms + self.config.target_ttl_ms,
            "baseUrl": target.base_url,
            "transport": target.transport,
            "maxRequestMs": target.max_request_ms,
            "podId": pod.pod_id,
        }

    def clear_owned_target(self) -> bool:
        if self.config.dry_run or not self._owns_target:
            return False
        assert self.config.gateway_endpoint is not None
        assert self.config.gateway_admin_token is not None
        issued_at = self._next_issued_at_ms()
        self.transport.clear_target(
            self.config.gateway_endpoint,
            self.config.gateway_admin_token,
            {"generation": self.generation, "issuedAt": issued_at},
        )
        self._owns_target = False
        self._busy_since_monotonic = None
        return True

    def reconcile(self) -> ReconcileResult:
        try:
            pod = self._list_and_select()
        except ControllerError as exc:
            if exc.code in {
                "pod_inventory_invalid",
                "pod_not_found",
                "pod_selector_ambiguous",
            }:
                self.clear_owned_target()
            raise

        if self._stop_attempted_pod_id == pod.pod_id:
            self._clear_owned_target_best_effort()
            if pod.desired_status in STOPPED_STATUSES:
                self._stop_attempted_pod_id = None
                self._owned_pod_id = None
                self._owned_pod_was_ready = False
                self._resume_deadline_monotonic = None
                self._resume_outcome_ambiguous = False
                self._busy_since_monotonic = None
                raise ControllerError("pod_stop_confirmed")
            # RunPod inventory can remain RUNNING after stop returns or fails
            # ambiguously. Never probe or publish while reconciliation is pending.
            raise ControllerError("pod_stop_reconciliation_pending")

        target = derive_worker_target(self.config, pod)
        resumed = False

        if pod.desired_status != RUNNING_STATUS:
            # A target owned by this generation must stop accepting work before
            # any resume decision or provider mutation is attempted.
            self.clear_owned_target()
            if (
                self._owned_pod_id == pod.pod_id
                and self._resume_deadline_monotonic is not None
            ):
                if self._monotonic() >= self._resume_deadline_monotonic:
                    self._stop_owned_pod(
                        pod.pod_id,
                        ControllerError(
                            "pod_bootstrap_deadline_exceeded_stop_requested"
                        ),
                    )
                raise ControllerError("pod_bootstrap_reconciliation_pending")
            if pod.desired_status not in STOPPED_STATUSES:
                raise ControllerError("pod_not_running")
            self._check_resume_guard(pod)
            if self._resume_attempted_pod_id == pod.pod_id:
                raise ControllerError("pod_resume_requires_reconciliation")
            if self.config.dry_run:
                return ReconcileResult(
                    ready=False,
                    published=False,
                    resumed=False,
                    dry_run=True,
                    reason="resume_policy_validated",
                )
            self._acquire_control(pod.pod_id)
            self._resume_attempted_pod_id = pod.pod_id
            self._resume_deadline_monotonic = (
                self._monotonic() + self.config.ready_timeout_seconds
            )
            self._resume_outcome_ambiguous = False
            self._owned_pod_id = pod.pod_id
            try:
                assert pod.gpu_count is not None
                self.provider.resume_pod(pod.pod_id, pod.gpu_count)
            except ControllerError:
                self._resume_outcome_ambiguous = True
                raise
            except Exception:
                self._resume_outcome_ambiguous = True
                raise ControllerError("pod_resume_outcome_ambiguous") from None
            self._renew_control_or_stop(pod.pod_id)
            resumed = True
        else:
            if self._owned_pod_id != pod.pod_id:
                self._resume_attempted_pod_id = None

        if self.config.dry_run:
            return ReconcileResult(
                ready=False,
                published=False,
                resumed=False,
                dry_run=True,
                reason="configuration_validated",
            )

        if not self._wait_for_health(pod, target.base_url):
            if self._owned_pod_id == pod.pod_id:
                self._stop_owned_pod(
                    pod.pod_id,
                    ControllerError("worker_not_ready_stop_requested"),
                )
            self.clear_owned_target()
            raise ControllerError("worker_not_ready")

        assert self.config.gateway_endpoint is not None
        assert self.config.gateway_admin_token is not None
        self._renew_control_or_stop(pod.pod_id)
        issued_at = self._next_issued_at_ms()
        try:
            self.transport.publish_target(
                self.config.gateway_endpoint,
                self.config.gateway_admin_token,
                self._target_payload(pod, target, issued_at),
            )
        except ControllerError as exc:
            if self._owned_pod_id == pod.pod_id:
                self._stop_owned_pod(pod.pod_id, exc)
            raise
        self._owns_target = True
        if self._owned_pod_id == pod.pod_id:
            self._owned_pod_was_ready = True
            self._resume_deadline_monotonic = None
            self._resume_outcome_ambiguous = False
        self._stop_attempted_pod_id = None
        if self._owned_pod_id != pod.pod_id:
            self._resume_attempted_pod_id = None
        return ReconcileResult(
            ready=True,
            published=True,
            resumed=resumed,
            dry_run=False,
            reason="target_published",
        )


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument(
        "--auto-provision",
        action="store_true",
        help="Provision the configured RunPod slot before starting reconciliation.",
    )
    parser.add_argument(
        "--max-hourly-cost-usd", type=_positive_decimal_argument, default=None
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=None)
    parser.add_argument("--ready-timeout-seconds", type=float, default=None)
    parser.add_argument("--probe-interval-seconds", type=float, default=None)
    return parser


def config_from_environment(
    arguments: argparse.Namespace, environ: Mapping[str, str]
) -> ControllerConfig:
    dry_run = bool(arguments.dry_run)
    gateway_url = environ.get("GATEWAY_ADMIN_URL")
    endpoint = None
    if gateway_url:
        endpoint = _normalize_endpoint(gateway_url)
    worker_port = _bounded_int(
        environ.get("RUNPOD_WORKER_PORT"),
        default=8_000,
        minimum=1,
        maximum=65_535,
        code="worker_port_invalid",
    )
    max_request_raw = environ.get("RUNPOD_WORKER_MAX_REQUEST_MS")
    max_request_ms = (
        None
        if max_request_raw in (None, "")
        else _bounded_int(
            max_request_raw,
            default=PROXY_MAX_REQUEST_MS,
            minimum=1_000,
            maximum=900_000,
            code="worker_request_limit_invalid",
        )
    )
    ttl_seconds = _bounded_int(
        environ.get("RUNPOD_TARGET_TTL_SECONDS"),
        default=90,
        minimum=30,
        maximum=300,
        code="target_ttl_invalid",
    )
    heartbeat_value = (
        arguments.heartbeat_seconds
        if arguments.heartbeat_seconds is not None
        else environ.get("RUNPOD_HEARTBEAT_SECONDS")
    )
    ready_timeout_value = (
        arguments.ready_timeout_seconds
        if arguments.ready_timeout_seconds is not None
        else environ.get("RUNPOD_READY_TIMEOUT_SECONDS")
    )
    unhealthy_grace_value = environ.get("RUNPOD_UNHEALTHY_GRACE_SECONDS")
    probe_interval_value = (
        arguments.probe_interval_seconds
        if arguments.probe_interval_seconds is not None
        else environ.get("RUNPOD_PROBE_INTERVAL_SECONDS")
    )
    return _validate_config(
        ControllerConfig(
            api_key=environ.get("RUNPOD_API_KEY", ""),
            pod_id=environ.get("RUNPOD_POD_ID"),
            pod_name=environ.get("RUNPOD_POD_NAME"),
            gateway_endpoint=endpoint,
            gateway_admin_token=environ.get("GATEWAY_ADMIN_TOKEN"),
            worker_url=environ.get("RUNPOD_WORKER_URL"),
            transport=environ.get("RUNPOD_WORKER_TRANSPORT") or None,
            worker_port=worker_port,
            max_request_ms=max_request_ms,
            target_ttl_ms=ttl_seconds * 1_000,
            heartbeat_seconds=_bounded_float(
                heartbeat_value,
                default=30.0,
                minimum=1.0,
                maximum=300.0,
                code="heartbeat_interval_invalid",
            ),
            ready_timeout_seconds=_bounded_float(
                ready_timeout_value,
                default=1_800.0,
                minimum=0.0,
                maximum=1_800.0,
                code="ready_timeout_invalid",
            ),
            unhealthy_grace_seconds=_bounded_float(
                unhealthy_grace_value,
                default=120.0,
                minimum=0.0,
                maximum=600.0,
                code="unhealthy_grace_invalid",
            ),
            probe_interval_seconds=_bounded_float(
                probe_interval_value,
                default=5.0,
                minimum=0.1,
                maximum=60.0,
                code="probe_interval_invalid",
            ),
            allow_resume=bool(arguments.allow_resume),
            max_hourly_cost_usd=arguments.max_hourly_cost_usd,
            dry_run=dry_run,
        )
    )


def _safe_json(document: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(document), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _backoff_seconds(attempt: int, *, rng: random.Random) -> float:
    base = min(60.0, 2.0 * (2 ** min(attempt, 5)))
    return min(60.0, max(1.0, base * rng.uniform(0.8, 1.2)))


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    provider: PodProvider | None = None,
    transport: ControllerTransport | None = None,
) -> int:
    environment: MutableMapping[str, str] = dict(
        os.environ if environ is None else environ
    )
    preowned_pod_id: str | None = None
    generation: str | None = None
    generation_started_at_ms: int | None = None
    try:
        arguments = _parser().parse_args(argv)
        if arguments.once and arguments.allow_resume:
            raise ControllerError("once_resume_not_allowed")
        if arguments.auto_provision:
            if arguments.once or arguments.dry_run:
                raise ControllerError("auto_provision_requires_daemon")
            if not arguments.allow_resume:
                raise ControllerError("auto_provision_requires_allow_resume")
            if environment.get("RUNPOD_POD_ID") or environment.get("RUNPOD_POD_NAME"):
                raise ControllerError("auto_provision_selector_conflict")
            generation = uuid.uuid4().hex
            generation_started_at_ms = int(time.time() * 1_000)
            try:
                try:
                    from .pod_provisioner import provision_from_environment
                except ImportError:
                    from pod_provisioner import provision_from_environment
                provisioned = provision_from_environment(
                    environment,
                    generation=generation,
                    generation_started_at_ms=generation_started_at_ms,
                )
            except Exception as error:
                code = getattr(error, "code", None)
                raise ControllerError(
                    code
                    if isinstance(code, str) and _REASON_RE.fullmatch(code)
                    else "auto_provision_failed"
                ) from None
            environment["RUNPOD_POD_ID"] = provisioned.pod_id
            environment.pop("RUNPOD_POD_NAME", None)
            if provisioned.status in {"PROVISIONING", "STARTING", "RUNNING"}:
                preowned_pod_id = provisioned.pod_id
        config = config_from_environment(
            arguments, environment
        )
        active_provider = provider or SdkRunPodProvider.load(config.api_key)
        controller = RunPodController(
            config,
            active_provider,
            transport or HttpxControllerTransport(),
            generation=generation,
            generation_started_at_ms=generation_started_at_ms,
            preowned_pod_id=preowned_pod_id,
        )
    except ControllerError as exc:
        print(
            _safe_json(
                {
                    "schema": "omnivoice.runpod-sdk-controller.v1",
                    "ready": False,
                    "reason": exc.code,
                }
            ),
            file=sys.stderr,
        )
        return 2

    once = bool(arguments.once or arguments.dry_run)
    rng = random.SystemRandom()
    failed_attempts = 0
    try:
        while True:
            try:
                result = controller.reconcile()
            except ControllerError as exc:
                report: dict[str, object] = {
                    "schema": "omnivoice.runpod-sdk-controller.v1",
                    "ready": False,
                    "published": False,
                    "dry_run": config.dry_run,
                    "reason": exc.code,
                }
                if once:
                    print(_safe_json(report), file=sys.stderr)
                    return 2
                delay = _backoff_seconds(failed_attempts, rng=rng)
                failed_attempts += 1
                report["retryAfterMs"] = int(delay * 1_000)
                print(_safe_json(report), file=sys.stderr, flush=True)
                time.sleep(delay)
                continue

            print(_safe_json(result.safe_report()), flush=True)
            if once:
                return 0
            failed_attempts = 0
            time.sleep(config.heartbeat_seconds)
    except KeyboardInterrupt:
        try:
            controller.clear_owned_target()
        except ControllerError as exc:
            print(
                _safe_json(
                    {
                        "schema": "omnivoice.runpod-sdk-controller.v1",
                        "ready": False,
                        "reason": exc.code,
                    }
                ),
                file=sys.stderr,
            )
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
