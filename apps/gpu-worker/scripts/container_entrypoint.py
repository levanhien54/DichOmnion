"""Fail-fast production entrypoint for RunPod and other NVIDIA container hosts."""

from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence


_MOSS_TTS_HOST = "127.0.0.1"
_MOSS_TTS_PORT = 9880
_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)
_HF_TOKEN_ENV = "HF_TOKEN"


def _enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise RuntimeError(f"{name} must be a boolean value.")


def _child_environment(*, include_hf_token: bool = False) -> dict[str, str]:
    """Build a child environment with explicit access to the gated-model credential."""
    child_env = dict(os.environ)
    if not include_hf_token:
        child_env.pop(_HF_TOKEN_ENV, None)
    return child_env


def _preload_if_requested() -> None:
    if not _enabled("OMNIVOICE_PRELOAD_MODELS"):
        return
    # The disposable preloader is one of only two children allowed to receive HF_TOKEN.
    preload_env = _child_environment(include_hf_token=True)
    preload_env["TRANSFORMERS_OFFLINE"] = "0"
    preload_env["HF_HUB_OFFLINE"] = "0"
    preload_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    subprocess.run(
        [sys.executable, "/app/scripts/preload_models.py"],
        check=True,
        env=preload_env,
    )


def _validate_gpu_in_subprocess() -> None:
    """Validate CUDA without creating a CUDA context in the long-lived PID 1."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.gpu_runtime import validate_production_gpu; "
                "validate_production_gpu()"
            ),
        ],
        check=True,
        env=_child_environment(),
    )


def _validate_model_cache() -> None:
    if not _enabled("OMNIVOICE_REQUIRE_MODEL_MARKER"):
        return
    from scripts.preload_models import validate_ready_marker

    validate_ready_marker()


def _uvicorn_argv() -> list[str]:
    host = os.environ.get("WORKER_HOST", "0.0.0.0")
    raw_port = os.environ.get("WORKER_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("WORKER_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("WORKER_PORT must be between 1 and 65535.")
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
        "--no-access-log",
    ]


def _moss_tts_endpoint() -> tuple[str, int]:
    host = os.environ.get("MOSS_TTS_HOST", _MOSS_TTS_HOST).strip()
    if host != _MOSS_TTS_HOST:
        raise RuntimeError("MOSS_TTS_HOST must be 127.0.0.1.")

    raw_port = os.environ.get("MOSS_TTS_PORT", str(_MOSS_TTS_PORT)).strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("MOSS_TTS_PORT must be 9880.") from exc
    if port != _MOSS_TTS_PORT:
        raise RuntimeError("MOSS_TTS_PORT must be 9880.")
    return host, port


def _moss_tts_argv() -> list[str]:
    host, port = _moss_tts_endpoint()
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "src.moss_tts_adapter:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
        "--no-access-log",
    ]


def _bounded_float(
    name: str, default: float, minimum: float, maximum: float
) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} is outside the allowed range.")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} is outside the allowed range.")
    return value


def _probe_moss_health() -> bool:
    """Return readiness from a bounded loopback response without retaining its body."""
    host, port = _moss_tts_endpoint()
    timeout = _bounded_float(
        "MOSS_TTS_HEALTH_TIMEOUT_SECONDS", 2.0, 0.1, 10.0
    )
    max_bytes = _bounded_int("MOSS_TTS_HEALTH_MAX_BYTES", 16_384, 256, 65_536)
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.request(
            "GET",
            "/health",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        body = response.read(max_bytes + 1)
        if response.status != 200 or len(body) > max_bytes:
            return False
        payload = json.loads(body.decode("utf-8"))
        return isinstance(payload, dict) and payload.get("ready") is True
    except (OSError, UnicodeError, ValueError, http.client.HTTPException):
        return False
    finally:
        if connection is not None:
            connection.close()


@dataclass
class _SignalState:
    signum: int | None = None


class _ShutdownRequested(RuntimeError):
    def __init__(self, signum: int):
        super().__init__("container shutdown requested")
        self.signum = signum


def _install_signal_handlers() -> tuple[_SignalState, dict[int, Any]]:
    state = _SignalState()
    previous: dict[int, Any] = {}

    def request_shutdown(signum: int, _frame: Any) -> None:
        if state.signum is None:
            state.signum = signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    return state, previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _spawn_process(
    argv: Sequence[str], *, include_hf_token: bool = False
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(argv),
        env=_child_environment(include_hf_token=include_hf_token),
        start_new_session=True,
    )


def _validate_supervisor_configuration() -> None:
    """Reject invalid runtime bounds before starting either resident GPU process."""
    _moss_tts_endpoint()
    _bounded_float("MOSS_TTS_STARTUP_TIMEOUT_SECONDS", 600.0, 1.0, 3600.0)
    _bounded_float("MOSS_TTS_HEALTH_TIMEOUT_SECONDS", 2.0, 0.1, 10.0)
    _bounded_int("MOSS_TTS_HEALTH_MAX_BYTES", 16_384, 256, 65_536)
    _bounded_float("MOSS_TTS_HEALTH_POLL_SECONDS", 0.25, 0.01, 5.0)
    shutdown_drain = _bounded_float(
        "WORKER_SHUTDOWN_DRAIN_SECONDS", 15.0, 0.0, 120.0
    )
    supervisor_grace = _bounded_float(
        "SUPERVISOR_SHUTDOWN_GRACE_SECONDS", 20.0, 0.0, 120.0
    )
    if shutdown_drain >= supervisor_grace:
        raise RuntimeError(
            "WORKER_SHUTDOWN_DRAIN_SECONDS must be less than "
            "SUPERVISOR_SHUTDOWN_GRACE_SECONDS."
        )
    _bounded_float("SUPERVISOR_POLL_SECONDS", 0.1, 0.01, 2.0)


def _wait_for_moss_ready(
    process: subprocess.Popen[bytes], signal_state: _SignalState
) -> None:
    timeout = _bounded_float(
        "MOSS_TTS_STARTUP_TIMEOUT_SECONDS", 600.0, 1.0, 3600.0
    )
    interval = _bounded_float(
        "MOSS_TTS_HEALTH_POLL_SECONDS", 0.25, 0.01, 5.0
    )
    deadline = time.monotonic() + timeout

    while True:
        if signal_state.signum is not None:
            raise _ShutdownRequested(signal_state.signum)
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError("MOSS TTS exited before becoming ready.")
        if _probe_moss_health():
            return
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError("MOSS TTS did not become ready before startup timeout.")
        time.sleep(min(interval, deadline - now))


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif signum == signal.SIGTERM:
            process.terminate()
        elif signum == _KILL_SIGNAL:
            process.kill()
        else:
            process.send_signal(signum)
    except (OSError, ProcessLookupError):
        pass


def _shutdown_children(
    children: Sequence[subprocess.Popen[bytes]],
    initial_signal: int = signal.SIGTERM,
) -> None:
    running = [process for process in children if process.poll() is None]
    for process in running:
        _signal_process_group(process, initial_signal)

    grace = _bounded_float(
        "SUPERVISOR_SHUTDOWN_GRACE_SECONDS", 20.0, 0.0, 120.0
    )
    poll_interval = _bounded_float(
        "SUPERVISOR_POLL_SECONDS", 0.1, 0.01, 2.0
    )
    deadline = time.monotonic() + grace
    while running and time.monotonic() < deadline:
        running = [process for process in running if process.poll() is None]
        if running:
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    for process in running:
        _signal_process_group(process, _KILL_SIGNAL)
    for process in children:
        try:
            process.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _child_exit_status(return_code: int) -> int:
    if return_code > 0:
        return return_code
    if return_code < 0:
        return 128 + abs(return_code)
    # A supervised server exiting cleanly is still an unexpected container failure.
    return 1


def _monitor_children(
    children: Sequence[subprocess.Popen[bytes]], signal_state: _SignalState
) -> int:
    poll_interval = _bounded_float(
        "SUPERVISOR_POLL_SECONDS", 0.1, 0.01, 2.0
    )
    while True:
        if signal_state.signum is not None:
            _shutdown_children(children, signal_state.signum)
            return 128 + signal_state.signum
        for process in children:
            return_code = process.poll()
            if return_code is not None:
                _shutdown_children(children, signal.SIGTERM)
                return _child_exit_status(return_code)
        time.sleep(poll_interval)


def _run_moss_supervisor() -> int:
    _validate_supervisor_configuration()
    signal_state, previous_handlers = _install_signal_handlers()
    children: list[subprocess.Popen[bytes]] = []
    shutdown_signal = signal.SIGTERM
    try:
        # MOSS uses only its pinned offline cache and must never inherit the gated pyannote
        # credential. The worker receives it just long enough to build its resident pipeline.
        moss_process = _spawn_process(_moss_tts_argv())
        children.append(moss_process)
        _wait_for_moss_ready(moss_process, signal_state)
        if signal_state.signum is not None:
            raise _ShutdownRequested(signal_state.signum)

        worker_process = _spawn_process(
            _uvicorn_argv(), include_hf_token=True
        )
        children.append(worker_process)
        os.environ.pop(_HF_TOKEN_ENV, None)
        return _monitor_children(children, signal_state)
    except _ShutdownRequested as exc:
        shutdown_signal = exc.signum
        return 128 + exc.signum
    finally:
        _shutdown_children(children, shutdown_signal)
        _restore_signal_handlers(previous_handlers)


def main() -> None:
    # Keep all CUDA imports and admission allocations out of the long-lived supervisor.
    _validate_gpu_in_subprocess()
    _preload_if_requested()
    _validate_model_cache()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if _enabled("MOSS_TTS_ENABLED"):
        raise SystemExit(_run_moss_supervisor())
    os.execv(sys.executable, _uvicorn_argv())


if __name__ == "__main__":
    main()
