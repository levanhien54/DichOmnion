import importlib.util
import json
import sys
import types
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import httpx
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "runpod" / "sdk_controller.py"
)
SPEC = importlib.util.spec_from_file_location("runpod_sdk_controller", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


NOW_MS = 1_800_000_000_000


def _pod(
    *,
    pod_id="pod_123",
    name="omnivoice-production",
    status="RUNNING",
    cost=Decimal("0.44"),
    gpu_count=1,
):
    return {
        "id": pod_id,
        "name": name,
        "desiredStatus": status,
        "costPerHr": cost,
        "gpuCount": gpu_count,
    }


def _config(**updates):
    base = controller.ControllerConfig(
        api_key="rpa_" + "K" * 48,
        pod_id="pod_123",
        pod_name=None,
        gateway_endpoint="https://gateway.example/api/admin/worker-target",
        gateway_admin_token="admin-token-value",
        worker_url="https://worker-tunnel.example",
        transport="https_tunnel",
        worker_port=8000,
        max_request_ms=900_000,
        target_ttl_ms=90_000,
        heartbeat_seconds=30.0,
        ready_timeout_seconds=0.0,
        unhealthy_grace_seconds=0.0,
        probe_interval_seconds=1.0,
        allow_resume=False,
        max_hourly_cost_usd=None,
        dry_run=False,
    )
    return replace(base, **updates)


class FakeProvider:
    def __init__(self, rows, *, resume_error=None, stop_error=None):
        self.rows = rows
        self.resume_error = resume_error
        self.stop_error = stop_error
        self.list_calls = 0
        self.resume_calls = []
        self.stop_calls = []

    def list_pods(self):
        self.list_calls += 1
        return self.rows

    def resume_pod(self, pod_id, gpu_count):
        self.resume_calls.append((pod_id, gpu_count))
        if self.resume_error is not None:
            raise self.resume_error

    def stop_pod(self, pod_id):
        self.stop_calls.append(pod_id)
        if self.stop_error is not None:
            raise self.stop_error


class FakeTransport:
    def __init__(self, health=True, *, lease_errors=None, liveness=None):
        self.health = health
        self.liveness = liveness
        self.lease_errors = list(lease_errors or [])
        self.leases = []
        self.probes = []
        self.liveness_probes = []
        self.published = []
        self.cleared = []

    def acquire_control(self, endpoint, admin_token, payload):
        self.leases.append((endpoint, admin_token, dict(payload)))
        if self.lease_errors:
            error = self.lease_errors.pop(0)
            if error is not None:
                raise error

    def probe_health(self, base_url):
        self.probes.append(base_url)
        if isinstance(self.health, list):
            return self.health.pop(0)
        return self.health

    def probe_liveness(self, base_url):
        self.liveness_probes.append(base_url)
        value = self.liveness
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, controller.LivenessResult):
            return value
        if isinstance(value, tuple):
            return controller.LivenessResult(*value)
        return controller.LivenessResult(bool(value))

    def publish_target(self, endpoint, admin_token, payload):
        self.published.append((endpoint, admin_token, dict(payload)))

    def clear_target(self, endpoint, admin_token, payload):
        self.cleared.append((endpoint, admin_token, dict(payload)))


class SafeSdkStub:
    api_key = None

    @staticmethod
    def get_pods(*, api_key):
        del api_key
        return []

    @staticmethod
    def resume_pod(pod_id, gpu_count):
        del pod_id, gpu_count

    @staticmethod
    def stop_pod(pod_id):
        del pod_id


def _active_controller(config, provider, transport, *, wall_time=None):
    return controller.RunPodController(
        config,
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        wall_time=wall_time or (lambda: NOW_MS / 1000),
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )


class AdvancingClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class SharedLeaseTransport(FakeTransport):
    def __init__(self):
        super().__init__(health=True)
        self.holder = None

    def acquire_control(self, endpoint, admin_token, payload):
        super().acquire_control(endpoint, admin_token, payload)
        identity = (
            payload["generation"],
            payload["generationStartedAt"],
            payload["podId"],
        )
        if self.holder is None:
            self.holder = identity
        elif self.holder != identity:
            raise controller.ControllerError("worker_control_lease_denied")


def test_exact_name_selection_fails_closed_on_duplicate_pods():
    rows = [
        _pod(pod_id="pod_123", name="same-name"),
        _pod(pod_id="pod_456", name="same-name"),
    ]

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.select_exact_pod(rows, pod_id=None, pod_name="same-name")

    assert exc_info.value.code == "pod_selector_ambiguous"


def test_auto_provision_handoff_marks_running_pod_owned_for_cost_containment():
    provider = FakeProvider([_pod(status="RUNNING")])
    transport = FakeTransport(health=True)
    active = controller.RunPodController(
        _config(),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        preowned_pod_id="pod_123",
        wall_time=lambda: NOW_MS / 1000,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    result = active.reconcile()

    assert result.published is True
    assert len(transport.leases) >= 2
    assert transport.published


def test_auto_provision_handoff_rejects_a_different_pod_id():
    with pytest.raises(controller.ControllerError) as exc_info:
        controller.RunPodController(
            _config(),
            FakeProvider([_pod()]),
            FakeTransport(),
            preowned_pod_id="pod_other",
        )

    assert exc_info.value.code == "preowned_pod_invalid"


def test_auto_provision_main_hands_exact_id_to_controller(monkeypatch):
    captured = {}

    class Provisioned:
        pod_id = "pod_new"
        status = "RUNNING"

    def provision_from_environment(environ, **kwargs):
        captured["environment"] = environ
        captured["kwargs"] = kwargs
        return Provisioned()

    fake_module = types.SimpleNamespace(
        provision_from_environment=provision_from_environment,
    )
    monkeypatch.setitem(sys.modules, "pod_provisioner", fake_module)

    class OneShotController:
        def __init__(self, config, provider, transport, **kwargs):
            captured["config"] = config
            captured["provider"] = provider
            captured["transport"] = transport
            captured["controller_kwargs"] = kwargs

        def reconcile(self):
            raise KeyboardInterrupt

        def clear_owned_target(self):
            return False

    monkeypatch.setattr(controller, "RunPodController", OneShotController)
    environment = {
        "RUNPOD_API_KEY": "rpa_" + "K" * 48,
        "GATEWAY_ADMIN_URL": "https://gateway.example",
        "GATEWAY_ADMIN_TOKEN": "gateway-admin-token",
        "RUNPOD_WORKER_URL": "https://worker-tunnel.example",
        "RUNPOD_WORKER_TRANSPORT": "https_tunnel",
        "RUNPOD_PROVISION_SPEC_PATH": "spec.json",
        "RUNPOD_PROVISION_PLACEMENT_PATH": "placement.json",
        "RUNPOD_PROVISION_STATE_PATH": "intent.json",
    }

    result = controller.main(
        ["--auto-provision", "--allow-resume", "--max-hourly-cost-usd", "0.50"],
        environ=environment,
        provider=FakeProvider([_pod(pod_id="pod_new")]),
        transport=FakeTransport(),
    )

    assert result == 0
    assert captured["config"].pod_id == "pod_new"
    assert captured["controller_kwargs"]["preowned_pod_id"] == "pod_new"
    assert captured["kwargs"]["generation"]
    assert captured["environment"]["RUNPOD_POD_ID"] == "pod_new"


def test_exact_id_selection_does_not_fall_back_to_similar_name():
    selected = controller.select_exact_pod(
        [
            _pod(pod_id="pod_other", name="pod_123"),
            _pod(pod_id="pod_123", name="expected"),
        ],
        pod_id="pod_123",
        pod_name=None,
    )

    assert selected.pod_id == "pod_123"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, "pod_resume_not_allowed"),
        (
            {
                "allow_resume": True,
                "max_hourly_cost_usd": Decimal("0.40"),
            },
            "pod_hourly_cost_guard_exceeded",
        ),
    ],
)
def test_stopped_pod_is_not_resumed_without_complete_cost_authorization(
    changes, reason
):
    provider = FakeProvider([_pod(status="EXITED", cost=Decimal("0.44"))])
    transport = FakeTransport()
    active = _active_controller(_config(**changes), provider, transport)

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == reason
    assert provider.resume_calls == []
    assert transport.probes == []
    assert transport.published == []
    assert transport.cleared == []


def test_missing_provider_cost_blocks_resume_even_with_allow_flag():
    provider = FakeProvider([_pod(status="STOPPED", cost=0)])
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        FakeTransport(),
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "pod_hourly_cost_unavailable"
    assert provider.resume_calls == []


def test_dry_run_validates_resume_guard_without_any_remote_mutation_or_probe():
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport()
    active = _active_controller(
        _config(
            gateway_endpoint=None,
            gateway_admin_token=None,
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
            dry_run=True,
        ),
        provider,
        transport,
    )

    result = active.reconcile()

    assert result.reason == "resume_policy_validated"
    assert result.dry_run is True
    assert provider.resume_calls == []
    assert transport.probes == []
    assert transport.published == []
    assert transport.cleared == []


def test_resume_uses_existing_gpu_count_and_publishes_only_after_cuda_health():
    provider = FakeProvider([_pod(status="EXITED", gpu_count=2)])
    transport = FakeTransport(health=True)
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )

    result = active.reconcile()

    assert result.ready is True
    assert result.resumed is True
    assert provider.resume_calls == [("pod_123", 2)]
    assert transport.probes == ["https://worker-tunnel.example"]
    assert len(transport.published) == 1
    assert len(transport.leases) == 4


def test_competing_generations_cannot_both_resume_the_same_pod():
    transport = SharedLeaseTransport()
    first_provider = FakeProvider([_pod(status="EXITED")])
    second_provider = FakeProvider([_pod(status="EXITED")])
    first = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        first_provider,
        transport,
    )
    second = controller.RunPodController(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        second_provider,
        transport,
        generation="generation-b",
        generation_started_at_ms=NOW_MS + 1,
        wall_time=lambda: (NOW_MS + 1) / 1000,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    first.reconcile()
    with pytest.raises(controller.ControllerError) as exc_info:
        second.reconcile()

    assert exc_info.value.code == "worker_control_lease_denied"
    assert first_provider.resume_calls == [("pod_123", 1)]
    assert second_provider.resume_calls == []


def test_kill_switch_denial_before_resume_makes_no_provider_mutation():
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(
        lease_errors=[
            controller.ControllerError("worker_control_kill_switch_active")
        ]
    )
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_control_kill_switch_active"
    assert provider.resume_calls == []
    assert provider.stop_calls == []
    assert transport.probes == []
    assert transport.published == []


def test_lease_conflict_after_resume_abandons_without_stopping_new_owner_pod():
    denial = controller.ControllerError("worker_control_lease_denied")
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(
        health=True,
        lease_errors=[None, denial, denial],
    )
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_control_lease_denied"
    assert provider.resume_calls == [("pod_123", 1)]
    assert provider.stop_calls == []
    assert transport.probes == []
    assert transport.published == []


def test_kill_switch_after_resume_stops_owned_pod_without_health_or_publish():
    blocked = controller.ControllerError("worker_control_kill_switch_active")
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(
        health=True,
        lease_errors=[None, blocked, blocked],
    )
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_control_kill_switch_active"
    assert provider.resume_calls == [("pod_123", 1)]
    assert provider.stop_calls == ["pod_123"]
    assert transport.probes == []
    assert transport.published == []


def test_control_lease_is_renewed_while_waiting_for_health():
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(health=[False, True])
    clock = AdvancingClock()
    active = controller.RunPodController(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
            ready_timeout_seconds=10.0,
            probe_interval_seconds=1.0,
        ),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    active.reconcile()

    assert transport.probes == [
        "https://worker-tunnel.example",
        "https://worker-tunnel.example",
    ]
    # Before resume, after resume, before each probe, and immediately before publish.
    assert len(transport.leases) == 5


def test_ambiguous_resume_uses_fixed_deadline_across_transitional_inventory():
    clock = AdvancingClock()
    provider = FakeProvider(
        [_pod(status="EXITED")],
        resume_error=controller.ControllerError("pod_resume_outcome_ambiguous"),
    )
    active = controller.RunPodController(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
            ready_timeout_seconds=10.0,
        ),
        provider,
        FakeTransport(),
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(controller.ControllerError) as first:
        active.reconcile()
    provider.rows[0]["desiredStatus"] = "CREATED"
    clock.value = 9.0
    with pytest.raises(controller.ControllerError) as second:
        active.reconcile()
    clock.value = 10.0
    with pytest.raises(controller.ControllerError) as third:
        active.reconcile()

    assert first.value.code == "pod_resume_outcome_ambiguous"
    assert second.value.code == "pod_bootstrap_reconciliation_pending"
    assert third.value.code == "pod_bootstrap_deadline_exceeded_stop_requested"
    assert provider.resume_calls == [("pod_123", 1)]
    assert provider.stop_calls == ["pod_123"]


def test_failed_bootstrap_stops_once_and_running_inventory_cannot_publish():
    clock = AdvancingClock()
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(health=False)
    active = controller.RunPodController(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
            ready_timeout_seconds=2.0,
            probe_interval_seconds=1.0,
        ),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(controller.ControllerError) as first:
        active.reconcile()
    probes_after_stop = list(transport.probes)
    provider.rows[0]["desiredStatus"] = "RUNNING"
    with pytest.raises(controller.ControllerError) as second:
        active.reconcile()

    assert first.value.code == "worker_not_ready_stop_requested"
    assert second.value.code == "pod_stop_reconciliation_pending"
    assert provider.resume_calls == [("pod_123", 1)]
    assert provider.stop_calls == ["pod_123"]
    assert transport.probes == probes_after_stop
    assert transport.published == []


def test_ambiguous_stop_latches_and_never_retries_or_publishes():
    provider = FakeProvider(
        [_pod(status="EXITED")],
        stop_error=controller.ControllerError("pod_stop_outcome_ambiguous"),
    )
    transport = FakeTransport(health=False)
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )

    with pytest.raises(controller.ControllerError) as first:
        active.reconcile()
    provider.rows[0]["desiredStatus"] = "RUNNING"
    probes_after_stop = list(transport.probes)
    with pytest.raises(controller.ControllerError) as second:
        active.reconcile()

    assert first.value.code == "pod_stop_outcome_ambiguous"
    assert second.value.code == "pod_stop_reconciliation_pending"
    assert provider.stop_calls == ["pod_123"]
    assert transport.probes == probes_after_stop
    assert transport.published == []


def test_owned_ready_pod_uses_short_unhealthy_grace_then_stops():
    clock = AdvancingClock()
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(health=True)
    active = controller.RunPodController(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
            ready_timeout_seconds=100.0,
            unhealthy_grace_seconds=3.0,
            probe_interval_seconds=1.0,
        ),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    active.reconcile()
    provider.rows[0]["desiredStatus"] = "RUNNING"
    transport.health = False
    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_not_ready_stop_requested"
    assert clock.value == 3.0
    assert provider.stop_calls == ["pod_123"]
    assert len(transport.published) == 1
    assert len(transport.cleared) == 1


def test_owned_ready_pod_keeps_target_while_liveness_reports_active_job():
    """A strict 503 during async inference must not stop a healthy Pod."""
    clock = AdvancingClock()
    provider = FakeProvider([_pod()])
    transport = FakeTransport(
        health=[True, False, False, True],
        liveness=[(True, True), (True, True)],
    )
    active = controller.RunPodController(
        _config(
            unhealthy_grace_seconds=1.0,
            probe_interval_seconds=1.0,
        ),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        preowned_pod_id="pod_123",
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    active.reconcile()
    result = active.reconcile()

    assert result.ready is True
    assert provider.stop_calls == []
    assert len(transport.liveness_probes) == 2
    assert len(transport.published) == 2


def test_owned_ready_pod_does_not_extend_grace_for_idle_liveness():
    clock = AdvancingClock()
    provider = FakeProvider([_pod()])
    transport = FakeTransport(
        health=[True, False, False, False], liveness=(True, False)
    )
    active = controller.RunPodController(
        _config(
            unhealthy_grace_seconds=2.0,
            probe_interval_seconds=1.0,
        ),
        provider,
        transport,
        generation="generation-a",
        generation_started_at_ms=NOW_MS,
        preowned_pod_id="pod_123",
        wall_time=lambda: NOW_MS / 1000,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    active.reconcile()
    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_not_ready_stop_requested"
    assert provider.stop_calls == ["pod_123"]
    assert clock.value == 2.0


def test_stale_published_owner_never_stops_pod_after_new_generation_takeover():
    provider = FakeProvider([_pod(status="EXITED")])
    transport = FakeTransport(health=True)
    active = _active_controller(
        _config(
            allow_resume=True,
            max_hourly_cost_usd=Decimal("0.50"),
        ),
        provider,
        transport,
    )
    active.reconcile()
    provider.rows[0]["desiredStatus"] = "RUNNING"
    transport.lease_errors = [
        controller.ControllerError("worker_control_lease_denied")
    ]
    probes_before_takeover = list(transport.probes)

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_control_lease_denied"
    assert provider.stop_calls == []
    assert transport.probes == probes_before_takeover
    assert len(transport.published) == 1
    assert len(transport.cleared) == 1


def test_probe_interval_must_renew_well_before_control_lease_expiry():
    with pytest.raises(controller.ControllerError) as exc_info:
        _active_controller(
            _config(
                target_ttl_ms=30_000,
                heartbeat_seconds=10.0,
                probe_interval_seconds=16.0,
            ),
            FakeProvider([_pod()]),
            FakeTransport(),
        )

    assert exc_info.value.code == "probe_interval_exceeds_lease_renewal_budget"


def test_heartbeat_interval_cannot_exceed_half_the_target_ttl():
    _active_controller(
        _config(target_ttl_ms=90_000, heartbeat_seconds=45.0),
        FakeProvider([_pod()]),
        FakeTransport(),
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        _active_controller(
            _config(target_ttl_ms=90_000, heartbeat_seconds=45.001),
            FakeProvider([_pod()]),
            FakeTransport(),
        )

    assert exc_info.value.code == "heartbeat_interval_invalid"


def test_initial_unready_worker_does_not_publish_or_clear_an_unowned_target():
    provider = FakeProvider([_pod()])
    transport = FakeTransport(health=False)
    active = _active_controller(_config(), provider, transport)

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "worker_not_ready"
    assert provider.resume_calls == []
    assert transport.published == []
    assert transport.cleared == []


def test_unready_worker_clears_only_the_generation_previously_published():
    provider = FakeProvider([_pod()])
    transport = FakeTransport(health=[True, False])
    active = _active_controller(_config(), provider, transport)

    active.reconcile()
    with pytest.raises(controller.ControllerError, match="worker_not_ready"):
        active.reconcile()

    published = transport.published[0][2]
    cleared = transport.cleared[0][2]
    assert published["generation"] == "generation-a"
    assert cleared["generation"] == "generation-a"
    assert cleared["issuedAt"] > published["issuedAt"]


def test_duplicate_inventory_clears_a_target_owned_by_this_generation():
    provider = FakeProvider([_pod()])
    transport = FakeTransport()
    active = _active_controller(_config(), provider, transport)
    active.reconcile()
    provider.rows = [
        _pod(pod_id="pod_123", name="first"),
        _pod(pod_id="pod_123", name="duplicate"),
    ]

    with pytest.raises(controller.ControllerError) as exc_info:
        active.reconcile()

    assert exc_info.value.code == "pod_selector_ambiguous"
    assert len(transport.cleared) == 1
    assert transport.cleared[0][2]["generation"] == "generation-a"


def test_publish_payload_has_generation_ttl_transport_and_request_budget():
    provider = FakeProvider([_pod()])
    transport = FakeTransport()
    active = _active_controller(_config(), provider, transport)

    active.reconcile()

    endpoint, token, payload = transport.published[0]
    assert endpoint == "https://gateway.example/api/admin/worker-target"
    assert token == "admin-token-value"
    assert payload == {
        "schema_version": 1,
        "provider": "runpod",
        "generation": "generation-a",
        "generationStartedAt": NOW_MS,
        "issuedAt": NOW_MS + 2,
        "validUntil": NOW_MS + 90_002,
        "baseUrl": "https://worker-tunnel.example",
        "transport": "https_tunnel",
        "maxRequestMs": 900_000,
        "podId": "pod_123",
    }


def test_gateway_origin_is_normalized_to_the_admin_target_path():
    provider = FakeProvider([_pod()])
    transport = FakeTransport()
    active = _active_controller(
        _config(gateway_endpoint="https://gateway.example/"), provider, transport
    )

    active.reconcile()

    assert (
        transport.published[0][0]
        == "https://gateway.example/api/admin/worker-target"
    )


def test_default_target_is_the_exact_pod_proxy_with_hard_90_second_budget():
    target = controller.derive_worker_target(
        _config(worker_url=None, transport=None, max_request_ms=None),
        controller.SelectedPod(
            pod_id="pod_123",
            desired_status="RUNNING",
            gpu_count=1,
            hourly_cost_usd=Decimal("0.44"),
        ),
    )

    assert target == controller.WorkerTarget(
        base_url="https://pod_123-8000.proxy.runpod.net",
        transport="runpod_proxy",
        max_request_ms=90_000,
    )


def test_proxy_target_rejects_wrong_pod_and_oversized_request_budget():
    pod = controller.SelectedPod(
        pod_id="pod_123",
        desired_status="RUNNING",
        gpu_count=1,
        hourly_cost_usd=Decimal("0.44"),
    )
    with pytest.raises(controller.ControllerError) as mismatch:
        controller.derive_worker_target(
            _config(
                worker_url="https://pod_other-8000.proxy.runpod.net",
                transport="runpod_proxy",
                max_request_ms=90_000,
            ),
            pod,
        )
    with pytest.raises(controller.ControllerError) as oversized:
        controller.derive_worker_target(
            _config(
                worker_url=None,
                transport="runpod_proxy",
                max_request_ms=90_001,
            ),
            pod,
        )

    assert mismatch.value.code == "worker_proxy_pod_mismatch"
    assert oversized.value.code == "worker_proxy_request_limit_invalid"


def test_sdk_is_lazy_loaded_and_adapter_exposes_list_resume_and_stop(monkeypatch):
    calls = []

    class FakeSdk:
        api_key = None

        @staticmethod
        def get_pods(*, api_key):
            calls.append(("list", api_key))
            return [_pod()]

        @staticmethod
        def resume_pod(pod_id, gpu_count):
            calls.append(("resume", pod_id, gpu_count))

        @staticmethod
        def stop_pod(pod_id):
            calls.append(("stop", pod_id))

        @staticmethod
        def create_pod(*_args, **_kwargs):
            raise AssertionError("controller must never call create_pod")

    monkeypatch.setattr(controller.importlib, "import_module", lambda name: FakeSdk)
    provider = controller.SdkRunPodProvider.load("rpa_" + "S" * 48)

    assert provider.list_pods()[0]["id"] == "pod_123"
    provider.resume_pod("pod_123", 1)
    provider.stop_pod("pod_123")
    assert calls == [
        ("list", "rpa_" + "S" * 48),
        ("resume", "pod_123", 1),
        ("stop", "pod_123"),
    ]


@pytest.mark.parametrize(
    "unsafe_environment",
    [
        {"HTTP_PROXY": "http://proxy.example"},
        {"https_proxy": "http://proxy.example"},
        {"ALL_PROXY": "socks5://proxy.example"},
        {"REQUESTS_CA_BUNDLE": "C:/intercept-ca.pem"},
        {"CURL_CA_BUNDLE": "C:/intercept-ca.pem"},
        {"SSL_CERT_FILE": "C:/intercept-ca.pem"},
        {"SSL_CERT_DIR": "C:/intercept-certs"},
    ],
)
def test_sdk_rejects_inherited_proxy_and_ca_environment(unsafe_environment):
    with pytest.raises(controller.ControllerError) as exc_info:
        controller.SdkRunPodProvider(
            SafeSdkStub,
            "rpa_" + "S" * 48,
            environ=unsafe_environment,
        )

    assert exc_info.value.code == "runpod_sdk_network_environment_unsafe"


def test_sdk_pins_official_api_base_and_rejects_override():
    clean_environment = {}
    controller.SdkRunPodProvider(
        SafeSdkStub,
        "rpa_" + "S" * 48,
        environ=clean_environment,
    )
    assert clean_environment["RUNPOD_API_BASE_URL"] == "https://api.runpod.io"

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.SdkRunPodProvider(
            SafeSdkStub,
            "rpa_" + "S" * 48,
            environ={"RUNPOD_API_BASE_URL": "https://proxy.example"},
        )

    assert exc_info.value.code == "runpod_sdk_api_base_invalid"


def test_http_control_lease_posts_to_lease_route_with_bounded_contract(monkeypatch):
    captured = {}
    real_client = httpx.Client

    def handler(request):
        captured["url"] = str(request.url)
        captured["token"] = request.headers["X-Admin-Token"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"granted": True, "outcome": "accepted"})

    mock = httpx.MockTransport(handler)
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )
    payload = {
        "schema_version": 1,
        "provider": "runpod",
        "generation": "generation-a",
        "generationStartedAt": NOW_MS,
        "issuedAt": NOW_MS,
        "validUntil": NOW_MS + 90_000,
        "podId": "pod_123",
    }

    controller.HttpxControllerTransport().acquire_control(
        "https://gateway.example/api/admin/worker-target",
        "secret-admin-token",
        payload,
    )

    assert captured == {
        "url": "https://gateway.example/api/admin/worker-target/lease",
        "token": "secret-admin-token",
        "payload": payload,
    }


@pytest.mark.parametrize(
    ("status", "body", "reason"),
    [
        (409, {"granted": False, "outcome": "conflict"}, "worker_control_lease_denied"),
        (
            423,
            {"granted": False, "outcome": "kill_switch_active"},
            "worker_control_kill_switch_active",
        ),
        (
            423,
            {"granted": False, "outcome": "control_blocked"},
            "worker_control_blocked",
        ),
        (503, {"error": "unavailable"}, "gateway_control_lease_failed"),
        (200, {"granted": False}, "gateway_control_lease_response_invalid"),
    ],
)
def test_http_control_lease_distinguishes_gateway_outcomes(
    monkeypatch, status, body, reason
):
    real_client = httpx.Client
    mock = httpx.MockTransport(
        lambda _request: httpx.Response(status, json=body)
    )
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.HttpxControllerTransport().acquire_control(
            "https://gateway.example/api/admin/worker-target",
            "secret-admin-token",
            {"schema_version": 1},
        )

    assert exc_info.value.code == reason


def test_http_control_lease_rejects_oversized_streamed_response(monkeypatch):
    real_client = httpx.Client
    mock = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"x" * (controller.MAX_CONTROL_RESPONSE_BYTES + 1),
        )
    )
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.HttpxControllerTransport().acquire_control(
            "https://gateway.example/api/admin/worker-target",
            "secret-admin-token",
            {"schema_version": 1},
        )

    assert exc_info.value.code == "gateway_control_lease_response_invalid"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"status": "ok", "models_loaded": True, "device": "cuda"}, True),
        ({"status": "ok", "models_loaded": False, "device": "cuda"}, False),
        ({"status": "ok", "models_loaded": True, "device": "cpu"}, False),
    ],
)
def test_http_health_probe_requires_real_cuda_readiness(monkeypatch, body, expected):
    real_client = httpx.Client
    mock = httpx.MockTransport(lambda _request: httpx.Response(200, json=body))

    def client_factory(**kwargs):
        return real_client(transport=mock, **kwargs)

    monkeypatch.setattr(controller.httpx, "Client", client_factory)

    assert (
        controller.HttpxControllerTransport().probe_health(
            "https://worker.example"
        )
        is expected
    )


def test_http_health_probe_rejects_an_oversized_body(monkeypatch):
    real_client = httpx.Client
    mock = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"x" * (controller.MAX_HEALTH_RESPONSE_BYTES + 1),
        )
    )

    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )

    assert (
        controller.HttpxControllerTransport().probe_health(
            "https://worker.example"
        )
        is False
    )


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            200,
            {
                "schema_version": 1,
                "status": "ok",
                "models_loaded": True,
                "device": "cuda",
                "busy": True,
            },
            controller.LivenessResult(True, True),
        ),
        (
            200,
            {
                "schema_version": 1,
                "status": "ok",
                "models_loaded": True,
                "device": "cuda",
                "busy": False,
            },
            controller.LivenessResult(True, False),
        ),
        (
            200,
            {"status": "not_ready", "models_loaded": False, "device": "cuda"},
            controller.LivenessResult(False, False),
        ),
        (404, {}, controller.LivenessResult(False, False)),
    ],
)
def test_http_liveness_probe_is_strict_and_reports_busy(
    monkeypatch, status, body, expected
):
    real_client = httpx.Client
    mock = httpx.MockTransport(lambda _request: httpx.Response(status, json=body))
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )

    assert (
        controller.HttpxControllerTransport().probe_liveness(
            "https://worker.example"
        )
        == expected
    )


def test_gateway_publish_uses_admin_header_and_exact_json_without_url_logging(
    monkeypatch,
):
    captured = {}
    real_client = httpx.Client

    def handler(request):
        captured["method"] = request.method
        captured["token"] = request.headers["X-Admin-Token"]
        captured["payload"] = request.content
        return httpx.Response(204)

    mock = httpx.MockTransport(handler)
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )
    transport = controller.HttpxControllerTransport()

    transport.publish_target(
        "https://gateway.example/api/admin/worker-target",
        "secret-admin-token",
        {"schema_version": 1, "baseUrl": "https://worker-secret.example"},
    )

    assert captured["method"] == "POST"
    assert captured["token"] == "secret-admin-token"
    assert json.loads(captured["payload"]) == {
        "schema_version": 1,
        "baseUrl": "https://worker-secret.example",
    }


def test_gateway_publish_rejects_oversized_streamed_response(monkeypatch):
    real_client = httpx.Client
    mock = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"x" * (controller.MAX_CONTROL_RESPONSE_BYTES + 1),
        )
    )
    monkeypatch.setattr(
        controller.httpx,
        "Client",
        lambda **kwargs: real_client(transport=mock, **kwargs),
    )

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.HttpxControllerTransport().publish_target(
            "https://gateway.example/api/admin/worker-target",
            "secret-admin-token",
            {"schema_version": 1},
        )

    assert exc_info.value.code == "gateway_target_response_invalid"


def test_main_error_report_never_contains_api_key_or_worker_url(capsys):
    secret = "rpa_" + "Z" * 48
    target = "https://private-worker.example"
    result = controller.main(
        ["--once"],
        environ={
            "RUNPOD_API_KEY": secret,
            "RUNPOD_POD_ID": "pod_123",
            "RUNPOD_WORKER_URL": target,
        },
        provider=FakeProvider([_pod()]),
        transport=FakeTransport(),
    )

    output = capsys.readouterr().err
    assert result == 2
    assert "gateway_configuration_invalid" in output
    assert secret not in output
    assert target not in output


def test_once_mode_refuses_resume_before_loading_sdk_or_touching_provider(capsys):
    provider = FakeProvider([_pod(status="EXITED")])
    result = controller.main(
        ["--once", "--allow-resume", "--max-hourly-cost-usd", "0.50"],
        environ={},
        provider=provider,
        transport=FakeTransport(),
    )

    output = capsys.readouterr().err
    assert result == 2
    assert "once_resume_not_allowed" in output
    assert provider.list_calls == 0
    assert provider.resume_calls == []
    assert provider.stop_calls == []


def test_environment_defaults_and_caps_unhealthy_grace():
    arguments = controller._parser().parse_args([])
    environment = {
        "RUNPOD_API_KEY": "rpa_" + "S" * 48,
        "RUNPOD_POD_ID": "pod_123",
        "RUNPOD_WORKER_URL": "https://worker.example",
        "GATEWAY_ADMIN_URL": "https://gateway.example",
        "GATEWAY_ADMIN_TOKEN": "admin-token-value",
    }

    config = controller.config_from_environment(arguments, environment)

    assert config.ready_timeout_seconds == 1_800.0
    assert config.unhealthy_grace_seconds == 120.0

    with pytest.raises(controller.ControllerError) as exc_info:
        controller.config_from_environment(
            arguments,
            {**environment, "RUNPOD_UNHEALTHY_GRACE_SECONDS": "601"},
        )

    assert exc_info.value.code == "unhealthy_grace_invalid"
