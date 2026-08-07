from fastapi.testclient import TestClient

import src.main as main_module


client = TestClient(main_module.app)


def test_liveness_is_lightweight_and_separate_from_readiness(monkeypatch):
    """Controller liveness stays green while strict readiness may be busy."""
    monkeypatch.setattr(
        main_module,
        "_model_health_snapshot",
        lambda: {"core_ready": True, "device": "cuda"},
    )
    main_module.WORKER_STATE["quarantined"] = False

    response = client.get("/api/worker/liveness")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": 1,
        "status": "ok",
        "models_loaded": True,
        "device": "cuda",
        "busy": False,
    }


def test_liveness_fails_closed_when_worker_is_quarantined(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_model_health_snapshot",
        lambda: {"core_ready": True, "device": "cuda"},
    )
    main_module.WORKER_STATE["quarantined"] = True
    try:
        response = client.get("/api/worker/liveness")
    finally:
        main_module.WORKER_STATE["quarantined"] = False

    assert response.status_code == 200
    assert response.json()["status"] == "not_ready"
    assert response.json()["models_loaded"] is True
