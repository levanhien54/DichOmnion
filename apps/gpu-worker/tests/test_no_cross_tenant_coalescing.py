"""Cách ly xuyên-tenant ở worker: KHÔNG gộp (coalesce) request theo job_id.

Bối cảnh (Đợt 12, F2 — hồi quy do một bản I8 trước): worker từng gộp các dispatch TRÙNG
job_id vào CÙNG một asyncio.Task để né render GPU trùng khi Gateway auto-retry. NHƯNG
job_id chỉ DUY NHẤT theo thiết bị (Gateway tự namespace state bằng job:<deviceId>:<jobId>),
còn JWT worker chỉ mang claim `jobId` (không có deviceId). Gộp theo job_id TRẦN nên phá vỡ
ranh giới tenant: hai thiết bị khác nhau lỡ chọn cùng chuỗi job_id (client dùng JOB-<ms> đoán
được) sẽ va vào nhau, và thiết bị thứ hai NHẬN kết quả (đường dẫn dubbed_audio) của thiết bị
đầu -> rò rỉ audio xuyên tenant.

Bản hiện tại đã BỎ coalescing: mỗi request chạy pipeline ĐỘC LẬP, tuần tự hoá qua semaphore.
Các bài dưới khoá bất biến đó lại — nếu coalescing quay lại, chúng sẽ đỏ. Phần dư (một lần
render trùng hiếm gặp khi retry mạng thoáng qua) là ranh giới G-03 đã ghi nhận & chấp nhận.

Cần request THẬT đồng thời nên dùng httpx.AsyncClient + ASGITransport + gather. Override xác
thực đọc job_id từ body để thoả ràng buộc token/job cho TỪNG payload.
"""
import asyncio

import httpx
from fastapi import Request

import src.main as main_mod
from src.main import app, verify_gateway_jwt


def _echo_jobid_auth():
    """Coi JWT Gateway hợp lệ và jobId KHỚP payload từng request (đọc body — Starlette cache
    lại nên model vẫn parse được sau đó)."""
    async def _auth(request: Request):
        body = await request.json()
        return {"role": "gateway", "jobId": body.get("job_id")}
    return _auth


def _payload(job_id: str, audio_url: str):
    return {
        "job_id": job_id,
        "audio_url": audio_url,
        "target_language": "Vietnamese",
        "translation_style": "Formal",
        "segments": [],
    }


async def test_same_jobid_distinct_tenants_do_not_leak(monkeypatch):
    """Hai request CÙNG job_id nhưng audio KHÁC nhau (hai tenant lỡ chọn cùng JOB-<ms>) phải
    chạy pipeline ĐỘC LẬP và mỗi bên nhận ĐÚNG kết quả của mình — KHÔNG bên nào nhận
    dubbed_audio của bên kia. Release-Event ép hai request cùng in-flight trước khi bên nào
    hoàn tất, để nếu coalescing quay lại thì bài này chắc chắn đỏ."""
    calls = []
    release = asyncio.Event()

    async def _fake_process_job(url, config):
        calls.append(url)
        await release.wait()  # giữ in-flight để cả hai request cùng vào handler
        name = url.rsplit("/", 1)[-1]
        return {"status": "success", "dubbed_audio": f"/tmp/out-for-{name}"}

    monkeypatch.setattr(main_mod.model_manager, "process_job", _fake_process_job)
    app.dependency_overrides[verify_gateway_jwt] = _echo_jobid_auth()

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            tV = asyncio.create_task(
                ac.post("/api/worker/process", json=_payload("JOB-COLLIDE", "http://local/victim.wav"))
            )
            tA = asyncio.create_task(
                ac.post("/api/worker/process", json=_payload("JOB-COLLIDE", "http://local/attacker.wav"))
            )
            await asyncio.sleep(0.1)  # cho cả hai tới điểm chờ
            release.set()
            rV, rA = await asyncio.gather(tV, tA)

        assert rV.status_code == 200, rV.text
        assert rA.status_code == 200, rA.text
        # Mỗi bên nhận ĐÚNG audio của mình — không chảy xuyên tenant.
        assert rV.json()["result"]["dubbed_audio"] == "/tmp/out-for-victim.wav"
        assert rA.json()["result"]["dubbed_audio"] == "/tmp/out-for-attacker.wav"
        # Pipeline chạy đủ HAI lần (không gộp làm một task chung).
        assert sorted(calls) == ["http://local/attacker.wav", "http://local/victim.wav"], calls
    finally:
        app.dependency_overrides.clear()


async def test_distinct_jobs_render_independently(monkeypatch):
    """Hai job_id KHÁC nhau chạy pipeline riêng (không gộp nhầm)."""
    calls = []

    async def _fake_process_job(url, config):
        calls.append(url)
        return {"status": "success", "url_echo": url}

    monkeypatch.setattr(main_mod.model_manager, "process_job", _fake_process_job)
    app.dependency_overrides[verify_gateway_jwt] = _echo_jobid_auth()

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r1, r2 = await asyncio.gather(
                ac.post("/api/worker/process", json=_payload("JOB-X", "http://local/JOB-X.wav")),
                ac.post("/api/worker/process", json=_payload("JOB-Y", "http://local/JOB-Y.wav")),
            )
        assert r1.status_code == 200 and r2.status_code == 200
        assert sorted(calls) == ["http://local/JOB-X.wav", "http://local/JOB-Y.wav"], calls
    finally:
        app.dependency_overrides.clear()


async def test_same_jobid_sequential_reprocess_runs_again(monkeypatch):
    """Gửi lại cùng job_id SAU khi lần đầu đã xong vẫn chạy pipeline lại (không dính task
    cũ, không có registry tồn dư)."""
    calls = []

    async def _fake_process_job(url, config):
        calls.append(url)
        return {"status": "success"}

    monkeypatch.setattr(main_mod.model_manager, "process_job", _fake_process_job)
    app.dependency_overrides[verify_gateway_jwt] = _echo_jobid_auth()

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r1 = await ac.post("/api/worker/process", json=_payload("SOLO-1", "http://local/a.wav"))
            r2 = await ac.post("/api/worker/process", json=_payload("SOLO-1", "http://local/b.wav"))
        assert r1.status_code == 200 and r2.status_code == 200
        assert calls == ["http://local/a.wav", "http://local/b.wav"], calls
    finally:
        app.dependency_overrides.clear()
