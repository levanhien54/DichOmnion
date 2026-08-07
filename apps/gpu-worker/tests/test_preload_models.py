import json
import sys
from contextlib import nullcontext
from types import ModuleType

import pytest

from scripts import preload_models


def _configure_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models" / "hf"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "models" / "torch"))
    monkeypatch.setenv("OMNIVOICE_MIN_MODEL_CACHE_FREE_BYTES", "0")
    monkeypatch.delenv("OMNIVOICE_MODEL_READY_FILE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


def _hydrate_fake_hf_cache():
    revisions = {}
    for index, spec in enumerate(preload_models.runtime_model_specs(), start=1):
        revision = spec["requested_revision"] or f"{index:040x}"
        repo = (
            preload_models._hf_hub_cache()
            / f"models--{spec['model_id'].replace('/', '--')}"
        )
        if spec["requested_revision"] is None:
            (repo / "refs").mkdir(parents=True, exist_ok=True)
            (repo / "refs" / "main").write_text(revision + "\n", encoding="ascii")
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.asset").write_bytes(b"cached")
        if spec["component"] == "pyannote_diarization":
            (snapshot / "config.yaml").write_text(
                "pipeline:\n"
                "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
                "  params:\n"
                "    embedding: pyannote/wespeaker-voxceleb-resnet34-LM\n"
                "    segmentation: pyannote/segmentation-3.0\n",
                encoding="utf-8",
            )
        revisions[spec["component"]] = revision
    return revisions


def test_v5_cache_fingerprint_pins_exact_model_assets(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("QWEN_MODEL_ID", "Qwen/test-model")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "small")
    monkeypatch.setattr(
        preload_models, "_package_version", lambda name: f"test-{name}"
    )

    assert preload_models.CACHE_SCHEMA_VERSION == 5
    assert (
        preload_models.MOSS_TTS_MODEL_ID
        == "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    )
    assert (
        preload_models.MOSS_TTS_MODEL_REVISION
        == "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
    )
    assert preload_models.MOSS_TTS_CODEC_ID == "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
    assert (
        preload_models.MOSS_TTS_CODEC_REVISION
        == "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    )
    assert preload_models.DEMUCS_REPO_ID == "adefossez/HTDemucs"
    assert preload_models.marker_path() == (
        tmp_path / "models" / ".omnivoice-model-cache-v5.json"
    ).resolve()
    assert preload_models.cache_fingerprint() == {
        "schema": 5,
        "qwen_model_id": "Qwen/test-model",
        "whisper_model_size": "small",
        "pyannote_pipeline_id": "pyannote/speaker-diarization-3.1",
        "pyannote_segmentation_id": "pyannote/segmentation-3.0",
        "pyannote_embedding_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "demucs_model_id": "hf://htdemucs",
        "audioseal_model_id": "audioseal_wm_16bits",
        "audioseal_detector_id": "audioseal_detector_16bits",
        "audioseal_repo_id": "facebook/audioseal",
        "audioseal_revision": "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba",
        "audioseal_nbits": 16,
        "audioseal_generator_filename": "generator_base.pth",
        "audioseal_generator_sha256": (
            "7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748"
        ),
        "audioseal_detector_filename": "detector_base.pth",
        "audioseal_detector_sha256": (
            "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9"
        ),
        "moss_tts_model_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "moss_tts_model_revision": "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        "moss_tts_codec_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        "moss_tts_codec_revision": "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
        "packages": {
            "audioseal": "test-audioseal",
            "demucs": "test-demucs",
            "faster-whisper": "test-faster-whisper",
            "huggingface-hub": "test-huggingface-hub",
            "torch": "test-torch",
            "torchaudio": "test-torchaudio",
            "transformers": "test-transformers",
            "whisperx": "test-whisperx",
        },
    }
    assert {
        spec["component"]: spec["model_id"]
        for spec in preload_models.runtime_model_specs()
    } == {
        "audioseal": "facebook/audioseal",
        "demucs": "adefossez/HTDemucs",
        "moss_codec": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        "moss_tts": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "pyannote_diarization": "pyannote/speaker-diarization-3.1",
        "pyannote_embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "pyannote_segmentation": "pyannote/segmentation-3.0",
        "qwen_translation": "Qwen/test-model",
        "whisper_asr": "Systran/faster-whisper-small",
    }


def test_moss_snapshot_downloads_use_pinned_revisions(monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    preload_models._download_moss_tts("hf_test_secret")
    preload_models._download_moss_codec("hf_test_secret")

    assert calls == [
        {
            "repo_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
            "revision": "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
            "token": "hf_test_secret",
        },
        {
            "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
            "revision": "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
            "token": "hf_test_secret",
        },
    ]
    assert [name for name, _stage in preload_models._default_stages()][-2:] == [
        "moss-tts",
        "moss-codec",
    ]


def test_diarization_preload_uses_explicit_audited_pipeline(monkeypatch):
    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_diarize = ModuleType("whisperx.diarize")
    fake_diarize.DiarizationPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)
    monkeypatch.setattr(
        "src.diarization_service._pyannote_checkpoint_context",
        nullcontext,
    )

    preload_models._download_diarization("hf_test_secret")

    assert calls == [
        {
            "model_name": "pyannote/speaker-diarization-3.1",
            "use_auth_token": "hf_test_secret",
            "device": "cpu",
        }
    ]


def test_audioseal_preload_hydrates_generator_and_detector(monkeypatch):
    calls = []

    class FakeAudioSeal:
        @staticmethod
        def load_generator(model_id, *, nbits):
            calls.append(("generator", model_id, nbits))
            return object()

        @staticmethod
        def load_detector(model_id, *, nbits):
            calls.append(("detector", model_id, nbits))
            return object()

    fake_audioseal = ModuleType("audioseal")
    fake_audioseal.AudioSeal = FakeAudioSeal
    monkeypatch.setitem(sys.modules, "audioseal", fake_audioseal)
    monkeypatch.setattr(
        preload_models,
        "resolve_audioseal_checkpoint_paths",
        lambda **_kwargs: ("generator_base.pth", "detector_base.pth"),
    )

    preload_models._download_audioseal("")

    assert calls == [
        ("generator", "generator_base.pth", 16),
        ("detector", "detector_base.pth", 16),
    ]


def test_legacy_v4_marker_does_not_satisfy_v5_contract(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    old_marker = tmp_path / "models" / ".omnivoice-model-cache-v4.json"
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text(json.dumps({"schema": 4}), encoding="utf-8")

    assert preload_models.marker_path().name == ".omnivoice-model-cache-v5.json"
    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_preload_is_idempotent_after_atomic_marker(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    calls = []
    stages = [("one", lambda token: calls.append(("one", token)))]

    assert preload_models.prepare_model_cache(stages=stages) is True
    monkeypatch.setenv("OMNIVOICE_MIN_MODEL_CACHE_FREE_BYTES", str(10**30))
    assert preload_models.prepare_model_cache(stages=stages) is False
    assert calls == [("one", "hf_test_secret")]

    marker = json.loads(preload_models.marker_path().read_text(encoding="utf-8"))
    assert marker == {
        "schema": 5,
        "fingerprint": preload_models.cache_fingerprint(),
        "models": preload_models.resolve_runtime_model_manifest(),
        "snapshots": preload_models._snapshot_inventories(
            preload_models.resolve_runtime_model_manifest()
        ),
    }
    assert "hf_test_secret" not in json.dumps(marker)
    preload_models.validate_ready_marker()


def test_manifest_accepts_single_snapshot_when_hf_loader_omits_main_ref(
    monkeypatch, tmp_path
):
    _configure_cache(monkeypatch, tmp_path)
    revision = "a" * 40
    spec = next(
        row
        for row in preload_models.runtime_model_specs()
        if row["component"] == "qwen_translation"
    )
    repo = preload_models._repo_cache_path(spec["model_id"])
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.asset").write_bytes(b"cached")

    assert preload_models._resolved_snapshot_revision(
        spec["model_id"], None
    ) == revision


def test_runtime_manifest_resolves_commits_and_exports_sanitized_m7_input(
    monkeypatch, tmp_path
):
    _configure_cache(monkeypatch, tmp_path)
    revisions = _hydrate_fake_hf_cache()

    assert preload_models.prepare_model_cache(stages=[]) is True
    destination = tmp_path / "reports" / "models.json"
    preload_models.export_m7_model_manifest(destination)

    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported == preload_models.resolve_runtime_model_manifest()
    assert {row["component"]: row["revision"] for row in exported} == revisions
    serialized = json.dumps(exported)
    assert "hf_test_secret" not in serialized
    assert str(tmp_path) not in serialized
    assert all(set(row) == {"component", "model_id", "revision"} for row in exported)


def test_ready_marker_fails_closed_when_resolved_snapshot_is_removed(
    monkeypatch, tmp_path
):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    preload_models.prepare_model_cache(stages=[])
    qwen = next(
        spec
        for spec in preload_models.runtime_model_specs()
        if spec["component"] == "qwen_translation"
    )
    revision = next(
        row["revision"]
        for row in preload_models.resolve_runtime_model_manifest()
        if row["component"] == "qwen_translation"
    )
    asset = (
        preload_models._hf_hub_cache()
        / f"models--{qwen['model_id'].replace('/', '--')}"
        / "snapshots"
        / revision
        / "model.asset"
    )
    asset.unlink()

    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_ready_marker_rejects_manifest_tampering(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    preload_models.prepare_model_cache(stages=[])
    marker_path = preload_models.marker_path()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["models"][0]["revision"] = "f" * 40
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_ready_marker_rejects_changed_pyannote_dependency_graph(
    monkeypatch, tmp_path
):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    preload_models.prepare_model_cache(stages=[])
    pipeline = next(
        row
        for row in preload_models.resolve_runtime_model_manifest()
        if row["component"] == "pyannote_diarization"
    )
    config = (
        preload_models._repo_cache_path(pipeline["model_id"])
        / "snapshots"
        / pipeline["revision"]
        / "config.yaml"
    )
    config.write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        "    embedding: unreviewed/example\n"
        "    segmentation: pyannote/segmentation-3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_ready_marker_rejects_deleted_weight_when_config_remains(
    monkeypatch, tmp_path
):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    preload_models.prepare_model_cache(stages=[])
    pipeline = next(
        row
        for row in preload_models.resolve_runtime_model_manifest()
        if row["component"] == "pyannote_diarization"
    )
    weight = (
        preload_models._repo_cache_path(pipeline["model_id"])
        / "snapshots"
        / pipeline["revision"]
        / "model.asset"
    )
    weight.unlink()

    assert (
        weight.parent / "config.yaml"
    ).is_file(), "test must leave pipeline metadata behind"
    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_export_never_overwrites_ready_marker(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    _hydrate_fake_hf_cache()
    preload_models.prepare_model_cache(stages=[])

    with pytest.raises(RuntimeError, match="must differ"):
        preload_models.export_m7_model_manifest(preload_models.marker_path())


def test_preload_failure_never_leaves_ready_marker(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)

    def fail(_token):
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        preload_models.prepare_model_cache(stages=[("broken", fail)])

    assert not preload_models.marker_path().exists()
    with pytest.raises(RuntimeError, match="marker is missing"):
        preload_models.validate_ready_marker()


def test_preload_requires_token_before_any_stage(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    monkeypatch.delenv("HF_TOKEN")

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        preload_models.prepare_model_cache(stages=[])


def test_preload_rejects_non_hub_model_id_before_network(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("QWEN_MODEL_ID", "https://example.test/model?token=secret")
    calls = []

    with pytest.raises(RuntimeError, match="model id is invalid"):
        preload_models.prepare_model_cache(
            stages=[("network", lambda token: calls.append(token))]
        )

    assert calls == []


def test_preload_rechecks_free_space_before_writing_marker(monkeypatch, tmp_path):
    _configure_cache(monkeypatch, tmp_path)
    checks = []

    def validate_storage():
        checks.append("checked")
        if len(checks) == 2:
            raise RuntimeError("Model cache volume does not have enough free space.")

    monkeypatch.setattr(preload_models, "validate_cache_storage", validate_storage)

    with pytest.raises(RuntimeError, match="enough free space"):
        preload_models.prepare_model_cache(stages=[])

    assert checks == ["checked", "checked"]
    assert not preload_models.marker_path().exists()
