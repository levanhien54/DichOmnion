"""Hydrate the complete OmniVoice model cache on a persistent volume.

This command is intentionally online. The serving process remains offline-only and starts only
after every download stage has completed and an atomic cache marker has been written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from src.audio_engine import (
    AUDIOSEAL_DETECTOR_FILENAME,
    AUDIOSEAL_DETECTOR_SHA256,
    AUDIOSEAL_DETECTOR_ID,
    AUDIOSEAL_GENERATOR_FILENAME,
    AUDIOSEAL_GENERATOR_SHA256,
    AUDIOSEAL_MODEL_ID,
    AUDIOSEAL_NBITS,
    AUDIOSEAL_REPO_ID,
    AUDIOSEAL_REVISION,
    DEMUCS_MODEL_ID,
    resolve_audioseal_checkpoint_paths,
)
from src.diarization_service import PYANNOTE_PIPELINE_ID


CACHE_SCHEMA_VERSION = 5
DEFAULT_MIN_FREE_BYTES = 12_000_000_000
MAX_MARKER_BYTES = 64 * 1024
PYANNOTE_SEGMENTATION_ID = "pyannote/segmentation-3.0"
PYANNOTE_EMBEDDING_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEMUCS_REPO_ID = "adefossez/HTDemucs"
MOSS_TTS_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MOSS_TTS_MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
MOSS_TTS_CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MOSS_TTS_CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
_HF_COMMIT_RE = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)
_HF_MODEL_ID_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,95}/)?"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def cache_fingerprint() -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "qwen_model_id": os.environ.get(
            "QWEN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"
        ),
        "whisper_model_size": os.environ.get("WHISPER_MODEL_SIZE", "base"),
        "pyannote_pipeline_id": PYANNOTE_PIPELINE_ID,
        "pyannote_segmentation_id": PYANNOTE_SEGMENTATION_ID,
        "pyannote_embedding_id": PYANNOTE_EMBEDDING_ID,
        "demucs_model_id": DEMUCS_MODEL_ID,
        "audioseal_model_id": AUDIOSEAL_MODEL_ID,
        "audioseal_detector_id": AUDIOSEAL_DETECTOR_ID,
        "audioseal_repo_id": AUDIOSEAL_REPO_ID,
        "audioseal_revision": AUDIOSEAL_REVISION,
        "audioseal_nbits": AUDIOSEAL_NBITS,
        "audioseal_generator_filename": AUDIOSEAL_GENERATOR_FILENAME,
        "audioseal_generator_sha256": AUDIOSEAL_GENERATOR_SHA256,
        "audioseal_detector_filename": AUDIOSEAL_DETECTOR_FILENAME,
        "audioseal_detector_sha256": AUDIOSEAL_DETECTOR_SHA256,
        "moss_tts_model_id": MOSS_TTS_MODEL_ID,
        "moss_tts_model_revision": MOSS_TTS_MODEL_REVISION,
        "moss_tts_codec_id": MOSS_TTS_CODEC_ID,
        "moss_tts_codec_revision": MOSS_TTS_CODEC_REVISION,
        "packages": {
            name: _package_version(name)
            for name in (
                "audioseal",
                "demucs",
                "faster-whisper",
                "huggingface-hub",
                "torch",
                "torchaudio",
                "transformers",
                "whisperx",
            )
        },
    }


def _cache_directories() -> tuple[Path, Path, Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/models/hf")).resolve()
    torch_home = Path(os.environ.get("TORCH_HOME", "/models/torch")).resolve()
    try:
        common_root = Path(os.path.commonpath((hf_home, torch_home)))
    except ValueError:
        common_root = hf_home.parent
    if common_root == Path(common_root.anchor):
        common_root = hf_home.parent
    return hf_home, torch_home, common_root


def marker_path() -> Path:
    configured = os.environ.get("OMNIVOICE_MODEL_READY_FILE", "").strip()
    if configured:
        return Path(configured).resolve()
    _, _, cache_root = _cache_directories()
    return cache_root / ".omnivoice-model-cache-v5.json"


def _hf_hub_cache() -> Path:
    configured = os.environ.get("HF_HUB_CACHE", "").strip()
    if configured:
        return Path(configured).resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _validated_hf_model_id(value: Any) -> str:
    if not isinstance(value, str) or not _HF_MODEL_ID_RE.fullmatch(value):
        raise RuntimeError("A runtime model id is invalid.")
    if ".." in value or value.endswith(".git"):
        raise RuntimeError("A runtime model id is invalid.")
    return value


def _whisper_repo_id() -> str:
    configured = os.environ.get("WHISPER_MODEL_SIZE", "base").strip()
    try:
        from faster_whisper.utils import _MODELS
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Faster-Whisper model registry is unavailable.") from exc
    return _validated_hf_model_id(_MODELS.get(configured, configured))


def runtime_model_specs() -> tuple[dict[str, str | None], ...]:
    """Return the audited Hugging Face repositories required by this worker image."""

    fingerprint = cache_fingerprint()
    specs = (
        {
            "component": "audioseal",
            "model_id": fingerprint["audioseal_repo_id"],
            "requested_revision": fingerprint["audioseal_revision"],
        },
        {
            "component": "demucs",
            "model_id": DEMUCS_REPO_ID,
            "requested_revision": None,
        },
        {
            "component": "moss_codec",
            "model_id": fingerprint["moss_tts_codec_id"],
            "requested_revision": fingerprint["moss_tts_codec_revision"],
        },
        {
            "component": "moss_tts",
            "model_id": fingerprint["moss_tts_model_id"],
            "requested_revision": fingerprint["moss_tts_model_revision"],
        },
        {
            "component": "pyannote_diarization",
            "model_id": fingerprint["pyannote_pipeline_id"],
            "requested_revision": None,
        },
        {
            "component": "pyannote_embedding",
            "model_id": fingerprint["pyannote_embedding_id"],
            "requested_revision": None,
        },
        {
            "component": "pyannote_segmentation",
            "model_id": fingerprint["pyannote_segmentation_id"],
            "requested_revision": None,
        },
        {
            "component": "qwen_translation",
            "model_id": fingerprint["qwen_model_id"],
            "requested_revision": None,
        },
        {
            "component": "whisper_asr",
            "model_id": _whisper_repo_id(),
            "requested_revision": None,
        },
    )
    for spec in specs:
        _validated_hf_model_id(spec["model_id"])
        revision = spec["requested_revision"]
        if revision is not None and not _HF_COMMIT_RE.fullmatch(revision):
            raise RuntimeError("A pinned runtime model revision is invalid.")
    return tuple(sorted(specs, key=lambda row: str(row["component"])))


def _snapshot_inventory(path: Path) -> dict[str, int | str]:
    """Describe a cached snapshot without hashing multi-gigabyte model contents."""

    try:
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError("A resolved model snapshot is missing or incomplete.")
        entries: list[tuple[str, str, int, str]] = []
        total_bytes = 0
        for candidate in path.rglob("*"):
            if candidate.is_dir():
                if candidate.is_symlink():
                    raise RuntimeError(
                        "A resolved model snapshot contains an unsafe directory link."
                    )
                continue
            if not candidate.is_file():
                raise RuntimeError("A resolved model snapshot is incomplete.")
            size = candidate.stat().st_size
            if size < 0:
                raise RuntimeError("A resolved model snapshot is incomplete.")
            link_identity = ""
            kind = "file"
            if candidate.is_symlink():
                kind = "symlink"
                target = Path(os.readlink(candidate))
                link_identity = target.as_posix() if not target.is_absolute() else target.name
            relative = candidate.relative_to(path).as_posix()
            entries.append((relative, kind, size, link_identity))
            total_bytes += size
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError("A resolved model snapshot is missing or incomplete.") from exc

    if not entries:
        raise RuntimeError("A resolved model snapshot is missing or incomplete.")
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(
            json.dumps(entry, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def _repo_cache_path(model_id: str) -> Path:
    model_id = _validated_hf_model_id(model_id)
    return _hf_hub_cache() / f"models--{model_id.replace('/', '--')}"


def _resolved_snapshot_revision(model_id: str, requested_revision: str | None) -> str:
    model_id = _validated_hf_model_id(model_id)
    repo_cache = _repo_cache_path(model_id)
    if requested_revision is None:
        ref = repo_cache / "refs" / "main"
        try:
            if ref.stat().st_size > 80:
                raise RuntimeError("A model cache reference is invalid.")
            revision = ref.read_text(encoding="ascii").strip().lower()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "A model cache is missing its resolved Hugging Face revision."
            ) from exc
    else:
        revision = requested_revision.lower()
    if not _HF_COMMIT_RE.fullmatch(revision):
        raise RuntimeError("A model cache revision is invalid.")
    _snapshot_inventory(repo_cache / "snapshots" / revision)
    return revision


def _validate_pyannote_dependency_graph(manifest: list[dict[str, str]]) -> None:
    revisions = {row["component"]: row["revision"] for row in manifest}
    pipeline_revision = revisions.get("pyannote_diarization")
    if pipeline_revision is None:
        raise RuntimeError("The pyannote runtime manifest is incomplete.")
    config = (
        _repo_cache_path(PYANNOTE_PIPELINE_ID)
        / "snapshots"
        / pipeline_revision
        / "config.yaml"
    )
    try:
        if not config.is_file() or not 1 <= config.stat().st_size <= 64 * 1024:
            raise RuntimeError("The pyannote pipeline configuration is invalid.")
        import yaml

        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except RuntimeError:
        raise
    except (ImportError, OSError, UnicodeError, ValueError, TypeError):
        raise RuntimeError("The pyannote pipeline configuration is invalid.") from None
    pipeline = document.get("pipeline") if isinstance(document, dict) else None
    params = pipeline.get("params") if isinstance(pipeline, dict) else None
    if (
        not isinstance(pipeline, dict)
        or pipeline.get("name") != "pyannote.audio.pipelines.SpeakerDiarization"
        or not isinstance(params, dict)
    ):
        raise RuntimeError("The pyannote pipeline dependency graph is invalid.")
    if (
        params.get("segmentation") != PYANNOTE_SEGMENTATION_ID
        or params.get("embedding") != PYANNOTE_EMBEDDING_ID
    ):
        raise RuntimeError("The pyannote pipeline dependency graph is invalid.")


def resolve_runtime_model_manifest() -> list[dict[str, str]]:
    """Resolve exact cached commits without exposing cache paths or credentials."""

    manifest = [
        {
            "component": str(spec["component"]),
            "model_id": str(spec["model_id"]),
            "revision": _resolved_snapshot_revision(
                str(spec["model_id"]), spec["requested_revision"]
            ),
        }
        for spec in runtime_model_specs()
    ]
    _validate_pyannote_dependency_graph(manifest)
    return manifest


def _snapshot_inventories(
    manifest: list[dict[str, str]],
) -> list[dict[str, int | str]]:
    inventories: list[dict[str, int | str]] = []
    for model in manifest:
        inventory = _snapshot_inventory(
            _repo_cache_path(model["model_id"])
            / "snapshots"
            / model["revision"]
        )
        inventories.append({"component": model["component"], **inventory})
    return inventories


def _ensure_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fd, probe = tempfile.mkstemp(prefix=".omnivoice-write-", dir=path)
    os.close(fd)
    Path(probe).unlink(missing_ok=True)


def validate_cache_storage() -> None:
    hf_home, torch_home, cache_root = _cache_directories()
    audioseal_home = Path(
        os.environ.get("AUDIOSEAL_CACHE_DIR", str(cache_root))
    ).resolve()
    _ensure_writable(hf_home)
    _ensure_writable(torch_home)
    _ensure_writable(audioseal_home)
    _ensure_writable(cache_root)

    try:
        minimum = int(
            os.environ.get(
                "OMNIVOICE_MIN_MODEL_CACHE_FREE_BYTES", str(DEFAULT_MIN_FREE_BYTES)
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "OMNIVOICE_MIN_MODEL_CACHE_FREE_BYTES must be an integer."
        ) from exc
    if minimum < 0:
        raise RuntimeError(
            "OMNIVOICE_MIN_MODEL_CACHE_FREE_BYTES must not be negative."
        )
    if shutil.disk_usage(cache_root).free < minimum:
        raise RuntimeError("Model cache volume does not have enough free space.")


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        if not 1 <= path.stat().st_size <= MAX_MARKER_BYTES:
            raise RuntimeError("Model cache marker has an invalid size.")
        data = json.loads(path.read_text(encoding="utf-8"))
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise RuntimeError("Model cache marker is unreadable.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Model cache marker has an invalid schema.")
    return data


def _validated_marker_document(
    path: Path, fingerprint: dict[str, Any]
) -> dict[str, Any]:
    data = _read_marker(path)
    expected_manifest = resolve_runtime_model_manifest()
    expected_snapshots = _snapshot_inventories(expected_manifest)
    if (
        set(data) != {"schema", "fingerprint", "models", "snapshots"}
        or data["schema"] != CACHE_SCHEMA_VERSION
        or data["fingerprint"] != fingerprint
        or data["models"] != expected_manifest
        or data["snapshots"] != expected_snapshots
    ):
        raise RuntimeError("Model cache marker does not match this image.")
    return data


def _marker_matches(path: Path, fingerprint: dict[str, Any]) -> bool:
    try:
        _validated_marker_document(path, fingerprint)
    except RuntimeError:
        return False
    return True


def validate_ready_marker() -> None:
    """Fail closed when the mounted cache was not completed by this image version."""

    ready_file = marker_path()
    try:
        _validated_marker_document(ready_file, cache_fingerprint())
    except RuntimeError:
        raise RuntimeError(
            "Model cache marker is missing or does not match this image."
        ) from None


def _write_json_atomic(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_marker(
    path: Path,
    fingerprint: dict[str, Any],
    models: list[dict[str, str]],
    snapshots: list[dict[str, int | str]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "models": models,
            "snapshots": snapshots,
        },
    )


def export_m7_model_manifest(path: Path) -> None:
    """Export the validated cache manifest in the strict M7 benchmark input format."""

    destination = path.resolve()
    ready_file = marker_path()
    if destination == ready_file:
        raise RuntimeError("Model manifest destination must differ from the cache marker.")
    document = _validated_marker_document(ready_file, cache_fingerprint())
    _write_json_atomic(destination, document["models"])


def _download_qwen(token: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"),
        token=token,
    )


def _download_whisper(_token: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        os.environ.get("WHISPER_MODEL_SIZE", "base"),
        device="cpu",
        compute_type="int8",
    )
    del model


def _download_diarization(token: str) -> None:
    from src.diarization_service import _pyannote_checkpoint_context
    from whisperx.diarize import DiarizationPipeline

    with _pyannote_checkpoint_context():
        pipeline = DiarizationPipeline(
            model_name=PYANNOTE_PIPELINE_ID,
            use_auth_token=token,
            device="cpu",
        )
    del pipeline


def _download_demucs(_token: str) -> None:
    from demucs.pretrained import get_model

    model = get_model(DEMUCS_MODEL_ID)
    del model


def _download_audioseal(_token: str) -> None:
    from audioseal import AudioSeal

    generator_path, detector_path = resolve_audioseal_checkpoint_paths(
        local_files_only=False
    )
    generator = AudioSeal.load_generator(generator_path, nbits=AUDIOSEAL_NBITS)
    detector = AudioSeal.load_detector(detector_path, nbits=AUDIOSEAL_NBITS)
    del generator, detector


def _download_moss_tts(token: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MOSS_TTS_MODEL_ID,
        revision=MOSS_TTS_MODEL_REVISION,
        token=token,
    )


def _download_moss_codec(token: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MOSS_TTS_CODEC_ID,
        revision=MOSS_TTS_CODEC_REVISION,
        token=token,
    )


def _default_stages() -> Iterable[tuple[str, Callable[[str], None]]]:
    return (
        ("qwen", _download_qwen),
        ("whisper", _download_whisper),
        ("diarization", _download_diarization),
        ("demucs", _download_demucs),
        ("audioseal", _download_audioseal),
        ("moss-tts", _download_moss_tts),
        ("moss-codec", _download_moss_codec),
    )


def prepare_model_cache(
    *,
    force: bool = False,
    stages: Iterable[tuple[str, Callable[[str], None]]] | None = None,
) -> bool:
    """Prepare all caches and return True when downloads ran, False on marker hit."""

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required to preload gated models.")

    fingerprint = cache_fingerprint()
    # Validate all configured model identifiers and immutable source pins before network I/O.
    runtime_model_specs()
    ready_file = marker_path()
    if not force and _marker_matches(ready_file, fingerprint):
        print("OmniVoice model cache already matches this image; skipping preload.")
        return False

    validate_cache_storage()
    ready_file.unlink(missing_ok=True)
    selected_stages = _default_stages() if stages is None else stages
    for stage_name, stage in selected_stages:
        print(f"Preloading model stage: {stage_name}")
        stage(token)
        gc.collect()

    # Preserve the configured working-space reserve after the model footprint is real.
    # A volume that only had enough room to begin downloading must not receive a ready marker.
    validate_cache_storage()
    manifest = resolve_runtime_model_manifest()
    snapshots = _snapshot_inventories(manifest)
    _write_marker(ready_file, fingerprint, manifest, snapshots)
    print("OmniVoice model cache is complete.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Revalidate every download stage."
    )
    parser.add_argument(
        "--export-m7-manifest",
        type=Path,
        help="Export the validated runtime model revisions for the M7 benchmark.",
    )
    args = parser.parse_args()

    if args.export_m7_manifest is not None:
        if args.force:
            parser.error("--force cannot be combined with --export-m7-manifest")
        export_m7_model_manifest(args.export_m7_manifest)
        print("OmniVoice M7 model manifest exported.")
        return

    # This command is the only online phase. Serving resets both flags to 1 before boot.
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    prepare_model_cache(force=args.force)


if __name__ == "__main__":
    main()
