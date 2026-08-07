"""Sanitized M7 source provenance and immutable-image syntax preflight.

Schema v2 separates Docker image inputs from RunPod provisioning controls. The local
preflight validates only image-reference syntax; it never proves registry pullability,
platform architecture, or authorization to create paid infrastructure.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any


REPORT_SCHEMA = "omnivoice.m7-worker-build-provenance.v2"
BUILD_DIGEST_FORMAT = "omnivoice.gpu-worker-build-inputs.v2"
CONTROL_DIGEST_FORMAT = "omnivoice.runpod-controls.v1"
BUILD_FIXED_FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock")
BUILD_COPY_FILES = ("pyproject.toml", "uv.lock")
BUILD_DIRECTORIES = ("src", "scripts")
BUILD_SCOPE = (*BUILD_FIXED_FILES, "src/**", "scripts/**")
CONTROL_DIRECTORY = "runpod"
CONTROL_SCOPE = ("runpod/**",)
CONTROL_EXCLUSIONS = ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")
MAX_FILE_SET_ENTRIES = 50_000
MAX_SCANNED_ENTRIES = 100_000
MAX_INPUT_FILE_BYTES = 256 * 1024**2
MAX_INPUT_BYTES = 2 * 1024**3
MAX_DOCKERIGNORE_BYTES = 256 * 1024
MAX_DOCKERIGNORE_RULES = 1_024
MAX_DOCKERIGNORE_LINE = 4_096
MAX_GIT_OUTPUT_BYTES = 4 * 1024**2
GIT_TIMEOUT_SECONDS = 10
HASH_CHUNK_BYTES = 1024**2

_HEX_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MANIFEST_SHA_RE = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{1,238}[a-z0-9]")
_REPOSITORY_COMPONENT_RE = re.compile(
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
)
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class ProvenanceError(RuntimeError):
    """A release-preflight failure represented by a sanitized reason code."""

    def __init__(self, code: str):
        if not _REASON_CODE_RE.fullmatch(code):
            raise ValueError("invalid provenance reason code")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FileSetDigest:
    sha256: str
    file_count: int
    total_bytes: int
    relative_files: tuple[str, ...]


@dataclass(frozen=True)
class ImageIdentity:
    repository_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class GitState:
    commit: str
    build_inputs_clean: bool
    runpod_controls_clean: bool

    @property
    def release_ready(self) -> bool:
        return self.build_inputs_clean and self.runpod_controls_clean


@dataclass(frozen=True)
class _DockerIgnoreRule:
    segments: tuple[str, ...]
    basename_only: bool
    directory_only: bool
    negated: bool

    def matches(self, relative_name: str, *, is_directory: bool) -> bool:
        parts = PurePosixPath(relative_name).parts
        if not parts or (self.directory_only and not is_directory):
            return False
        if self.basename_only:
            return fnmatch.fnmatchcase(parts[-1], self.segments[0])
        return _match_path_segments(self.segments, parts)


@dataclass(frozen=True)
class _DockerIgnore:
    rules: tuple[_DockerIgnoreRule, ...]

    def _directly_ignored(self, relative_name: str, *, is_directory: bool) -> bool:
        ignored = False
        for rule in self.rules:
            if rule.matches(relative_name, is_directory=is_directory):
                ignored = not rule.negated
        return ignored

    def is_ignored(self, relative_name: str, *, is_directory: bool) -> bool:
        parts = PurePosixPath(relative_name).parts
        for end in range(1, len(parts)):
            parent = "/".join(parts[:end])
            if self._directly_ignored(parent, is_directory=True):
                return True
        return self._directly_ignored(relative_name, is_directory=is_directory)


def _match_path_segments(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    @lru_cache(maxsize=None)
    def match(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern):
            return path_index == len(path)
        segment = pattern[pattern_index]
        if segment == "**":
            return match(pattern_index + 1, path_index) or (
                path_index < len(path) and match(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path)
            and fnmatch.fnmatchcase(path[path_index], segment)
            and match(pattern_index + 1, path_index + 1)
        )

    return match(0, 0)


def _is_reparse_stat(metadata: os.stat_result | Any) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _REPARSE_POINT_FLAG)


def _normalized_worker_root(worker_root: Path) -> Path:
    try:
        unresolved = Path(os.path.abspath(worker_root))
        metadata = unresolved.lstat()
        if _is_reparse_stat(metadata):
            raise ProvenanceError("worker_root_reparse_unsupported")
        root = unresolved.resolve(strict=True)
    except ProvenanceError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ProvenanceError("worker_root_unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode) or not root.is_dir():
        raise ProvenanceError("worker_root_invalid")
    return root


def _regular_file_stat(path: Path, *, too_large_code: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise ProvenanceError("input_file_unavailable") from None
    if _is_reparse_stat(metadata):
        raise ProvenanceError("input_reparse_point_unsupported")
    if not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError("input_special_file_unsupported")
    if metadata.st_size < 0 or metadata.st_size > MAX_INPUT_FILE_BYTES:
        raise ProvenanceError(too_large_code)
    return metadata


def _same_file_version(first: os.stat_result, second: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(first, name, None) == getattr(second, name, None) for name in fields)


def _open_validated_file(path: Path, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise ProvenanceError("input_file_unreadable") from None
    if _is_reparse_stat(opened) or not _same_file_version(expected, opened):
        os.close(descriptor)
        raise ProvenanceError("input_changed_during_hash")
    return descriptor


def _stream_regular_file(
    path: Path,
    metadata: os.stat_result,
    consume: Callable[[bytes], None],
) -> None:
    descriptor = _open_validated_file(path, metadata)
    remaining = metadata.st_size
    try:
        with os.fdopen(descriptor, "rb") as handle:
            while remaining:
                chunk = handle.read(min(HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ProvenanceError("input_changed_during_hash")
                consume(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise ProvenanceError("input_changed_during_hash")
    except ProvenanceError:
        raise
    except OSError:
        raise ProvenanceError("input_file_unreadable") from None
    try:
        after = path.lstat()
    except OSError:
        raise ProvenanceError("input_changed_during_hash") from None
    if _is_reparse_stat(after) or not _same_file_version(metadata, after):
        raise ProvenanceError("input_changed_during_hash")


def _read_bounded_regular_file(path: Path, *, maximum: int) -> bytes:
    metadata = _regular_file_stat(path, too_large_code="dockerignore_too_large")
    if metadata.st_size > maximum:
        raise ProvenanceError("dockerignore_too_large")
    chunks: list[bytes] = []
    _stream_regular_file(path, metadata, chunks.append)
    return b"".join(chunks)


def _normalize_dockerignore_pattern(value: str) -> tuple[str, ...]:
    if "\\" in value or "\x00" in value:
        raise ProvenanceError("dockerignore_pattern_unsupported")
    value = value.strip("/")
    if not value or value == ".":
        return ()
    normalized: list[str] = []
    for component in value.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not normalized:
                raise ProvenanceError("dockerignore_pattern_unsupported")
            normalized.pop()
        else:
            normalized.append(component)
    return tuple(normalized)


def _parse_dockerignore(payload: bytes) -> _DockerIgnore:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError:
        raise ProvenanceError("dockerignore_encoding_invalid") from None
    rules: list[_DockerIgnoreRule] = []
    for raw_line in text.splitlines():
        if len(raw_line) > MAX_DOCKERIGNORE_LINE:
            raise ProvenanceError("dockerignore_pattern_too_long")
        if raw_line.startswith("#") or not raw_line.strip():
            continue
        line = raw_line.strip()
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        directory_only = line.endswith("/")
        segments = _normalize_dockerignore_pattern(line)
        if not segments:
            if negated:
                raise ProvenanceError("dockerignore_pattern_unsupported")
            continue
        rules.append(
            _DockerIgnoreRule(
                segments=segments,
                basename_only=len(segments) == 1,
                directory_only=directory_only,
                negated=negated,
            )
        )
        if len(rules) > MAX_DOCKERIGNORE_RULES:
            raise ProvenanceError("dockerignore_rule_count_invalid")
    return _DockerIgnore(tuple(rules))


def _load_dockerignore(root: Path) -> tuple[_DockerIgnore, bytes]:
    payload = _read_bounded_regular_file(
        root / ".dockerignore", maximum=MAX_DOCKERIGNORE_BYTES
    )
    return _parse_dockerignore(payload), payload


def _is_generated_control_path(relative_name: str) -> bool:
    path = PurePosixPath(relative_name)
    return "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}


def _scan_directory(
    root: Path,
    directory_name: str,
    *,
    include: Callable[[str], bool],
    prune: Callable[[str], bool] | None = None,
) -> list[tuple[str, Path]]:
    directory = root / directory_name
    try:
        directory_metadata = directory.lstat()
    except OSError:
        raise ProvenanceError("input_directory_unavailable") from None
    if _is_reparse_stat(directory_metadata):
        raise ProvenanceError("input_reparse_point_unsupported")
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise ProvenanceError("input_directory_invalid")

    discovered: list[tuple[str, Path]] = []
    pending = [directory]
    scanned_entries = 0
    while pending:
        current = pending.pop()
        try:
            current_metadata = current.lstat()
            if _is_reparse_stat(current_metadata):
                raise ProvenanceError("input_reparse_point_unsupported")
            if not stat.S_ISDIR(current_metadata.st_mode):
                raise ProvenanceError("input_directory_invalid")
            with os.scandir(current) as iterator:
                for entry in iterator:
                    scanned_entries += 1
                    if scanned_entries > MAX_SCANNED_ENTRIES:
                        raise ProvenanceError("input_tree_entry_count_invalid")
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        relative_name = Path(entry.path).relative_to(root).as_posix()
                    except (OSError, RuntimeError, UnicodeError, ValueError):
                        raise ProvenanceError("input_tree_scan_failed") from None
                    if _is_reparse_stat(metadata):
                        raise ProvenanceError("input_reparse_point_unsupported")
                    if stat.S_ISDIR(metadata.st_mode):
                        if prune is None or not prune(relative_name):
                            pending.append(Path(entry.path))
                    elif stat.S_ISREG(metadata.st_mode):
                        if include(relative_name):
                            discovered.append((relative_name, Path(entry.path)))
                    else:
                        raise ProvenanceError("input_special_file_unsupported")
        except ProvenanceError:
            raise
        except OSError:
            raise ProvenanceError("input_tree_scan_failed") from None
    return discovered


def _sort_and_validate_files(files: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    if not files or len(files) > MAX_FILE_SET_ENTRIES:
        raise ProvenanceError("input_file_count_invalid")
    names = [name for name, _path in files]
    if len(names) != len(set(names)):
        raise ProvenanceError("input_path_collision")
    try:
        return sorted(files, key=lambda item: item[0].encode("utf-8"))
    except UnicodeError:
        raise ProvenanceError("input_path_encoding_invalid") from None


def _discover_build_files(root: Path, dockerignore: _DockerIgnore) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for relative_name in BUILD_FIXED_FILES:
        path = root / relative_name
        _regular_file_stat(path, too_large_code="input_file_too_large")
        if relative_name in BUILD_COPY_FILES and dockerignore.is_ignored(
            relative_name, is_directory=False
        ):
            raise ProvenanceError("docker_required_input_ignored")
        discovered.append((relative_name, path))

    for directory_name in BUILD_DIRECTORIES:
        discovered.extend(
            _scan_directory(
                root,
                directory_name,
                include=lambda name: not dockerignore.is_ignored(
                    name, is_directory=False
                ),
            )
        )
    return _sort_and_validate_files(discovered)


def _discover_control_files(root: Path) -> list[tuple[str, Path]]:
    files = _scan_directory(
        root,
        CONTROL_DIRECTORY,
        include=lambda name: not _is_generated_control_path(name),
        prune=_is_generated_control_path,
    )
    return _sort_and_validate_files(files)


def _digest_files(
    files: list[tuple[str, Path]], *, digest_format: str
) -> FileSetDigest:
    digest = hashlib.sha256()
    digest.update(digest_format.encode("ascii") + b"\0")
    total_bytes = 0
    versions: list[tuple[Path, os.stat_result]] = []
    for relative_name, path in files:
        metadata = _regular_file_stat(path, too_large_code="input_file_too_large")
        if total_bytes + metadata.st_size > MAX_INPUT_BYTES:
            raise ProvenanceError("input_file_set_too_large")
        relative_bytes = relative_name.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(metadata.st_size.to_bytes(8, "big"))
        _stream_regular_file(path, metadata, digest.update)
        total_bytes += metadata.st_size
        versions.append((path, metadata))
    for path, expected in versions:
        try:
            current = path.lstat()
        except OSError:
            raise ProvenanceError("input_changed_during_hash") from None
        if _is_reparse_stat(current) or not _same_file_version(expected, current):
            raise ProvenanceError("input_changed_during_hash")
    return FileSetDigest(
        sha256=digest.hexdigest(),
        file_count=len(files),
        total_bytes=total_bytes,
        relative_files=tuple(name for name, _path in files),
    )


def digest_build_context(worker_root: Path) -> FileSetDigest:
    """Hash exactly the declared Docker inputs that survive ``.dockerignore``."""

    root = _normalized_worker_root(worker_root)
    dockerignore, dockerignore_payload = _load_dockerignore(root)
    files = _discover_build_files(root, dockerignore)
    result = _digest_files(files, digest_format=BUILD_DIGEST_FORMAT)
    after_rules, after_payload = _load_dockerignore(root)
    after_files = _discover_build_files(root, after_rules)
    if dockerignore_payload != after_payload or result.relative_files != tuple(
        name for name, _path in after_files
    ):
        raise ProvenanceError("input_changed_during_hash")
    return result


def digest_runpod_controls(worker_root: Path) -> FileSetDigest:
    """Hash RunPod provisioning scripts and artifacts separately from the image."""

    root = _normalized_worker_root(worker_root)
    files = _discover_control_files(root)
    result = _digest_files(files, digest_format=CONTROL_DIGEST_FORMAT)
    after_files = _discover_control_files(root)
    if result.relative_files != tuple(name for name, _path in after_files):
        raise ProvenanceError("input_changed_during_hash")
    return result


def validate_image_reference(image: str) -> ImageIdentity:
    """Validate syntax only; no registry or platform claim is made here."""

    if not isinstance(image, str) or image.count("@sha256:") != 1:
        raise ProvenanceError("image_manifest_invalid")
    repository, manifest_sha256 = image.rsplit("@sha256:", 1)
    if (
        not _REPOSITORY_RE.fullmatch(repository)
        or not _MANIFEST_SHA_RE.fullmatch(manifest_sha256)
        or "/" not in repository
        or "://" in repository
        or "//" in repository
        or ".." in repository
        or "\\" in repository
        or "@" in repository
        or ":" in repository.rsplit("/", 1)[-1]
    ):
        raise ProvenanceError("image_manifest_invalid")

    components = repository.split("/")
    first_component = components[0]
    if ":" in first_component:
        host, port = first_component.rsplit(":", 1)
        if (
            not _REPOSITORY_COMPONENT_RE.fullmatch(host)
            or not port.isascii()
            or not port.isdecimal()
            or not 1 <= int(port) <= 65_535
        ):
            raise ProvenanceError("image_manifest_invalid")
    elif not _REPOSITORY_COMPONENT_RE.fullmatch(first_component):
        raise ProvenanceError("image_manifest_invalid")
    if any(
        not _REPOSITORY_COMPONENT_RE.fullmatch(component)
        for component in components[1:]
    ):
        raise ProvenanceError("image_manifest_invalid")
    return ImageIdentity(
        repository_sha256=hashlib.sha256(repository.encode("ascii")).hexdigest(),
        manifest_sha256=manifest_sha256,
    )


def _run_git(
    git_root: Path,
    arguments: Sequence[str],
    *,
    runner: Callable[..., Any],
) -> bytes:
    try:
        completed = runner(
            ["git", *arguments],
            cwd=str(git_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        raise ProvenanceError("git_command_unavailable") from None
    stdout = completed.stdout
    if (
        completed.returncode != 0
        or not isinstance(stdout, bytes)
        or len(stdout) > MAX_GIT_OUTPUT_BYTES
    ):
        raise ProvenanceError("git_command_failed")
    return stdout


def _decode_scoped_git_path(raw_name: bytes, prefix: str) -> str:
    try:
        name = raw_name.decode("utf-8")
    except UnicodeError:
        raise ProvenanceError("git_path_encoding_invalid") from None
    if prefix:
        if not name.startswith(prefix):
            raise ProvenanceError("git_path_scope_invalid")
        name = name[len(prefix) :]
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ProvenanceError("git_path_scope_invalid")
    return name


def _porcelain_paths(payload: bytes, prefix: str) -> list[str]:
    records = payload.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ProvenanceError("git_status_invalid")
        status_code = record[:2]
        paths.append(_decode_scoped_git_path(record[3:], prefix))
        if b"R" in status_code or b"C" in status_code:
            if index >= len(records) or not records[index]:
                raise ProvenanceError("git_status_invalid")
            paths.append(_decode_scoped_git_path(records[index], prefix))
            index += 1
    return paths


def _classify_release_path(
    relative_name: str, dockerignore: _DockerIgnore
) -> tuple[bool, bool]:
    path = PurePosixPath(relative_name)
    build_input = relative_name in BUILD_FIXED_FILES
    if path.parts and path.parts[0] in BUILD_DIRECTORIES:
        build_input = not dockerignore.is_ignored(relative_name, is_directory=False)
    control = bool(
        path.parts
        and path.parts[0] == CONTROL_DIRECTORY
        and not _is_generated_control_path(relative_name)
    )
    return build_input, control


def inspect_git_state(
    worker_root: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> GitState:
    """Read HEAD and classify dirtiness for image inputs and RunPod controls."""

    root = _normalized_worker_root(worker_root)
    dockerignore, _payload = _load_dockerignore(root)
    root_output = _run_git(root, ["rev-parse", "--show-toplevel"], runner=runner)
    try:
        git_root = Path(root_output.strip().decode("utf-8")).resolve(strict=True)
        worker_relative = root.relative_to(git_root).as_posix()
    except (OSError, RuntimeError, UnicodeError, ValueError):
        raise ProvenanceError("git_root_invalid") from None

    commit_output = _run_git(
        git_root, ["rev-parse", "--verify", "HEAD"], runner=runner
    )
    try:
        commit = commit_output.strip().decode("ascii")
    except UnicodeError:
        raise ProvenanceError("git_commit_invalid") from None
    if not _HEX_SHA_RE.fullmatch(commit):
        raise ProvenanceError("git_commit_invalid")

    prefix = f"{worker_relative}/" if worker_relative else ""
    pathspecs = [
        prefix + name
        for name in (*BUILD_FIXED_FILES, *BUILD_DIRECTORIES, CONTROL_DIRECTORY)
    ]
    status = _run_git(
        git_root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ],
        runner=runner,
    )
    ignored = _run_git(
        git_root,
        [
            "ls-files",
            "--full-name",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        runner=runner,
    )

    build_clean = True
    controls_clean = True
    names = _porcelain_paths(status, prefix)
    names.extend(
        _decode_scoped_git_path(raw, prefix)
        for raw in ignored.split(b"\0")
        if raw
    )
    for name in names:
        build_input, control = _classify_release_path(name, dockerignore)
        build_clean = build_clean and not build_input
        controls_clean = controls_clean and not control
    return GitState(
        commit=commit,
        build_inputs_clean=build_clean,
        runpod_controls_clean=controls_clean,
    )


def build_release_provenance(
    worker_root: Path,
    image: str,
    *,
    git_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Build a bounded report with no deployment authorization side effect."""

    image_identity = validate_image_reference(image)
    build_inputs = digest_build_context(worker_root)
    controls = digest_runpod_controls(worker_root)
    git_state = inspect_git_state(worker_root, runner=git_runner)
    return {
        "schema": REPORT_SCHEMA,
        "build_inputs": {
            "digest_format": BUILD_DIGEST_FORMAT,
            "sha256": build_inputs.sha256,
            "file_count": build_inputs.file_count,
            "total_bytes": build_inputs.total_bytes,
            "scope": list(BUILD_SCOPE),
            "dockerignore_applied": True,
        },
        "runpod_controls": {
            "digest_format": CONTROL_DIGEST_FORMAT,
            "sha256": controls.sha256,
            "file_count": controls.file_count,
            "total_bytes": controls.total_bytes,
            "scope": list(CONTROL_SCOPE),
            "excluded": list(CONTROL_EXCLUSIONS),
        },
        "source": {
            "git_commit": git_state.commit,
            "build_inputs_clean": git_state.build_inputs_clean,
            "runpod_controls_clean": git_state.runpod_controls_clean,
        },
        "image": {
            "digest_pinned": True,
            "repository_sha256": image_identity.repository_sha256,
            "manifest_digest": f"sha256:{image_identity.manifest_sha256}",
            "verification_evidence": "syntax_only_unverified",
            "pullability_verified": False,
            "linux_amd64_verified": False,
        },
        "release_ready": git_state.release_ready,
        "provisioning_authorized": False,
    }


def _canonical_json(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _is_release_area(relative_name: str) -> bool:
    path = PurePosixPath(relative_name)
    return relative_name in BUILD_FIXED_FILES or bool(
        path.parts
        and path.parts[0] in (*BUILD_DIRECTORIES, CONTROL_DIRECTORY)
    )


def _preflight_report_destination(worker_root: Path, destination: Path) -> Path:
    root = _normalized_worker_root(worker_root)
    try:
        resolved = destination.resolve(strict=False)
    except (OSError, RuntimeError, UnicodeError):
        raise ProvenanceError("report_destination_invalid") from None
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved
    if _is_release_area(relative):
        raise ProvenanceError("report_destination_inside_release_inputs")
    return resolved


def _write_report(worker_root: Path, destination: Path, document: dict[str, Any]) -> None:
    resolved = _preflight_report_destination(worker_root, destination)
    parent = resolved.parent
    if not parent.is_dir() or resolved.is_dir():
        raise ProvenanceError("report_destination_invalid")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".m7-provenance-", suffix=".tmp", dir=parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, resolved)
        temporary_name = None
    except OSError:
        raise ProvenanceError("report_write_failed") from None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ProvenanceError("arguments_invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="OCI image in repository@sha256:<64 lowercase hex> syntax.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination outside image inputs and RunPod controls.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Emit diagnostic provenance without failing for dirty release inputs.",
    )
    return parser


def _safe_error(code: str) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "release_ready": False,
        "provisioning_authorized": False,
        "reason_code": code,
    }
    print(_canonical_json(payload).decode("ascii"), file=sys.stderr)


def main(argv: Sequence[str] | None = None, *, worker_root: Path | None = None) -> int:
    root = worker_root or Path(__file__).resolve().parents[1]
    try:
        args = _parser().parse_args(argv)
        document = build_release_provenance(root, args.image)
        if args.output is not None:
            _write_report(root, args.output, document)
        print(_canonical_json(document).decode("ascii"))
        if not document["release_ready"] and not args.allow_dirty:
            return 2
        return 0
    except ProvenanceError as exc:
        _safe_error(exc.code)
        return 2
    except Exception:
        _safe_error("internal_error")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
