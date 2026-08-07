"""Delete MOSS reference data whose authorization or retention has ended.

This is a trusted, offline administration command. It is deliberately dry-run by
default and emits only a bounded, machine-readable audit document. Registry-owned
values such as paths, reference IDs, transcripts, hashes, and tokens never enter the
audit output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence

from src import moss_tts_adapter as moss_policy


_MAX_REGISTRY_BYTES = 256 * 1024
_MAX_PROFILES = 128
_MAX_PRIVATE_TEXT_CHARS = 2_000
_EXPECTED_ROW_KEYS = {
    "profileId",
    "referenceId",
    "audioFile",
    "sha256",
    "transcript",
    "provenance",
    "consent",
    "retention",
}
_CONSENT_SCOPES = frozenset(
    {"voice-cloning", "synthetic-no-natural-person"}
)
_TRIGGER_ORDER = (
    "consent_revoked",
    "consent_expired",
    "retention_due",
)


class ReferenceReaperError(RuntimeError):
    """A fail-closed error with a fixed, non-sensitive machine code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _RegistrySnapshot:
    document: dict[str, Any]
    raw_sha256: str
    mode: int


@dataclass(frozen=True)
class _Candidate:
    row_index: int
    row: dict[str, Any]
    target: Path
    target_key: str
    exists: bool
    triggers: tuple[str, ...]


def _fail(code: str = "invalid_registry") -> None:
    raise ReferenceReaperError(code)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        _fail("unsafe_path")
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


def _read_registry_bytes(registry_path: Path) -> tuple[bytes, int]:
    if not registry_path.is_absolute():
        _fail("unsafe_registry")
    try:
        metadata = registry_path.lstat()
    except OSError:
        _fail("unsafe_registry")
    if _is_link_or_reparse(registry_path) or not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe_registry")
    if metadata.st_size > _MAX_REGISTRY_BYTES:
        _fail()
    try:
        with registry_path.open("rb") as handle:
            raw = handle.read(_MAX_REGISTRY_BYTES + 1)
    except OSError:
        _fail("unsafe_registry")
    if len(raw) > _MAX_REGISTRY_BYTES:
        _fail()
    return raw, stat.S_IMODE(metadata.st_mode)


def _private_text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_PRIVATE_TEXT_CHARS
    ):
        _fail()
    return value


def _timestamp(value: Any) -> datetime | None:
    try:
        return moss_policy._parse_timestamp(value)
    except Exception:
        _fail()


def _parse_registry(registry_path: Path) -> _RegistrySnapshot:
    raw, mode = _read_registry_bytes(registry_path)
    try:
        text = raw.decode("utf-8", errors="strict")
        document = moss_policy._strict_json(text)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        _fail()

    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "profiles",
    }:
        _fail()
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        _fail()
    rows = document["profiles"]
    if not isinstance(rows, list) or len(rows) > _MAX_PROFILES:
        _fail()

    seen_profile_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _EXPECTED_ROW_KEYS:
            _fail()
        profile_id = row.get("profileId")
        if (
            not isinstance(profile_id, str)
            or not moss_policy._PROFILE_ID_RE.fullmatch(profile_id)
            or profile_id in seen_profile_ids
        ):
            _fail()
        seen_profile_ids.add(profile_id)

        _private_text(row.get("referenceId"))
        _private_text(row.get("audioFile"))
        _private_text(row.get("transcript"))
        _private_text(row.get("provenance"))
        sha256 = row.get("sha256")
        if not isinstance(sha256, str) or not moss_policy._SHA256_RE.fullmatch(sha256):
            _fail()

        consent = row.get("consent")
        if not isinstance(consent, dict) or set(consent) != {
            "granted",
            "revoked",
            "scope",
            "expiresAt",
        }:
            _fail()
        if (
            consent.get("granted") is not True
            or type(consent.get("revoked")) is not bool
        ):
            _fail()
        if consent.get("scope") not in _CONSENT_SCOPES:
            _fail()
        _timestamp(consent.get("expiresAt"))

        retention = row.get("retention")
        if not isinstance(retention, dict) or set(retention) != {"deleteAfter"}:
            _fail()
        _timestamp(retention.get("deleteAfter"))

    return _RegistrySnapshot(
        document=document,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        mode=mode,
    )


def _safe_relative_audio_file(value: str) -> tuple[str, ...]:
    if (
        "\x00" in value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        _fail("unsafe_path")
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        _fail("unsafe_path")
    if not value.casefold().endswith(".wav"):
        _fail("unsafe_path")
    return parts


def _reference_root(path: Path) -> Path:
    if not path.is_absolute():
        _fail("unsafe_root")
    try:
        metadata = path.lstat()
        if _is_link_or_reparse(path) or not stat.S_ISDIR(metadata.st_mode):
            _fail("unsafe_root")
        resolved = path.resolve(strict=True)
    except ReferenceReaperError:
        raise
    except (OSError, RuntimeError):
        _fail("unsafe_root")
    return resolved


def _inspect_target(root: Path, audio_file: str) -> tuple[Path, bool]:
    parts = _safe_relative_audio_file(audio_file)
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        try:
            metadata = parent.lstat()
        except OSError:
            _fail("unsafe_path")
        if _is_link_or_reparse(parent) or not stat.S_ISDIR(metadata.st_mode):
            _fail("unsafe_path")

    target = parent / parts[-1]
    try:
        target.relative_to(root)
    except ValueError:
        _fail("unsafe_path")

    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return target, False
    except OSError:
        _fail("unsafe_path")
    if _is_link_or_reparse(target) or not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe_path")
    if metadata.st_nlink != 1:
        _fail("unsafe_path")
    try:
        target.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        _fail("unsafe_path")
    return target, True


def _due_triggers(row: dict[str, Any], now: datetime) -> tuple[str, ...]:
    consent = row["consent"]
    expires_at = _timestamp(consent["expiresAt"])
    delete_after = _timestamp(row["retention"]["deleteAfter"])
    present = {
        "consent_revoked": consent["revoked"] is True,
        "consent_expired": expires_at is not None and expires_at <= now,
        "retention_due": delete_after is not None and delete_after <= now,
    }
    return tuple(trigger for trigger in _TRIGGER_ORDER if present[trigger])


def _current_registry_digest(registry_path: Path) -> str:
    raw, _mode = _read_registry_bytes(registry_path)
    return hashlib.sha256(raw).hexdigest()


def _verify_unchanged(registry_path: Path, expected_digest: str) -> None:
    if _current_registry_digest(registry_path) != expected_digest:
        _fail("registry_changed")


def _delete_target(root: Path, candidate: _Candidate) -> str:
    current_target, exists = _inspect_target(root, candidate.row["audioFile"])
    if current_target != candidate.target:
        _fail("unsafe_path")
    if not exists:
        return "already_absent"
    try:
        current_target.unlink()
    except OSError:
        _fail("delete_failed")
    try:
        current_target.lstat()
    except FileNotFoundError:
        return "deleted"
    except OSError:
        _fail("delete_verification_failed")
    _fail("delete_verification_failed")


def _atomic_write_registry(
    registry_path: Path,
    document: dict[str, Any],
    *,
    mode: int,
    expected_digest: str,
) -> None:
    _verify_unchanged(registry_path, expected_digest)
    encoded = (
        json.dumps(document, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_REGISTRY_BYTES:
        _fail()

    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            dir=registry_path.parent,
        )
        os.chmod(temporary_name, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_unchanged(registry_path, expected_digest)
        os.replace(temporary_name, registry_path)
        temporary_name = ""
    except ReferenceReaperError:
        raise
    except OSError:
        _fail("registry_update_failed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass

    if _is_link_or_reparse(registry_path):
        _fail("registry_update_failed")
    try:
        actual = registry_path.read_bytes()
    except OSError:
        _fail("registry_update_failed")
    if not hashlib.sha256(actual).digest() == hashlib.sha256(encoded).digest():
        _fail("registry_update_failed")


def reap_references(
    registry_path: Path,
    reference_root: Path,
    *,
    execute: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Plan or execute reference erasure and return a sanitized audit document."""

    effective_now = now or datetime.now(timezone.utc)
    if effective_now.tzinfo is None:
        _fail("invalid_clock")
    effective_now = effective_now.astimezone(timezone.utc)

    registry_path = Path(registry_path)
    root = _reference_root(Path(reference_root))
    snapshot = _parse_registry(registry_path)

    candidates: list[_Candidate] = []
    target_users: dict[str, list[tuple[int, bool]]] = {}
    for row_index, row in enumerate(snapshot.document["profiles"]):
        target, exists = _inspect_target(root, row["audioFile"])
        triggers = _due_triggers(row, effective_now)
        if not triggers and not exists:
            _fail("invalid_registry")
        target_key = os.path.normcase(str(target))
        target_users.setdefault(target_key, []).append((row_index, bool(triggers)))
        candidates.append(
            _Candidate(
                row_index=row_index,
                row=row,
                target=target,
                target_key=target_key,
                exists=exists,
                triggers=triggers,
            )
        )

    due = [candidate for candidate in candidates if candidate.triggers]
    for candidate in due:
        if any(not is_due for _index, is_due in target_users[candidate.target_key]):
            _fail("reference_still_active")

    outcomes_by_target: dict[str, str] = {}
    if execute and due:
        _verify_unchanged(registry_path, snapshot.raw_sha256)
        for candidate in due:
            if candidate.target_key not in outcomes_by_target:
                outcomes_by_target[candidate.target_key] = _delete_target(
                    root, candidate
                )

        due_indexes = {candidate.row_index for candidate in due}
        updated_document = {
            "schema_version": 1,
            "profiles": [
                row
                for index, row in enumerate(snapshot.document["profiles"])
                if index not in due_indexes
            ],
        }
        _atomic_write_registry(
            registry_path,
            updated_document,
            mode=snapshot.mode,
            expected_digest=snapshot.raw_sha256,
        )

    items = []
    for candidate in due:
        if not execute:
            outcome = "planned" if candidate.exists else "already_absent"
        else:
            outcome = outcomes_by_target[candidate.target_key]
        items.append(
            {
                "row_index": candidate.row_index,
                "triggers": list(candidate.triggers),
                "outcome": outcome,
            }
        )

    return {
        "schema_version": 1,
        "mode": "execute" if execute else "dry_run",
        "evaluated": len(candidates),
        "due": len(due),
        "files_deleted": sum(
            1 for outcome in outcomes_by_target.values() if outcome == "deleted"
        ),
        "registry_rows_removed": len(due) if execute else 0,
        "registry_updated": bool(execute and due),
        "items": items,
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ReferenceReaperError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Dry-run or execute MOSS reference retention cleanup."
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete due WAV files and atomically remove their registry rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    mode = "unknown"
    try:
        args = _parser().parse_args(argv)
        mode = "execute" if args.execute else "dry_run"
        audit = reap_references(
            args.registry,
            args.reference_root,
            execute=args.execute,
        )
    except ReferenceReaperError as error:
        audit = {
            "schema_version": 1,
            "mode": mode,
            "status": "error",
            "error": error.code,
        }
        print(json.dumps(audit, ensure_ascii=True, separators=(",", ":")))
        return 2
    except Exception:
        audit = {
            "schema_version": 1,
            "mode": mode,
            "status": "error",
            "error": "internal_error",
        }
        print(json.dumps(audit, ensure_ascii=True, separators=(",", ":")))
        return 2

    print(json.dumps(audit, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
