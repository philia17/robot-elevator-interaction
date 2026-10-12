#!/usr/bin/env python3
"""Incrementally import and validate raw RGB-D capture sessions from AGX.

The remote side is read-only: discovery uses ``ssh`` and transfer uses rsync
without any delete option.  A session is accepted only when metadata.jsonl is
an exact manifest for the RGB and depth files copied into it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


DEFAULT_HOST = "agx-wifi"
DEFAULT_REMOTE_ROOT = "/home/robot/datasets/raw_capture"
_SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*_session$")
_RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}
_DEPTH_SUFFIXES = {".png"}


class ImportCaptureError(RuntimeError):
    """Raised when discovery, transfer, or validation cannot be completed."""


class SessionValidationError(ImportCaptureError):
    """Raised when a local session violates the raw-capture contract."""


@dataclass(frozen=True)
class SessionValidationReport:
    session: str
    pair_count: int
    rgb_count: int
    depth_count: int
    metadata_count: int


def normalize_date(value: str) -> str:
    """Return a capture date as YYYYMMDD, accepting YYYY-MM-DD as input."""
    compact = value.replace("-", "")
    if not re.fullmatch(r"\d{8}", compact):
        raise argparse.ArgumentTypeError("date must be YYYYMMDD or YYYY-MM-DD")
    try:
        dt.datetime.strptime(compact, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date: {value}") from exc
    return compact


def _check_connection_inputs(host: str, remote_root: str) -> None:
    if not _SAFE_HOST_RE.fullmatch(host):
        raise ImportCaptureError(f"unsafe SSH host: {host!r}")
    if not remote_root.startswith("/") or "\n" in remote_root or "\r" in remote_root:
        raise ImportCaptureError("remote root must be an absolute path without newlines")


def discover_remote_sessions(
    host: str,
    remote_root: str,
    capture_date: str,
    *,
    runner: Any = subprocess.run,
) -> list[str]:
    """List remote session basenames for ``capture_date`` without changing AGX."""
    capture_date = normalize_date(capture_date)
    _check_connection_inputs(host, remote_root)
    command = (
        f"find {shlex.quote(remote_root)} -mindepth 1 -maxdepth 1 -type d "
        f"-name {shlex.quote('*' + capture_date + '*_session')} -printf '%f\\n'"
    )
    try:
        completed = runner(
            ["ssh", host, command],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ImportCaptureError(f"remote session discovery failed: {detail.strip()}") from exc

    sessions: list[str] = []
    for raw_name in completed.stdout.splitlines():
        name = raw_name.strip()
        if not name:
            continue
        if not _SAFE_SESSION_RE.fullmatch(name) or capture_date not in name:
            raise ImportCaptureError(f"remote returned an unsafe session name: {name!r}")
        sessions.append(name)
    return sorted(set(sessions))


def build_rsync_command(
    host: str,
    remote_root: str,
    session: str,
    destination: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Build the incremental rsync command for one session.

    Only raw RGB images, depth PNGs, and metadata.jsonl are included.  In
    particular, ``images/overlay`` is never imported.
    """
    _check_connection_inputs(host, remote_root)
    if not _SAFE_SESSION_RE.fullmatch(session):
        raise ImportCaptureError(f"unsafe session name: {session!r}")
    remote_session = f"{remote_root.rstrip('/')}/{session}/"
    command = [
        "rsync",
        "-a",
        "--partial",
        "--itemize-changes",
        "--prune-empty-dirs",
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "--exclude=/images/overlay/",
            "--exclude=/images/overlay/***",
            "--include=/images/",
            "--include=/images/***",
            "--include=/depth/",
            "--include=/depth/***",
            "--include=/metadata.jsonl",
            "--exclude=*",
            f"{host}:{shlex.quote(remote_session)}",
            str(destination / session) + "/",
        ]
    )
    return command


def _manifest_path(value: Any, field: str, line_no: int) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SessionValidationError(
            f"metadata line {line_no}: {field} must be a non-empty string"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise SessionValidationError(
            f"metadata line {line_no}: unsafe {field} path {value!r}"
        )
    return path


def _files_below(directory: Path, suffixes: set[str]) -> set[PurePosixPath]:
    if not directory.is_dir():
        return set()
    result: set[PurePosixPath] = set()
    for path in directory.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts):
            if path.suffix.lower() in suffixes:
                result.add(PurePosixPath(path.relative_to(directory.parent).as_posix()))
    return result


def validate_session(session_dir: Path) -> SessionValidationReport:
    """Strictly validate one imported raw-capture session.

    The metadata references and supported files physically present below
    images/ and depth/ must be identical sets.  Symlinks and paths outside the
    session are rejected.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise SessionValidationError(f"session directory not found: {session_dir}")
    if session_dir.is_symlink():
        raise SessionValidationError(f"session directory may not be a symlink: {session_dir}")
    metadata_path = session_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise SessionValidationError(f"missing metadata.jsonl: {session_dir}")
    if metadata_path.is_symlink():
        raise SessionValidationError(f"metadata.jsonl may not be a symlink: {session_dir}")

    expected_rgb: set[PurePosixPath] = set()
    expected_depth: set[PurePosixPath] = set()
    metadata_count = 0
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SessionValidationError(f"cannot read {metadata_path}: {exc}") from exc

    for line_no, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SessionValidationError(
                f"invalid JSON at {metadata_path}:{line_no}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise SessionValidationError(
                f"metadata line {line_no}: expected a JSON object"
            )
        rgb = _manifest_path(record.get("filename"), "filename", line_no)
        depth = _manifest_path(record.get("depth_filename"), "depth_filename", line_no)
        if not rgb.parts or rgb.parts[0] != "images" or rgb.suffix.lower() not in _RGB_SUFFIXES:
            raise SessionValidationError(
                f"metadata line {line_no}: RGB path must be a supported file below images/"
            )
        if "overlay" in rgb.parts:
            raise SessionValidationError(
                f"metadata line {line_no}: overlay captures are not imported"
            )
        if not depth.parts or depth.parts[0] != "depth" or depth.suffix.lower() not in _DEPTH_SUFFIXES:
            raise SessionValidationError(
                f"metadata line {line_no}: depth path must be a PNG below depth/"
            )
        if rgb in expected_rgb:
            raise SessionValidationError(f"duplicate RGB reference at metadata line {line_no}: {rgb}")
        if depth in expected_depth:
            raise SessionValidationError(
                f"duplicate depth reference at metadata line {line_no}: {depth}"
            )
        expected_rgb.add(rgb)
        expected_depth.add(depth)
        metadata_count += 1

    if metadata_count == 0:
        raise SessionValidationError(f"metadata contains no capture records: {metadata_path}")

    actual_rgb = _files_below(session_dir / "images", _RGB_SUFFIXES)
    actual_depth = _files_below(session_dir / "depth", _DEPTH_SUFFIXES)
    errors: list[str] = []
    missing_rgb = expected_rgb - actual_rgb
    extra_rgb = actual_rgb - expected_rgb
    missing_depth = expected_depth - actual_depth
    extra_depth = actual_depth - expected_depth
    if missing_rgb:
        errors.append("missing RGB: " + ", ".join(map(str, sorted(missing_rgb))))
    if extra_rgb:
        errors.append("unreferenced RGB: " + ", ".join(map(str, sorted(extra_rgb))))
    if missing_depth:
        errors.append("missing depth: " + ", ".join(map(str, sorted(missing_depth))))
    if extra_depth:
        errors.append("unreferenced depth: " + ", ".join(map(str, sorted(extra_depth))))
    if errors:
        raise SessionValidationError(f"invalid session {session_dir.name}: " + "; ".join(errors))

    # Ensure no manifest path resolves through a symlink or outside the session.
    resolved_session = session_dir.resolve()
    for relative in expected_rgb | expected_depth:
        candidate = session_dir / relative.as_posix()
        try:
            candidate.resolve(strict=True).relative_to(resolved_session)
        except (OSError, ValueError) as exc:
            raise SessionValidationError(f"unsafe or missing capture file: {relative}") from exc
        if candidate.is_symlink():
            raise SessionValidationError(f"capture files may not be symlinks: {relative}")

    return SessionValidationReport(
        session=session_dir.name,
        pair_count=metadata_count,
        rgb_count=len(actual_rgb),
        depth_count=len(actual_depth),
        metadata_count=metadata_count,
    )


def import_sessions(
    host: str,
    remote_root: str,
    capture_date: str,
    destination: Path,
    *,
    dry_run: bool = False,
    runner: Any = subprocess.run,
) -> list[SessionValidationReport]:
    """Discover, incrementally rsync, and validate all sessions for a date."""
    capture_date = normalize_date(capture_date)
    destination = Path(destination).expanduser().resolve()
    sessions = discover_remote_sessions(
        host, remote_root, capture_date, runner=runner
    )
    if not sessions:
        raise ImportCaptureError(f"no remote sessions found for {capture_date}")
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    reports: list[SessionValidationReport] = []
    for session in sessions:
        command = build_rsync_command(
            host, remote_root, session, destination, dry_run=dry_run
        )
        try:
            runner(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ImportCaptureError(f"rsync failed for {session}: {exc}") from exc
        if not dry_run:
            reports.append(validate_session(destination / session))
    return reports


def _default_destination(capture_date: str) -> Path:
    return Path.home() / "Datasets" / "Lift_Robot" / "annotation_workspace" / capture_date


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    today = dt.date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="Import raw RGB-D sessions from AGX and strictly validate them"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--date", type=normalize_date, default=today)
    parser.add_argument(
        "--destination",
        type=Path,
        help="local directory for this date (default: ~/Datasets/Lift_Robot/annotation_workspace/YYYYMMDD)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show rsync changes without writing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.destination or _default_destination(args.date)
    try:
        reports = import_sessions(
            args.host,
            args.remote_root,
            args.date,
            destination,
            dry_run=args.dry_run,
        )
    except ImportCaptureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"Dry run complete; no files written to {destination}")
    else:
        total = sum(report.pair_count for report in reports)
        for report in reports:
            print(f"OK {report.session}: {report.pair_count} RGB-D pairs")
        print(f"Imported and validated {total} pairs across {len(reports)} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
