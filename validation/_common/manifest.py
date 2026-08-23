#!/usr/bin/env python3
"""Record the inputs and outputs used by each cached campaign step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable


def _is_generated(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_state(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path)
    state: dict[str, Any] = {"path": raw_path}
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {**state, "exists": False, "type": "missing", "sha256": None}

    state["exists"] = True
    mode = metadata.st_mode
    if stat.S_ISREG(mode):
        state.update(type="file", sha256=_hash_file(path))
        return state
    if stat.S_ISLNK(mode):
        target = os.readlink(path)
        digest = hashlib.sha256(target.encode("utf-8", "surrogateescape"))
        state.update(type="symlink", target=target, sha256=digest.hexdigest())
        return state
    if stat.S_ISDIR(mode):
        digest = hashlib.sha256()
        entries = sorted(
            (entry for entry in path.rglob("*")
             if not _is_generated(entry.relative_to(path))),
            key=lambda item: item.as_posix(),
        )
        for entry in entries:
            relative = entry.relative_to(path).as_posix()
            entry_metadata = entry.lstat()
            digest.update(relative.encode("utf-8", "surrogateescape"))
            if entry.is_symlink():
                digest.update(b"L")
                digest.update(
                    os.readlink(entry).encode("utf-8", "surrogateescape")
                )
            elif entry.is_file():
                digest.update(b"F")
                digest.update(_hash_file(entry).encode("ascii"))
            elif entry.is_dir():
                digest.update(b"D")
            else:
                digest.update(f"M{entry_metadata.st_mode}".encode("ascii"))
        state.update(type="directory", sha256=digest.hexdigest())
        return state
    state.update(type="other", sha256=None)
    return state


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "step": args.step,
        "arguments": list(args.argument),
        "configuration": list(args.configuration),
        "inputs": [_path_state(path) for path in _ordered_unique(args.input)],
        "implementations": [
            _path_state(path)
            for path in _ordered_unique(args.implementation)
        ],
        "outputs": [_path_state(path) for path in _ordered_unique(args.output)],
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("matches", "write"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--argument", action="append", default=[])
    parser.add_argument("--configuration", action="append", default=[])
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--implementation", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = Path(args.manifest)
    payload = _payload(args)
    if args.action == "write":
        _atomic_write(manifest, payload)
        return 0
    try:
        with manifest.open(encoding="utf-8") as stream:
            existing = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 1
    return 0 if existing == payload else 1


if __name__ == "__main__":
    raise SystemExit(main())
