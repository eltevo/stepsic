#!/usr/bin/env python3
"""Validated, crash-safe cache management for StePS S1 x R2 Ewald tables."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

SCHEMA_VERSION = 1
RUN_MANIFEST = ".ewald-run-v1.json"
REQUIRED_PARAMS = (
    "HubbleConstant",
    "IS_PERIODIC",
    "L_BOX",
    "R_SIM",
    "H_INDEPENDENT_UNITS",
)
MODE_NZ = {"lowres": 128, "medres": 256, "higres": 512}
STAGE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
PROVENANCE_ATTRS = (
    "ProgramName",
    "ProgramVersion",
    "ProgramCommitID",
    "ProgramBranchName",
    "ProgramBuildDate",
    "ProgramCompiler",
)


class EwaldCacheError(Exception):
    """Base class for expected CLI failures."""

    exit_code = 4


class InputError(EwaldCacheError):
    """Invalid caller-controlled path, option, or parameter file."""

    exit_code = 2


class NoTableError(EwaldCacheError):
    """No generated table is available to promote."""

    exit_code = 3


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise InputError(f"{label} is not a regular file: {candidate}")
    return candidate


def _directory(path: str | Path, label: str, *, create: bool) -> Path:
    candidate = Path(path).expanduser().resolve()
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir():
        raise InputError(f"{label} is not a directory: {candidate}")
    return candidate


def _canonical_decimal(raw: str, key: str, *, positive: bool = True) -> str:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise InputError(f"{key} must be numeric, got {raw!r}") from exc
    if not value.is_finite():
        raise InputError(f"{key} must be finite, got {raw!r}")
    if positive and value <= 0:
        raise InputError(f"{key} must be positive, got {raw!r}")
    normalized = format(value.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def _parse_params(path: Path) -> dict[str, str | int]:
    found: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields or fields[0] not in REQUIRED_PARAMS:
                continue
            key = fields[0]
            if len(fields) != 2:
                raise InputError(f"{path}:{line_number}: {key} requires exactly one value")
            if key in found:
                raise InputError(f"{path}:{line_number}: duplicate parameter {key}")
            found[key] = fields[1]
    missing = [key for key in REQUIRED_PARAMS if key not in found]
    if missing:
        raise InputError(f"{path}: missing required parameters: {', '.join(missing)}")

    try:
        periodic = int(found["IS_PERIODIC"])
        h_units = int(found["H_INDEPENDENT_UNITS"])
    except ValueError as exc:
        raise InputError("IS_PERIODIC and H_INDEPENDENT_UNITS must be integers") from exc
    if periodic < 2:
        raise InputError(f"IS_PERIODIC must be at least 2 for an Ewald table, got {periodic}")
    if h_units not in (0, 1):
        raise InputError(f"H_INDEPENDENT_UNITS must be 0 or 1, got {h_units}")

    return {
        "HubbleConstant": _canonical_decimal(found["HubbleConstant"], "HubbleConstant"),
        "IS_PERIODIC": periodic,
        "L_BOX": _canonical_decimal(found["L_BOX"], "L_BOX"),
        "R_SIM": _canonical_decimal(found["R_SIM"], "R_SIM"),
        "H_INDEPENDENT_UNITS": h_units,
    }


def _table_name(mode: str) -> str:
    if mode not in MODE_NZ:
        raise InputError(f"mode must be one of {', '.join(MODE_NZ)}, got {mode!r}")
    return f"S1R2_Ewald_table_{mode}.hdf5"


def _validate_mode(mode: str, periodic: int) -> None:
    expected = "lowres" if periodic == 2 else "medres" if periodic == 3 else "higres"
    if mode != expected:
        raise InputError(f"IS_PERIODIC={periodic} requires mode {expected}, got {mode}")


def _identity(binary: Path, param: Path, mode: str) -> dict[str, Any]:
    parameters = _parse_params(param)
    _validate_mode(mode, int(parameters["IS_PERIODIC"]))
    core = {
        "schema_version": SCHEMA_VERSION,
        "topology": "S1R2",
        "mode": mode,
        "table_name": _table_name(mode),
        "binary_sha256": _sha256(binary),
        "parameters": parameters,
    }
    fingerprint = hashlib.sha256(_canonical_json(core)).hexdigest()
    return {"fingerprint": fingerprint, "identity": core}


def _identity_from_args(args: argparse.Namespace) -> dict[str, Any]:
    binary = _regular_file(args.binary, "binary")
    param = _regular_file(args.param, "parameter file")
    result = _identity(binary, param, args.mode)
    result["binary_path"] = str(binary)
    result["param_path"] = str(param)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    return value


def _close_enough(actual: float, expected: float, precision: int) -> bool:
    relative = 2e-6 if precision == 0 else 1e-12
    absolute = 1e-5 if precision == 0 else 1e-10
    return math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute)


def _inspect_table(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EwaldCacheError(f"Ewald table is not a regular, non-symlink file: {path}")
    parameters = identity["parameters"]
    expected_nz = MODE_NZ[str(identity["mode"])]
    try:
        with h5py.File(path, "r") as handle:
            if "Header" not in handle or "s1r2_ewald/S1R2_EWALD_FORCE_TABLE" not in handle:
                raise EwaldCacheError(f"Ewald table has missing groups or dataset: {path}")
            header = handle["Header"]
            required_attrs = (
                "BoxSize",
                "SimulationRadius",
                "EWALD_GRID_SIZE_RHO",
                "EWALD_GRID_SIZE_Z",
                "EWALD_PRECISION",
            )
            missing = [name for name in required_attrs if name not in header.attrs]
            if missing:
                raise EwaldCacheError(f"Ewald table is missing attributes: {', '.join(missing)}")
            precision = int(header.attrs["EWALD_PRECISION"])
            if precision not in (0, 1):
                raise EwaldCacheError(f"unsupported EWALD_PRECISION={precision}")
            nrho = int(header.attrs["EWALD_GRID_SIZE_RHO"])
            nz = int(header.attrs["EWALD_GRID_SIZE_Z"])
            if nrho <= 0 or nz != expected_nz:
                raise EwaldCacheError(
                    f"Ewald grid metadata ({nrho}, {nz}) is incompatible with {identity['mode']}"
                )
            dataset = handle["s1r2_ewald/S1R2_EWALD_FORCE_TABLE"]
            expected_dtype = np.dtype(np.float32 if precision == 0 else np.float64)
            if dataset.shape != (nrho, nz, 2):
                raise EwaldCacheError(
                    f"Ewald dataset shape {dataset.shape} does not match header {(nrho, nz, 2)}"
                )
            if dataset.dtype != expected_dtype:
                raise EwaldCacheError(
                    f"Ewald dataset dtype {dataset.dtype} does not match precision {precision}"
                )
            for start in range(0, nrho, 64):
                if not np.isfinite(dataset[start : start + 64]).all():
                    raise EwaldCacheError("Ewald dataset contains non-finite values")

            box_size = float(header.attrs["BoxSize"])
            simulation_radius = float(header.attrs["SimulationRadius"])
            expected_box = float(Decimal(str(parameters["L_BOX"])))
            expected_radius = float(Decimal(str(parameters["R_SIM"])))
            if int(parameters["H_INDEPENDENT_UNITS"]) == 1:
                expected_radius *= float(Decimal(str(parameters["HubbleConstant"]))) / 100.0
            if not _close_enough(box_size, expected_box, precision):
                raise EwaldCacheError(
                    f"Ewald BoxSize {box_size} does not match expected {expected_box}"
                )
            if not _close_enough(simulation_radius, expected_radius, precision):
                raise EwaldCacheError(
                    f"Ewald SimulationRadius {simulation_radius} does not match expected {expected_radius}"
                )
            provenance = {
                name: _json_value(header.attrs[name])
                for name in PROVENANCE_ATTRS
                if name in header.attrs
            }
    except OSError as exc:
        raise EwaldCacheError(f"cannot read Ewald HDF5 table {path}: {exc}") from exc
    return {
        "box_size": box_size,
        "simulation_radius": simulation_radius,
        "grid_size_rho": nrho,
        "grid_size_z": nz,
        "precision": precision,
        "dtype": expected_dtype.name,
        "shape": [nrho, nz, 2],
        "provenance": provenance,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EwaldCacheError(f"{label} is not a regular, non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EwaldCacheError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EwaldCacheError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_identity_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = record.get("identity")
    fingerprint = record.get("fingerprint")
    if not isinstance(identity, dict) or not isinstance(fingerprint, str):
        raise EwaldCacheError("manifest lacks identity or fingerprint")
    expected = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or fingerprint != expected:
        raise EwaldCacheError("manifest fingerprint does not match its identity")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise EwaldCacheError("unsupported Ewald cache schema")
    mode = identity.get("mode")
    if identity.get("table_name") != _table_name(str(mode)):
        raise EwaldCacheError("manifest table name is inconsistent with mode")
    return fingerprint, identity


@contextmanager
def _entry_lock(entry: Path) -> Iterator[None]:
    entry.mkdir(parents=True, exist_ok=True)
    lock_path = entry / ".lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cache_entry(cache_root: Path, fingerprint: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise EwaldCacheError("invalid cache fingerprint")
    return cache_root / "v1" / fingerprint


def _validate_cache_entry(entry: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(entry / "manifest.json", "cache manifest")
    fingerprint, identity = _validate_identity_record(manifest)
    if fingerprint != expected["fingerprint"] or identity != expected["identity"]:
        raise EwaldCacheError("cache manifest identity does not match requested run")
    table = entry / str(identity["table_name"])
    metadata = _inspect_table(table, identity)
    checksum = _sha256(table)
    if manifest.get("table_sha256") != checksum or manifest.get("hdf5") != metadata:
        raise EwaldCacheError("cache table checksum or HDF5 metadata does not match manifest")
    return {"manifest": manifest, "table": table, "checksum": checksum, "metadata": metadata}


def command_identity(args: argparse.Namespace) -> int:
    print(json.dumps(_identity_from_args(args), sort_keys=True))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    if not STAGE_NAME_RE.fullmatch(args.stage_name):
        raise InputError("stage name may contain only letters, digits, dot, underscore, and hyphen")
    stage = _directory(args.stage_dir, "stage directory", create=True)
    cache_root = _directory(args.cache_root, "cache root", create=True)
    record = _identity_from_args(args)
    fingerprint = str(record["fingerprint"])
    identity = record["identity"]
    entry = _cache_entry(cache_root, fingerprint)
    staged = False
    with _entry_lock(entry):
        manifest_exists = (entry / "manifest.json").exists()
        table_exists = (entry / str(identity["table_name"])).exists()
        if manifest_exists:
            validated = _validate_cache_entry(entry, record)
            _atomic_copy(validated["table"], stage / str(identity["table_name"]))
            staged = True
        elif table_exists:
            # A crash may leave the atomically-renamed table visible before the
            # manifest commit. Without the manifest it is not a cache entry;
            # leave it untouched and let a later promotion replace it safely.
            staged = False
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "binary_path": record["binary_path"],
        "param_path": record["param_path"],
        "stage_name": args.stage_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(stage / RUN_MANIFEST, run_manifest)
    print(json.dumps({"fingerprint": fingerprint, "staged": staged}, sort_keys=True))
    return 0


def _promote(stage: Path, cache_root: Path, attempt_log: str | None) -> dict[str, Any]:
    run_manifest = _read_json(stage / RUN_MANIFEST, "run manifest")
    fingerprint, identity = _validate_identity_record(run_manifest)
    if not isinstance(run_manifest.get("stage_name"), str):
        raise EwaldCacheError("run manifest lacks stage_name")
    source = stage / str(identity["table_name"])
    if not source.exists():
        raise NoTableError(f"no generated Ewald table to promote: {source}")
    metadata = _inspect_table(source, identity)
    checksum = _sha256(source)
    cache_manifest = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "table_sha256": checksum,
        "hdf5": metadata,
        "origin": {
            "stage_name": run_manifest["stage_name"],
            "binary_path": run_manifest.get("binary_path"),
            "param_path": run_manifest.get("param_path"),
            "run_created_at": run_manifest.get("created_at"),
            "attempt_log": str(Path(attempt_log).expanduser().resolve()) if attempt_log else None,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    entry = _cache_entry(cache_root, fingerprint)
    destination = entry / str(identity["table_name"])
    with _entry_lock(entry):
        if (entry / "manifest.json").exists():
            existing = _validate_cache_entry(
                entry, {"fingerprint": fingerprint, "identity": identity}
            )
            if existing["checksum"] != checksum:
                raise EwaldCacheError(
                    f"cache conflict for {fingerprint}: existing and generated checksums differ"
                )
            status = "existing"
        else:
            _atomic_copy(source, destination)
            _write_json(entry / "manifest.json", cache_manifest)
            status = "promoted"
    return {
        "fingerprint": fingerprint,
        "status": status,
        "table": str(destination),
        "table_sha256": checksum,
    }


def command_promote(args: argparse.Namespace) -> int:
    stage = _directory(args.stage_dir, "stage directory", create=False)
    cache_root = _directory(args.cache_root, "cache root", create=True)
    print(json.dumps(_promote(stage, cache_root, args.attempt_log), sort_keys=True))
    return 0


def command_recover(args: argparse.Namespace) -> int:
    stage = _directory(args.stage_dir, "stage directory", create=True)
    cache_root = _directory(args.cache_root, "cache root", create=True)
    manifest = stage / RUN_MANIFEST
    if not manifest.exists():
        print(json.dumps({"status": "no-manifest"}, sort_keys=True))
        return 0
    identity_record = _read_json(manifest, "run manifest")
    _, identity = _validate_identity_record(identity_record)
    if not (stage / str(identity["table_name"])).exists():
        print(json.dumps({"status": "no-table"}, sort_keys=True))
        return 0
    print(json.dumps(_promote(stage, cache_root, None), sort_keys=True))
    return 0


def command_check_missing_error(args: argparse.Namespace) -> int:
    log = _regular_file(args.log, "attempt log")
    table = Path(args.table).expanduser().resolve()
    if table.exists() or table.is_symlink():
        return 1
    text = log.read_text(encoding="utf-8", errors="replace")
    markers = (
        f"Ewald lookup table file ({table}) found.",
        f"name = '{table}', errno = 2, error message = 'No such file or directory'",
        f"HDF5: cannot open file {table}",
        f"Error: Failed to load the Ewald lookup table from file {table}",
    )
    return 0 if all(marker in text for marker in markers) else 1


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_NZ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity")
    _add_identity_args(identity)
    identity.set_defaults(handler=command_identity)

    prepare = subparsers.add_parser("prepare")
    _add_identity_args(prepare)
    prepare.add_argument("--stage-dir", required=True)
    prepare.add_argument("--cache-root", required=True)
    prepare.add_argument("--stage-name", required=True)
    prepare.set_defaults(handler=command_prepare)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--stage-dir", required=True)
    promote.add_argument("--cache-root", required=True)
    promote.add_argument("--attempt-log")
    promote.set_defaults(handler=command_promote)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--stage-dir", required=True)
    recover.add_argument("--cache-root", required=True)
    recover.set_defaults(handler=command_recover)

    missing = subparsers.add_parser("check-missing-error")
    missing.add_argument("--log", required=True)
    missing.add_argument("--table", required=True)
    missing.set_defaults(handler=command_check_missing_error)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except EwaldCacheError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"ERROR: filesystem operation failed: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
