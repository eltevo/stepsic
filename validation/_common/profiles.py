"""Load campaign settings for the small, medium, and large run sizes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Mapping

import toml


Scalar = bool | int | float | str
ProfileValue = Scalar | list[Scalar]
_SIZES = ("small", "medium", "large")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


@dataclass(frozen=True)
class Campaign:
    entrypoint: str
    group: str
    prerequisites: str
    keys: tuple[str, ...]
    evaluates: bool = True


CAMPAIGNS: dict[str, Campaign] = {
    "particle-loads": Campaign(
        "particle-loads/run.sh", "paper", "stepsic environment",
        ("NGRID", "NPART", "NRBINS", "NSHELL"),
    ),
    "shell-mass-profiles": Campaign(
        "shell-mass-profiles/run.sh", "paper", "stepsic environment",
        ("NRBINS", "NSHELL"),
    ),
    "field-statistics": Campaign(
        "field-statistics/run.sh", "supplementary", "stepsic environment",
        ("NMESH_3D", "NREAL", "NBINS"),
    ),
    "slab-sampling": Campaign(
        "slab-sampling/run.sh", "paper", "stepsic environment",
        (
            "FIELD_NMESH", "FIELD_NREAL", "FIELD_NBINS",
            "TRANSFER_NSTEPS", "TRANSFER_NMESH", "TRANSFER_NREAL",
            "TRANSFER_PAIRED", "FAIR_NMESH", "FAIR_ASPECT",
            "FAIR_NREAL", "FAIR_PAIRED",
        ),
    ),
    "grid-power-recovery": Campaign(
        "grid-power-recovery/run.sh", "paper", "stepsic environment",
        (
            "NREAL", "PAIRED",
            "PANEL_A_NMESH", "PANEL_A_LPT", "PANEL_A_Z", "PANEL_A_METHOD",
            "PANEL_B_NMESH", "PANEL_B_LPT", "PANEL_B_Z", "PANEL_B_METHOD",
            "PANEL_C_NMESH", "PANEL_C_LPT", "PANEL_C_Z", "PANEL_C_METHOD",
            "PANEL_D_NMESH", "PANEL_D_LPT", "PANEL_D_Z", "PANEL_D_METHOD",
        ),
    ),
    "glass-quality": Campaign(
        "glass-quality/run.sh", "supplementary",
        "stepsic and StePS environments; StePS source",
        (
            "NGRID", "NPART", "NRBINS_STEPS", "NSHELL_STEPS",
            "NRBINS_CUBICAL", "NSHELL_CUBICAL", "GLASS_TIME_LIMIT_MIN",
            "GLASS_FIRST_T_OUT", "GLASS_H_OUT", "DIAG_NZONES",
            "DIAG_NCUBES", "DIAG_PK_NMESH", "DIAG_NPROF",
            "RESCALE_ASPECTS", "FORCE_TIME_LIMIT_MIN",
        ),
    ),
    "cylindrical-lpt": Campaign(
        "cylindrical-lpt/run.sh", "paper",
        "stepsic and StePS environments; StePS source or GLASS_SNAP",
        (
            "NRBINS", "NSHELL", "SIM_NMESH", "PK_NMESH",
            "PK_RANDOMS_NFACTOR", "PK_NFKP_RADIAL_BINS",
            "GLASS_TIME_LIMIT_MIN", "SIM_TIME_LIMIT_MIN",
        ),
    ),
    "monofonic-agreement": Campaign(
        "monofonic-agreement/run.sh", "paper",
        "stepsic and monofonic environments; network for first build",
        ("LBOX", "NMESH"),
    ),
    "periodic-embedding": Campaign(
        "periodic-embedding/run.sh", "supplementary", "stepsic environment",
        ("NGRID0", "ALPHAS", "NSEEDS", "NRBINS", "NSHELL", "PK_NMESH", "NPROF"),
    ),
    "spherical-ic": Campaign(
        "spherical-ic/run.sh", "supplementary", "stepsic environment",
        (
            "R_3D", "RCRIT", "NRBINS", "NSHELL", "NMESH",
            "NMESHSAMPLES", "NCMP", "PK_NMESH",
        ),
    ),
    "periodic-lpt": Campaign(
        "periodic-lpt/run.sh", "supplementary",
        "Gadget-4 source and build environment",
        ("REF_NGRID", "REF_NMESH", "REF_PMGRID", "PK_NMESH"),
    ),
    "spherical-evolution": Campaign(
        "spherical-evolution/run.sh", "extended",
        "REFERENCE_MANIFEST, spherical GLASS_SNAP, StePS source and environment",
        ("PAIR_NGRID", "SIM_NMESH", "CL_NSIDE", "CL_LMAX", "PK_NMESH", "PK_RANDOM_FACTOR"),
        False,
    ),
}


@dataclass(frozen=True)
class LoadedProfile:
    path: Path
    sha256: str
    status: str
    tables: Mapping[str, Mapping[str, ProfileValue]]


@dataclass(frozen=True)
class ProfileSet:
    profiles: Mapping[str, LoadedProfile]


@dataclass(frozen=True)
class ResolvedCampaign:
    size: str
    campaign: str
    profile_file: Path
    profile_sha256: str
    values: Mapping[str, ProfileValue]
    sources: Mapping[str, str]


def _scalar_type(value: object) -> type[Scalar]:
    if type(value) not in (bool, int, float, str):
        raise ValueError(
            "profile values must be typed scalars or non-empty flat arrays"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("profile numeric values must be finite")
    return type(value)


def _type_signature(value: object) -> tuple[str, type[Scalar]]:
    if isinstance(value, list):
        if not value:
            raise ValueError("profile arrays must be non-empty")
        if any(isinstance(item, (list, dict)) for item in value):
            raise ValueError("profile values may contain only flat arrays")
        item_types = {_scalar_type(item) for item in value}
        if len(item_types) != 1:
            raise ValueError("profile arrays must contain one scalar type")
        return "array", item_types.pop()
    return "scalar", _scalar_type(value)


def _load_one(path: Path, size: str) -> LoadedProfile:
    try:
        data = toml.load(path)
    except (OSError, toml.TomlDecodeError) as error:
        raise ValueError(f"cannot load profile {path}: {error}") from error
    metadata = data.pop("profile", None)
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name}: missing [profile] metadata")
    unknown_metadata = set(metadata) - {"status", "message"}
    if unknown_metadata:
        raise ValueError(
            f"{path.name}: unknown profile metadata: {sorted(unknown_metadata)}"
        )
    status = metadata.get("status")
    if status not in ("defined", "reserved"):
        raise ValueError(f"{path.name}: profile status must be defined or reserved")
    if size != "large" and status != "defined":
        raise ValueError(f"{path.name}: profile must be defined")
    if status == "reserved" and data:
        raise ValueError(f"{path.name}: reserved profile cannot define campaigns")

    unknown_campaigns = set(data) - set(CAMPAIGNS)
    if unknown_campaigns:
        raise ValueError(
            f"{path.name}: unknown campaign tables: {sorted(unknown_campaigns)}"
        )
    if status == "defined":
        if size != "large":
            missing_campaigns = set(CAMPAIGNS) - set(data)
            if missing_campaigns:
                raise ValueError(
                    f"{path.name}: missing campaign tables: {sorted(missing_campaigns)}"
                )
        for campaign_name, table in data.items():
            campaign = CAMPAIGNS[campaign_name]
            if not isinstance(table, dict):
                raise ValueError(f"{path.name}: [{campaign_name}] must be a table")
            unknown_keys = set(table) - set(campaign.keys)
            if unknown_keys:
                raise ValueError(
                    f"{path.name}: [{campaign_name}] unknown keys: {sorted(unknown_keys)}"
                )
            missing_keys = set(campaign.keys) - set(table)
            if missing_keys:
                raise ValueError(
                    f"{path.name}: [{campaign_name}] missing keys: {sorted(missing_keys)}"
                )
            for key, value in table.items():
                try:
                    _type_signature(value)
                except ValueError as error:
                    raise ValueError(
                        f"{path.name}: [{campaign_name}].{key}: {error}"
                    ) from error

    content = path.read_bytes()
    return LoadedProfile(
        path=path.resolve(),
        sha256=hashlib.sha256(content).hexdigest(),
        status=status,
        tables=data,
    )


def load_profile_set(directory: str | Path) -> ProfileSet:
    directory = Path(directory)
    loaded = {
        size: _load_one(directory / f"{size}.toml", size) for size in _SIZES
    }
    for campaign_name, campaign in CAMPAIGNS.items():
        for key in campaign.keys:
            small_value = loaded["small"].tables[campaign_name][key]
            medium_value = loaded["medium"].tables[campaign_name][key]
            if _type_signature(small_value) != _type_signature(medium_value):
                raise ValueError(
                    f"inconsistent type for [{campaign_name}].{key} "
                    "between small and medium profiles"
                )
    return ProfileSet(loaded)


def _parse_override(text: str, example: ProfileValue, key: str) -> ProfileValue:
    signature = _type_signature(example)
    scalar_type = signature[1]

    def parse_scalar(raw: str) -> Scalar:
        if scalar_type is str:
            return raw
        if scalar_type is bool:
            lowered = raw.lower()
            if lowered not in ("true", "false"):
                raise ValueError(f"environment override {key} must be true or false")
            return lowered == "true"
        if scalar_type is int:
            if not _INTEGER.fullmatch(raw):
                raise ValueError(f"environment override {key} must be an integer")
            return int(raw)
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"environment override {key} must be finite")
        return value

    if signature[0] == "array":
        try:
            parts = shlex.split(text)
        except ValueError as error:
            raise ValueError(f"environment override {key}: {error}") from error
        if not parts:
            raise ValueError(f"environment override {key} must be non-empty")
        return [parse_scalar(part) for part in parts]
    return parse_scalar(text)


def _validate_resolved(campaign: str, values: Mapping[str, ProfileValue]) -> None:
    if campaign == "slab-sampling":
        nmesh = int(values["FAIR_NMESH"])
        aspect = int(values["FAIR_ASPECT"])
        if aspect <= 0 or nmesh <= 0 or nmesh % (2 * aspect) != 0:
            raise ValueError(
                "slab-sampling FAIR_NMESH must be positive and divisible by "
                "2*FAIR_ASPECT"
            )


def resolve_campaign(
    profile_set: ProfileSet,
    size: str,
    campaign: str,
    environ: Mapping[str, str] | None = None,
) -> ResolvedCampaign:
    if size not in _SIZES:
        raise ValueError(
            f"invalid size {size!r}; expected one of: {', '.join(_SIZES)}"
        )
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}")
    profile = profile_set.profiles[size]
    if profile.status == "reserved" or campaign not in profile.tables:
        raise ValueError(
            f"{size} profile has no approved configuration for {campaign}"
        )
    environ = os.environ if environ is None else environ
    values: dict[str, ProfileValue] = {}
    sources: dict[str, str] = {}
    for key in CAMPAIGNS[campaign].keys:
        profile_value = profile.tables[campaign][key]
        if key in environ:
            values[key] = _parse_override(environ[key], profile_value, key)
            sources[key] = "environment"
        else:
            values[key] = profile_value
            sources[key] = "profile"
    _validate_resolved(campaign, values)
    return ResolvedCampaign(
        size=size,
        campaign=campaign,
        profile_file=profile.path,
        profile_sha256=profile.sha256,
        values=values,
        sources=sources,
    )


def resolve_custom(
    profile_set: ProfileSet,
    path: str | Path,
    campaign: str,
    environ: Mapping[str, str] | None = None,
) -> ResolvedCampaign:
    """Load one campaign table from a custom TOML profile."""
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}")
    custom_path = Path(path)
    try:
        data = toml.load(custom_path)
        content = custom_path.read_bytes()
    except (OSError, toml.TomlDecodeError) as error:
        raise ValueError(f"cannot load custom profile {custom_path}: {error}") from error
    if set(data) != {campaign} or not isinstance(data[campaign], dict):
        raise ValueError(
            f"custom profile must contain exactly one [{campaign}] table"
        )
    table = data[campaign]
    expected = set(CAMPAIGNS[campaign].keys)
    unknown = set(table) - expected
    missing = expected - set(table)
    if unknown:
        raise ValueError(f"custom profile unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"custom profile missing keys: {sorted(missing)}")
    examples = profile_set.profiles["small"].tables[campaign]
    for key, value in table.items():
        try:
            if _type_signature(value) != _type_signature(examples[key]):
                raise ValueError("type differs from the registered profile key")
        except ValueError as error:
            raise ValueError(f"custom profile [{campaign}].{key}: {error}") from error

    environ = os.environ if environ is None else environ
    values: dict[str, ProfileValue] = {}
    sources: dict[str, str] = {}
    for key in CAMPAIGNS[campaign].keys:
        if key in environ:
            values[key] = _parse_override(environ[key], table[key], key)
            sources[key] = "environment"
        else:
            values[key] = table[key]
            sources[key] = "custom"
    _validate_resolved(campaign, values)
    digest = hashlib.sha256(content).hexdigest()
    return ResolvedCampaign(
        size="custom",
        campaign=campaign,
        profile_file=custom_path.resolve(),
        profile_sha256=digest,
        values=values,
        sources=sources,
    )


def shell_value(value: ProfileValue) -> str:
    if isinstance(value, list):
        return " ".join(shell_value(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _atomic_document(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(toml.dumps(data))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_toml(path: Path, resolved: ResolvedCampaign) -> None:
    _atomic_document(path, {
        "profile": {
            "size": resolved.size,
            "campaign": resolved.campaign,
            "sha256": resolved.profile_sha256,
        },
        "values": dict(resolved.values),
        "sources": dict(resolved.sources),
    })


def _write_runtime_config(
    path: Path,
    entries: list[tuple[str, str, str]],
) -> None:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key, source, value in entries:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid resolved configuration key: {key!r}")
        if key in values:
            raise ValueError(f"duplicate resolved configuration key: {key}")
        if source not in ("environment", "config.env", "cosmology"):
            raise ValueError(f"invalid resolved configuration source: {source!r}")
        if "\0" in value:
            raise ValueError(f"resolved configuration value for {key} contains NUL")
        values[key] = value
        sources[key] = source
    _atomic_document(path, {"values": values, "sources": sources})


def _emit_nul(resolved: ResolvedCampaign) -> None:
    records = {
        "VLIB_PROFILE_FILE": str(resolved.profile_file),
        "VLIB_PROFILE_SHA256": resolved.profile_sha256,
        **{key: shell_value(value) for key, value in resolved.values.items()},
    }
    output = sys.stdout.buffer
    for key, value in records.items():
        output.write(key.encode() + b"\0" + value.encode() + b"\0")


def _parser() -> argparse.ArgumentParser:
    default_directory = Path(__file__).parents[1] / "_profiles"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=default_directory)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    selection = resolve.add_mutually_exclusive_group(required=True)
    selection.add_argument("--size")
    selection.add_argument("--config", type=Path)
    resolve.add_argument("--campaign", required=True)
    resolve.add_argument("--write", type=Path)
    runtime = subparsers.add_parser("runtime-config")
    runtime.add_argument("--output", type=Path, required=True)
    runtime.add_argument("--entry", nargs=3, action="append", default=[])
    subparsers.add_parser("catalog")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile_set = load_profile_set(args.profile_dir)
        if args.command == "runtime-config":
            _write_runtime_config(args.output, args.entry)
            return 0
        if args.command == "catalog":
            for name, campaign in CAMPAIGNS.items():
                print(
                    f"{name}\t{campaign.entrypoint}\t{campaign.group}\t"
                    f"{campaign.prerequisites}\t{str(campaign.evaluates).lower()}"
                )
            return 0
        if args.config is None:
            resolved = resolve_campaign(profile_set, args.size, args.campaign)
        else:
            resolved = resolve_custom(profile_set, args.config, args.campaign)
        if args.write is not None:
            _atomic_toml(args.write, resolved)
        _emit_nul(resolved)
        return 0
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
