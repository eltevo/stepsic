"""Read, select, compare, and write particle snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Mapping

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.uint64]
_GEOMETRIES = ("spherical", "cylindrical")


@dataclass(frozen=True)
class Snapshot:
    """The PartType1 fields and metadata needed by the validation tools.

    Coordinates are in Mpc/h, velocities in km/s times Gadget's
    ``sqrt(a)`` convention, and masses in 1e11 Msun/h.
    """

    coordinates: ArrayF
    velocities: ArrayF
    masses: ArrayF
    particle_ids: ArrayI
    header: Mapping[str, object]
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_part = len(self.particle_ids)
        if self.coordinates.shape != (n_part, 3):
            raise ValueError("coordinates must have shape (n_part, 3)")
        if self.velocities.shape != (n_part, 3):
            raise ValueError("velocities must have shape (n_part, 3)")
        if self.masses.shape != (n_part,):
            raise ValueError("masses must have shape (n_part,)")
        if not np.all(np.isfinite(self.coordinates)):
            raise ValueError("coordinates must be finite")
        if not np.all(np.isfinite(self.velocities)):
            raise ValueError("velocities must be finite")
        if not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0.0):
            raise ValueError("masses must be finite and positive")


def _copy_attributes(attributes: h5py.AttributeManager) -> dict[str, object]:
    return {name: value for name, value in attributes.items()}


def load_snapshot(path: str | Path) -> Snapshot:
    """Load one Gadget-HDF5 PartType1 catalog."""
    with h5py.File(path, "r") as handle:
        group = handle["PartType1"]
        parameters = (
            _copy_attributes(handle["Parameters"].attrs)
            if "Parameters" in handle else {}
        )
        return Snapshot(
            coordinates=group["Coordinates"][:].astype(np.float64),
            velocities=group["Velocities"][:].astype(np.float64),
            masses=group["Masses"][:].astype(np.float64),
            particle_ids=group["ParticleIDs"][:].astype(np.uint64),
            header=_copy_attributes(handle["Header"].attrs),
            parameters=parameters,
        )


def subset_snapshot(snapshot: Snapshot, selection: NDArray[np.bool_]) -> Snapshot:
    """Return a particle subset with its source metadata."""
    selection = np.asarray(selection, dtype=np.bool_)
    if selection.shape != (len(snapshot.particle_ids),):
        raise ValueError("selection must contain one boolean per particle")
    return Snapshot(
        coordinates=snapshot.coordinates[selection],
        velocities=snapshot.velocities[selection],
        masses=snapshot.masses[selection],
        particle_ids=snapshot.particle_ids[selection],
        header=dict(snapshot.header),
        parameters=dict(snapshot.parameters),
    )


def _validate_geometry(geometry: str) -> None:
    if geometry not in _GEOMETRIES:
        raise ValueError(
            f"geometry must be one of {_GEOMETRIES}, got {geometry!r}"
        )


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def transform_periodic_coordinates(
    coordinates_mpc_h: ArrayF,
    centre_mpc_h: ArrayF,
    box_size_mpc_h: float,
    geometry: str,
    length_mpc_h: float | None,
) -> ArrayF:
    """Minimum-image coordinates in the StePS estimator's output frame."""
    _validate_geometry(geometry)
    box_size_mpc_h = _validate_positive(box_size_mpc_h, "box_size_mpc_h")
    centre_mpc_h = np.asarray(centre_mpc_h, dtype=np.float64)
    if centre_mpc_h.shape != (3,) or not np.all(np.isfinite(centre_mpc_h)):
        raise ValueError("centre_mpc_h must contain three finite coordinates")
    coordinates_mpc_h = np.asarray(coordinates_mpc_h, dtype=np.float64)
    if coordinates_mpc_h.ndim != 2 or coordinates_mpc_h.shape[1:] != (3,):
        raise ValueError("coordinates_mpc_h must have shape (n_part, 3)")
    if not np.all(np.isfinite(coordinates_mpc_h)):
        raise ValueError("coordinates_mpc_h must be finite")

    centred = (
        coordinates_mpc_h - centre_mpc_h + box_size_mpc_h / 2.0
    ) % box_size_mpc_h - box_size_mpc_h / 2.0
    if geometry == "cylindrical":
        if length_mpc_h is None:
            raise ValueError("cylindrical geometry requires length_mpc_h")
        length_mpc_h = _validate_positive(length_mpc_h, "length_mpc_h")
        if length_mpc_h > box_size_mpc_h:
            raise ValueError("length_mpc_h cannot exceed box_size_mpc_h")
        centred[:, 2] += length_mpc_h / 2.0
    elif length_mpc_h is not None:
        raise ValueError("spherical geometry does not use length_mpc_h")
    return centred


def region_mask(
    coordinates_mpc_h: ArrayF,
    geometry: str,
    radius_mpc_h: float,
    length_mpc_h: float | None,
) -> NDArray[np.bool_]:
    """Select a centred sphere or a cylinder with corner-origin z."""
    _validate_geometry(geometry)
    radius_mpc_h = _validate_positive(radius_mpc_h, "radius_mpc_h")
    coordinates_mpc_h = np.asarray(coordinates_mpc_h, dtype=np.float64)
    if coordinates_mpc_h.ndim != 2 or coordinates_mpc_h.shape[1:] != (3,):
        raise ValueError("coordinates_mpc_h must have shape (n_part, 3)")
    if geometry == "spherical":
        if length_mpc_h is not None:
            raise ValueError("spherical geometry does not use length_mpc_h")
        return np.linalg.norm(coordinates_mpc_h, axis=1) <= radius_mpc_h

    if length_mpc_h is None:
        raise ValueError("cylindrical geometry requires length_mpc_h")
    length_mpc_h = _validate_positive(length_mpc_h, "length_mpc_h")
    radial = np.hypot(coordinates_mpc_h[:, 0], coordinates_mpc_h[:, 1])
    return (
        (radial <= radius_mpc_h)
        & (coordinates_mpc_h[:, 2] >= 0.0)
        & (coordinates_mpc_h[:, 2] < length_mpc_h)
    )


def match_snapshot_ids(snapshot: Snapshot, target_ids: ArrayI) -> Snapshot:
    """Reorder a snapshot to an exact requested particle-ID sequence."""
    target_ids = np.asarray(target_ids, dtype=np.uint64)
    if target_ids.ndim != 1:
        raise ValueError("target_ids must be one-dimensional")
    if len(np.unique(snapshot.particle_ids)) != len(snapshot.particle_ids):
        raise ValueError("source snapshot contains duplicate particle IDs")
    if len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("target IDs contain duplicate particle IDs")

    order = np.argsort(snapshot.particle_ids)
    sorted_ids = snapshot.particle_ids[order]
    locations = np.searchsorted(sorted_ids, target_ids)
    valid = locations < len(sorted_ids)
    valid[valid] &= sorted_ids[locations[valid]] == target_ids[valid]
    if not np.all(valid):
        missing = target_ids[~valid]
        raise ValueError(f"source snapshot is missing particle IDs {missing[:5]}")
    matched = order[locations]
    return Snapshot(
        coordinates=snapshot.coordinates[matched],
        velocities=snapshot.velocities[matched],
        masses=snapshot.masses[matched],
        particle_ids=snapshot.particle_ids[matched],
        header=dict(snapshot.header),
        parameters=dict(snapshot.parameters),
    )


def sample_uniform_region(
    n_part: int,
    geometry: str,
    radius_mpc_h: float,
    rng: np.random.Generator,
    *,
    length_mpc_h: float | None = None,
) -> ArrayF:
    """Sample a uniform sphere or a uniform cylinder."""
    _validate_geometry(geometry)
    if n_part < 0:
        raise ValueError("n_part cannot be negative")
    radius_mpc_h = _validate_positive(radius_mpc_h, "radius_mpc_h")
    if n_part == 0:
        return np.empty((0, 3), dtype=np.float64)

    azimuth = rng.uniform(0.0, 2.0 * np.pi, n_part)
    if geometry == "spherical":
        if length_mpc_h is not None:
            raise ValueError("spherical geometry does not use length_mpc_h")
        cos_polar = rng.uniform(-1.0, 1.0, n_part)
        sin_polar = np.sqrt(1.0 - cos_polar**2)
        radius = radius_mpc_h * np.cbrt(rng.uniform(size=n_part))
        return np.column_stack((
            radius * sin_polar * np.cos(azimuth),
            radius * sin_polar * np.sin(azimuth),
            radius * cos_polar,
        ))

    if length_mpc_h is None:
        raise ValueError("cylindrical geometry requires length_mpc_h")
    length_mpc_h = _validate_positive(length_mpc_h, "length_mpc_h")
    radius = radius_mpc_h * np.sqrt(rng.uniform(size=n_part))
    return np.column_stack((
        radius * np.cos(azimuth),
        radius * np.sin(azimuth),
        rng.uniform(0.0, length_mpc_h, n_part),
    ))


def _validate_pair_inputs(
    first: ArrayF,
    second: ArrayF,
    edges_mpc_h: ArrayF,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    arrays = []
    for name, value in (("first", first), ("second", second)):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[1:] != (3,):
            raise ValueError(f"{name} must have shape (n_part, 3)")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        arrays.append(value)
    edges_mpc_h = np.asarray(edges_mpc_h, dtype=np.float64)
    if (
        edges_mpc_h.ndim != 1
        or len(edges_mpc_h) == 0
        or not np.all(np.isfinite(edges_mpc_h))
        or np.any(edges_mpc_h <= 0.0)
        or np.any(np.diff(edges_mpc_h) <= 0.0)
    ):
        raise ValueError("edges_mpc_h must be finite, positive, and increasing")
    return arrays[0], arrays[1], edges_mpc_h


def cumulative_pair_counts(
    first: ArrayF,
    second: ArrayF,
    edges_mpc_h: ArrayF,
    *,
    same_catalog: bool,
    periodic_length_mpc_h: float | None,
) -> NDArray[np.int64]:
    """Count ordered pairs up to each edge, optionally periodic along z."""
    first, second, edges_mpc_h = _validate_pair_inputs(
        first, second, edges_mpc_h,
    )
    if same_catalog and (
        first.shape != second.shape or not np.array_equal(first, second)
    ):
        raise ValueError("same_catalog requires identical input coordinates")
    if len(first) == 0 or len(second) == 0:
        return np.zeros(len(edges_mpc_h), dtype=np.int64)

    if periodic_length_mpc_h is not None:
        periodic_length_mpc_h = _validate_positive(
            periodic_length_mpc_h, "periodic_length_mpc_h",
        )
        if edges_mpc_h[-1] >= periodic_length_mpc_h / 2.0:
            raise ValueError(
                "largest pair edge must be below half the periodic length"
            )
        tiled = np.concatenate((
            second - np.array([0.0, 0.0, periodic_length_mpc_h]),
            second,
            second + np.array([0.0, 0.0, periodic_length_mpc_h]),
        ))
        second_tree = cKDTree(tiled)
    else:
        second_tree = cKDTree(second)

    counts = np.asarray(
        cKDTree(first).count_neighbors(
            second_tree, edges_mpc_h, cumulative=True,
        ),
        dtype=np.int64,
    )
    if same_catalog:
        counts -= len(first)
    return counts


def landy_szalay(
    dd: NDArray[np.integer],
    dr: NDArray[np.integer],
    rr: NDArray[np.integer],
    n_data: int,
    n_random: int,
) -> ArrayF:
    """Landy-Szalay xi from ordered DD/RR and ordinary cross DR counts."""
    if n_data < 2 or n_random < 2:
        raise ValueError("Landy-Szalay requires at least two data and random points")
    dd = np.asarray(dd, dtype=np.float64)
    dr = np.asarray(dr, dtype=np.float64)
    rr = np.asarray(rr, dtype=np.float64)
    if dd.shape != dr.shape or dd.shape != rr.shape:
        raise ValueError("DD, DR, and RR counts must have identical shapes")
    if np.any(dd < 0.0) or np.any(dr < 0.0) or np.any(rr < 0.0):
        raise ValueError("pair counts cannot be negative")

    dd_norm = dd / (n_data * (n_data - 1))
    dr_norm = dr / (n_data * n_random)
    rr_norm = rr / (n_random * (n_random - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            rr_norm > 0.0,
            (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm,
            np.nan,
        )


def _updated_header(
    snapshot: Snapshot,
    n_part: int,
    box_size_mpc_h: float,
    simulation_radius_mpc_h: float,
) -> dict[str, object]:
    header = dict(snapshot.header)
    ntypes = 6
    for key in ("NumPart_ThisFile", "NumPart_Total", "MassTable"):
        if key in header:
            value = np.asarray(header[key])
            if value.ndim == 1:
                ntypes = max(ntypes, len(value))
    counts = np.zeros(ntypes, dtype=np.uint64)
    counts[1] = n_part
    header["NumPart_ThisFile"] = counts
    header["NumPart_Total"] = counts
    header["NumPart_Total_HighWord"] = np.zeros(ntypes, dtype=np.uint32)
    header["MassTable"] = np.zeros(ntypes, dtype=np.float64)
    header["NumFilesPerSnapshot"] = 1
    header["BoxSize"] = float(box_size_mpc_h)
    header["SimulationRadius"] = float(simulation_radius_mpc_h)
    for key in ("Omega0", "OmegaLambda", "HubbleParam"):
        if key not in header and key in snapshot.parameters:
            header[key] = snapshot.parameters[key]
    return header


def write_snapshot(
    output: str | Path,
    snapshot: Snapshot,
    *,
    box_size_mpc_h: float,
    simulation_radius_mpc_h: float,
) -> None:
    """Write a reproducible Gadget-HDF5 PartType1 snapshot atomically."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    box_size_mpc_h = _validate_positive(box_size_mpc_h, "box_size_mpc_h")
    simulation_radius_mpc_h = _validate_positive(
        simulation_radius_mpc_h, "simulation_radius_mpc_h",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w", track_order=True) as handle:
            header_group = handle.create_group("Header", track_order=True)
            header = _updated_header(
                snapshot, len(snapshot.particle_ids),
                box_size_mpc_h, simulation_radius_mpc_h,
            )
            for name, value in header.items():
                header_group.attrs[name] = value
            if snapshot.parameters:
                parameter_group = handle.create_group(
                    "Parameters", track_order=True,
                )
                for name, value in snapshot.parameters.items():
                    parameter_group.attrs[name] = value
            particle_group = handle.create_group("PartType1", track_order=True)
            particle_group.create_dataset(
                "Coordinates", data=snapshot.coordinates,
            )
            particle_group.create_dataset(
                "Velocities", data=snapshot.velocities,
            )
            particle_group.create_dataset("Masses", data=snapshot.masses)
            particle_group.create_dataset(
                "ParticleIDs", data=snapshot.particle_ids,
            )
            handle.flush()
        file_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_glass_snapshot(
    path: str | Path,
    *,
    geometry: str,
    radius_mpc_h: float,
    length_mpc_h: float | None = None,
) -> None:
    """Validate the geometry and metadata required of a glass input."""
    snapshot = load_snapshot(path)
    if len(snapshot.particle_ids) == 0:
        raise ValueError("glass snapshot contains no particles")
    if len(np.unique(snapshot.particle_ids)) != len(snapshot.particle_ids):
        raise ValueError("glass snapshot contains duplicate particle IDs")
    radius_mpc_h = _validate_positive(radius_mpc_h, "radius_mpc_h")
    try:
        header_radius = float(snapshot.header["SimulationRadius"])
        topology_value = snapshot.header["TopologicalManifold"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("glass snapshot lacks geometry metadata") from error
    if isinstance(topology_value, bytes):
        topology = topology_value.decode("ascii")
    else:
        topology = str(topology_value)
    if not np.isclose(header_radius, radius_mpc_h, rtol=1e-12, atol=0.0):
        raise ValueError("glass snapshot has the wrong simulation radius")
    if geometry == "spherical":
        if length_mpc_h is not None:
            raise ValueError("spherical glass does not use length_mpc_h")
        if topology != "R^3":
            raise ValueError("spherical glass has the wrong topology")
    elif geometry == "cylindrical":
        if length_mpc_h is None:
            raise ValueError("cylindrical glass requires length_mpc_h")
        length_mpc_h = _validate_positive(length_mpc_h, "length_mpc_h")
        if topology != "S^1xR^2":
            raise ValueError("cylindrical glass has the wrong topology")
        try:
            box_size = float(snapshot.header["BoxSize"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cylindrical glass lacks its axial period") from error
        if not np.isclose(box_size, length_mpc_h, rtol=1e-12, atol=0.0):
            raise ValueError("cylindrical glass has the wrong axial period")
        z = snapshot.coordinates[:, 2]
        tolerance = 32.0 * np.finfo(np.float64).eps * length_mpc_h
        if np.any(z < -tolerance) or np.any(z >= length_mpc_h + tolerance):
            raise ValueError("cylindrical glass lies outside its axial period")
    else:
        raise ValueError("glass geometry must be spherical or cylindrical")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate a pre-generated glass snapshot.")
    parser.add_argument("path")
    parser.add_argument("--geometry", choices=("spherical", "cylindrical"), required=True)
    parser.add_argument("--radius-mpc-h", type=float, required=True)
    parser.add_argument("--length-mpc-h", type=float)
    args = parser.parse_args()
    validate_glass_snapshot(
        args.path,
        geometry=args.geometry,
        radius_mpc_h=args.radius_mpc_h,
        length_mpc_h=args.length_mpc_h,
    )


if __name__ == "__main__":
    _main()
