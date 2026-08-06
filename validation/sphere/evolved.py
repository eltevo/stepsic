"""Pure contracts and estimators for the evolved spherical validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import h5py
import numpy as np
from numpy.typing import NDArray


PAIR_KIND = "stepsic-periodic-matched-pair"
FIELD_HASH_ALGORITHM = "sha256-canonical-hdf5-dataset"


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically publish strict JSON in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def snapshot_diagnostics(path: str | Path) -> dict[str, Any]:
    """Read epoch and numerical-resolution diagnostics from one snapshot."""
    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        particles = handle["PartType1"]
        particle_count = int(len(particles["Coordinates"]))
        if "Masses" in particles:
            masses = particles["Masses"][:].astype(np.float64)
        else:
            mass_table = np.asarray(header["MassTable"], dtype=np.float64)
            masses = np.full(particle_count, mass_table[1], dtype=np.float64)
        scale_factor = float(header.get("Time", np.nan))
        if "Redshift" in header:
            redshift = float(header["Redshift"])
        elif np.isfinite(scale_factor) and scale_factor > 0.0:
            redshift = 1.0 / scale_factor - 1.0
        else:
            redshift = np.nan
    if (
        particle_count == 0
        or not np.isfinite(scale_factor)
        or not np.isfinite(redshift)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError(f"{path} has invalid epoch or particle metadata")
    return {
        "final_scale_factor": scale_factor,
        "final_redshift": redshift,
        "particle_count": particle_count,
        "mass_min_1e11_msun_h": float(np.min(masses)),
        "mass_median_1e11_msun_h": float(np.median(masses)),
        "mass_max_1e11_msun_h": float(np.max(masses)),
    }


def canonical_dataset_sha256(
    path: str | Path,
    dataset_name: str,
) -> str:
    """Hash the logical HDF5 dataset, independently of container metadata."""
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"{path} contains no {dataset_name!r} dataset")
        values = np.asarray(handle[dataset_name][...])
    if values.dtype.hasobject:
        raise ValueError("object-valued HDF5 datasets cannot be canonicalized")

    canonical_dtype = values.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(values.astype(canonical_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(FIELD_HASH_ALGORITHM.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
    name: str,
) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing {', '.join(missing)}")


def validate_pair_configuration(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the reference manifest against the spherical campaign."""
    if manifest.get("kind") != PAIR_KIND:
        raise ValueError(f"reference manifest kind must be {PAIR_KIND!r}")

    paths = _mapping(manifest.get("paths"), "paths")
    _require_keys(
        paths, ("snapshot", "ic", "load", "white_noise", "delta_k"), "paths",
    )
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a nonempty path")

    extent = _mapping(manifest.get("extent"), "extent")
    radius_mpc_h = float(expected["radius_mpc_h"])
    expected_box_size_mpc_h = 2.0 * radius_mpc_h
    if extent.get("box_size_mpc_h") != expected_box_size_mpc_h:
        raise ValueError(
            "periodic box size must equal the StePS diameter "
            f"(2 * R_3D = {expected_box_size_mpc_h} Mpc/h)"
        )
    expected_periodic_origin = [radius_mpc_h] * 3
    if extent.get("origin_mpc_h") != expected_periodic_origin:
        raise ValueError(
            "periodic origin must be the box centre corresponding to the "
            "StePS origin"
        )
    if expected.get("origin_mpc_h") != [0.0, 0.0, 0.0]:
        raise ValueError("StePS origin must be [0, 0, 0]")

    for section_name in ("cosmology", "initial_conditions"):
        actual_section = _mapping(manifest.get(section_name), section_name)
        expected_section = _mapping(expected.get(section_name), section_name)
        for key, expected_value in expected_section.items():
            if key not in actual_section:
                raise ValueError(f"{section_name}.{key} is missing")
            if actual_section[key] != expected_value:
                raise ValueError(
                    f"{section_name}.{key} mismatch: "
                    f"{actual_section[key]!r} != {expected_value!r}"
                )

    fields = _mapping(manifest.get("fields"), "fields")
    _require_keys(
        fields,
        ("hash_algorithm", "ic_white_noise", "ic_delta_k"),
        "fields",
    )
    if fields["hash_algorithm"] != FIELD_HASH_ALGORITHM:
        raise ValueError(
            f"fields.hash_algorithm must be {FIELD_HASH_ALGORITHM!r}"
        )
    for key in ("ic_white_noise", "ic_delta_k"):
        value = fields[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"fields.{key} must be a lowercase SHA-256 digest")

    evolution = _mapping(manifest.get("evolution"), "evolution")
    _require_keys(
        evolution,
        (
            "final_scale_factor",
            "final_redshift",
            "particle_count",
            "particle_mass_1e11_msun_h",
            "softening_mpc_h",
            "pm_grid",
            "requested_final_redshift",
        ),
        "evolution",
    )
    if (
        "final_redshift" in expected
        and evolution["requested_final_redshift"] != expected["final_redshift"]
    ):
        raise ValueError(
            "evolution.requested_final_redshift mismatch: "
            f"{evolution['requested_final_redshift']!r} != "
            f"{expected['final_redshift']!r}"
        )
    revisions = _mapping(manifest.get("revisions"), "revisions")
    _require_keys(revisions, ("stepsic", "simulation_code"), "revisions")
    return manifest


def verify_field_hashes(
    manifest: Mapping[str, Any],
    white_noise_path: str | Path,
    white_noise_dataset: str,
    delta_k_path: str | Path,
    delta_k_dataset: str,
) -> dict[str, str]:
    """Require exact field-content agreement with the periodic reference."""
    fields = _mapping(manifest.get("fields"), "fields")
    actual = {
        "ic_white_noise": canonical_dataset_sha256(
            white_noise_path, white_noise_dataset,
        ),
        "ic_delta_k": canonical_dataset_sha256(delta_k_path, delta_k_dataset),
    }
    for key, digest in actual.items():
        if fields.get(key) != digest:
            raise ValueError(
                f"{key} hash mismatch: {digest} != {fields.get(key)!r}"
            )
    return actual


def parse_shells(
    specification: str,
    *,
    radius_mpc_h: float,
    rcrit_mpc_h: float,
) -> NDArray[np.float64]:
    """Parse two ordered, disjoint representative spherical shells."""
    radius_mpc_h = float(radius_mpc_h)
    rcrit_mpc_h = float(rcrit_mpc_h)
    if (
        not np.isfinite(radius_mpc_h)
        or not np.isfinite(rcrit_mpc_h)
        or not 0.0 < rcrit_mpc_h < radius_mpc_h
    ):
        raise ValueError("RCRIT and R_3D must satisfy 0 < RCRIT < R_3D")
    if specification.strip():
        bounds = []
        for shell in specification.split(","):
            pieces = shell.split(":")
            if len(pieces) != 2:
                raise ValueError("CL_SHELLS must use inner:outer pairs")
            try:
                bounds.append([float(pieces[0]), float(pieces[1])])
            except ValueError as error:
                raise ValueError("CL_SHELLS bounds must be numeric") from error
        shells = np.asarray(bounds, dtype=np.float64)
    else:
        outer_span_mpc_h = radius_mpc_h - rcrit_mpc_h
        shells = np.asarray(
            [
                [0.4 * rcrit_mpc_h, 0.6 * rcrit_mpc_h],
                [
                    rcrit_mpc_h + 0.4 * outer_span_mpc_h,
                    rcrit_mpc_h + 0.6 * outer_span_mpc_h,
                ],
            ],
            dtype=np.float64,
        )
    if shells.shape != (2, 2):
        raise ValueError("CL_SHELLS must define exactly two shells")
    if not np.all(np.isfinite(shells)):
        raise ValueError("CL_SHELLS bounds must be finite")
    if np.any(shells[:, 0] < 0.0) or np.any(shells[:, 1] <= shells[:, 0]):
        raise ValueError("each CL_SHELLS interval must have 0 <= inner < outer")
    if shells[0, 1] > shells[1, 0]:
        raise ValueError("CL_SHELLS intervals must be ordered and non-overlapping")
    if shells[-1, 1] > radius_mpc_h:
        raise ValueError("CL_SHELLS bounds must lie inside R_3D")
    if shells[0, 1] > rcrit_mpc_h:
        raise ValueError("the inner CL_SHELLS interval must lie inside RCRIT")
    if shells[1, 0] < rcrit_mpc_h:
        raise ValueError("the outer CL_SHELLS interval must lie outside RCRIT")
    return shells


def minimum_image_displacements(
    coordinates_mpc_h: NDArray[np.floating],
    *,
    origin_mpc_h: NDArray[np.floating],
    box_size_mpc_h: float,
) -> NDArray[np.float64]:
    """Return displacements from one periodic origin using minimum images."""
    coordinates = np.asarray(coordinates_mpc_h, dtype=np.float64)
    origin = np.asarray(origin_mpc_h, dtype=np.float64)
    box_size_mpc_h = float(box_size_mpc_h)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("coordinates_mpc_h must have shape (n_part, 3)")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_mpc_h must contain three finite coordinates")
    if (
        not np.all(np.isfinite(coordinates))
        or not np.isfinite(box_size_mpc_h)
        or box_size_mpc_h <= 0.0
    ):
        raise ValueError("coordinates and box_size_mpc_h must be finite")
    displacement = coordinates - origin
    displacement -= box_size_mpc_h * np.rint(
        displacement / box_size_mpc_h
    )
    return displacement


def measure_angular_spectra(
    displacements_mpc_h: NDArray[np.floating],
    masses_1e11_msun_h: NDArray[np.floating],
    shells_mpc_h: NDArray[np.floating],
    *,
    nside: int,
    lmax: int,
    omega_m: float,
) -> dict[str, NDArray[Any]]:
    """Measure mass-weighted HEALPix ``C_ell`` in centered radial shells.

    Positions are displacements from the selected origin in Mpc/h. Particle
    masses are in ``1e11 Msun/h``. The density normalization uses
    ``rho_crit = 2.77536627e11 Msun h^2 / Mpc^3``, which is numerically
    ``2.77536627`` in the mass and length units used here.
    """
    import healpy as hp

    displacements = np.asarray(displacements_mpc_h, dtype=np.float64)
    masses = np.asarray(masses_1e11_msun_h, dtype=np.float64)
    shells = np.asarray(shells_mpc_h, dtype=np.float64)
    if displacements.ndim != 2 or displacements.shape[1:] != (3,):
        raise ValueError("displacements_mpc_h must have shape (n_part, 3)")
    if masses.shape != (len(displacements),) or len(masses) == 0:
        raise ValueError("masses must contain one value per particle")
    if (
        not np.all(np.isfinite(displacements))
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("particle displacements and positive masses must be finite")
    if shells.ndim != 2 or shells.shape[1:] != (2,) or len(shells) == 0:
        raise ValueError("shells_mpc_h must have shape (n_shell, 2)")
    if (
        not np.all(np.isfinite(shells))
        or np.any(shells[:, 0] < 0.0)
        or np.any(shells[:, 1] <= shells[:, 0])
        or np.any(shells[1:, 0] < shells[:-1, 1])
    ):
        raise ValueError("shell bounds must be finite, ordered, and non-overlapping")
    if not hp.isnsideok(nside) or lmax < 0 or lmax > 3 * nside - 1:
        raise ValueError("nside must be valid and 0 <= lmax <= 3*nside-1")
    omega_m = float(omega_m)
    if not np.isfinite(omega_m) or omega_m <= 0.0:
        raise ValueError("omega_m must be finite and positive")

    npix = hp.nside2npix(nside)
    mean_density_1e11_msun_h_per_mpc_h3 = 2.77536627 * omega_m
    radii = np.linalg.norm(displacements, axis=1)
    spectra = []
    density_contrasts = []
    particle_counts = []
    mass_sums = []
    mass_means = []
    mass_minima = []
    mass_maxima = []
    sector_volumes = []
    for inner_mpc_h, outer_mpc_h in shells:
        selected = (radii >= inner_mpc_h) & (radii < outer_mpc_h)
        if not np.any(selected):
            raise ValueError(
                f"shell [{inner_mpc_h}, {outer_mpc_h}) contains no particles"
            )
        shell_displacements = displacements[selected]
        shell_radii = radii[selected]
        directions = shell_displacements / shell_radii[:, None]
        pixels = hp.vec2pix(
            nside,
            directions[:, 0],
            directions[:, 1],
            directions[:, 2],
            nest=False,
        )
        shell_masses = masses[selected]
        mass_per_pixel = np.bincount(
            pixels, weights=shell_masses, minlength=npix,
        ).astype(np.float64)
        sector_volume_mpc_h3 = (
            (4.0 * np.pi / npix)
            * (outer_mpc_h**3 - inner_mpc_h**3)
            / 3.0
        )
        delta = (
            mass_per_pixel
            / sector_volume_mpc_h3
            / mean_density_1e11_msun_h_per_mpc_h3
            - 1.0
        )
        spectra.append(hp.anafast(delta, lmax=lmax, iter=3))
        density_contrasts.append(delta)
        particle_counts.append(int(np.count_nonzero(selected)))
        mass_sums.append(float(np.sum(shell_masses, dtype=np.float64)))
        mass_means.append(float(np.mean(shell_masses)))
        mass_minima.append(float(np.min(shell_masses)))
        mass_maxima.append(float(np.max(shell_masses)))
        sector_volumes.append(sector_volume_mpc_h3)

    return {
        "ell": np.arange(lmax + 1, dtype=np.int64),
        "cl": np.asarray(spectra, dtype=np.float64),
        "delta_maps": np.asarray(density_contrasts, dtype=np.float64),
        "particle_count": np.asarray(particle_counts, dtype=np.int64),
        "mass_sum_1e11_msun_h": np.asarray(mass_sums, dtype=np.float64),
        "mass_mean_1e11_msun_h": np.asarray(mass_means, dtype=np.float64),
        "mass_min_1e11_msun_h": np.asarray(mass_minima, dtype=np.float64),
        "mass_max_1e11_msun_h": np.asarray(mass_maxima, dtype=np.float64),
        "pixel_sector_volume_mpc_h3": np.asarray(
            sector_volumes, dtype=np.float64,
        ),
    }


def physical_pk_to_h_units(
    k_per_mpc: NDArray[np.floating],
    power_mpc3: NDArray[np.floating],
    *,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ``1/Mpc, Mpc^3`` to ``h/Mpc, (Mpc/h)^3``."""
    k = np.asarray(k_per_mpc, dtype=np.float64)
    power = np.asarray(power_mpc3, dtype=np.float64)
    h = float(h)
    if k.shape != power.shape or k.ndim != 1 or k.size == 0:
        raise ValueError("k and power must be nonempty one-dimensional peers")
    if (
        not np.all(np.isfinite(k))
        or not np.all(np.isfinite(power))
        or not np.isfinite(h)
        or h <= 0.0
    ):
        raise ValueError("k, power, and h must be finite, with h positive")
    return k / h, power * h**3


def measure_full_periodic_pk(
    coordinates_mpc_h: NDArray[np.floating],
    masses_1e11_msun_h: NDArray[np.floating],
    *,
    box_size_mpc_h: float,
    nmesh: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Pass the complete periodic particle arrays to the native estimator."""
    from stepsic.pk import measure_pk

    coordinates = np.asarray(coordinates_mpc_h, dtype=np.float64)
    masses = np.asarray(masses_1e11_msun_h, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or len(coordinates) == 0:
        raise ValueError("coordinates must be a nonempty (n_part, 3) array")
    if masses.shape != (len(coordinates),):
        raise ValueError("masses must contain one value per periodic particle")
    if (
        not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
        or not np.isfinite(box_size_mpc_h)
        or box_size_mpc_h <= 0.0
        or nmesh < 1
    ):
        raise ValueError("periodic estimator inputs must be finite and positive")
    return measure_pk(
        coordinates,
        np.full(3, box_size_mpc_h),
        nmesh=nmesh,
        mass=masses,
        method="cic",
        deconvolve=True,
        interlace=True,
        subtract_shot=True,
    )


def generate_full_selection_randoms(
    coordinates_mpc_h: NDArray[np.floating],
    masses_1e11_msun_h: NDArray[np.floating],
    *,
    factor: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample every glass radius and mass with isotropic directions."""
    coordinates = np.asarray(coordinates_mpc_h, dtype=np.float64)
    masses = np.asarray(masses_1e11_msun_h, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("coordinates_mpc_h must have shape (n_part, 3)")
    if masses.shape != (len(coordinates),) or len(masses) == 0:
        raise ValueError("masses must contain one value per glass particle")
    if (
        not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("glass coordinates and positive masses must be finite")
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError("factor must be a positive integer")

    rng = np.random.default_rng(seed)
    source = np.tile(np.arange(len(coordinates)), int(factor))
    rng.shuffle(source)
    radii = np.linalg.norm(coordinates[source], axis=1)
    cos_polar = rng.uniform(-1.0, 1.0, len(source))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, len(source))
    sin_polar = np.sqrt(1.0 - cos_polar**2)
    random_coordinates = np.column_stack(
        (
            radii * sin_polar * np.cos(azimuth),
            radii * sin_polar * np.sin(azimuth),
            radii * cos_polar,
        )
    )
    return random_coordinates, masses[source].copy()
