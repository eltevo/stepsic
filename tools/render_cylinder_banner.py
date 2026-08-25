#!/usr/bin/env python3
"""Render the cylindrical StePS volume used for the project banner.

Install the renderer with ``pip install -e '.[visualization]'``. The ``render``
command projects a stepsic snapshot in its default Mpc/h and 1e11 Msun/h
units. The ``plot`` command changes the presentation without repeating that
projection.

Examples:
    python tools/render_cylinder_banner.py render snapshot.hdf5 \
        -o docs/assets/banner.png
    python tools/render_cylinder_banner.py plot output/cylinder-banner.npz \
        -o /tmp/banner.pdf --cmap inferno
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import tempfile
from typing import Iterator

import h5py
import numpy as np
from scipy.spatial import cKDTree


FIGURE_WIDTH_INCHES = 7.09
DEFAULT_ASPECT = 3.5
DEFAULT_ELEVATION_DEG = 25.0
DEFAULT_AZIMUTH_DEG = 0.0
DEFAULT_NEIGHBOURS = 32
DEFAULT_RESOLUTION_LEVEL = 12
DEFAULT_THREADS = 8
DEFAULT_FOV_PADDING = 1.05
IMAGE_SUFFIXES = {".pdf", ".png", ".svg"}

log = logging.getLogger(__name__)


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float_arg(value: str) -> float:
    try:
        return _positive_float(float(value), "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _int_arg(value: str) -> int:
    try:
        return _positive_int(int(value), "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def load_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load PartType1 and the cylinder dimensions from a stepsic snapshot."""
    with h5py.File(path, "r") as handle:
        if "Header" not in handle or "PartType1" not in handle:
            raise ValueError(f"{path}: expected Header and PartType1 groups")
        header = handle["Header"].attrs
        particles = handle["PartType1"]
        if "Coordinates" not in particles:
            raise ValueError(f"{path}: missing PartType1/Coordinates")
        positions_mpc_h = particles["Coordinates"][:].astype(np.float64)
        if "Masses" in particles:
            masses_1e11_msun_h = particles["Masses"][:].astype(np.float64)
        else:
            mass_table = np.asarray(header.get("MassTable", []))
            if mass_table.shape != (6,) or not np.isfinite(mass_table[1]):
                raise ValueError(f"{path}: missing PartType1 masses")
            masses_1e11_msun_h = np.full(len(positions_mpc_h), mass_table[1])
        radius_mpc_h = float(header.get("SimulationRadius", np.nan))
        height_mpc_h = float(header.get("BoxSize", np.nan))

    if positions_mpc_h.ndim != 2 or positions_mpc_h.shape[1:] != (3,):
        raise ValueError(f"{path}: Coordinates must have shape (n_part, 3)")
    if len(positions_mpc_h) == 0:
        raise ValueError(f"{path}: snapshot contains no particles")
    if masses_1e11_msun_h.shape != (len(positions_mpc_h),):
        raise ValueError(f"{path}: Masses must contain one value per particle")
    if not np.all(np.isfinite(positions_mpc_h)):
        raise ValueError(f"{path}: Coordinates must be finite")
    if not np.all(np.isfinite(masses_1e11_msun_h)) or np.any(
        masses_1e11_msun_h <= 0.0
    ):
        raise ValueError(f"{path}: Masses must be finite and positive")

    diameter_mpc_h = 2.0 * _positive_float(
        radius_mpc_h, "Header/SimulationRadius"
    )
    height_mpc_h = _positive_float(height_mpc_h, "Header/BoxSize")
    return positions_mpc_h, masses_1e11_msun_h, diameter_mpc_h, height_mpc_h


def compute_fov_width(
    diameter_mpc_h: float,
    height_mpc_h: float,
    elevation_deg: float,
    padding: float = DEFAULT_FOV_PADDING,
) -> float:
    """Return the square field width containing the tilted cylinder."""
    diameter_mpc_h = _positive_float(diameter_mpc_h, "diameter_mpc_h")
    height_mpc_h = _positive_float(height_mpc_h, "height_mpc_h")
    padding = _positive_float(padding, "padding")
    if not np.isfinite(elevation_deg):
        raise ValueError("elevation_deg must be finite")
    elevation_rad = np.radians(elevation_deg)
    projected_height_mpc_h = (
        height_mpc_h * abs(np.cos(elevation_rad))
        + diameter_mpc_h * abs(np.sin(elevation_rad))
    )
    return max(diameter_mpc_h, projected_height_mpc_h) * padding


def estimate_smoothing_lengths(
    positions_mpc_h: np.ndarray,
    neighbours: int = DEFAULT_NEIGHBOURS,
) -> np.ndarray:
    """Use non-periodic nearest-neighbour distances as smoothing lengths."""
    neighbours = _positive_int(neighbours, "neighbours")
    if neighbours >= len(positions_mpc_h):
        raise ValueError("neighbours must be smaller than the particle count")
    distances_mpc_h, _ = cKDTree(positions_mpc_h).query(
        positions_mpc_h, k=neighbours + 1, workers=-1
    )
    smoothing_lengths_mpc_h = distances_mpc_h[:, -1]
    if not np.all(np.isfinite(smoothing_lengths_mpc_h)) or np.any(
        smoothing_lengths_mpc_h <= 0.0
    ):
        raise ValueError("smoothing lengths must be finite and positive")
    return smoothing_lengths_mpc_h


def render_image(
    positions_mpc_h: np.ndarray,
    masses_1e11_msun_h: np.ndarray,
    smoothing_lengths_mpc_h: np.ndarray,
    fov_width_mpc_h: float,
    *,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    threads: int = DEFAULT_THREADS,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the cylinder without wrapping its open radial boundaries."""
    import sphviewer2

    if masses_1e11_msun_h.shape != (len(positions_mpc_h),):
        raise ValueError("masses must contain one value per particle")
    if smoothing_lengths_mpc_h.shape != (len(positions_mpc_h),):
        raise ValueError("smoothing lengths must contain one value per particle")
    if not np.isfinite(elevation_deg) or not np.isfinite(azimuth_deg):
        raise ValueError("camera angles must be finite")
    renderer_box_mpc_h = _positive_float(
        np.ptp(positions_mpc_h, axis=0).max() * 1.01, "particle extent"
    )
    image, extent_mpc_h = sphviewer2.render(
        positions_mpc_h[:, 0],
        positions_mpc_h[:, 1],
        positions_mpc_h[:, 2],
        smoothing_lengths_mpc_h.astype(np.float32),
        masses_1e11_msun_h.astype(np.float32),
        Lbox=renderer_box_mpc_h,
        extent=_positive_float(fov_width_mpc_h, "fov_width_mpc_h"),
        xc=0.0,
        yc=0.0,
        zc=0.0,
        azimuth=azimuth_deg,
        elevation=elevation_deg,
        periodic=False,
        r_max=_positive_int(resolution_level, "resolution_level"),
        num_threads=_positive_int(threads, "threads"),
    )
    image = np.asarray(image, dtype=np.float64)
    extent_mpc_h = np.asarray(extent_mpc_h, dtype=np.float64)
    _validate_projection(image, extent_mpc_h)
    return image, extent_mpc_h


def _validate_projection(image: np.ndarray, extent_mpc_h: np.ndarray) -> None:
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("image values must be finite and non-negative")
    if not np.any(image > 0.0):
        raise ValueError("image must contain at least one positive value")
    if extent_mpc_h.shape != (4,) or not np.all(np.isfinite(extent_mpc_h)):
        raise ValueError("extent_mpc_h must contain four finite values")
    if extent_mpc_h[0] >= extent_mpc_h[1] or extent_mpc_h[2] >= extent_mpc_h[3]:
        raise ValueError("extent_mpc_h bounds must be increasing")


def crop_to_banner(
    image: np.ndarray, extent_mpc_h: np.ndarray, aspect: float
) -> tuple[np.ndarray, np.ndarray]:
    _validate_projection(image, extent_mpc_h)
    aspect = _positive_float(aspect, "aspect")
    rows, columns = image.shape
    target_rows = int(columns / aspect)
    if target_rows < 1 or target_rows > rows:
        raise ValueError("aspect does not fit within the projection")
    first_row = (rows - target_rows) // 2
    last_row = first_row + target_rows
    pixel_height_mpc_h = (extent_mpc_h[3] - extent_mpc_h[2]) / rows
    cropped_extent_mpc_h = extent_mpc_h.copy()
    cropped_extent_mpc_h[2] += first_row * pixel_height_mpc_h
    cropped_extent_mpc_h[3] = extent_mpc_h[2] + last_row * pixel_height_mpc_h
    return image[first_row:last_row, :], cropped_extent_mpc_h


@contextmanager
def _atomic_target(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_banner(
    image: np.ndarray,
    extent_mpc_h: np.ndarray,
    output: Path,
    *,
    aspect: float,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    if output.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"output must use one of {sorted(IMAGE_SUFFIXES)}")
    if vmin is not None and not np.isfinite(vmin):
        raise ValueError("vmin must be finite")
    if vmax is not None and not np.isfinite(vmax):
        raise ValueError("vmax must be finite")
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")

    banner, banner_extent_mpc_h = crop_to_banner(image, extent_mpc_h, aspect)
    log_image = np.full(banner.shape, np.nan)
    np.log10(banner, out=log_image, where=banner > 0.0)
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_WIDTH_INCHES / aspect))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(
        log_image,
        extent=banner_extent_mpc_h,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_axis_off()
    try:
        with _atomic_target(output) as temporary:
            figure.savefig(temporary, dpi=_positive_int(dpi, "dpi"), transparent=True)
    finally:
        plt.close(figure)
    log.info("Saved banner to %s", output)


def save_archive(
    path: Path,
    image: np.ndarray,
    extent_mpc_h: np.ndarray,
) -> None:
    _validate_projection(image, extent_mpc_h)
    with _atomic_target(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                image=image,
                extent_mpc_h=extent_mpc_h,
            )
    log.info("Saved projection archive to %s", path)


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a projection archive without allowing object deserialization."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"image", "extent_mpc_h"}:
                raise ValueError("archive has unexpected fields")
            image = archive["image"].astype(np.float64)
            extent_mpc_h = archive["extent_mpc_h"].astype(np.float64)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid projection archive") from error

    _validate_projection(image, extent_mpc_h)
    return image, extent_mpc_h


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aspect", type=_float_arg, default=DEFAULT_ASPECT)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--dpi", type=_int_arg, default=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render", help="project a cylindrical snapshot")
    render.add_argument("snapshot", type=Path)
    render.add_argument("-o", "--output", type=Path, required=True)
    render.add_argument(
        "--archive", type=Path, default=Path("output/cylinder-banner.npz")
    )
    render.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION_DEG)
    render.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH_DEG)
    render.add_argument(
        "--resolution-level", type=_int_arg, default=DEFAULT_RESOLUTION_LEVEL
    )
    render.add_argument("--neighbours", type=_int_arg, default=DEFAULT_NEIGHBOURS)
    render.add_argument("--threads", type=_int_arg, default=DEFAULT_THREADS)
    render.add_argument("--fov-padding", type=_float_arg, default=DEFAULT_FOV_PADDING)
    _add_presentation_arguments(render)

    plot = commands.add_parser("plot", help="replot an existing projection")
    plot.add_argument("archive", type=Path)
    plot.add_argument("-o", "--output", type=Path, required=True)
    _add_presentation_arguments(plot)
    return parser


def _render(args: argparse.Namespace) -> None:
    if args.output.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"output must use one of {sorted(IMAGE_SUFFIXES)}")
    if args.archive.suffix.lower() != ".npz" or args.archive == args.output:
        raise ValueError("archive must be a separate .npz file")
    snapshot = args.snapshot.resolve(strict=True)
    positions_mpc_h, masses_1e11_msun_h, diameter_mpc_h, height_mpc_h = (
        load_snapshot(snapshot)
    )
    centre_mpc_h = 0.5 * (
        positions_mpc_h.min(axis=0) + positions_mpc_h.max(axis=0)
    )
    positions_mpc_h -= centre_mpc_h
    field_width_mpc_h = compute_fov_width(
        diameter_mpc_h, height_mpc_h, args.elevation, args.fov_padding
    )
    smoothing_lengths_mpc_h = estimate_smoothing_lengths(
        positions_mpc_h, args.neighbours
    )
    image, extent_mpc_h = render_image(
        positions_mpc_h,
        masses_1e11_msun_h,
        smoothing_lengths_mpc_h,
        field_width_mpc_h,
        elevation_deg=args.elevation,
        azimuth_deg=args.azimuth,
        resolution_level=args.resolution_level,
        threads=args.threads,
    )
    save_archive(args.archive, image, extent_mpc_h)
    save_banner(
        image,
        extent_mpc_h,
        args.output,
        aspect=args.aspect,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
    )


def _plot(args: argparse.Namespace) -> None:
    image, extent_mpc_h = load_archive(args.archive)
    save_banner(
        image,
        extent_mpc_h,
        args.output,
        aspect=args.aspect,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    (_render if args.command == "render" else _plot)(args)


if __name__ == "__main__":
    main()
