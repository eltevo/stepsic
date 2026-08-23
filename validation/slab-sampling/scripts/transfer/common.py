"""Calculate slab power-transfer curves relative to the cubic box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]

# Geometry metadata is written as float64 in Mpc/h. These tolerances accept
# round-off in that serialization while remaining far below physical scales.
_GEOMETRY_RTOL = 1.0e-12
_GEOMETRY_ATOL_MPC_H = 1.0e-9


@dataclass(frozen=True)
class AspectTransferCurve:
    """One transfer curve normalized by the matched cubic control."""

    lz_mpc_h: float
    k_h_mpc: ArrayF
    transfer: ArrayF
    aspect_mode_counts: ArrayF
    cubic_mode_counts: ArrayF
    is_cubic: bool


@dataclass(frozen=True)
class _RawCurve:
    lz_mpc_h: float
    k_h_mpc: ArrayF
    transfer: ArrayF
    mode_counts: ArrayF
    is_cubic: bool


def cubic_normalized_transfer_curves(
    data: Mapping[str, np.ndarray],
) -> list[AspectTransferCurve]:
    """Return validated transfer curves relative to the cubic geometry.

    Cubic transfer and mode counts are interpolated at slab bin centres only where the cubic curve covers the same k values.
    """
    l_cube_mpc_h = _scalar_float(data["meta_l_cube"], "meta_l_cube")
    lz_values_mpc_h = np.asarray(
        data["meta_lz_values"], dtype=np.float64
    ).reshape(-1)
    if lz_values_mpc_h.size < 2:
        raise ValueError("the slab comparison requires at least two box shapes")
    if not np.isfinite(l_cube_mpc_h) or l_cube_mpc_h <= 0.0:
        raise ValueError("cubic box length must be finite and positive")
    if not np.all(np.isfinite(lz_values_mpc_h)) or np.any(
        lz_values_mpc_h <= 0.0
    ):
        raise ValueError("geometry lengths must be finite and positive")

    for index, lz_mpc_h in enumerate(lz_values_mpc_h):
        if np.any(
            np.isclose(
                lz_mpc_h,
                lz_values_mpc_h[index + 1 :],
                rtol=_GEOMETRY_RTOL,
                atol=_GEOMETRY_ATOL_MPC_H,
            )
        ):
            raise ValueError("each z length must be unique")

    cubic_mask = np.isclose(
        lz_values_mpc_h,
        l_cube_mpc_h,
        rtol=_GEOMETRY_RTOL,
        atol=_GEOMETRY_ATOL_MPC_H,
    )
    if np.count_nonzero(cubic_mask) != 1:
        raise ValueError("the slab comparison requires exactly one cubic box")

    raw_curves = [
        _load_raw_curve(data, float(lz_mpc_h), bool(is_cubic))
        for lz_mpc_h, is_cubic in zip(lz_values_mpc_h, cubic_mask, strict=True)
    ]
    raw_curves.sort(key=lambda curve: -curve.lz_mpc_h)
    cubic = next(curve for curve in raw_curves if curve.is_cubic)

    curves = []
    for curve in raw_curves:
        if curve.is_cubic:
            curves.append(
                AspectTransferCurve(
                    lz_mpc_h=curve.lz_mpc_h,
                    k_h_mpc=curve.k_h_mpc.copy(),
                    transfer=np.ones_like(curve.transfer),
                    aspect_mode_counts=curve.mode_counts.copy(),
                    cubic_mode_counts=curve.mode_counts.copy(),
                    is_cubic=True,
                )
            )
            continue

        shared = (
            (curve.k_h_mpc >= cubic.k_h_mpc[0])
            & (curve.k_h_mpc <= cubic.k_h_mpc[-1])
        )
        if not np.any(shared):
            raise ValueError("the slab and cubic curves do not cover any of the same k values")
        k_h_mpc = curve.k_h_mpc[shared]
        cubic_transfer = np.interp(
            k_h_mpc, cubic.k_h_mpc, cubic.transfer
        )
        if not np.all(np.abs(cubic_transfer) > 0.0):
            raise ValueError("cubic power transfer must be nonzero over the shared k range")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized_transfer = curve.transfer[shared] / cubic_transfer
        if not np.all(np.isfinite(normalized_transfer)):
            raise ValueError("power transfer relative to the cubic box must be finite")
        curves.append(
            AspectTransferCurve(
                lz_mpc_h=curve.lz_mpc_h,
                k_h_mpc=k_h_mpc.copy(),
                transfer=normalized_transfer,
                aspect_mode_counts=curve.mode_counts[shared].copy(),
                cubic_mode_counts=np.interp(
                    k_h_mpc, cubic.k_h_mpc, cubic.mode_counts
                ),
                is_cubic=False,
            )
        )
    return curves


def _load_raw_curve(
    data: Mapping[str, np.ndarray],
    lz_mpc_h: float,
    is_cubic: bool,
) -> _RawCurve:
    tag = f"lz{lz_mpc_h:g}"
    k_h_mpc = np.asarray(data[f"lpt_{tag}_k"], dtype=np.float64)
    measured = np.asarray(data[f"lpt_{tag}_pk"], dtype=np.float64)
    reference = np.asarray(data[f"ref_{tag}_pk"], dtype=np.float64)
    mode_counts = np.asarray(data[f"lpt_{tag}_nmodes"], dtype=np.float64)
    if (
        k_h_mpc.ndim != 1
        or measured.shape != k_h_mpc.shape
        or reference.shape != k_h_mpc.shape
        or mode_counts.shape != k_h_mpc.shape
        or k_h_mpc.size == 0
    ):
        raise ValueError("k, measured power, reference power, and mode counts must be non-empty arrays of the same shape")
    if not np.all(np.isfinite(k_h_mpc)) or not np.all(np.isfinite(measured)):
        raise ValueError("power and k values must be finite")
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("reference power must be finite and positive")
    if not np.all(np.isfinite(mode_counts)) or np.any(mode_counts <= 0.0):
        raise ValueError("mode counts must be finite and positive")
    if np.any(np.diff(k_h_mpc) <= 0.0):
        raise ValueError("k values must be strictly increasing")
    with np.errstate(over="ignore", invalid="ignore"):
        transfer = measured / reference
    if not np.all(np.isfinite(transfer)):
        raise ValueError("measured power divided by reference power must be finite")
    return _RawCurve(
        lz_mpc_h=lz_mpc_h,
        k_h_mpc=k_h_mpc,
        transfer=transfer,
        mode_counts=mode_counts,
        is_cubic=is_cubic,
    )


def _scalar_float(value: np.ndarray, name: str) -> float:
    values = np.asarray(value, dtype=np.float64)
    if values.size != 1:
        raise ValueError(f"{name} must contain one value")
    return float(values.reshape(-1)[0])
