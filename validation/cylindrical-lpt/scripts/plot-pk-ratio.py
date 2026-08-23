#!/usr/bin/env python3
'''Plot measured-to-reference power ratios for cylindrical StePS snapshots.'''

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# If running from the stepsic repo, use its matplotlib setup
from validation._common.cosmology_fields import VALIDATION_COSMOLOGY
from validation._common.plotting import atomic_savefig, setup_matplotlib


# A&A single-column width [inches]
AA_COL_WIDTH = 3.5


def _derive_output_paths(
    base: str,
) -> tuple[str, str, str]:
    '''
    Derive the three output paths from the user-supplied base path.
 
    ``base = 'dir/validation.pdf'`` yields:
      - ``dir/validation-measured.pdf``
      - ``dir/validation-reference.pdf``
      - ``dir/validation.pdf``  (ratio)
    '''
    p = Path(base)
    stem = p.stem
    suffix = p.suffix or ".pdf"
    parent = p.parent
    return (
        str(parent / f"{stem}-measured{suffix}"),
        str(parent / f"{stem}-reference{suffix}"),
        str(parent / f"{stem}{suffix}"),
    )


def load_pk_measured(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''Load the k, P(k) columns from a StePS_Pk ASCII output file.'''
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    k = data[:, 0]
    pk = data[:, 1]
    # Remove NaN and negative entries
    mask = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    return k[mask], pk[mask]


def get_reference_pk_camb(z: float, kmax: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generate the z=0 non-linear (or linear) reference P(k) using CAMB.
    
    For z=0 comparison of an evolved simulation, we want the non-linear
    P(k) from Halofit/HMCode as the reference. For high-z IC validation,
    the linear P(k) is the correct reference.
    
    Returns k [1/Mpc], P(k) [Mpc^3].
    '''
    import camb

    # Shared Planck 2018 EE+BAO parameters used by all validation campaigns.
    cosmo = VALIDATION_COSMOLOGY
    h = cosmo["H0"] / 100.0
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=cosmo["H0"],
        ombh2=cosmo["OMEGA_B"] * h**2,
        omch2=(cosmo["OMEGA_M"] - cosmo["OMEGA_B"]) * h**2,
        omk=0.0,
        mnu=cosmo["MNU"],
        nnu=cosmo["NNU"],
        YHe=cosmo["YHE"],
        TCMB=cosmo["TCMB"],
    )
    pars.InitPower.set_params(As=cosmo["AS"], ns=cosmo["NS"], r=0)
    pars.set_matter_power(
        redshifts=[z],
        kmax=kmax,
        nonlinear=True,
    )
    pars.NonLinearModel.set_params(halofit_version="mead2020")

    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=kmax, npoints=500,
    )
    # kh is in [h/Mpc], pk is in [(Mpc/h)^3]
    # Convert to physical units [1/Mpc] and [Mpc^3]
    k_phys = kh * h        # h/Mpc -> 1/Mpc
    pk_phys = pk[0] / h**3  # (Mpc/h)^3 -> Mpc^3

    return k_phys, pk_phys


def get_redshift_from_snapshot(path: str) -> float:
    '''Read the redshift from a StePS HDF5 snapshot header.'''
    import h5py
    with h5py.File(path, "r") as f:
        hdr = f["/Header"].attrs
        # StePS writes 'Redshift' as a header attribute
        if "Redshift" in hdr:
            return float(hdr["Redshift"])
        # Fallback: compute from scale factor
        if "Time" in hdr:
            a = float(hdr["Time"])
            if a > 0:
                return 1.0 / a - 1.0
    raise ValueError(f"Cannot determine redshift from {path}")


def detect_discontinuities(
    k: np.ndarray,
    pk: np.ndarray,
    *,
    sigma_threshold: float = 4.0,
) -> np.ndarray:
    r'''
    Detect discontinuities in a measured P(k) using second-order
    finite differences in log-log space.
 
    A jump at bin *i* shows up as a spike in
 
    .. math::
        \Delta_i = \log P_{i+1} - 2\,\log P_i + \log P_{i-1}
 
    normalised by the local spacing :math:`\Delta\!\log k`. Points whose
    absolute deviation from the median exceeds ``sigma_threshold`` x MAD
    are flagged.
 
    Parameters
    ----------
    k : ndarray
        Wavenumber array (positive, monotonically increasing).
    pk : ndarray
        Power spectrum values.
    sigma_threshold : float
        Detection threshold in units of MAD. Default 4.0 - aggressive
        enough to catch shell-boundary jumps, conservative enough to
        ignore ordinary BAO wiggles.
 
    Returns
    -------
    k_disc : ndarray
        Wavenumber values at detected discontinuities.
    '''
    if len(k) < 5:
        return np.array([])
 
    logk = np.log(k)
    logpk = np.log(pk)
 
    # Second-order finite difference (curvature in log-log)
    d2 = np.diff(logpk, n=2)
    # Normalise by local spacing to handle non-uniform k-bins
    dk = np.diff(logk)
    dk_mid = 0.5 * (dk[:-1] + dk[1:])
    d2_norm = d2 / dk_mid**2
 
    # Use median absolute deviation so a few outliers do not set the scale.
    med = np.median(d2_norm)
    mad = np.median(np.abs(d2_norm - med))
    if mad == 0:
        return np.array([])
 
    # 0.6745 converts MAD to a Gaussian-equivalent sigma
    z_score = np.abs(d2_norm - med) / (mad / 0.6745)
    disc_mask = z_score > sigma_threshold
 
    # Map back: diff(n=2) index i corresponds to original index i+1
    disc_indices = np.where(disc_mask)[0] + 1
    return k[disc_indices]


def estimate_cosmic_variance(
    k: np.ndarray,
    dk: float,
    V_eff: float,
) -> np.ndarray:
    r'''
    Estimate the fractional 1-sigma cosmic variance on P(k).

    For a Gaussian field measured in volume V_eff with bin width dk,
    the number of independent modes in a spherical shell is

    .. math::

        N_{\rm modes}
        =
        \frac{4\pi k^2 \, dk \, V_{\rm eff}}{(2\pi)^3}

    and the fractional uncertainty is
    
    .. math::
        
        \sigma_P / P
        =
        \sqrt{2 / N_{\rm modes}}

    Parameters
    ----------
    k : np.ndarray
        Wavenumber bin centers [1/Mpc].
    dk : float
        Wavenumber bin width [1/Mpc].
    V_eff : float
        Effective survey volume [Mpc^3].

    Returns
    -------
    sigma_frac : np.ndarray
        Fractional 1-sigma uncertainty on P(k).
    '''
    N_modes = 4.0 * np.pi * k**2 * dk * V_eff / (2.0 * np.pi) ** 3
    # Floor at 1 mode to avoid div-by-zero
    N_modes = np.maximum(N_modes, 1.0)
    return np.sqrt(2.0 / N_modes)


def _volume_from_snapshot(path: str) -> float | None:
    '''
    Estimate the geometric survey volume from a StePS snapshot header.

    Returns V_eff in Mpc^3, or None if the geometry cannot be determined.
    '''
    import h5py
    try:
        with h5py.File(path, "r") as f:
            hdr = f["/Header"].attrs
            topo = hdr.get("TopologicalManifold", b"").decode()
            r_sim = float(hdr.get("SimulationRadius", 0))
            l_box = float(hdr.get("BoxSize", 0))
            if "S^1" in topo or "S1" in topo:
                # Cylindrical: V = pi * R^2 * Lz
                vol = np.pi * r_sim**2 * l_box
            elif r_sim > 0:
                # Spherical: V = 4/3 pi R^3
                vol = 4.0 / 3.0 * np.pi * r_sim**3
            else:
                # Periodic box
                vol = l_box**3
            log.info(f"V_eff from snapshot: {vol:.3e} Mpc^3 (topo={topo!r})")
            return vol
    except Exception as e:
        log.warning(f"Could not determine volume from {path}: {e}")
        return None


def plot_pk_measured(
    pk_files: list[str],
    snapshot_files: list[str],
    labels: list[str],
    output: str,
    *,
    sigma_threshold: float = 4.0,
) -> None:
    '''
    Plot the measured P(k) from StePS_Pk output, with vertical dashed
    lines marking detected discontinuities.
 
    Parameters
    ----------
    pk_files : list of str
        Paths to StePS_Pk ASCII output files.
    snapshot_files : list of str
        Corresponding StePS HDF5 snapshots (used only for redshift).
    labels : list of str
        Legend labels for each curve.
    output : str
        Output figure path.
    sigma_threshold : float
        Sensitivity for discontinuity detection (see
        :func:`detect_discontinuities`).
    '''
    import matplotlib.pyplot as plt
 
    setup_matplotlib()
 
    fig, ax = plt.subplots(1, 1, figsize=(AA_COL_WIDTH, 2.8))
 
    for i, (pk_file, snap_file, label) in enumerate(
        zip(pk_files, snapshot_files, labels)
    ):
        k, pk = load_pk_measured(pk_file)
        log.info(f"Loaded {len(k)} k-bins from {pk_file}")
 
        ax.loglog(k, pk, lw=1.2, label=label)
 
        # Detect and mark discontinuities
        k_disc = detect_discontinuities(
            k, pk, sigma_threshold=sigma_threshold,
        )
        for kd in k_disc:
            ax.axvline(kd, color='tab:red', ls="--", lw=0.7, alpha=0.6)
        if len(k_disc) > 0:
            log.info(
                f"  {len(k_disc)} discontinuit{'y' if len(k_disc) == 1 else 'ies'} "
                f"detected at k = {k_disc}"
            )
 
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [Mpc$^{3}$]")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
 
    fig.tight_layout()
    atomic_savefig(fig, output, dpi=300, bbox_inches="tight")
    log.info(f"Measured P(k) figure saved to {output}")
    plt.close(fig)
 
 
def plot_pk_reference(
    snapshot_files: list[str],
    labels: list[str],
    output: str,
    *,
    kmax_global: float = 10.0,
) -> None:
    '''
    Plot the CAMB Halofit / HMCode non-linear reference P(k) for each
    redshift extracted from the corresponding snapshot headers.
 
    Parameters
    ----------
    snapshot_files : list of str
        StePS HDF5 snapshots (for reading z).
    labels : list of str
        Legend labels for each curve.
    output : str
        Output figure path.
    kmax_global : float
        Maximum wavenumber passed to CAMB [h/Mpc].
    '''
    import matplotlib.pyplot as plt
 
    setup_matplotlib()
 
    fig, ax = plt.subplots(1, 1, figsize=(AA_COL_WIDTH, 2.8))
 
    for i, (snap_file, label) in enumerate(zip(snapshot_files, labels)):
        try:
            z = max(0.0, get_redshift_from_snapshot(snap_file))
        except Exception:
            z = 0.0
            log.warning(
                f"Could not read z from {snap_file}, assuming z=0"
            )
        log.info(f"Generating CAMB reference for z = {z:.4f}")
 
        k_ref, pk_ref = get_reference_pk_camb(z, kmax=kmax_global)
 
        ax.loglog(k_ref, pk_ref, lw=1.2, label=f"{label} (HMCode)")
 
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [Mpc$^{3}$]")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
 
    fig.tight_layout()
    atomic_savefig(fig, output, dpi=300, bbox_inches="tight")
    log.info(f"Reference P(k) figure saved to {output}")
    plt.close(fig)
 


def plot_pk_ratio(
    pk_files: list[str],
    snapshot_files: list[str],
    labels: list[str],
    output: str,
    ylim: tuple[float, float] = (0.8, 1.2),
    V_eff: float | None = None,
):
    '''
    Plot P_meas / P_ref for one or more redshifts.

    Parameters
    ----------
    V_eff : float, optional
        Effective survey volume [Mpc^3] for cosmic variance band.
        Auto-detected from snapshot header if None.
    '''
    import matplotlib.pyplot as plt

    setup_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    for i, (pk_file, snap_file, label) in enumerate(
        zip(pk_files, snapshot_files, labels)
    ):
        # Load measured P(k)
        k_meas, pk_meas = load_pk_measured(pk_file)
        log.info(f"Loaded {len(k_meas)} k-bins from {pk_file}")

        # Get redshift (clamp z<0 to 0: StePS can slightly overshoot a=1)
        try:
            z = get_redshift_from_snapshot(snap_file)
            if z < 0:
                log.warning(f"Negative z={z:.6f} from {snap_file}, clamping to 0")
                z = 0.0
        except Exception:
            z = 0.0
            log.warning(f"Could not read z from {snap_file}, assuming z=0")
        log.info(f"Redshift: z = {z:.2f}")

        # Generate reference P(k)
        k_ref, pk_ref = get_reference_pk_camb(z, kmax=k_meas.max() * 1.5)

        # Interpolate reference onto measured k-bins (log-log)
        log_pk_ref_interp = np.interp(
            np.log(k_meas), np.log(k_ref), np.log(pk_ref),
            left=np.nan, right=np.nan,
        )
        pk_ref_at_k = np.exp(log_pk_ref_interp)

        # Ratio
        ratio = pk_meas / pk_ref_at_k
        valid = np.isfinite(ratio)

        ax.plot(k_meas[valid], ratio[valid], lw=1.2, label=label)

        # Cosmic-variance band (first dataset only to keep the plot clean)
        if i == 0:
            vol = V_eff if V_eff is not None else _volume_from_snapshot(snap_file)
            if vol is not None and vol > 0:
                dk = np.median(np.diff(k_meas))
                sigma = estimate_cosmic_variance(k_meas[valid], dk, vol)
                ax.fill_between(
                    k_meas[valid], 1.0 - sigma, 1.0 + sigma,
                    color="gray", alpha=0.18, lw=0,
                    label=r"$1\sigma$ cosmic var.",
                    zorder=0,
                )

    # Reference line
    ax.axhline(1.0, color="gray", ls="--", lw=0.5, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_title(r"$P_{\rm meas}(k) \,/\, P_{\rm ref}(k)$", loc="left", fontsize=10)
    #ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=False)

    fig.tight_layout()
    atomic_savefig(fig, output, dpi=300, bbox_inches="tight")
    log.info(f"Figure saved to {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot P(k) ratio for cylindrical StePS simulation."
    )
    parser.add_argument(
        "--pk", type=str, nargs="+", required=True,
        help="Path(s) to StePS_Pk output ASCII file(s).",
    )
    parser.add_argument(
        "--snapshot", type=str, nargs="+", required=True,
        help="Path(s) to corresponding StePS HDF5 snapshot(s) (for reading z).",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Labels for the legend. If not given, uses 'z=X.X'.",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, default=[0.8, 1.2],
        help="Y-axis limits for the ratio plot (default: 0.8 1.2).",
    )
    parser.add_argument(
        "--disc-sigma", type=float, default=4.0,
        help=(
            "Discontinuity detection threshold in MAD-sigma units "
            "(default: 4.0)."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="validation_cylindrical_pk.pdf",
        help="Output figure path.",
    )
    parser.add_argument(
        "--V_eff", type=float, default=None,
        help="Effective survey volume [Mpc^3] for cosmic variance band. "
             "Auto-detected from snapshot if not given.",
    )
    args = parser.parse_args()

    if len(args.pk) != len(args.snapshot):
        parser.error("Number of --pk and --snapshot files must match.")

    labels = args.labels or [rf"$z \approx {i}$" for i in range(len(args.pk))]
    if len(labels) != len(args.pk):
        parser.error("Number of --labels must match number of --pk files.")

    out_measured, out_reference, out_ratio = _derive_output_paths(args.output)

    plot_pk_measured(
        pk_files=args.pk,
        snapshot_files=args.snapshot,
        labels=labels,
        output=out_measured,
        sigma_threshold=args.disc_sigma,
    )
 
    plot_pk_reference(
        snapshot_files=args.snapshot,
        labels=labels,
        output=out_reference,
    )
 
    plot_pk_ratio(
        pk_files=args.pk,
        snapshot_files=args.snapshot,
        labels=labels,
        output=args.output,
        ylim=tuple(args.ylim),
    )


if __name__ == "__main__":
    main()
