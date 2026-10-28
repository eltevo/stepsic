from __future__ import annotations

import runpy
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import toml
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from conftest import growth_rate, linear_growth


DRIVER = Path(__file__).resolve().parents[1] / "stepsic.py"
# tiny-config.toml cosmology: flat, Omega_m = 0.3, H0 = 70 km/s/Mpc, z = 9.
OMEGA_M, H, A = 0.3, 0.7, 0.1
AMPLITUDE = 1e-3


@pytest.fixture
def stepsic_run(write_config, monkeypatch):
    """Run the stepsic driver on the tiny config with overrides and read its IC."""

    def run(**overrides) -> dict:
        config = write_config(**overrides)
        monkeypatch.setattr(sys, "argv", ["stepsic.py", str(config)])
        runpy.run_path(str(DRIVER), run_name="__main__")
        (path,) = Path(toml.load(config)["IC_DIR"]).glob("*/ic.hdf5")
        with h5py.File(path, "r") as f:
            particles = f["PartType1"]
            return {
                "header": dict(f["Header"].attrs),
                "pos": particles["Coordinates"][:],
                "vel": particles["Velocities"][:],
                "mass": particles["Masses"][:],
            }

    return run


@pytest.fixture
def power_law(tmp_path) -> str:
    """Input spectrum P(k) = AMPLITUDE k^-2 (Mpc/h)^3, as rows ln k and ln(k^3 P / 2 pi^2)."""
    k = np.geomspace(1e-3, 1e2, 64)
    path = tmp_path / "spectrum.txt"
    np.savetxt(path, np.log([k, k**3 * AMPLITUDE * k**-2 / (2.0 * np.pi**2)]))
    return str(path)


def _hubble(a: float) -> float:
    return FlatLambdaCDM(H0=70.0, Om0=OMEGA_M, Tcmb0=0.0).H(1.0 / a - 1.0).value


def _velocity_per_mpc(a: float, f: float) -> float:
    """Snapshot velocity per Mpc of displacement: a H(a) f / sqrt(a) in km/s (Gadget convention)."""
    return a * _hubble(a) * f / np.sqrt(a)


def _lattice_displacement(ic: dict, box: float, ngrid: int):
    """Displacements and velocities ordered by lattice site.

    The unperturbed load is the lattice of ngrid^3 cubic cells tiling
    [0, box)^3 with particles at the cell centres; each particle is assigned
    to the site whose cell contains it, which is its own site while |Psi| is
    below half a cell.
    """
    cell = box / ngrid
    assert len(ic["pos"]) == ngrid**3
    site = np.floor(ic["pos"] / cell).astype(int)
    assert site.min() >= 0 and site.max() < ngrid
    index = np.ravel_multi_index(site.T, (ngrid, ngrid, ngrid))
    order = np.argsort(index)
    np.testing.assert_array_equal(index[order], np.arange(ngrid**3))
    return ic["pos"][order] - (site[order] + 0.5) * cell, ic["vel"][order]


def _grid(power_law, **overrides) -> dict:
    config = dict(
        TYPE="grid", NGRID=16, NMESH=16, LBOX=[12.0, 12.0, 12.0], COI=[0.0, 0.0, 0.0],
        LPTORDER=1, INTERPOLATION="cic", COMPENSATE=False, FIXED=True,
        SPECTRUM="input", INPUT_SPECTRUM=power_law,
    )
    config.update(overrides)
    return config


def _write_spherical_glass(path: Path, pos: np.ndarray, mass: np.ndarray, radius: float) -> None:
    """Write a centred spherical glass with the header StePS gives it."""
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = np.array([0, len(pos), 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["BoxSize"] = 50.0
        header.attrs["SimulationRadius"] = radius
        particles = f.create_group("PartType1")
        particles["ParticleIDs"] = np.arange(len(pos), dtype=np.uint64)
        particles["Coordinates"] = pos
        particles["Velocities"] = np.zeros_like(pos)
        particles["Masses"] = mass


@pytest.mark.parametrize("hindependent", [True, False])
def test_grid_ic__particles_carry_the_input_spectrum(stepsic_run, power_law, hindependent) -> None:
    """A lattice IC carries the input P(k) with consistent units and velocities.

    Lengths are Mpc/h (HINDEPENDENT) or Mpc, masses 1e11 Msun/h or 1e11 Msun,
    velocities km/s over sqrt(a). With fixed amplitudes and particles on the
    mesh nodes, Parseval makes the mean squared displacement exactly
    (1/V) sum_k P(k)/k^2 over the mesh modes (Nyquist planes excluded). The
    Zel'dovich velocity is a H f Psi, the total mass is Omega_m rho_crit V,
    and the paired IC has exactly the opposite displacements.
    """
    box, ngrid = 12.0, 16
    common = _grid(power_law, HINDEPENDENT=hindependent)
    ic = stepsic_run(**common)
    psi, vel = _lattice_displacement(ic, box, ngrid)
    psi_paired, _ = _lattice_displacement(stepsic_run(**common, PAIRED=True), box, ngrid)
    mpc_h = 1.0 if hindependent else H  # output length unit in Mpc/h

    n = np.rint(np.fft.fftfreq(ngrid) * ngrid)
    n = n[n != -ngrid // 2]
    n2 = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    k2 = (2.0 * np.pi / (box * mpc_h)) ** 2 * n2[n2 > 0]
    expected_var = np.sum(AMPLITUDE / k2**2) / (box * mpc_h) ** 3
    np.testing.assert_allclose(np.mean(np.sum((psi * mpc_h) ** 2, axis=1)), expected_var, rtol=1e-10)

    v_expected = psi * mpc_h / H * _velocity_per_mpc(A, growth_rate(A, OMEGA_M))
    # F_omega approximates the exact growth rate to 2e-5 at z = 9.
    np.testing.assert_allclose(vel, v_expected, rtol=0.0, atol=1e-4 * np.abs(v_expected).max())

    rho_crit = FlatLambdaCDM(H0=100.0 * mpc_h, Om0=OMEGA_M).critical_density0
    total_mass = OMEGA_M * rho_crit.to_value(u.Msun / u.Mpc**3) * box**3 / 1e11
    # Grid masses are held in float32.
    np.testing.assert_allclose(ic["mass"].sum(), total_mass, rtol=1e-6)
    np.testing.assert_allclose(ic["header"]["BoxSize"], box, rtol=1e-12)
    np.testing.assert_allclose(psi_paired, -psi, rtol=0.0, atol=1e-12 * box)


def test_grid_ic__lattice_has_ngrid_cells_per_shortest_axis(stepsic_run) -> None:
    """NGRID is the number of grid cells along the shortest box dimension.

    The lattice must tile the periodic box with cubic cells and exactly
    NGRID particles per axis of a cube. With the Planck H0 = 67.74 and a box
    given in Mpc, the internal box is 14 * 0.6774 Mpc/h, where evaluating
    ceil(L / (L / 14)) in floating point overshoots 14.
    """
    ic = stepsic_run(
        TYPE="grid", NGRID=14, NMESH=14, LBOX=[14.0, 14.0, 14.0], H0=67.74,
        HINDEPENDENT=False, LPTORDER=0,
    )
    psi, _ = _lattice_displacement(ic, 14.0, 14)
    np.testing.assert_allclose(psi, 0.0, rtol=0.0, atol=1e-12)


def _density_mode(pos: np.ndarray, box: float, mode) -> complex:
    """Eulerian density contrast (1/N) sum_j exp(-i k.x_j) of equal-mass particles."""
    return np.mean(np.exp(-2j * np.pi / box * (pos @ np.asarray(mode, dtype=np.float64))))


def test_grid_ic__second_order_term_has_second_order_growth(stepsic_run, tmp_path) -> None:
    """The 2LPT correction has the amplitude and growth rate of D2 at the IC epoch.

    The spectrum holds power only in the three fundamental waves. For two
    orthogonal waves the (k1 + k2) density mode gains -(D2/D1^2) delta(k1)
    delta(k2) from the second-order displacement, i.e. F2 = 1/2 - D2/(2 D1^2),
    which is 5/7 for the Einstein-de Sitter value D2 = -3/7 D1^2 that holds to
    2e-5 at z = 9. D2 grows as D1^2, so the correction also moves with the
    growth rate 2f: velocity a H 2f / sqrt(a) per unit displacement.
    """
    box = 12.0
    k = np.geomspace(1e-3, 1e2, 4096)
    k_fund = 2.0 * np.pi / box
    # Waves of amplitude ~3e-3; every other mode carries 1e-60 of the power.
    log_pk = np.log(4e-3) + np.maximum(-(((k - k_fund) / (0.05 * k_fund)) ** 2), np.log(1e-60))
    spectrum = tmp_path / "waves.txt"
    np.savetxt(spectrum, [np.log(k), 3.0 * np.log(k) + log_pk - np.log(2.0 * np.pi**2)])
    first = stepsic_run(**_grid(str(spectrum), LPTORDER=1))
    second = stepsic_run(**_grid(str(spectrum), LPTORDER=2))

    for i, j in ((0, 1), (0, 2), (1, 2)):
        ki, kj = np.eye(3)[i], np.eye(3)[j]
        delta_i, delta_j = (_density_mode(first["pos"], box, m) for m in (ki, kj))
        change = _density_mode(second["pos"], box, ki + kj) - _density_mode(first["pos"], box, ki + kj)
        # Higher orders enter at eps^2 ~ 1e-5 relative.
        np.testing.assert_allclose(-change / (delta_i * delta_j), -3.0 / 7.0, rtol=1e-3)

    psi1, v1 = _lattice_displacement(first, box, 16)
    psi2, v2 = _lattice_displacement(second, box, 16)
    expected = (psi2 - psi1) / H * _velocity_per_mpc(A, 2.0 * growth_rate(A, OMEGA_M))
    np.testing.assert_allclose(v2 - v1, expected, rtol=0.0, atol=1e-3 * np.abs(expected).max())


@pytest.mark.parametrize("hindependent", [True, False])
def test_grid_ic__physical_coordinates_add_the_hubble_flow(stepsic_run, power_law, hindependent) -> None:
    """Physical ICs are the comoving ones seen by an observer at the origin.

    Proper positions are r = a x and velocities u = H(a) r + a dx/dt, where
    a dx/dt = sqrt(a) times the comoving snapshot velocity; H r is in km/s
    with r converted to Mpc.
    """
    common = _grid(power_law, HINDEPENDENT=hindependent)
    comoving = stepsic_run(**common, COMOVING=True)
    physical = stepsic_run(**common, COMOVING=False)
    np.testing.assert_allclose(physical["pos"], A * comoving["pos"], rtol=1e-12)
    r_mpc = physical["pos"] / (H if hindependent else 1.0)
    expected = np.sqrt(A) * comoving["vel"] + _hubble(A) * r_mpc
    np.testing.assert_allclose(physical["vel"], expected, rtol=1e-9)


@pytest.mark.camb
def test_camb_ic__displacement_and_velocity_scale_with_linear_growth(stepsic_run) -> None:
    """The CAMB z = 0 spectrum is backscaled with the linear growth factor.

    The same seed at two redshifts gives displacements in the ratio
    D(z1)/D(z2) particle by particle, and velocities in the ratio of
    D a H f / sqrt(a) (Heath growth integral, flat Lambda CDM).
    """
    common = dict(TYPE="grid", NGRID=16, NMESH=16, LBOX=[12.0, 12.0, 12.0], COI=[0.0, 0.0, 0.0])
    lattice = stepsic_run(**common, LPTORDER=0)["pos"]

    def displacement(z):
        ic = stepsic_run(**common, LPTORDER=1, REDSHIFT=z)
        psi = ic["pos"] - lattice
        return psi - 12.0 * np.rint(psi / 12.0), ic["vel"]

    (psi9, v9), (psi19, v19) = displacement(9.0), displacement(19.0)
    a9, a19 = 0.1, 0.05
    growth = linear_growth(a9, OMEGA_M) / linear_growth(a19, OMEGA_M)
    # Colossus tabulates D to 2e-4 at these redshifts.
    np.testing.assert_allclose(psi9, growth * psi19, rtol=0.0, atol=5e-4 * np.abs(psi9).max())
    velocity = growth * _velocity_per_mpc(a9, growth_rate(a9, OMEGA_M)) / _velocity_per_mpc(
        a19, growth_rate(a19, OMEGA_M)
    )
    np.testing.assert_allclose(v9, velocity * v19, rtol=0.0, atol=5e-4 * np.abs(v9).max())


def test_camb_ic__target_spectrum_covers_every_mesh_mode(stepsic_run, tmp_path) -> None:
    """The CAMB spectrum is tabulated over every mode of the mesh, so none is extrapolated.

    In a 120 x 120 x 12 box with NMESH = 8 cells along the short side, the
    cubic cells are 1.5 Mpc/h wide. The mesh modes then run from the box
    fundamental 2 pi / 120 to the corner of the Nyquist cube, sqrt(3) pi / 1.5.
    """
    stepsic_run(
        TYPE="grid", NGRID=8, NMESH=8, LBOX=[120.0, 120.0, 12.0], COI=[0.0, 0.0, 0.0],
        SAVE_SPECTRUM=True,
    )
    (path,) = tmp_path.glob("ics-*/*/ic_target_pk.txt")
    k = np.loadtxt(path)[:, 0]
    assert k[0] <= 2.0 * np.pi / 120.0
    assert k[-1] >= np.sqrt(3.0) * np.pi / 1.5


def test_spherical_glass_ic__keeps_the_load_and_returns_input_units(
    stepsic_run, power_law, tmp_path
) -> None:
    """A multiresolution spherical IC stays on its glass and in the glass's units.

    The StePS convention stores lengths in Mpc, masses in 1e11 Msun and
    velocities in km/s, declared through UNIT_*. The sphere must stay centred
    where the glass is, whatever the centre of interest, so the output
    positions are the glass positions plus small displacements. The glass
    masses are rescaled to Omega_m rho_crit (4/3) pi R^3, the header carries
    R, and velocities are a H f Psi / sqrt(a) in the input velocity unit.
    """
    radius = 20.0
    axis = np.arange(-19.0, 20.0, 2.0)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    glass = grid[np.linalg.norm(grid, axis=1) < radius]
    r = np.linalg.norm(glass, axis=1)
    masses = np.select([r < 8.0, r < 14.0], [1.0, 8.0], 27.0)
    path = tmp_path / "glass.hdf5"
    _write_spherical_glass(path, glass, masses, radius)

    ic = stepsic_run(
        GEOMETRY="spherical", TYPE="glass", INPUT_GLASS=str(path), NMESH=0, NMESHSAMPLES=3,
        R_3D=radius, D_4D=40.0, LBOX=[50.0, 50.0, 50.0], PERIODIC=False, COI=[3.0, -2.0, 1.0],
        HINDEPENDENT=False, LPTORDER=1, INTERPOLATION="cic", COMPENSATE=True,
        SPECTRUM="input", INPUT_SPECTRUM=power_law, UNIT_V_IN_KMPS=1.0,
    )
    psi = ic["pos"] - glass
    assert np.abs(psi).max() < 0.1
    rho_crit = FlatLambdaCDM(H0=70.0, Om0=OMEGA_M).critical_density0.to_value(u.Msun / u.Mpc**3)
    np.testing.assert_allclose(
        ic["mass"].sum(), OMEGA_M * rho_crit * 4.0 / 3.0 * np.pi * radius**3 / 1e11, rtol=1e-10
    )
    np.testing.assert_allclose(ic["header"]["SimulationRadius"], radius, rtol=1e-12)
    expected = psi * _velocity_per_mpc(A, growth_rate(A, OMEGA_M))
    np.testing.assert_allclose(ic["vel"], expected, rtol=0.0, atol=1e-4 * np.abs(expected).max())


def test_spherical_glass_ic__is_stretched_to_r3d(stepsic_run, tmp_path) -> None:
    """A spherical glass is stretched uniformly to the sphere of radius R_3D.

    R_3D is the radius of the simulated sphere (Template-config.toml), and a
    StePS glass records the radius it was relaxed at as SimulationRadius. A
    glass relaxed at radius 10 and run at R_3D = 20 must come out of the
    unperturbed mode with every position doubled about the sphere centre.
    Left at its own size, it would fill an eighth of the volume its masses
    are normalised to.
    """
    radius = 20.0
    axis = np.arange(-9.5, 10.0, 1.0)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    glass = grid[np.linalg.norm(grid, axis=1) < radius / 2]
    path = tmp_path / "glass.hdf5"
    _write_spherical_glass(path, glass, np.ones(len(glass)), radius / 2)

    ic = stepsic_run(
        GEOMETRY="spherical", TYPE="glass", INPUT_GLASS=str(path), NMESH=0, NMESHSAMPLES=3,
        R_3D=radius, D_4D=40.0, LBOX=[50.0, 50.0, 50.0], PERIODIC=False, LPTORDER=0,
    )
    np.testing.assert_array_equal(ic["pos"], 2.0 * glass)

