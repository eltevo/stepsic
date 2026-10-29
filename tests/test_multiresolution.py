from __future__ import annotations

import numpy as np
import pytest

from stepsic.field import cubic_voxels
from stepsic.multiresolution import create_nres_mass_map, density_samples, multiresolution_lpt
from stepsic.particles import create_grid


def test_multiresolution_lpt__aligns_levels_and_blends_linearly_in_mass(tiny_chunks) -> None:
    """Mesh levels describe one displacement field, mixed linearly in mass.

    Both levels draw the same white-noise modes, and the spectrum has no
    power at |k| >= 4 k_f, so the 8^3 and 16^3 meshes represent the same
    continuous field. The particles sit on even-indexed voxel centres of the
    finest mesh, which are nodes of both meshes once the origin shift puts
    each mesh's FFT origin on the finest voxel centre; CIC then reads exact node
    values, so a particle must get the same displacement from either level.
    A particle's mass selects the levels by linear interpolation on the mass
    table (constant beyond its ends).
    """
    box = 16.0
    k_fund = 2.0 * np.pi / box
    hierarchy = [(np.array([8, 8, 8]), 2.0), (np.array([16, 16, 16]), 1.0)]
    level_masses = np.array([2.0, 1.0])
    kh = np.geomspace(1e-3, 5.5, 256)
    pk = np.exp(-3.0 * (kh / k_fund) ** 2)
    params = {"SEED": 7, "DTYPE": np.float64, "PAIRED": False, "FIXED": False}
    positions = create_grid((8, 8, 8), 2.0, return_coords=False) - 0.5

    def displacement(masses, fields):
        disp, _ = multiresolution_lpt(
            positions, masses, level_masses, hierarchy, fields,
            order=1, g1=1.0, g2=0.0, aHf1=1.0, aHf2=0.0,
            method="cic", compensate=False,
        )
        return disp

    coarse = displacement(np.full(len(positions), 2.0), density_samples(kh, pk, hierarchy, params))
    fine = displacement(np.full(len(positions), 1.0), density_samples(kh, pk, hierarchy, params))
    # Power above 4 k_f is exp(-48) ~ 1e-21, i.e. amplitude ~ 3e-11.
    np.testing.assert_allclose(coarse, fine, rtol=0.0, atol=1e-9 * np.abs(fine).max())

    # Tripling the fine level makes each particle's level weights visible.
    delta_coarse, delta_fine = density_samples(kh, pk, hierarchy, params)
    masses = np.resize([2.0, 1.0, 1.5, 1.25, 5.0, 0.3], len(positions))
    mixed = displacement(masses, iter([delta_coarse, 3.0 * delta_fine]))
    fine_weight = np.interp(masses, [1.0, 2.0], [1.0, 0.0])
    expected = fine * ((1.0 - fine_weight) + 3.0 * fine_weight)[:, None]
    np.testing.assert_allclose(mixed, expected, rtol=0.0, atol=1e-9 * np.abs(expected).max())


def test_mesh_hierarchy__cells_hold_one_particle_of_their_mass(capsys) -> None:
    """Each mass level gets a mesh whose cell is that mass's mean volume.

    For masses m = 1, 8, 64 filling a 32^3 box of total mass 32^3, the mean
    interparticle spacing is m^(1/3) = 1, 2, 4, so the meshes are 8^3, 16^3
    and 32^3 from coarse to fine, and rho_mean * dk^3 = m on every level.
    """
    box = np.array([32.0, 32.0, 32.0])
    masses = np.array([1.0, 8.0, 64.0])
    total_mass = np.prod(box)
    nres, level_masses = create_nres_mass_map(3, masses, total_mass, box)
    capsys.readouterr()
    cells = [cubic_voxels(n, box)[1] for n in nres]
    np.testing.assert_array_equal(level_masses, [64.0, 8.0, 1.0])
    np.testing.assert_array_equal(np.array(cells) ** 3 * total_mass / np.prod(box), level_masses)


@pytest.mark.parametrize(
    ("box", "masses", "total_mass"),
    [
        (np.array([500.0, 500.0, 500.0]), np.geomspace(1.0, 4096.0, 40), 1.25e8),
        (np.array([1500.0, 1500.0, 200.0]), np.geomspace(1.0, 4096.0, 40), 1.5e7),
    ],
    ids=["cube", "cylinder"],
)
def test_mesh_hierarchy__levels_span_the_same_physical_box(box, masses, total_mass, capsys) -> None:
    """All mesh levels cover one physical volume.

    White-noise modes are shared between levels by integer index, so an
    index is the same physical wave on every level only if every mesh has
    the same extent nvox * dk; the mass-mixed displacement is otherwise a
    blend of different realisations. Each mesh must also cover the box.
    The cylinder is the 750 Mpc/h radius, 200 Mpc/h height StePS setup.
    """
    nres, _ = create_nres_mass_map(7, masses, total_mass, box)
    capsys.readouterr()
    extents = np.array([nvox * cell for nvox, cell in (cubic_voxels(n, box) for n in nres)])
    np.testing.assert_allclose(extents, np.broadcast_to(extents[0], extents.shape), rtol=1e-12, atol=0.0)
    assert np.all(extents >= box * (1.0 - 1e-12))
