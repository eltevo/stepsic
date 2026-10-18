#!/usr/bin/env python3
"""Compare fixed-mesh glass displacements with a direct low-mode 2LPT sum."""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

from stepsic.field import cubic_voxels, generate_delta_k, white_noise
from stepsic.lpt import lpt2
from validation._common.fields import VALIDATION_COSMOLOGY, init_cosmology
from validation._common.evaluation import atomic_savez


BOX = np.array([1000.0, 1000.0, 1000.0])
EDGES = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0])
REDSHIFT = 31.0


def direct_lpt(delta_k, anchor_cell_mpc_h, q_mpc_h, cutoff_h_mpc,
               g2, velocity_1, velocity_2):
    """Evaluate retained Fourier modes and their explicit 2LPT convolution.

    Mode n has the wave vector 2 pi n / BOX. All modes are held on the integer
    lattice |n_i| <= 2 reach, which contains every sum of two retained modes.
    """
    n = delta_k.shape[0]
    origin = -BOX / 2 + anchor_cell_mpc_h / 2
    reach = int(np.ceil(cutoff_h_mpc * np.max(BOX) / (2 * np.pi)))
    side = 4 * reach + 1
    lattice = np.arange(side) - 2 * reach
    numbers = np.indices((side,) * 3).reshape(3, -1).T - 2 * reach
    grid = 2 * np.pi * numbers / BOX
    grid2 = np.einsum('ij,ij->i', grid, grid)
    retained = np.flatnonzero((grid2 > 0) & (np.sqrt(grid2) <= cutoff_h_mpc))
    if not len(retained):
        raise ValueError('cutoff retains no Fourier modes')
    vectors, k2 = grid[retained], grid2[retained]

    # delta_k stores n_z >= 0; a mode with n_z < 0 is the conjugate of its mirror.
    mirror = numbers[retained, 2] < 0
    nx, ny, nz = np.where(mirror[:, None], -numbers[retained], numbers[retained]).T
    density = delta_k[nx % n, ny % n, nz] / n**3
    phi = -np.where(mirror, density.conj(), density) * np.exp(-1j * vectors @ origin) / k2

    # Mode a adds phi_a phi_b [k_a^2 k_b^2 - (k_a.k_b)^2] / 2 to the source at a + b for
    # every mode b; the flat lattice index of a + b is index(a) + index(b) - index(0).
    centre = side**3 // 2
    source = np.zeros(side**3, dtype=complex)
    reached = np.zeros(side**3, dtype=bool)
    for a, index_a in enumerate(retained):
        index = index_a + retained - centre
        source[index] += 0.5 * phi[a] * phi * (k2[a] * k2 - (vectors @ vectors[a])**2)
        reached[index] = True
    reached[centre] = False
    summed = np.flatnonzero(reached)

    # First- and second-order displacement modes, indexed [order, component, lattice point].
    psi = np.zeros((2, 3, side**3), dtype=complex)
    psi[0][:, retained] = -1j * phi * vectors.T
    psi[1][:, summed] = -1j * source[summed] / grid2[summed] * grid[summed].T

    # exp(i k.q) is a product over the axes, so the sum over the lattice is
    # three successive contractions with the phases of each axis.
    phase = np.exp(1j * (2 * np.pi * q_mpc_h / BOX)[:, :, None] * lattice)
    cube = psi.reshape(2, 3, side, side, side)
    values = np.empty((2, len(q_mpc_h), 3))
    for start in range(0, len(q_mpc_h), 128):
        block = slice(start, start + 128)
        partial = np.tensordot(cube, phase[block, 2], axes=(4, 1))
        partial = np.einsum('ocxyp,py->ocxp', partial, phase[block, 1])
        values[:, block] = np.einsum('ocxp,px->opc', partial, phase[block, 0]).real
    first, second = values

    predicted = cic_prediction(grid, (psi[0] + g2 * psi[1]).T, anchor_cell_mpc_h)
    return (first + g2 * second, velocity_1 * first + g2 * velocity_2 * second, g2 * second,
            len(retained), len(summed), predicted)


def cic_prediction(vectors, coefficients, cell_mpc_h):
    """Predict the relative RMS residual and the transfer of CIC interpolation.

    A mesh that holds the exact field at its cell centers returns the mode
    exp(ikx) multiplied by prod_i t(k_i, u_i) at fractional cell offsets u,
    with t = (1 - u) exp(-iku cell) + u exp(ik(1 - u) cell). Over uniformly
    distributed offsets the mean of t is sinc^2(k cell / 2) and the mean of
    |t|^2 is (2 + cos k cell) / 3, and different modes do not correlate.
    """
    angle = vectors * cell_mpc_h
    weight = np.sum(np.abs(coefficients) ** 2, axis=1)
    weight /= weight.sum()
    mean = np.prod(np.sinc(angle / (2 * np.pi)) ** 2, axis=1)
    square = np.prod((2 + np.cos(angle)) / 3, axis=1)
    return np.sqrt(weight @ (square - 2 * mean + 1)), weight @ mean


def tail_fraction(delta_k, support_h_mpc):
    """Return the half-complex density L2 fraction beyond the retained support."""
    n = delta_k.shape[0]
    freq = np.rint(np.fft.fftfreq(n) * n)
    xy2 = np.add.outer(freq**2, freq**2)
    total = tail = 0.0
    for index, nz in enumerate(np.rint(np.fft.rfftfreq(n) * n)):
        weight = 1.0 if index == 0 or (n % 2 == 0 and index == n // 2) else 2.0
        power = np.abs(delta_k[:, :, index]) ** 2
        modulus = 2 * np.pi * np.sqrt(xy2 + nz**2) / BOX[0]
        total += weight * np.sum(power)
        tail += weight * np.sum(power[modulus > support_h_mpc])
    return np.sqrt(tail / total)


def omitted_lpt_bounds(delta_k, support_h_mpc):
    """Bound the omitted finite-sum displacement-vector norm at every point.

    The symmetrized quadratic kernel has magnitude at most one half, so
    retained--omitted pairs contribute at most K*T and omitted pairs T**2/2.
    """
    n = delta_k.shape[0]
    freq = np.rint(np.fft.fftfreq(n) * n)
    xy2 = np.add.outer(freq**2, freq**2)
    retained = omitted = first = 0.0
    for index, nz in enumerate(np.rint(np.fft.rfftfreq(n) * n)):
        weight = 1.0 if index == 0 or (n % 2 == 0 and index == n // 2) else 2.0
        modulus = 2 * np.pi * np.sqrt(xy2 + nz**2) / BOX[0]
        amplitude = np.abs(delta_k[:, :, index]) / n**3
        use = (modulus > 0) & (modulus <= support_h_mpc)
        tail = modulus > support_h_mpc
        retained += weight * np.sum(amplitude[use])
        omitted += weight * np.sum(amplitude[tail])
        first += weight * np.sum(amplitude[tail] / modulus[tail])
    fundamental = 2 * np.pi / np.max(BOX)
    second = (retained * omitted + omitted**2 / 2) / fundamental
    return first, second, retained, omitted


def radial_rms(values, radii):
    answer = np.empty(len(EDGES) - 1)
    for index, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        use = (radii >= lo) & (radii < hi)
        answer[index] = np.sqrt(np.mean(np.sum(values[use] ** 2, axis=1)))
    return answer


def metrics(measured, reference, radii):
    transfer = np.full(len(EDGES) - 1, np.nan)
    correlation = transfer.copy()
    residual = transfer.copy()
    for index, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        use = (radii >= lo) & (radii < hi)
        if not np.any(use):
            raise ValueError(f'empty radial bin [{lo}, {hi}) Mpc/h')
        product = np.sum(measured[use] * reference[use])
        reference_power = np.sum(reference[use] ** 2)
        measured_power = np.sum(measured[use] ** 2)
        if reference_power <= 0 or measured_power <= 0:
            raise ValueError(f'zero vector power in radial bin [{lo}, {hi}) Mpc/h')
        transfer[index] = product / reference_power
        correlation[index] = product / np.sqrt(measured_power * reference_power)
        residual[index] = np.sqrt(np.sum((measured[use] - reference[use]) ** 2) / reference_power)
    return transfer, correlation, residual


def select_glass(path, geometry, source_radius, samples_per_bin):
    """Sample a scaled glass, including independent axial copies for a cylinder."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    scale = EDGES[-1] / source_radius
    with h5py.File(path) as handle:
        q = handle['PartType1/Coordinates'][:].astype(float)
        ids = handle['PartType1/ParticleIDs'][:]
        period = float(handle['Header'].attrs.get('BoxSize', 0)) * scale
    q *= scale
    repeats = 1
    if geometry == 'cylindrical':
        if not np.isfinite(period) or period <= 0:
            raise ValueError('cylindrical glass needs a positive axial BoxSize')
        repeats = int(round(BOX[2] / period))
        if repeats < 1 or not np.isclose(repeats * period, BOX[2], rtol=0, atol=1e-9):
            raise ValueError('scaled axial glass period must divide the Fourier box')
        q[:, 2] %= period
    radii = np.linalg.norm(q[:, :2] if geometry == 'cylindrical' else q, axis=1)
    if not np.all(np.isfinite(q)) or np.any(radii > EDGES[-1] * (1 + 1e-8)):
        raise ValueError('glass coordinates exceed the declared source radius')
    rng = np.random.default_rng(0)
    positions, selected, replicas, counts = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        indices = np.flatnonzero((radii >= lo) & (radii < hi))
        population = len(indices) * repeats
        if population < samples_per_bin:
            raise ValueError(f'not enough glass particles in radial bin [{lo}, {hi})')
        chosen = np.sort(rng.choice(population, size=samples_per_bin, replace=False))
        replica, offset = np.divmod(chosen, len(indices))
        index = indices[offset]
        points = q[index].copy()
        if geometry == 'cylindrical':
            points[:, 2] += replica * period - BOX[2] / 2
        positions.append(points)
        selected.append(ids[index])
        replicas.append(replica)
        counts.append(population)
    return dict(q_mpc_h=np.concatenate(positions), selected_ids=np.concatenate(selected),
        selected_replicas=np.concatenate(replicas), radial_population=np.array(counts),
        source_glass_path=np.array(str(path.resolve())), source_glass_sha256=np.array(digest.hexdigest()),
        source_glass_particles=np.int64(len(ids)), source_glass_radius=np.float64(source_radius),
        meta_glass_scale=np.float64(scale), meta_axial_repeats=np.int64(repeats),
        meta_geometry=np.array(geometry), meta_samples_per_bin=np.int64(samples_per_bin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=('spherical', 'cylindrical'), required=True)
    parser.add_argument('--glass', type=Path, required=True)
    parser.add_argument('--glass-radius', type=float, required=True,
                        help='radius in the source coordinate units, before isotropic scaling')
    parser.add_argument('--nmesh', type=int, choices=(128, 256), required=True)
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--samples-per-bin', type=int, default=256)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_bin < 1 or not np.isfinite(args.glass_radius) or args.glass_radius <= 0:
        parser.error('sample count and source radius must be positive')
    if not np.isfinite(args.cutoff) or args.cutoff <= 0:
        parser.error('cutoff must be finite and positive')
    args.work.mkdir(parents=True, exist_ok=True)
    selected = select_glass(args.glass, args.geometry, args.glass_radius, args.samples_per_bin)
    q = selected['q_mpc_h']
    radii = np.linalg.norm(q[:, :2] if args.geometry == 'cylindrical' else q, axis=1)
    print(f"Selected {len(q)} particles from {int(selected['source_glass_particles'])} "
          f"prepared glass particles; axial repeats={int(selected['meta_axial_repeats'])}", flush=True)
    growth = init_cosmology([REDSHIFT], BOX[0], kmax=100., npoints=2048)[REDSHIFT]
    power = np.maximum(growth.pk_input * np.exp(-np.minimum((growth.kh_camb / args.cutoff)**16, 700)), 1e-250)
    nvox, cell = cubic_voxels(args.nmesh, BOX)
    reference_cutoff = 1.5 * args.cutoff
    if 2 * reference_cutoff >= np.pi / cell:
        raise ValueError('quadratic reference support reaches the mesh Nyquist frequency')
    field = white_noise(nvox=nvox, seed=42, dtype=np.float64)
    delta_k = generate_delta_k(growth.kh_camb, power, nvox, cell, field=field,
                              paired=False, fixed=False, dtype=np.float64)
    del field
    delta_path = args.work / 'delta-k.npz'
    atomic_savez(delta_path, delta_k=delta_k, box_mpc_h=BOX,
                 kh_h_mpc=growth.kh_camb, power_mpc_h3=power)
    x, velocity = lpt2(q, delta_k=delta_k, nvox=nvox, dk=cell, g1=growth.g1,
        g2=growth.g2, aHf1=growth.aHf1, aHf2=growth.aHf2,
        method='cic', compensate=False, dtype=np.float64)
    # a H f [km/s/Mpc] times a displacement in Mpc/h, in km/s over sqrt(a).
    velocity_scale = 1 / (VALIDATION_COSMOLOGY['H0'] / 100 * np.sqrt(1 / (1 + REDSHIFT)))
    velocity *= velocity_scale
    reference_x, reference_v, second_order, modes_1, modes_2, predicted = direct_lpt(
        delta_k, cell, q, reference_cutoff, growth.g2,
        growth.aHf1 * velocity_scale, growth.aHf2 * velocity_scale)
    dx = x - q
    mx = metrics(dx, reference_x, radii)
    mv = metrics(velocity, reference_v, radii)
    omitted_1, omitted_2, retained_l1, omitted_l1 = omitted_lpt_bounds(delta_k, reference_cutoff)
    displacement_bound = omitted_1 + abs(growth.g2) * omitted_2
    velocity_bound = velocity_scale * (abs(growth.aHf1) * omitted_1 + abs(growth.g2 * growth.aHf2) * omitted_2)
    atomic_savez(args.output, **selected, radial_edges_mpc_h=EDGES, box_mpc_h=BOX,
        cutoff_h_mpc=np.float64(args.cutoff),
        displacement_transfer=mx[0], displacement_correlation=mx[1], displacement_relative_rms=mx[2],
        velocity_transfer=mv[0], velocity_correlation=mv[1], velocity_relative_rms=mv[2],
        measured_displacement=dx, reference_displacement=reference_x,
        predicted_displacement_relative_rms=np.float64(predicted[0]),
        predicted_displacement_transfer=np.float64(predicted[1]),
        second_order_relative_rms=np.sqrt(np.sum(second_order**2) / np.sum(reference_x**2)),
        measured_velocity=velocity, reference_velocity=reference_v,
        meta_nmesh=np.int64(args.nmesh), meta_samples=np.int64(1),
        meta_mode_count_1=np.int64(modes_1), meta_mode_count_2=np.int64(modes_2),
        meta_seed=np.int64(42), meta_redshift=np.float64(REDSHIFT),
        meta_reference_support_h_mpc=np.float64(reference_cutoff),
        meta_density_tail_relative_l2=np.float64(tail_fraction(delta_k, reference_cutoff)),
        meta_retained_density_l1=np.float64(retained_l1), meta_omitted_density_l1=np.float64(omitted_l1),
        oracle_displacement_bound_mpc_h=np.float64(displacement_bound),
        oracle_velocity_bound_km_s=np.float64(velocity_bound),
        oracle_displacement_relative_bound=displacement_bound / radial_rms(reference_x, radii),
        oracle_velocity_relative_bound=velocity_bound / radial_rms(reference_v, radii),
        meta_taper_power_exponent=np.int64(16), meta_reference_support_multiplier=np.float64(1.5),
        meta_lpt_order=np.int64(2), meta_interpolation=np.array('cic'),
        meta_selection_seed=np.int64(0), meta_selection=np.array('uniform_without_replacement_per_radial_bin'),
        source_delta_path=np.array(str(delta_path.resolve())),
        source_delta_sha256=np.array(hashlib.sha256(delta_k.tobytes()).hexdigest()))
    overall = np.sqrt(np.sum((dx - reference_x)**2) / np.sum(reference_x**2))
    print(f'{args.geometry} N={args.nmesh} cutoff={args.cutoff}: maximum RMS={mx[2].max():.8g}, '
          f'all particles {overall:.8g}, predicted {predicted[0]:.8g}', flush=True)


if __name__ == '__main__':
    main()
