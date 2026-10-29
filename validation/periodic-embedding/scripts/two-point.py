#!/usr/bin/env python3
"""RMS relative displacement of two points in padded periodic boxes and in open space.

The displacement trace of the linear field at redshift 31 has the covariance

    C(r) = sum_k P(k) F(k) / k^2 exp(i k.r) / V,

with the CAMB spectrum P and the Gaussian filter F(k) = exp(-k^2 / k_s^2) that
suppresses the modes near the mesh Nyquist frequency. The relative displacement
of two points separated by r has the mean square sigma_12^2(r) = 2 [C(0) - C(r)]
and the root mean square sigma_12, in Mpc/h. No random field enters.

For a box with the stepsic mesh the sum runs over the retained modes: the zero
mode and the Nyquist planes are removed. One inverse real FFT gives C at every
lattice separation. A sphere of radius R sits in a cubic box of side
alpha * 2R; a cylinder of radius R and axial period L_z sits in a box whose two
transverse sides are alpha * 2R and whose axial side is the period L_z. The
reference for the sphere is open space,

    C(r) = 1 / (2 pi^2) int P F j0(k r) dk,

and for the cylinder open transverse axes with the periodic axis,

    C(r_perp, z) = 1 / L_z sum_n cos(k_z z) int q dq / (2 pi) P F / k^2 J0(q r_perp),

with k_z = 2 pi n / L_z and k^2 = q^2 + k_z^2. The integrals run from the first
tabulated wavenumber to the point where F falls below 1e-12.

Direction cuts give sigma_12 at lattice separations up to the domain diameter.
Given the work directory of the relative-motion measurement, the script also
gives the mean of sigma_12^2 over the particle pairs of that measurement,
evaluating C at each pair's
separation vector by cubic spline interpolation of the periodic lattice (box)
or of a table (reference). For the axis cut the component of the displacement
covariance along the separation,

    C_xx(r) = sum_k P F k_x^2 / k^4 exp(i k.r) / V,

is also evaluated. Its open-space form is 1 / (2 pi^2) int P F [j0(kr) - 2 j1(kr) / (kr)] dk
for the sphere and, for the cylinder with the separation across the axis,
1 / L_z sum_n int q dq / (2 pi) P F q^2 / k^4 [J0(q r) - J2(q r)] / 2. Finally the
correlation C(2R n) / C(0) of the displacements at opposite boundary points is
given for directions n at several angles to a lattice axis. Radii and the
period are given in Mpc and converted to Mpc/h with the validation cosmology.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import irfftn
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import j0, jv, spherical_jn

from validation._common.evaluation import atomic_savez, load_npz
from validation._common.fields import VALIDATION_COSMOLOGY, init_cosmology

GEOMETRIES = ('spherical', 'cylindrical')
DIRECTIONS = {
    'spherical': {'axis': (1, 0, 0), 'face_diagonal': (1, 1, 0), 'body_diagonal': (1, 1, 1)},
    'cylindrical': {'axis': (1, 0, 0), 'diagonal': (1, 1, 0)},
}
REGIONS = ('central', 'whole')
REDSHIFT = 31.
KMIN = 1e-5
KMAX = 50.
FILTER_FLOOR = 1e-12
PAIR_PADDINGS = (1, 2, 3)
TABLE_TARGETS = (10., 50., 100., 200.)
THETAS = (0., 5., 10., 15., 22.5, 30., 45.)
BODY_DIAGONAL_DEGREES = 54.7356103172
OPPOSITE_FAMILIES = {'spherical': ('face', 'body'), 'cylindrical': ('transverse',)}
# Largest separation [Mpc/h] that sets the quadrature step, and radians per step.
STEP_SEPARATION = 1100.
STEP_RADIANS = .25
LOG_JOIN = 1e-1
LOG_INTERVALS = 40960
TABLE_STEP = 1.
CHUNK = 128


def simpson_weights(count, step):
    weights = np.ones(count)
    weights[1:-1:2] = 4.
    weights[2:-1:2] = 2.
    return weights * step / 3.


def quadrature(kmax, refine=1):
    """Nodes and weights for integrals over [KMIN, kmax]: Simpson in ln k below
    LOG_JOIN, then uniformly spaced so that k r advances by STEP_RADIANS per step
    at r = STEP_SEPARATION."""
    log_nodes = np.linspace(np.log(KMIN), np.log(LOG_JOIN), LOG_INTERVALS * refine + 1)
    k_log = np.exp(log_nodes)
    w_log = simpson_weights(len(k_log), log_nodes[1] - log_nodes[0]) * k_log
    intervals = 2 * int(np.ceil((kmax - LOG_JOIN) * STEP_SEPARATION * refine / STEP_RADIANS / 2))
    k_lin = np.linspace(LOG_JOIN, kmax, intervals + 1)
    w_lin = simpson_weights(len(k_lin), k_lin[1] - k_lin[0])
    return np.concatenate([k_log, k_lin]), np.concatenate([w_log, w_lin])


def filtered_spectrum(spectrum, k, filter_scale):
    return spectrum(k) * np.exp(-k**2 / filter_scale**2)


def cutoff_wavenumber(filter_scale):
    return filter_scale * np.sqrt(-np.log(FILTER_FLOOR))


def lattice_covariance(nvox, cell, spectrum, filter_scale, longitudinal=False):
    """C on every lattice separation of a periodic box [(Mpc/h)^2]; with
    ``longitudinal`` the xx component C_xx instead of the trace."""
    nvox = tuple(int(n) for n in nvox)
    kx, ky = (2. * np.pi * np.fft.fftfreq(n, cell) for n in nvox[:2])
    kz = 2. * np.pi * np.fft.rfftfreq(nvox[2], cell)
    ky2, kz2 = ky[:, None]**2, kz[None, :]**2
    retained = ((np.arange(nvox[1]) != nvox[1] // 2)[:, None]
                & (np.arange(len(kz)) != nvox[2] // 2)[None, :])
    weights = np.zeros((nvox[0], nvox[1], len(kz)), dtype=np.complex128)
    # The weight is even in kx: fill both signs from one evaluation.
    for i in range(nvox[0] // 2):
        k2 = kx[i]**2 + ky2 + kz2
        if i == 0:
            k2[0, 0] = 1.
        slab = filtered_spectrum(spectrum, np.sqrt(k2), filter_scale) / k2
        if longitudinal:
            slab *= kx[i]**2 / k2
        slab[~retained] = 0.
        if i == 0:
            slab[0, 0] = 0.
        weights[i].real = slab
        if i:
            weights[nvox[0] - i].real = slab
    covariance = irfftn(weights, s=nvox, workers=-1, overwrite_x=True)
    return covariance / cell**3


def longitudinal_kernel(x):
    """j0(x) - 2 j1(x) / x, with the limit 1/3 at x = 0."""
    small = x < 1e-6
    safe = np.where(small, 1., x)
    return np.sinc(x / np.pi) - 2. * np.where(small, 1. / 3., spherical_jn(1, safe) / safe)


def sphere_reference(r, spectrum, filter_scale, refine=1, longitudinal=False):
    k, w = quadrature(cutoff_wavenumber(filter_scale), refine)
    f = w * filtered_spectrum(spectrum, k, filter_scale) / (2. * np.pi**2)
    r = np.atleast_1d(r)
    kernel = longitudinal_kernel if longitudinal else lambda x: np.sinc(x / np.pi)
    return np.concatenate([kernel(r[i:i + CHUNK, None] * k) @ f
                           for i in range(0, len(r), CHUNK)])


def cylinder_reference(r_perp, z, spectrum, filter_scale, period, refine=1, longitudinal=False):
    """C on the grid z (rows) by r_perp (columns); with ``longitudinal`` the
    component along the separation, which lies across the axis."""
    kmax = cutoff_wavenumber(filter_scale)
    q, w = quadrature(kmax, refine)
    r_perp, z = np.atleast_1d(r_perp), np.atleast_1d(z)
    result = np.zeros((len(z), len(r_perp)))
    for n in range(int(kmax * period / (2. * np.pi)) + 1):
        kz = 2. * np.pi * n / period
        k2 = q**2 + kz**2
        g = w * q * filtered_spectrum(spectrum, np.sqrt(k2), filter_scale) / k2 / (2. * np.pi * period)
        if longitudinal:
            g = g * q**2 / k2
            kernel = lambda x: (j0(x) - jv(2, x)) / 2.
        else:
            kernel = j0
        radial = np.concatenate([kernel(r_perp[i:i + CHUNK, None] * q) @ g
                                 for i in range(0, len(r_perp), CHUNK)])
        result += (1 if n == 0 else 2) * np.cos(kz * z)[:, None] * radial[None, :]
    return result


def direction_cut(covariance, cell, direction, maximum):
    """Separations [Mpc/h] and C along a lattice direction up to ``maximum``."""
    step = np.linalg.norm(direction) * cell
    n = np.arange(1, int(np.floor(maximum / step + 1e-6)) + 1)
    index = tuple(n * d % size for d, size in zip(direction, covariance.shape))
    return n * step, covariance[index]


def pair_structure(covariance_at, c0, positions, pairs, counts):
    """Mean of sigma_12^2 = 2 [C(0) - C(q_i - q_j)] over the pairs of each bin."""
    result = np.full(len(counts), np.nan)
    for b, n in enumerate(counts):
        if n:
            selected = pairs[b, :n]
            delta = positions[selected[:, 0]] - positions[selected[:, 1]]
            result[b] = np.mean(2. * (c0 - covariance_at(delta)))
    return result


def lattice_at(covariance, cell):
    coefficients = spline_filter(covariance, order=3, mode='grid-wrap')
    return lambda delta: map_coordinates(coefficients, (delta / cell).T, order=3,
                                         mode='grid-wrap', prefilter=False)


def opposite_directions(family, degrees):
    t = np.radians(np.asarray(degrees))
    zero = np.zeros_like(t)
    if family == 'body':
        return np.stack([np.cos(t), np.sin(t) / np.sqrt(2.), np.sin(t) / np.sqrt(2.)], axis=1)
    return np.stack([np.cos(t), np.sin(t), zero], axis=1)


def cylinder_table_at(table, period):
    return lambda delta: table.ev(np.hypot(delta[:, 0], delta[:, 1]),
                                  np.abs(delta[:, 2] - period * np.rint(delta[:, 2] / period)))


def even_mesh(cells, name):
    if abs(cells - round(cells)) > 1e-6 or round(cells) % 2:
        raise ValueError(f'{name} = {cells} is not an even integer')
    return int(round(cells))


def spherical_mesh(alpha, spec):
    n = even_mesh(alpha * spec['cells'], 'padding times spherical cells')
    return np.array([n, n, n])


def cylindrical_mesh(alpha, spec):
    even_mesh(spec['cells'], 'cylindrical cells')
    n = even_mesh(alpha * spec['diameter'] / spec['cell'], 'transverse cells')
    return np.array([n, n, spec['cells']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spherical-radius-mpc', type=float, required=True)
    parser.add_argument('--cylindrical-radius-mpc', type=float, required=True)
    parser.add_argument('--cylindrical-period-mpc', type=float, required=True)
    parser.add_argument('--spherical-cells', type=int, default=360,
                        help='cells across the sphere diameter')
    parser.add_argument('--cylindrical-cells', type=int, default=48,
                        help='cells along the axial period')
    parser.add_argument('--paddings', type=float, nargs='+', required=True,
                        help='box widths in domain diameters')
    parser.add_argument('--filter-scale', type=float, default=.4,
                        help='k_s of the Gaussian filter [h/Mpc]')
    parser.add_argument('--glass-work', type=Path,
                        help='work directory of relative-motion.py after its measure action')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    h = VALIDATION_COSMOLOGY['H0'] / 100
    period = args.cylindrical_period_mpc * h
    cylinder_radius = args.cylindrical_radius_mpc * h
    sphere_radius = args.spherical_radius_mpc * h
    domains = {
        'spherical': dict(radius=sphere_radius, diameter=2 * sphere_radius, cells=args.spherical_cells,
                          cell=2 * sphere_radius / args.spherical_cells, mesh=spherical_mesh),
        'cylindrical': dict(radius=cylinder_radius, diameter=2 * cylinder_radius, cells=args.cylindrical_cells,
                            cell=period / args.cylindrical_cells, mesh=cylindrical_mesh),
    }
    paddings = np.array(args.paddings)
    longest = max(paddings) * max(spec['diameter'] for spec in domains.values())
    growth = init_cosmology([REDSHIFT], longest, kmin=KMIN, kmax=KMAX, npoints=4096)[REDSHIFT]
    log_spectrum = CubicSpline(np.log(growth.kh_camb), np.log(growth.pk_input))
    spectrum = lambda k: np.exp(log_spectrum(np.log(k)))
    ks = args.filter_scale

    pair_paddings = [a for a in paddings if a in PAIR_PADDINGS]
    with_pairs = args.glass_work is not None
    if with_pairs:
        inputs = load_npz(args.glass_work / 'inputs.npz')
        pair_count = int(load_npz(args.glass_work / 'relative-motion.npz')['pair_count'])
        pair_data = load_npz(args.glass_work / f'pairs-{pair_count}.npz')

    result = dict(paddings=paddings, filter_scale_h_mpc=ks, redshift=REDSHIFT,
                  pair_paddings=np.array(pair_paddings))
    if with_pairs:
        result['separation_edges_mpc_h'] = inputs['separation_edges_mpc_h']
    for geometry, spec in domains.items():
        cell = spec['cell']
        cylinder = geometry == 'cylindrical'
        if cylinder:
            reference = lambda s: cylinder_reference(s, [0.], spectrum, ks, period)[0]
        else:
            reference = lambda s: sphere_reference(s, spectrum, ks)
        c0_reference = reference(np.zeros(1))[0]
        if cylinder:
            parallel_reference = lambda s: cylinder_reference(s, [0.], spectrum, ks, period, longitudinal=True)[0]
        else:
            parallel_reference = lambda s: sphere_reference(s, spectrum, ks, longitudinal=True)
        sqrt_s = {name: [] for name in DIRECTIONS[geometry]}
        sqrt_s_parallel = []
        correlation = {family: [] for family in OPPOSITE_FAMILIES[geometry]}
        c0 = []
        pair_s = {region: [] for region in REGIONS}
        for alpha in paddings:
            nvox = spec['mesh'](alpha, spec)
            parallel = lattice_covariance(nvox, cell, spectrum, ks, longitudinal=True)
            _, c = direction_cut(parallel, cell, DIRECTIONS[geometry]['axis'], spec['diameter'])
            sqrt_s_parallel.append(np.sqrt(2. * (parallel[0, 0, 0] - c)))
            del parallel
            covariance = lattice_covariance(nvox, cell, spectrum, ks)
            c0.append(covariance[0, 0, 0])
            for name, direction in DIRECTIONS[geometry].items():
                s, c = direction_cut(covariance, cell, direction, spec['diameter'])
                sqrt_s[name].append(np.sqrt(2. * (c0[-1] - c)))
                result[f'{geometry}_{name}_separation_mpc_h'] = s
            interpolate = lattice_at(covariance, cell)
            for family in OPPOSITE_FAMILIES[geometry]:
                degrees = THETAS + ((BODY_DIAGONAL_DEGREES,) if family == 'body' else ())
                value = interpolate(spec['diameter'] * opposite_directions(family, degrees))
                value[0] = covariance[int(round(spec['diameter'] / cell)) % nvox[0], 0, 0]
                correlation[family].append(value / c0[-1])
                result[f'{geometry}_opposite_{family}_degrees'] = np.array(degrees)
            if with_pairs and alpha in pair_paddings:
                for region in REGIONS:
                    key = f'{geometry}_{region}'
                    pair_s[region].append(pair_structure(
                        interpolate, c0[-1], inputs[f'{geometry}_q_mpc_h'],
                        pair_data[f'{key}_pairs'], pair_data[f'{key}_counts']))
            print(f'{geometry} alpha={alpha:g} mesh={nvox.tolist()} C(0)={c0[-1]:.6f}', flush=True)
            del covariance, interpolate
        result[f'{geometry}_c0'] = np.array(c0)
        result[f'{geometry}_c0_reference'] = c0_reference
        for name, direction in DIRECTIONS[geometry].items():
            s = result[f'{geometry}_{name}_separation_mpc_h']
            result[f'{geometry}_{name}_sqrt_s'] = np.array(sqrt_s[name])
            result[f'{geometry}_{name}_sqrt_s_reference'] = np.sqrt(2. * (c0_reference - reference(s)))
        s = result[f'{geometry}_axis_separation_mpc_h']
        result[f'{geometry}_axis_sqrt_s_parallel'] = np.array(sqrt_s_parallel)
        result[f'{geometry}_axis_sqrt_s_parallel_reference'] = np.sqrt(
            2. * (parallel_reference(np.zeros(1))[0] - parallel_reference(s)))
        for family, values in correlation.items():
            result[f'{geometry}_opposite_{family}_correlation'] = np.array(values)
        result[f'{geometry}_opposite_correlation_reference'] = (
            reference(np.array([spec['diameter']]))[0] / c0_reference)
        if with_pairs:
            if cylinder:
                r_perp = np.arange(0., spec['diameter'] + 2 * TABLE_STEP, TABLE_STEP)
                z = np.arange(0., period / 2 + TABLE_STEP, TABLE_STEP)
                table = RectBivariateSpline(r_perp, z, cylinder_reference(r_perp, z, spectrum, ks, period).T)
                reference_at = cylinder_table_at(table, period)
            else:
                r = np.arange(0., spec['diameter'] + 2 * TABLE_STEP, TABLE_STEP)
                line = CubicSpline(r, sphere_reference(r, spectrum, ks))
                reference_at = lambda delta: line(np.linalg.norm(delta, axis=1))
            for region in REGIONS:
                key = f'{geometry}_{region}'
                counts = pair_data[f'{key}_counts']
                result[f'{key}_counts'] = counts
                result[f'{key}_pair_s'] = np.array(pair_s[region]).reshape(len(pair_paddings), len(counts))
                result[f'{key}_pair_s_reference'] = pair_structure(
                    reference_at, c0_reference, inputs[f'{geometry}_q_mpc_h'], pair_data[f'{key}_pairs'], counts)
    atomic_savez(args.output, **result)

    for geometry, spec in domains.items():
        print(f'\n{geometry}: C(0) [(Mpc/h)^2] reference {result[f"{geometry}_c0_reference"]:.4f}; '
              + ', '.join(f'alpha={a:g} {c:.4f}' for a, c in zip(paddings, result[f'{geometry}_c0'])))
        targets = np.array(TABLE_TARGETS + (spec['radius'], 1.5 * spec['radius'], spec['diameter']))
        blocks = [(name, f'{name}_sqrt_s') for name in DIRECTIONS[geometry]]
        blocks.append(('axis, component along the separation', 'axis_sqrt_s_parallel'))
        for title, key in blocks:
            s = result[f'{geometry}_{key.split("_sqrt_s")[0]}_separation_mpc_h']
            print(f'{geometry} {title}: sigma_12 [Mpc/h]\n{"s":>10}{"reference":>12}'
                  + ''.join(f'{"a=" + format(a, "g"):>12}' for a in paddings))
            for index in sorted(set(np.argmin(abs(s[:, None] - targets), axis=0))):
                print(f'{s[index]:10.2f}{result[f"{geometry}_{key}_reference"][index]:12.4f}'
                      + ''.join(f'{v:12.4f}' for v in result[f'{geometry}_{key}'][:, index]))
        for family in OPPOSITE_FAMILIES[geometry]:
            print(f'{geometry} opposite points, {family}: C(2R n)/C(0), reference '
                  f'{result[f"{geometry}_opposite_correlation_reference"]:.5f}\n{"deg":>10}'
                  + ''.join(f'{"a=" + format(a, "g"):>12}' for a in paddings))
            for i, degrees in enumerate(result[f'{geometry}_opposite_{family}_degrees']):
                print(f'{degrees:10.2f}' + ''.join(
                    f'{v:12.5f}' for v in result[f'{geometry}_opposite_{family}_correlation'][:, i]))


if __name__ == '__main__':
    main()
