#!/usr/bin/env python3
"""Mean density contrast and mean strain of an open domain after the production displacement.

For a sphere or a full-period cylinder inside periodic boxes of each requested
padding, the script measures over an ensemble of Gaussian seeds, with d = 3 for
the sphere and 2 for the cylinder (only the transverse axes) and V the domain
volume:

* the monopole implied by the production 2LPT displacements on the domain
  boundary, -(d/R) <Psi.n>, the same average taken directly from the density
  modes with the Fourier window of the domain, and the predicted standard
  deviation sqrt((1/V_box) sum_k P(k) W(k)^2) over the retained modes, with its
  continuum limit;
* the mean strain E_ij = (1/V) int d_i Psi_j dV = (d/R) <n_i Psi_j> of the
  domain, whose trace is minus the monopole and which is also computed per seed
  from the density modes, -sum_k khat_i khat_j delta_k W, and the variance of its
  traceless part, predicted by (1/V_box) sum_k P W^2 (khat_i khat_j - delta_ij/d)^2
  and by the isotropic continuum;
* the correlation of the displacements at opposite boundary points, +-R nhat,
  for directions at several angles from the x axis, averaged over the
  directions equivalent under the symmetry of the box. It is compared with
  rho = C(+R nhat, -R nhat) / C(R nhat, R nhat) of the linear displacement, and
  with the continuum value C(2R) / C(0), C(r) = (1/V_box) sum_k P(k) / k^2
  cos(k.r), cut at the Nyquist wavenumber of the padding-1 mesh. The
  prediction is exact for the points used: each point reads the 8 mesh nodes
  of its CIC stencil, so the covariance of two points is the weighted sum of
  the lattice covariance of the mesh field, (1/V_box) sum_k P/k^2 exp(i k.n cell)
  over the retained modes, between their stencil nodes.

Radii and the axial period are given in Mpc and converted to Mpc/h with the
validation cosmology. The boundary quadrature points are at most half a mesh
cell apart. The opposite points are interpolated in the same 2LPT call.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.fft import irfftn
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, simpson
from scipy.special import j0, j1

from stepsic.field import cubic_voxels, fourier_vectors, generate_delta_k
from stepsic.lpt import lpt2
from validation._common.fields import VALIDATION_COSMOLOGY, init_cosmology
from validation._common.evaluation import atomic_savez

GEOMETRIES = ('spherical', 'cylindrical')
FIRST_SEED = 1000
REDSHIFT = 31.
THETAS = (0, 5, 10, 15, 22.5, 30, 45)
BODY_DIAGONAL = 54.7356103172
FAMILIES = {'spherical': ('face', 'body'), 'cylindrical': ('transverse',)}
CONTINUUM_STEP = 2e-4


def window(geometry, kx, ky, kz, radius):
    """Fourier transform of the normalized domain indicator."""
    if geometry == 'spherical':
        x = np.sqrt(kx**2 + ky**2 + kz**2) * radius
        with np.errstate(invalid='ignore', divide='ignore'):
            w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        return np.where(x > 0, w, 1.)
    # A full axial period averages every mode with k_z != 0 to zero.
    x = np.sqrt(kx**2 + ky**2) * radius
    with np.errstate(invalid='ignore', divide='ignore'):
        w = 2 * j1(x) / x
    return np.where(kz == 0, np.where(x > 0, w, 1.), 0.)


def box(spec, alpha):
    side = 2 * alpha * spec['radius']
    lbox = np.array([side, side, side if spec['length'] is None else spec['length']])
    nvox, cell = cubic_voxels(spec['cells'] * (alpha if spec['length'] is None else 1), lbox)
    return nvox, cell


def modes(nvox, cell, spline):
    """Wavevectors, retention mask and spectrum of the modes of the box."""
    # Integer mode indices; the white noise zeroes the Nyquist planes and the zero mode.
    index = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in nvox]
    kept = [abs(i) < n // 2 for i, n in zip(index, nvox)]
    kx, ky, kz = np.meshgrid(*[2 * np.pi * i / (n * cell) for i, n in zip(index, nvox)],
                             indexing='ij', sparse=True)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    keep = (k > 0) & kept[0][:, None, None] & kept[1][None, :, None] & kept[2][None, None, :]
    power = np.exp(spline(np.log(np.where(k > 0, k, 1.))))
    return kx, ky, kz, k, keep, power


def retained_variance(geometry, radius, nvox, cell, spline):
    """(1/V) sum of P W^2 over the modes the white noise retains."""
    kx, ky, kz, k, keep, power = modes(nvox, cell, spline)
    w = window(geometry, kx, ky, kz, radius)
    return float(np.sum(np.where(keep, power * w**2, 0.)) / np.prod(nvox * cell))


def strain_variance(geometry, radius, nvox, cell, spline):
    """Variance of each component of the traceless mean strain, (d, d)."""
    kx, ky, kz, k, keep, power = modes(nvox, cell, spline)
    dimension = 3 if geometry == 'spherical' else 2
    weight = np.where(keep, power * window(geometry, kx, ky, kz, radius)**2, 0.)
    norm = k if dimension == 3 else np.sqrt(kx**2 + ky**2)
    unit = [c / np.where(norm > 0, norm, 1.) for c in (kx, ky, kz)[:dimension]]
    variance = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            term = unit[i] * unit[j] - (i == j) / dimension
            variance[i, j] = variance[j, i] = np.sum(weight * term**2) / np.prod(nvox * cell)
    return variance


def cic_stencil(points, nvox, cell):
    """Mesh nodes (N, 8, 3) and CIC weights (N, 8) of ``points``, as used by the production interpolation."""
    box = nvox * cell
    g = np.mod(points + box / 2, box) / cell - .5
    base = np.floor(g).astype(int)
    frac = g - base
    corners = np.array(list(itertools.product((0, 1), repeat=3)))
    nodes = (base[:, None, :] + corners) % nvox
    weight = np.prod(np.where(corners == 1, frac[:, None, :], 1 - frac[:, None, :]), axis=2)
    return nodes, weight


def displacement_covariance(nvox, cell, spline, groups):
    """Mean product and mean square of the CIC-interpolated displacement, for each (plus, minus) group of points."""
    kx, ky, kz, k, keep, power = modes(nvox, cell, spline)
    spectrum = np.where(keep, power / np.where(k > 0, k, 1.)**2, 0.)
    grid = irfftn(spectrum[:, :, :nvox[2] // 2 + 1], s=tuple(nvox)) / cell**3

    def covariance(x, y):
        (nx, wx), (ny, wy) = cic_stencil(x, nvox, cell), cic_stencil(y, nvox, cell)
        lattice = (nx[:, :, None, :] - ny[:, None, :, :]) % nvox
        return np.mean(np.sum(wx[:, :, None] * wy[:, None, :] * grid[tuple(np.moveaxis(lattice, -1, 0))], axis=(1, 2)))

    product = np.array([covariance(plus, minus) for plus, minus in groups])
    square = np.array([(covariance(plus, plus) + covariance(minus, minus)) / 2 for plus, minus in groups])
    return product, square


def continuum_variance(geometry, spec, spline, kmin, kmax):
    radius = spec['radius']
    power = lambda k: np.exp(spline(np.log(k)))
    if geometry == 'spherical':
        integrand = lambda k: k**2 * power(k) * window(geometry, k, 0., 0., radius)**2 / (2 * np.pi**2)
    else:
        integrand = lambda k: k * power(k) * window(geometry, k, 0., 0., radius)**2 / (2 * np.pi * spec['length'])
    return quad(integrand, kmin, kmax, limit=2000, points=np.geomspace(kmin, kmax, 40)[1:-1])[0]


def k_grid(lo, hi, step):
    """Logarithmic grid up to 2e-3 h/Mpc, then uniform with spacing ``step``."""
    knee = min(2e-3, hi)
    count = int(np.ceil(np.log(knee / lo) * 2e-3 / step)) + 1
    return np.unique(np.concatenate([np.geomspace(lo, knee, count), np.arange(knee, hi, step), [hi]]))


def continuum_correlation(geometry, spec, spline, kmin, kmax, step=CONTINUUM_STEP):
    """C_inf(2R) / C_inf(0) of the displacement, with the spectrum cut at ``kmax``."""
    separation = 2 * spec['radius']
    power = lambda k: np.exp(spline(np.log(k)))
    if geometry == 'spherical':
        k = k_grid(kmin, kmax, step)
        return simpson(power(k) * np.sinc(k * separation / np.pi), x=k) / simpson(power(k), x=k)
    length = spec['length']
    num = den = 0.
    for n in range(-int(kmax * length / (2 * np.pi)), int(kmax * length / (2 * np.pi)) + 1):
        kz = 2 * np.pi * n / length
        top = np.sqrt(kmax**2 - kz**2)
        if top <= kmin:
            continue
        q = k_grid(kmin, top, step)
        f = q * power(np.hypot(q, kz)) / (q**2 + kz**2)
        num += simpson(f * j0(q * separation), x=q)
        den += simpson(f, x=q)
    return num / den


def surface(geometry, spec, spacing):
    """Quadrature points on the lateral boundary, at most ``spacing`` apart, and their normals."""
    radius = spec['radius']
    if geometry == 'spherical':
        # Fibonacci lattice with one point per spacing^2 of area.
        count = int(np.ceil(4 * np.pi * radius**2 / spacing**2))
        i = np.arange(count) + .5
        z = 1 - 2 * i / count
        phi = np.pi * (3 - np.sqrt(5)) * i
        normal = np.column_stack([np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z])
        return radius * normal, normal
    nphi = int(np.ceil(2 * np.pi * radius / spacing))
    nz = int(np.ceil(spec['length'] / spacing))
    phi = 2 * np.pi * (np.arange(nphi) + .5) / nphi
    z = spec['length'] * ((np.arange(nz) + .5) / nz - .5)
    phi, z = (a.ravel() for a in np.meshgrid(phi, z, indexing='ij'))
    normal = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    return np.column_stack([radius * normal[:, :2], z]), normal


def opposite_directions(geometry, family, theta):
    """Directions at ``theta`` degrees equivalent under the box symmetry, one per antipodal pair."""
    t = np.radians(theta)
    if family == 'body':
        nhat = np.array([np.cos(t), np.sin(t) / np.sqrt(2), np.sin(t) / np.sqrt(2)])
    else:
        nhat = np.array([np.cos(t), np.sin(t), 0.])
    dimension = 3 if geometry == 'spherical' else 2
    images = {}
    for perm in itertools.permutations(range(dimension)):
        for signs in itertools.product((1., -1.), repeat=dimension):
            image = np.zeros(3)
            image[:dimension] = np.array(signs) * nhat[list(perm)]
            image = np.round(image, 9) + 0.
            first = image[np.nonzero(abs(image) > 1e-9)[0][0]]
            images[tuple(np.sign(first) * image + 0.)] = None
    return np.array(list(images))


def opposite_points(geometry, radius, directions, zs):
    plus = radius * directions
    if geometry == 'spherical':
        return plus, -plus
    plus = np.repeat(plus, len(zs), axis=0)
    plus[:, 2] = np.tile(zs, len(directions))
    return plus, plus * [-1, -1, 1]


def direct_monopole(geometry, radius, delta_k, nvox, cell):
    """Domain average of the band-limited density field and of the displacement gradient, from the modes."""
    kx, ky, kz = fourier_vectors(nvox, cell, hermitian=True)
    kx, ky, kz = kx[:, None, None], ky[None, :, None], kz[None, None, :]
    # Mesh index 0 lies at -L/2 + cell/2; the domain is centered at the origin.
    x0 = -nvox * cell / 2 + cell / 2
    shifted = delta_k * window(geometry, kx, ky, kz, radius)
    shifted = shifted * np.exp(-1j * (kx * x0[0] + ky * x0[1] + kz * x0[2]))
    # The strain is the point value of -khat_i khat_j times the same field; the
    # half-complex array counts every plane except k_z = 0 and Nyquist twice.
    dimension = 3 if geometry == 'spherical' else 2
    norm = np.sqrt(sum(c**2 for c in (kx, ky, kz)[:dimension]))
    unit = [c / np.where(norm > 0, norm, 1.) for c in (kx, ky, kz)[:dimension]]
    multiplicity = np.full(delta_k.shape[2], 2.)
    multiplicity[0] = 1.
    if nvox[2] % 2 == 0:
        multiplicity[-1] = 1.
    monopole = float(irfftn(shifted, s=tuple(nvox))[0, 0, 0])
    shifted = shifted * multiplicity
    strain = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            strain[i, j] = strain[j, i] = -np.sum((unit[i] * unit[j] * shifted).real) / np.prod(nvox)
    return monopole, strain


def measure(geometry, spec, alphas, seeds, growth):
    spline = CubicSpline(np.log(growth.kh_camb), np.log(growth.pk_input))
    dimension = 3 if geometry == 'spherical' else 2
    radius = spec['radius']
    families = FAMILIES[geometry]
    angles = {family: THETAS + (BODY_DIAGONAL,) * (family == 'body') for family in families}
    result = {}
    for alpha in alphas:
        nvox, cell = box(spec, alpha)
        positions, normals = surface(geometry, spec, cell / 2)
        boundary = len(positions)
        zs = np.unique(positions[:, 2])
        points, spans, groups = [positions], {}, []
        for family in families:
            for theta in angles[family]:
                plus, minus = opposite_points(geometry, radius, opposite_directions(geometry, family, theta), zs)
                spans[family, theta] = (sum(map(len, points)), len(plus))
                points += [plus, minus]
                groups.append((plus, minus))
        everything = np.concatenate(points)
        flux = np.empty(len(seeds))
        direct = np.empty(len(seeds))
        strain_direct = np.empty((len(seeds), dimension, dimension))
        strain = np.empty((len(seeds), dimension, dimension))
        product = {key: np.empty(len(seeds)) for key in spans}
        square = {key: np.empty(len(seeds)) for key in spans}
        for i, seed in enumerate(seeds):
            delta_k = generate_delta_k(growth.kh_camb, growth.pk_input, nvox, cell, seed=int(seed))
            x, _ = lpt2(everything.copy(), delta_k, nvox, cell, g1=growth.g1, g2=growth.g2,
                        aHf1=growth.aHf1, aHf2=growth.aHf2, compensate=False, method='cic')
            psi = x - everything
            flux[i] = -dimension / radius * np.mean(np.sum(psi[:boundary] * normals, axis=1))
            strain[i] = dimension / radius * normals[:, :dimension].T @ psi[:boundary, :dimension] / boundary
            for key, (start, count) in spans.items():
                plus, minus = psi[start:start + count], psi[start + count:start + 2 * count]
                product[key][i] = np.mean(np.sum(plus * minus, axis=1))
                square[key][i] = np.mean(np.sum(plus**2 + minus**2, axis=1)) / 2
            direct[i], strain_direct[i] = direct_monopole(geometry, radius, delta_k, nvox, cell)
        predicted = np.sqrt(retained_variance(geometry, radius, nvox, cell, spline))
        strain_predicted = strain_variance(geometry, radius, nvox, cell, spline)
        product_predicted, square_predicted = displacement_covariance(nvox, cell, spline, groups)
        rho = product_predicted / square_predicted
        rms = np.sqrt(np.mean(flux**2))
        print(f'{geometry} alpha={alpha} mesh={nvox.tolist()} cell={cell:.3f} '
              f'boundary points={len(positions)}: '
              f'predicted {predicted:.4e}, measured rms {rms:.4e} '
              f'(+/- {rms / np.sqrt(2 * len(seeds)):.1e}), mean {flux.mean():+.2e}, '
              f'max |flux-direct|/predicted {np.max(abs(flux - direct)) / predicted:.2e}', flush=True)
        traceless = strain - np.trace(strain, axis1=1, axis2=2)[:, None, None] / dimension * np.eye(dimension)
        total = np.sum(traceless**2, axis=(1, 2))
        offdiagonal = total - np.sum(np.diagonal(traceless, axis1=1, axis2=2)**2, axis=1)
        predicted_offdiagonal = strain_predicted.sum() - np.trace(strain_predicted)
        line = (f'  strain rms {np.sqrt(np.mean(total)):.4e} (predicted {np.sqrt(strain_predicted.sum()):.4e}), '
                f'off-diagonal share {np.mean(offdiagonal) / np.mean(total):.3f} '
                f'(predicted {predicted_offdiagonal / strain_predicted.sum():.3f})')
        scale = np.sqrt(strain_predicted.sum())
        gap = abs(strain - strain_direct) / scale
        on_diagonal = np.eye(dimension, dtype=bool)
        line += (f', max |strain-direct|/rms diagonal {gap[:, on_diagonal].max():.2e} '
                 f'off-diagonal {gap[:, ~on_diagonal].max():.2e}')
        offset = 0
        for family in families:
            n = len(angles[family])
            ratio = [product[family, t].sum() / square[family, t].sum() for t in angles[family]]
            line += f'; {family} corr ' + ' '.join(
                f'{t:g}:{m:.3f}/{p:.3f}' for t, m, p in zip(angles[family], ratio, rho[offset:offset + n]))
            offset += n
        print(line, flush=True)
        # RMS over the realizations with standard errors from the fourth
        # moments, the off-diagonal share of the traceless strain, and the
        # largest boundary-minus-direct differences in units of the prediction.
        count = len(seeds)
        strain_rms = np.sqrt(np.mean(total))
        share = np.mean(offdiagonal) / np.mean(total)
        result.update({f'{geometry}_a{alpha}_{key}': value for key, value in dict(
            flux=flux, direct=direct, predicted=predicted, mesh=nvox, cell_mpc_h=cell,
            boundary_points=len(positions), strain=strain, strain_direct=strain_direct, strain_predicted_variance=strain_predicted,
            rms=rms, rms_error=np.sqrt(np.mean((flux**2 - rms**2)**2) / count) / (2 * rms),
            strain_rms=strain_rms, strain_rms_predicted=scale,
            strain_rms_error=np.sqrt(np.mean((total - strain_rms**2)**2) / count) / (2 * strain_rms),
            offdiagonal_share=share, offdiagonal_share_predicted=predicted_offdiagonal / strain_predicted.sum(),
            offdiagonal_share_error=np.sqrt(np.mean((offdiagonal - share * total)**2) / count) / np.mean(total),
            direct_difference=np.max(abs(flux - direct)) / predicted, strain_direct_difference=gap.max()).items()})
        offset = 0
        for family in families:
            n = len(angles[family])
            result[f'{geometry}_opposite_{family}_degrees'] = np.array(angles[family])
            result.update({
                f'{geometry}_a{alpha}_opposite_{family}_product': np.column_stack([product[family, t] for t in angles[family]]),
                f'{geometry}_a{alpha}_opposite_{family}_square': np.column_stack([square[family, t] for t in angles[family]]),
                f'{geometry}_a{alpha}_opposite_{family}_predicted': rho[offset:offset + n],
                f'{geometry}_a{alpha}_opposite_{family}_square_predicted': square_predicted[offset:offset + n]})
            offset += n
    kmin, kmax = growth.kh_camb[0], growth.kh_camb[-1]
    variance = continuum_variance(geometry, spec, spline, kmin, kmax)
    result[f'{geometry}_continuum'] = np.sqrt(variance)
    result[f'{geometry}_radius_mpc_h'] = radius
    diagonal, off = ((4 / 45, 1 / 15) if dimension == 3 else (1 / 8, 1 / 8))
    result[f'{geometry}_strain_continuum_variance'] = variance * np.where(np.eye(dimension, dtype=bool), diagonal, off)
    continuum_strain = result[f'{geometry}_strain_continuum_variance']
    result[f'{geometry}_strain_rms_continuum'] = np.sqrt(continuum_strain.sum())
    result[f'{geometry}_offdiagonal_share_continuum'] = 1 - np.trace(continuum_strain) / continuum_strain.sum()
    nyquist = np.pi / box(spec, 1)[1]
    result[f'{geometry}_opposite_continuum'] = continuum_correlation(geometry, spec, spline, kmin, nyquist)
    print(f'{geometry} continuum: {result[f"{geometry}_continuum"]:.4e}, '
          f'opposite-point correlation {result[f"{geometry}_opposite_continuum"]:.3f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for geometry in GEOMETRIES:
        parser.add_argument(f'--{geometry}-radius-mpc', type=float, required=True)
        parser.add_argument(f'--{geometry}-cells', type=int, required=True,
                            help='mesh cells along the shortest box axis at padding 1')
    parser.add_argument('--cylindrical-period-mpc', type=float, required=True)
    parser.add_argument('--paddings', type=int, nargs='+', required=True,
                        help='box widths in domain diameters')
    parser.add_argument('--seeds', type=int, required=True,
                        help=f'number of Gaussian seeds, starting at {FIRST_SEED}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    h = VALIDATION_COSMOLOGY['H0'] / 100
    domains = {
        'spherical': dict(radius=args.spherical_radius_mpc * h, length=None, cells=args.spherical_cells),
        'cylindrical': dict(radius=args.cylindrical_radius_mpc * h, length=args.cylindrical_period_mpc * h,
                            cells=args.cylindrical_cells),
    }
    longest = 2 * max(args.paddings) * max(spec['radius'] for spec in domains.values())
    growth = init_cosmology([REDSHIFT], longest, kmin=1e-5, kmax=50., npoints=4096)[REDSHIFT]
    seeds = FIRST_SEED + np.arange(args.seeds)
    result = dict(seeds=seeds, redshift=REDSHIFT, alphas=np.array(args.paddings))
    for geometry, spec in domains.items():
        result.update(measure(geometry, spec, args.paddings, seeds, growth))
    atomic_savez(args.output, **result)


if __name__ == '__main__':
    main()
