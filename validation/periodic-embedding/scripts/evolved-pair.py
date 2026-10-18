#!/usr/bin/env python3
"""Compare two evolved cylinders whose initial fields were computed in boxes of different width.

Both runs evolve the same glass particles to z = 0 with StePS. Member a comes from a box
of one domain diameter, member b from a box of two diameters whose white noise equals
a's inside the domain. Lengths are reported in Mpc/h; the snapshots are in Mpc.

A. FKP power spectra (the estimator of StePS_Pk.py, unblended) of the particles within
   a fixed distance of the axis at z = 0, for a, for b, and for b moved by minus the shift
   that this region has in common between the runs. Both runs use the same shells.
B. The displacement difference of the same particle, b minus a, with b moved to a's scale
   factor along its velocity: the common shift of the region, the RMS about it, and the RMS
   about the mean in the outermost layer, at the initial redshift and at z = 0.
C. The transverse centre of mass of the whole domain. StePS adds the force of the
   homogeneous exterior, so X'' + 2 H X' = 2 pi G rho X / a^3 holds for each run and for
   the difference of two runs; it is integrated from the first pair of files and compared
   with the measured difference at every output and with linear growth.
D. The mean density contrast of the region at z = 0, and of the whole domain at the initial
   redshift from the radial displacement of the outermost particles.
"""

import argparse
import glob
import logging
import sys
import types
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import quad, solve_ivp

from stepsic.cosmology import hubble_a
from validation._common.evaluation import atomic_savez

log = logging.getLogger(__name__)

OUTER_FRACTION = 0.97


def load(path):
    with h5py.File(path, 'r') as f:
        group = f['PartType1']
        return {'ids': group['ParticleIDs'][:], 'x': group['Coordinates'][:].astype('f8'),
                'u': group['Velocities'][:].astype('f8'), 'mass': group['Masses'][:],
                'head': dict(f['Header'].attrs)}


def hubble(a, head):
    return hubble_a(a, 100 * head['HubbleParam'], head['Omega0'], head['OmegaLambda'])


def growth(a, head):
    """Linear growth factor for matter and a cosmological constant, not normalised."""
    omega_m, omega_l = head['Omega0'], head['OmegaLambda']
    return hubble_a(a, 1., omega_m, omega_l) * quad(
        lambda x: (x * hubble_a(x, 1., omega_m, omega_l))**-3, 0, a)[0]


def rms(v):
    return np.sqrt(np.mean(np.sum(v**2, 1)))


def difference(a, b, region, outer, radius_of, period):
    """Common shift and RMS about it, of b minus a, in Mpc/h with b at a's scale factor."""
    h, a_ref, a_other = a['head']['HubbleParam'], a['head']['Time'], b['head']['Time']
    move = b['u'] * (a_ref - a_other) / (a_other**1.5 * hubble(a_other, b['head'])) * h
    delta = b['x'] - a['x']
    delta[:, 2] -= period * np.rint(delta[:, 2] / period)
    moved = delta * h + move
    inside = radius_of < region
    layer = (radius_of >= outer[0]) & (radius_of < outer[1])
    shift = moved[inside].mean(0)
    return {'shift': shift, 'rms_region': rms(moved[inside] - shift),
            'rms_layer': rms(moved[layer] - moved[layer].mean(0))}


def centre_of_mass(pairs):
    """Measured, homogeneous-exterior and linear-growth shifts of b minus a [Mpc/h] per output."""
    measured, exterior, linear, scale = [], [], [], []
    start = None
    for a, b in pairs:
        head = a['head']
        h = head['HubbleParam']
        a_ref, a_other = head['Time'], b['head']['Time']
        weight = (a['mass'] / a['mass'].sum())[:, None]
        x_other = b['x'][:, :2] + b['u'][:, :2] * (a_ref - a_other) / (a_other**1.5 * hubble(a_other, b['head']))
        shift = np.sum(weight * (x_other - a['x'][:, :2]), 0)
        if start is None:
            start = a_ref, np.concatenate((shift, a_ref**1.5 * np.sum(weight * (b['u'][:, :2] - a['u'][:, :2]), 0)))

        def rhs(t, y):
            return np.concatenate((y[2:] / (t**3 * hubble(t, head)),
                                   0.75 * head['Omega0'] * (100 * h)**2 * y[:2] / (t**2 * hubble(t, head))))

        solved = solve_ivp(rhs, (start[0], a_ref), start[1], rtol=1e-10, atol=1e-14).y[:2, -1]
        measured.append(shift * h)
        exterior.append(solved * h)
        linear.append(start[1][:2] * growth(a_ref, head) / growth(start[0], head) * h)
        scale.append(a_ref)
    return (np.array(scale), np.array(measured), np.array(exterior), np.array(linear),
            growth(scale[-1], head) / growth(scale[0], head))


def region_contrast(x, mass, radius, domain_radius):
    """Mass within radius of the axis over the mass of the mean density in that volume, minus one."""
    inside = np.hypot(x[:, 0], x[:, 1]) <= radius
    return mass[inside].sum() / mass.sum() * (domain_radius / radius)**2 - 1


def domain_contrast(snapshot, glass_xy, domain_radius):
    """Mean density contrast of the domain from the radial displacement of the outermost particles."""
    r = np.hypot(glass_xy[:, 0], glass_xy[:, 1])
    s = r > OUTER_FRACTION * domain_radius
    normal = glass_xy[s] / r[s, None]
    return -2 / domain_radius * np.mean(np.sum((snapshot['x'][s, :2] - glass_xy[s]) * normal, 1))


def fkp_catalog(path, radius, shift=(0., 0.)):
    """Particles within radius of the axis, moved by a transverse shift [Mpc], and the box period."""
    from nbodykit.lab import ArrayCatalog

    with h5py.File(path, 'r') as f:
        position = f['PartType1/Coordinates'][:].astype('f8')
        period = f['Header'].attrs['BoxSize']
    position[:, :2] += shift
    distance = np.hypot(position[:, 0], position[:, 1])
    inside = distance <= radius
    return ArrayCatalog({'Radius': distance[inside], 'Position': position[inside]}), period


def spectra(snapshot_path, shift, references, args):
    """Wavenumbers, and FKP spectra against the randoms and against the glass, of the region."""
    from StePS_Pk import estimate_power_spectrum_fkp

    data, period = fkp_catalog(snapshot_path, args.radius, shift)
    randoms, glass = (estimate_power_spectrum_fkp(
        data, reference, max(2 * args.radius, period), P0=args.p0, n_radial_bins=args.bins,
        Nmesh=args.nmesh, Geometry='cylindrical', Lzsim=period) for reference in references)
    return (randoms.poles['k'], randoms.poles['power_0'].real - randoms.attrs['shotnoise'] / 2,
            glass.poles['power_0'].real)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair-dir', required=True, help='Directory with sim_a, sim_b, ics/a and ics/b.')
    parser.add_argument('--ic', required=True, help='Initial conditions file below ics/a and ics/b.')
    parser.add_argument('--randoms', required=True, help='Randoms of the FKP estimator.')
    parser.add_argument('--glass', required=True, help='Glass the particles were displaced from.')
    parser.add_argument('--steps-src', required=True, help='StePS source tree with tools/PowerSpectra/StePS_Pk.py.')
    parser.add_argument('--domain-radius', type=float, required=True, help='Cylinder radius [Mpc].')
    parser.add_argument('--radius', type=float, default=157., help='Region about the axis [Mpc].')
    parser.add_argument('--outer', type=float, nargs=2, default=[700., 750.], help='Outermost layer [Mpc].')
    parser.add_argument('--bins', type=int, default=48, help='Number of n(r) bins of the FKP weights.')
    parser.add_argument('--nmesh', type=int, default=512, help='FKP mesh cells per axis.')
    parser.add_argument('--p0', type=float, default=1e4, help='FKP weight amplitude P0 [Mpc^3].')
    parser.add_argument('--k-max', type=float, default=1.46, help='Largest shell wavenumber [h/Mpc].')
    parser.add_argument('--output', required=True, help='Output .npz archive.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    pair = Path(args.pair_dir)
    names_a = sorted(glob.glob(str(pair / 'sim_a' / 'snapshot_*.hdf5')))
    names_b = sorted(glob.glob(str(pair / 'sim_b' / 'snapshot_*.hdf5')))
    assert len(names_a) == len(names_b) > 0
    files = [(str(pair / 'ics' / 'a' / args.ic), str(pair / 'ics' / 'b' / args.ic))] + list(zip(names_a, names_b))
    pairs = [(load(path_a), load(path_b)) for path_a, path_b in files]
    glass = load(args.glass)
    for a, b in pairs:
        assert np.array_equal(a['ids'], glass['ids']) and np.array_equal(b['ids'], glass['ids'])
        assert np.array_equal(a['mass'], b['mass'])
    head = pairs[0][0]['head']
    h = head['HubbleParam']
    period = float(head['BoxSize'])
    glass_radius = np.hypot(glass['x'][:, 0], glass['x'][:, 1])

    # B. displacement differences at the initial redshift and at z = 0
    diffs = [difference(*pairs[i], args.radius, args.outer, glass_radius, period) for i in (0, -1)]
    shift_final = diffs[1]['shift']
    move = -shift_final[:2] / h
    log.info('b moved by %s Mpc', move)

    # A. power spectra in the fixed region
    sys.modules['pygadgetreader'] = types.ModuleType('pygadgetreader')
    sys.path.insert(0, f'{args.steps_src}/tools/PowerSpectra')
    references = [fkp_catalog(path, args.radius)[0] for path in (args.randoms, args.glass)]
    measured = {}
    for name, path, shift in (('a', files[-1][0], (0., 0.)), ('b', files[-1][1], (0., 0.)),
                              ('b_moved', files[-1][1], move)):
        measured[name] = spectra(path, shift, references, args)
        log.info('spectra of %s done', name)
    k = measured['a'][0]
    for name in ('b', 'b_moved'):
        assert np.allclose(measured[name][0], k, rtol=1e-3, equal_nan=True)
    shells = np.isfinite(k) & (k / h < args.k_max)
    k_h = k[shells] / h
    power = {(name, estimate): measured[name][1 + estimate][shells]
             for name in measured for estimate in (0, 1)}
    ratios = {(name, estimate): power[(name, estimate)] / power[('a', estimate)]
              for name in ('b', 'b_moved') for estimate in (0, 1)}

    # C. centre of mass
    scale, com_measured, com_exterior, com_linear, growth_ratio = centre_of_mass(pairs)
    com_difference = np.linalg.norm(com_measured - com_exterior, axis=1) / np.linalg.norm(com_exterior, axis=1)

    # D. density contrasts
    final_a, final_b = pairs[-1]
    moved_b = final_b['x'][:, :2] + move
    contrast = {'a': region_contrast(final_a['x'], final_a['mass'], args.radius, args.domain_radius),
                'b': region_contrast(final_b['x'], final_b['mass'], args.radius, args.domain_radius),
                'b_moved': region_contrast(moved_b, final_b['mass'], args.radius, args.domain_radius)}
    glass_xy = glass['x'][:, :2]
    initial = [domain_contrast(pairs[0][i], glass_xy, args.domain_radius) for i in (0, 1)]

    atomic_savez(
        args.output, k_h=k_h, move=move, scale_factor=scale,
        ratio_same_randoms=ratios[('b', 0)], ratio_moved_randoms=ratios[('b_moved', 0)],
        ratio_same_glass=ratios[('b', 1)], ratio_moved_glass=ratios[('b_moved', 1)],
        **{f'power_{name}_{estimate}': power[(name, i)]
           for name in ('a', 'b', 'b_moved') for i, estimate in enumerate(('randoms', 'glass'))},
        shift=np.array([d['shift'] for d in diffs]), rms_region=np.array([d['rms_region'] for d in diffs]),
        rms_layer=np.array([d['rms_layer'] for d in diffs]),
        com_measured=com_measured, com_exterior=com_exterior, com_linear=com_linear,
        growth_ratio=growth_ratio, com_difference=com_difference,
        region_contrast=np.array([contrast[n] for n in ('a', 'b', 'b_moved')]),
        domain_contrast=np.array(initial))


if __name__ == '__main__':
    main()
