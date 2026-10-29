#!/usr/bin/env python3
"""Finite-box sensitivity of production multiresolution relative displacements."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import toml

from stepsic.data import CosmoData
from stepsic.field import cubic_voxels
from stepsic.multiresolution import (
    _mass_interp_weights, create_nres_mass_map, density_samples, multiresolution_lpt,
)
from stepsic.parameters import CosmoParameters
from stepsic.units import UNIT_L, UNIT_M
from validation._common.fields import VALIDATION_COSMOLOGY, init_cosmology
from validation._common.evaluation import atomic_savez, load_npz
from validation._common.displacement import (
    pair_separations, rms_changes, sample_pairs, simultaneous_intervals, structure_components,
)

GEOMETRIES = ('spherical', 'cylindrical')
PADDINGS = (1, 2, 3)
MEMBERS = (0, 1)
REGIONS = ('central', 'whole')
FIRST_SEED = 137
REDSHIFT = 31.
EDGES = np.geomspace(10., 500., 17)
CENTRAL_RADIUS_MPC_H = 50.
PAIR_SEED = 7251
PAIR_COUNTS = (16384, 32768, 65536)
SAMPLING_THRESHOLD_PERCENTAGE_POINTS = .2
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 8163
SIMULTANEOUS_CONFIDENCE = .95
# Mandatory StePS compactification diameters [Mpc]; a glass load does not use them.
D_4D_MPC = {'spherical': 19.48571428571428, 'cylindrical': 35.}
H = VALIDATION_COSMOLOGY['H0'] / 100


def glass_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def coarse_resolutions(resolutions, geometry, cap):
    """Scale the mass levels' resolutions so that the finest has about ``cap`` cells."""
    # Native cylindrical aspect ratio is 7.5; multiples of four give even transverse meshes.
    multiple = 2 if geometry == 'spherical' else 4
    values = np.asarray(resolutions, dtype=np.float64)
    return np.maximum(multiple, np.rint(values / max(values) * cap / multiple) * multiple).astype(int)


def check_mesh_budget(meshes):
    # A single real field is a strict lower bound, not a peak-RSS estimate.
    gib = np.prod(np.asarray(meshes, dtype=np.float64), axis=-1) * 8 / 1024**3
    if np.max(gib) > 48.:
        raise MemoryError(f'one required float64 mesh needs {np.max(gib):.3f} GiB; '
                          'the process-session limit is 48 GiB')


def cached(path, calculate):
    """Load ``path``, or calculate and save it, so an interrupted run resumes."""
    if not path.exists():
        atomic_savez(path, **calculate())
    return load_npz(path)


def configuration(geometry, domain, work):
    """Driver configuration for a glass with radius and axial period in Mpc."""
    radius = domain.radius
    return dict(
        GEOMETRY=geometry, LBOX=[2*radius, 2*radius, 2*radius if geometry == 'spherical' else domain.period],
        PERIODIC=[False, False, geometry == 'cylindrical'], COI=[0., 0., 0.],
        R_3D=radius, RCRIT=CENTRAL_RADIUS_MPC_H/H, D_4D=D_4D_MPC[geometry],
        TYPE='glass', INPUT_GLASS=str(domain.glass), SEED=42, USE_DOUBLE=True, COMOVING=True,
        HINDEPENDENT=False, ROTATE=0., NMESH=0, NMESHSAMPLES=4, LPTORDER=2,
        REDSHIFT=REDSHIFT, INTERPOLATION='cic', COMPENSATE=False, SPHEREMODE=False,
        PAIRED=False, FIXED=False, IC_FORMAT='hdf5', SAVE_WHITE_NOISE=False, SAVE_SPECTRUM=False,
        COSMOLOGY='Planck2018EE+BAO', SPECTRUM='camb', RESCALE_SIGMA8=True, NONLINEAR=False,
        UNIT_L_IN_CM=float(UNIT_L), UNIT_M_IN_G=float(UNIT_M),
        UNIT_V_IN_KMPS=1.0, IC_DIR=str(work.resolve()), IC_PREFIX=geometry,
    )


def parameters(config, work):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', dir=work) as stream:
        toml.dump(config, stream)
        stream.flush()
        return CosmoParameters(Path(stream.name)).get_parameters()


def load_glass(config, params):
    data = CosmoData.load_snapshot(Path(config['INPUT_GLASS']))
    # These StePS glasses store comoving Mpc. Convert units without deforming the load.
    data.pos = np.asarray(data.pos, dtype=np.float64) * params['H']
    data.mass = np.asarray(data.mass, dtype=np.float64)
    data.rescale_snapshot_mass(params)
    data.center_snapshot(params)
    return data


def prepare(work, domains, seed_count):
    arrays = dict(separation_edges_mpc_h=EDGES, seeds=FIRST_SEED + np.arange(seed_count))
    for geometry in GEOMETRIES:
        domain = domains[geometry]
        config = configuration(geometry, domain, work)
        params = parameters(config, work)
        data = load_glass(config, params)
        resolutions, levels = create_nres_mass_map(4, data.mass_list, data.M_box, params['LBOX'])
        automatic_resolutions = resolutions.copy()
        resolutions = coarse_resolutions(resolutions, geometry, domain.cells)
        hierarchy = [cubic_voxels(n, params['LBOX']) for n in resolutions]
        meshes = np.array([n for n, cell in hierarchy])
        cells = np.array([cell for n, cell in hierarchy])
        factor = np.array([3, 3, 3 if geometry == 'spherical' else 1])
        check_mesh_budget(meshes*factor)
        # Production rounds each mesh to even dimensions. Record its actual extent.
        for name, value in dict(meshes=meshes, cells_mpc_h=cells,
                                automatic_resolutions=automatic_resolutions,
                                box_extents_mpc_h=meshes*cells[:, None],
                                particle_count=data.N_part, glass_path=str(domain.glass),
                                glass_sha256=glass_identity(domain.glass), source_length_unit='Mpc',
                                radius_mpc_h=params['R_3D'], central_radius_mpc_h=CENTRAL_RADIUS_MPC_H,
                                axial_period_mpc_h=params['LBOX'][2],
                                q_mpc_h=data.pos, masses=data.mass, ids=data.id,
                                mass_levels=levels,
                                weights=np.array(_mass_interp_weights(data.mass, levels))).items():
            arrays[f'{geometry}_{name}'] = value
        print(f'{geometry}: {data.N_part} particles, meshes {meshes.tolist()}, '
              f'padding 3 {(meshes*factor).tolist()}', flush=True)
    longest = max(np.max(arrays[f'{g}_box_extents_mpc_h'])*3 for g in GEOMETRIES)
    growth = init_cosmology([REDSHIFT], longest, kmax=100., npoints=2048)[REDSHIFT]
    for geometry in GEOMETRIES:
        if growth.kh_camb[0] > 2*np.pi/longest or growth.kh_camb[-1] < np.sqrt(3)*np.pi/min(arrays[f'{geometry}_cells_mpc_h']):
            raise ValueError('spectrum does not cover represented frequencies')
    arrays.update(k_h_mpc=growth.kh_camb, power_mpc_h3=growth.pk_input,
                  growth=np.array([growth.g1, growth.g2, growth.aHf1, growth.aHf2]))
    atomic_savez(work / 'inputs.npz', **arrays)


def run_member(inputs, geometry, alpha, seed, member):
    factor = np.array([alpha, alpha, alpha if geometry == 'spherical' else 1])
    hierarchy = list(zip(inputs[f'{geometry}_meshes'] * factor, inputs[f'{geometry}_cells_mpc_h']))
    fields = density_samples(inputs['k_h_mpc'], inputs['power_mpc_h3'], hierarchy,
                              dict(SEED=int(seed), DTYPE=np.float64, PAIRED=bool(member), FIXED=False))
    g1, g2, aHf1, aHf2 = inputs['growth']
    displacement, velocity = multiresolution_lpt(
        inputs[f'{geometry}_q_mpc_h'], inputs[f'{geometry}_masses'], inputs[f'{geometry}_mass_levels'],
        hierarchy, fields, order=2, g1=g1, g2=g2, aHf1=aHf1, aHf2=aHf2,
        method='cic', compensate=False)
    velocity /= H * np.sqrt(1/(1+REDSHIFT))
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
        raise ValueError('non-finite particle output')
    return dict(displacement_mpc_h=displacement, stored_velocity=velocity,
                ids=inputs[f'{geometry}_ids'], meshes=np.array([n for n, cell in hierarchy]),
                cells_mpc_h=inputs[f'{geometry}_cells_mpc_h'])


def particle_path(work, geometry, alpha, seed, member):
    return work / 'particles' / f'{geometry}-a{alpha}-seed{seed}-member{member}.npz'


def run(work, inputs):
    for seed in inputs['seeds']:
        for geometry in GEOMETRIES:
            for alpha in PADDINGS:
                for member in MEMBERS:
                    path = particle_path(work, geometry, alpha, seed, member)
                    start = time.monotonic()
                    if not path.exists():
                        atomic_savez(path, **run_member(inputs, geometry, alpha, seed, member))
                    print(f'{path.name}: {time.monotonic()-start:.2f} s', flush=True)


def pairs_for(work, inputs, count):
    def calculate():
        arrays = {}
        for geometry in GEOMETRIES:
            q, masses = inputs[f'{geometry}_q_mpc_h'], inputs[f'{geometry}_masses']
            cylindrical = geometry == 'cylindrical'
            radius = np.linalg.norm(q[:, :2 if cylindrical else 3], axis=1)
            period = float(inputs[f'{geometry}_axial_period_mpc_h']) if cylindrical else None
            for region, extent in zip(REGIONS, (float(inputs[f'{geometry}_central_radius_mpc_h']),
                                                float(inputs[f'{geometry}_radius_mpc_h']))):
                selected = np.flatnonzero(radius <= extent)
                max_sep = np.hypot(2*extent, period/2) if cylindrical else 2*extent
                pairs, counts = sample_pairs(q[selected], masses[selected], EDGES, count,
                                             max_separation_mpc_h=max_sep, seed=PAIR_SEED,
                                             periodic_length_mpc_h=period)
                good = pairs >= 0
                pairs[good] = selected[pairs[good]]
                arrays[f'{geometry}_{region}_pairs'] = pairs
                arrays[f'{geometry}_{region}_counts'] = counts
                means = np.full(len(counts), np.nan)
                for b, n in enumerate(counts):
                    if n:
                        means[b] = np.mean(pair_separations(
                            q, pairs[b, :n], period))
                arrays[f'{geometry}_{region}_separation_mpc_h'] = means
                print(f'{geometry} {region}: pair counts {counts.tolist()}', flush=True)
        return arrays
    return cached(work / f'pairs-{count}.npz', calculate)


def measure(work, inputs, count):
    pairs = pairs_for(work, inputs, count)
    seeds = inputs['seeds']
    # Mean squared relative displacement per Cartesian component [(Mpc/h)^2].
    components = np.full((len(seeds), len(MEMBERS), len(PADDINGS), len(GEOMETRIES),
                          len(REGIONS), len(EDGES) - 1, 3), np.nan)
    for si, seed in enumerate(seeds):
        for ai, alpha in enumerate(PADDINGS):
            for gi, geometry in enumerate(GEOMETRIES):
                for member in MEMBERS:
                    data = load_npz(particle_path(work, geometry, alpha, seed, member))
                    np.testing.assert_array_equal(data['ids'], inputs[f'{geometry}_ids'])
                    for ri, region in enumerate(REGIONS):
                        key = f'{geometry}_{region}'
                        components[si, member, ai, gi, ri] = structure_components(
                            data['displacement_mpc_h'], pairs[f'{key}_pairs'], pairs[f'{key}_counts'])
    seed_components = components.mean(axis=1)
    seed_s = seed_components.sum(axis=-1)
    estimate, lower, upper = simultaneous_intervals(
        seed_s, nbootstrap=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        confidence=SIMULTANEOUS_CONFIDENCE)
    result = dict(seed_component_s_mpc_h2=seed_components, seed_s_mpc_h2=seed_s,
                  rms_mpc_h=np.sqrt(seed_s.mean(axis=0)), change_percent=estimate,
                  lower_percent=lower, upper_percent=upper,
                  component_change_percent=rms_changes(seed_components.mean(axis=0)),
                  separation_edges_mpc_h=EDGES, pair_count=np.int64(count),
                  **{k: v for k, v in pairs.items() if k.endswith(('_counts', '_separation_mpc_h'))})
    atomic_savez(work / f'measurement-{count}.npz', **result)
    return result


def refine(work, inputs):
    """Double the pair count until the plotted changes are stable or the counts run out."""
    previous = measure(work, inputs, PAIR_COUNTS[0])
    result = measure(work, inputs, PAIR_COUNTS[1])
    changes = [result['change_percent'] - previous['change_percent']]
    if np.nanmax(abs(changes[0])) > SAMPLING_THRESHOLD_PERCENTAGE_POINTS:
        previous, result = result, measure(work, inputs, PAIR_COUNTS[2])
        changes.append(result['change_percent'] - previous['change_percent'])
    result.update(
        sampling_changes_percentage_points=np.array(changes),
        sampling_sensitive=np.abs(changes[-1]) > SAMPLING_THRESHOLD_PERCENTAGE_POINTS,
        sampling_threshold_percentage_points=np.float64(SAMPLING_THRESHOLD_PERCENTAGE_POINTS),
        bootstrap_resamples=np.int64(BOOTSTRAP_RESAMPLES), bootstrap_seed=np.int64(BOOTSTRAP_SEED),
        simultaneous_confidence=np.float64(SIMULTANEOUS_CONFIDENCE),
        member_count=np.int64(len(MEMBERS)), fixed_amplitudes=np.bool_(False),
        **{f'{geometry}_radius_mpc_h': inputs[f'{geometry}_radius_mpc_h'] for geometry in GEOMETRIES})
    atomic_savez(work / 'relative-motion.npz', **result)
    print(f'pair sampling maximum change: {np.nanmax(abs(changes[-1])):.4f} percentage points', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('prepare', 'run', 'measure'):
        actions.add_parser(action).add_argument('--work', type=Path, required=True)
    prepare_parser = actions.choices['prepare']
    for geometry in GEOMETRIES:
        prepare_parser.add_argument(f'--{geometry}-glass', type=Path, required=True,
                                    help='relaxed glass snapshot with coordinates in Mpc')
        prepare_parser.add_argument(f'--{geometry}-radius-mpc', type=float, required=True)
        prepare_parser.add_argument(f'--{geometry}-cells', type=int, required=True,
                                    help='finest-mesh cells along the shortest box axis at padding 1')
    prepare_parser.add_argument('--cylindrical-period-mpc', type=float, required=True)
    prepare_parser.add_argument('--seeds', type=int, required=True,
                                help=f'number of Gaussian seeds, starting at {FIRST_SEED}')
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        domains = {geometry: SimpleNamespace(
            glass=getattr(args, f'{geometry}_glass'), radius=getattr(args, f'{geometry}_radius_mpc'),
            cells=getattr(args, f'{geometry}_cells'), period=getattr(args, 'cylindrical_period_mpc'))
            for geometry in GEOMETRIES}
        prepare(work, domains, args.seeds)
        return
    inputs = load_npz(work / 'inputs.npz')
    {'run': run, 'measure': refine}[args.action](work, inputs)


if __name__ == '__main__':
    main()
