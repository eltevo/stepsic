#!/usr/bin/env python3
"""Compare mass-mixed multiresolution displacements with one finest mesh.

With ``NMESH=0`` stepsic assigns each particle mass a mesh whose cell holds
one particle of that mass and interpolates the displacement linearly in mass
between ``NMESHSAMPLES`` such meshes. The same particle load is displaced
once on the finest of those meshes alone, and the difference is binned by
Lagrangian radius. All runs read one input spectrum tapered at 0.03 h/Mpc, a
scale every mesh resolves, so the meshes carry the same modes and the
difference measures the interpolation on the coarser cells.
"""

import argparse
from pathlib import Path
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import toml

from stepsic.field import cubic_voxels
from stepsic.multiresolution import create_nres_mass_map
from validation._common.fields import init_cosmology
from validation._common.evaluation import atomic_savez
from validation._common.plotting import atomic_savefig, setup_matplotlib


TAPER_H_MPC = 0.03
SAMPLES = (2, 4, 8)
NBINS = 10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True,
                   help='stepsic parameter file of the undisplaced shell load')
    p.add_argument('--stepsic', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    config = toml.load(a.config)
    cylindrical = config['GEOMETRY'] == 'cylindrical'
    box = np.array(config['LBOX'], dtype=float)
    radius, rcrit, redshift = config['R_3D'], config['RCRIT'], float(config['REDSHIFT'])

    growth = init_cosmology([redshift], float(np.min(box)), kmax=100., npoints=2048)[redshift]
    k = growth.kh_camb
    power = np.maximum(growth.pk_input*np.exp(-np.minimum((k/TAPER_H_MPC)**16, 700)), 1e-250)
    spectrum = (a.work/'spectrum.txt').resolve()
    np.savetxt(spectrum, np.array([np.log(k), np.log(power*k**3/(2*np.pi**2))]))
    config.update(SPECTRUM='input', INPUT_SPECTRUM=str(spectrum))

    def run(name, **overrides):
        parameters = a.work/f'{name}.toml'
        parameters.write_text(toml.dumps(dict(config, IC_DIR=str(a.work/name), **overrides)))
        with (a.work/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(a.stepsic), str(parameters)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        snapshot, = (a.work/name).rglob('ic.hdf5')
        return snapshot

    def read(snapshot, key):
        with h5py.File(snapshot, 'r') as handle:
            return handle[f'PartType1/{key}'][:]

    def displacement(name, **overrides):
        dx = read(run(name, LPTORDER=2, **overrides), 'Coordinates') - q
        if cylindrical:
            dx[:, 2] -= box[2]*np.rint(dx[:, 2]/box[2])
        return dx

    # Mass mixing needs a stored load, so the shell load is written once and
    # read back as the particle input of every displaced run.
    load = run('load')
    config.update(TYPE='glass', INPUT_GLASS=str(load))
    q, masses = read(load, 'Coordinates'), read(load, 'Masses')
    volume = np.pi*radius**2*box[2] if cylindrical else 4*np.pi*radius**3/3
    box_mass = masses.sum()/volume*np.prod(box)
    resolutions, _ = create_nres_mass_map(2, np.unique(masses), box_mass, box)
    finest = int(cubic_voxels(resolutions[-1], box)[0].min())
    reference = displacement('finest', NMESH=finest)

    radii = np.linalg.norm(q[:, :2 if cylindrical else 3], axis=1)
    edges = np.linspace(0., radius, NBINS + 1)
    residual = np.empty((len(SAMPLES), NBINS))
    for i, count in enumerate(SAMPLES):
        mixed = displacement(f'mixed_{count}', NMESH=0, NMESHSAMPLES=count)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (radii >= lo) & (radii < hi)
            residual[i, j] = np.sqrt(np.sum((mixed[sel]-reference[sel])**2)/np.sum(reference[sel]**2))
    atomic_savez(a.output, radial_edges_mpc_h=edges, samples=np.array(SAMPLES),
                 displacement_relative_residual=residual,
                 meta_geometry=np.array(config['GEOMETRY']), meta_radius_mpc_h=np.float64(radius),
                 meta_rcrit_mpc_h=np.float64(rcrit), meta_box_mpc_h=box,
                 meta_seed=np.int64(config['SEED']), meta_redshift=np.float64(redshift),
                 meta_taper_h_mpc=np.float64(TAPER_H_MPC), meta_finest_mesh=np.int64(finest),
                 meta_mass_ratio=np.float64(masses.max()/masses.min()))

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    centers = (edges[1:]+edges[:-1])/2
    for count, row in zip(SAMPLES, residual):
        ax.plot(centers, 100*row, 'o-', lw=1.2, ms=3, label=f'{count} mass samples')
    ax.axvline(rcrit, color='0.35', ls=':', lw=0.8, label=r'$R_{\rm crit}$')
    ax.set_xlabel(r'Lagrangian radius [$h^{-1}\mathrm{Mpc}$]')
    ax.set_ylabel(r'Displacement residual [\%]')
    ax.legend(fontsize=7, frameon=False)
    atomic_savefig(fig, a.figure)
    plt.close(fig)


if __name__ == '__main__':
    main()
