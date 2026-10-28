#!/usr/bin/env python3

#*****************************************************************************#
#  stepsic - An initial condition generator for                               #
#           STEreographically Projected cosmological Simulations              #
#    Copyright (C) 2017-2026 Balazs Pal, Gabor Racz                           #
#                                                                             #
#    This program is free software; you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation; either version 2 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#*****************************************************************************#

from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path

import h5py
import scipy
import numpy as np

import stepsic
from stepsic.io import ensure_run_dir
from stepsic.cosmology import (
    CAMBCosmology,
    ColossusCosmology,
    F2_omega,
    F_omega,
    G2_omega,
    hubble_a,
)
from stepsic.data import CosmoData
from stepsic.field import (
    cubic_voxels,
    generate_delta_k,
    white_noise,
)
from stepsic.particles import create_grid, create_particles, create_shell_particles
from stepsic.lpt import log_lpt, lpt1, lpt2
from stepsic.parameters import CosmoParameters
from stepsic.multiresolution import create_nres_mass_map, density_samples, multiresolution_lpt

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    start = time.time()
    print(stepsic.__header__)

    # Reading in input parameter file
    if len(sys.argv) != 2:
        raise ValueError(
            f"Error: missing toml file!\nUsage: "
            f"./{stepsic.__programname__} <input toml file>\nExiting."
        )
    params = CosmoParameters(path=Path(sys.argv[1])).get_parameters()

    # Construct the initial particle load
    log.info('Constructing the initial particle load...')
    if params['TYPE'] == 'glass':
        ic_orig = CosmoData.load_snapshot(Path(params['INPUT_GLASS']))
        ic_orig.to_internal_units(params)
        ic_orig.rescale_snapshot_size(params)
        #converting the glass to 32bit or 64bit floats, depending on the DTYPE parameter
        ic_orig.pos = ic_orig.pos.astype(params['DTYPE'])
        ic_orig.vel = ic_orig.vel.astype(params['DTYPE'])
        ic_orig.mass = ic_orig.mass.astype(params['DTYPE'])
    elif params['TYPE'] == 'shell':
        # Shell-based particle generation for StePS geometries
        pos, mass = create_shell_particles(params)
        ic_orig = CosmoData(
            pos=pos.astype(params['DTYPE']),
            mass=mass.astype(params['DTYPE']),
        )
    elif params['TYPE'] == 'grid':
        nvox, dk = cubic_voxels(params['NGRID'], params['LBOX'])
        pos = create_grid(nvox, dk, dtype=params['DTYPE'], return_coords=False)
        ic_orig = CosmoData(pos=pos)
        del pos
    elif params['TYPE'] == 'random':
        pos = create_particles(
            npart=params['NPART'], boxsize=params['LBOX'], seed=params['SEED'])
        ic_orig = CosmoData(pos=pos.astype(params['DTYPE']))
    ic_orig.rescale_snapshot_mass(params)
    ic_orig.center_snapshot(params)
    # The unperturbed path shifts the original arrays in place. LPT paths bind
    # fresh position and velocity arrays and can share the remaining metadata.
    if params['LPTORDER'] == 0:
        ic = copy.deepcopy(ic_orig)
    else:
        ic = copy.copy(ic_orig)

    if params['LPTORDER'] == 0:
        # No perturbations applied. The unperturbed particle load is
        # written directly, ready for reverse-gravity relaxation into
        # a glass. Velocities remain zero.
        log.info(
            'LPTORDER=0: skipping perturbation theory. '
            'Writing unperturbed particle load for glass-making.'
        )
        ic.periodic_shift(params)

    else:
        # Initialize cosmology models and calculate growth parameters
        cosmo_colossus = ColossusCosmology(
            H0=params['H0'], Om0=params['OMEGA_M'], Ob0=params['OMEGA_B'],
            Ol0=params['OMEGA_L'], sigma8=params['SIGMA8'], ns=params['NS'],
            Neff=params['NNU'], w0=params['W0'], wa=params['WA'], Tcmb0=1e-6)
        g1 = 1
        D1 = g1 * cosmo_colossus.Dzplus0(params['REDSHIFT'])
        g2 = G2_omega(params['SCALE'], params['OMEGA_M'], params['OMEGA_L'])
        D2 = g2 * D1**2  # Bernardeau et al. 2002, eq. 97
        log.info(f"D1(z={params['REDSHIFT']}) = {D1:.6f}")
        log.info(f"D2(z={params['REDSHIFT']}) = {D2:.6f}")

        # Bernardeau et al. 2002, eq. 99
        # Velocity prefactors (a*H*f) should be in km/s/Mpc
        Hz = hubble_a(params['SCALE'], params['H0'], params['OMEGA_M'], params['OMEGA_L'])
        log.info(f'Initial Hubble parameter: {Hz} km/s/Mpc')
        aHf1 = params['SCALE'] * Hz * F_omega(params['SCALE'], params['OMEGA_M'], params['OMEGA_L'])
        aHf2 = params['SCALE'] * Hz * F2_omega(params['SCALE'], params['OMEGA_M'], params['OMEGA_L'])
        log.info(f"1st vel. prefac(z={params['REDSHIFT']}) = {aHf1:.6f}")
        log.info(f"2nd vel. prefac(z={params['REDSHIFT']}) = {aHf2:.6f}")

        # Meshes from coarse to fine: one per sampled particle mass for the
        # multiresolution IC (NMESH = 0), otherwise the single NMESH mesh.
        if params['NMESH'] == 0:
            nres_tab, mass_tab = create_nres_mass_map(
                params['NMESHSAMPLES'], ic_orig.mass_list, ic_orig.M_box, params['LBOX'])
            hierarchy = [cubic_voxels(res, params['LBOX']) for res in nres_tab]
        else:
            hierarchy = [cubic_voxels(params['NMESH'], params['LBOX'])]

        # Construct the linear power spectrum and backscale it to `z`
        if params['SPECTRUM'] == 'camb':
            cosmo_camb = CAMBCosmology(
                H0=params['H0'], ombh2=params['OMBH2'], omch2=params['OMCH2'],
                omk=params.get('OMK', 0.0), mnu=params['MNU'], nnu=params['NNU'],
                YHe=params['YHE'], TCMB=params['TCMB'], zrei=params['ZREI'],
                w0=params['W0'], wa=params['WA'], nonlinear=False)
            # From half the box fundamental mode to 4x the Nyquist frequency
            # of the finest mesh, beyond its corner mode sqrt(3) pi / dk, so
            # no mesh mode falls outside the tabulated range.
            kmin = np.pi / np.max(params['LBOX'])
            kmax = 4 * np.pi / hierarchy[-1][1]
            npoints=2048
            log.info("Generating the linear power spectrum using CAMB: kmin = %.4e h/Mpc, kmax = %.4e h/Mpc, npoints = %d" % (kmin, kmax, npoints))
            kh, pk, pk3 = cosmo_camb.get_spectrum( z=0, As=params['AS'], ns=params['NS'],
                                sigma8_init=params['SIGMA8'] if params['RESCALE_SIGMA8'] else None,
                                kmin=kmin, kmax=kmax, npoints=npoints)
            pk = pk[0]*D1**2  # Backscale P(k,z=0) with D1^2 to desired `z`
        elif params['SPECTRUM'] == 'input':
            # Should contain 2 rows or columns: log(k) and a scaled log(P^3(k))
            kh_log, pk3_log = np.genfromtxt(params['INPUT_SPECTRUM'])
            kh, pk3 = np.exp(kh_log), np.exp(pk3_log)
            pk = pk3 / (kh**3/(2*np.pi**2))

        log.info('Calculating the displacement and velocity field...')
        if params['NMESH'] == 0:
            # If the number of mesh points is not specified, the script
            # will generate NMESHSAMPLES number of ICs with different
            # resolutions. This is the standard method to generate a
            # variable resolution IC for StePS simulations.
            # 
            # Then it calculates the displacement and velocity fields for
            # each grid, which are then interpolated on top of each other to
            # create the final IC.
            displacement, velocity = multiresolution_lpt(
                ic_orig.pos, ic_orig.mass, mass_tab, hierarchy,
                density_samples(kh, pk, hierarchy, params),
                order=params['LPTORDER'], g1=g1, g2=g2,
                aHf1=aHf1, aHf2=aHf2,
                method=params['INTERPOLATION'], compensate=params['COMPENSATE'])
            np.add(displacement, ic_orig.pos, out=displacement)
            ic.pos = displacement
            ic.vel = velocity
            del ic_orig
            if params['SAVE_WHITE_NOISE']:
                # Only the final mass sample is exported.
                nvox, dk = hierarchy[-1]
                field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
                delta_k = generate_delta_k(
                    kh, pk, nvox, dk, field=field,
                    paired=params['PAIRED'], fixed=params['FIXED'],
                    dtype=params['DTYPE'])

        else:
            # If the number of mesh points is specified, the script will
            # generate a single IC with a grid of the specified resolution.
            # This is useful for testing purposes or for generating ICs with
            # a specific resolution.
            nvox, dk = hierarchy[0]
            # White noise field for complete reproducibility
            field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
            delta_k = generate_delta_k(kh, pk, nvox, dk, field=field, paired=params['PAIRED'], fixed=params['FIXED'], dtype=params['DTYPE'])
            if not params['SAVE_WHITE_NOISE']:
                del field  # only read again when saving the white noise

            lpt_kwargs = dict(
                delta_k=delta_k, nvox=nvox, dk=dk, g1=g1, aHf1=aHf1,
                method=params['INTERPOLATION'], compensate=params['COMPENSATE'], dtype=params['DTYPE']
            )
            if params['LPTORDER'] == 1:
                # Use 1st order Lagrangian PT (Zel'dovich approximation)
                xpert, vpert = lpt1(x=ic_orig.pos, **lpt_kwargs)
                log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='1LPT')
            elif params['LPTORDER'] == 2:
                # Use 2nd order Lagrangian PT
                lpt_kwargs.update(dict(g2=g2, aHf2=aHf2))
                xpert, vpert = lpt2(x=ic_orig.pos, **lpt_kwargs)
                log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='2LPT')
            ic.pos = xpert
            ic.vel = vpert
            # LPT is done, so drop the unperturbed load and the Fourier-space
            # inputs (lpt_kwargs pins delta_k) unless they are saved below.
            del ic_orig, lpt_kwargs
            if not params['SAVE_WHITE_NOISE']:
                del delta_k

        # Prepare the IC for final output
        ic.vel /= params['H']  # Convert to [km/s] in all cases
        ic.vel /= np.sqrt(params['SCALE'])  # Gadget/StePS convention
        ic.periodic_shift(params)

        if params['TYPE'] == 'glass':
            ic.from_internal_units(params)

    if not params['HINDEPENDENT']:
        log.info('Converting the IC to H0 dependent units...')
        ic.pos /= params['H']
        ic.mass /= params['H']
        params['LBOX'] /= params['H']
        params['COI'] /= params['H']
        params['R_3D'] /= params['H']
        params['D_4D'] /= params['H']

    if params['GEOMETRY'] == 'spherical':
        log.info('Restoring the sphere to its center of interest...')
        ic.pos += params['COI']

    if not params['COMOVING'] and params['LPTORDER'] > 0:
        log.info('Converting the IC to proper coordinates...')
        ic.pos *= params['SCALE']
        ic.vel *= np.sqrt(params['SCALE'])
        ic.vel += ic.pos * Hz

    # Save the generated files
    run_dir = ensure_run_dir(params)
    header = {
        'BoxSize': params['LBOX'][2],
        'Redshift': params['REDSHIFT'],
        'Omega0': params['OMEGA_M'],
        'OmegaLambda': params['OMEGA_L'],
        'HubbleParam': params['H'],
        'dtype': params['DTYPE'],
        'SimulationRadius': params['R_3D'],
    }
    ic.save_snapshot(path=run_dir / 'ic.hdf5', fmt=params['IC_FORMAT'], **header)
    if params['LPTORDER'] > 0 and params['SAVE_WHITE_NOISE']:
        with h5py.File(run_dir / 'ic_white_noise.hdf5', 'w') as f:
            f.create_dataset('ic_white_noise', data=scipy.fft.irfftn(field, workers=-1))
        with h5py.File(run_dir / 'ic_delta_k.hdf5', 'w') as f:
            f.create_dataset('ic_delta_k', data=delta_k)
    if params['LPTORDER'] > 0 and params['SAVE_SPECTRUM']:
        np.savetxt(run_dir / 'ic_target_pk.txt', np.column_stack((kh, pk)),
                   header=f"k [h/Mpc], P(k) [(Mpc/h)^3] at z = {params['REDSHIFT']}")

    log.info(f'The IC building took {(time.time() - start):.4f} s.')

if __name__ == "__main__":
    main()
