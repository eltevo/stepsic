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
from stepsic._util import ensure_run_dir
from stepsic.cosmology import (
    CAMBCosmology,
    ColossusCosmology,
    F2_omega,
    F_omega,
    hubble_a,
)
from stepsic.data import CosmoData
from stepsic.field import (
    _mass_interp_weights,
    create_grid,
    create_nres_mass_map,
    create_particles,
    cubic_voxels,
    generate_delta_k,
    white_noise,
)
from stepsic.geometry import create_shell_particles
from stepsic.lpt import log_lpt, lpt1, lpt2
from stepsic.parameters import CosmoParameters

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
    # The output IC will be stored here. The LPTORDER=0 and NMESH=0 paths
    # mutate ic's arrays in place, so they need full copies; the single-grid
    # LPT path rebinds ic.pos/ic.vel to fresh arrays and only reads the
    # rest, so a shallow copy avoids duplicating every particle array.
    if params['LPTORDER'] == 0 or params['NMESH'] == 0:
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
        g2 = - 3.0/7.0 * params['OMEGA_M']**(-1/143)
        D2 = g2 * D1**2  # Bernardeau et al. 2002, eq. 97  # Unused!
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

        # Construct the linear power spectrum and backscale it to `z`
        if params['SPECTRUM'] == 'camb':
            cosmo_camb = CAMBCosmology(
                H0=params['H0'], ombh2=params['OMBH2'], omch2=params['OMCH2'],
                omk=params.get('OMK', 0.0), mnu=params['MNU'], nnu=params['NNU'],
                YHe=params['YHE'], TCMB=params['TCMB'], zrei=params['ZREI'],
                w0=params['W0'], wa=params['WA'], nonlinear=False)
            kh, pk, pk3 = cosmo_camb.get_spectrum(
                z=0, As=params['AS'], ns=params['NS'], sigma8_init=params['SIGMA8'],
                kmin=1/np.min(params['LBOX']), kmax=100, npoints=2048)
            pk = pk[0]*D1**2  # Backscale P(k,z=0) with D1^2 to desired `z`
        elif params['SPECTRUM'] == 'input':
            # Should contain 2 rows or columns: log(k) and a scaled log(P^3(k))
            kh_log, pk3_log = np.genfromtxt(params['INPUT_SPECTRUM'])
            kh, pk3 = np.exp(kh_log), np.exp(pk3_log)
            pk = pk3 / (kh**3/(2*np.pi**2))
        # CAMB uses [U/h] units
        #kh, pk, pk3 = kh/params['H'], pk/params['H']**3, pk3/params['H']**3

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
            nres_tab, mass_tab = create_nres_mass_map(
                params['NMESHSAMPLES'], ic_orig.mass_list, ic_orig.M_box, params['LBOX'])

            # Each particle linearly mixes the two samples bracketing its
            # mass, so every sample's contribution can be applied as soon
            # as its fields are computed.
            j_lo, j_hi, w_hi = _mass_interp_weights(ic.mass, mass_tab)

            for si, (res, mass) in enumerate(zip(nres_tab, mass_tab)):
                log.info(f"Generating sample {si+1}/{params['NMESHSAMPLES']}...")
                log.info(f'Resolution: {res:.0f} voxels, Mass: {mass:.6f} 1e11 Msol/h')
                nvox, dk = cubic_voxels(res, params['LBOX'])
                # White noise field for complete reproducibility
                field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
                delta_k = generate_delta_k(kh, pk, nvox, dk, field=field, paired=params['PAIRED'], fixed=params['FIXED'])

                if params['LPTORDER'] == 1:
                    # Use 1st order Lagrangian PT (Zel'dovich approximation)
                    xpert, vpert = lpt1(
                        ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk,
                        g1=g1, aHf1=aHf1,
                        method=params['INTERPOLATION'], compensate=params['COMPENSATE'])
                    log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='1LPT')
                elif params['LPTORDER'] == 2:
                    # Use 2nd order Lagrangian PT
                    xpert, vpert = lpt2(
                        ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk,
                        g1=g1, g2=g2, aHf1=aHf1, aHf2=aHf2,
                        method=params['INTERPOLATION'], compensate=params['COMPENSATE'])
                    log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='2LPT')

                # Accumulate this sample's displacement and velocity
                # contribution for the particles whose mass it brackets
                coeff = (np.where(j_lo == si, 1.0 - w_hi, 0.0)
                         + np.where(j_hi == si, w_hi, 0.0))
                sel = coeff != 0.0
                cw = coeff[sel, None]
                ic.pos[sel] += cw * (xpert[sel] - ic_orig.pos[sel])
                ic.vel[sel] += cw * (vpert[sel] - ic_orig.vel[sel])
        else:
            # If the number of mesh points is specified, the script will
            # generate a single IC with a grid of the specified resolution.
            # This is useful for testing purposes or for generating ICs with
            # a specific resolution.
            nvox, dk = cubic_voxels(params['NMESH'], params['LBOX'])
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
        # shifting back the center of the sphere to the origin
        log.info('Shifting the center of the sphere back to the origin...')
        ic.pos += params['COI'] + params['LBOX']/2

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

    log.info(f'The IC building took {(time.time() - start):.4f} s.')

if __name__ == "__main__":
    main()


#*******************************************************************************#

# Redshift cones code from old stepsic

# if GEOMETRY == 'spherical':
#     print("Calculating redshifts for the spherical shells...")
#     #calculating the comoving distances of the particles
#     shell_limits = np.zeros(np.uint64(Params['NRBINS'])+np.uint64(1), dtype=np.float64)
#     z_list = np.zeros(np.uint64(Params['NRBINS']), dtype=np.float64)
#     if Params['BIN_MODE'] == 0:
#         last_cell_size = Params['NRBINS']*np.pi/(2*np.arctan(Params['RSIM']/Params['D_S']))-Params['NRBINS']
#     i = np.arange(Params['NRBINS'])
#     if Params['BIN_MODE'] == 0:
#         r_list = Calculate_r_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
#         if Params['HINDEPENDENTUNITS'] == 1:
#             r_list *= h
#     if Params['BIN_MODE'] == 1:
#         r_list = Calculate_r_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
#         if Params['HINDEPENDENTUNITS'] == 1:
#             r_list *= h
#     del(i)
#     i = np.arange(Params['NRBINS']+1)
#     if Params['BIN_MODE'] == 0:
#         shell_limits = Calculate_rlimits_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
#         if Params['HINDEPENDENTUNITS'] == 1:
#             shell_limits *= h
#     if Params['BIN_MODE'] == 1:
#         shell_limits = Calculate_rlimits_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
#         if Params['HINDEPENDENTUNITS'] == 1:
#             shell_limits *= h
#     #calculating redshift-comoving distance function for the redshift cone
#     for i in range(0,len(z_list)):
#         z_list[i] = z_at_value(cosmo.comoving_distance, r_list[i]*u.Mpc)
#     if Params['OUTPUTFORMAT'] == 0:
#         outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".dat_zbins"
#     if Params['OUTPUTFORMAT'] == 2:
#         outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".hdf5_zbins"
#     np.savetxt(outputfilename, z_list)
#     if Params['OUTPUTFORMAT'] == 0:
#         outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".dat_zbins_rlimits"
#     if Params['OUTPUTFORMAT'] == 2:
#         outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".hdf5_zbins_rlimits"
#     np.savetxt(outputfilename, shell_limits)
#     print("...done\n")