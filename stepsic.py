#!/usr/bin/env python3

#*******************************************************************************#
#  stepsic - An initial condition generator for                                 #
#            STEreographically Projected cosmological Simulations               #
#    Copyright (C) 2017-2025 Gabor Racz, Balazs Pal                             #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

import sys
import copy
import time
import h5py
import numpy as np
from pathlib import Path

import stepsic
from stepsic.parameters import CosmoParameters
from stepsic.data import CosmoData
from stepsic.cosmology import \
    hubble_a, F_omega, F2_omega, CAMBCosmology, ColossusCosmology
from stepsic.field import \
    cubic_voxels, white_noise, generate_delta_k, \
    create_grid, create_particles, create_nres_mass_map
from stepsic.lpt import lpt1, lpt2, log_lpt

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_filename(params):
    '''
    Construct a filename for the output IC based on its parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the ``stepsic`` simulation parameters.

    Returns
    -------
    str
        The generated filename.
    '''
    fname = f"{params['IC_PREFIX']}_"
    fname += "Lx{}_Ly{}_Lz{}_".format(*map(int, params['LBOX']))
    fname += f"R3D{params['R_3D']:.0f}_D4D{params['D_4D']:.0f}_"
    fname += f"z{params['REDSHIFT']:.0f}"
    return fname

def main():
    start = time.time()
    print(stepsic.__header__)
    # Reading in input parameter file
    if len(sys.argv) != 2:
        raise ValueError('Error: missing toml file!\nUsage: ./StePS_IC.py <input toml file>\nExiting.')
    params = CosmoParameters(path=Path(sys.argv[1])).get_parameters()

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
    # velocity prefactors (a*H*f) should be in km/s/Mpc
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

    # Construct the initial conditions
    if params['TYPE'] == 'glass':
        ic_orig = CosmoData.load_snapshot(Path(params['INPUT_GLASS']))
        ic_orig.to_internal_units(params)
        ic_orig.rescale_snapshot_mass(params)
        ic_orig.center_snapshot(params)
        ic = copy.deepcopy(ic_orig)  # The output IC will be stored here
    if params['TYPE'] == 'grid':
        raise NotImplementedError
        x, _ = create_grid(nvox, dk)
    elif params['TYPE'] == 'random':
        raise NotImplementedError
        x = create_particles(
            npart=params['NPART'], Lbox=params['LBOX'], seed=params['SEED'])

    log.info('Calculating the displacement and velocity field...')
    if params['NMESH'] == 0:
        # If the number of mesh points is not specified, the script will
        # generate NGRIDSAMPLES number of ICs with different resolutions.
        # This is the standard method to generate a variable resolution
        # IC for StePS simulations.
        # 
        # Then it calculates the displacement and velocity fields for
        # each grid, which are then interpolated on top of each other to
        # create the final IC.
        nres_tab, mass_tab = create_nres_mass_map(
            params['NGRIDSAMPLES'], ic_orig.mass_list, ic_orig.M_box, params['LBOX'])

        dis_field = np.zeros((params['NGRIDSAMPLES'], ic_orig.N_part, 3), dtype=params['DTYPE'])
        vel_field = np.zeros((params['NGRIDSAMPLES'], ic_orig.N_part, 3), dtype=params['DTYPE'])

        for si, (res, mass) in enumerate(zip(nres_tab, mass_tab)):
            log.info(f"Generating sample {si+1}/{params['NGRIDSAMPLES']}...")
            log.info(f'Resolution: {res:.0f} voxels, Mass: {mass:.6f} 1e11 Msol')
            nvox, dk = cubic_voxels(res, params['LBOX'])
            # White noise field for complete reproducibility
            field = white_noise(nvox=nvox, seed=params['SEED'])
            with h5py.File(Path(params['IC_DIR'], 'ic_white_noise.hdf5'), 'w') as f:
                f.create_dataset('ic_white_noise', data=np.fft.irfftn(field))
            delta_k = generate_delta_k(kh, pk, nvox, dk, field=field)

            if params['LPTORDER'] == 1:
                # Use 1st order Lagrangian PT (Zel'dovich approximation)
                xpert, vpert = lpt1(
                    ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk, g1=g1, aHf1=aHf1,
                    counter=params['COUNTER'])
                log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='1LPT')
            elif params['LPTORDER'] == 2:
                # Use 2nd order Lagrangian PT
                xpert, vpert = lpt2(
                    ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk, g1=g1, g2=g2,
                    aHf1=aHf1, aHf2=aHf2, counter=params['COUNTER'])
                log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='2LPT')

            # Calculating the displacement field for every grid
            dis_field[si, ...] = xpert - ic_orig.pos
            vel_field[si, ...] = vpert - ic_orig.vel

        log.info('Interpolating between the different resolutions...')
        for i in range(0, ic.N_part):  # TODO: vectorize this!
            for k in range(0, 3):
                ic.pos[i, k] += np.interp(ic.mass[i], mass_tab, dis_field[:, i, k])
                ic.vel[i, k] += np.interp(ic.mass[i], mass_tab, vel_field[:, i, k])
    else:
        # If the number of mesh points is specified, the script will
        # generate a single IC with a grid of the specified resolution.
        # This is useful for testing purposes or for generating ICs with
        # a specific resolution.
        nvox, dk = cubic_voxels(params['NMESH'], params['LBOX'])
        # White noise field for complete reproducibility
        field = white_noise(nvox=nvox, seed=params['SEED'])
        with h5py.File(Path(params['IC_DIR'], 'ic_white_noise.hdf5'), 'w') as f:
            f.create_dataset('ic_white_noise', data=np.fft.irfftn(field))
        delta_k = generate_delta_k(kh, pk, nvox, dk, field=field)

        if params['LPTORDER'] == 1:
            # Use 1st order Lagrangian PT (Zel'dovich approximation)
            xpert, vpert = lpt1(
                ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk, g1=g1, aHf1=aHf1,
                counter=params['COUNTER'])
            log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='1LPT')
        elif params['LPTORDER'] == 2:
            # Use 2nd order Lagrangian PT
            xpert, vpert = lpt2(
                ic_orig.pos, delta_k=delta_k, nvox=nvox, dk=dk, g1=g1, g2=g2,
                aHf1=aHf1, aHf2=aHf2, counter=params['COUNTER'])
            log_lpt(x=ic_orig.pos, xpert=xpert, vpert=vpert, title='2LPT')
        ic.pos = xpert
        ic.vel = vpert

    # Prepare the IC for final output
    ic.periodic_shift(params)
    ic.from_internal_units(params)

    if not params['COMOVING']:
        log.info('Converting the IC to proper coordinates...')
        ic.pos *= params['SCALE']
        ic.vel *= np.sqrt(params['SCALE'])  # StePS/Gadget convention
        ic.vel += ic.pos * Hz

    if params['HINDEPENDENT']:
        log.info('Converting the IC to H0 independent units...')
        ic.pos *= params['H']
        ic.mass *= params['H']

    # Save the IC to a file
    header = {
        'BoxSize': np.max(params['LBOX']),
        'Redshift': params['REDSHIFT'],
        'Omega0': params['OMEGA_M'],
        'OmegaLambda': params['OMEGA_L'],
        'HubbleParam': params['H'],
        'dtype': params['DTYPE']
    }
    path = Path(params['IC_DIR'], create_filename(params))
    ic.save_snapshot(path=path, fmt=params['IC_FORMAT'], **header)

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