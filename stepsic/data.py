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
from pathlib import Path

import numpy as np

from stepsic.field import wrap
from stepsic.io import CosmoIO
from stepsic.units import UNIT_L, UNIT_M, UNIT_V

log = logging.getLogger(__name__)


class CosmoData:
    '''
    Implements a class for handling cosmological snapshot data stored in
    a conventional Gadget-like structure for StePS simulations.

    Parameters:
    -----------
    data : ndarray of shape (N, 7)
        Array containing the particle data, where N is the number of particles.
    '''
    def __init__(self, id=None, pos=None, vel=None, mass=None, Lbox=None):
        if pos is None:
            raise ValueError('Particle positions must be provided!')
        if id is None:
            id = np.arange(pos.shape[0], dtype=np.uint64)
        if vel is None:
            vel = np.zeros_like(pos, dtype=np.float32)
        if mass is None:
            mass = np.ones(pos.shape[0], dtype=np.float32)
        self.id = id      # Particle IDs
        self.pos = pos    # Particle positions
        self.vel = vel    # Particle velocities
        self.mass = mass  # Particle masses
        self.Lbox = Lbox  # Linear size of the simulation volume (in z direction)

        # Calculated values
        self.N_part = self.id.size  # Number of particles
        self.mass_list = None       # List of unique particle masses
        self.M_box = None           # Total mass in the box (in Msol)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def to_internal_units(self, params):
        '''TODO'''
        self.pos *= params['UNIT_L_IN_CM'] / UNIT_L
        self.vel *= params['UNIT_V_IN_KMPS'] / UNIT_V
        self.mass *= params['UNIT_M_IN_G'] / UNIT_M

    def from_internal_units(self, params):
        '''TODO'''
        self.pos /= UNIT_L / params['UNIT_L_IN_CM']
        self.vel /= UNIT_V / params['UNIT_V_IN_KMPS']
        self.mass /= UNIT_M / params['UNIT_M_IN_G']

    @classmethod
    def load_snapshot(cls, path: Path, **io_kwargs):
        '''Load snapshot data from a file.'''
        ids, pos, vel, mass = CosmoIO.load_snapshot(path, **io_kwargs)
        Lbox = CosmoIO.get_box_size(path)
        if Lbox is None:
            # Fallback: infer box size from the particle extent along the
            # shortest axis (consistent with how StePS glasses are built)
            Lbox = float(np.max(pos[:, 2]) - np.min(pos[:, 2]))
            log.warning(
                f'BoxSize not found or invalid in snapshot header. '
                f'Inferred Lbox={Lbox:.6f} from particle positions.'
            )
        log.info(f'Loaded snapshot from {path} with box size {Lbox} (internal units).')
        instance = cls(id=ids, pos=pos, vel=vel, mass=mass, Lbox=Lbox)
        return instance
    def save_snapshot(self, path: Path, **io_kwargs):
        '''Save the snapshot data to a file.'''
        CosmoIO.save_snapshot(path, self, **io_kwargs)
        log.info(f'Snapshot saved to {path}.')

    def rescale_snapshot_size(self, params):
        '''
        Rescale the particle positions to fit the input box size.
        
        Parameters
        ----------
        params : dict
            Dictionary containing the cosmological parameters.
        '''
        if self.Lbox is None or not np.isfinite(self.Lbox) or self.Lbox <= 0:
            raise ValueError(
                f'Cannot rescale snapshot: invalid Lbox={self.Lbox}. '
                f'Check the input snapshot header.'
            )
        if params['GEOMETRY'] == 'cylindrical' or params['GEOMETRY'] == 'cubical':
            log.info('Rescaling the snapshot size to fit the desired box size...')
            log.info(f'Target box size (along the z axis): {params["LBOX"][2]} (internal units), loaded box size: {self.Lbox} (internal units).')
            self.pos *= params['LBOX'][2] / self.Lbox
            self.Lbox = params['LBOX'][2]
            return
        elif params['GEOMETRY'] == 'spherical':
            # in spherical (R^3) geometry, we only need to rescale the the glass to h-independent units, if the input snapshot is in h-dependent units.
            if not params['HINDEPENDENT']:
                self.pos *= params['H']
            return
        else:
            # this should never happen 
            raise ValueError(f'Unknown geometry type: {params["GEOMETRY"]}')

    def rescale_snapshot_mass(self, params):
        '''
        Rescale the particle masses to fit the cosmological parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the data array.
        '''
        M_tot = np.sum(self.mass)
        if params['GEOMETRY'] == 'spherical':
            V_sim = 4/3 * params['R_3D']**3 * np.pi
        elif params['GEOMETRY'] == 'cylindrical':
            V_sim = params['R_3D']**2 * np.min(params['LBOX']) * np.pi
        elif params['GEOMETRY'] == 'cubical':
            V_sim = np.prod(params['LBOX'])
        omega_m_box = (M_tot / V_sim) / params['RHO_CRIT']
        if np.isclose(omega_m_box, params['OMEGA_M'], rtol=1e-9):
            log.info(f'Omega_m calculated from particle masses: {omega_m_box = :.6f}')
        else:
            self.mass *= params['OMEGA_M'] / omega_m_box
            log.info(f'Particle masses were rescaled to fit Omega_m = {params["OMEGA_M"]}')
        log.info(f'Minimal particle mass: {np.min(self.mass)*1e11:.6e} Msol')
        log.info(f'Maximal particle mass: {np.max(self.mass)*1e11:.6e} Msol')
        log.info(f'Total mass in the box: {np.sum(self.mass)*1e11:.6e} Msol')
        # Calculate mass statistics after rescaling
        self.mass_list = np.unique(self.mass)
        log.info(f'Number of different masses: {self.mass_list.size}')
        self.M_box = params['RHO_MEAN'] * np.prod(params['LBOX'])

    def center_snapshot(self, params):
        '''
        Shift data to a Center-of-Interest (COI). First, normalize
        particles to the interval :math:`[-L/2, L/2]`, if they are in
        :math:`[0, L]`. Otherwise, assume they are already in this
        interval.

        Parameters
        ----------
        params : dict
            Dictionary containing the cosmological parameters.
        '''
        log.info('Centering the particles around the Center-of-Interest...')
        # move the particles to the center of the box
        Lbox_half = np.multiply(params['LBOX'], 0.5)
        mask = self.pos.max(axis=0) > Lbox_half  # `>` filters grid ICs too
        self.pos = np.where(mask, self.pos - Lbox_half, self.pos)
        # center the particles around a Center-of-Interest
        self.pos -= params['COI']

    def periodic_shift(self, params, periodic=None):
        '''
        Periodically shift the input glass to the desired box size and
        center it in the box. 
        
        Parameters
        ----------
        params : dict
            Dictionary containing the cosmological parameters.
        periodic: ndarray, optional
            Boolean array indicating whether the periodic boundary
            conditions are applied in each dimension.
        '''
        log.info('Periodically shifting the input glass...')
        if periodic is None:
            periodic = params['PERIODIC']
        if np.any(params['PERIODIC']):
            self.pos[:, periodic] = wrap(self.pos[:, periodic], params['LBOX'][periodic])