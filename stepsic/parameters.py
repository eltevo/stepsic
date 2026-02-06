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

from __future__ import annotations
import importlib.resources

import toml
import numpy as np
from pathlib import Path
from textwrap import dedent

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CosmoParameters:
    def __init__(self, path: Path):
        with open(path, 'r') as file:
            self.P = toml.load(file)
        self._load_default_cosmology(parameter_set=self.P['COSMOLOGY'])
        self._process_cosmo_params()
        self._process_ic_params()

    def get_parameters(self):
        '''Returns the parameters as a dictionary.'''
        return self.P
    
    def _check_exists(self, name):
        '''Check if the parameter exists in the parameter list.'''
        if name not in self.P:
            raise KeyError(f'{name} is not defined in the parameters!')
        return self.P[name]

    def _check_path(self, name, makedir=False):
        '''Check if the parameter is a valid file path.'''
        value = self.P[name]
        if not isinstance(value, (str, Path)):
            raise TypeError(f'{name} must be a string or a Path object!')
        path = Path(value)
        if not path.exists():
            if makedir:
                path.mkdir(parents=True)
            else:
                raise FileNotFoundError(f'{name} does not exist: {path}')
        return path

    def _check_string(self, name):
        '''Check if the parameter is a string.'''
        value = self.P[name]
        if not isinstance(value, str):
            raise TypeError(f'{name} must be a string!')
        self.P[name] = value.strip().lower()
    
    def _check_boolean(self, name):
        '''Check if the parameter is a boolean value.'''
        value = self.P[name]
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a boolean!')

    def _check_scalar(self, name, dtype=np.float64):
        '''Check if the parameter is a scalar value.'''
        value = self.P[name]
        if not np.isscalar(value):
            raise TypeError(f'{name} must be a scalar!')
        self.P[name] = dtype(value)

    def _check_array_or_scalar(self, name, length:int = 3, dtype=np.float64):
        '''
        Check if the parameter is a scalar or a container of a given length.
        If it is a scalar, convert it to an array of the specified length.
        '''
        value = self.P[name]
        if not (np.isscalar(value) or (hasattr(value, '__len__') and len(value) == length)):
            raise TypeError(f'{name} must be a scalar or a container of {length} elements!')
        if not np.issubdtype(np.asarray(value).dtype, np.number):
            raise TypeError(f'{name} must be a scalar or numeric array!')
        self.P[name] = np.broadcast_to(value, (length,)).astype(dtype)

    def _load_default_cosmology(self, *,
            parameter_set='Planck2018EE+BAO+SN', value_type='best'):
        '''TODO'''
        if value_type not in ['best', 'mean']:
            raise ValueError("The 'value_type' must be either 'best' or 'mean'!")

        with importlib.resources.path('stepsic.config', 'cosmology.toml') as path:
            with open(path, 'r') as f:
                config_data = toml.load(f)

        if parameter_set not in config_data:
            raise KeyError(f"Parameter set '{parameter_set}' not found in the TOML data.")

        config_data = config_data[parameter_set]

        for key, value in config_data.items():
            if isinstance(value, dict):
                # Values with best fit/68% limit values
                self.P.setdefault(key, value[value_type])
                if value_type == 'mean' and key not in self.P:
                    self.P[f'{key}_err'] = value['error']
            else:
                # Single value parameters
                self.P.setdefault(key, value)

    def _process_cosmo_params(self):
        '''
        Print all cosmological parameters found in the parameter list.
        '''
        __COSMO_PARAMS = [
            'H0', 'OMEGA_M', 'OMEGA_B', 'OMEGA_L', 'NS', 'AS',
            'SIGMA8', 'YHE', 'MNU', 'NNU', 'ZREI', 'TCMB',
            'KPIVOT', 'W0', 'WA'
        ]
        for key in __COSMO_PARAMS:
            self._check_scalar(key)

        self.P['H'] = self.P['H0'] / 100.0
        self.P['OMMH2'] = self.P['OMEGA_M'] * self.P['H']**2
        self.P['OMBH2'] = self.P['OMEGA_B'] * self.P['H']**2
        self.P['OMEGA_NU'] = self.P.get('MNU', 0.06) / 93.14 / self.P['H']**2
        self.P['OMEGA_C'] = self.P['OMEGA_M'] - self.P['OMEGA_B'] - self.P['OMEGA_NU']
        self.P['OMCH2'] = self.P['OMEGA_C'] * self.P['H']**2
        self.P['OMEGA_K'] = 1.0 - self.P['OMEGA_M'] - self.P['OMEGA_L']

        text = dedent(f'''
        Cosmological Parameters
        -----------------------
        H0:                {self.P['H0']:.3f} km s^-1 Mpc^-1
        Omega_m:           {self.P['OMEGA_M']:.6f}
        Omega_m h^2:       {self.P['OMMH2']:.6f}
        Omega_c h^2:       {self.P['OMCH2']:.6f}
        Omega_L:           {self.P['OMEGA_L']:.6f}
        Omega_k:           {self.P['OMEGA_K']:.6f}
        Omega_b:           {self.P['OMEGA_B']:.6f}
        Omega_b h^2:       {self.P['OMBH2']:.6f}
        n_s:               {self.P['NS']:.6f}
        A_s:               {self.P['AS']:.6e}
        sigma_8:           {self.P['SIGMA8']:.3f}
        Omega_nu:          {self.P['OMEGA_NU']:.6f}
        M_nu:              {self.P['MNU']:.3f} eV
        N_nu:              {self.P['NNU']:.3f}
        Y_He:              {self.P['YHE']:.6f}
        z_reionization:    {self.P['ZREI']:.3f}
        T_CMB:             {self.P['TCMB']:.3f} K
        k_pivot:           {self.P['KPIVOT']:.3f} Mpc^-1
        w0:                {self.P['W0']:.3f}
        wa:                {self.P['WA']:.3f}
        ''')
        log.info(text)

    def _process_ic_params(self):
        '''Process initial condition parameters from the config file.'''
        self._check_string('GEOMETRY')
        if self.P['GEOMETRY'] not in ['cylindrical', 'spherical', 'cubical']:
            raise ValueError(f"Error: unknown geometry `{self.P['GEOMETRY']}`!\nExiting.")
        self._check_string('BIN_MODE')
        if self.P['BIN_MODE'] not in ['omega', 'volume']:
            raise ValueError(f"Error: unknown binning mode `{self.P['BIN_MODE']}`!\nExiting.")

        self._check_boolean('HINDEPENDENT')
        unit = 'Mpc/h' if self.P['HINDEPENDENT'] else 'Mpc'

        self._check_scalar('NMESH', dtype=int)
        self._check_array_or_scalar('LBOX', length=3)
        if not self.P['HINDEPENDENT']:
            self.P['LBOX'] *= self.P['H']
        self._check_array_or_scalar('PERIODIC', length=3, dtype=bool)
        self._check_array_or_scalar('COI', length=3)
        if not self.P['HINDEPENDENT']:
            self.P['COI'] *= self.P['H']
        self._check_scalar('LPTORDER', dtype=int)

        self._check_scalar('REDSHIFT')
        self.P['SCALE'] = 1.0 / (self.P['REDSHIFT'] + 1.0)

        self._check_scalar('R_3D')
        if not self.P['HINDEPENDENT']:
            self.P['R_3D'] *= self.P['H']
        self._check_scalar('D_4D')
        if not self.P['HINDEPENDENT']:
            self.P['D_4D'] *= self.P['H']
        self._check_scalar('NRBINS', dtype=int)

        self._check_string('TYPE')
        if self.P['TYPE'] not in ['grid', 'random', 'glass']:
            raise ValueError(f"Error: unknown IC pattern `{self.P['TYPE']}`!\nExiting.")
        if self.P['TYPE'] == 'glass':
            self._check_exists('INPUT_GLASS')
            self._check_path('INPUT_GLASS')
        
        self._check_string('IC_PREFIX')
        self._check_path('IC_DIR', makedir=True)
        self._check_string('IC_FORMAT')
        if self.P['IC_FORMAT'] not in ['ascii', 'gadget', 'hdf5']:
            raise ValueError(f"Error: unknown output file format `{self.P['IC_FORMAT']}`!\nExiting.")
        self._check_boolean('USE_DOUBLE')
        if self.P['USE_DOUBLE']:
            self.P['DTYPE'] = np.float64
        else:
            self.P['DTYPE'] = np.float32

        self._check_scalar('NPART', dtype=int)
        self._check_scalar('SEED', dtype=int)

        self._check_string('INTERPOLATION')
        if self.P['INTERPOLATION'] not in ['ngp', 'cic', 'tsc']:
            raise ValueError(f"Error: unknown interpolation method `{self.P['INTERPOLATION']}`!\nExiting.")
        self._check_boolean('COMPENSATE')

        self._check_boolean('SPHEREMODE')
        self._check_boolean('COMOVING')

        self._check_boolean('COUNTER')
        self._check_scalar('PHASE_SHIFT')

        self._check_string('SPECTRUM')
        if self.P['SPECTRUM'] not in ['input', 'camb']:
            raise ValueError(f"Error: unknown spectrum type `{self.P['SPECTRUM']}`!\nExiting.")
        if self.P['SPECTRUM'] == 'input':
            self._check_path('INPUT_SPECTRUM')
            self._check_scalar('INPUT_SPECTRUM_UNIT_L_IN_CM')

        self._check_scalar('UNIT_L_IN_CM')
        self._check_scalar('UNIT_M_IN_G')
        self._check_scalar('UNIT_V_IN_KMPS')

        text = dedent(f'''
        IC parameters
        -------------
        Random seed:                   {self.P['SEED']:d}
        Mesh size:                     {self.P['NMESH']} voxels
        Box size:                      {self.P['LBOX']} {unit}
        Periodicity along x-y-z axis:  {self.P['PERIODIC']}
        Target redshift:               {self.P['REDSHIFT']:.3f}
        Target scale factor:           {self.P['SCALE']:.6f}
        Center of interest:            {self.P['COI']} {unit}
        Euclidean simulation radius:   {self.P['R_3D']} {unit}
        Compact. simulation diameter:  {self.P['D_4D']} {unit}
        Number of grid samples:        {self.P['NGRIDSAMPLES']:d}
        Glass input file:              {self.P['INPUT_GLASS']}
        IC output directory:           {self.P['IC_DIR']}
        IC name prefix:                {self.P['IC_PREFIX']}
        Comoving IC:                   {self.P['COMOVING']}
        Counter phase simulation:      {self.P['COUNTER']}
        Phase shift:                   {self.P['PHASE_SHIFT']:.2f} degrees
        ''')
        log.info(text)