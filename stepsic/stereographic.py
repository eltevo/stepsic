#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2025 Gabor Racz                                         #
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
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root

import logging
log = logging.getLogger(__name__)


class SphericalBinner(ABC):
    '''TODO
    '''
    @abstractmethod
    def r_limit(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @abstractmethod
    def r_centroid(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @staticmethod
    def centroid(r0: float, r1: float):
        r'''
        Calculates the centroid of a conical frustum along the radius in
        3D space for the i-th bin between r0 and r1.
        The formula used is:
        .. math::
            r_i = \frac{1}{4} \frac{r_1^3 - r_0^3}{r_1^2 - r_0^2} + r_0
        where :math:`r_0` and :math:`r_1` are the lower and upper limits
        of the bin.
        
        Parameters:
        -----------
        r0 : float
            The lower limit of the bin.
        r1 : float
            The upper limit of the bin.
        
        Returns:
        --------
        r_i : float
            The centroid of the i-th bin.
        '''
        r_i = 0.25 * (r1-r0) * (r0*r0 + 2*r0*r1 + 3*r1*r1) / (r0*r0 + r0*r1 + r1*r1) + r0
        return r_i
    
    def invert_x_minus_sin_x(y, *, method='hybr', tol=1e-06):
        r'''
        Solve :math:`x - sin(x) = y for x` in :math:`[0, \pi]`, using
        a root-finding algorithm. 

        Parameters
        ----------
        y : float or ndarray
            Target value; must satisfy :math:`0 \leq y \leq \pi`.
        method : str, optional
            The method to use for root finding. Default is 'hybr'.
            See `scipy.optimize.root` for more options.
        tol : float, optional
            Absolute tolerance for the solver.

        Returns
        -------
        x : ndarray
            The unique solution in :math:`[0, \pi]` such that
            :math:`x - sin(x) = y`.
        '''
        if not np.asarray((y >= 0.0) & (y <= np.pi)).all():
            raise ValueError(f"y must be in [0, Pi], got y={y}")

        f = lambda x: x - np.sin(x) - y
        sol = root(f, x0=np.ones_like(y), method=method, tol=tol)
        if not sol.success:
            raise RuntimeError(f"Root finding failed: {sol.message}")
        return sol.x

class SphericalLinear(SphericalBinner):
    '''Equal step in angular space binning in the non-compact space'''
    def __init__(self, d_s, n_bins, last_cell_size):
        self.d_s = d_s
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i):
        '''
        Calculate the radius limit for the i-th bin.
        '''
        omega = i * self.d_omega
        return self.d_s * np.tan(omega)

    def r_centroid(self, i):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        return self.centroid(r0, r1)

class SphericalConstantVolume(SphericalBinner):
    '''
    Constant volume binning in the non-compact space (constant volume
    in the compact space)
    '''
    def __init__(self, d_s, n_bins, R_sim):
        self.d_s = d_s
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(R_sim/d_s)
        self.unit_bin = (2*self.omega_max - np.sin(2*self.omega_max)) / n_bins

    def r_limit(self, i):
        omega = self.invert_x_minus_sin_x(i * self.unit_bin) / 2  # or 1/4??
        return self.d_s * np.tan(omega)

    def r_centroid(self, i):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        return self.centroid(r0, r1)


class CylindricalBinner(ABC):
    '''TODO'''
    @abstractmethod
    def r_limit(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @abstractmethod
    def r_centroid(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @staticmethod
    def centroid(r0: float, r1: float):
        '''
        Centroid of a cylindrical shell between r0 and r1 (assuming
        uniform height).
        The formula used is:
        .. math::
            r_i = \frac{2}{3} \frac{r_1^3 - r_0^3}{r_1^2 - r_0^2}
        where :math:`r_0` and :math:`r_1` are the lower and upper limits
        of the bin.
        '''
        return 2.0/3.0 * (r1*r1*r1 - r0*r0*r0) / (r1*r1 - r0*r0)

class CylindricalLinear(CylindricalBinner):
    '''TODO'''
    def __init__(self, R_max: float, n_bins: int):
        self.R_max = R_max
        self.n_bins = n_bins
        self.dr = R_max / n_bins

    def r_limit(self, i: int):
        return i * self.dr

    def r_centroid(self, i: int):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        # centroid of a cylindrical shell
        return self.centroid(r0, r1)

class CylindricalConstantVolume(CylindricalBinner):
    '''TODO'''
    def __init__(self, R_max: float, n_bins: int):
        self.R_max = R_max
        self.n_bins = n_bins
        self.dr = (R_max**2) / n_bins

    def r_limit(self, i: int):
        # invert A(r) = pi * r^2  ->  r_i = R_max * sqrt(i/n_bins)
        return np.sqrt(i * self.dr)

    def r_centroid(self, i: int):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        # centroid of a planar annulus
        return self.centroid(r0, r1)