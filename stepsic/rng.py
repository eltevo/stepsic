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

import numpy as np


class RNG:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
    def _get_rng(self, seed=None):
        '''
        Returns a new generator if seed is provided, otherwise returns
        the existing one.
        '''
        return np.random.default_rng(seed) if seed is not None else self.rng
    def uniform(self, size=None, seed=None):
        '''Generate uniformly distributed random numbers.'''
        return self._get_rng(seed).uniform(size=size)
    def normal(self, mean=0.0, std=1.0, size=None, seed=None):
        '''Generate normally distributed random numbers.'''
        return self._get_rng(seed).normal(loc=mean, scale=std, size=size)
    def integers(self, low, high=None, size=None, seed=None):
        '''Generate random integers.'''
        return self._get_rng(seed).integers(low, high=high, size=size)