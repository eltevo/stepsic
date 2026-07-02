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

import numpy as np

from stepsic._typing import Seed

_UINT64_MASK = (1 << 64) - 1
_UINT32_MASK = np.uint64((1 << 32) - 1)

_PHILOX_M0 = 0xD2E7470EE14C6C93
_PHILOX_M1 = 0xCA5A826395121157
_PHILOX_W0 = 0x9E3779B97F4A7C15
_PHILOX_W1 = 0xBB67AE8584CAA73B


def _mulhilo64(x: np.ndarray, multiplier: int) -> tuple[np.ndarray, np.ndarray]:
    '''Return the high and low uint64 words of x * multiplier.'''
    x = np.asarray(x, dtype=np.uint64)
    m = np.uint64(multiplier)
    x_lo = x & _UINT32_MASK
    x_hi = x >> np.uint64(32)
    m_lo = m & _UINT32_MASK
    m_hi = m >> np.uint64(32)

    t = x_lo * m_lo
    w0 = t & _UINT32_MASK
    carry = t >> np.uint64(32)

    t = x_hi * m_lo + carry
    w1 = t & _UINT32_MASK
    w2 = t >> np.uint64(32)

    t = x_lo * m_hi + w1
    lo = (t << np.uint64(32)) | w0
    hi = x_hi * m_hi + w2 + (t >> np.uint64(32))
    return hi.astype(np.uint64, copy=False), lo.astype(np.uint64, copy=False)


def philox4x64_10(counter, key) -> np.ndarray:
    '''Evaluate Philox4x64-10 for one or more 256-bit counters.

    Parameters
    ----------
    counter : array-like of uint64, shape (..., 4)
        Counter words in little-endian order. The returned block for counter
        c is bit-exact with np.random.Philox(counter=c-1) followed by one
        random_raw(4) call, matching NumPy pre-increment convention.
    key : array-like of uint64, shape (2,)
        Two-word Philox key. StePS uses (seed, 0) for white-noise modes.
    '''
    counter = np.asarray(counter, dtype=np.uint64)
    if counter.shape[-1:] != (4,):
        raise ValueError("counter must have shape (..., 4)")
    key = np.asarray(key, dtype=np.uint64)
    if key.shape != (2,):
        raise ValueError("key must have shape (2,)")

    c0 = counter[..., 0].copy()
    c1 = counter[..., 1].copy()
    c2 = counter[..., 2].copy()
    c3 = counter[..., 3].copy()
    k0 = int(key[0])
    k1 = int(key[1])

    for _ in range(10):
        hi0, lo0 = _mulhilo64(c0, _PHILOX_M0)
        hi1, lo1 = _mulhilo64(c2, _PHILOX_M1)
        n0 = hi1 ^ c1 ^ np.uint64(k0)
        n1 = lo1
        n2 = hi0 ^ c3 ^ np.uint64(k1)
        n3 = lo0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = (k0 + _PHILOX_W0) & _UINT64_MASK
        k1 = (k1 + _PHILOX_W1) & _UINT64_MASK

    return np.stack((c0, c1, c2, c3), axis=-1)



class RNG:
    def __init__(self, seed: Seed = None):
        self.rng = np.random.default_rng(seed)
    def _get_rng(self, seed: Seed = None):
        '''
        Returns a new generator if seed is provided, otherwise returns
        the existing one.
        '''
        return np.random.default_rng(seed) if seed is not None else self.rng
    def uniform(self, size=None, seed: Seed = None):
        '''Generate uniformly distributed random numbers.'''
        return self._get_rng(seed).uniform(size=size)
    def normal(self, mean=0.0, std=1.0, size=None, seed: Seed = None):
        '''Generate normally distributed random numbers.'''
        return self._get_rng(seed).normal(loc=mean, scale=std, size=size)
    def integers(self, low, high=None, size=None, seed: Seed = None):
        '''Generate random integers.'''
        return self._get_rng(seed).integers(low, high=high, size=size)