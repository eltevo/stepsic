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

import astropy.units as u
import numpy as np
from astropy.units import cds

# Gravitational constant in cm^3 g^-1 s^-2
G = 1 * cds.G

# This IC generator uses the same internal units as the StePS simulator code
UNIT_L = (1 * u.Mpc).to(u.cm)                  # Unit distance (1 Mpc in cm)
UNIT_M = (1e11 * u.M_sun).to(u.g)              # Unit mass (1e11 Msol in g)
UNIT_T = np.sqrt(UNIT_L**3 / G / UNIT_M).to(u.Gyr)  # Unit time
UNIT_V = UNIT_L.to(u.km) / UNIT_T.to(u.s)      # Unit velocity

# Float values for the units
G = G.value
UNIT_L = UNIT_L.value
UNIT_T = UNIT_T.value
UNIT_V = UNIT_V.value
UNIT_M = UNIT_M.value