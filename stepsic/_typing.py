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

from typing import TypeAlias, Literal

import numpy as np
from numpy.typing import NDArray

Seed: TypeAlias = int | None

# Anything broadcastable to a length-3 int array; e.g. grid shapes
IntVec3: TypeAlias = int | tuple[int, int, int] | NDArray[np.integer]

# Anything broadcastable to a length-3 float array; e.g. boxsize
FloatVec3: TypeAlias = float | tuple[float, float, float] | NDArray[np.floating]

# Anything broadcastable to a length-3 bool array; e.g. periodicity flag
BoolVec3: TypeAlias = bool | tuple[bool, bool, bool] | NDArray[np.bool_]

# Field / particle arrays (shape documented in docstrings)
RealField: TypeAlias = NDArray[np.floating]
ComplexField: TypeAlias = NDArray[np.complexfloating]

# Interpolation/deposit method
MASMethod: TypeAlias = Literal['ngp', 'cic', 'tsc']