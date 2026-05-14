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

from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

Seed: TypeAlias = int | np.integer | None
PathInput: TypeAlias = str | Path | PathLike[str]

# Broadcastable to length-3 inputs
IntVec3: TypeAlias = int | tuple[int, int, int] | list[int] | NDArray[np.integer]
FloatVec3: TypeAlias = float | tuple[float, float, float] | list[float] | NDArray[np.floating]
BoolVec3: TypeAlias = bool | tuple[bool, bool, bool] | list[bool] | NDArray[np.bool_]

RealField: TypeAlias = NDArray[np.floating]
ComplexField: TypeAlias = NDArray[np.complexfloating]

MASMethod: TypeAlias = Literal['ngp', 'cic', 'tsc']
GeometryName: TypeAlias = Literal['cubical', 'cylindrical', 'spherical']
SpectrumKind: TypeAlias = Literal['camb', 'input']
ICType: TypeAlias = Literal['grid', 'random', 'shell', 'glass']