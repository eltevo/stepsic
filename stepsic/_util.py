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

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _resolution_tag(params: dict) -> str:
    '''
    Build a compact resolution descriptor for the directory name.

    The tag depends on the IC generation mode:

    - ``grid``:  ``Ng{NGRID}`` — mesh drives particle count
    - ``random``:  ``Np{NPART}`` — explicit particle count
    - ``glass``:  ``Nm{NMESH}`` or ``Ng{NMESHSAMPLES}`` for multiscale
    - ``shell``:  ``Nsh{NSHELL}_Nr{NRBINS}`` — shell structure

    Parameters
    ----------
    params : dict
        Full parameter dictionary.

    Returns
    -------
    str
        Resolution tag without leading underscore.
    '''
    ic_type = params['TYPE']
    parts: list[str] = []

    if ic_type == 'grid':
        parts.append(f'Ng{params["NGRID"]}')

    elif ic_type == 'random':
        parts.append(f'Np{params["NPART"]}')

    elif ic_type == 'glass':
        pass

    elif ic_type == 'shell':
        parts.append(f'Nsh{params["NSHELL"]}')
        parts.append(f'Nr{params["NRBINS"]}')

    # Add FFT grid size or multigrid sample count at the end
    if params["NMESH"] > 0:
        parts.append(f'Nm{params["NMESH"]}')
    else:
        parts.append(f'Ng{params["NMESHSAMPLES"]}')

    return '_'.join(parts)


def _run_dirname(params: dict) -> str:
    '''
    Construct a run-specific directory name from simulation parameters.

    Encodes geometry, resolution, redshift, and numerical method so
    that the directory name uniquely identifies the run at a glance.

    Parameters
    ----------
    params : dict
        Dictionary containing the ``stepsic`` simulation parameters.

    Returns
    -------
    str
        The generated directory name (no path separators).
    '''
    parts = [params['IC_PREFIX']]

    parts.append(params['GEOMETRY'])
    if params['LPTORDER'] == 0:
        parts.append('preglass')

    if params['GEOMETRY'] == 'cubical':
        parts.append('Lx{}_Ly{}_Lz{}'.format(*map(int, params['LBOX'])))
    elif params['GEOMETRY'] in ['spherical', 'cylindrical']:
        parts.append(f'R3D{params["R_3D"]:.0f}_D4D{params["D_4D"]:.0f}')
        parts.append(f'Lz{params["LBOX"][2]:.0f}')

    parts.append(_resolution_tag(params))

    if params['LPTORDER'] > 0:
        parts.append(f'z{params["REDSHIFT"]:.0f}')
        parts.append(f'LPT{params["LPTORDER"]}')
        parts.append(f'{params["INTERPOLATION"]}')

    return '_'.join(parts)


def ensure_run_dir(params: dict) -> Path:
    '''
    Build, create, and return the full output directory for this IC run.

    Constructs the path ``{IC_DIR}/{run_name}/`` where ``run_name`` is
    derived from the simulation parameters via :func:`_run_dirname`,
    then creates the entire tree if it does not already exist.

    Parameters
    ----------
    params : dict
        Full stepsic parameter dictionary.  Must contain at least
        ``IC_DIR`` and all keys required by :func:`_run_dirname`.

    Returns
    -------
    pathlib.Path
        The resolved run directory (guaranteed to exist on return).
    '''
    run_dir = Path(params['IC_DIR']) / _run_dirname(params)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'Output directory: {run_dir}')
    return run_dir