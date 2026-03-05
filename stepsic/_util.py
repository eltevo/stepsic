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


def _resolution_tag(params: dict, *, n_part: int | None = None) -> str:
    '''
    Build a compact resolution descriptor for the filename.

    The tag depends on the IC generation mode:

    - ``grid``:      ``Nm{NMESH}`` — mesh drives particle count
    - ``random``:    ``Np{NPART}`` — explicit particle count
    - ``glass``:     ``Nm{NMESH}`` or ``Ng{NGRIDSAMPLES}`` for multiscale
    - ``shell``:     ``Nsh{NSHELL}_Nr{NRBINS}`` — shell structure

    Parameters
    ----------
    params : dict
        Full parameter dictionary.
    n_part : int or None
        If provided (e.g. from a loaded glass), appended as ``Np{n_part}``.

    Returns
    -------
    str
        Resolution tag without leading underscore.
    '''
    ic_type = params['TYPE']
    nmesh = params.get('NMESH', 0)
    parts: list[str] = []

    if ic_type == 'grid':
        parts.append(f'Nm{nmesh}')

    elif ic_type == 'random':
        parts.append(f'Np{params["NPART"]}')

    elif ic_type == 'glass':
        if nmesh > 0:
            parts.append(f'Nm{nmesh}')
        else:
            if n_part is not None:
                parts.append(f'Np{n_part}')
            parts.append(f'Ng{params["NGRIDSAMPLES"]}')

    elif ic_type == 'shell':
        parts.append(f'Nsh{params["NSHELL"]}')
        parts.append(f'Nr{params["NRBINS"]}')
        if nmesh > 0:
            parts.append(f'Nm{nmesh}')

    return '_'.join(parts)


def create_filename(params: dict, *, n_part: int | None = None) -> str:
    '''
    Construct a filename for the output IC based on its parameters.

    The filename encodes geometry, resolution, redshift, and numerical
    method in a way that uniquely identifies the run at a glance.

    Parameters
    ----------
    params : dict
        Dictionary containing the ``stepsic`` simulation parameters.
    n_part : int or None
        Actual particle count from the loaded snapshot. If provided,
        included in the resolution tag for glass/shell ICs.

    Returns
    -------
    str
        The generated filename (without extension).
    '''
    parts = [params['IC_PREFIX']]

    if params['LPTORDER'] == 0:
        parts.append('preglass')

    parts.append('Lx{}_Ly{}_Lz{}'.format(*map(int, params['LBOX'])))
    parts.append(f'R3D{params["R_3D"]:.0f}_D4D{params["D_4D"]:.0f}')

    parts.append(_resolution_tag(params, n_part=n_part))

    if params['LPTORDER'] > 0:
        parts.append(f'z{params["REDSHIFT"]:.0f}')
        parts.append(f'LPT{params["LPTORDER"]}')
        parts.append(f'INT{params["INTERPOLATION"]}')

    return '_'.join(parts)