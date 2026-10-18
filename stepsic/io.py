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

import re
from pathlib import Path
from collections.abc import Sequence

import h5py
import numpy as np

from stepsic import __githash__, __programname__, __version__
from stepsic._typing import PathInput

# Gadget IO library for reading Gadget snapshots
# Download from https://www.github.com/masterdesky/glio
try:
    import glio
except ImportError:
    glio = None

import logging

log = logging.getLogger(__name__)


def _resolution_tag(params: dict) -> str:
    """
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
    """
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
    """
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
    """
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
    """
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
    """
    run_dir = Path(params['IC_DIR']) / _run_dirname(params)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'Output directory: {run_dir}')
    return run_dir


class UnsupportedFormatError(RuntimeError):
    """Raised when CosmoIO encounters an unknown file format."""


class CosmoIO:
    """Read snapshot arrays and write HDF5 snapshots.

    Readers handle ASCII tables, HDF5 datasets, and Gadget binaries through
    optional ``glio``. Snapshot data are returned as tuples of arrays.
    """
    @staticmethod
    def load_snapshot(path: Path, *args, **kwargs):
        """Load arrays from the snapshot files matching a path.

        Parameters
        ----------
        path : pathlib.Path
            Path to a snapshot file. Matching numbered parts are collected and
            sorted before dispatch to the reader.
        *args : str
            HDF5 dataset names, in return order. Defaults to ``ParticleIDs``,
            ``Coordinates``, ``Velocities``, and ``Masses``.
        **kwargs
            Reader options. ``part_type`` defaults to 1 (dark matter) for HDF5
            and Gadget. ASCII uses ``dtype`` (default float32); HDF5 retains
            dataset dtypes and accepts ``constant_res`` for mass scaling.

        Returns
        -------
        tuple of ndarray or None
            IDs, positions, velocities, and masses by default. HDF5 returns
            requested datasets in order, with ``None`` for missing datasets.

        Notes
        -----
        ASCII files use columns ID, x, y, z, vx, vy, vz, mass. HDF5 reads
        ``PartType{part_type}`` groups. Other extensions use the optional
        Gadget reader.
        """
        path = path.expanduser().resolve()
        if not path.parent.exists():
            raise FileNotFoundError(path.parent)

        ext = CosmoIO._get_extension(path)
        loader = CosmoIO._find_loader(ext)
        files = CosmoIO._collect_files(path)
        if not files:
            raise FileNotFoundError(f'No files found matching {path.name}.')
        return loader(files, *args, **kwargs)
    
    @staticmethod
    def save_snapshot(path: Path, data: "CosmoData", fmt: str, **kwargs):
        """
        Saves a cosmological snapshot to a file. Currently, only HDF5
        format is supported for writing.

        Parameters
        ----------
        path : pathlib.Path
            Path to the output snapshot file.
        data : stepsic.data.CosmoData
            Particle arrays and snapshot metadata.
        fmt : str
            The file format to save the snapshot in.
        """
        path = path.expanduser().resolve()
        saver = CosmoIO._find_saver(fmt)
        saver(path, data=data, **kwargs)

    @staticmethod
    def get_box_size(path: Path):
        """
        Retrieves the box size from the snapshot file.

        If the header contains a missing, zero, or clearly invalid
        ``BoxSize`` (e.g. a denormalized float), the value is logged
        as a warning and ``None`` is returned so the caller can decide
        how to handle it.

        Parameters
        ----------
        path : pathlib.Path
            Path to the snapshot file.

        Returns
        -------
        boxsize : float or None
            The box size in internal units, or ``None`` if the header
            value is missing or invalid.
        """
        path = path.expanduser().resolve()
        ext = CosmoIO._get_extension(path)
        boxsize = None
        if ext in ['hdf5', 'h5']:
            with h5py.File(path, 'r') as hdf:
                if '/Header' in hdf and 'BoxSize' in hdf['/Header'].attrs:
                    boxsize = float(hdf['/Header'].attrs['BoxSize'])
        elif ext in ['dat', 'txt']:
            raise NotImplementedError('Box size retrieval from ASCII files is not implemented.')
        else:
            if glio is None:
                raise UnsupportedFormatError(
                    f'`{ext}` is an unsupported extension and glio is not installed.'
                )
            s = glio.GadgetSnapshot(path)
            boxsize = float(s.header.BoxSize)

        # Validate: reject zero, negative, subnormal, inf, nan
        if boxsize is not None and (
            not np.isfinite(boxsize) or boxsize <= 0.0 or np.isclose(boxsize, 0.0)
        ):
            log.warning(
                f'Invalid BoxSize={boxsize} in header of {path.name}. '
                f'Treating as unset.'
            )
            boxsize = None
        return boxsize

    @staticmethod
    def get_simulation_radius(path: Path):
        """
        Retrieves the simulation radius from an HDF5 snapshot header.

        StePS and stepsic store the radius of spherical and cylindrical
        volumes as ``SimulationRadius``. Other formats carry no radius.

        Parameters
        ----------
        path : pathlib.Path
            Path to the snapshot file.

        Returns
        -------
        radius : float or None
            The radius in the length unit of the file, or ``None`` if the
            header has no positive, finite ``SimulationRadius``.
        """
        path = path.expanduser().resolve()
        if CosmoIO._get_extension(path) not in ['hdf5', 'h5']:
            return None
        with h5py.File(path, 'r') as hdf:
            header = hdf['/Header'].attrs if '/Header' in hdf else {}
            radius = float(header.get('SimulationRadius', 0.0))
        return radius if np.isfinite(radius) and radius > 0.0 else None


    @staticmethod
    def _match_extension(path: Path):
        """
        Detects the file extension of the given path.

        Parameters
        ----------
        path : pathlib.Path
            The path to the file whose extension is to be detected.

        Returns
        -------
        match : re.Match or None
            A match object if the path matches a valid pattern, or `None` if
            it does not.
        """
        parser_re = re.compile(
            r"^(?P<stem>.*?)"
            r"(?:"
            # Pattern 1: .<index>.<ext> (e.g. ".0.hdf5")
            r"\.(?P<index1>\d+)\.(?P<ext1>[a-zA-Z_][a-zA-Z0-9_]*)"
            r"|"
            # Pattern 2: .<ext>.<index> (e.g. ".hdf5.0")
            r"\.(?P<ext2>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<index2>\d+)"
            r"|"
            # Pattern 3: .<ext> (e.g. ".hdf5")
            r"\.(?P<ext3>[a-zA-Z_][a-zA-Z0-9_]*)"
            r")$"
        )
        return parser_re.match(path.name)
    
    @staticmethod
    def _get_extension(path: Path):
        """
        Returns the file extension of the given path without the leading dot.

        Parameters
        ----------
        path : pathlib.Path
            The path to the file whose extension is to be returned.

        Returns
        -------
        ext : str or None
            The file extension without the leading dot, or `None` if no
            valid extension is found.
        """
        match = CosmoIO._match_extension(path)
        if match:
            parts = match.groupdict()
            return re.escape(parts['ext1'] or parts['ext2'] or parts['ext3'])
        return None

    @staticmethod
    def _collect_files(path: Path):
        """
        Gathers all snapshot files belonging to the same group.

        Parameters
        ----------
        path : pathlib.Path
            The path to any single file in the snapshot set.

        Returns
        -------
        files : list of pathlib.Path
            A sorted list of Path objects for all files in the snapshot.
            Returns an empty list if the filepath does not match a valid
            pattern.
        """
        match = CosmoIO._match_extension(path)

        if match:
            # Regex looking for: \.digits\.ext OR \.ext\.digits OR \.ext
            parts = match.groupdict()
            stem = re.escape(parts['stem'])
            ext = re.escape(parts['ext1'] or parts['ext2'] or parts['ext3'])
            search_pattern = rf'^{stem}(\.\d+\.{ext}|\.{ext}\.\d+|\.{ext})$'
        else:
            # If no extension is provided, treat it as a gadget file
            ext_pattern = re.compile(r"^(?P<stem>.*?)(?:\.(?P<index>\d+))?$")
            match = ext_pattern.match(path.name)
            if not match:
                return [path.name]  # Standalone gadget snapshot
            search_pattern = rf'^{re.escape(match.group("stem"))}(?:\.\d+)?$'
        search_re = re.compile(search_pattern)

        files = []
        for f in path.parent.iterdir():
            if f.is_file() and search_re.match(f.name):
                files.append(f)
        return sorted(files)

    @staticmethod
    def _find_loader(ext: str):
        """Return the backend reader for a file extension."""
        _LOADER_MAP = {
            "dat": lambda f, **kw: CosmoIO._load_ascii(f, **kw),
            "txt": lambda f, **kw: CosmoIO._load_ascii(f, **kw),
            "hdf5": lambda f, *args, **kw: CosmoIO._load_hdf5(f, *args, **kw),
            "h5": lambda f, *args, **kw: CosmoIO._load_hdf5(f, *args, **kw)
        }

        if ext in _LOADER_MAP:
            return _LOADER_MAP[ext]

        # Anything else we treat as Gadget if glio is present
        if glio is None:
            raise UnsupportedFormatError(
                f'`{ext}` is an unsupported extension and glio is not installed.'
            )
        return lambda f, **kw: CosmoIO._load_gadget(f, **kw)

    @staticmethod
    def _find_saver(ext: str):
        """Return the appropriate backend writer for the save method."""
        _SAVER_MAP = {
            "ascii": lambda p, **kw: CosmoIO._save_ascii(p, **kw),
            "gadget": lambda p, **kw: CosmoIO._save_gadget(p, **kw),
            "hdf5": lambda p, **kw: CosmoIO._save_hdf5(p, **kw),
        }

        if ext in _SAVER_MAP:
            return _SAVER_MAP[ext]
        raise UnsupportedFormatError('Only HDF5 files are supported.')

    @staticmethod
    def _load_ascii(files: Sequence[PathInput], **kwargs):
        """Load a cosmological snapshot from an ASCII file."""
        dtype = kwargs.get('dtype', np.float32)
        particleIDs, coordinates, velocities, masses = [], [], [], []
        log.info(f'Reading the input ASCII files ...')
        for path in files:
            log.info(f'Opening ASCII file {path}...')
            data = np.loadtxt(path)
            particleIDs.append(np.array(data[:, 0], dtype=np.uint64))
            coordinates.append(np.array(data[:, 1:4], dtype=dtype))
            velocities.append(np.array(data[:, 4:7], dtype=dtype))
            masses.append(np.array(data[:, 7], dtype=dtype))
        particleIDs = np.concatenate(particleIDs, dtype=np.uint64)
        coordinates = np.concatenate(coordinates, dtype=dtype)
        velocities = np.concatenate(velocities, dtype=dtype)
        masses = np.concatenate(masses, dtype=dtype)
        return particleIDs, coordinates, velocities, masses

    @staticmethod
    def _load_gadget(path: Path, **kwargs):
        """Load a cosmological snapshot from a Gadget binary."""
        part_type = kwargs.get('part_type', 1)
        log.info(f'Reading the input Gadget files ...')
        log.info(f'Opening Gadget file {path}...')
        s = glio.GadgetSnapshot(path)
        particleIDs = s.ID[part_type]
        coordinates = s.pos[part_type]
        velocities = s.vel[part_type]
        masses = s.mass[part_type]
        return particleIDs, coordinates, velocities, masses

    @staticmethod
    def _load_hdf5(files: Sequence[PathInput], *args, **kwargs):
        """Load a cosmological snapshot from an HDF5 file."""
        log.info(f'Reading the input HDF5 files ...')
        part_type = kwargs.get('part_type', 1)
        if not args:
            args = ['ParticleIDs', 'Coordinates', 'Velocities', 'Masses']
        arguments = {ai: [] for ai in args}
        dtypes = {ai: None for ai in args}
        for path in files:
            log.info(f'Opening HDF file {path}...')
            with h5py.File(path, 'r') as hdf:
                for ai in args:
                    g = f'/PartType{part_type}/{ai}'
                    if g not in hdf:
                        log.warning(f'Group {g} not found in HDF5 file {path}.')
                        continue
                    arguments[ai].append(hdf[g][:])
                    dtypes[ai] = hdf[f'/PartType{part_type}/{ai}'].dtype
                if 'Masses' in args and 'Masses' in hdf[f'/PartType{part_type}']:
                    N_part = hdf['/Header'].attrs['NumPart_ThisFile'][part_type]
                    mass_part_type = hdf['/Header'].attrs['MassTable'][part_type]
                    if np.all(arguments['Masses'] == 0):
                        arguments['Masses'] = np.ones(N_part) * mass_part_type
                    if kwargs.get('constant_res', False):
                        arguments['Masses'] *= mass_part_type
        for ai in args:
            if not arguments[ai]:
                arguments[ai] = None
            else:
                arguments[ai] = np.concatenate(arguments[ai], dtype=dtypes[ai])
        return tuple(arguments.values())
    
    @staticmethod
    def _save_ascii(path: Path, data: "CosmoData", **kwargs):
        """Save a cosmological snapshot to an ASCII file."""
        path = path.with_suffix('.dat')
        raise NotImplementedError
    
    @staticmethod
    def _save_gadget(path: Path, data: "CosmoData", **kwargs):
        raise NotImplementedError

    @staticmethod
    def _save_hdf5(path: Path, data: "CosmoData", **kwargs):
        """
        Saves data of a single particle type in GADGET-format HDF5 file.

        Compatible with GADGET, GIZMO and StePS.

        Parameters
        ----------
        path : Path
            The file path to save the snapshot.
        data : stepsic.data.CosmoData
            The cosmological data to save.

        part_type : int
            The particle type to save (e.g. 1 for dark matter in Gadget).
        dtype : numpy.dtype, optional
            Precision for positions, velocities, and masses (default float32).
        **kwargs
            Header values: Redshift, BoxSize, SimulationRadius,
            NumFilesPerSnapshot, Omega0, OmegaLambda, HubbleParam, and
            the Gadget Flag_* fields written below. Numeric values default
            to zero, except NumFilesPerSnapshot, which defaults to one.
        """
        path = path.with_suffix('.hdf5')
        with h5py.File(path, 'w') as hdf_file:
            part_type = kwargs.get('part_type', 1)
            dtype = kwargs.get('dtype', np.float32)

            num_part_array = np.zeros(6, dtype=np.uint32)
            num_part_array[part_type] = data.N_part

            h = hdf_file.create_group("/Header")
            h.attrs['ProgramName'] = __programname__
            h.attrs['ProgramVersion'] = __version__
            h.attrs['ProgramCommitID'] = __githash__
            h.attrs['NumPart_ThisFile'] = num_part_array
            h.attrs['NumPart_Total'] = num_part_array
            h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
            h.attrs['MassTable'] = np.zeros(6, dtype=dtype)
            h.attrs['Time'] = 1.0 / (kwargs.get('Redshift', 0) + 1.0)
            h.attrs['Redshift'] = float(kwargs.get('Redshift', 0.0))
            h.attrs['BoxSize'] = float(kwargs.get('BoxSize', 0.0))  # L_z; used in T^3 and S^1xR^2 simulations
            h.attrs['SimulationRadius'] = float(kwargs.get('SimulationRadius', 0.0))  # R_sim; used in R^1xR^2 and R^3 simulations
            h.attrs['NumFilesPerSnapshot'] = kwargs.get('NumFilesPerSnapshot', 1)
            h.attrs['Omega0'] = float(kwargs.get('Omega0', 0.0))
            h.attrs['OmegaLambda'] = float(kwargs.get('OmegaLambda', 0.0))
            h.attrs['HubbleParam'] = float(kwargs.get('HubbleParam', 0.0))
            h.attrs['Flag_Sfr'] = float(kwargs.get('Flag_Sfr', 0.0))
            h.attrs['Flag_Cooling'] = float(kwargs.get('Flag_Cooling', 0.0))
            h.attrs['Flag_StellarAge'] = float(kwargs.get('Flag_StellarAge', 0.0))
            h.attrs['Flag_Metals'] = float(kwargs.get('Flag_Metals', 0.0))
            h.attrs['Flag_Feedback'] = float(kwargs.get('Flag_Feedback', 0.0))
            h.attrs['Flag_Entropy_ICs'] = float(kwargs.get('Flag_Entropy_ICs', 0.0))

            p = hdf_file.create_group(f"/PartType{part_type}")
            p.create_dataset('ParticleIDs', data=data.id)
            p.create_dataset('Coordinates', data=data.pos, dtype=dtype)
            p.create_dataset('Velocities', data=data.vel, dtype=dtype)
            p.create_dataset('Masses', data=data.mass, dtype=dtype)