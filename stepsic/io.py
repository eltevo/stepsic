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
from typing import List, Dict, Any

import re
import h5py
import numpy as np
from pathlib import Path

# Gadget IO library for reading Gadget snapshots
# Download from https://www.github.com/masterdesky/glio
try:
    import glio
except ImportError as _err:
    glio = None
    # _GLIO_IMPORT_ERROR = _err

import logging
log = logging.getLogger(__name__)


class UnsupportedFormatError(RuntimeError):
    '''Raised when CosmoIO encounters an unknown file format.'''


class CosmoIO:
    '''
    A stateless class for loading cosmological snapshots from various
    formats (ASCII, HDF5, NPY, Gadget) and returning them as a
    structured numpy array.
    '''
    @staticmethod
    def load_snapshot(path: Path, *args, **kwargs):
        '''
        Loads a Gadget-format snapshot of a cosmological simulation from
        either an ASCII, HDF5, NPY or Gadget-format input file.

        Parameters
        ----------
        fname : str
            Path to the *first* snapshot file on disk. All accompanying
            parts (e.g. ``snap_001``, ``snap_002``, etc.) are detected
            and concatenated automatically for Gadget or HDF5 inputs.
        part_type : int
            The particle type to load. For Gadget snapshots, this is the
            part type index (1 for gas, 2 for dark matter, etc.). For HDF5
            snapshots, this is the part type group (e.g. 'PartType1').
        constant_res : bool
            If True, the snapshot is assumed to have constant mass resolution.
        dtype : numpy.dtype
            The data type to use for the snapshot. If not specified, float32 is used.
        '''
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
        '''
        Saves a cosmological snapshot to a file. Currently, only HDF5
        format is supported for writing.

        Parameters
        ----------
        fname : pathlib.Path
            Path to the output snapshot file.
        data : stepsic.CosmoData
            A structured array containing the particle data.
        fmt : str
            The file format to save the snapshot in.
        '''
        path = path.expanduser().resolve()
        saver = CosmoIO._find_saver(fmt)
        saver(path, data=data, **kwargs)

    @staticmethod
    def get_box_size(path: Path):
        '''
        Retrieves the box size from the snapshot file.

        Parameters
        ----------
        path : pathlib.Path
            Path to the snapshot file.

        Returns
        -------
        box_size : float
            The box size in internal units.
        '''
        path = path.expanduser().resolve()
        ext = CosmoIO._get_extension(path)
        if ext in ['hdf5', 'h5']:
            with h5py.File(path, 'r') as hdf:
                box_size = hdf['/Header'].attrs['BoxSize']
            return box_size
        elif ext in ['dat', 'txt']:
            raise NotImplementedError('Box size retrieval from ASCII files is not implemented.')
        else:
            if glio is None:
                raise UnsupportedFormatError(
                    f'`{ext}` is an unsupported extension and glio is not installed.'
                )
            s = glio.GadgetSnapshot(path)
            box_size = s.header.BoxSize
            return box_size
    
    def get_simulation_radius(path: Path):
        '''
        Retrieves the simulation radius from the snapshot file.
        This will be useful for re-scaling spherical snapshots.

        Parameters
        ----------
        path : pathlib.Path
            Path to the snapshot file.

        Returns
        -------
        box_size : float
            The box size in internal units.
        '''
        path = path.expanduser().resolve()
        ext = CosmoIO._get_extension(path)
        if ext in ['hdf5', 'h5']:
            with h5py.File(path, 'r') as hdf:
                sim_radius = hdf['/Header'].attrs['SimulationRadius']
            return sim_radius
        elif ext in ['dat', 'txt']:
            raise NotImplementedError('Box size retrieval from ASCII files is not implemented.')
        else:
            if glio is None:
                raise UnsupportedFormatError(
                    f'`{ext}` is an unsupported extension and glio is not installed.'
                )
            s = glio.GadgetSnapshot(path)
            box_size = s.header.BoxSize
            return box_size / 2.0

    @staticmethod
    def _match_extension(path: Path):
        '''
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
        '''
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
        '''
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
        '''
        match = CosmoIO._match_extension(path)
        if match:
            parts = match.groupdict()
            return re.escape(parts['ext1'] or parts['ext2'] or parts['ext3'])
        return None

    @staticmethod
    def _collect_files(path: Path):
        '''
        Gathers all snapshot files belonging to the same group.

        Parameters
        ----------
        path : pathlib.Path
            The path to any single file in the snapshot set.

        Returns
        -------
        files : List[pathlib.Path]
            A sorted list of Path objects for all files in the snapshot.
            Returns an empty list if the filepath does not match a valid
            pattern.
        '''
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
        '''Return the appropriate backend reader for ``path``.'''
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
        '''Return the appropriate backend writer for the save method.'''
        _SAVER_MAP = {
            "ascii": lambda p, **kw: CosmoIO._save_ascii(p, **kw),
            "gadget": lambda p, **kw: CosmoIO._save_gadget(p, **kw),
            "hdf5": lambda p, **kw: CosmoIO._save_hdf5(p, **kw),
        }

        if ext in _SAVER_MAP:
            return _SAVER_MAP[ext]
        raise UnsupportedFormatError('Only HDF5 files are supported.')

    @staticmethod
    def _load_ascii(files: List[Path], **kwargs):
        '''Load a cosmological snapshot from an ASCII file.'''
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
        '''Load a cosmological snapshot from a Gadget binary.'''
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
    def _load_hdf5(files: List[Path], *args, **kwargs):
        '''Load a cosmological snapshot from an HDF5 file.'''
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
        return arguments.values()
    
    @staticmethod
    def _save_ascii(path: Path, data: "CosmoData", **kwargs):
        path = path.with_suffix('.dat')
        '''Save a cosmological snapshot to an ASCII file.'''
        raise NotImplementedError
    
    @staticmethod
    def _save_gadget(path: Path, data: "CosmoData", **kwargs):
        raise NotImplementedError

    @staticmethod
    def _save_hdf5(path: Path, data: "CosmoData", **kwargs):
        '''
        Saves data of a single particle type in GADGET-format HDF5 file.

        Compatible with GADGET, GIZMO and StePS.

        Parameters
        ----------
        path : Path
            The file path to save the snapshot.
        data : stepsic.CosmoData
            The cosmological data to save.

        .. Optional Parameters:
        *header arguments*
            Additional parameters for the snapshot header. TODO.
        part_type : int
            The particle type to save (e.g. 1 for dark matter in Gadget).
        dtype : numpy.dtype
            The data type to use for the snapshot positions, velocities
            and masses.
        '''
        path = path.with_suffix('.hdf5')
        with h5py.File(path, 'w') as hdf_file:
            part_type = kwargs.get('part_type', 1)
            dtype = kwargs.get('dtype', np.float32)

            num_part_array = np.zeros(6, dtype=np.uint32)
            num_part_array[part_type] = data.N_part

            h = hdf_file.create_group("/Header")
            h.attrs['NumPart_ThisFile'] = num_part_array
            h.attrs['NumPart_Total'] = num_part_array
            h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
            h.attrs['MassTable'] = np.zeros(6, dtype=dtype)
            h.attrs['Time'] = 1.0 / (kwargs.get('Redshift', 0) + 1.0)
            h.attrs['Redshift'] = float(kwargs.get('Redshift', 0.0))
            h.attrs['BoxSize'] = float(kwargs.get('BoxSize', 0.0)) # only Lz is stored to have compatibility with both T^3 and S^1xR^2 simulations
            h.attrs['SimulationRadius'] = float(kwargs.get('SimulationRadius', 0.0)) # Rsim, only relevant for R^1xR^2 and R^3 simulations
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
