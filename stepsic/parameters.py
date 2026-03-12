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
import importlib.resources

import sys  # only used for dataset "slots" argument compatibility
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import toml

import logging

from stepsic.units import UNIT_V
log = logging.getLogger(__name__)


class PType(Enum):
    '''
    Parameter storage types.

    Each variant maps to a specific validation and casting rule.
    '''
    SCALAR = 'scalar'          # Cast to dtype (default: float64)
    INT = 'int'                # Cast to int
    BOOL = 'bool'              # Must be bool
    STRING = 'string'          # Stripped and lowered
    ARRAY = 'array'            # Broadcast to shape-(length,) with given dtype
    PATH = 'path'              # Must be a valid filesystem path
    PATH_MKDIR = 'path_mkdir'  # Path that will be created if missing


@dataclass(frozen=True, **({"slots": True} if sys.version_info >= (3, 10) else {}))
class Param:
    r'''
    Declarative descriptor for a single configuration parameter.

    Parameters
    ----------
    key : str
        Internal dictionary key that matches keys in config.
    ptype : PType
        Storage type, controls validation and casting.
    label : str or None
        Human-readable label for display.  If ``None``, the parameter
        is validated but not printed.
    fmt : str or None
        Format spec applied to the value when printing, e.g., ``'.6f'``.
        Ignored when ``formatter`` is set.
    group : str
        Display group name (e.g., ``'cosmo'``, ``'ic'``).
    unit : str
        Physical unit string appended after the value (e.g., ``'eV'``).
    choices : tuple of str or None
        For STRING parameters: set of valid values.
    array_length : int
        For ARRAY parameters: number of elements the value is broadcast
        to.  Ignored for non-ARRAY types.  Default is 3 (spatial dims).
    array_dtype : type
        For ARRAY parameters: NumPy dtype to cast elements to.
        Default is ``np.float64``.  Use ``bool`` for boolean arrays.
    h_scaled : bool
        If ``True`` and ``HINDEPENDENT`` is False, the value is
        multiplied by ``h`` during processing.
    h_display : bool
        If ``True``, use the dual "X unit = Y unit/h" display format.
    h_precision : int
        Precision passed to the dual h-display formatter.
    condition : callable or None
        A ``Callable[[dict], bool]`` predicate.  The parameter is only
        validated and displayed when the predicate returns ``True``.
        Receives the full parameter dict.
    formatter : callable or None
        Optional ``Callable[[Any, dict], str]`` to fully override
        display formatting.  Receives ``(value, params_dict)``.
    '''
    key: str
    ptype: PType = PType.SCALAR
    label: str | None = None
    fmt: str | None = None
    unit: str = ""
    group: str = "ic"
    choices: tuple[str, ...] | None = None
    array_length: int = 3
    array_dtype: type = np.float64
    h_scaled: bool = False
    h_display: bool = False
    h_precision: int = 2
    condition: Callable[[dict], bool] | None = None
    formatter: Callable[[Any, dict], str] | None = None


@dataclass(frozen=True, **({"slots": True} if sys.version_info >= (3, 10) else {}))
class Constraint:
    r'''
    A cross-parameter validation rule.

    A ``Constraint`` enforces compatibility between two or more already
    validated parameters.

    Parameters
    ----------
    check : Callable[[dict], bool]
        Predicate that returns ``True`` when the constraint is
        satisfied.  Receives the full parameter dict.
    message : Callable[[dict], str]
        Callable that produces the error message when the constraint
        is violated.  Receives the full parameter dict so it can
        interpolate actual values into the message.
    level : str
        - ``'error'`` (default) raises ``ValueError``
        - ``'warning'`` broadcasts a log warning and continues.
    '''
    check: Callable[[dict], bool]
    message: Callable[[dict], str]
    level: str = "error"


# Parameter table - cosmological parameters
COSMO_PARAMS: tuple[Param, ...] = (
    Param('H0',      label='H0',       fmt=".3f", group='cosmo', unit='km s^-1 Mpc^-1'),
    Param('OMEGA_M', label='Omega_m',  fmt=".6f", group='cosmo'),
    Param('OMEGA_B', label='Omega_b',  fmt=".6f", group='cosmo'),
    Param('OMEGA_L', label='Omega_L',  fmt=".6f", group='cosmo'),
    Param('NS',      label='n_s',      fmt=".6f", group='cosmo'),
    Param('AS',      label='A_s',      fmt=".6e", group='cosmo'),
    Param('SIGMA8',  label='sigma_8',  fmt=".3f", group='cosmo'),
    Param('YHE',     label='Y_He',     fmt=".6f", group='cosmo'),
    Param('MNU',     label='M_nu',     fmt=".3f", group='cosmo', unit='eV'),
    Param('NNU',     label='N_nu',     fmt=".3f", group='cosmo'),
    Param('ZREI',    label='z_reion.', fmt=".3f", group='cosmo'),
    Param('TCMB',    label='T_CMB',    fmt=".3f", group='cosmo', unit='K'),
    Param('KPIVOT',  label='k_pivot',  fmt=".3f", group='cosmo', unit='Mpc^-1'),
    Param('W0',      label='w0',       fmt=".3f", group='cosmo'),
    Param('WA',      label='wa',       fmt=".3f", group='cosmo'),
)

COSMO_DERIVED: tuple[Param, ...] = (
    Param('H',        label='h',           fmt=".6f", group='cosmo'),
    Param('RHO_CRIT', label='rho_crit',    fmt=".3e", group='cosmo', unit='(internal units)'),
    Param('RHO_MEAN', label='rho_mean',    fmt=".3e", group='cosmo', unit='(internal units)'),
    Param('OMMH2',    label='Omega_m h^2', fmt=".6f", group='cosmo'),
    Param('OMCH2',    label='Omega_c h^2', fmt=".6f", group='cosmo'),
    Param('OMBH2',    label='Omega_b h^2', fmt=".6f", group='cosmo'),
    Param('OMEGA_K',  label='Omega_k',     fmt=".6f", group='cosmo'),
    Param('OMEGA_NU', label='Omega_nu',    fmt=".6f", group='cosmo'),
    Param('OMEGA_C',  label='Omega_c',     fmt=".6f", group='cosmo'),
)

#  The display order for the cosmological parameter block.
#
#  This is separate from COSMO_PARAMS because the display order mixes
#  base and derived quantities (e.g., h comes right after H0, OMMH2
#  right after OMEGA_M, etc.).  Keeping a flat display list avoids
#  contorting the validation tables just to get the log output pretty.
_COSMO_DISPLAY_KEYS: tuple[str, ...] = (
    'H0', 'H',
    'RHO_CRIT', 'RHO_MEAN',
    'OMEGA_M', 'OMMH2',
    'OMEGA_C', 'OMCH2',
    'OMEGA_L', 'OMEGA_K',
    'OMEGA_B', 'OMBH2',
    'NS', 'AS', 'SIGMA8',
    'OMEGA_NU', 'MNU', 'NNU',
    'YHE', 'ZREI', 'TCMB', 'KPIVOT',
    'W0', 'WA',
)

_COSMO_LOOKUP: dict[str, Param] = {
    p.key: p for p in (*COSMO_PARAMS, *COSMO_DERIVED)
}

_COSMO_DISPLAY: tuple[Param, ...] = tuple(
    _COSMO_LOOKUP[k] for k in _COSMO_DISPLAY_KEYS
)

# Parameter table - IC parameters
IC_PARAMS: tuple[Param, ...] = (
    # -- Geometry, box, and scale --------------------------------------
    Param('GEOMETRY', ptype=PType.STRING, label="Geometry", choices=('cylindrical', 'spherical', 'cubical')),
    Param('LBOX', ptype=PType.ARRAY, label="Box size [X, Y, Z]", h_scaled=True, h_display=True),
    Param('PERIODIC', ptype=PType.ARRAY, label="Periodicity [X, Y, Z]", array_dtype=bool),
    Param('REDSHIFT',                  label="Target redshift", fmt=".2f"),
    Param('LPTORDER', ptype=PType.INT, label="LPT order", choices=(0, 1, 2)),
    Param('COI', ptype=PType.ARRAY, label="Center of Interest [X, Y, Z]", h_scaled=True, h_display=True),

    # -- IC type and generation ----------------------------------------
    Param('TYPE', ptype=PType.STRING, label="IC type", choices=('grid', 'random', 'shell', 'glass')),
    Param('NMESH', ptype=PType.INT, label="FFT mesh size", unit="voxels"),
    Param('NGRID', ptype=PType.INT, label="Grid size", unit="voxels"),
    Param('NPART', ptype=PType.INT, label="N particles (random)", condition=lambda P: P.get('TYPE') == 'random'),
    Param('NSHELL', ptype=PType.INT, label="Particles per shell", condition=lambda P: P.get('TYPE') == 'shell'),
    Param('NMESHSAMPLES', ptype=PType.INT, label="Grid samples", condition=lambda P: P.get('TYPE') == 'glass'),
    Param('INTERPOLATION', ptype=PType.STRING, label="Interpolation", choices=('ngp', 'cic', 'tsc'), condition=lambda P: P.get('LPTORDER') > 0),
    Param('COMPENSATE', ptype=PType.BOOL, label="Compensation kernel", condition=lambda P: P.get('LPTORDER') > 0),
    Param('SPHEREMODE', ptype=PType.BOOL, label="Sphere mode", condition=lambda P: P.get('LPTORDER') > 0),
    Param('COMOVING', ptype=PType.BOOL, label="Comoving IC"),
    Param('COUNTER', ptype=PType.BOOL, label="Counter phase", condition=lambda P: P.get('LPTORDER') > 0),
    Param('PHASE_SHIFT', label="Phase shift",  fmt=".2f", unit="degrees", condition=lambda P: P.get('LPTORDER') > 0),
    Param('HINDEPENDENT', ptype=PType.BOOL, label="H-independent units"),
    Param('SEED', ptype=PType.INT, label="Random seed"),

    # -- Input/output files and format ---------------------------------
    Param('INPUT_GLASS', ptype=PType.PATH, label="Glass input file", condition=lambda P: P.get('TYPE') == 'glass'),
    # TODO: Param('INPUT_WHITE_NOISE', ptype=PType.PATH, label="White noise input file"),
    # TODO: Param('INPUT_DELTA_K', ptype=PType.PATH, label="Delta(k) input file"),
    Param('IC_DIR', ptype=PType.PATH_MKDIR, label="IC output directory"),
    Param('IC_PREFIX', ptype=PType.STRING, label="IC name prefix"),
    Param('IC_FORMAT', ptype=PType.STRING, label="Output format", choices=('ascii', 'gadget', 'hdf5')),
    Param('USE_DOUBLE', ptype=PType.BOOL, label="Double precision"),

    # -- Stereographic projection --------------------------------------
    Param('R_3D', label="Euclidean sim. radius", h_scaled=True, h_display=True, h_precision=4),
    Param('D_4D', label="Compact. sim. diameter", h_scaled=True, h_display=True, h_precision=4),
    Param('BIN_MODE', ptype=PType.STRING, label="Binning mode", choices=('omega', 'volume'), condition=lambda P: P.get('TYPE') == 'shell'),
    Param('NRBINS', ptype=PType.INT, label="Radial bins", condition=lambda P: P.get('TYPE') == 'shell'),

    # -- Rotation ------------------------------------------------------
    Param('ROTATE', label="Rotation", fmt=".4f", unit="rad/Gyr"),

    # -- Cosmology default ---------------------------------------------
    # The "choices" here are checked at load time against cosmology.toml,
    # so no need to define them here.
    Param('COSMOLOGY', ptype=PType.STRING, label="Cosmology set"),

    # -- Power spectrum ------------------------------------------------
    Param('SPECTRUM', ptype=PType.STRING, label="Spectrum type", choices=('input', 'camb')),
    Param('NONLINEAR', ptype=PType.BOOL, label="Use nonlinear spectrum", condition=lambda P: P.get('SPECTRUM') == 'camb'),
    Param('HALOFIT', ptype=PType.STRING, label="Halofit version", condition=lambda P: P.get('SPECTRUM') == 'camb' and P.get('NONLINEAR', False)),
    Param('INPUT_SPECTRUM', ptype=PType.PATH, label="Input spectrum file", condition=lambda P: P.get('SPECTRUM') == 'input'),
    Param('INPUT_SPECTRUM_UNIT_L_IN_CM', label=None, condition=lambda P: P.get('SPECTRUM') == 'input'),

    # -- Internal unit system for validation ---------------------------
    Param('UNIT_L_IN_CM', label=None),
    Param('UNIT_M_IN_G', label=None),
    Param('UNIT_V_IN_KMPS', label=None),
)

IC_DERIVED: tuple[Param, ...] = (
    Param('SCALE', label="Target scale factor", fmt=".6f"),
)


# Cross-parameter constraints
#
#  Each entry enforces a compatibility rule between two or more parameters.
#  The ``check`` predicate must return True when the constraint is met.
#  The ``message`` callable produces a human-readable error/warning when
#  the constraint is violated.

IC_CONSTRAINTS: tuple[Constraint, ...] = (
    # -- TYPE / GEOMETRY compatibility ---------------------------------
    Constraint(
        check=lambda P: P.get('TYPE') != 'shell' or P.get('GEOMETRY') != 'cubical',
        message=lambda P: (
            "TYPE='shell' is not valid for cubical geometry. "
            "Use TYPE='grid' or TYPE='random' instead."
        ),
        level='error',
    ),
    Constraint(
        check=lambda P: P.get('TYPE') != 'grid' or P.get('GEOMETRY') == 'cubical',
        message=lambda P: (
            f"TYPE='grid' is only valid for cubical geometry, "
            f"got GEOMETRY='{P['GEOMETRY']}'. "
            f"Use TYPE='shell' for cylindrical/spherical geometries."
        ),
        level='error',
    ),
    Constraint(
        check=lambda P: P.get('TYPE') != 'random' or P.get('GEOMETRY') == 'cubical',
        message=lambda P: (
            f"TYPE='random' is only valid for cubical geometry, "
            f"got GEOMETRY='{P['GEOMETRY']}'. "
            f"Use TYPE='shell' for cylindrical/spherical geometries."
        ),
        level='error',
    ),

    # -- NMESH / TYPE compatibility ------------------------------------
    Constraint(  # Include shells mode too if ever implement glass generation
        check=lambda P: P.get('NMESH', 1) != 0 or P.get('TYPE') == 'glass',
        message=lambda P: (
            f"NMESH=0 (variable-resolution mode) requires TYPE='glass'. "
            f"Got TYPE='{P['TYPE']}'."
        ),
        level='error',
    ),

    # -- Sanity checks -------------------------------------------------
    Constraint(
        check=lambda P: P.get('REDSHIFT', 0) >= 0,
        message=lambda P: (
            f"REDSHIFT={P['REDSHIFT']:.2f} is negative. "
            f"Redshift must be >= 0."
        ),
        level='error',
    ),
    Constraint(
        check=lambda P: P.get('NMESH') >= P.get('NGRID'),
        message=lambda P: (
            f"NMESH ({P['NMESH']}) < NGRID ({P['NGRID']}). "
            f"The FFT grid should be >= the particle grid to avoid aliasing."
        ),
        level="warning",
    ),

    # -- Non-fatal warnings --------------------------------------------
    Constraint(
        check=lambda P: P.get('COMOVING', True) or P.get('LPTORDER', 1) > 0,
        message=lambda P: (
            "COMOVING=false has no effect for LPTORDER=0 (glass-making mode). "
            "Output will be written in comoving coordinates."
        ),
        level="warning",
    ),
)


def _validate_constraints(
    constraints: Sequence[Constraint], P: dict,
) -> None:
    '''
    Run all cross-parameter constraints against the parameter dict.

    Parameters
    ----------
    constraints : sequence of Constraint
        The constraint rules to check.
    P : dict
        The full parameter dictionary.

    Raises
    ------
    ValueError
        On the first ``level='error'`` constraint that fails.
    '''
    for c in constraints:
        if not c.check(P):
            if c.level == "error":
                raise ValueError(c.message(P))
            log.warning(c.message(P))


def _fmt_arr(x: object, *, precision: int = 2) -> str:
    '''Pretty fixed-width formatting for scalars or small arrays.'''
    return np.array2string(np.asarray(x), precision=precision, floatmode='fixed')


def _fmt_with_h(
    value: object, h: float, unit: str, *, hindependent: bool, precision: int = 2,
) -> str:
    r'''
    Format a value with optional dual h-dependent / h-independent view.

    When ``HINDEPENDENT`` is True the value is already in ``unit``
    (typically Mpc/h) and is printed as-is.  When False the internal
    value is in ``unit`` (Mpc) but was multiplied by ``h``, so both the
    physical and ``h``-scaled representations are shown.
    '''
    if hindependent:
        return f'{_fmt_arr(value, precision=precision)} {unit}'
    left = _fmt_arr(np.asarray(value) / h, precision=precision)
    right = _fmt_arr(value, precision=precision)
    return f"{left} {unit} = {right} {unit}/h"


def _format_value(param: Param, value: Any, P: dict) -> str:
    '''
    Produce a display string for a single parameter, respecting
    all formatting rules encoded in the ``Param`` descriptor.
    '''
    # Custom override
    if param.formatter is not None:
        return param.formatter(value, P)

    # Dual h-display
    if param.h_display:
        h = float(P['H'])
        unit = 'Mpc/h' if P['HINDEPENDENT'] else 'Mpc'
        return _fmt_with_h(
            value, h, unit,
            hindependent=P['HINDEPENDENT'],
            precision=param.h_precision,
        )

    # Bool / string / path
    if param.ptype in (PType.BOOL, PType.STRING, PType.PATH, PType.PATH_MKDIR):
        return str(value)

    # Array types
    if param.ptype is PType.ARRAY:
        s = _fmt_arr(value)
        return f'{s} {param.unit}'.strip()

    # Scalar with format spec
    if param.fmt is not None:
        s = f'{value:{param.fmt}}'
    elif param.ptype is PType.INT:
        s = f'{value:d}'
    else:
        s = str(value)
    return f'{s} {param.unit}'.strip()


def _render_kv_block(title: str, rows: list[tuple[str, str]]) -> str:
    '''Render a titled key/value block with aligned colon columns.'''
    if not rows:
        return title
    width = max(len(k) for k, _ in rows)
    lines = [title, '-' * len(title)]
    lines += [f'{k:<{width}}:  {v}' for k, v in rows]
    return '\n'.join(lines)


def _validate_and_cast(param: Param, P: dict) -> None:
    '''
    Validate and cast a single parameter in-place inside ``P``.

    Raises
    ------
    KeyError
        If the parameter is required but missing.
    TypeError
        If the raw value has the wrong Python type.
    ValueError
        If a string parameter has an invalid choice.
    FileNotFoundError
        If a path parameter points to a non-existent location.
    '''
    if param.condition is not None and not param.condition(P):
        return  # Skip conditional params whose predicate is False

    if param.key not in P:
        # Parameters with conditions that passed but key is missing
        # are truly required and should raise an error.
        raise KeyError(f"'{param.key}' is not defined in the parameters!")

    value = P[param.key]

    if param.ptype is PType.SCALAR:
        if not np.isscalar(value):
            raise TypeError(f"'{param.key}' must be a scalar!")
        P[param.key] = np.float64(value)

    elif param.ptype is PType.INT:
        if not np.isscalar(value):
            raise TypeError(f"'{param.key}' must be a scalar integer!")
        P[param.key] = int(value)

    elif param.ptype is PType.BOOL:
        if not isinstance(value, bool):
            raise TypeError(f"'{param.key}' must be a boolean!")

    elif param.ptype is PType.STRING:
        if not isinstance(value, str):
            raise TypeError(f"'{param.key}' must be a string!")
        value = value.strip().lower()
        P[param.key] = value
        if param.choices and value not in param.choices:
            raise ValueError(
                f"Invalid value for '{param.key}': '{value}'. "
                f"Must be one of {param.choices}."
            )

    elif param.ptype is PType.ARRAY:
        arr = np.asarray(value)
        n = param.array_length
        dt = param.array_dtype
        if not (np.isscalar(value) or arr.size == n):
            raise TypeError(
                f"'{param.key}' must be a scalar or length-{n} array!"
            )
        if np.issubdtype(dt, np.floating) and not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"'{param.key}' must be numeric!")
        P[param.key] = np.broadcast_to(value, (n,)).astype(dt)

    elif param.ptype in (PType.PATH, PType.PATH_MKDIR):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"'{param.key}' must be a string or Path!")
        path = Path(value)
        if not path.exists():
            if param.ptype is PType.PATH_MKDIR:
                path.mkdir(parents=True)
            else:
                raise FileNotFoundError(
                    f"'{param.key}' does not exist: {path}"
                )

    # h-scaling when HINDEPENDENT is False
    if param.h_scaled and not P.get("HINDEPENDENT", True):
        P[param.key] = np.asarray(P[param.key]) * P["H"]


def _validate_group(params: Sequence[Param], P: dict) -> None:
    '''Validate and cast every parameter in a group.'''
    for p in params:
        _validate_and_cast(p, P)


def _compute_derived_cosmo(P: dict) -> None:
    '''Compute derived cosmological quantities from base parameters.'''
    h = P['H0'] / 100.0
    P["H"] = h
    P["RHO_CRIT"] = 3 * (P['H0'] / 100.0)**2 / (8*np.pi) / UNIT_V / UNIT_V
    P["RHO_CRIT"] /= P['H']**2  # Since H0 is in km/s/Mpc instead of km/s/(Mpc/h)
    P["RHO_MEAN"] = P["OMEGA_M"] * P["RHO_CRIT"]
    P["OMMH2"] = P["OMEGA_M"] * h**2
    P["OMBH2"] = P["OMEGA_B"] * h**2
    P["OMEGA_NU"] = P.get("MNU", 0.06) / 93.14 / h**2
    P["OMEGA_C"] = P["OMEGA_M"] - P["OMEGA_B"] - P["OMEGA_NU"]
    P["OMCH2"] = P["OMEGA_C"] * h**2
    P["OMEGA_K"] = 1.0 - P["OMEGA_M"] - P["OMEGA_L"]


def _compute_derived_ic(P: dict) -> None:
    '''Compute derived IC quantities from base parameters.'''
    P["SCALE"] = 1.0 / (P["REDSHIFT"] + 1.0)
    P["DTYPE"] = np.float64 if P["USE_DOUBLE"] else np.float32


def _display_params(title: str, params: Sequence[Param], P: dict) -> None:
    '''
    Format and log a group of parameters as an aligned key/value block.

    Parameters
    ----------
    title : str
        Section heading printed above the parameter block.
    params : sequence of Param
        Ordered parameter descriptors to display. The iteration order
        determines the display order, so callers can pass a custom
        sequence to control the order without changing the original
        tables.
    P : dict
        The full parameter dictionary (values + metadata).
    '''
    rows: list[tuple[str, str]] = []
    for p in params:
        if p.label is None or p.key not in P:
            continue
        if p.condition is not None and not p.condition(P):
            continue
        rows.append((p.label, _format_value(p, P[p.key], P)))
    if rows:
        log.info('\n' + _render_kv_block(title, rows))

class CosmoParameters:
    '''
    Reads, validates, casts, derives, and pretty-prints the full set of
    parameters needed by ``stepsic``.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Notes
    -----
    Every parameter is declared exactly once in the module-level
    ``COSMO_PARAMS``, ``IC_PARAMS``, ``COSMO_DERIVED`` or ``IC_DERIVED``
    tuples.  Adding a new parameter is a one-liner: append a ``Param``
    entry to the appropriate table and, if it's derived, add its
    computation in ``_compute_derived_*``.
    '''

    def __init__(self, path: Path) -> None:
        with open(path, 'r') as f:
            self.P: dict[str, Any] = toml.load(f)
        self._load_default_cosmology(cosmology=self.P["COSMOLOGY"])
        self._process()

    def get_parameters(self) -> dict[str, Any]:
        '''Return the fully-processed parameter dictionary.'''
        return self.P

    def _load_default_cosmology(
        self,
        *,
        cosmology: str = "Planck2018EE+BAO+SN",
        estimate: str = "best",
    ) -> None:
        '''
        Populate ``self.P`` with default cosmological values from the
        bundled ``cosmology.toml`` file.  User-supplied values in the
        input config take precedence via ``dict.setdefault``.

        Parameters
        ----------
        cosmology : str
            Name of the cosmology block in the TOML file.
        estimate : str
            ``'best'`` for best-fit values, ``'mean'`` for the mean of
            the posterior.
        '''
        if estimate not in ('best', 'mean'):
            raise ValueError("'estimate' must be 'best' or 'mean'!")

        with importlib.resources.path('stepsic.config', 'cosmology.toml') as p:
            with open(p, 'r') as fh:
                config_data = toml.load(fh)

        if cosmology not in config_data:
            raise KeyError(
                f"Parameter set '{cosmology}' not found in cosmology.toml."
            )

        for key, value in config_data[cosmology].items():
            if isinstance(value, dict):
                self.P.setdefault(key, value[estimate])
                if estimate == 'mean' and key not in self.P:
                    self.P[f'{key}_err'] = value['error']
            else:
                self.P.setdefault(key, value)

    def _process(self) -> None:
        '''
        Validate, cast, compute derived parameters, and display all
        parameters in the correct order.
        '''
        # cosmo parameters
        _validate_group(COSMO_PARAMS, self.P)
        _compute_derived_cosmo(self.P)
        _display_params('Cosmological Parameters', _COSMO_DISPLAY, self.P)

        # stepsic parameters
        _validate_group(IC_PARAMS, self.P)
        _validate_constraints(IC_CONSTRAINTS, self.P)
        _compute_derived_ic(self.P)
        _display_params('IC Parameters', (*IC_PARAMS, *IC_DERIVED), self.P)