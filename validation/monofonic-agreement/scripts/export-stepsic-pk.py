#!/usr/bin/env python3
'''
Export the CAMB linear transfer function in monofonIC's expected format.

monofonIC's `CAMB_file` transfer-function plugin reads a 13-column ASCII
table at the target redshift (`ztarget`, default 0):

    k/h   T_cdm   T_b   T_g   T_nu   T_mass_nu   T_tot
          T_no_nu   T_tot_de   T_Weyl   v_cdm   v_b   v_b - v_cdm

This script reproduces stepsic's CAMB cosmology setup (so transfers
are computed with the same parameters stepsic uses internally), then
saves the 13-column table at z=0. monofonIC handles backscaling to
zstart on its own via D+(astart) / D+(atarget).

Notes
-----
The CAMB linear transfer function is independent of the primordial
amplitude (A_s, sigma_8), so no As-rescaling is performed here.
monofonIC re-normalizes the resulting power spectrum to match
[cosmology] / sigma_8 in its own config (tf_isnormalised_=false in
the CAMB_file plugin), so make sure that value is set there.

Usage
-----
::

    python export-stepsic-pk.py \\
        --cosmology Planck2018EE+BAO \\
        --lbox 1000 \\
        -o camb_transfer.dat
'''

from __future__ import annotations

import argparse
import logging

import camb
import numpy as np


from stepsic.cosmology import CAMBCosmology

from validation._common.cosmology_fields import VALIDATION_COSMOLOGY, VALIDATION_COSMOLOGY_NAME
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# CAMB transfer-function column constants (1-indexed); see
# camb.model.Transfer_*. Order matches monofonIC's CAMB_file plugin.
_TRANSFER_COLUMNS = [
    'Transfer_kh',
    'Transfer_cdm',
    'Transfer_b',
    'Transfer_g',
    'Transfer_r',
    'Transfer_nu',
    'Transfer_tot',
    'Transfer_nonu',
    'Transfer_tot_de',
    'Transfer_Weyl',
    'Transfer_Newt_vel_cdm',
    'Transfer_Newt_vel_baryon',
    'Transfer_vel_baryon_cdm',
]


def _load_cosmology(name: str) -> dict[str, float]:
    '''Return the shared validation cosmology after checking its name.'''
    if name != VALIDATION_COSMOLOGY_NAME:
        raise ValueError(f'unsupported validation cosmology: {name}')
    return VALIDATION_COSMOLOGY


def export_transfer(
    cosmology_name: str,
    lbox: float,
    output: str,
) -> None:
    '''
    Compute CAMB linear transfer functions at z=0 and save in CAMB's
    13-column ASCII format (the format monofonIC's CAMB_file plugin
    expects).
    '''
    cosmo = _load_cosmology(cosmology_name)
    h = cosmo['H0'] / 100.0
    ombh2 = cosmo['OMEGA_B'] * h**2
    omch2 = (cosmo['OMEGA_M'] - cosmo['OMEGA_B']) * h**2

    log.info('Cosmology: %s', cosmology_name)
    log.info('  H0=%.3f, Omega_m=%.4f, sigma8=%.4f, ns=%.5f',
             cosmo['H0'], cosmo['OMEGA_M'], cosmo['SIGMA8'], cosmo['NS'])

    # Use stepsic's CAMB wrapper for parameter consistency.
    camb_cosmo = CAMBCosmology(
        H0=cosmo['H0'], ombh2=ombh2, omch2=omch2,
        omk=0.0, mnu=cosmo['MNU'], nnu=cosmo['NNU'],
        YHe=cosmo['YHE'], TCMB=cosmo['TCMB'],
        zrei=cosmo.get('ZREI', 7.90),
        w0=cosmo.get('W0', -1.0), wa=cosmo.get('WA', 0.0),
        nonlinear=False,
    )

    # CAMB needs to know the k-range for transfer-function evaluation.
    # Cover well below kmin = 2*pi / Lbox and well above the Nyquist
    # of any reasonable mesh.
    kmin = 1.0 / lbox / 10.0          # h/Mpc, well below box scale
    kmax = 500.0                       # h/Mpc, well above Nyquist
    camb_cosmo.params.InitPower.set_params(
        As=cosmo['AS'], ns=cosmo['NS'], pivot_scalar=cosmo.get('KPIVOT', 0.05))
    camb_cosmo.params.set_matter_power(
        redshifts=[0.0], kmax=kmax, k_per_logint=20)
    camb_cosmo.params.WantTransfer = True

    results = camb.get_results(camb_cosmo.params)
    td = results.get_matter_transfer_data()

    # transfer_data has shape (n_components, n_k, n_redshifts).
    # Slice z=0 (only redshift requested) and verify column count.
    data = td.transfer_data[:, :, 0]
    n_components_have = data.shape[0]
    n_components_want = len(_TRANSFER_COLUMNS)
    if n_components_have < n_components_want:
        raise RuntimeError(
            f'CAMB returned only {n_components_have} transfer components, '
            f'expected at least {n_components_want}. '
            f'the required transfer components are unavailable.'
        )

    # Take the first 13 components in CAMB's native order. Drop k
    # samples below kmin set by Lbox (with margin) to keep the table
    # tight without affecting interpolation in the relevant range.
    cols = data[:n_components_want, :]   # (13, n_k)

    # Restrict to k >= kmin and sort by k.
    kh = cols[0, :]
    mask = (kh >= kmin) & (kh <= kmax)
    cols = cols[:, mask]
    order = np.argsort(cols[0, :])
    cols = cols[:, order]

    table = cols.T  # (n_k, 13)
    log.info(
        'Transfer table: %d k-bins, k=[%.4e, %.4e] h/Mpc',
        table.shape[0], table[0, 0], table[-1, 0],
    )

    header = (
        f'CAMB linear transfer function at z=0 for {cosmology_name}\n'
        f'Generated by stepsic export-stepsic-pk.py\n'
        f'Format: monofonIC CAMB_file plugin (13 columns)\n'
        f'Columns: 1:k/h  2:T_cdm  3:T_b  4:T_g  5:T_nu  6:T_mass_nu  '
        f'7:T_tot  8:T_no_nu  9:T_tot_de  10:T_Weyl  11:v_cdm  '
        f'12:v_b  13:v_b-v_cdm\n'
        f'Note: monofonIC normalizes amplitude to [cosmology]/sigma_8'
    )
    np.savetxt(output, table, header=header, fmt='%.10e')
    log.info('Saved to %s', output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Export CAMB linear transfer functions in monofonIC '
            'CAMB_file 13-column format.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--cosmology', type=str, default=VALIDATION_COSMOLOGY_NAME,
                   help='Shared validation cosmology name.')
    p.add_argument('--lbox', type=float, default=1000.0,
                   help='Box side length [Mpc/h] (sets kmin floor).')
    p.add_argument('-o', '--output', type=str, required=True,
                   help='Output ASCII file path.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_transfer(
        cosmology_name=args.cosmology,
        lbox=args.lbox,
        output=args.output,
    )
