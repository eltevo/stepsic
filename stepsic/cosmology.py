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

import camb
import numpy as np
from colossus.cosmology import cosmology

log = logging.getLogger(__name__)


def hubble_a(a, H0, omega_m, omega_l):
    r'''
    Computes the Hubble parameter :math:`H(a)` at scale factor :math:`a`.

    The Hubble parameter is given by

    .. math::
        H(a) = H_0\,\sqrt{\Omega_m\,a^3 + (1 - \Omega_m - \Omega_\Lambda)\,a^2 + \Omega_\Lambda},

    where :math:`a` is the scale factor normalized to 1 at present. :math:`H_0`
    is the Hubble constant, :math:`\Omega_m` is the present-day matter
    density parameter and :math:`\Omega_\Lambda` is the present-day dark
    energy density parameter.

    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    H0 : float
        Hubble constant in km/s/Mpc.
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The Hubble parameter :math:`H(a)`.
    '''
    return H0 * np.sqrt(omega_m / a**3 + (1 - omega_m - omega_l) / a**2 + omega_l)


def F_omega(a, omega_m, omega_l):
    r'''
    Computes the linear growth rate factor for first-order Lagrangian
    perturbation.

    This function returns the factor :math:`F_\omega(a)`, defined by

    .. math::
        F_\omega(a)  = \frac{d\ln(D_1)}{d\ln(a)},

    where the effective matter density parameter :math:`\omega(a)` is computed
    as

    .. math::
        \Omega(a) = \frac{\omega_m}{\omega_m + a\,(1 - \omega_m - \omega_l) + \omega_l\,a^3}.

    :math:`F_\omega` approximates the logarithmic derivative of the linear
    growth factor :math:`D_1` with respect to the scale factor :math:`a`, i.e.

    .. math::
        f \equiv \frac{d\ln(D_1)}{d\ln(a)}.


    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The linear growth rate :math:`F_\omega(a)`.
    '''
    omega_a = omega_m / (omega_m + a * (1 - omega_m - omega_l) + a**3 * omega_l)
    return np.power(omega_a, 5.0/9.0)  # Bernardeau et al. 2002, eq. 101a


def F2_omega(a, omega_m, omega_l):
    r'''
    Computes the second-order growth rate factor for second-order
    Lagrangian perturbation theory corrections.

    This function returns the factor :math:`F2_\omega(a)`, defined by

    .. math::
        F2_\omega(a) = \frac{d\ln(D_2)}{d\ln(a)},

    where the effective matter density parameter :math:`\omega(a)` is computed
    as

    .. math::
        \omega(a) = \frac{\omega_m}{\omega_m + a\,(1 - \omega_m - \omega_l) + \omega_l\,a^3}.

    :math:`F2_\omega` is used in second-order Lagrangian perturbation theory
    to scale the second-order displacement field and its time derivative,
    thereby accounting for non-linear corrections to the growth of structure.

    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The second-order growth rate :math:`F2_\omega(a)`.
    '''
    omega_a = omega_m / (omega_m + a * (1 - omega_m - omega_l) + a**3 * omega_l)
    return 2 * np.power(omega_a, 6.0/11.0)  # Bernardeau et al. 2002, eq. 101b

class ColossusCosmology:
    def __init__(self, *,
            H0=67.742, Om0=0.3099, Ob0=0.048891, Ol0=0.6901, sigma8=0.8105,
            ns=0.96822, Neff=3.046, w0=-1.0, wa=0.0, **kwargs):
        '''
        Wrapper to initialize a Colossus cosmology object.

        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc.
        Om0 : float
            Total matter density parameter today divided by the critical density.
        Ob0 : float
            Baryon density parameter today divided by the critical density.
        Ol0 : float
            Dark energy density parameter today divided by the critical density.
        sigma8 : float
            RMS matter fluctuation amplitude at redshift 0.
        ns : float
            Scalar spectrum power-law index for :math:`k = 0.05\,\mathrm{Mpc}^{-1}`.
        Neff : float
            Total effective number of massive and massless neutrinos.
        w0 : float
            Dark energy equation of state parameter at redshift 0.
        wa : float
            Dark energy equation of state parameter evolution.
        '''
        flat = np.isclose(1 - Ol0 - Om0, 0.0, rtol=1e-5, atol=1e-8)
        de_model = 'w0wa' if wa != 0.0 else 'w0' if w0 != -1.0 else 'lambda'
        self.cosmo = cosmology.setCosmology(
            'steps', H0=H0, Om0=Om0, Ob0=Ob0, sigma8=sigma8, ns=ns, Neff=Neff,
            w0=w0, wa=wa, flat=flat, de_model=de_model, **kwargs)
    
    def Dzplus0(self, z : float) -> float:
        '''
        Calculate the linear growth factor :math:`D_+(z)` normalized to
        1 at present.

        For :math:`w_0` and CPL dark energy, we use the Colossus
        implementation of Eq. (11) from Linder & Jenkins (2003).
        See paper at https://arxiv.org/pdf/astro-ph/0305286.pdf.
        
        Parameters
        ----------
        z : float
            Redshift at which to compute the growth factor.
        
        Returns
        -------
        float or np.ndarray
            The linear growth factor D_+(z).
        '''
        Dzplus = self.cosmo.growthFactorUnnormalized(z)
        D0plus = self.cosmo.growthFactorUnnormalized(0.0)

        log.info(f'D_+({z=:.2f})/D_+(z=0) = {Dzplus / D0plus:.2e}')
        return Dzplus / D0plus

class CAMBCosmology:
    def __init__(self, *,
            H0=67.742, ombh2=0.022436, omch2=0.11914, omk=0.0, mnu=0.06,
            nnu=3.046, YHe=0.245421, zrei=7.89, TCMB=2.775, w0=-1.0, wa=0.0,
            nonlinear=False, halofit_version='mead2020', **kwargs):
        '''
        Wrapper to initialize a CAMB cosmology object.
        
        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc.
        ombh2 : float
            Baryon density parameter today.
        omch2 : float
            Cold dark matter density parameter today.
        omk : float
            Curvature density today divided by the critical density.
        mnu : float
            Sum of active neutrino masses in eV.
        nnu : float
            Total effective number of massive and massless neutrinos.
        YHe : float
            Fraction of baryonic mass in helium. Set to ``None`` to be
            calculated internally for BBN consistency.
        TCMB : float
            CMB temperature in Kelvin.
        zrei : float
            Redshift at which the Universe is half reionized.
        w0 : float
            Dark energy equation of state parameter at redshift 0.
        wa : float
            Dark energy equation of state parameter evolution.
        nonlinear : bool
            If ``True``, include non-linear corrections using Halofit.
        halofit_version : str
            Version of the Halofit model to use for non-linear corrections.
            Check ``camb.nonlinear.Halofit`` for available models.
        '''
        self.params = camb.CAMBparams()
        self.params.set_cosmology(
            H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, mnu=mnu, nnu=nnu,
            YHe=YHe, TCMB=TCMB, zrei=zrei, **kwargs)
        if w0 != -1.0 or wa != 0.0:
            log.info(f'Using single fluid dark energy model with w0={w0} and wa={wa}')
            self.params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
            self.params.DarkEnergy.set_params(w=w0, wa=wa)

        if nonlinear:
            log.info(f'Using non-linear corrections with Halofit model `{halofit_version}`')
            self.params.NonLinear = camb.model.NonLinear_both
            self.params.NonLinearModel = camb.nonlinear.Halofit()
            self.params.NonLinearModel.set_params(halofit_version=halofit_version)
        else:
            log.info('Using linear theory only.')
            self.params.NonLinear = camb.model.NonLinear_none

    def get_sigma8(self, z=0, *, As=2.1064e-09, ns=0.96822, kmax=1.0):
        r'''
        Calculate the RMS matter fluctuation amplitude :math:`\sigma_8`
        at given redshift ``z`` using CAMB.
        
        Parameters
        ----------
        z : float
            Target redshift.
        As : float
            Comoving curvature power at :math:`k = 0.05\,\mathrm{Mpc}^{-1}`.
            This is the amplitude of the primordial power spectrum at large
            scales, typically set to match the observed :math:`\sigma_8`.
        ns : float
            Scalar spectral index.
        kmax : float
            Maximum wavenumber in :math:`h^{-1}\,\mathrm{Mpc}`.

        Returns
        -------
        float
            The RMS matter fluctuation amplitude :math:`\sigma_8` at
            redshift :math:`z`.
        '''
        self.params.InitPower.set_params(As=As, ns=ns)
        self.params.set_matter_power(redshifts=[z], kmax=kmax)
        results = camb.get_results(self.params)
        sigma8 = results.get_sigma8()[0]
        return sigma8
    
    def _rescale_As(self, target_sigma8, As=2.1064e-09, ns=0.96822, kmax=1.0):
        '''
        Rescale the matter fluctuation amplitude :math:`A_s` such that
        :math:`\sigma_8(z=0, A_s=\mathrm{new\_As}) = \mathrm{target\_sigma8}`.
        '''
        sigma8_now = self.get_sigma8(z=0, As=As, ns=ns, kmax=kmax)
        if target_sigma8 is None or np.isclose(sigma8_now, target_sigma8, rtol=1e-4):
            return sigma8_now, As
        scale = (target_sigma8 / sigma8_now)**2
        log.info(f'Rescaling matter-fluctuation amplitude by {scale:.3g}')
        As_new = As * scale
        sigma8_new = self.get_sigma8(z=0, As=As_new, ns=ns, kmax=kmax)
        return sigma8_new, As_new

    def get_spectrum(self, *,
            z=127, As=2.1064e-09, ns=0.96822, sigma8_init=None,
            kmin=0.01, kmax=1.0, npoints=512, component='delta_cdm'):
        r'''
        Calculate the matter power spectrum using CAMB.

        Parameters
        ----------
        z : float or list of float
            Redshifts at which the linear power spectrum is calculated.
        As : float
            Comoving curvature power at :math:`k = 0.05\,\mathrm{Mpc}^{-1}`.
            This is the amplitude of the primordial power spectrum at large
            scales, typically set to match the observed :math:`\sigma_8`.
        ns : float
            Scalar spectral index.
        sigma8_init : float
            Rescale the matter power spectrum to the given :math:`\sigma_8`
            value.
        kmin : float
            Minimum wavenumber in :math:`h^{-1}\,\mathrm{Mpc}`.
        kmax : float
            Maximum wavenumber in :math:`h^{-1}\,\mathrm{Mpc}`.
        npoints : int
            Number of wavenumber points.
        component : str
            The component of the power spectrum to return. Options are:
            'delta_tot' for total matter, 'delta_cdm' for cold dark matter,
            'delta_baryon' for baryonic matter, etc. See CAMB documentation
            for more details.
        '''
        # Optional rescaling of the `As` amplitude to match a desired sigma8
        sigma8, As = self._rescale_As(sigma8_init, As=As, ns=ns, kmax=kmax)
        log.info(f'Value for matter fluctuation amplitude used: {sigma8 = :.4f}')

        # Calculating P(k) at redshift `z`
        self.params.set_matter_power(redshifts=np.atleast_1d(z).tolist(), kmax=kmax)
        results = camb.get_results(self.params)
        kh, _, pk = results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints=npoints, var1=component, var2=component)
        pk3 = pk * kh**3/(2*np.pi**2)  # Save (log(kh), log(pk3)).T for StePS/Gadget
        #np.savetxt(f'output/power_spectrum_z{z}.txt', np.log10(kh, pk, pk3).T)
        return kh, pk, pk3