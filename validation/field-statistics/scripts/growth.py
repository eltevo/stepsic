#!/usr/bin/env python3
"""Plot the error of stepsic's growth fits against integrated growth equations.

stepsic evaluates the second-order growth coefficient D2/D1² and the growth rates f1 and f2 from power-law fits in the matter fraction Ω_m(a). The reference integrates the first- and second-order growth equations for matter, curvature and Λ without radiation.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from stepsic.cosmology import G2_omega, F_omega, F2_omega
from validation._common.fields import VALIDATION_COSMOLOGY
from validation._common.plotting import atomic_savefig, setup_matplotlib


def integrate_growth(scale, omega_m, omega_lambda, *, initial_scale=1e-5, rtol=1e-11):
    """Return D1, dD1/dln(a), D2, dD2/dln(a) at increasing scale factors.

    The integration starts in the matter era on the growing mode, D1 = a and D2 = -3/7 a².
    """
    def rhs(log_a, state):
        a = np.exp(log_a)
        matter = omega_m / a**3
        curvature = (1 - omega_m - omega_lambda) / a**2
        e2 = matter + curvature + omega_lambda
        drag = 2 - (3*matter + 2*curvature)/(2*e2)
        source = 1.5*matter/e2
        d, dp, e, ep = state
        return [dp, source*d-drag*dp, ep, source*(e-d*d)-drag*ep]

    a0 = initial_scale
    solution = solve_ivp(rhs, (np.log(a0), np.log(scale[-1])),
                         [a0, a0, -3*a0*a0/7, -6*a0*a0/7],
                         t_eval=np.log(scale), method='DOP853',
                         rtol=rtol, atol=rtol*1e-7)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    scale = np.geomspace(1/128, 1, 200)
    om, ol = (VALIDATION_COSMOLOGY[key] for key in ('OMEGA_M', 'OMEGA_L'))
    d, dp, e, ep = integrate_growth(scale, om, ol)
    reference = np.array([e/d**2, dp/d, ep/e])
    approximation = np.array([G2_omega(scale, om, ol), F_omega(scale, om, ol),
                              F2_omega(scale, om, ol)])
    error_percent = 100*(approximation/reference-1)

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.46, 2.4), constrained_layout=True)
    for row, label, style in zip(
        error_percent, (r'$D_2/D_1^2$', r'$f_1$', r'$f_2$'), ('-', '--', ':'),
    ):
        ax.plot(1/scale, row, ls=style, label=label)
    ax.axhline(0, color='0.65', lw=.6, zorder=0)
    ax.set(xscale='log', xlabel=r'$1+z$', ylabel=r'Fit / integrated $-\,1$ [\%]')
    ax.legend(frameon=False)
    atomic_savefig(fig, args.figure)
    plt.close(fig)


if __name__ == '__main__':
    main()
