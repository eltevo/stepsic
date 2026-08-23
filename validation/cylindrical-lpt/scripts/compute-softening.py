#!/usr/bin/env python3
"""Compute gravitational softening for a cylindrical StePS initial condition.

Use a softening value stored in the HDF5 header when available. Otherwise,
measure the mean particle spacing in the constant-resolution core and divide it
by 40. The result uses the snapshot's coordinate units.
"""
import sys
import h5py
import numpy as np


def main() -> None:
    snap_file = sys.argv[1]

    with h5py.File(snap_file, "r") as f:
        hdr = f["/Header"].attrs
        for key in ("SofteningLength", "PARTICLE_RADII", "Softening"):
            if key in hdr:
                val = float(np.asarray(hdr[key]).reshape(-1)[0])
                print(f"{val:.8f}")
                return

        pos = f["PartType1/Coordinates"][:]
        masses = f["PartType1/Masses"][:]

    m_min = float(np.min(masses))
    mask = np.isclose(masses, m_min, rtol=1e-6)
    n_central = int(np.sum(mask))

    rho = np.sqrt(pos[mask, 0] ** 2 + pos[mask, 1] ** 2)
    z = pos[mask, 2]
    r_max = float(np.max(rho))
    lz = float(np.max(z) - np.min(z))
    v_central = np.pi * r_max**2 * lz

    eps = float(np.cbrt(v_central / n_central)) / 40.0
    print(f"{eps:.8f}")


if __name__ == "__main__":
    main()
