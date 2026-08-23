#!/usr/bin/env python3
"""Generate an FKP random catalogue for a cylindrical simulation.

The random particles fill the same cylinder and follow the radial mass profile
measured from the input glass.
"""

import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate random catalog for cylindrical FKP P(k) estimation."
    )
    parser.add_argument("glass_path", type=str,
                        help="Path to the glass HDF5 file.")
    parser.add_argument("output_path", type=str,
                        help="Path for the output random catalog HDF5 file.")
    parser.add_argument("--nfactor", type=int, default=10,
                        help="Oversampling factor: N_randoms = nfactor * N_glass (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Load the glass ---
    with h5py.File(args.glass_path, "r") as f:
        coords = f["PartType1/Coordinates"][:]
        masses = f["PartType1/Masses"][:]
        # Copy header attributes for later
        header_attrs = dict(f["/Header"].attrs)

    N_glass = len(masses)
    N_rand = args.nfactor * N_glass
    print(f"Glass: {N_glass} particles")
    print(f"Generating {N_rand} random particles (factor {args.nfactor}x)")

    # --- Measure the cylindrical geometry ---
    R_cyl = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    R_max = R_cyl.max()
    Lz = z_max - z_min
    print(f"Cylinder: R_max = {R_max:.2f} Mpc, z in [{z_min:.2f}, {z_max:.2f}] Mpc, Lz = {Lz:.2f} Mpc")

    # --- Bin the glass by cylindrical radius to get the mass profile ---
    n_bins = 200
    r_edges = np.linspace(0, R_max * 1.01, n_bins + 1)
    bin_idx = np.digitize(R_cyl, r_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Count particles per bin and compute the mass in each bin
    mass_per_bin = np.zeros(n_bins)
    count_per_bin = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_idx == i
        mass_per_bin[i] = masses[mask].sum() if mask.any() else 0.0
        count_per_bin[i] = mask.sum()

    # Annulus volumes for mass assignment to randoms
    annulus_vol = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2) * Lz

    # PDF for radial sampling: proportional to particle count per bin
    # (this samples the n(r) profile used by the FKP estimator)
    pdf = count_per_bin.astype(float)
    pdf /= pdf.sum()

    # Mean mass per particle in each bin (for assigning masses to randoms)
    mean_mass_per_bin = np.zeros_like(mass_per_bin)
    np.divide(
        mass_per_bin,
        count_per_bin,
        out=mean_mass_per_bin,
        where=count_per_bin > 0,
    )

    # --- Generate random particles ---
    # Step 1: Draw radial bin for each random particle
    bin_choices = rng.choice(n_bins, size=N_rand, p=pdf)

    # Step 2: Uniform in r^2 within each annulus
    r0 = r_edges[bin_choices]
    r1 = r_edges[bin_choices + 1]
    u = rng.uniform(size=N_rand)
    r_rand = np.sqrt(u * (r1**2 - r0**2) + r0**2)

    # Step 3: Uniform angle
    theta = rng.uniform(0, 2 * np.pi, size=N_rand)

    # Step 4: Uniform z
    z_rand = rng.uniform(z_min, z_max, size=N_rand)

    # Cartesian coordinates
    x_rand = r_rand * np.cos(theta)
    y_rand = r_rand * np.sin(theta)

    rand_coords = np.column_stack([x_rand, y_rand, z_rand])
    rand_masses = mean_mass_per_bin[bin_choices]

    print(f"Random catalog: {N_rand} particles generated")
    print(f"  R range: [{np.sqrt(x_rand**2 + y_rand**2).min():.4f}, {np.sqrt(x_rand**2 + y_rand**2).max():.2f}] Mpc")
    print(f"  z range: [{z_rand.min():.2f}, {z_rand.max():.2f}] Mpc")
    print(f"  Mass range: [{rand_masses.min():.6e}, {rand_masses.max():.6e}]")

    # --- Write output HDF5 ---
    with h5py.File(args.output_path, "w") as f:
        # Header (copy from glass)
        hdr = f.create_group("Header")
        for k, v in header_attrs.items():
            if k == "NumPart_ThisFile" or k == "NumPart_Total":
                npart = np.zeros(6, dtype=np.uint32)
                npart[1] = N_rand
                hdr.attrs.create(k, npart)
            else:
                hdr.attrs.create(k, v)

        # PartType1 group
        grp = f.create_group("PartType1")
        grp.create_dataset("Coordinates", data=rand_coords.astype(np.float64))
        grp.create_dataset("Masses", data=rand_masses.astype(np.float64))
        grp.create_dataset("Velocities", data=np.zeros_like(rand_coords, dtype=np.float64))

    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
