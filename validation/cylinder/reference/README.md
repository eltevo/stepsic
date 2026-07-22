# Historical Fig. 7 reference inputs

These spectra are the inputs behind the manuscript's original cylindrical 1LPT/2LPT ratio. They are retained as audit evidence only. They were generated with the pre-fix 2LPT implementation and must not be reused as revised-paper evidence.

## Provenance

- Recovered on 2026-07-18 from `/home/nebula/projects/stepsic-bak/validation/cylinder/pk/`.
- The preserved simulations are:
  - `/data/steps/cylindrical-1M/LCDM_cylindrical_Lz200_D1500_z31_1LPT`
  - `/data/steps/cylindrical-1M/LCDM_cylindrical_Lz200_D1500_z31_2LPT`
- Their ICs contain 1,983,884 particles, use `R_3D=750 Mpc`, `L_z=200 Mpc`, `z_init=31`, and record StePSIC commit `78e7d40e8033570db32d0c8f53a1a0debb370502`.
- Both spectrum headers record the same FKP settings: `P0=10000`, three radial bins, 32 radial n(r) bins, a 512 mesh, and shot-noise subtraction.
- The original IC TOMLs were not preserved. In particular, the value of `COMPENSATE` cannot be proven from the HDF5 headers. Both the recovered and current cylinder validation drivers set it to `false`.

## Integrity

```text
44133d75fb09031ac47a46685865323820593e242b88b46d1f757cdc77c8e200  fig7-pk-1lpt.txt
2ef6e97542bd3c69c52e60085b6b0879ae6839b358c2cc82d1579329a7637107  fig7-pk-2lpt.txt
```

## Reproduce the audited plot

From the repository root:

```bash
conda run -n stepsic python validation/cylinder/scripts/plot-lpt-ratio.py \
  --pk-1lpt validation/cylinder/reference/fig7-pk-1lpt.txt \
  --pk-2lpt validation/cylinder/reference/fig7-pk-2lpt.txt \
  --output /tmp/article-review-lpt-ratio.pdf
```

The plotting script explicitly forms `P_1LPT/P_2LPT`; it does not invert the curve. Regeneration on 2026-07-18 loaded 375 bins from each input and wrote a 227,958-byte PDF. The numerical audit is recorded in `docs/.agents/article-review/findings.md`.
