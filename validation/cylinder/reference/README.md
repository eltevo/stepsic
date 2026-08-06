# Manuscript Fig. 7 spectra

These spectra reproduce the cylindrical 1LPT/2LPT ratio in manuscript Fig. 7. Evaluators exclude them from scientific gates because the available metadata does not establish the current initial-condition requirements.

## Integrity

```text
44133d75fb09031ac47a46685865323820593e242b88b46d1f757cdc77c8e200  fig7-pk-1lpt.txt
2ef6e97542bd3c69c52e60085b6b0879ae6839b358c2cc82d1579329a7637107  fig7-pk-2lpt.txt
```

## Reproduce the plot

From the repository root:

```bash
conda run -n stepsic python validation/cylinder/scripts/plot-lpt-ratio.py \
  --pk-1lpt validation/cylinder/reference/fig7-pk-1lpt.txt \
  --pk-2lpt validation/cylinder/reference/fig7-pk-2lpt.txt \
  --output /tmp/cylinder-lpt-ratio.pdf
```

The plotting script explicitly forms `P_1LPT/P_2LPT`; it does not invert the curve.
