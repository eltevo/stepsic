#!/usr/bin/env bash
# =============================================================================
#  validation/smoke_test.sh - lightweight smoke test for all 8 pipelines
#
#  Calls each run.sh with minimal parameters.
#
#  Usage:
#    bash validation/smoke-test.sh
# =============================================================================
set -euo pipefail

VDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "  Validation smoke test - $(date)"
echo "========================================================================"

# 1. field - displacement/velocity field histograms
echo ""
echo "[1/8] field"
(cd "${VDIR}/field" &&
    NREAL=1 NMESH_3D=8 NMESH_SLAB=8 \
    bash run.sh --force)

# 2. grid - P(k) recovery, one nmesh/z/LPT/MAS per panel
echo ""
echo "[2/8] grid"
(cd "${VDIR}/grid" &&
    PANEL_A_NMESH=8 \
    PANEL_B_NMESH=8 PANEL_B_Z=31 \
    PANEL_C_NMESH=8 PANEL_C_LPT=2 \
    PANEL_D_NMESH=8 PANEL_D_METHOD=cic \
    NREAL=1 PAIRED=false \
    bash run.sh --force)

# 3. glass - run full pipeline.
# Tune first snapshot output; default 999 would never be reached in a 1-minute run.
echo ""
echo "[3/8] glass"
(cd "${VDIR}/glass" &&
    GLASS_TIME_LIMIT_MIN=1 \
    GLASS_FIRST_T_OUT=99000 GLASS_H_OUT=99000 \
    bash run.sh --force)

# 4. cylinder - end-to-end validation; compare 1lpt vs 2lpt simulation
echo ""
echo "[4/8] cylinder"
(cd "${VDIR}/cylinder" &&
    NRBINS=32 NSHELL=8 SIM_NMESH=16 \
    PK_NMESH=8 PK_RANDOMS_NFACTOR=2 \
    GLASS_TIME_LIMIT_MIN=1 SIM_TIME_LIMIT_MIN=1 \
    bash run.sh --force)

# 5. squish - P(k) vs aspect ratio; full cube versus flat slab
echo ""
echo "[5/8] squish"
(cd "${VDIR}/squish" &&
    NMESH=8 NSTEPS=2 NREAL=1 PAIRED=false \
    bash run.sh --force)

# 6. particle-load - all 4 IC geometry types
echo ""
echo "[6/8] particle-load"
(cd "${VDIR}/particle-load" &&
    NGRID=8 NPART=256 NRBINS=32 NSHELL=8 \
    bash run.sh --force)

# 7. shell-mass - shell mass distribution
echo ""
echo "[7/8] shell-mass"
(cd "${VDIR}/shell-mass" &&
    bash run.sh --force)

# 8. monofonic - tiny 8^3 ICs
echo ""
echo "[8/8] monofonic"
(cd "${VDIR}/monofonic" &&
    LBOX=100 NMESH=8 \
    bash run.sh --force)

echo ""
echo "========================================================================"
echo "  All 8 pipelines completed successfully."
echo "========================================================================"
