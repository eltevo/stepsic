#!/usr/bin/env bash
set -euo pipefail

# glass/geometries/cylindrical.sh - Cylindrical (S^1 x R^2) glass generation pipeline
#
# Generates a cylindrical pre-glass IC (shell discretization), compiles a
# PERIODIC_Z StePS binary with GLASS_MAKING, and runs glass relaxation.
#
# Steps:
#   1. Generate pre-glass IC via stepsic (LPTORDER=0, TYPE=shell)
#   2. Write StePS parameter file
#   3. Compile StePS GPU binary (PERIODIC_Z + GLASS_MAKING)
#   4. Run StePS glass relaxation
#
# Called by glass/run.sh. Reads GEOM_START_STEP from env (default 1).

GEOM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GEOM_DIR}/../../_common/lib.sh"

START_STEP="${GEOM_START_STEP:-1}"
if ! [[ "${START_STEP}" =~ ^[1-4]$ ]]; then
    echo "ERROR: GEOM_START_STEP must be 1–4 (got '${START_STEP}')." >&2
    exit 1
fi

# Precision build flag (empty for the default double)
STEPS_PRECISION_FLAGS="$(vlib::steps::precision_flags)"
GLASS_BIN_NAME="StePS_glass_cylindrical$(vlib::steps::precision_suffix)"

echo ""
echo "--------------------------------------------------------------------"
echo "  Cylindrical (S^1 x R^2) glass pipeline - starting from step ${START_STEP}"
echo "--------------------------------------------------------------------"

CYLINDRICAL_DIR="${OUTDIR}/cylindrical"
CYLINDRICAL_EWALD_DIR="${CYLINDRICAL_DIR}/ewald"
mkdir -p "${CYLINDRICAL_DIR}/preglass" "${CYLINDRICAL_DIR}/glass" \
    "${CYLINDRICAL_EWALD_DIR}"

# -- Step 1: Generate pre-glass IC -----------------------------------------
if (( START_STEP <= 1 )); then
    echo ""
    echo "-- Step 1: Generating cylindrical pre-glass IC --"

    vlib::clear_files "${TOML_DIR}" "cylindrical.toml"
    vlib::clear_dir "${CYLINDRICAL_DIR}/preglass"

    local_diam=$(( R_3D * 2 ))
    vlib::stepsic::write_toml "${TOML_DIR}/cylindrical.toml" \
        "GEOMETRY=cylindrical" \
        "LBOX=[${local_diam}, ${local_diam}, ${LZ}]" \
        "PERIODIC=[0, 0, 1]" \
        "BIN_MODE=${BIN_MODE_STEPS}" \
        "NRBINS=${NRBINS_STEPS}" \
        "TYPE=shell" \
        "NGRID=${NGRID}" \
        "NPART=${NPART}" \
        "NSHELL=${NSHELL_STEPS}" \
        "IC_DIR=${CYLINDRICAL_DIR}/preglass" \
        "OMEGA_M=1.0" \
        "OMEGA_L=0.0"

    echo "  Generating cylindrical shell pre-glass IC..."
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cylindrical.toml"
fi

# -- Step 2: Write StePS parameter file ------------------------------------
if (( START_STEP <= 2 )); then
    echo ""
    echo "-- Step 2: Writing StePS parameter file --"

    vlib::clear_files "${PARAM_DIR}" "cylindrical.param"

    IC_CYLINDRICAL="$(vlib::find_ic "${CYLINDRICAL_DIR}/preglass")"

    # S^1 x R^2: IS_PERIODIC=4 (high-precision Ewald), L_BOX=Lz, R_SIM=R_3D.
    vlib::steps::write_glass_param \
        "cylindrical" "${IC_CYLINDRICAL}" "${CYLINDRICAL_DIR}/glass/" \
        4 "${LZ}" "${R_3D}" \
        "${GLASS_SOFT_SHELL}" \
        "1000" "1000"

    echo "  -> ${PARAM_DIR}/cylindrical.param"
fi

# -- Step 3: Compile StePS binary ------------------------------------------
if (( START_STEP <= 3 )); then
    echo ""
    echo "-- Step 3: Compiling StePS cylindrical glass binary --"

    vlib::steps::detect_toolchain
    vlib::clear_files "${BUILD_DIR}" "${GLASS_BIN_NAME}"

    vlib::steps::build "${GLASS_BIN_NAME}" PERIODIC_Z GLASS_MAKING ${STEPS_PRECISION_FLAGS}
fi

# Always rewrite the param file so env-var overrides (e.g. GLASS_TIME_LIMIT_MIN)
# take effect even when step 2 was skipped via GEOM_START_STEP=4.
IC_CYLINDRICAL="$(vlib::find_ic "${CYLINDRICAL_DIR}/preglass")"
vlib::steps::write_glass_param \
    "cylindrical" "${IC_CYLINDRICAL}" "${CYLINDRICAL_DIR}/glass/" \
    4 "${LZ}" "${R_3D}" \
    "${GLASS_SOFT_SHELL}" \
    "1000" "1000"

# -- Step 4: Run glass relaxation ------------------------------------------
if (( START_STEP <= 4 )); then
    echo ""
    echo "-- Step 4: Running cylindrical glass relaxation --"

    vlib::steps::recover_ewald_cache \
        "${CYLINDRICAL_DIR}/glass" "${CYLINDRICAL_EWALD_DIR}"
    vlib::clear_dir "${CYLINDRICAL_DIR}/glass"

    vlib::steps::run_binary_with_ewald_cache \
        "${BUILD_DIR}/${GLASS_BIN_NAME}" "${PARAM_DIR}/cylindrical.param" \
        "${CYLINDRICAL_DIR}/glass" "${CYLINDRICAL_EWALD_DIR}" \
        "run_glass_cylindrical" "higres"
fi

echo ""
echo "Cylindrical glass pipeline complete."