#!/usr/bin/env bash
set -euo pipefail

# glass/geometries/cubical.sh - Cubical (T^3) glass generation pipeline
#
# Generates two cubical pre-glass ICs (random + grid), compiles a periodic
# StePS binary with GLASS_MAKING, and runs glass relaxation on the random
# IC only (the grid IC is plotted directly as-is).
#
# Steps:
#   1. Generate pre-glass ICs via stepsic (LPTORDER=0)
#   2. Write StePS parameter file for the random-IC glass run
#   3. Compile StePS GPU binary (PERIODIC + GLASS_MAKING)
#   4. Run StePS glass relaxation
#
# Called by glass/run.sh. Reads GEOM_START_STEP from env (default 1).
# All other required env vars are set by the controller - see run.sh.

GEOM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_RUN_ROOT="${VLIB_RUN_ROOT}"
source "${GEOM_DIR}/../../_common/lib.sh"
VLIB_RUN_ROOT="${PARENT_RUN_ROOT}"

START_STEP="${GEOM_START_STEP:-1}"
if ! [[ "${START_STEP}" =~ ^[1-4]$ ]]; then
    echo "ERROR: GEOM_START_STEP must be 1–4 (got '${START_STEP}')." >&2
    exit 1
fi

# Precision build flag (empty for the default double)
STEPS_PRECISION_FLAGS="$(vlib::steps::precision_flags)"
STEPS_BACKEND_SUFFIX=""
STEPS_BACKEND_FLAGS=()
if [[ "${STEPS_BACKEND}" == "bh" ]]; then
    STEPS_BACKEND_SUFFIX="_bh"
    STEPS_BACKEND_FLAGS=("USE_BH=0.25" "RANDOMIZE_BH=123456")
fi
GLASS_BIN_NAME="StePS_glass_periodic${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"

echo ""
echo "--------------------------------------------------------------------"
echo "  Cubical (T^3) glass pipeline - starting from step ${START_STEP}"
echo "--------------------------------------------------------------------"

CUBIC_RANDOM_DIR="${OUTDIR}/cubic_random"
CUBIC_GRID_DIR="${OUTDIR}/cubic_grid"
mkdir -p "${CUBIC_RANDOM_DIR}/preglass" "${CUBIC_RANDOM_DIR}/glass"
mkdir -p "${CUBIC_GRID_DIR}/preglass"

# -- Step 1: Generate pre-glass ICs ----------------------------------------
if (( START_STEP <= 1 )); then
    echo ""
    echo "-- Step 1: Generating cubical pre-glass ICs --"

    vlib::clear_files "${TOML_DIR}" "cubic_random.toml" "cubic_grid.toml"
    vlib::clear_dir "${CUBIC_RANDOM_DIR}/preglass"
    vlib::clear_dir "${CUBIC_GRID_DIR}/preglass"

    vlib::stepsic::write_toml "${TOML_DIR}/cubic_random.toml" \
        "GEOMETRY=cubical" \
        "LBOX=[${LBOX}, ${LBOX}, ${LBOX}]" \
        "PERIODIC=[1, 1, 1]" \
        "BIN_MODE=${BIN_MODE_CUBICAL}" \
        "NRBINS=${NRBINS_CUBICAL}" \
        "TYPE=random" \
        "NGRID=${NGRID}" \
        "NPART=${NPART}" \
        "NSHELL=${NSHELL_CUBICAL}" \
        "IC_DIR=${CUBIC_RANDOM_DIR}/preglass" \
        "OMEGA_M=1.0" \
        "OMEGA_L=0.0"

    vlib::stepsic::write_toml "${TOML_DIR}/cubic_grid.toml" \
        "GEOMETRY=cubical" \
        "LBOX=[${LBOX}, ${LBOX}, ${LBOX}]" \
        "PERIODIC=[1, 1, 1]" \
        "BIN_MODE=${BIN_MODE_CUBICAL}" \
        "NRBINS=${NRBINS_CUBICAL}" \
        "TYPE=grid" \
        "NGRID=${NGRID}" \
        "NPART=${NPART}" \
        "NSHELL=${NSHELL_CUBICAL}" \
        "IC_DIR=${CUBIC_GRID_DIR}/preglass" \
        "OMEGA_M=1.0" \
        "OMEGA_L=0.0"

    echo "  [1/2] Random pre-glass IC..."
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cubic_random.toml"

    echo "  [2/2] Grid pre-glass IC..."
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cubic_grid.toml"
fi

# -- Step 2: Write StePS parameter file ------------------------------------
if (( START_STEP <= 2 )); then
    echo ""
    echo "-- Step 2: Writing StePS parameter file --"

    vlib::clear_files "${PARAM_DIR}" "cubic_random.param"

    IC_CUBIC_RANDOM="$(vlib::find_ic "${CUBIC_RANDOM_DIR}/preglass")"

    # T^3: IS_PERIODIC=3 (Ewald), L_BOX=box side, R_SIM=0 (unused).
    vlib::steps::write_glass_param \
        "cubic_random" "${IC_CUBIC_RANDOM}" "${CUBIC_RANDOM_DIR}/glass/" \
        3 "${LBOX}" "0" \
        "${GLASS_SOFT_CUBIC}" \
        "1000" "1000"

    echo "  -> ${PARAM_DIR}/cubic_random.param"
fi

# -- Step 3: Compile StePS binary ------------------------------------------
if (( START_STEP <= 3 )); then
    echo ""
    echo "-- Step 3: Compiling StePS periodic glass binary --"

    vlib::steps::detect_toolchain
    vlib::clear_files "${BUILD_DIR}" "${GLASS_BIN_NAME}"

    vlib::steps::build "${GLASS_BIN_NAME}" PERIODIC GLASS_MAKING \
        "${STEPS_BACKEND_FLAGS[@]}" ${STEPS_PRECISION_FLAGS}
fi

# Always rewrite the param file so env-var overrides (e.g. GLASS_TIME_LIMIT_MIN)
# take effect even when step 2 was skipped via GEOM_START_STEP=4.
IC_CUBIC_RANDOM="$(vlib::find_ic "${CUBIC_RANDOM_DIR}/preglass")"
vlib::steps::write_glass_param \
    "cubic_random" "${IC_CUBIC_RANDOM}" "${CUBIC_RANDOM_DIR}/glass/" \
    3 "${LBOX}" "0" \
    "${GLASS_SOFT_CUBIC}" \
    "1000" "1000"

# -- Step 4: Run glass relaxation ------------------------------------------
if (( START_STEP <= 4 )); then
    echo ""
    echo "-- Step 4: Running cubic random glass relaxation --"
    echo "   (Grid IC is used as-is for plotting; no relaxation needed.)"

    vlib::clear_dir "${CUBIC_RANDOM_DIR}/glass"

    vlib::steps::run_binary_with_fresh_ewald \
        "${BUILD_DIR}/${GLASS_BIN_NAME}" "${PARAM_DIR}/cubic_random.param" \
        "${CUBIC_RANDOM_DIR}/glass" "Ewald_table_medres.hdf5"
fi

echo ""
echo "Cubical glass pipeline complete."
