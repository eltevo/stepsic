#!/usr/bin/env bash
set -euo pipefail

# glass/geometries/spherical.sh - Spherical (R^3) glass generation pipeline
#
# Generates a spherical pre-glass IC (shell discretization), compiles a
# non-periodic StePS binary with GLASS_MAKING, and runs glass relaxation.
#
# Steps:
#   1. Generate pre-glass IC via stepsic (LPTORDER=0, TYPE=shell)
#   2. Write StePS parameter file
#   3. Compile StePS GPU binary (GLASS_MAKING, no periodicity flag)
#   4. Run StePS glass relaxation
#
# Called by glass/run.sh. Reads GEOM_START_STEP from env (default 1).

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
GLASS_BIN_NAME="StePS_glass_spherical${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"

echo ""
echo "--------------------------------------------------------------------"
echo "  Spherical (R^3) glass pipeline - starting from step ${START_STEP}"
echo "--------------------------------------------------------------------"

SPHERICAL_DIR="${OUTDIR}/spherical"
mkdir -p "${SPHERICAL_DIR}/preglass" "${SPHERICAL_DIR}/glass"

# -- Step 1: Generate pre-glass IC -----------------------------------------
if (( START_STEP <= 1 )); then
    echo ""
    echo "-- Step 1: Generating spherical pre-glass IC --"

    vlib::clear_files "${TOML_DIR}" "spherical.toml"
    vlib::clear_dir "${SPHERICAL_DIR}/preglass"

    local_diam=$(( R_3D * 2 ))
    vlib::stepsic::write_toml "${TOML_DIR}/spherical.toml" \
        "GEOMETRY=spherical" \
        "LBOX=[${local_diam}, ${local_diam}, ${local_diam}]" \
        "PERIODIC=[0, 0, 0]" \
        "BIN_MODE=${BIN_MODE_STEPS}" \
        "NRBINS=${NRBINS_STEPS}" \
        "TYPE=shell" \
        "NGRID=${NGRID}" \
        "NPART=${NPART}" \
        "NSHELL=${NSHELL_STEPS}" \
        "IC_DIR=${SPHERICAL_DIR}/preglass" \
        "OMEGA_M=1.0" \
        "OMEGA_L=0.0"

    echo "  Generating spherical shell pre-glass IC..."
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/spherical.toml"
fi

# -- Step 2: Write StePS parameter file ------------------------------------
if (( START_STEP <= 2 )); then
    echo ""
    echo "-- Step 2: Writing StePS parameter file --"

    vlib::clear_files "${PARAM_DIR}" "spherical.param"

    IC_SPHERICAL="$(vlib::find_ic "${SPHERICAL_DIR}/preglass")"

    # R^3: IS_PERIODIC=0 (vacuum), L_BOX=0 (unused), R_SIM=sphere radius.
    vlib::steps::write_glass_param \
        "spherical" "${IC_SPHERICAL}" "${SPHERICAL_DIR}/glass/" \
        0 "0" "${R_3D}" \
        "${GLASS_SOFT_SHELL}" \
        "1000" "1000"

    echo "  -> ${PARAM_DIR}/spherical.param"
fi

# -- Step 3: Compile StePS binary ------------------------------------------
if (( START_STEP <= 3 )); then
    echo ""
    echo "-- Step 3: Compiling StePS spherical glass binary --"

    vlib::steps::detect_toolchain
    vlib::clear_files "${BUILD_DIR}" "${GLASS_BIN_NAME}"

    vlib::steps::build "${GLASS_BIN_NAME}" GLASS_MAKING \
        "${STEPS_BACKEND_FLAGS[@]}" ${STEPS_PRECISION_FLAGS}
fi

# Always rewrite the param file so env-var overrides (e.g. GLASS_TIME_LIMIT_MIN)
# take effect even when step 2 was skipped via GEOM_START_STEP=4.
IC_SPHERICAL="$(vlib::find_ic "${SPHERICAL_DIR}/preglass")"
vlib::steps::write_glass_param \
    "spherical" "${IC_SPHERICAL}" "${SPHERICAL_DIR}/glass/" \
    0 "0" "${R_3D}" \
    "${GLASS_SOFT_SHELL}" \
    "1000" "1000"

# -- Step 4: Run glass relaxation ------------------------------------------
if (( START_STEP <= 4 )); then
    echo ""
    echo "-- Step 4: Running spherical glass relaxation --"

    vlib::clear_dir "${SPHERICAL_DIR}/glass"

    vlib::steps::run_binary "${BUILD_DIR}/${GLASS_BIN_NAME}" \
        "${PARAM_DIR}/spherical.param"
fi

echo ""
echo "Spherical glass pipeline complete."
