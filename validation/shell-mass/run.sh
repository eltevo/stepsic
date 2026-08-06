#!/usr/bin/env bash
# ============================================================================
#  validation/shell-mass/run.sh - shell mass distribution (omega vs volume)
#
#  Plots the particle mass distribution across radial shells for spherical
#  and cylindrical StePS geometries, comparing constant-omega vs
#  constant-volume radial binning strategies.
#
#  No simulation data required - the figure is computed purely from the
#  binning geometry and cosmological model.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N          Start from step N (default 1)
#    --plot-only       Alias for --step=1 (single step, always a plot)
#    --force           Re-run the plot step, ignoring cached output
#    --force-step=plot Re-run only the plot step
#    --clean           Delete output/ before running
#    --config=PATH     Source an alternative config.env
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Configuration (edit config.env or export before running):
#    D4D           Compactification diameter D_4D [Mpc/h]  (default: 75)
#    R3D           Simulation radius R_3D [Mpc/h]          (default: 500)
#    NRBINS        Number of radial bins                   (default: 224)
#    NSHELL        Particles per shell                     (default: 12288)
#    LZ            Cylinder height [Mpc/h]                 (default: 200)
#    COSMO_OMEGA_M Matter density; overrides the common default if exported
#    RCRIT         Critical radius for inner zone [Mpc/h]  (default: "")
#    RCRIT_MODES   Binning methods receiving RCRIT         (default: omega)
#    STEPSIC_ENV   Conda env for stepsic                   (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

vlib::declare_steps measure plot evaluate
if (( VLIB_LIST_STEPS )); then
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

VLIB_CACHE_DIR="${BASEDIR}/cache"
OUTPUT="${BASEDIR}/output"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${OUTPUT}"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Shell mass distribution: omega vs. constant-volume binning"
echo "========================================================================"
echo ""
echo "  D_4D:    ${D4D} Mpc/h"
echo "  R_3D:    ${R3D} Mpc/h"
echo "  N_rbins: ${NRBINS}"
echo "  N_shell: ${NSHELL}"
echo "  Lz:      ${LZ} Mpc/h"
echo "  RCRIT:   ${RCRIT:-none}"
echo "  output:  ${OUTPUT}/shell-mass.pdf"
echo ""

EXTRA_FLAGS=(--omega-m "${COSMO_OMEGA_M}")
if [[ -n "${RCRIT:-}" ]]; then
    EXTRA_FLAGS+=(--rcrit "${RCRIT}")
    # shellcheck disable=SC2086
    EXTRA_FLAGS+=(--rcrit-modes ${RCRIT_MODES})
fi

if vlib::step_check "measure" "${VLIB_CACHE_DIR}/shell-mass.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/measure.py" \
        --D4D "${D4D}" \
        --R3D "${R3D}" \
        --nrbins "${NRBINS}" \
        --nshell "${NSHELL}" \
        --Lz "${LZ}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        -o "${VLIB_CACHE_DIR}/shell-mass.npz"
    vlib::step_done "measure"
fi

if vlib::step_check "plot" "${OUTPUT}/shell-mass.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        --archive "${VLIB_CACHE_DIR}/shell-mass.npz" \
        --D4D "${D4D}" \
        --R3D "${R3D}" \
        --nrbins "${NRBINS}" \
        --nshell "${NSHELL}" \
        --Lz "${LZ}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        -o "${OUTPUT}/shell-mass.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/shell-mass.npz" \
        --figure "${OUTPUT}/shell-mass.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
