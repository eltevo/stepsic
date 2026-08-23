#!/usr/bin/env bash
# ============================================================================
#  Compare shell masses from constant-angle and constant-volume bins.
#
#  The campaign computes the figure from the binning geometry and cosmology;
#  it does not read simulation snapshots.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE       small, medium (default), or approved large profile
#    --step=N          Start from step N (default 1)
#    --plot-only       Alias for --step=1 (single step, always a plot)
#    --force           Re-run the plot step, ignoring cached output
#    --force-step=plot Re-run only the plot step
#    --clean           Delete output/ before running
#    --config=PATH     Select a typed custom TOML profile
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
vlib::prepare_campaign shell-mass-profiles "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps measure plot evaluate
vlib::manifest_implementation measure "${BASEDIR}/scripts/measure.py"
vlib::manifest_implementation plot "${BASEDIR}/scripts/plot.py"
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache"
OUTPUT="${VLIB_RUN_ROOT}/output"

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
echo "  output:  ${OUTPUT}/shell-mass-profiles.pdf"
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

if vlib::step_check "plot" "${OUTPUT}/shell-mass-profiles.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        --archive "${VLIB_CACHE_DIR}/shell-mass.npz" \
        --D4D "${D4D}" \
        --R3D "${R3D}" \
        --nrbins "${NRBINS}" \
        --nshell "${NSHELL}" \
        --Lz "${LZ}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        -o "${OUTPUT}/shell-mass-profiles.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/shell-mass.npz" \
        --figure "${OUTPUT}/shell-mass-profiles.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
