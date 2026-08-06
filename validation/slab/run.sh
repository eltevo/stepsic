#!/usr/bin/env bash
# ============================================================================
#  validation/slab/run.sh - native-slab vs cube-cut fair-sample comparison
#
#  Quantifies how fairly a native L x L x L/s stepsic slab samples an
#  isotropic Universe by comparing it, through an identical windowed
#  estimator, against the central slab cut of a full L^3 periodic
#  realization at the same cell size: isotropic and direction-resolved
#  windowed P(k) ratios plus per-component displacement variances.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N          Start from step N (default 1; steps: run, plot)
#    --plot-only       Skip the run step; regenerate the PDF from cache
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Source an alternative config.env
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Configuration (edit config.env or export before running):
#    LCUBE, ASPECT, NMESH, LPT, Z, SEED, NREAL, PAIRED,
#    TAPER_FRAC, MU_SPLIT, KMAX_FRAC_NY, STEPSIC_ENV
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

vlib::declare_steps run plot evaluate
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
echo "  Slab fair-sample validation: native slab vs cube cut"
echo "========================================================================"
echo ""
echo "  L: ${LCUBE}   aspect: ${ASPECT}:1   nmesh: ${NMESH}"
echo "  lpt: ${LPT}   z: ${Z}   nreal: ${NREAL}   paired: ${PAIRED}"
echo "  cache:  ${VLIB_CACHE_DIR}"
echo "  output: ${OUTPUT}/slab.pdf"
echo ""

PAIRED_FLAG=()
if [[ "${PAIRED:-false}" == "true" ]]; then
    PAIRED_FLAG=(--paired)
fi

# --------------------------------------------------------------------------
if vlib::step_check "run" "${VLIB_CACHE_DIR}/data.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lcube "${LCUBE}" \
        --aspect "${ASPECT}" \
        --nmesh "${NMESH}" \
        --lpt "${LPT}" \
        --z "${Z}" \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --taper-frac "${TAPER_FRAC}" \
        --mu-split "${MU_SPLIT}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/data.npz"
    vlib::step_done "run"
fi

# --------------------------------------------------------------------------
if vlib::step_check "plot" "${OUTPUT}/slab.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/data.npz" \
        -o "${OUTPUT}/slab.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/data.npz" \
        --figure "${OUTPUT}/slab.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
