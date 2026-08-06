#!/usr/bin/env bash
# ============================================================================
#  validation/squish/run.sh - P(k) recovery vs box aspect ratio
#
#  Starting from a cubic box L_cube^3, the z-dimension is linearly
#  decreased to L_z_min while x and y stay fixed. Each slab geometry
#  gets its own band-averaged theory reference so the comparison is
#  apples-to-apples.
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
#    LCUBE             Cubic base side [Mpc/h]        (default: 1000)
#    LZ_MIN            Minimum z-dimension [Mpc/h]    (default: 100)
#    NSTEPS            Number of aspect-ratio steps   (default: 4)
#    NMESH             Grid cells (shortest dim)      (default: 48)
#    LPT               LPT order                      (default: 2)
#    Z                 Target redshift                (default: 31)
#    METHOD            Mass-assignment scheme         (default: cic)
#    SEED              RNG seed                       (default: 137)
#    NREAL             Realisations to average        (default: 1)
#    PAIRED            Paired-fixed averaging         (default: true)
#    KMAX_FRAC_NY      k_max / k_Nyquist              (default: 0.8)
#    YLIM              Plot y-axis limits             (default: 0.998 1.002)
#    PERCENT_BAND      Shaded band half-width         (default: 0.0005)
#    STEPSIC_ENV       Conda env for stepsic          (default: stepsic)
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

# --clean: wipe cache and output before running
if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${OUTPUT}"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Squish validation: P(k) vs box aspect ratio"
echo "========================================================================"
echo ""
echo "  L_cube:      ${LCUBE} Mpc/h"
echo "  L_z_min:     ${LZ_MIN} Mpc/h"
echo "  N_steps:     ${NSTEPS}"
echo "  nmesh:       ${NMESH}   lpt: ${LPT}   z: ${Z}"
echo "  method:      ${METHOD}   paired: ${PAIRED}"
echo "  cache:       ${VLIB_CACHE_DIR}"
echo "  output:      ${OUTPUT}/squish.pdf"
echo ""

# Build paired flag
PAIRED_FLAG=()
if [[ "${PAIRED:-false}" == "true" ]]; then
    PAIRED_FLAG=(--paired)
fi

# --------------------------------------------------------------------------
if vlib::step_check "run" "${VLIB_CACHE_DIR}/data.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lcube "${LCUBE}" \
        --Lz-min "${LZ_MIN}" \
        --nsteps "${NSTEPS}" \
        --nmesh "${NMESH}" \
        --lpt "${LPT}" \
        --z "${Z}" \
        --method "${METHOD}" \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/data.npz"
    vlib::step_done "run"
fi

# --------------------------------------------------------------------------
if vlib::step_check "plot" "${OUTPUT}/squish.pdf"; then
    # shellcheck disable=SC2086
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/data.npz" \
        --ylim ${YLIM} \
        --percent-band "${PERCENT_BAND}" \
        --title "" \
        -o "${OUTPUT}/squish.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/data.npz" \
        --figure "${OUTPUT}/squish.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
