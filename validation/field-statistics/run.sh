#!/usr/bin/env bash
# ============================================================================
#  LPT displacement and velocity field statistics.
#
#  Generates cubic LPT-displaced particles without periodic wrapping.
#
#    output/supplementary/field-statistics.pdf      2LPT field histograms
#    output/supplementary/field-statistics-lpt.pdf  1LPT/2LPT comparison
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE       small, medium (default), or approved large profile
#    --step=N          Start from step N (default 1)
#    --plot-only       Regenerate PDFs from cached .npz files
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Select a typed custom TOML profile
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Steps (in order):
#    1. run_2lpt         - 2LPT displacement and velocity histograms
#    2. run_1lpt         - 1LPT histograms for comparison
#    3. plot_fields      - displacement/velocity distributions
#    4. plot_comparison  - 1LPT vs 2LPT overlay
#    5. evaluate         - isotropy and histogram integrity
#
#  Configuration (edit config.env or export before running):
#    LBOX_3D           3D box [Lx Ly Lz] in Mpc/h    (default: 1000 1000 1000)
#    NMESH_3D          Grid cells, 3D box             (default: 128)
#    Z                 Target redshift                (default: 31)
#    METHOD            Mass-assignment scheme         (default: cic)
#    SEED              RNG seed                       (default: 137)
#    NREAL             Realisations to average        (default: 200)
#    NBINS             Histogram bins                 (default: 120)
#    STEPSIC_ENV       Conda env for stepsic          (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign field-statistics "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps run_2lpt run_1lpt plot_fields plot_comparison evaluate
for _step in run_2lpt run_1lpt; do
    vlib::manifest_implementation "${_step}" "${BASEDIR}/scripts/run.py"
done
for _step in plot_fields plot_comparison; do
    vlib::manifest_implementation "${_step}" "${BASEDIR}/scripts/plot.py"
done
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache"
OUTPUT="${VLIB_RUN_ROOT}/output"
FIGURES="${OUTPUT}/supplementary"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${FIGURES}"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Field validation: displacement and velocity distributions"
echo "========================================================================"
echo ""
echo "  3D box:   ${LBOX_3D} Mpc/h   nmesh: ${NMESH_3D}"
echo "  z: ${Z}   method: ${METHOD}   nreal: ${NREAL}   nbins: ${NBINS}"
echo "  cache:  ${VLIB_CACHE_DIR}"
echo "  output: ${OUTPUT}"
echo ""

# shellcheck disable=SC2086
if vlib::step_check "run_2lpt" "${VLIB_CACHE_DIR}/2lpt.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox ${LBOX_3D} \
        --nmesh "${NMESH_3D}" \
        --lpt 2 \
        --z "${Z}" \
        --method "${METHOD}" \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        --nbins "${NBINS}" \
        -o "${VLIB_CACHE_DIR}/2lpt.npz"
    vlib::step_done "run_2lpt"
fi

# shellcheck disable=SC2086
if vlib::step_check "run_1lpt" "${VLIB_CACHE_DIR}/1lpt.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox ${LBOX_3D} \
        --nmesh "${NMESH_3D}" \
        --lpt 1 \
        --z "${Z}" \
        --method "${METHOD}" \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        --nbins "${NBINS}" \
        -o "${VLIB_CACHE_DIR}/1lpt.npz"
    vlib::step_done "run_1lpt"
fi

if vlib::step_check "plot_fields" "${FIGURES}/field-statistics.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/2lpt.npz" \
        -o "${FIGURES}/field-statistics.pdf"
    vlib::step_done "plot_fields"
fi

if vlib::step_check "plot_comparison" "${FIGURES}/field-statistics-lpt.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/1lpt.npz" "${VLIB_CACHE_DIR}/2lpt.npz" \
        --labels "1LPT" "2LPT" \
        -o "${FIGURES}/field-statistics-lpt.pdf"
    vlib::step_done "plot_comparison"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/2lpt.npz" \
        --figure "${FIGURES}/field-statistics.pdf" \
        --figure "${FIGURES}/field-statistics-lpt.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
