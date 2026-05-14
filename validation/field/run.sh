#!/usr/bin/env bash
# ============================================================================
#  validation/field/run.sh - displacement and velocity field histogram validation
#
#  Generates LPT-displaced particles WITHOUT periodic wrapping so the full
#  displacement distribution is visible. Produces three output figures:
#
#    output/fields.pdf          3-panel |Ψ_x|, |Ψ|, |v| histograms (2LPT)
#    output/comparison.pdf      1LPT vs 2LPT histogram overlay
#    output/slab_anisotropy.pdf x vs z displacement ratio for slab geometry
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N          Start from step N (default 1)
#    --plot-only       Regenerate PDFs from cached .npz files
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Source an alternative config.env
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Steps (in order):
#    1. run_2lpt   - 200-realisation 2LPT displacement/velocity histograms
#    2. run_1lpt   - 200-realisation 1LPT histograms (for comparison)
#    3. run_slab   - 200-realisation slab box histograms
#    4. plot_fields      - validation-fields PDF (2LPT only)
#    5. plot_comparison  - 1LPT vs 2LPT overlay PDF
#    6. plot_slab        - slab anisotropy PDF
#
#  Configuration (edit config.env or export before running):
#    LBOX_3D           3D box [Lx Ly Lz] in Mpc/h    (default: 1000 1000 1000)
#    LBOX_SLAB         Slab box [Lx Ly Lz] in Mpc/h  (default: 1000 1000 200)
#    NMESH_3D          Grid cells, 3D box             (default: 128)
#    NMESH_SLAB        Grid cells, slab               (default: 64)
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
vlib::source_config "${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"

VLIB_CACHE_DIR="${BASEDIR}/cache"
OUTPUT="${BASEDIR}/output"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${OUTPUT}"

if (( VLIB_LIST_STEPS )); then
    vlib::step_check "run_2lpt"         "${VLIB_CACHE_DIR}/2lpt.npz" || :
    vlib::step_check "run_1lpt"         "${VLIB_CACHE_DIR}/1lpt.npz" || :
    vlib::step_check "run_slab"         "${VLIB_CACHE_DIR}/slab.npz" || :
    vlib::step_check "plot_fields"      "${OUTPUT}/fields.pdf"        || :
    vlib::step_check "plot_comparison"  "${OUTPUT}/comparison.pdf"    || :
    vlib::step_check "plot_slab"        "${OUTPUT}/slab_anisotropy.pdf" || :
    exit 0
fi

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Field validation: displacement and velocity distributions"
echo "========================================================================"
echo ""
echo "  3D box:   ${LBOX_3D} Mpc/h   nmesh: ${NMESH_3D}"
echo "  Slab box: ${LBOX_SLAB} Mpc/h   nmesh: ${NMESH_SLAB}"
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

# shellcheck disable=SC2086
if vlib::step_check "run_slab" "${VLIB_CACHE_DIR}/slab.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox ${LBOX_SLAB} \
        --nmesh "${NMESH_SLAB}" \
        --lpt 2 \
        --z "${Z}" \
        --method "${METHOD}" \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        --nbins "${NBINS}" \
        -o "${VLIB_CACHE_DIR}/slab.npz"
    vlib::step_done "run_slab"
fi

if vlib::step_check "plot_fields" "${OUTPUT}/fields.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/2lpt.npz" \
        -o "${OUTPUT}/fields.pdf"
    vlib::step_done "plot_fields"
fi

if vlib::step_check "plot_comparison" "${OUTPUT}/comparison.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/1lpt.npz" "${VLIB_CACHE_DIR}/2lpt.npz" \
        --labels "1LPT" "2LPT" \
        -o "${OUTPUT}/comparison.pdf"
    vlib::step_done "plot_comparison"
fi

if vlib::step_check "plot_slab" "${OUTPUT}/slab_anisotropy.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot_slab.py" \
        -i "${VLIB_CACHE_DIR}/slab.npz" \
        -o "${OUTPUT}/slab_anisotropy.pdf"
    vlib::step_done "plot_slab"
fi

vlib::report_done
