#!/usr/bin/env bash
# ============================================================================
#  Measure P(k) recovery on cubic grids.
#
#  Each panel isolates one parameter dimension while holding the others
#  fixed. All panels use Angulo & Pontzen (2016) paired-fixed averaging.
#
#    Panel A: resolution dependence   (multiple nmesh, fixed z/LPT/MAS)
#    Panel B: redshift dependence     (multiple z,     fixed nmesh/LPT/MAS)
#    Panel C: LPT order comparison    (1LPT vs 2LPT,   fixed nmesh/z/MAS)
#    Panel D: MAS scheme comparison   (NGP/CIC/TSC,    fixed nmesh/z/LPT)
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE       small, medium (default), or approved large profile
#    --step=N          Start from step N (default 1)
#    --plot-only       Regenerate the PDF from cached .npz files
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Select a typed custom TOML profile
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Steps (in order):
#    1. panel_a  - resolution dependence
#    2. panel_b  - redshift dependence
#    3. panel_c  - LPT order comparison
#    4. panel_d  - MAS scheme comparison
#    5. plot     - four-panel P(k) ratio figure
#
#  Configuration (edit config.env or export before running):
#    LBOX              Box side [Mpc/h]         (default: 1000)
#    SEED              RNG seed                 (default: 137)
#    PAIRED            Paired-fixed averaging   (default: true)
#    KMAX_FRAC_NY      k_max / k_Nyquist        (default: 0.8)
#    PANEL_A_NMESH     Panel A nmesh values     (default: 128 256 512)
#    PANEL_A_LPT       Panel A LPT order        (default: 2)
#    PANEL_A_Z         Panel A redshift         (default: 31)
#    PANEL_A_METHOD    Panel A MAS scheme       (default: cic)
#    PANEL_B_NMESH … PANEL_D_METHOD  (see config.env for all panel params)
#    YLIM              Plot y-axis limits       (default: 0.99 1.01)
#    STEPSIC_ENV       Conda env                (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign grid-power-recovery "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps panel_a panel_b panel_c panel_d plot evaluate
for _step in panel_a panel_b panel_c panel_d; do
    vlib::manifest_implementation "${_step}" "${BASEDIR}/scripts/run.py"
done
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

PAIRED_FLAG=()
if [[ "${PAIRED:-false}" == "true" ]]; then
    PAIRED_FLAG=(--paired)
fi

echo ""
echo "========================================================================"
echo "  Grid P(k) recovery"
echo "========================================================================"
echo ""
echo "  Lbox: ${LBOX} Mpc/h   seed: ${SEED}   paired: ${PAIRED}"
echo "  cache:  ${VLIB_CACHE_DIR}"
echo "  output: ${OUTPUT}/grid.pdf"
echo ""

# shellcheck disable=SC2086
if vlib::step_check "panel_a" "${VLIB_CACHE_DIR}/panel_a.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox "${LBOX}" \
        --nmesh ${PANEL_A_NMESH} \
        --lpt ${PANEL_A_LPT} \
        --z ${PANEL_A_Z} \
        --method ${PANEL_A_METHOD} \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/panel_a.npz"
    vlib::step_done "panel_a"
fi

# shellcheck disable=SC2086
if vlib::step_check "panel_b" "${VLIB_CACHE_DIR}/panel_b.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox "${LBOX}" \
        --nmesh ${PANEL_B_NMESH} \
        --lpt ${PANEL_B_LPT} \
        --z ${PANEL_B_Z} \
        --method ${PANEL_B_METHOD} \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/panel_b.npz"
    vlib::step_done "panel_b"
fi

# shellcheck disable=SC2086
if vlib::step_check "panel_c" "${VLIB_CACHE_DIR}/panel_c.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox "${LBOX}" \
        --nmesh ${PANEL_C_NMESH} \
        --lpt ${PANEL_C_LPT} \
        --z ${PANEL_C_Z} \
        --method ${PANEL_C_METHOD} \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/panel_c.npz"
    vlib::step_done "panel_c"
fi

# shellcheck disable=SC2086
if vlib::step_check "panel_d" "${VLIB_CACHE_DIR}/panel_d.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --Lbox "${LBOX}" \
        --nmesh ${PANEL_D_NMESH} \
        --lpt ${PANEL_D_LPT} \
        --z ${PANEL_D_Z} \
        --method ${PANEL_D_METHOD} \
        --seed "${SEED}" \
        --nreal "${NREAL}" \
        "${PAIRED_FLAG[@]+"${PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" \
        -o "${VLIB_CACHE_DIR}/panel_d.npz"
    vlib::step_done "panel_d"
fi

# shellcheck disable=SC2086
if vlib::step_check "plot" "${OUTPUT}/grid.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        --panel-a "${VLIB_CACHE_DIR}/panel_a.npz" \
            --nmesh-a ${PANEL_A_NMESH} --z-a ${PANEL_A_Z} \
        --panel-b "${VLIB_CACHE_DIR}/panel_b.npz" \
            --nmesh-b ${PANEL_B_NMESH} --z-b ${PANEL_B_Z} \
        --panel-c "${VLIB_CACHE_DIR}/panel_c.npz" \
            --nmesh-c ${PANEL_C_NMESH} --z-c ${PANEL_C_Z} \
        --panel-d "${VLIB_CACHE_DIR}/panel_d.npz" \
            --nmesh-d ${PANEL_D_NMESH} --z-d ${PANEL_D_Z} \
            --methods-d ${PANEL_D_METHOD} \
        --ylim ${YLIM} \
        -o "${OUTPUT}/grid.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/panel_a.npz" \
        --archive "${VLIB_CACHE_DIR}/panel_b.npz" \
        --archive "${VLIB_CACHE_DIR}/panel_c.npz" \
        --archive "${VLIB_CACHE_DIR}/panel_d.npz" \
        --figure "${OUTPUT}/grid.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
