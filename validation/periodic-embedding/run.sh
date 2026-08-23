#!/usr/bin/env bash
# ============================================================================
#  Measure how Fourier-box padding affects a non-periodic domain.
#
#  The campaign varies alpha = LBOX/(2*R_3D) while keeping voxel size and
#  white noise inside the domain fixed. It compares each box with the largest:
#    - displacement/velocity error in radial shells,
#    - correlation between opposite sides of the boundary shell,
#    - central-region mass-weighted P(k),
#    - mass remaining inside the domain and radial motion at its boundary.
#  The reported values combine NSEEDS random fields.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE       small, medium (default), or approved large profile
#    --step=N          Start from step N (default 1; steps: run, plot)
#    --plot-only       Skip the run step; regenerate the PDF from cache
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Select a typed custom TOML profile
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Configuration (edit config.env or export before running):
#    GEOMETRY          spherical | cylindrical       (default: spherical)
#    R_3D, D_4D, RCRIT, BIN_MODE, NRBINS, NSHELL, LZ
#    NGRID0            Voxels across 2*R_3D          (default: 128)
#    ALPHAS            Padding ratios, ascending     (default: 1.0 .. 2.0)
#    LPT, Z, SEED, NSEEDS, LOAD_SEED
#    BOUNDARY_FRAC, CORE_FRAC, PK_NMESH, NPROF
#    STEPSIC_ENV       Conda env for stepsic         (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign periodic-embedding "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps run plot evaluate
vlib::manifest_implementation run "${BASEDIR}/scripts/run.py"
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
FIGURES="${OUTPUT}/supplementary"

# --clean: wipe cache and output before running
if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${FIGURES}"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Padding validation: periodic-embedding error vs alpha"
echo "========================================================================"
echo ""
echo "  geometry:    ${GEOMETRY}   R_3D: ${R_3D} Mpc/h   RCRIT: ${RCRIT}"
echo "  ngrid0:      ${NGRID0}   alphas: ${ALPHAS}"
echo "  lpt:         ${LPT}   z: ${Z}   seeds: ${SEED}+${NSEEDS}"
echo "  cache:       ${VLIB_CACHE_DIR}"
echo "  output:      ${FIGURES}/periodic-embedding.pdf"
echo ""

# --------------------------------------------------------------------------
if vlib::step_check "run" "${VLIB_CACHE_DIR}/data.npz"; then
    # shellcheck disable=SC2086
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --geometry "${GEOMETRY}" \
        --R3d "${R_3D}" \
        --D4d "${D_4D}" \
        --rcrit "${RCRIT}" \
        --bin-mode "${BIN_MODE}" \
        --nrbins "${NRBINS}" \
        --nshell "${NSHELL}" \
        --Lz "${LZ}" \
        --ngrid0 "${NGRID0}" \
        --alphas ${ALPHAS} \
        --lpt "${LPT}" \
        --z "${Z}" \
        --seed "${SEED}" \
        --nseeds "${NSEEDS}" \
        --load-seed "${LOAD_SEED}" \
        --boundary-frac "${BOUNDARY_FRAC}" \
        --core-frac "${CORE_FRAC}" \
        --pk-nmesh "${PK_NMESH}" \
        --nprof "${NPROF}" \
        -o "${VLIB_CACHE_DIR}/data.npz"
    vlib::step_done "run"
fi

# --------------------------------------------------------------------------
if vlib::step_check "plot" "${FIGURES}/periodic-embedding.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/data.npz" \
        -o "${FIGURES}/periodic-embedding.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/data.npz" \
        --figure "${FIGURES}/periodic-embedding.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
