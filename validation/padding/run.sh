#!/usr/bin/env bash
# ============================================================================
#  validation/padding/run.sh - periodic-embedding error vs bounding-box padding
#
#  Sweeps the padding ratio alpha = LBOX/(2*R_3D) of the periodic Fourier
#  box around a non-periodic (spherical or cylindrical) domain, holding
#  the voxel size and the physical white-noise realization fixed (the
#  smaller boxes crop the reference box's real-space noise). Measures,
#  against the largest-alpha reference:
#    - displacement/velocity error in radial shells,
#    - antipodal boundary-shell correlation (periodic-image signature),
#    - central-region mass-weighted P(k),
#    - post-displacement monopole diagnostics (mass flux, boundary drift),
#  ensembled over NSEEDS realizations.
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
echo "  Padding validation: periodic-embedding error vs alpha"
echo "========================================================================"
echo ""
echo "  geometry:    ${GEOMETRY}   R_3D: ${R_3D} Mpc/h   RCRIT: ${RCRIT}"
echo "  ngrid0:      ${NGRID0}   alphas: ${ALPHAS}"
echo "  lpt:         ${LPT}   z: ${Z}   seeds: ${SEED}+${NSEEDS}"
echo "  cache:       ${VLIB_CACHE_DIR}"
echo "  output:      ${OUTPUT}/padding.pdf"
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
if vlib::step_check "plot" "${OUTPUT}/padding.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/data.npz" \
        -o "${OUTPUT}/padding.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/data.npz" \
        --figure "${OUTPUT}/padding.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
