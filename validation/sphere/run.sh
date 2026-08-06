#!/usr/bin/env bash
# ============================================================================
#  validation/sphere/run.sh - IC-level fair-sample validation (IC level)
#
#  Generates, with an identical Fourier box, seed, cosmology, and
#  redshift, (a) a periodic cubical grid-load IC and (b) an embedded
#  spherical (or cylindrical) shell-load IC, so that both share their
#  Gaussian field mode-for-mode. The comparison inside the central
#  constant-resolution region then isolates the geometry handling
#  (embedding, multi-resolution mixing, off-grid interpolation):
#    - mass-weighted velocity-field residual on a common grid,
#    - windowed core P(k) ratio,
#    - per-component velocity variances.
#
#  The evolved comparison against an N-body reference lives
#  in evolved.sh; this driver is laptop-scale.
#
#  Pipeline steps:
#    1. ic_ref   - periodic cubical grid IC          (stepsic)
#    2. ic_geom  - embedded spherical/cylindrical IC (stepsic)
#    3. compare  - field/pk/variance comparison
#    4. plot     - IC-level figure
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options (standard vlib):
#    --step=N / --plot-only / --force / --force-step=A,B / --clean
#    --config=PATH / --list-steps / -h
#
#  Configuration (edit config.env or export before running):
#    GEOMETRY (spherical|cylindrical), R_3D, D_4D, RCRIT, BIN_MODE,
#    NRBINS, NSHELL, LZ, ALPHA, NMESH, NMESHSAMPLES, LPT, Z, SEED,
#    RCORE, NCMP, PK_NMESH, NPROF, STEPSIC_ENV
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

vlib::declare_steps ic_ref ic_geom compare plot evaluate
if (( VLIB_LIST_STEPS )); then
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

VLIB_CACHE_DIR="${BASEDIR}/cache"
OUTPUT="${BASEDIR}/output"
PARAM_DIR="${BASEDIR}/configs"
IC_REF_DIR="${VLIB_CACHE_DIR}/ic_ref"
IC_GEOM_DIR="${VLIB_CACHE_DIR}/ic_geom"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi
mkdir -p "${VLIB_CACHE_DIR}" "${OUTPUT}" "${PARAM_DIR}" \
         "${IC_REF_DIR}" "${IC_GEOM_DIR}"

# Shared Fourier box: LBOX = ALPHA * 2 * R_3D on the non-periodic axes.
LBOX_SIDE="$(awk "BEGIN {printf \"%.10g\", ${ALPHA} * 2.0 * ${R_3D}}")"
if [[ "${GEOMETRY}" == "spherical" ]]; then
    LBOX_TOML="[${LBOX_SIDE}, ${LBOX_SIDE}, ${LBOX_SIDE}]"
    PERIODIC_GEOM="[0, 0, 0]"
else
    LBOX_TOML="[${LBOX_SIDE}, ${LBOX_SIDE}, ${LZ}]"
    PERIODIC_GEOM="[0, 0, 1]"
fi


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  IC-level fair-sample validation: ${GEOMETRY} vs periodic cube"
echo "========================================================================"
echo ""
echo "  R_3D: ${R_3D}   RCRIT: ${RCRIT}   alpha: ${ALPHA}  -> LBOX: ${LBOX_TOML}"
echo "  nmesh: ${NMESH} (samples: ${NMESHSAMPLES})   lpt: ${LPT}   z: ${Z}   seed: ${SEED}"
echo "  core: r < ${RCORE}   cmp grid: ${NCMP}^3"
echo ""

# -- Step 1: ic_ref (periodic cubical grid load) -----------------------------
if vlib::step_check "ic_ref" "${IC_REF_DIR}"; then
    vlib::clear_dir "${IC_REF_DIR}"
    vlib::stepsic::write_toml "${PARAM_DIR}/ic_ref.toml" \
        "GEOMETRY=cubical" \
        "LBOX=${LBOX_TOML}" \
        "PERIODIC=[1, 1, 1]" \
        "TYPE=grid" \
        "NGRID=${NMESH}" \
        "IC_DIR=${IC_REF_DIR}" \
        "SEED=${SEED}" \
        "LPTORDER=${LPT}" \
        "NMESH=${NMESH}" \
        "REDSHIFT=${Z}"
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${PARAM_DIR}/ic_ref.toml"
    vlib::breadcrumb_set "IC_REF" "$(vlib::find_ic "${IC_REF_DIR}")"
    vlib::step_done "ic_ref"
fi

# -- Step 2: ic_geom (embedded shell load, shared field) ---------------------
if vlib::step_check "ic_geom" "${IC_GEOM_DIR}"; then
    vlib::clear_dir "${IC_GEOM_DIR}"
    vlib::stepsic::write_toml "${PARAM_DIR}/ic_geom.toml" \
        "GEOMETRY=${GEOMETRY}" \
        "LBOX=${LBOX_TOML}" \
        "PERIODIC=${PERIODIC_GEOM}" \
        "R_3D=${R_3D}" \
        "D_4D=${D_4D}" \
        "RCRIT=${RCRIT}" \
        "BIN_MODE=${BIN_MODE}" \
        "NRBINS=${NRBINS}" \
        "TYPE=shell" \
        "NSHELL=${NSHELL}" \
        "IC_DIR=${IC_GEOM_DIR}" \
        "SEED=${SEED}" \
        "LPTORDER=${LPT}" \
        "NMESH=${NMESH}" \
        "NMESHSAMPLES=${NMESHSAMPLES}" \
        "REDSHIFT=${Z}"
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${PARAM_DIR}/ic_geom.toml"
    vlib::breadcrumb_set "IC_GEOM" "$(vlib::find_ic "${IC_GEOM_DIR}")"
    vlib::step_done "ic_geom"
fi

# -- Step 3: compare ---------------------------------------------------------
if vlib::step_check "compare" "${VLIB_CACHE_DIR}/compare.npz"; then
    IC_REF="$(vlib::find_ic "${IC_REF_DIR}")"
    IC_GEOM="$(vlib::find_ic "${IC_GEOM_DIR}")"
    CMP_ARGS=(--geometry "${GEOMETRY}" --lbox "${LBOX_SIDE}"
              --rcore "${RCORE}" --ncmp "${NCMP}"
              --pk-nmesh "${PK_NMESH}" --nprof "${NPROF}")
    if [[ "${GEOMETRY}" == "cylindrical" ]]; then
        CMP_ARGS+=(--Lz "${LZ}")
    fi
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/compare-ic.py" \
        --geom-ic "${IC_GEOM}" --ref-ic "${IC_REF}" \
        "${CMP_ARGS[@]}" \
        -o "${VLIB_CACHE_DIR}/compare.npz"
    vlib::step_done "compare"
fi

# -- Step 4: plot ------------------------------------------------------------
if vlib::step_check "plot" "${OUTPUT}/sphere-ic.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-ic.py" \
        -i "${VLIB_CACHE_DIR}/compare.npz" \
        -o "${OUTPUT}/sphere-ic.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/compare.npz" \
        --figure "${OUTPUT}/sphere-ic.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
