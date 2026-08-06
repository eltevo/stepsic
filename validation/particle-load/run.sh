#!/usr/bin/env bash
# ============================================================================
#  validation/particle-load/run.sh - particle load 4-panel 3D scatter figure
#
#  Generates the four particle load types supported by stepsic at LPTORDER=0
#  (no StePS glass relaxation) and plots them as a 4-panel 3D scatter figure.
#
#    cubic_random  - uniformly random positions in a periodic cube
#    cubic_grid    - regular SC lattice in a periodic cube
#    spherical     - omega-binned shells in a non-periodic sphere
#    cylindrical   - omega-binned shells in a non-periodic cylinder
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N          Start from step N (default 1)
#    --plot-only       Skip IC generation; regenerate the PDF from cached ICs
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Source an alternative config.env
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Steps (in order):
#    1. cubic_random - generate cubical random IC
#    2. cubic_grid   - generate cubical grid IC
#    3. spherical    - generate spherical shell IC
#    4. cylindrical  - generate cylindrical shell IC
#    5. plot         - 4-panel 3D scatter figure
#    6. plot_2d      - 4-panel 2D slice figure (xy plane, z slab)
#
#  Configuration (edit config.env or export before running):
#    LBOX            Cubic box side [Mpc/h]       (default: 500)
#    NGRID           Grid cells per dimension     (default: 24)
#    NPART           Target particle count        (default: 8192)
#    R_3D            Simulation radius [Mpc/h]    (default: 250)
#    D_4D            Compactification D [Mpc/h]   (default: 35)
#    NRBINS          Radial bins                  (default: 128)
#    NSHELL          Particles per shell          (default: 512)
#    RCRIT           Constant-res. radius [Mpc/h] (default: 100)
#    LZ              Cylinder height [Mpc/h]      (default: LBOX)
#    STEPSIC_SRC     stepsic source directory     (default: ~/projects/stepsic)
#    STEPSIC_PY      stepsic.py entry point       (default: STEPSIC_SRC/stepsic.py)
#    STEPSIC_ENV     Conda env for stepsic        (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

vlib::declare_steps cubic_random cubic_grid spherical cylindrical plot plot_2d evaluate
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

# IC output directories (each geometry gets its own subdir)
IC_CUBIC_RANDOM="${VLIB_CACHE_DIR}/cubic_random"
IC_CUBIC_GRID="${VLIB_CACHE_DIR}/cubic_grid"
IC_SPHERICAL="${VLIB_CACHE_DIR}/spherical"
IC_CYLINDRICAL="${VLIB_CACHE_DIR}/cylindrical"

TOML_DIR="${VLIB_CACHE_DIR}/configs"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

if [[ ! -f "${STEPSIC_PY}" ]]; then
    echo "ERROR: stepsic.py not found at '${STEPSIC_PY}'." >&2
    echo "  Set STEPSIC_SRC or STEPSIC_PY in config.env or environment." >&2
    exit 1
fi

echo ""
echo "========================================================================"
echo "  Particle load validation: 4-panel 3D scatter figure"
echo "========================================================================"
echo ""
echo "  LBOX: ${LBOX}   R_3D: ${R_3D}   D_4D: ${D_4D}   RCRIT: ${RCRIT}"
echo "  NGRID: ${NGRID}   NPART: ${NPART}   NSHELL: ${NSHELL}"
echo "  cache:  ${VLIB_CACHE_DIR}"
echo "  output: ${OUTPUT}/particle-load.pdf"
echo "          ${OUTPUT}/particle-load-2d.pdf"
echo ""

mkdir -p "${TOML_DIR}" \
    "${IC_CUBIC_RANDOM}" "${IC_CUBIC_GRID}" \
    "${IC_SPHERICAL}"   "${IC_CYLINDRICAL}"

LBOX_SQ="$((LBOX * 2))"  # 2*R_3D for non-periodic geometries

# Write TOML configs (inline because LBOX/PERIODIC are arrays here)
_write_toml_cubic_random() {
    vlib::atomic_text "${TOML_DIR}/cubic_random.toml" <<EOF
GEOMETRY = "cubical"
LBOX = [${LBOX}, ${LBOX}, ${LBOX}]
PERIODIC = [1, 1, 1]
R_3D = ${R_3D}
D_4D = ${D_4D}
BIN_MODE = "omega"
NRBINS = ${NRBINS}
COI = [0, 0, 0]
TYPE = "random"
NGRID = ${NGRID}
NPART = ${NPART}
NSHELL = ${NSHELL}
INPUT_GLASS = "none"
IC_DIR = "${IC_CUBIC_RANDOM}"
IC_PREFIX = "stepsic"
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = 42
COMOVING = true
HINDEPENDENT = true
LPTORDER = 0
NMESH = 32
REDSHIFT = 0
INTERPOLATION = "cic"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
IC_FORMAT = "hdf5"
USE_DOUBLE = false
SAVE_WHITE_NOISE = false
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF
}

_write_toml_cubic_grid() {
    vlib::atomic_text "${TOML_DIR}/cubic_grid.toml" <<EOF
GEOMETRY = "cubical"
LBOX = [${LBOX}, ${LBOX}, ${LBOX}]
PERIODIC = [1, 1, 1]
R_3D = ${R_3D}
D_4D = ${D_4D}
BIN_MODE = "omega"
NRBINS = ${NRBINS}
COI = [0, 0, 0]
TYPE = "grid"
NGRID = ${NGRID}
NPART = ${NPART}
NSHELL = ${NSHELL}
INPUT_GLASS = "none"
IC_DIR = "${IC_CUBIC_GRID}"
IC_PREFIX = "stepsic"
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = 42
COMOVING = true
HINDEPENDENT = true
LPTORDER = 0
NMESH = 32
REDSHIFT = 0
INTERPOLATION = "cic"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
IC_FORMAT = "hdf5"
USE_DOUBLE = false
SAVE_WHITE_NOISE = false
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF
}

_write_toml_spherical() {
    vlib::atomic_text "${TOML_DIR}/spherical.toml" <<EOF
GEOMETRY = "spherical"
LBOX = [${LBOX_SQ}, ${LBOX_SQ}, ${LBOX_SQ}]
PERIODIC = [0, 0, 0]
R_3D = ${R_3D}
D_4D = ${D_4D}
BIN_MODE = "omega"
NRBINS = ${NRBINS}
RCRIT = ${RCRIT}
COI = [0, 0, 0]
TYPE = "shell"
NGRID = ${NGRID}
NPART = ${NPART}
NSHELL = ${NSHELL}
INPUT_GLASS = "none"
IC_DIR = "${IC_SPHERICAL}"
IC_PREFIX = "stepsic"
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = 42
COMOVING = true
HINDEPENDENT = true
LPTORDER = 0
NMESH = 32
REDSHIFT = 0
INTERPOLATION = "cic"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
IC_FORMAT = "hdf5"
USE_DOUBLE = false
SAVE_WHITE_NOISE = false
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF
}

_write_toml_cylindrical() {
    vlib::atomic_text "${TOML_DIR}/cylindrical.toml" <<EOF
GEOMETRY = "cylindrical"
LBOX = [${LBOX_SQ}, ${LBOX_SQ}, ${LZ}]
PERIODIC = [0, 0, 1]
R_3D = ${R_3D}
D_4D = ${D_4D}
BIN_MODE = "omega"
NRBINS = ${NRBINS}
RCRIT = ${RCRIT}
COI = [0, 0, 0]
TYPE = "shell"
NGRID = ${NGRID}
NPART = ${NPART}
NSHELL = ${NSHELL}
INPUT_GLASS = "none"
IC_DIR = "${IC_CYLINDRICAL}"
IC_PREFIX = "stepsic"
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = 42
COMOVING = true
HINDEPENDENT = true
LPTORDER = 0
NMESH = 32
REDSHIFT = 0
INTERPOLATION = "cic"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
IC_FORMAT = "hdf5"
USE_DOUBLE = false
SAVE_WHITE_NOISE = false
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF
}

if vlib::step_check "cubic_random" "${IC_CUBIC_RANDOM}"; then
    _write_toml_cubic_random
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cubic_random.toml"
    vlib::step_done "cubic_random"
fi

if vlib::step_check "cubic_grid" "${IC_CUBIC_GRID}"; then
    _write_toml_cubic_grid
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cubic_grid.toml"
    vlib::step_done "cubic_grid"
fi

if vlib::step_check "spherical" "${IC_SPHERICAL}"; then
    _write_toml_spherical
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/spherical.toml"
    vlib::step_done "spherical"
fi

if vlib::step_check "cylindrical" "${IC_CYLINDRICAL}"; then
    _write_toml_cylindrical
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" \
        "${TOML_DIR}/cylindrical.toml"
    vlib::step_done "cylindrical"
fi

_configured_ic() {
    local directory="${1}" token="${2}"
    shopt -s nullglob
    local matches=("${directory}"/*"${token}"*/ic.hdf5)
    shopt -u nullglob
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "ERROR: expected one configured IC matching ${token} under ${directory}; found ${#matches[@]}." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

_resolve_ic_paths() {
    IC_CR="$(_configured_ic "${IC_CUBIC_RANDOM}" "_Np${NPART}_")"
    IC_CG="$(_configured_ic "${IC_CUBIC_GRID}" "_Ng${NGRID}_")"
    IC_SP="$(_configured_ic "${IC_SPHERICAL}" "_Nsh${NSHELL}_Nr${NRBINS}_")"
    IC_CY="$(_configured_ic "${IC_CYLINDRICAL}" "_Nsh${NSHELL}_Nr${NRBINS}_")"
}

if vlib::step_check "plot" "${OUTPUT}/particle-load.pdf"; then
    _resolve_ic_paths
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        --cubic-random "${IC_CR}" \
        --cubic-grid   "${IC_CG}" \
        --spherical    "${IC_SP}" \
        --cylindrical  "${IC_CY}" \
        --fraction "${PLOT_FRACTION}" \
        -o "${OUTPUT}/particle-load.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "plot_2d" "${OUTPUT}/particle-load-2d.pdf"; then
    _resolve_ic_paths
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        --cubic-random "${IC_CR}" \
        --cubic-grid   "${IC_CG}" \
        --spherical    "${IC_SP}" \
        --cylindrical  "${IC_CY}" \
        --plot-2d \
        --slice-thickness "${SLICE_THICKNESS}" \
        -o "${OUTPUT}/particle-load-2d.pdf"
    vlib::step_done "plot_2d"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    _resolve_ic_paths
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --cubic-random "${IC_CR}" \
        --cubic-grid "${IC_CG}" \
        --spherical "${IC_SP}" \
        --cylindrical "${IC_CY}" \
        --box-size "${LBOX}" \
        --grid-size "${NGRID}" \
        --random-count "${NPART}" \
        --radius "${R_3D}" \
        --cylinder-length "${LZ}" \
        --figure "${OUTPUT}/particle-load.pdf" \
        --figure "${OUTPUT}/particle-load-2d.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
