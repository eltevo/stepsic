#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  validation/monofonic/run.sh - stepsic <-> monofonIC cross-validation
#
#  Generates matched periodic cubic ICs with both stepsic and monofonIC
#  from the SAME white noise field, then compares three diagnostics:
#    (a) Power spectrum ratio       P_stepsic(k) / P_monofonic(k)
#    (b) Per-particle displacement residuals   (Ψ_stepsic - Ψ_monofonic)
#    (c) Per-particle velocity residuals       (v_stepsic - v_monofonic)
#
#  White-noise sharing:
#    stepsic saves the real-space noise to ic_white_noise.hdf5 (dataset
#    'ic_white_noise'). monofonIC reads it via ConstraintFieldFile /
#    ConstraintFieldName - no format conversion needed.
#
#  Transfer function:
#    By default, stepsic's CAMB-generated P(k) is exported and fed to
#    monofonIC via `transfer = CAMB_file` to isolate LPT differences from
#    Boltzmann-solver differences. Pass --use-class to let monofonIC use
#    its built-in CLASS instead (introduces a ~0.1% baseline from CAMB/CLASS
#    differences).
#
#  Pipeline steps:
#    1. run_stepsic     - IC + white noise + CAMB P(k) export
#    2. build_monofonic - clone (if needed) + cmake + compile
#    3. run_monofonic   - write config + run monofonIC
#    4. compare         - headless: P(k), Ψ, v -> .npz archive
#    5. plot            - 3-panel comparison figure
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N       Start from step N (1–5, default 1)
#    --plot-only    Jump to step 5 (plotting only)
#    --force        Re-run all steps regardless of cached outputs
#    --clean        Delete cache/ before running
#    --config=PATH  Source an alternative config.env
#    --list-steps   Print the step list and exit
#    -h, --help     Print this help text and exit
#
#  Options (monofonic-specific, pass as extra args):
#    --skip-build   Skip step 2 (assume monofonIC binary already exists)
#    --use-class    Use CLASS instead of CAMB_file for the transfer function
#
#  Configuration (edit config.env or export before running):
#    STEPSIC_SRC, STEPSIC_PY, STEPSIC_ENV
#    MONOFONIC_REPO, MONOFONIC_ENV
#    LBOX, NMESH, Z_INIT, LPT_ORDER, MAS_METHOD, SEED, NUM_THREADS
#    COSMO_H0, COSMO_OMEGA_M, COSMO_OMEGA_B, COSMO_OMEGA_L,
#    COSMO_NS, COSMO_AS, COSMO_SIGMA8, COSMO_TCMB, COSMO_YHE,
#    COSMO_MNU, COSMO_KPIVOT, COSMO_W0, COSMO_WA, COSMO_NUR
#
#  Examples:
#    # Full pipeline from scratch
#    bash run.sh
#
#    # Skip monofonIC build (already compiled) and rerun from step 3
#    bash run.sh --skip-build --step=3
#
#    # Replot from the cached .npz archive
#    bash run.sh --plot-only
# ============================================================================

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

vlib::declare_steps run_stepsic build_monofonic run_monofonic compare plot evaluate
if (( VLIB_LIST_STEPS )); then
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

# Parse monofonic-specific flags from VLIB_EXTRA_ARGS.
SKIP_BUILD=0
USE_CLASS=0

for _arg in "${VLIB_EXTRA_ARGS[@]+"${VLIB_EXTRA_ARGS[@]}"}"; do
    case "${_arg}" in
        --skip-build) SKIP_BUILD=1 ;;
        --use-class)  USE_CLASS=1  ;;
        *)
            echo "ERROR: Unknown argument '${_arg}'." >&2
            echo "Usage: $0 [--skip-build] [--use-class] [standard options...]" >&2
            exit 1
            ;;
    esac
done

# -- Directory layout --------------------------------------------------------
export VLIB_CACHE_DIR="${VLIB_CACHE_DIR:-${BASEDIR}/cache}"
CONFIGS_DIR="${BASEDIR}/configs"
OUTPUT="${BASEDIR}/output"
export CONFIGS_DIR

STEPSIC_CACHE="${VLIB_CACHE_DIR}/stepsic"
MONO_CACHE="${VLIB_CACHE_DIR}/monofonic"

# monofonIC source and build live under ${BASEDIR}/monofonic/.
MONOFONIC_DIR="${MONOFONIC_DIR:-${BASEDIR}/monofonic}"
MONOFONIC_BUILD="${MONOFONIC_BUILD:-${MONOFONIC_DIR}/build}"
MONOFONIC_BIN="${MONOFONIC_BUILD}/monofonIC"

# Stable paths for outputs.
STEPSIC_TOML="${CONFIGS_DIR}/stepsic_monofonic.toml"
MONOFONIC_CONF="${CONFIGS_DIR}/monofonic.conf"
STEPSIC_IC="${STEPSIC_CACHE}/ic.hdf5"
STEPSIC_NOISE="${STEPSIC_CACHE}/ic_white_noise.hdf5"
CAMB_PK_FILE="${STEPSIC_CACHE}/camb_transfer.dat"
MONOFONIC_IC="${MONO_CACHE}/monofonic_ics.hdf5"
COMPARISON_NPZ="${OUTPUT}/monofonic_comparison.npz"
PLOT_OUTPUT="${OUTPUT}/validation-monofonic.pdf"

mkdir -p "${CONFIGS_DIR}" "${OUTPUT}" "${STEPSIC_CACHE}" "${MONO_CACHE}"


# -- Conda envs --------------------------------------------------------------
vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

# -- Banner ------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  stepsic <-> monofonIC cross-validation"
echo "========================================================================"
echo ""
echo "  Box:         L = ${LBOX} Mpc/h  (cubic, fully periodic)"
echo "  Mesh:        ${NMESH}^3, LPT order ${LPT_ORDER}"
echo "  Redshift:    z = ${Z_INIT}"
echo "  MAS (P(k)):  ${MAS_METHOD},  seed = ${SEED}"
if (( USE_CLASS )); then
    echo "  Transfer:    CLASS (--use-class)"
else
    echo "  Transfer:    CAMB_file (from stepsic export)"
fi
echo "  Cosmology:   Planck 2018 EE+BAO (table 2.18)  H0=${COSMO_H0}"
echo ""

# -- Helpers -----------------------------------------------------------------

# Write the stepsic TOML for the monofonic cross-validation.
# Uses USE_DOUBLE=true and HINDEPENDENT=true (h-dependent units = Mpc/h).
_write_stepsic_toml() {
    vlib::atomic_text "${STEPSIC_TOML}" <<EOF
GEOMETRY = "cubical"
LBOX = [${LBOX}, ${LBOX}, ${LBOX}]
PERIODIC = [1, 1, 1]
R_3D = 0
D_4D = 0
RCRIT = 0
BIN_MODE = "omega"
NRBINS = 1
COI = [0, 0, 0]
TYPE = "grid"
NGRID = ${NMESH}
NPART = 0
NSHELL = 0
INPUT_GLASS = "none"
IC_DIR = "${STEPSIC_CACHE}"
IC_PREFIX = "stepsic"
IC_FORMAT = "hdf5"
SAVE_WHITE_NOISE = true
USE_DOUBLE = true
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = ${SEED}
COMOVING = true
HINDEPENDENT = true
LPTORDER = ${LPT_ORDER}
NMESH = ${NMESH}
REDSHIFT = ${Z_INIT}
INTERPOLATION = "${MAS_METHOD}"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
FIXED = false
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF
}

# Enforce the grid_fft.cc reset ordering required for valid FFT dimensions.
_prepare_monofonic_source() {
    local source_path="${MONOFONIC_DIR}/src/grid_fft.cc"
    if [[ ! -f "${source_path}" ]]; then
        echo "  WARNING: monofonIC source file not found: ${source_path}" >&2
        return 0
    fi
    python3 - "${source_path}" <<\PYEOF
import os
from pathlib import Path
import re
import sys
import tempfile

source = Path(sys.argv[1])
text = source.read_text()
required_pattern = re.compile(
    r"this->reset\(\);\s*\n(\s*)for \(size_t i = 0; i < 3; \+\+i\)"
)
if required_pattern.search(text):
    print(f"source invariant satisfied: {source}")
    raise SystemExit(0)

source_pattern = (
    r"([ \t]+)(for \(size_t i = 0; i < 3; \+\+i\)\n"
    r"[ \t]+this->n_\[i\] = dimsize\[i\];\n"
    r"[ \t]+this->space_ = rspace_id;\n"
    r"\n)"
    r"[ \t]+this->reset\(\);\n"
)
required_block = r"\1this->reset();\n\1\2"
updated, count = re.subn(source_pattern, required_block, text)
if count != 1:
    raise RuntimeError(
        f"expected one grid_fft reset-ordering target, found {count}"
    )

descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{source.name}.tmp-", dir=source.parent,
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(updated)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, source)
    directory_fd = os.open(source.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
print(f"source prepared: {source}")
PYEOF
}

# Create the monofonIC conda build environment if it doesn't exist.
_ensure_monofonic_env() {
    if conda env list | awk '{print $1}' | grep -Fxq "${MONOFONIC_ENV}"; then
        return 0
    fi
    echo "  Creating conda env '${MONOFONIC_ENV}' with build dependencies..."
    conda create --yes -n "${MONOFONIC_ENV}" -c conda-forge \
        cmake compilers openmpi "fftw=*=mpi_openmpi_*" \
        gsl "hdf5=*=mpi_openmpi_*" pkg-config
    echo "  -> ${MONOFONIC_ENV} created."
}

# Remove the monofonIC build environment; it is only needed to compile the
# binary and is unnecessary once the build succeeds.
_cleanup_monofonic_env() {
    if ! conda env list | awk '{print $1}' | grep -Fxq "${MONOFONIC_ENV}"; then
        return 0
    fi
    echo "  Removing build env '${MONOFONIC_ENV}'..."
    conda env remove --yes -n "${MONOFONIC_ENV}"
    echo "  -> ${MONOFONIC_ENV} removed."
}

# ============================================================================
# Steps
# ============================================================================

# -- Step 1: run_stepsic -----------------------------------------------------
if vlib::step_check "run_stepsic" "${STEPSIC_IC}"; then
    vlib::clear_dir "${STEPSIC_CACHE}"
    _write_stepsic_toml
    echo "  Config: ${STEPSIC_TOML}"
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" "${STEPSIC_TOML}"

    # stepsic writes into a named subdirectory; create stable symlinks here.
    _run_dir="$(find "${STEPSIC_CACHE}" -maxdepth 1 \
        -name "stepsic_*Ng${NMESH}_Nm${NMESH}_*" -type d \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -z "${_run_dir}" ]]; then
        echo "ERROR: stepsic did not produce an output directory matching Ng=${NMESH}." >&2
        exit 1
    fi
    echo "  Output dir: ${_run_dir}"
    ln -sf "${_run_dir}/ic.hdf5"             "${STEPSIC_IC}"
    ln -sf "${_run_dir}/ic_white_noise.hdf5" "${STEPSIC_NOISE}"
    vlib::breadcrumb_set "STEPSIC_RUN_DIR" "${_run_dir}"

    if ! (( USE_CLASS )); then
        echo "  Exporting CAMB linear transfer function at z=0..."
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/export-stepsic-pk.py" \
            --cosmology "${COSMOLOGY_NAME}" \
            --lbox "${LBOX}" \
            -o "${CAMB_PK_FILE}"
        echo "  -> ${CAMB_PK_FILE}"
        vlib::breadcrumb_set "CAMB_PK_FILE" "${CAMB_PK_FILE}"
    fi

    vlib::step_done "run_stepsic"
fi

# -- Step 2: build_monofonic -------------------------------------------------
if (( SKIP_BUILD )); then
    if [[ ! -f "${MONOFONIC_BIN}" ]]; then
        echo "ERROR: --skip-build specified but binary not found: ${MONOFONIC_BIN}" >&2
        exit 1
    fi
    echo "  build_monofonic - skipped (--skip-build; binary: ${MONOFONIC_BIN})"
elif vlib::step_check "build_monofonic" "${MONOFONIC_BIN}"; then
    if [[ ! -d "${MONOFONIC_DIR}" ]]; then
        echo "  Cloning monofonIC from ${MONOFONIC_REPO}..."
        git clone --recursive "${MONOFONIC_REPO}" "${MONOFONIC_DIR}"
    else
        echo "  Using clone: ${MONOFONIC_DIR}"
    fi

    _prepare_monofonic_source
    _ensure_monofonic_env

    echo "  Configuring with CMake..."
    rm -rf "${MONOFONIC_BUILD}"
    mkdir -p "${MONOFONIC_BUILD}"
    _mono_prefix="$(vlib::run_shell_in_env "${MONOFONIC_ENV}" 'echo "${CONDA_PREFIX}"')"

    vlib::run_in_env "${MONOFONIC_ENV}" cmake \
        -S "${MONOFONIC_DIR}" \
        -B "${MONOFONIC_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${_mono_prefix}" \
        -DENABLE_PANPHASIA=OFF \
        -DENABLE_CLASS=ON \
        -DBUILD_TESTING=OFF

    echo "  Compiling..."
    vlib::run_in_env "${MONOFONIC_ENV}" cmake \
        --build "${MONOFONIC_BUILD}" \
        --parallel "$(nproc)"

    if [[ ! -f "${MONOFONIC_BIN}" ]]; then
        echo "ERROR: monofonIC binary not found after build: ${MONOFONIC_BIN}" >&2
        exit 1
    fi
    echo "  -> ${MONOFONIC_BIN}"
    vlib::step_done "build_monofonic"
fi

# -- Step 3: run_monofonic ---------------------------------------------------
if vlib::step_check "run_monofonic" "${MONOFONIC_IC}"; then
    _ensure_monofonic_env
    if [[ ! -f "${MONOFONIC_BIN}" ]]; then
        echo "ERROR: monofonIC binary not found; run step 2 first." >&2; exit 1
    fi
    if [[ ! -f "${STEPSIC_NOISE}" ]]; then
        echo "ERROR: White noise file not found: ${STEPSIC_NOISE}" >&2
        echo "  Run step 1 first." >&2; exit 1
    fi

    _noise_abs="$(readlink -f "${STEPSIC_NOISE}")"

    if (( USE_CLASS )); then
        _transfer_block="transfer        = CLASS
ztarget         = 0.0"
    else
        _camb="${CAMB_PK_FILE:-$(vlib::breadcrumb_get CAMB_PK_FILE)}"
        if [[ -z "${_camb}" || ! -f "${_camb}" ]]; then
            echo "ERROR: CAMB transfer file not found: ${_camb}" >&2
            echo "  Run step 1 without --use-class, or add --use-class." >&2
            exit 1
        fi
        _camb_abs="$(readlink -f "${_camb}")"
        _transfer_block="transfer        = CAMB_file
transfer_file   = ${_camb_abs}"
    fi

    vlib::clear_dir "${MONO_CACHE}"

    vlib::atomic_text "${MONOFONIC_CONF}" <<EOF
########################################################################
# monofonIC config for stepsic cross-validation
# Auto-generated by validation/monofonic/run.sh
########################################################################

[setup]
GridRes         = ${NMESH}
BoxLength       = ${LBOX}
zstart          = ${Z_INIT}.0
LPTorder        = ${LPT_ORDER}
DoBaryons       = no
DoBaryonVrel    = no
DoFixing        = no
DoInversion     = no
DoRemoveCornerModes = no
ParticleLoad    = sc

[cosmology]
ParameterSet    = none
Omega_m         = ${COSMO_OMEGA_M}
Omega_b         = ${COSMO_OMEGA_B}
Omega_L         = ${COSMO_OMEGA_L}
H0              = ${COSMO_H0}
n_s             = ${COSMO_NS}
A_s             = ${COSMO_AS}
sigma_8         = ${COSMO_SIGMA8}
Tcmb            = ${COSMO_TCMB}
YHe             = ${COSMO_YHE}
k_p             = ${COSMO_KPIVOT}
N_ur            = ${COSMO_NUR}
m_nu1           = ${COSMO_MNU}
m_nu2           = 0.0
m_nu3           = 0.0
w_0             = ${COSMO_W0}
w_a             = ${COSMO_WA}
ZeroRadiation   = false

${_transfer_block}

[random]
generator       = NGENIC
seed            = ${SEED}
ConstraintFieldFile = ${_noise_abs}
ConstraintFieldName = ic_white_noise

[execution]
NumThreads      = ${NUM_THREADS}

[output]
format          = gadget_hdf5
filename        = ${MONOFONIC_IC}
EOF

    echo "  Config: ${MONOFONIC_CONF}"
    echo "  Noise:  ${_noise_abs}"
    (
        cd "${OUTPUT}"
        vlib::run_in_env "${MONOFONIC_ENV}" "${MONOFONIC_BIN}" "${MONOFONIC_CONF}"
    )

    if [[ ! -f "${MONOFONIC_IC}" ]]; then
        echo "ERROR: monofonIC did not produce ${MONOFONIC_IC}." >&2; exit 1
    fi
    echo "  -> ${MONOFONIC_IC}"
    _cleanup_monofonic_env
    vlib::step_done "run_monofonic"
fi

# -- Step 4: compare ---------------------------------------------------------
if vlib::step_check "compare" "${COMPARISON_NPZ}"; then
    for _f in "${STEPSIC_IC}" "${MONOFONIC_IC}"; do
        if [[ ! -f "${_f}" ]]; then
            echo "ERROR: Missing IC file: ${_f}" >&2; exit 1
        fi
    done

    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/compare-monofonic-run.py" \
        --stepsic-ic "${STEPSIC_IC}" \
        --monofonic-ic "${MONOFONIC_IC}" \
        --lbox "${LBOX}" \
        --nmesh "${NMESH}" \
        --redshift "${Z_INIT}" \
        --method "${MAS_METHOD}" \
        --lpt-order "${LPT_ORDER}" \
        -o "${COMPARISON_NPZ}"

    echo "  -> ${COMPARISON_NPZ}"
    vlib::step_done "compare"
fi

# -- Step 5: plot ------------------------------------------------------------
if vlib::step_check "plot" "${PLOT_OUTPUT}"; then
    if [[ ! -f "${COMPARISON_NPZ}" ]]; then
        echo "ERROR: Comparison archive not found: ${COMPARISON_NPZ}" >&2
        exit 1
    fi

    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/compare-monofonic-plot.py" \
        -i "${COMPARISON_NPZ}" \
        -o "${PLOT_OUTPUT}"

    echo "  -> ${PLOT_OUTPUT}"
    vlib::step_done "plot"
fi

# -- Step 6: evaluate --------------------------------------------------------
if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${COMPARISON_NPZ}" \
        --figure "${PLOT_OUTPUT}" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
