#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Compare periodic initial conditions from stepsic and monofonIC.
#
#  Both programs use the same white-noise field. The campaign compares:
#    (a) Power spectrum ratio       P_stepsic(k) / P_monofonic(k)
#    (b) Per-particle displacement residuals   (Ψ_stepsic - Ψ_monofonic)
#    (c) Per-particle velocity residuals       (v_stepsic - v_monofonic)
#
#  Shared white noise:
#    stepsic saves the real-space noise to ic_white_noise.hdf5 (dataset
#    'ic_white_noise'). monofonIC reads it via ConstraintFieldFile /
#    ConstraintFieldName - no format conversion needed.
#
#  By default, monofonIC reads the CAMB transfer function exported from
#  stepsic. This keeps CAMB/CLASS differences out of the LPT comparison.
#  --use-class instead uses monofonIC's built-in CLASS calculation.
#
#  Pipeline steps:
#    1. run_stepsic     - IC + white noise + CAMB P(k) export
#    2. build_monofonic - clone (if needed) + cmake + compile
#    3. run_monofonic   - write config + run monofonIC
#    4. compare         - measure P(k), displacement, and velocity differences
#    5. plot            - 3-panel comparison figure
#    6. evaluate        - checks written to result.json
#    7. isolated_orders      - per-order amplitude ratio and correlation of the
#                              displacement modes of the 1LPT and 2LPT runs
#    8. plot_isolated_orders - article figure isolated-orders.pdf
#  Steps 7-8 run only in the medium run root, from the custom runs of the
#  profiles monofonic-{64,128,256}-lpt{1,2} (ISOLATED_NMESH); a run of any
#  other profile skips them.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE    small or medium (default)
#    --step=N       Start from step N (1–6, default 1)
#    --plot-only    Jump to step 5 (plotting and evaluation)
#    --force        Re-run all steps regardless of cached outputs
#    --force-step=A,B  Re-run the named steps
#    --clean        Delete cache/ before running
#    --config=PATH  Select a typed custom TOML profile
#    --evaluation=MODE  gate (default), report, or skip
#    --list-steps   Print the step list and exit
#    -h, --help     Print this help text and exit
#
#  Options (monofonic-specific, pass as extra args):
#    --skip-build   Skip step 2 (assume monofonIC binary already exists)
#    --use-class    Use CLASS instead of CAMB_file for the transfer function
#
#  Configuration:
#    Profile keys:  LBOX, NMESH, LPT_ORDER
#    config.env:    STEPSIC_SRC, STEPSIC_PY, STEPSIC_ENV,
#                   MONOFONIC_REPO, MONOFONIC_ENV, Z_INIT, MAS_METHOD, SEED,
#                   ISOLATED_NMESH
#    Environment:   N_MPI, OMP_NUM_THREADS, and any value above
#    Cosmology:     Planck2018EE+BAO from _common/cosmology/
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
vlib::prepare_campaign monofonic-agreement "${BASEDIR}" "${BASEDIR}/config.env"

# The isolated-orders steps aggregate finished custom runs into the medium run.
ISOLATED_ORDERS=0
if [[ "${VLIB_SIZE}" == medium && -n "${ISOLATED_NMESH}" ]]; then
    ISOLATED_ORDERS=1
fi
vlib::declare_steps run_stepsic build_monofonic run_monofonic compare plot evaluate \
    isolated_orders plot_isolated_orders
vlib::manifest_implementation run_stepsic \
    "${BASEDIR}/scripts/export-stepsic-pk.py"
vlib::manifest_implementation build_monofonic \
    "${BASEDIR}/scripts/prepare-monofonic-source.py"
vlib::manifest_implementation compare \
    "${BASEDIR}/scripts/compare-monofonic-run.py" \
    "${BASEDIR}/scripts/particles.py"
vlib::manifest_implementation plot \
    "${BASEDIR}/scripts/compare-monofonic-plot.py"
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
vlib::manifest_implementation isolated_orders \
    "${BASEDIR}/scripts/compare-orders.py" \
    "${BASEDIR}/scripts/particles.py"
vlib::manifest_implementation plot_isolated_orders \
    "${BASEDIR}/scripts/plot-isolated-orders.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

# Run root of the custom run of profile monofonic-<N>-lpt<order>.
_isolated_run() {
    local profile="${BASEDIR}/../_profiles/experiments/monofonic-${1}-lpt${2}.toml"
    echo "${BASEDIR}/runs/custom/$(sha256sum "${profile}" | awk '{print $1}')"
}
if (( ISOLATED_ORDERS )); then
    for _n in ${ISOLATED_NMESH}; do
        for _order in 1 2; do
            vlib::manifest_input isolated_orders \
                "$(_isolated_run "${_n}" "${_order}")/cache/stepsic/ic.hdf5" \
                "$(_isolated_run "${_n}" "${_order}")/cache/monofonic/monofonic_ics.hdf5"
        done
    done
fi

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
export VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache"
CONFIGS_DIR="${VLIB_RUN_ROOT}/config/generated"
OUTPUT="${VLIB_RUN_ROOT}/output"
export CONFIGS_DIR

STEPSIC_CACHE="${VLIB_CACHE_DIR}/stepsic"
MONO_CACHE="${VLIB_CACHE_DIR}/monofonic"

# monofonIC source and build live under ${BASEDIR}/monofonic/.
MONOFONIC_DIR="${MONOFONIC_DIR:-${BASEDIR}/monofonic}"
MONOFONIC_BUILD="${MONOFONIC_BUILD:-${MONOFONIC_DIR}/build}"
MONOFONIC_BIN="${MONOFONIC_BUILD}/monofonIC"
vlib::manifest_implementation run_monofonic "${MONOFONIC_BIN}"

# Stable paths for outputs.
STEPSIC_TOML="${CONFIGS_DIR}/stepsic_monofonic.toml"
MONOFONIC_CONF="${CONFIGS_DIR}/monofonic.conf"
STEPSIC_IC="${STEPSIC_CACHE}/ic.hdf5"
STEPSIC_NOISE="${STEPSIC_CACHE}/ic_white_noise.hdf5"
CAMB_PK_FILE="${STEPSIC_CACHE}/camb_transfer.dat"
MONOFONIC_OUTPUT_BASE="${MONO_CACHE}/monofonic_ics.hdf5"
if (( N_MPI == 1 )); then
    MONOFONIC_IC="${MONOFONIC_OUTPUT_BASE}"
else
    MONOFONIC_IC="${MONO_CACHE}/monofonic_ics.0.hdf5"
fi
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

# Prepare grid_fft.cc: FFT reset ordering and unscaled constraint-field import.
_prepare_monofonic_source() {
    local source_path="${MONOFONIC_DIR}/src/grid_fft.cc"
    python3 "${BASEDIR}/scripts/prepare-monofonic-source.py" "${source_path}"
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

# ============================================================================
# Steps
# ============================================================================

# -- Step 1: run_stepsic -----------------------------------------------------
if vlib::step_check "run_stepsic" "${STEPSIC_IC}"; then
    vlib::clear_dir "${STEPSIC_CACHE}"
    vlib::stepsic::write_toml "${STEPSIC_TOML}" \
        "GEOMETRY=cubical" \
        "LBOX=[${LBOX}, ${LBOX}, ${LBOX}]" \
        "PERIODIC=[1, 1, 1]" \
        "R_3D=0" \
        "D_4D=0" \
        "RCRIT=0" \
        "NRBINS=1" \
        "TYPE=grid" \
        "NGRID=${NMESH}" \
        "NPART=0" \
        "NSHELL=0" \
        "IC_DIR=${STEPSIC_CACHE}" \
        "SAVE_WHITE_NOISE=true" \
        "USE_DOUBLE=true" \
        "SEED=${SEED}" \
        "LPTORDER=${LPT_ORDER}" \
        "NMESH=${NMESH}" \
        "REDSHIFT=${Z_INIT}" \
        "INTERPOLATION=${MAS_METHOD}"
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
    vlib::step_skip "build_monofonic" "--skip-build" "${MONOFONIC_BIN}"
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
# Auto-generated by validation/monofonic-agreement/run.sh
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
ZeroRadiation   = true

${_transfer_block}

[random]
generator       = NGENIC
seed            = ${SEED}
ConstraintFieldFile = ${_noise_abs}
ConstraintFieldName = ic_white_noise

[execution]
NumThreads      = ${OMP_NUM_THREADS}

[output]
format          = gadget_hdf5_double
filename        = ${MONOFONIC_OUTPUT_BASE}
EOF

    echo "  Config: ${MONOFONIC_CONF}"
    echo "  Noise:  ${_noise_abs}"
    (
        cd "${OUTPUT}"
        vlib::run_in_env "${MONOFONIC_ENV}" \
            mpirun -np "${N_MPI}" "${MONOFONIC_BIN}" "${MONOFONIC_CONF}"
    )

    if [[ ! -f "${MONOFONIC_IC}" ]]; then
        echo "ERROR: monofonIC did not produce ${MONOFONIC_IC}." >&2; exit 1
    fi
    echo "  -> ${MONOFONIC_IC}"
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
# -- Steps 7-8: isolated orders (medium run root only) -----------------------
ISOLATED_ARCHIVES=()
if (( ISOLATED_ORDERS )); then
    for _n in ${ISOLATED_NMESH}; do
        ISOLATED_ARCHIVES+=("${VLIB_CACHE_DIR}/isolated-orders-${_n}.npz")
    done
    if vlib::step_check "isolated_orders" "${ISOLATED_ARCHIVES[@]}"; then
        for _n in ${ISOLATED_NMESH}; do
            _one="$(_isolated_run "${_n}" 1)/cache"
            _two="$(_isolated_run "${_n}" 2)/cache"
            vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/compare-orders.py" \
                --stepsic-one "${_one}/stepsic/ic.hdf5" \
                --stepsic-two "${_two}/stepsic/ic.hdf5" \
                --monofonic-one "${_one}/monofonic/monofonic_ics.hdf5" \
                --monofonic-two "${_two}/monofonic/monofonic_ics.hdf5" \
                --nmesh "${_n}" \
                --box-mpc-h "${LBOX}" \
                --redshift "${Z_INIT}" \
                --output "${VLIB_CACHE_DIR}/isolated-orders-${_n}.npz"
        done
        vlib::step_done "isolated_orders"
    fi
    if vlib::step_check "plot_isolated_orders" "${OUTPUT}/isolated-orders.pdf"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-isolated-orders.py" \
            "${ISOLATED_ARCHIVES[@]}" \
            --output "${OUTPUT}/isolated-orders.pdf"
        vlib::step_done "plot_isolated_orders"
    fi
else
    vlib::step_skip "isolated_orders" "medium run root with ISOLATED_NMESH only"
    vlib::step_skip "plot_isolated_orders" "medium run root with ISOLATED_NMESH only"
fi

vlib::report_done
